# src/retrieval_demo.py
import os
import pickle
import yaml
import numpy as np
import torch
import torch.nn as nn
import faiss
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
from tqdm import tqdm


# ── Config ─────────────────────────────────────────────────
def load_config(path="configs/reid_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Reuse model + dataset from build_gallery ───────────────
# (copy IBN, BNNeck, ReIDModel, parse_reid_folder here)
def parse_reid_folder(folder):
    samples = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".jpg"):
            continue
        if fname.startswith("-1") or fname.startswith("0000"):
            continue
        pid = int(fname.split("_")[0])
        cam = int(re.search(r'c(\d+)', fname).group(1)) - 1
        samples.append((os.path.join(folder, fname), pid, cam))
    return samples


class IBN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        half    = channels // 2
        self.IN = nn.InstanceNorm2d(half, affine=True)
        self.BN = nn.BatchNorm2d(half)

    def forward(self, x):
        half = x.size(1) // 2
        return torch.cat([self.IN(x[:, :half]), self.BN(x[:, half:])], dim=1)


def add_ibn_a(model):
    for layer_name in ["layer1", "layer2", "layer3"]:
        for block in getattr(model, layer_name):
            if hasattr(block, "bn1"):
                block.bn1 = IBN(block.bn1.num_features)
    return model


class BNNeck(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)

    def forward(self, x):
        return self.bn(x)


class ReIDModel(nn.Module):
    def __init__(self, num_classes=751, feat_dim=2048):
        super().__init__()
        import timm
        backbone        = timm.create_model("resnet50", pretrained=False,
                                            num_classes=0, global_pool="avg")
        self.backbone   = add_ibn_a(backbone)
        self.bnneck     = BNNeck(feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)

    def forward(self, x):
        feat    = self.backbone(x)
        feat_bn = self.bnneck(feat)
        if self.training:
            return self.classifier(feat_bn), feat
        return nn.functional.normalize(feat_bn, dim=1)


# ── Query embedding ────────────────────────────────────────
@torch.no_grad()
def get_embedding(model, img_path, tf, device):
    img = Image.open(img_path).convert("RGB")
    x   = tf(img).unsqueeze(0).to(device)
    emb = model(x).cpu().numpy().astype(np.float32)
    faiss.normalize_L2(emb)
    return emb


# ── Retrieval grid ─────────────────────────────────────────
def draw_retrieval_grid(query_path, query_pid,
                        top_paths, top_pids, top_scores,
                        save_path, top_k=10):
    img_h, img_w = 256, 128
    pad          = 6
    label_h      = 22
    n_cols       = top_k + 1        # query + top_k results
    canvas_w     = n_cols * (img_w + pad) + pad
    canvas_h     = img_h + label_h + pad * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw   = ImageDraw.Draw(canvas)

    def paste_img(img_path, col, border_color, label):
        img = Image.open(img_path).convert("RGB").resize((img_w, img_h))
        x   = pad + col * (img_w + pad)
        y   = pad
        # Border
        bordered = Image.new("RGB", (img_w + 4, img_h + 4), border_color)
        bordered.paste(img, (2, 2))
        canvas.paste(bordered, (x - 2, y - 2))
        # Label
        draw.rectangle([x, y + img_h, x + img_w, y + img_h + label_h],
                       fill=(50, 50, 50))
        draw.text((x + 3, y + img_h + 3), label,
                  fill=(255, 255, 255))

    # Query image — blue border
    paste_img(query_path, 0, (66, 133, 244), f"QUERY  ID:{query_pid}")

    # Top-k results
    for i, (path, pid, score) in enumerate(zip(top_paths, top_pids, top_scores)):
        correct = (pid == query_pid)
        border  = (52, 168, 83) if correct else (234, 67, 53)   # green/red
        paste_img(path, i + 1, border,
                  f"{'✓' if correct else '✗'} {score:.3f} ID:{pid}")

    canvas.save(save_path)


# ── Main ───────────────────────────────────────────────────
def main(args):
    cfg = load_config(args.config)

    device_str = cfg["inference"]["device"]
    if device_str == "mps" and not torch.backends.mps.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    h, w = cfg["inference"]["input_size"]
    tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Load model
    model = ReIDModel().to(device)
    ckpt  = torch.load(cfg["paths"]["checkpoint"],
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"✅ Model loaded (epoch {ckpt['epoch']})")

    # Load FAISS index + metadata
    index = faiss.read_index(cfg["paths"]["gallery_index"])
    with open(cfg["paths"]["gallery_meta"], "rb") as f:
        meta = pickle.load(f)
    g_paths = meta["paths"]
    g_pids  = meta["pids"]
    print(f"✅ Gallery loaded: {index.ntotal} vectors")

    # Select query examples from Market-1501 query split
    query_samples = parse_reid_folder(
        f"{cfg['paths']['market1501_root']}/query"
    )

    # Pick diverse queries — sample from different IDs
    top_k     = cfg["retrieval"]["top_k"]
    n_queries = cfg["retrieval"]["num_query_examples"]

    # Sample one image per ID, take first n_queries
    seen_pids, selected = set(), []
    for path, pid, cam in query_samples:
        if pid not in seen_pids:
            selected.append((path, pid, cam))
            seen_pids.add(pid)
        if len(selected) == n_queries:
            break

    os.makedirs(cfg["paths"]["output_plots"], exist_ok=True)

    for i, (q_path, q_pid, _) in enumerate(selected):
        print(f"\nQuery {i+1}/{n_queries} — ID: {q_pid}")

        # Extract query embedding
        q_emb = get_embedding(model, q_path, tf, device)

        # Search FAISS
        scores, idxs = index.search(q_emb, top_k)
        scores = scores[0]
        idxs   = idxs[0]

        top_paths  = [g_paths[j] for j in idxs]
        top_pids   = [g_pids[j]  for j in idxs]

        # Print results
        matches = sum(1 for p in top_pids if p == q_pid)
        print(f"  Top-{top_k} matches: {matches}/{top_k} correct")
        for rank, (pid, score) in enumerate(zip(top_pids, scores), 1):
            correct = "✓" if pid == q_pid else "✗"
            print(f"  Rank {rank:2d}: {correct} score={score:.4f} pid={pid}")

        # Save grid
        save_path = f"{cfg['paths']['output_plots']}/retrieval_query{i+1}_pid{q_pid}.png"
        draw_retrieval_grid(
            q_path, q_pid,
            top_paths, top_pids, scores,
            save_path, top_k=top_k
        )
        print(f"  Grid saved: {save_path}")

    print(f"\n✅ Retrieval demo complete — {n_queries} grids saved")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reid_config.yaml")
    args   = parser.parse_args()
    main(args)