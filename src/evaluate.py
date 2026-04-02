# src/evaluate.py
import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import re


# ── Config ─────────────────────────────────────────────────
def load_config(path="configs/reid_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Dataset ────────────────────────────────────────────────
def parse_reid_folder(folder, relabel=False):
    samples = []
    pid_set = set()
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".jpg"):
            continue
        if fname.startswith("-1") or fname.startswith("0000"):
            continue
        pid = int(fname.split("_")[0])
        cam = int(re.search(r'c(\d+)', fname).group(1)) - 1
        samples.append((os.path.join(folder, fname), pid, cam))
        pid_set.add(pid)
    if relabel:
        pid_map = {p: i for i, p in enumerate(sorted(pid_set))}
        samples = [(p, pid_map[pid], cam) for p, pid, cam in samples]
    return samples


class ReIDDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pid, camid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), pid, camid


# ── Model ──────────────────────────────────────────────────
class IBN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        half     = channels // 2
        self.IN  = nn.InstanceNorm2d(half, affine=True)
        self.BN  = nn.BatchNorm2d(half)

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
    def __init__(self, num_classes, feat_dim=2048):
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


# ── Inference ──────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embeddings, pids, camids = [], [], []
    for imgs, pid, camid in tqdm(loader, desc="Extracting", leave=False):
        imgs = imgs.to(device)
        emb  = model(imgs)
        embeddings.append(emb.cpu())
        pids.extend(pid.tolist())
        camids.extend(camid.tolist())
    return torch.cat(embeddings), np.array(pids), np.array(camids)


# ── Evaluation ─────────────────────────────────────────────
def eval_reid(q_emb, q_pids, q_cams, g_emb, g_pids, g_cams, max_rank=10):
    sim    = torch.mm(q_emb, g_emb.t()).numpy()
    n_q    = len(q_pids)
    cmc    = np.zeros(max_rank)
    ap_sum = 0.0

    for q in range(n_q):
        order  = np.argsort(-sim[q])
        gp     = g_pids[order]
        gc     = g_cams[order]
        keep   = ~((gp == q_pids[q]) & (gc == q_cams[q]))
        gp     = gp[keep]
        matches = (gp == q_pids[q]).astype(np.float32)

        for r in range(max_rank):
            if matches[:r+1].sum() > 0:
                cmc[r:] += 1
                break

        num_rel = matches.sum()
        if num_rel == 0:
            continue
        cum     = np.cumsum(matches)
        prec    = cum / (np.arange(len(matches)) + 1)
        ap_sum += (prec * matches).sum() / num_rel

    return cmc / n_q, ap_sum / n_q


# ── CMC Curve ──────────────────────────────────────────────
def plot_cmc(cmc, label, save_path):
    ranks = list(range(1, len(cmc) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(ranks, cmc * 100, marker="o", linewidth=2, label=label)
    plt.xlabel("Rank")
    plt.ylabel("Recognition Rate (%)")
    plt.title("CMC Curve")
    plt.xticks(ranks)
    plt.ylim([0, 100])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  CMC curve saved: {save_path}")


# ── Main ───────────────────────────────────────────────────
def main(args):
    cfg    = load_config(args.config)
    device_str = cfg["inference"]["device"]

    # MPS fallback
    if device_str == "mps" and not torch.backends.mps.is_available():
        print("⚠ MPS not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    # Dataset paths from config
    if args.dataset == "market":
        data_root = cfg["paths"]["market1501_root"]
        label     = "Market-1501"
        num_classes = 751
    else:
        data_root = cfg["paths"]["duke_root"]
        label     = "DukeMTMC-reID"
        num_classes = 702

    # Transform
    h, w = cfg["inference"]["input_size"]
    tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Loaders
    q_samples = parse_reid_folder(f"{data_root}/query")
    g_samples = parse_reid_folder(f"{data_root}/bounding_box_test")
    q_loader  = DataLoader(ReIDDataset(q_samples, tf),
                           batch_size=cfg["inference"]["batch_size"],
                           shuffle=False,
                           num_workers=cfg["inference"]["num_workers"])
    g_loader  = DataLoader(ReIDDataset(g_samples, tf),
                           batch_size=cfg["inference"]["batch_size"],
                           shuffle=False,
                           num_workers=cfg["inference"]["num_workers"])

    # Model
    model = ReIDModel(num_classes=num_classes).to(device)
    ckpt  = torch.load(cfg["paths"]["checkpoint"],
                       map_location=device, weights_only=False)
    # With this:
    if args.dataset == "duke":
        # Cross-dataset: classifier head has different size, skip it
        state = {k: v for k, v in ckpt["state_dict"].items()
                if not k.startswith("classifier")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Skipped keys : {[k for k in missing if 'classifier' in k]}")
    else:
        model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"✅ Checkpoint loaded (epoch {ckpt['epoch']}, "
          f"mAP {ckpt['mAP']*100:.2f}%)")

    # Extract
    print(f"\nExtracting query embeddings  ({len(q_samples)} images)...")
    q_emb, q_pids, q_cams = extract_embeddings(model, q_loader, device)
    print(f"Extracting gallery embeddings ({len(g_samples)} images)...")
    g_emb, g_pids, g_cams = extract_embeddings(model, g_loader, device)

    # Evaluate
    cmc, mAP = eval_reid(q_emb, q_pids, q_cams, g_emb, g_pids, g_cams)

    print(f"\n── {label} Results ──────────────────────────")
    print(f"  Rank-1 : {cmc[0]*100:.2f}%")
    print(f"  Rank-5 : {cmc[4]*100:.2f}%")
    print(f"  Rank-10: {cmc[9]*100:.2f}%")
    print(f"  mAP    : {mAP*100:.2f}%")

    # Save CMC curve
    os.makedirs(cfg["paths"]["output_plots"], exist_ok=True)
    plot_cmc(
        cmc,
        label=f"{label} (Rank-1: {cmc[0]*100:.1f}%, mAP: {mAP*100:.1f}%)",
        save_path=f"{cfg['paths']['output_plots']}/cmc_{args.dataset}.png"
    )

    # Save results as numpy for cross-dataset comparison later
    np.save(f"{cfg['paths']['output_plots']}/results_{args.dataset}.npy",
            {"cmc": cmc, "mAP": mAP, "label": label})

    return cmc, mAP


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/reid_config.yaml")
    parser.add_argument("--dataset", default="market",
                        choices=["market", "duke"],
                        help="Which dataset to evaluate on")
    args = parser.parse_args()
    main(args)