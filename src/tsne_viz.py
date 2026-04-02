# src/tsne_viz.py
import os
import pickle
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re


# ── Config ─────────────────────────────────────────────────
def load_config(path="configs/reid_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Shared utilities ───────────────────────────────────────
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


# ── Sample gallery embeddings ──────────────────────────────
def sample_embeddings(model, gallery_dir, num_ids, imgs_per_id,
                      tf, device, seed=42):
    np.random.seed(seed)
    all_samples = parse_reid_folder(gallery_dir)

    # Group by pid
    from collections import defaultdict
    pid_to_samples = defaultdict(list)
    for s in all_samples:
        pid_to_samples[s[1]].append(s)

    # Sample num_ids random IDs that have enough images
    eligible = [p for p, s in pid_to_samples.items() if len(s) >= imgs_per_id]
    chosen_pids = np.random.choice(eligible, min(num_ids, len(eligible)),
                                   replace=False)

    selected = []
    for pid in chosen_pids:
        samples = pid_to_samples[pid]
        chosen  = np.random.choice(len(samples), imgs_per_id, replace=False)
        selected.extend([samples[i] for i in chosen])

    print(f"Selected {len(chosen_pids)} IDs × {imgs_per_id} images "
          f"= {len(selected)} total")

    # Extract embeddings
    dataset = ReIDDataset(selected, tf)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=0)

    embeddings, pids = [], []
    model.eval()
    with torch.no_grad():
        for imgs, pid, _ in tqdm(loader, desc="Extracting for t-SNE"):
            imgs = imgs.to(device)
            emb  = model(imgs)
            embeddings.append(emb.cpu().numpy())
            pids.extend(pid.tolist())

    return np.vstack(embeddings), np.array(pids), chosen_pids


# ── t-SNE plot ─────────────────────────────────────────────
def plot_tsne(embeddings, pids, chosen_pids, save_path,
              perplexity=30, n_iter=1000, random_state=42):
    print(f"\nRunning t-SNE (perplexity={perplexity}, "
          f"n_iter={n_iter})...")
    # With this:
    tsne = TSNE(n_components=2, perplexity=perplexity,
                max_iter=n_iter, random_state=random_state,
                metric="cosine", init="pca")
    coords = tsne.fit_transform(embeddings)
    print(f"t-SNE complete | KL divergence: {tsne.kl_divergence_:.4f}")

    # Assign a color per identity
    n_ids   = len(chosen_pids)
    cmap    = cm.get_cmap("tab20", n_ids) if n_ids <= 20 \
              else cm.get_cmap("hsv",  n_ids)
    pid_to_color = {pid: cmap(i) for i, pid in enumerate(chosen_pids)}

    fig, ax = plt.subplots(figsize=(12, 10))
    for pid in chosen_pids:
        mask = pids == pid
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[pid_to_color[pid]],
            s=40, alpha=0.85,
            label=f"ID {pid}"
        )

    ax.set_title(
        f"t-SNE of ReID Embeddings\n"
        f"{n_ids} identities × {len(embeddings)//n_ids} images  "
        f"(ResNet-50 IBN-a, Market-1501)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.grid(True, alpha=0.2)

    # Legend — only show if ≤ 20 IDs to avoid clutter
    if n_ids <= 20:
        ax.legend(loc="best", fontsize=7, ncol=2,
                  markerscale=1.5, framealpha=0.7)
    else:
        ax.text(0.01, 0.99,
                f"{n_ids} identities shown",
                transform=ax.transAxes,
                va="top", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"t-SNE plot saved: {save_path}")


# ── Main ───────────────────────────────────────────────────
def main(args):
    cfg = load_config(args.config)

    device_str = cfg["inference"]["device"]
    if device_str == "mps" and not torch.backends.mps.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device     : {device}")

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
    print(f"Model      : epoch {ckpt['epoch']} loaded")

    # t-SNE config from YAML
    num_ids     = cfg["tsne"]["num_ids"]
    imgs_per_id = cfg["tsne"]["images_per_id"]
    perplexity  = cfg["tsne"]["perplexity"]
    n_iter      = cfg["tsne"]["n_iter"]
    random_state= cfg["tsne"]["random_state"]

    gallery_dir = f"{cfg['paths']['market1501_root']}/bounding_box_test"

    embeddings, pids, chosen_pids = sample_embeddings(
        model, gallery_dir, num_ids, imgs_per_id,
        tf, device, seed=random_state
    )

    os.makedirs(cfg["paths"]["output_plots"], exist_ok=True)
    save_path = f"{cfg['paths']['output_plots']}/tsne_embeddings.png"

    plot_tsne(embeddings, pids, chosen_pids, save_path,
              perplexity=perplexity, n_iter=n_iter,
              random_state=random_state)

    print("\n✅ Phase 7 complete")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reid_config.yaml")
    args   = parser.parse_args()
    main(args)