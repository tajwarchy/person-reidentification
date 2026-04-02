# src/build_gallery.py
import os
import pickle
import yaml
import numpy as np
import torch
import torch.nn as nn
import faiss
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import re


# ── Config ─────────────────────────────────────────────────
def load_config(path="configs/reid_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Shared utilities (same as evaluate.py) ─────────────────
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
        return self.transform(img), pid, camid, path


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


# ── Build gallery ──────────────────────────────────────────
def build_gallery(cfg):
    # Device
    device_str = cfg["inference"]["device"]
    if device_str == "mps" and not torch.backends.mps.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    # Transform
    h, w = cfg["inference"]["input_size"]
    tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Load gallery images
    gallery_dir = f"{cfg['paths']['market1501_root']}/bounding_box_test"
    samples     = parse_reid_folder(gallery_dir)
    dataset     = ReIDDataset(samples, tf)
    loader      = DataLoader(dataset,
                             batch_size=cfg["inference"]["batch_size"],
                             shuffle=False,
                             num_workers=cfg["inference"]["num_workers"])
    print(f"Gallery: {len(samples)} images")

    # Load model
    model = ReIDModel().to(device)
    ckpt  = torch.load(cfg["paths"]["checkpoint"],
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"✅ Model loaded (epoch {ckpt['epoch']})")

    # Extract embeddings
    all_embs, all_pids, all_cams, all_paths = [], [], [], []
    with torch.no_grad():
        for imgs, pids, cams, paths in tqdm(loader, desc="Building gallery"):
            imgs = imgs.to(device)
            embs = model(imgs)
            all_embs.append(embs.cpu())
            all_pids.extend(pids.tolist())
            all_cams.extend(cams.tolist())
            all_paths.extend(paths)

    embeddings = torch.cat(all_embs).numpy().astype(np.float32)

    # Ensure contiguous float32 numpy array on CPU
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    # L2 normalize manually with numpy instead of faiss.normalize_L2
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    # Build index
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    # Save
    os.makedirs(cfg["paths"]["output_embeddings"], exist_ok=True)
    faiss.write_index(index, cfg["paths"]["gallery_index"])

    meta = {
        "paths": all_paths,
        "pids":  np.array(all_pids),
        "cams":  np.array(all_cams),
    }
    with open(cfg["paths"]["gallery_meta"], "wb") as f:
        pickle.dump(meta, f)

    print(f"✅ Gallery index saved : {cfg['paths']['gallery_index']}")
    print(f"✅ Gallery meta saved  : {cfg['paths']['gallery_meta']}")
    return index, meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reid_config.yaml")
    args   = parser.parse_args()
    cfg    = load_config(args.config)
    build_gallery(cfg)