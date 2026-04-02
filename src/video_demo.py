# src/video_demo.py
import os
import re
import yaml
import pickle
import numpy as np
import torch
import torch.nn as nn
import faiss
import cv2
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import defaultdict


# ── Simple IoU tracker ─────────────────────────────────────
class Track:
    def __init__(self, gid, box, emb):
        self.gid        = gid
        self.box        = box        # (x1,y1,x2,y2)
        self.emb        = emb
        self.lost       = 0          # frames since last matched

def iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
    inter = max(0,ix2-ix1) * max(0,iy2-iy1)
    if inter == 0: return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua

# ── Config ─────────────────────────────────────────────────
def load_config(path="configs/reid_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Model (same as all prior phases) ──────────────────────
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


# ── Color palette for persistent IDs ──────────────────────
PALETTE = [
    (255,  56,  56), (255, 157,  51), (255, 221,  51), ( 99, 204,  72),
    ( 60, 200, 165), ( 57, 147, 255), (157,  99, 255), (255,  99, 182),
    (255, 140, 100), (100, 220, 255), (200, 255, 100), (255, 200,  60),
    (120, 120, 255), (255, 120, 120), (120, 255, 120), (200, 100, 255),
    (255, 180,  40), ( 40, 180, 255), (180, 255,  40), (255,  40, 180),
]

def get_color(global_id):
    return PALETTE[global_id % len(PALETTE)]


# ── Crop preprocessing ─────────────────────────────────────
def preprocess_crop(crop_bgr, input_size):
    h, w   = input_size
    img    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img    = cv2.resize(img, (w, h))
    img    = img.astype(np.float32) / 255.0
    mean   = np.array([0.485, 0.456, 0.406])
    std    = np.array([0.229, 0.224, 0.225])
    img    = (img - mean) / std
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    return tensor


# ── Draw annotated box ─────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, global_id, score, unknown=False):
    color      = (128, 128, 128) if unknown else get_color(global_id)
    label      = "Unknown" if unknown else f"ID:{global_id} {score:.2f}"
    fh, fw     = frame.shape[:2]
    thickness  = max(2, fw // 300)        # adaptive to resolution
    font_scale = max(0.8, fw / 1200)      # adaptive to resolution
    font_thick = max(2, fw // 600)
    font       = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)
    cv2.rectangle(frame,
                  (x1, y1 - th - 10),
                  (x1 + tw + 6, y1),
                  color, -1)
    cv2.putText(frame, label,
                (x1 + 3, y1 - 5),
                font, font_scale,
                (255, 255, 255), font_thick,
                cv2.LINE_AA)

# ── Main pipeline ──────────────────────────────────────────
def main(args):
    cfg = load_config(args.config)

    # ── Device ────────────────────────────────────────────
    device_str = cfg["inference"]["device"]
    if device_str == "mps" and not torch.backends.mps.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    input_size = tuple(cfg["inference"]["input_size"])  # (H, W)

    # ── Load ReID model ───────────────────────────────────
    reid_model = ReIDModel().to(device)
    ckpt       = torch.load(cfg["paths"]["checkpoint"],
                            map_location=device, weights_only=False)
    reid_model.load_state_dict(ckpt["state_dict"])
    reid_model.eval()
    print(f"ReID model loaded (epoch {ckpt['epoch']})")

    # ── Load YOLOv8 ───────────────────────────────────────
    from ultralytics import YOLO
    yolo = YOLO(cfg["video"]["yolo_model"])
    print("YOLOv8 loaded")

    # ── Load FAISS gallery ────────────────────────────────
    index = faiss.read_index(cfg["paths"]["gallery_index"])
    with open(cfg["paths"]["gallery_meta"], "rb") as f:
        meta = pickle.load(f)
    g_paths = meta["paths"]
    g_pids  = meta["pids"]
    print(f"Gallery: {index.ntotal} vectors")

    sim_thresh   = cfg["video"]["reid_similarity_threshold"]
    unknown_lbl  = cfg["video"]["unknown_label"]

    # ── Open video ────────────────────────────────────────
    cap = cv2.VideoCapture(cfg["video"]["input_path"])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg['video']['input_path']}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or cfg["video"]["fps_output"]
    fw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {fw}x{fh} @ {fps:.1f}fps, {total} frames")
    
    # ── Resize scale for 4K processing ─────────────────────── 
    proc_w     = cfg["video"].get("processing_width", fw)
    scale      = min(1.0, proc_w / fw)   # never upscale
    proc_h     = int(fh * scale)
    do_resize  = (scale < 1.0)
    print(f"Processing at: {proc_w}x{proc_h} (scale={scale:.2f})")

    # ── Output writer ─────────────────────────────────────
    os.makedirs(cfg["paths"]["output_videos"], exist_ok=True)
    out_path = cfg["video"]["output_annotated"]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc,
                         cfg["video"]["fps_output"], (proc_w, proc_h))

    # ── Tracking state ────────────────────────────────────────
    active_tracks        = []            # list of Track objects
    next_global_id       = 0
    global_id_counts     = defaultdict(int)
    global_id_crops      = {}
    from collections import deque
    global_id_embeddings = defaultdict(lambda: deque(maxlen=60))
    IOU_THRESH           = 0.3
    REID_THRESH          = cfg["video"]["reid_similarity_threshold"]
    MAX_LOST             = 30            # frames to keep lost track alive

    # ── Process frames ────────────────────────────────────
    frame_idx = 0
    pbar      = tqdm(total=total, desc="Processing video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if do_resize:
            frame = cv2.resize(frame, (proc_w, proc_h))
        fw, fh = frame.shape[1], frame.shape[0]

        # YOLOv8 detection
        results = yolo(
            frame,
            classes=[0],                              # person only
            conf=cfg["video"]["yolo_conf_threshold"],
            iou=cfg["video"]["yolo_iou_threshold"],
            device=device_str,
            verbose=False
        )

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy().astype(int)

            # Batch extract ReID embeddings for all crops
            crops, valid_boxes = [], []
            for x1, y1, x2, y2 in xyxy:
                x1c = max(0, x1); y1c = max(0, y1)
                x2c = min(fw, x2); y2c = min(fh, y2)
                if (x2c - x1c) < 20 or (y2c - y1c) < 40:
                    continue                           # skip tiny crops
                crop = frame[y1c:y2c, x1c:x2c]
                crops.append(preprocess_crop(crop, input_size))
                valid_boxes.append((x1c, y1c, x2c, y2c, crop))
            det_results = []
            if crops:
                batch = torch.stack(crops).to(device)
                with torch.no_grad():
                    embs = reid_model(batch).cpu().numpy().astype(np.float32)

                # Normalize for cosine search
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs  = embs / np.clip(norms, 1e-12, None)

                # FAISS search
                scores_batch, idxs_batch = index.search(embs, 1)

                used_gids_this_frame = set()
                

                for i, (x1, y1, x2, y2, crop) in enumerate(valid_boxes):
                    emb   = embs[i]
                    box   = (x1, y1, x2, y2)
                    score = float(scores_batch[i][0])
                    g_idx = int(idxs_batch[i][0])

                    # ── Step 1: match to active track via IoU ──
                    best_iou   = 0.0
                    best_track = None
                    for t in active_tracks:
                        v = iou(box, t.box)
                        if v > best_iou:
                            best_iou   = v
                            best_track = t

                    if best_iou >= IOU_THRESH and best_track is not None \
                            and best_track.gid not in used_gids_this_frame:
                        # Matched to existing track via IoU
                        gid = best_track.gid
                        best_track.box  = box
                        best_track.emb  = emb
                        best_track.lost = 0
                        match_score = float(np.dot(emb, best_track.emb))

                    else:
                        # ── Step 2: match via ReID embedding ──
                        best_reid_score = -1.0
                        best_gid        = -1
                        for t in active_tracks:
                            if t.gid in used_gids_this_frame:
                                continue
                            sim = float(np.dot(emb, t.emb))
                            if sim > best_reid_score:
                                best_reid_score = sim
                                best_gid        = t.gid
                                best_track      = t

                        if best_reid_score >= REID_THRESH and best_gid != -1:
                            gid = best_gid
                            best_track.box  = box
                            best_track.emb  = emb
                            best_track.lost = 0
                            match_score     = best_reid_score

                        else:
                            # ── Step 3: new track via gallery ──
                            if score >= REID_THRESH:
                                gid = next_global_id
                                next_global_id += 1
                                active_tracks.append(Track(gid, box, emb))
                                match_score = score
                            else:
                                det_results.append(
                                    (x1,y1,x2,y2,-1,score,True))
                                continue

                    used_gids_this_frame.add(gid)
                    global_id_embeddings[gid].append(emb)
                    global_id_counts[gid] += 1

                    if gid not in global_id_crops:
                        global_id_crops[gid] = {
                            "crop":    crop.copy(),
                            "gallery": g_paths[g_idx],
                            "pid":     int(g_pids[g_idx]),
                            "score":   match_score,
                        }
                    det_results.append((x1,y1,x2,y2,gid,match_score,False))

            # Age lost tracks
            for t in active_tracks:
                if t.gid not in used_gids_this_frame:
                    t.lost += 1
            active_tracks = [t for t in active_tracks if t.lost < MAX_LOST]

            # Draw
            for x1,y1,x2,y2,gid,score,unknown in det_results:
                draw_box(frame, x1, y1, x2, y2, gid, score, unknown)

        writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    print(f"\n✅ Annotated video saved: {out_path}")
    print(f"   Unique global IDs assigned: {next_global_id}")
    print(f"   Top re-identified IDs:")
    top5 = sorted(global_id_counts.items(), key=lambda x: -x[1])[:5]
    for gid, cnt in top5:
        print(f"     Global ID {gid}: {cnt} frames")

    # Save tracking state for side-by-side export
    state_path = f"{cfg['paths']['output_videos']}/tracking_state.pkl"
    with open(state_path, "wb") as f:
        pickle.dump({
            "global_id_counts": dict(global_id_counts),
            "global_id_crops":  global_id_crops,
        }, f)
    print(f"   Tracking state saved: {state_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reid_config.yaml")
    args   = parser.parse_args()
    main(args)