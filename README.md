# Project V2.1 — Person Re-Identification with ResNet-50 IBN-a

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Platform](https://img.shields.io/badge/Platform-M1%20Mac%20%2B%20Colab-lightgrey)

> **Portfolio Project** — Computer Vision Engineer Track  
> Training: Google Colab T4 | Inference & Demo: Apple M1 MacBook Air (MPS)

---

## Overview

End-to-end Person Re-Identification system built from scratch in pure PyTorch.
Trains a ResNet-50 IBN-a backbone with BNNeck on Market-1501, evaluates
cross-dataset generalization on DukeMTMC-reID, and deploys a full video
demo pipeline with persistent global ID assignment across frames.

---

## Results

### Market-1501 (trained & evaluated)

| Metric  | Score  |
|---------|--------|
| Rank-1  | 81.80% |
| Rank-5  | 92.64% |
| Rank-10 | 95.64% |
| mAP     | 60.38% |

### Cross-Dataset Generalization (Market-1501 → DukeMTMC-reID, zero-shot)

| Train        | Test          | Rank-1 | mAP    |
|--------------|---------------|--------|--------|
| Market-1501  | Market-1501   | 81.80% | 60.38% |
| Market-1501  | DukeMTMC-reID | 37.48% | 20.42% |

---

## What I Built

- **ResNet-50 IBN-a from scratch** — Instance-Batch Normalization
  implemented manually and patched into ResNet-50 (timm backbone)
- **BNNeck** — separate feature paths for metric loss and classification
- **Label smoothing CE + Triplet loss** — combined loss with hard mining
- **Cosine LR with warmup** — 60 epoch training recipe on Colab T4
- **FAISS gallery index** — fast cosine similarity search at inference
- **Video demo pipeline** — YOLOv8m detection + ReID embedding +
  IoU-anchored temporal tracking + persistent global ID assignment
- **t-SNE embedding visualization** — 30 identities, cosine metric
- **Cross-dataset evaluation** — zero-shot transfer to DukeMTMC-reID

---

## Architecture
```
Input Frame
    │
    ▼
YOLOv8m (person detection)
    │
    ▼ crops
ResNet-50 IBN-a backbone
    │
    ▼
Global Average Pool → 2048-d
    │
    ▼
BNNeck (BatchNorm1d)
    │
    ├──► L2-normalize → cosine similarity → FAISS gallery search
    │
    └──► Linear classifier (training only)
```

---

## Tech Stack

| Component       | Tool                          |
|-----------------|-------------------------------|
| Framework       | PyTorch 2.x                   |
| Backbone        | ResNet-50 IBN-a (timm)        |
| Detection       | YOLOv8m (ultralytics)         |
| Similarity Search | FAISS (faiss-cpu)           |
| Training        | Google Colab T4               |
| Inference       | Apple M1 MPS                  |
| Environment     | conda                         |

---

## Project Structure
```
person-reidentification/
├── configs/
│   └── reid_config.yaml          # all parameters, no hardcoded values
├── notebooks/
│   └── train_reid.ipynb          # Colab training notebook
├── src/
│   ├── prepare_datasets.py       # dataset verification
│   ├── evaluate.py               # CMC + mAP evaluation
│   ├── cross_dataset_table.py    # comparison table + CMC overlay
│   ├── build_gallery.py          # FAISS gallery index builder
│   ├── retrieval_demo.py         # top-10 image retrieval demo
│   ├── tsne_viz.py               # t-SNE embedding visualization
│   ├── video_demo.py             # full video ReID pipeline
│   └── side_by_side_export.py    # side-by-side grid video export
├── outputs/
│   ├── plots/                    # CMC curves, t-SNE, retrieval grids
│   └── videos/                   # annotated demo videos
├── checkpoints/                  # download reid_best.pth from Drive
├── environment.yml
└── README.md
```

---

## Quickstart

### 1. Setup environment
```bash
conda env create -f environment.yml
conda activate project-v2.1-reid
```

### 2. Download datasets

Place datasets at:
```
data/Market-1501-v15.09.15/
data/DukeMTMC-reID/
```

### 3. Download checkpoint

Download `reid_best.pth` from [Google Drive](https://drive.google.com/file/d/1f7RtrGtsEVQ-a99zuSMz3fQo_jij-n5T/view?usp=sharing) and place at:
```
checkpoints/reid_best.pth
```

### 4. Run evaluation
```bash
python src/evaluate.py --dataset market
python src/evaluate.py --dataset duke
```

### 5. Build gallery & run retrieval demo
```bash
python src/build_gallery.py
python src/retrieval_demo.py
```

### 6. Run video demo
```bash
python src/video_demo.py
python src/side_by_side_export.py
```

---

## Training (Colab)

Open `notebooks/train_reid.ipynb` in Google Colab with T4 GPU runtime.
Upload datasets to `MyDrive/reid_project/` and run all cells.
Training takes ~2 hours on T4.

**Final metrics after 60 epochs:**
- Rank-1: 81.80% | mAP: 60.38% on Market-1501

---

## Key Learning Outcomes

- IBN-a implementation from scratch for domain generalization
- BNNeck architecture for separating metric and classification features
- Hard triplet mining for metric learning
- FAISS flat index for efficient embedding retrieval
- IoU-anchored temporal tracking for stable video ID assignment
- Split training/inference environment (Colab T4 + M1 MPS)
- Hardware-aware optimization throughout (MPS, num_workers=0)

---

## Outputs

| Output | Description |
|--------|-------------|
| `outputs/plots/cmc_market.png` | CMC curve on Market-1501 |
| `outputs/plots/cmc_crossdataset.png` | CMC overlay — both datasets |
| `outputs/plots/cross_dataset_table.png` | Generalization table |
| `outputs/plots/tsne_embeddings.png` | t-SNE of 30 identities |
| `outputs/plots/retrieval_query*.png` | Top-10 retrieval grids |
| `outputs/videos/reid_demo_annotated.mp4` | Full annotated video demo |
| `outputs/videos/reid_demo_sidebyside.mp4` | Side-by-side export |
