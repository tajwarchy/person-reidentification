# run as: python src/cross_dataset_table.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, yaml

def load_config(path="configs/reid_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

cfg = load_config()
plots_dir = cfg["paths"]["output_plots"]

market = np.load(f"{plots_dir}/results_market.npy", allow_pickle=True).item()
duke   = np.load(f"{plots_dir}/results_duke.npy",   allow_pickle=True).item()

# ── Print table ────────────────────────────────────────────
print("\n── Cross-Dataset Generalization ─────────────────────")
print(f"{'':30s} {'Rank-1':>8} {'Rank-5':>8} {'Rank-10':>8} {'mAP':>8}")
print("-" * 58)
for res, train_set, test_set in [
    (market, "Market-1501", "Market-1501"),
    (duke,   "Market-1501", "DukeMTMC-reID"),
]:
    print(f"  Train: {train_set:12s} → Test: {test_set:13s} "
          f"{res['cmc'][0]*100:7.2f}% "
          f"{res['cmc'][4]*100:7.2f}% "
          f"{res['cmc'][9]*100:7.2f}% "
          f"{res['mAP']*100:7.2f}%")

# ── Plot comparison table as image ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 2.5))
ax.axis("off")

rows = [
    ["Train → Test", "Rank-1", "Rank-5", "Rank-10", "mAP"],
    [f"Market-1501 → Market-1501",
     f"{market['cmc'][0]*100:.2f}%",
     f"{market['cmc'][4]*100:.2f}%",
     f"{market['cmc'][9]*100:.2f}%",
     f"{market['mAP']*100:.2f}%"],
    [f"Market-1501 → DukeMTMC-reID",
     f"{duke['cmc'][0]*100:.2f}%",
     f"{duke['cmc'][4]*100:.2f}%",
     f"{duke['cmc'][9]*100:.2f}%",
     f"{duke['mAP']*100:.2f}%"],
]

tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.2, 2.0)

# Header styling
for j in range(5):
    tbl[0, j].set_facecolor("#2c3e50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Row styling
for j in range(5):
    tbl[1, j].set_facecolor("#eaf4fb")
    tbl[2, j].set_facecolor("#fef9e7")

plt.title("Cross-Dataset Generalization (IBN-a ResNet-50, trained on Market-1501)",
          fontsize=12, fontweight="bold", pad=15)
plt.tight_layout()
save_path = f"{plots_dir}/cross_dataset_table.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Table saved: {save_path}")

# ── Overlay CMC curves ─────────────────────────────────────
plt.figure(figsize=(8, 5))
ranks = list(range(1, 11))
plt.plot(ranks, market["cmc"]*100, marker="o", linewidth=2,
         label=f"Market→Market (Rank-1: {market['cmc'][0]*100:.1f}%)")
plt.plot(ranks, duke["cmc"]*100,   marker="s", linewidth=2,
         label=f"Market→Duke  (Rank-1: {duke['cmc'][0]*100:.1f}%)")
plt.xlabel("Rank")
plt.ylabel("Recognition Rate (%)")
plt.title("CMC Curves — Cross-Dataset Generalization")
plt.xticks(ranks)
plt.ylim([0, 100])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
cmc_path = f"{plots_dir}/cmc_crossdataset.png"
plt.savefig(cmc_path, dpi=150)
plt.close()
print(f"  CMC overlay saved: {cmc_path}")