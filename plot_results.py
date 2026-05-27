"""Generate all result plots for the paper."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path("outputs_ablation/figures")
OUT.mkdir(parents=True, exist_ok=True)

GENERATORS = ["glide", "biggan", "adm", "vqdm"]
GEN_LABELS = ["GLIDE\n(in-dist)", "BigGAN\n(cross)", "ADM\n(cross)", "VQDM\n(cross)"]

# ── load all results ──────────────────────────────────────────────────────────
def load_acc(path):
    try:
        with open(path) as f:
            return json.load(f)["accuracy"] * 100
    except:
        return None

results = {
    "ResNet-18\nFull FT":   [load_acc(f"outputs/test_resnet_{g}.json")   for g in GENERATORS],
    "ViT-Small\nFull FT":   [load_acc(f"outputs_ablation/test_vit_{g}.json")      for g in GENERATORS],
    "ResNet-18\nLinear Probe": [load_acc(f"outputs_ablation/test_resnet_lp_{g}.json") for g in GENERATORS],
    "CLIP-ViT\nLinear Probe":  [load_acc(f"outputs_ablation/test_clip_lp_{g}.json")  for g in GENERATORS],
    "CLIP-ViT\nKNN (k=1)":     [load_acc(f"outputs_ablation/knn_clip_{g}.json")      for g in GENERATORS],
}

patch_acc = [load_acc(f"outputs_ablation/patch_resnet_{g}.json") for g in GENERATORS]

# ── Figure 1: Grouped bar chart – accuracy per method per generator ──────────
fig, ax = plt.subplots(figsize=(11, 4.5))

methods = list(results.keys())
n_methods = len(methods)
n_gens = len(GENERATORS)
x = np.arange(n_gens)
width = 0.15
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

for i, (method, accs) in enumerate(results.items()):
    offset = (i - n_methods / 2 + 0.5) * width
    bars = ax.bar(x + offset, accs, width, label=method,
                  color=colors[i], edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(GEN_LABELS, fontsize=10)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("Detection Accuracy by Method and Generator", fontsize=12, fontweight="bold")
ax.set_ylim(45, 102)
ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Random baseline")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "fig1_accuracy_comparison.png", dpi=300)
plt.close()
print("Saved fig1_accuracy_comparison.png")

# ── Figure 2: Training curves – ResNet-18 full FT ────────────────────────────
with open("outputs/history.json") as f:
    hist = json.load(f)

epochs    = [h["epoch"]     for h in hist]
tr_loss   = [h["train_loss"] for h in hist]
val_loss  = [h["val_loss"]   for h in hist]
tr_acc    = [h["train_acc"]  for h in hist]
val_acc   = [h["val_acc"]    for h in hist]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, (y1, y2, ylabel, title) in zip(axes, [
    (tr_loss, val_loss, "Cross-Entropy Loss", "Training & Validation Loss"),
    (tr_acc,  val_acc,  "Accuracy",           "Training & Validation Accuracy"),
]):
    ax.plot(epochs, y1, label="Train", linewidth=1.8, color="#4C72B0")
    ax.plot(epochs, y2, label="Val",   linewidth=1.8, color="#C44E52")
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11); ax.legend(); ax.grid(alpha=0.3)
    if "Accuracy" in ylabel: ax.set_ylim(0.94, 1.01)
plt.tight_layout()
plt.savefig(OUT / "fig2_training_curves.png", dpi=300)
plt.close()
print("Saved fig2_training_curves.png")

# ── Figure 3: Cross-generator generalisation (in-dist vs cross-gen) ──────────
fig, ax = plt.subplots(figsize=(8, 4))

cross_gens = ["BigGAN", "ADM", "VQDM"]
method_short = ["ResNet-18 FT", "ViT-Small FT", "ResNet-18 LP", "CLIP-ViT LP", "CLIP-ViT KNN"]
all_accs = list(results.values())
colors2 = colors

x = np.arange(len(cross_gens))
width = 0.15
for i, (method, accs) in enumerate(zip(method_short, all_accs)):
    cross_accs = accs[1:]  # skip glide (in-dist)
    offset = (i - len(method_short) / 2 + 0.5) * width
    ax.bar(x + offset, cross_accs, width, label=method,
           color=colors2[i], edgecolor="white", linewidth=0.5)

ax.set_xticks(x); ax.set_xticklabels(cross_gens, fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("Cross-Generator Generalisation (Trained on GLIDE)", fontsize=12, fontweight="bold")
ax.set_ylim(45, 102)
ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "fig3_cross_gen.png", dpi=300)
plt.close()
print("Saved fig3_cross_gen.png")

# ── Figure 4: Patch consistency ───────────────────────────────────────────────
patch_acc_vals = []
patch_cons_vals = []
for g in GENERATORS:
    try:
        with open(f"outputs_ablation/patch_resnet_{g}.json") as f:
            d = json.load(f)
        patch_acc_vals.append(d["accuracy"] * 100)
        patch_cons_vals.append(d["mean_consistency"] * 100)
    except:
        patch_acc_vals.append(0); patch_cons_vals.append(0)

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(GENERATORS))
w = 0.35
ax.bar(x - w/2, patch_acc_vals,  w, label="Patch-vote Accuracy", color="#4C72B0", edgecolor="white")
ax.bar(x + w/2, patch_cons_vals, w, label="Mean Patch Consistency", color="#55A868", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(GEN_LABELS, fontsize=10)
ax.set_ylabel("%", fontsize=11)
ax.set_title("Patch-Level Voting: Accuracy vs Consistency (ResNet-18, 3×3 Grid)", fontsize=11, fontweight="bold")
ax.set_ylim(40, 100); ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "fig4_patch_analysis.png", dpi=300)
plt.close()
print("Saved fig4_patch_analysis.png")

print("\nAll figures saved to", OUT)
