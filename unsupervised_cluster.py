"""
Unsupervised clustering to verify MoE feasibility.
Extracts CLIP-ViT features from 4 generators, runs K-Means(k=4),
and visualises with t-SNE. Measures purity and NMI.
"""

import os, random, zipfile, io, argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── config ──────────────────────────────────────────────────────────────────
GENERATORS = ["glide", "BigGAN", "ADM", "VQDM"]
RAW_DIR    = "/root/ml-project/data/raw/shimei"   # contains *.zip files
SAMPLES    = 200          # images per generator
SEED       = 42
OUT_DIR    = "/root/ml-project/outputs_ablation/figures"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)

# ── helpers ──────────────────────────────────────────────────────────────────

def sample_from_zip(zip_path: str, n: int, seed: int = SEED) -> list[Image.Image]:
    """Randomly sample n fake images from a generator zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [
            f for f in zf.namelist()
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
            and "1_fake" in f
        ]
        if not names:
            names = [f for f in zf.namelist()
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        rng = random.Random(seed)
        chosen = rng.sample(names, min(n, len(names)))
        imgs = []
        for name in chosen:
            data = zf.read(name)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            imgs.append(img)
    return imgs


def load_glide_images(n: int) -> list[Image.Image]:
    return sample_from_zip(os.path.join(RAW_DIR, "glide.zip"), n)


# ── load CLIP ────────────────────────────────────────────────────────────────
print("Loading CLIP model …")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["NO_PROXY"] = "*"

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir="/root/.cache/huggingface"
).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir="/root/.cache/huggingface"
)
print(f"CLIP loaded on {DEVICE}")


# ── extract features ──────────────────────────────────────────────────────────
def extract_features(images: list[Image.Image], batch=32) -> np.ndarray:
    feats = []
    for i in range(0, len(images), batch):
        batch_imgs = images[i:i+batch]
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
            out = out / out.norm(dim=-1, keepdim=True)  # L2 normalise
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


all_features = []
all_labels   = []   # integer 0-3
all_names    = []

print("\nSampling and extracting features …")
for idx, gen in enumerate(GENERATORS):
    print(f"  [{idx+1}/4] {gen} …", flush=True)
    if gen.lower() == "glide":
        images = load_glide_images(SAMPLES)
    else:
        zip_path = os.path.join(RAW_DIR, f"{gen}.zip")
        images = sample_from_zip(zip_path, SAMPLES)

    print(f"        loaded {len(images)} images", flush=True)
    feats = extract_features(images)
    all_features.append(feats)
    all_labels.extend([idx] * len(feats))
    all_names.extend([gen] * len(feats))

X = np.concatenate(all_features, axis=0)   # (N, 512)
y = np.array(all_labels)
print(f"\nFeature matrix: {X.shape}")


# ── K-Means clustering ────────────────────────────────────────────────────────
print("Running K-Means (k=4) …")
km = KMeans(n_clusters=4, n_init=20, random_state=SEED)
pred = km.fit_predict(X)

# purity
def purity_score(y_true, y_pred):
    from scipy.stats import mode
    total = 0
    for c in np.unique(y_pred):
        mask = y_pred == c
        majority = mode(y_true[mask], keepdims=True).count[0]
        total += majority
    return total / len(y_true)

purity = purity_score(y, pred)
nmi    = normalized_mutual_info_score(y, pred)
print(f"  Purity : {purity:.4f}")
print(f"  NMI    : {nmi:.4f}")


# ── t-SNE visualisation ───────────────────────────────────────────────────────
print("Running t-SNE …")
tsne = TSNE(n_components=2, perplexity=40, max_iter=1000,
            random_state=SEED, init="pca")
X2d = tsne.fit_transform(X)

COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
MARKERS = ["o", "s", "^", "D"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (color_by, title) in zip(axes, [
    ("true",    "t-SNE coloured by Generator (ground truth)"),
    ("cluster", "t-SNE coloured by K-Means cluster"),
]):
    labels = y if color_by == "true" else pred
    for i, gen in enumerate(GENERATORS):
        mask = labels == i
        ax.scatter(X2d[mask, 0], X2d[mask, 1],
                   c=COLORS[i], marker=MARKERS[i],
                   alpha=0.6, s=18, linewidths=0)
    patches = [mpatches.Patch(color=COLORS[i],
               label=GENERATORS[i] if color_by == "true" else f"Cluster {i}")
               for i in range(4)]
    ax.legend(handles=patches, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.3)

fig.suptitle(
    f"Unsupervised Generator Clustering  |  Purity={purity:.3f}  NMI={nmi:.3f}",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "fig_unsupervised_cluster.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved figure → {out_path}")


# ── per-cluster breakdown ─────────────────────────────────────────────────────
print("\nCluster composition (rows=cluster, cols=generator):")
from collections import Counter
header = f"{'':10s}" + "".join(f"{g:>10s}" for g in GENERATORS)
print(header)
for c in range(4):
    mask = pred == c
    counts = Counter(y[mask].tolist())
    row = f"Cluster {c}  " + "".join(f"{counts.get(i,0):>10d}" for i in range(4))
    print(row)

print(f"\nDone. Purity={purity:.4f}, NMI={nmi:.4f}")
