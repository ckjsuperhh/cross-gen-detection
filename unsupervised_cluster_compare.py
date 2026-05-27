"""
Compare unsupervised clustering quality:
  (A) Pretrained ViT-Small features  (timm, no task supervision)
  (B) Fine-tuned ViT-Small features  (full-finetune checkpoint)

Metric: K-Means(k=4) Purity + NMI, visualised with t-SNE 2x2 grid.
"""

import os, random, zipfile, io
import numpy as np
import torch
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── config ────────────────────────────────────────────────────────────────────
GENERATORS  = ["glide", "BigGAN", "ADM", "VQDM"]
RAW_DIR     = "/root/ml-project/data/raw/shimei"
CKPT        = "/root/ml-project/outputs_ablation/vit_small_full_finetune/best.pth"
TIMM_MODEL  = "vit_small_patch16_224.augreg_in21k_ft_in1k"
SAMPLES     = 200
SEED        = 42
OUT_DIR     = "/root/ml-project/outputs_ablation/figures"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED); np.random.seed(SEED)
os.environ["NO_PROXY"]    = "*"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# ── data loading ──────────────────────────────────────────────────────────────
def sample_from_zip(zip_path: str, n: int) -> list[Image.Image]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [f for f in zf.namelist()
                 if f.lower().endswith((".png", ".jpg", ".jpeg")) and "1_fake" in f]
        if not names:
            names = [f for f in zf.namelist()
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        rng = random.Random(SEED)
        chosen = rng.sample(names, min(n, len(names)))
        return [Image.open(io.BytesIO(zf.read(n))).convert("RGB") for n in chosen]


def load_all_images():
    all_imgs, all_labels = [], []
    for idx, gen in enumerate(GENERATORS):
        print(f"  [{idx+1}/4] {gen} …", flush=True)
        imgs = sample_from_zip(os.path.join(RAW_DIR, f"{gen}.zip"), SAMPLES)
        all_imgs.extend(imgs)
        all_labels.extend([idx] * len(imgs))
        print(f"        {len(imgs)} images", flush=True)
    return all_imgs, np.array(all_labels)


# ── feature extraction ────────────────────────────────────────────────────────
def build_transform(model):
    cfg = resolve_data_config({}, model=model)
    return create_transform(**cfg)


def extract(model, transform, images, batch=32):
    feats = []
    model.eval()
    for i in range(0, len(images), batch):
        tensors = torch.stack([transform(img) for img in images[i:i+batch]]).to(DEVICE)
        with torch.no_grad():
            out = model(tensors)          # timm num_classes=0 → (B, feat_dim)
            out = out / out.norm(dim=-1, keepdim=True)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


# ── metrics ───────────────────────────────────────────────────────────────────
def purity_score(y_true, y_pred):
    from scipy.stats import mode
    total = 0
    for c in np.unique(y_pred):
        mask = y_pred == c
        total += mode(y_true[mask], keepdims=True).count[0]
    return total / len(y_true)


def cluster_and_eval(X, y, k=4):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED)
    pred = km.fit_predict(X)
    return pred, purity_score(y, pred), normalized_mutual_info_score(y, pred)


# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Step 1: Loading images …")
images, y = load_all_images()

# ── (A) pretrained ViT-Small ──────────────────────────────────────────────────
print("\nStep 2: Extracting PRETRAINED ViT-Small features …")
vit_pre = timm.create_model(
    TIMM_MODEL,
    pretrained=True,
    num_classes=0,
    cache_dir="/root/.cache/timm"
).to(DEVICE)
tfm = build_transform(vit_pre)

X_pre = extract(vit_pre, tfm, images)
print(f"  Shape: {X_pre.shape}")
del vit_pre; torch.cuda.empty_cache()

# ── (B) fine-tuned ViT-Small ──────────────────────────────────────────────────
print("\nStep 3: Extracting FINE-TUNED ViT-Small features …")
vit_ft = timm.create_model(
    TIMM_MODEL,
    pretrained=False,
    num_classes=0,
).to(DEVICE)

ckpt  = torch.load(CKPT, map_location=DEVICE)
state = ckpt["model_state_dict"]
# strip "backbone." prefix; timm model keys have no such prefix
bb_state = {k[len("backbone."):]: v
            for k, v in state.items()
            if k.startswith("backbone.")}
missing, unexpected = vit_ft.load_state_dict(bb_state, strict=False)
print(f"  missing={len(missing)}, unexpected={len(unexpected)}")

X_ft = extract(vit_ft, tfm, images)
print(f"  Shape: {X_ft.shape}")
del vit_ft; torch.cuda.empty_cache()

# ── clustering ────────────────────────────────────────────────────────────────
print("\nStep 4: K-Means (k=4) …")
pred_pre, pur_pre, nmi_pre = cluster_and_eval(X_pre, y)
pred_ft,  pur_ft,  nmi_ft  = cluster_and_eval(X_ft,  y)
print(f"  Pretrained → Purity={pur_pre:.4f}, NMI={nmi_pre:.4f}")
print(f"  Fine-tuned → Purity={pur_ft:.4f},  NMI={nmi_ft:.4f}")

# ── t-SNE ─────────────────────────────────────────────────────────────────────
print("\nStep 5: Running t-SNE …")
def run_tsne(X):
    return TSNE(n_components=2, perplexity=40, max_iter=1000,
                random_state=SEED, init="pca").fit_transform(X)

X2d_pre = run_tsne(X_pre)
X2d_ft  = run_tsne(X_ft)

# ── plot ──────────────────────────────────────────────────────────────────────
COLORS  = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
MARKERS = ["o", "s", "^", "D"]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
configs = [
    (axes[0,0], X2d_pre, y,        "Pretrained ViT-Small — ground truth"),
    (axes[0,1], X2d_pre, pred_pre, f"Pretrained ViT-Small — K-Means  (Purity={pur_pre:.3f}, NMI={nmi_pre:.3f})"),
    (axes[1,0], X2d_ft,  y,        "Fine-tuned ViT-Small — ground truth"),
    (axes[1,1], X2d_ft,  pred_ft,  f"Fine-tuned ViT-Small — K-Means  (Purity={pur_ft:.3f}, NMI={nmi_ft:.3f})"),
]

for ax, X2d, labels, title in configs:
    is_gt = "ground truth" in title
    for i in range(4):
        mask = labels == i
        lbl  = GENERATORS[i] if is_gt else f"Cluster {i}"
        ax.scatter(X2d[mask,0], X2d[mask,1],
                   c=COLORS[i], marker=MARKERS[i],
                   alpha=0.6, s=18, linewidths=0, label=lbl)
    ax.legend(fontsize=8, markerscale=1.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.3)

fig.suptitle(
    "Pretrained vs Fine-tuned ViT-Small Feature Space\n"
    "Can unsupervised clustering recover generator identity?",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "fig_cluster_compare.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path}")

# ── breakdown ─────────────────────────────────────────────────────────────────
for label, pred in [("Pretrained", pred_pre), ("Fine-tuned", pred_ft)]:
    print(f"\n{label} cluster composition:")
    print(f"{'':12s}" + "".join(f"{g:>10s}" for g in GENERATORS))
    for c in range(4):
        counts = Counter(y[pred == c].tolist())
        print(f"Cluster {c}    " + "".join(f"{counts.get(i,0):>10d}" for i in range(4)))

print(f"\nResult: Pretrained Purity={pur_pre:.4f} NMI={nmi_pre:.4f} | "
      f"Fine-tuned Purity={pur_ft:.4f} NMI={nmi_ft:.4f}")
