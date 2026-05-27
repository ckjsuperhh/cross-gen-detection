# Why Detectors Fail on Unseen Generators

> Ablation, Feature-Space Analysis, and a Route Towards Mixture-of-Experts Detection

Systematic ablation study on the GenImage benchmark comparing ResNet-18, ViT-Small, and CLIP-ViT for cross-generator fake image detection. Includes patch-level consistency analysis and unsupervised clustering experiments that validate a Mixture-of-Experts (MoE) routing approach.

---

## 🔬 Key Findings

1. **Full fine-tuning dominates in-distribution** — ResNet-18 and ViT-Small achieve ~98–99% on GLIDE (training generator).
2. **All methods collapse on unseen diffusion generators** — ADM and VQDM drop to 52–65%, barely above chance.
3. **CLIP KNN generalises best on hard generators** — Frozen vision-language features reach 64.9% on ADM and 64.7% on VQDM, outperforming all fine-tuned models.
4. **Patch errors are systematic, not localised** — 3×3 grid majority vote shows ~90% patch consistency despite ~61% accuracy, pointing to global feature mismatch.
5. **Fine-tuned features implicitly encode generator identity** — Unsupervised K-Means clustering of ViT-Small features achieves **Purity=0.559 / NMI=0.378** with no generator labels, providing direct empirical support for MoE routing.

---

## 📊 Results at a Glance

| Method | Backbone | Strategy | GLIDE | BigGAN | ADM | VQDM |
|---|---|---|---|---|---|---|
| Baseline | ResNet-18 | Full FT | 98.2 | **99.3** | 56.2 | 53.0 |
| Ablation | ViT-Small | Full FT | 98.7 | 97.8 | 54.9 | 52.1 |
| Ablation | ResNet-18 | Linear Probe | 92.0 | 91.9 | 59.4 | 54.7 |
| Ablation | CLIP-ViT | Linear Probe | **99.3** | 98.3 | 61.3 | 55.6 |
| Ojha-style | CLIP-ViT | KNN (k=1) | 85.4 | 81.7 | **64.9** | **64.7** |
| Patch | ResNet-18 | 3×3 Vote | 61.5 | 54.0 | 60.8 | 61.8 |

---

## 🏗️ Project Structure

```
.
├── prepare_dataset.py               # Download GenImage subset from HuggingFace
├── train.py                         # ResNet-18 baseline full fine-tuning
├── train_ablation.py                # Multi-backbone / multi-strategy trainer
├── evaluate.py                      # Standard evaluation (acc, precision, recall, F1, AUC)
├── evaluate_knn.py                  # KNN evaluation on frozen features
├── evaluate_patch.py                # Patch-level 3×3 majority vote
├── plot_results.py                  # Generate comparison figures
├── plot_curves.py                   # Plot training curves
├── unsupervised_cluster.py          # Pretrained CLIP K-Means clustering
├── unsupervised_cluster_compare.py  # Pretrained vs fine-tuned ViT-Small clustering
├── project_report.md                # Full workflow documentation
└── format/
    ├── paper.tex                    # Mini paper (NeurIPS 2023 format, 9 pages)
    ├── paper.pdf                    # Compiled PDF
    ├── ref.bib                      # Bibliography
    └── fig*.png                     # All publication figures
```

---

## 🛠️ Environment

- Python 3.12
- PyTorch 2.6.0+cu124
- CUDA 12.6
- GPU: RTX 4060 Laptop (8 GB)

### Setup

```bash
pip install torch torchvision timm transformers scikit-learn matplotlib pillow
```

> **Note:** If direct HuggingFace access is blocked, set `HF_ENDPOINT=https://hf-mirror.com` and `NO_PROXY="*"`.

---

## 🚀 Quick Start

### 1. Prepare Dataset

```bash
python prepare_dataset.py
```

Downloads GLIDE, BigGAN, ADM, and VQDM from `shimei123/Genimage` (via hf-mirror.com) and samples balanced train/val/test splits.

### 2. Train Baseline (ResNet-18 Full FT)

```bash
python train.py
```

### 3. Run Ablation Experiments

```bash
python train_ablation.py --backbone resnet18 --strategy full_finetune
python train_ablation.py --backbone vit_small --strategy full_finetune
python train_ablation.py --backbone clip_vit --strategy linear_probe
```

### 4. Evaluate

```bash
# Standard evaluation
python evaluate.py --checkpoint outputs_ablation/vit_small_full_finetune/best.pth \
                   --backbone vit_small --strategy full_finetune --test_dir data/test/adm

# KNN
python evaluate_knn.py --backbone clip_vit --test_dir data/test/vqdm

# Patch-level analysis
python evaluate_patch.py --checkpoint outputs/best.pth --test_dir data/test/biggan
```

### 5. Unsupervised Clustering (MoE Feasibility)

```bash
# Compare pretrained vs fine-tuned ViT-Small features
python unsupervised_cluster_compare.py
```

Generates `outputs_ablation/figures/fig_cluster_compare.png` showing t-SNE projections and K-Means purity/NMI metrics.

### 6. Plot Figures

```bash
python plot_results.py
python plot_curves.py
```

---

## 📈 Figures

| Figure | Description |
|---|---|
| Fig 1 | Accuracy comparison across all methods and generators |
| Fig 2 | ResNet-18 full fine-tune training curves |
| Fig 3 | Cross-generator generalisation bar chart |
| Fig 4 | Patch accuracy vs. consistency |
| Fig 5 | Pretrained CLIP clustering (t-SNE) |
| Fig 6 | **Pretrained vs fine-tuned ViT-Small clustering** (key result) |

---

## 📄 Paper

The complete mini paper is in `format/paper.tex` (NeurIPS 2023 template) and compiled to `format/paper.pdf` (9 pages).

To recompile:

```bash
cd format
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

---

## 🔮 Future Work: Mixture-of-Experts Framework

Based on the clustering evidence (Section 4.5 of the paper), we propose a three-component MoE architecture:

1. **GAN Expert** — CNN exploiting frequency-domain spectral artifacts
2. **Diffusion Expert** — DIRE/FIRE-style reconstruction-error detector
3. **Universal Router** — Lightweight head on fine-tuned detection features; routes by generator family without explicit generator labels

The key insight: **detection-oriented supervision implicitly structures the feature space by generator family**, making routing feasible even without generator-identity training data.

---

## 📚 References

- Wang et al., "CNN-generated images are surprisingly easy to spot... for now" (CVPR 2020)
- Ojha et al., "Towards Universal Fake Image Detectors that Generalize Across Generative Models" (CVPR 2023)
- Zhu et al., "GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image" (NeurIPS 2023)
- Wang et al., "DIRE for Diffusion-Generated Image Detection" (ICCV 2023)
- He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
- Dosovitskiy et al., "An Image is Worth 16×16 Words" (ICLR 2021)
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (ICML 2021)

---

## 📝 License

This project is for academic / coursework purposes.
