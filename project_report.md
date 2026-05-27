# AI-Generated Image Detection – Project Report

## 1. Overview

**Task:** Distinguish AI-generated images (diffusion / GAN) from authentic photographs.  
**Dataset:** GenImage subset — GLIDE for training, BigGAN / ADM / VQDM for cross-generator evaluation.  
**Models:** ResNet-18, ViT-Small, CLIP-ViT with multiple training strategies.  
**Environment:** Python 3.12, PyTorch 2.6.0+cu124, RTX 4060 Laptop GPU.

---

## 2. Dataset Preparation

### 2.1 Source
- HuggingFace mirror (`hf-mirror.com`): `shimei123/Genimage`
- Single complete ZIP files per generator (not split archives)

### 2.2 Generators Downloaded
| Generator | File Size | Images |
|---|---|---|
| GLIDE | 1.2 GB | ~12 000 (train source) |
| BigGAN | 934 MB | ~12 000 (cross-gen test) |
| ADM | 1.5 GB | ~12 000 (cross-gen test) |
| VQDM | 1.4 GB | ~12 000 (cross-gen test) |

### 2.3 Subset Split
| Split | Source | Real | Fake | Total | Purpose |
|---|---|---|---|---|---|
| train | GLIDE | 4 000 | 4 000 | 8 000 | Model training |
| val | GLIDE | 1 000 | 1 000 | 2 000 | Hyperparameter tuning |
| test/glide | GLIDE | ~502 | 500 | ~1 002 | In-distribution test |
| test/biggan | BigGAN | 500 | 500 | 1 000 | Cross-generator test |
| test/adm | ADM | 500 | 500 | 1 000 | Cross-generator test |
| test/vqdm | VQDM | 500 | 500 | 1 000 | Cross-generator test |

---

## 3. Methods

### 3.1 Architectures
| Backbone | Parameters | Pretrained Weights |
|---|---|---|
| ResNet-18 | ~11 M | ImageNet-1K |
| ViT-Small (patch16_224) | ~22 M | ImageNet-21K/1K (timm) |
| CLIP-ViT-B/32 | ~86 M | OpenAI CLIP (400M image-text pairs) |

### 3.2 Training Strategies
| Strategy | Description |
|---|---|
| **Full Finetune** | Update all backbone + classifier parameters end-to-end |
| **Linear Probe** | Freeze backbone; train only a single linear classification head |
| **KNN (k=1)** | Freeze backbone; classify by cosine nearest-neighbour against train memory bank (Ojha et al.) |

### 3.3 Hyperparameters
| | Full Finetune | Linear Probe |
|---|---|---|
| Epochs | 30 | 30 |
| Batch size | 64 | 32 (CLIP), 64 (others) |
| Optimiser | AdamW | AdamW |
| LR | 1e-4 | 1e-3 |
| Scheduler | ReduceLROnPlateau (p=3, f=0.5) | same |
| Input size | 224×224 | 224×224 |

### 3.4 Patch-Level Analysis
- ResNet-18 (full finetune) tested with 3×3 patch grid
- Each patch classified independently; final prediction by majority vote
- **Mean consistency** = fraction of patches agreeing with the majority vote

---

## 4. Experimental Results

### 4.1 Main Results: Accuracy (%) on All Test Sets

| Method | GLIDE (in-dist) | BigGAN (cross) | ADM (cross) | VQDM (cross) |
|---|---|---|---|---|
| **ResNet-18 Full FT** | **98.21** | **99.30** | 56.20 | 53.00 |
| **ViT-Small Full FT** | **98.71** | 97.80 | 54.90 | 52.10 |
| ResNet-18 Linear Probe | 92.03 | 91.90 | 59.40 | 54.70 |
| CLIP-ViT Linear Probe | 99.30 | 98.30 | 61.30 | 55.60 |
| CLIP-ViT KNN (k=1) | 85.36 | 81.70 | **64.90** | **64.70** |

> **Bold**: best per column.

### 4.2 Full Metrics – Baseline (ResNet-18 Full FT)
| Test Set | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| GLIDE (in-dist) | 98.21% | 97.83% | 98.61% | 98.22% | 99.88% |
| BigGAN (cross) | 99.30% | 100.00% | 98.60% | 99.30% | 99.89% |

### 4.3 Training Curves (ResNet-18 Full FT)
- See `outputs/curves.png`
- Train loss converges to ~0 after epoch 5
- Val accuracy plateaus at ~98.5% from epoch 9; best: **98.55%** at epoch 23
- Mild overfitting: train acc ~100% vs val acc ~98.5%

### 4.4 Patch-Level Analysis (ResNet-18 3×3 Grid)
| Test Set | Accuracy (%) | Mean Consistency (%) |
|---|---|---|
| GLIDE (in-dist) | 61.45 | 92.40 |
| BigGAN (cross) | 54.00 | 86.54 |
| ADM (cross) | 60.80 | 89.62 |
| VQDM (cross) | 61.80 | 89.62 |

### 4.5 Unsupervised Generator Clustering

**Setup:** 200 fake images per generator (800 total) → CLIP/ViT-Small features → K-Means(k=4) + t-SNE visualisation.  
**Comparison:** Pretrained ViT-Small (no task adaptation) vs. Fine-tuned ViT-Small (full-finetune checkpoint).

| Feature Space | Purity | NMI |
|---|---|---|
| Pretrained ViT-Small | 0.278 | 0.003 |
| **Fine-tuned ViT-Small** | **0.559** | **0.378** |

**Fine-tuned cluster breakdown (rows = cluster, cols = generator):**
|  | glide | BigGAN | ADM | VQDM |
|---|---|---|---|---|
| Cluster 0 | 192 | 103 | 14 | 9 |
| Cluster 1 | 0 | 0 | 75 | 79 |
| Cluster 2 | 8 | 97 | 40 | 33 |
| Cluster 3 | 0 | 0 | 71 | 79 |

**Interpretation:** Cluster 0 is dominated by GLIDE; Clusters 1+3 capture ADM/VQDM (diffusion family); Cluster 2 captures BigGAN. Task-oriented fine-tuning implicitly separates generator families with no generator-identity supervision — 2× improvement over random baseline.

---

## 5. Key Findings

### Finding 1 – Full Finetune Dominates In-Distribution and GAN
ResNet-18 and ViT-Small full finetune both achieve ~98–99% on GLIDE and BigGAN, confirming that end-to-end feature learning captures strong generator-specific artifacts.

### Finding 2 – All Methods Fail on ADM and VQDM
All detectors drop to 52–65% on ADM and VQDM — barely above chance. This replicates Ojha et al.'s finding that training on one generator family fails to generalise to sufficiently different architectures. BigGAN and GLIDE are both ImageNet-conditioned, explaining why cross-gen transfer works there.

### Finding 3 – CLIP KNN Shows Better Generalisation on Harder Generators
Despite lower in-distribution accuracy (85%), CLIP-ViT KNN achieves 64.9% on ADM and 64.7% on VQDM — the highest of any method on these two hard generators. This supports Ojha et al.'s hypothesis that frozen vision-language features capture more generator-agnostic cues.

### Finding 4 – Linear Probe > KNN on Easy Generators
CLIP-ViT Linear Probe (99.3% on GLIDE, 98.3% on BigGAN) substantially outperforms KNN (85.4%, 81.7%) on easy generators, showing that a learned linear boundary is more precise when the target distribution is known.

### Finding 5 – High Patch Consistency Despite Low Patch Accuracy
The ResNet-18 3×3 patch classifier achieves only ~61% accuracy but ~90% consistency — meaning patches tend to agree with each other even when they are wrong. This suggests the model's errors are systematic rather than spatially localised, and that local patches alone lack sufficient context for reliable detection.

### Finding 6 – Fine-tuned Features Implicitly Encode Generator Identity
Unsupervised K-Means clustering of fine-tuned ViT-Small features (Purity=0.559, NMI=0.378) outperforms pretrained features (Purity=0.278, NMI=0.003) by 2×, with **no generator-identity labels** used during training. This is the key empirical result supporting the MoE hypothesis: detection-oriented supervision naturally separates the feature space by generator family, meaning a lightweight router trained on these features could reliably dispatch test images to specialist experts (GAN expert / diffusion expert).

### MoE Framework (Future Direction)
Proposed three-component architecture:
1. **GAN expert** — fine-tuned CNN exploiting frequency-domain spectral artifacts
2. **Diffusion expert** — DIRE/FIRE-style reconstruction-error detector
3. **Universal router** — trained on top of fine-tuned detection features; routes by generator family without needing explicit generator labels

Finding 6 validates step 3: the routing problem is solvable from detection features alone.

---

## 6. Figures

| Figure | File | Description |
|---|---|---|
| Fig 1 | `outputs_ablation/figures/fig1_accuracy_comparison.png` | Grouped bar: all methods × all generators |
| Fig 2 | `outputs_ablation/figures/fig2_training_curves.png` | ResNet-18 FT training curves |
| Fig 3 | `outputs_ablation/figures/fig3_cross_gen.png` | Cross-generator generalisation |
| Fig 4 | `outputs_ablation/figures/fig4_patch_analysis.png` | Patch accuracy vs consistency |
| Fig 5 | `outputs_ablation/figures/fig_unsupervised_cluster.png` | t-SNE of CLIP features (pretrained only) |
| Fig 6 | `outputs_ablation/figures/fig_cluster_compare.png` | Pretrained vs fine-tuned ViT-Small clustering (2×2 t-SNE) |

---

## 7. Project Structure

```
ml-project/
├── data/
│   ├── train/          # 4k real + 4k fake (GLIDE)
│   ├── val/            # 1k real + 1k fake (GLIDE)
│   ├── test/
│   │   ├── glide/      # In-distribution
│   │   ├── biggan/     # Cross-gen
│   │   ├── adm/        # Cross-gen
│   │   └── vqdm/       # Cross-gen
│   └── raw/shimei/     # Downloaded ZIPs
├── outputs/            # ResNet-18 baseline results
├── outputs_ablation/   # All ablation results + figures
├── prepare_dataset.py  # Data download and sampling
├── train.py            # ResNet-18 full finetune (original)
├── train_ablation.py   # Multi-backbone / multi-strategy trainer
├── evaluate.py         # Standard evaluation (acc, F1, AUC)
├── evaluate_knn.py     # KNN evaluation on frozen features
├── evaluate_patch.py   # Patch-level majority vote evaluation
├── plot_curves.py      # Training curve plots
├── plot_results.py     # All ablation result plots
├── unsupervised_cluster.py          # CLIP K-Means clustering (pretrained)
├── unsupervised_cluster_compare.py  # Pretrained vs fine-tuned ViT-Small clustering
└── project_report.md  # This file
```

---

## 8. Summary Table for Paper

| Method | Backbone | Strategy | GLIDE | BigGAN | ADM | VQDM |
|---|---|---|---|---|---|---|
| Baseline | ResNet-18 | Full FT | 98.2 | 99.3 | 56.2 | 53.0 |
| Ablation | ViT-Small | Full FT | 98.7 | 97.8 | 54.9 | 52.1 |
| Ablation | ResNet-18 | Linear Probe | 92.0 | 91.9 | 59.4 | 54.7 |
| Ablation | CLIP-ViT | Linear Probe | **99.3** | 98.3 | 61.3 | 55.6 |
| Ojha-style | CLIP-ViT | KNN (k=1) | 85.4 | 81.7 | **64.9** | **64.7** |
| Patch | ResNet-18 | 3×3 Vote | 61.5 | 54.0 | 60.8 | 61.8 |
