"""Patch-level consistency evaluation.

Divides each image into a grid of patches, classifies each patch
independently, and aggregates by majority vote.
This tests whether local (patch-level) artifacts are sufficient for
detection and measures prediction consistency across regions.

Usage:
    python evaluate_patch.py --model_path outputs/best.pth --test_dir data/test/biggan
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--test_dir",   type=str, required=True)
    p.add_argument("--backbone",   type=str, default="resnet18")
    p.add_argument("--grid",       type=int, default=3, help="Grid size (3=3x3 patches)")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_results", type=str, default=None)
    return p.parse_args()


def build_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model(checkpoint_path, backbone, device):
    if backbone == "resnet18":
        model = models.resnet18(weights=None)
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, 2)
    else:
        raise NotImplementedError(backbone)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return model.to(device).eval()


@torch.no_grad()
def evaluate_patch_voting(model, dataset, device, grid=3):
    model.eval()
    all_preds = []
    all_labels = []
    consistency_scores = []

    from torchvision.transforms import functional as TF

    for idx in range(len(dataset)):
        img, label = dataset[idx]
        # img is already a tensor [C, H, W] from ImageFolder transform
        # We need to undo normalization to get back to [0,1] for patch extraction
        # Actually simpler: just work on the raw PIL image before transform

    # Re-open dataset without transform to get PIL images
    raw_ds = datasets.ImageFolder(dataset.root, transform=None)

    for idx in range(len(raw_ds)):
        img_pil, label = raw_ds[idx]
        w, h = img_pil.size
        patch_w, patch_h = w // grid, h // grid

        patch_preds = []
        for row in range(grid):
            for col in range(grid):
                left = col * patch_w
                upper = row * patch_h
                right = left + patch_w
                lower = upper + patch_h
                patch = img_pil.crop((left, upper, right, lower))

                # Apply same transform
                t = build_transforms()
                patch_tensor = t(patch).unsqueeze(0).to(device)
                output = model(patch_tensor)
                _, pred = output.max(1)
                patch_preds.append(pred.item())

        # Majority vote
        vote = max(set(patch_preds), key=patch_preds.count)
        all_preds.append(vote)
        all_labels.append(label)
        # Consistency: fraction of patches agreeing with the final vote
        consistency = patch_preds.count(vote) / len(patch_preds)
        consistency_scores.append(consistency)

    return torch.tensor(all_preds), torch.tensor(all_labels), torch.tensor(consistency_scores)


def compute_metrics(preds, labels, consistency):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds)),
        "recall": float(recall_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "mean_consistency": float(consistency.mean()),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def main():
    args = get_args()
    device = torch.device(args.device)
    print(f"Model: {args.model_path} | Test: {args.test_dir} | Grid: {args.grid}x{args.grid}")

    model = build_model(args.model_path, args.backbone, device)
    test_ds = datasets.ImageFolder(args.test_dir, transform=build_transforms())

    preds, labels, consistency = evaluate_patch_voting(model, test_ds, device, grid=args.grid)
    metrics = compute_metrics(preds, labels, consistency)

    print(f"\n{'Metric':<20} {'Value':>8}")
    print("-" * 30)
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k:<20} {v:>8.4f}")
    print(f"{'confusion_matrix':<20} {metrics['confusion_matrix']}")

    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved to {args.save_results}")


if __name__ == "__main__":
    main()
