"""KNN evaluation on frozen features (following Ojha et al.).

Extracts embeddings from a frozen backbone and classifies test samples
by cosine-similarity nearest-neighbor matching against a memory bank
built from the training set.

Usage:
    python evaluate_knn.py --backbone clip_vit --test_dir data/test/biggan
    python evaluate_knn.py --backbone resnet18 --test_dir data/test/glide
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone",   type=str, default="clip_vit",
                   choices=["resnet18", "vit_small", "clip_vit"])
    p.add_argument("--data_root",  type=str, default="data")
    p.add_argument("--test_dir",   type=str, required=True)
    p.add_argument("--k",          type=int, default=1, help="K for KNN")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_results", type=str, default=None)
    return p.parse_args()


def build_transforms(backbone):
    if backbone == "clip_vit":
        from transformers import CLIPImageProcessor
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        def transform(img):
            return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return transform
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_feature_extractor(backbone, device):
    if backbone == "resnet18":
        from torchvision import models
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
    elif backbone == "vit_small":
        import timm
        model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
    elif backbone == "clip_vit":
        from transformers import CLIPModel
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model = clip.vision_model
    else:
        raise ValueError(backbone)
    return model.to(device).eval()


@torch.no_grad()
def extract_features(model, loader, device, backbone):
    feats, labels = [], []
    for images, ys in loader:
        images = images.to(device)
        if backbone == "clip_vit":
            out = model(images).pooler_output
        elif backbone == "vit_small":
            out = model(images)
        else:
            out = model(images)
        out = F.normalize(out, dim=1)
        feats.append(out.cpu())
        labels.append(ys)
    return torch.cat(feats), torch.cat(labels)


def knn_predict(train_feats, train_labels, test_feats, k=1):
    """Cosine-similarity KNN (features are already L2-normalized)."""
    sim = test_feats @ train_feats.T  # [N_test, N_train]
    topk_vals, topk_idx = sim.topk(k, dim=1)
    pred = train_labels[topk_idx].mode(dim=1).values
    return pred


def compute_metrics(preds, labels):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    # AUC not directly available for KNN without scores; skip or use sim scores
    cm = confusion_matrix(labels, preds)
    return {"accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1),
            "confusion_matrix": cm.tolist()}


def main():
    args = get_args()
    device = torch.device(args.device)
    print(f"Backbone: {args.backbone} | K={args.k} | Device: {device}")

    transform = build_transforms(args.backbone)

    train_dir = Path(args.data_root) / "train"
    train_ds = datasets.ImageFolder(str(train_dir), transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    test_ds = datasets.ImageFolder(args.test_dir, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    model = build_feature_extractor(args.backbone, device)

    print("Extracting train features …")
    train_feats, train_labels = extract_features(model, train_loader, device, args.backbone)
    print("Extracting test features …")
    test_feats, test_labels = extract_features(model, test_loader, device, args.backbone)

    print("Running KNN …")
    preds = knn_predict(train_feats, train_labels, test_feats, k=args.k)

    metrics = compute_metrics(preds.numpy(), test_labels.numpy())
    print(f"\n{'Metric':<15} {'Value':>8}")
    print("-" * 25)
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k:<15} {v:>8.4f}")
    print(f"{'confusion_matrix':<15} {metrics['confusion_matrix']}")

    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved to {args.save_results}")


if __name__ == "__main__":
    main()
