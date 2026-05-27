"""Evaluate trained model on a test set.

Supports checkpoints from both train.py (ResNet-18, flat keys)
and train_ablation.py (Wrapper with backbone.* / head.* keys).

Usage:
    python evaluate.py --model_path outputs/best.pth --test_dir data/test/glide
    python evaluate.py --model_path outputs_ablation/vit_small_full_finetune/best.pth --test_dir data/test/biggan
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ── Wrapper (must match train_ablation.py exactly) ────────────────────────────

class Wrapper(nn.Module):
    def __init__(self, bb, head, bb_name):
        super().__init__()
        self.backbone = bb
        self.head = head
        self.bb_name = bb_name

    def forward(self, x):
        if self.bb_name == "clip_vit":
            out = self.backbone(x).pooler_output
        else:
            out = self.backbone(x)
        return self.head(out)


# ── transforms ────────────────────────────────────────────────────────────────

def build_transforms(backbone="resnet18"):
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


# ── model builder ─────────────────────────────────────────────────────────────

def build_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args  = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    backbone   = ckpt_args.get("backbone", "resnet18")
    state      = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    if backbone == "resnet18":
        bb = models.resnet18(weights=None)
        in_feat = bb.fc.in_features
        bb.fc = nn.Identity()
        head = nn.Linear(in_feat, 2)
        model = Wrapper(bb, head, backbone)
        # old train.py uses flat keys (conv1.weight, fc.weight)
        if not any(k.startswith("backbone.") for k in state):
            new_state = {}
            for k, v in state.items():
                if k.startswith("fc."):
                    new_state["head." + k[3:]] = v
                else:
                    new_state["backbone." + k] = v
            state = new_state
        model.load_state_dict(state)

    elif backbone == "vit_small":
        import timm
        bb = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
        head = nn.Linear(bb.num_features, 2)
        model = Wrapper(bb, head, backbone)
        model.load_state_dict(state)

    elif backbone == "clip_vit":
        from transformers import CLIPModel
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        bb = clip.vision_model
        head = nn.Linear(bb.config.hidden_size, 2)
        model = Wrapper(bb, head, backbone)
        model.load_state_dict(state)

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return model.to(device), backbone


# ── evaluation ────────────────────────────────────────────────────────────────

def run_eval(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = outputs.max(1)
            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_probs.append(probs.cpu())
    return (torch.cat(all_preds).numpy(),
            torch.cat(all_labels).numpy(),
            torch.cat(all_probs).numpy())


def compute_metrics(preds, labels, probs):
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score,
                                  roc_auc_score, confusion_matrix)
    return {
        "accuracy":         float(accuracy_score(labels, preds)),
        "precision":        float(precision_score(labels, preds)),
        "recall":           float(recall_score(labels, preds)),
        "f1":               float(f1_score(labels, preds)),
        "auc":              float(roc_auc_score(labels, probs)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   type=str, required=True)
    parser.add_argument("--test_dir",     type=str, required=True)
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--device",       type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_results", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device} | Model: {args.model_path} | Test: {args.test_dir}")

    model, backbone = build_model(args.model_path, device)
    dataset = datasets.ImageFolder(args.test_dir, transform=build_transforms(backbone))
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_workers)
    print(f"Test samples: {len(dataset)}")

    preds, labels, probs = run_eval(model, loader, device)
    metrics = compute_metrics(preds, labels, probs)

    print(f"\n{'Metric':<15} {'Value':>8}")
    print("-" * 25)
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k:<15} {v:>8.4f}")
    print(f"{'confusion_matrix':<15} {metrics['confusion_matrix']}")

    if args.save_results:
        Path(args.save_results).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_results, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved to {args.save_results}")


if __name__ == "__main__":
    main()
