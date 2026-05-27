"""Ablation training script supporting multiple backbones and strategies.

Backbones:
  resnet18   – torchvision ResNet-18 (IMAGENET1K_V1)
  vit_small  – timm vit_small_patch16_224
  clip_vit   – CLIP ViT-B/32 (OpenAI)

Strategies:
  full_finetune – update all parameters (default)
  linear_probe  – freeze backbone, train only a linear head

Usage:
  python train_ablation.py --backbone clip_vit --strategy linear_probe --epochs 30
  python train_ablation.py --backbone vit_small --strategy full_finetune --epochs 30
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  type=str, default="data")
    p.add_argument("--output_dir", type=str, default="outputs_ablation")
    p.add_argument("--backbone",   type=str, default="resnet18",
                   choices=["resnet18", "vit_small", "clip_vit"])
    p.add_argument("--strategy",   type=str, default="full_finetune",
                   choices=["full_finetune", "linear_probe"])
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── transforms ────────────────────────────────────────────────────────────────

def build_transforms(is_train=True, backbone="resnet18"):
    if backbone == "clip_vit":
        # CLIP expects 224x224, normalized with its own stats
        from transformers import CLIPImageProcessor
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        def transform(img):
            return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return transform

    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── model builders ────────────────────────────────────────────────────────────

def build_model(backbone, strategy, device):
    if backbone == "resnet18":
        from torchvision import models
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_feat = model.fc.in_features
        model.fc = nn.Identity()
        head = nn.Linear(in_feat, 2)

    elif backbone == "vit_small":
        import timm
        model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
        in_feat = model.num_features
        head = nn.Linear(in_feat, 2)

    elif backbone == "clip_vit":
        from transformers import CLIPModel
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model = clip.vision_model
        in_feat = model.config.hidden_size
        head = nn.Linear(in_feat, 2)

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    # strategy: freeze or not
    if strategy == "linear_probe":
        for p in model.parameters():
            p.requires_grad = False

    class Wrapper(nn.Module):
        def __init__(self, backbone, head, bb_name):
            super().__init__()
            self.backbone = backbone
            self.head = head
            self.bb_name = bb_name

        def forward(self, x):
            if self.bb_name == "clip_vit":
                out = self.backbone(x).pooler_output  # [B, hidden]
            elif self.bb_name == "vit_small":
                out = self.backbone(x)  # timm returns [B, num_features] when num_classes=0
            else:
                out = self.backbone(x)
            return self.head(out)

    wrapped = Wrapper(model, head, backbone).to(device)
    return wrapped


# ── train / eval ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, backbone):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device} | Backbone: {args.backbone} | Strategy: {args.strategy}")

    train_dir = Path(args.data_root) / "train"
    val_dir   = Path(args.data_root) / "val"

    transform = build_transforms(is_train=True, backbone=args.backbone)
    val_transform = build_transforms(is_train=False, backbone=args.backbone)

    train_dataset = datasets.ImageFolder(str(train_dir), transform=transform)
    val_dataset   = datasets.ImageFolder(str(val_dir),   transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    model = build_model(args.backbone, args.strategy, device)

    criterion = nn.CrossEntropyLoss()
    # linear probe uses higher lr
    lr = args.lr * 10 if args.strategy == "linear_probe" else args.lr
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    out_dir = Path(args.output_dir) / f"{args.backbone}_{args.strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_path = out_dir / "best.pth"
    history = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tloss, tacc = train_epoch(model, train_loader, criterion, optimizer, device, args.backbone)
        vloss, vacc = evaluate(model, val_loader, criterion, device)
        scheduler.step(vacc)
        elapsed = time.time() - start
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={tloss:.4f} train_acc={tacc:.4f} | "
              f"val_loss={vloss:.4f} val_acc={vacc:.4f} | {elapsed:.1f}s")
        history.append({"epoch": epoch, "train_loss": tloss, "train_acc": tacc,
                        "val_loss": vloss, "val_acc": vacc})
        if vacc > best_acc:
            best_acc = vacc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_acc": vacc, "args": vars(args)}, best_path)
            print(f"  -> Saved best (val_acc={vacc:.4f})")

    torch.save(model.state_dict(), out_dir / "last.pth")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val_acc: {best_acc:.4f} | Saved to {best_path}")


if __name__ == "__main__":
    main()
