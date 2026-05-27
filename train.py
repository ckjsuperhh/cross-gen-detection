"""Train a ResNet-18 binary classifier for AI-generated image detection.

Usage:
    python train.py --data_root data --epochs 30 --batch_size 64
    python evaluate.py --model_path best.pth --test_dir data/test/glide
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


# ── config ────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str, default="data",     help="Path to data/ directory")
    parser.add_argument("--output_dir", type=str, default="outputs",  help="Directory to save checkpoints")
    parser.add_argument("--epochs",     type=int, default=30,         help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64,         help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-4,   help="Learning rate")
    parser.add_argument("--num_workers",type=int, default=4,          help="DataLoader workers")
    parser.add_argument("--seed",       type=int, default=42,         help="Random seed")
    parser.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# ── data ──────────────────────────────────────────────────────────────────────

def build_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ── model ───────────────────────────────────────────────────────────────────

def build_model(device):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Replace final FC for binary classification
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model.to(device)


# ── train / val ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
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
    print(f"Using device: {device}")

    # Data
    train_dir = Path(args.data_root) / "train"
    val_dir   = Path(args.data_root) / "val"

    train_dataset = datasets.ImageFolder(str(train_dir), transform=build_transforms(is_train=True))
    val_dataset   = datasets.ImageFolder(str(val_dir),   transform=build_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Model
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # Training
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_path = output_dir / "best.pth"
    history = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        epoch_time = time.time() - start
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"time={epoch_time:.1f}s")

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
            }, best_path)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    # Save final model and history
    torch.save(model.state_dict(), output_dir / "last.pth")
    import json
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best val_acc: {best_acc:.4f}")
    print(f"Checkpoint saved to {best_path}")


if __name__ == "__main__":
    main()
