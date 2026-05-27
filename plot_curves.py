"""Plot training curves from history.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    history_path = Path("outputs/history.json")
    save_path = Path("outputs/curves.png")

    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    train_acc  = [h["train_acc"]  for h in history]
    val_acc    = [h["val_acc"]    for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Loss
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train", linewidth=1.5)
    ax.plot(epochs, val_loss,   label="Val",   linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, train_acc, label="Train", linewidth=1.5)
    ax.plot(epochs, val_acc,   label="Val",   linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.94, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure to {save_path}")


if __name__ == "__main__":
    main()
