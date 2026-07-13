"""
Generate training curves (fig10) from CSV training logs.

Input: training CSV file with columns epoch, loss, val_loss, accuracy, val_accuracy
Output: paper/figures/fig10_training_curves.png

Usage:
  python paper/scripts/exp7_training_curves.py --log path/to/training_log.csv
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="training_log.csv", help="Path to training CSV log")
    parser.add_argument("--output", default="paper/figures/fig10_training_curves.png")
    parser.add_argument("--title", default="Training and Validation Curves")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"ERROR: training log not found: {args.log}")
        print("Provide a CSV with columns: epoch, loss, val_loss, accuracy, val_accuracy")
        return

    df = pd.read_csv(args.log)
    required = {"epoch", "loss", "val_loss", "accuracy", "val_accuracy"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: log missing columns: {missing}")
        print(f"Found columns: {list(df.columns)}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax = axes[0]
    ax.plot(df["epoch"], df["loss"], "-", color="#e94560", linewidth=2, label="Training loss")
    ax.plot(df["epoch"], df["val_loss"], "--", color="#4ecca3", linewidth=2, label="Validation loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Loss", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(df["epoch"], df["accuracy"] * 100, "-", color="#e94560", linewidth=2, label="Training accuracy")
    ax.plot(df["epoch"], df["val_accuracy"] * 100, "--", color="#4ecca3", linewidth=2, label="Validation accuracy")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)

    fig.suptitle(args.title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
