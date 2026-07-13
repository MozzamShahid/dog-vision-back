"""
Generate data-dependent figures from exp1_validation_suite.py outputs.

Produces:
  paper/figures/fig2_confusion_matrix_top20.png  (top-20 most confused breeds)
  paper/figures/fig7_agreement_heatmap.png       (model agreement matrix)
  paper/figures/fig8_per_breed_accuracy.png      (sorted per-breed accuracy)

Usage:
  python paper/scripts/exp1_figures.py --results-dir paper/results
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def top_confused_breeds(confusion: np.ndarray, breeds: list, n: int = 20):
    """Return indices of top-n breeds by total misclassifications."""
    off_diag_sum = confusion.sum(axis=1) - np.diag(confusion)
    top_indices = np.argsort(off_diag_sum)[-n:][::-1]
    return top_indices


def plot_confusion_matrix_top20(confusion: np.ndarray, breeds: list, output_path: str):
    top_indices = top_confused_breeds(confusion, breeds, n=20)
    sub = confusion[np.ix_(top_indices, top_indices)]
    sub_labels = [breeds[i].replace("_", " ").title() for i in top_indices]

    # Normalize by row (true label) for readability
    row_sums = sub.sum(axis=1, keepdims=True)
    sub_norm = np.divide(sub, row_sums, out=np.zeros_like(sub, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(sub_norm, annot=False, cmap="YlOrRd", xticklabels=sub_labels,
                yticklabels=sub_labels, cbar_kws={"label": "Fraction of true class"}, ax=ax)
    ax.set_title("Normalised Confusion Matrix — Top 20 Most Confused Breeds", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_agreement_heatmap(per_model_probs: np.ndarray, labels: np.ndarray, breeds: list,
                           output_path: str):
    """Plot agreement matrix: how often each pair of models predict the same breed."""
    n_models = per_model_probs.shape[1]
    preds = np.argmax(per_model_probs, axis=2)  # (N, M)

    agreement = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            agreement[i, j] = (preds[:, i] == preds[:, j]).mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(agreement, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=[f"M{i+1}" for i in range(n_models)],
                yticklabels=[f"M{i+1}" for i in range(n_models)],
                vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Agreement"})
    ax.set_title("Inter-Model Prediction Agreement", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_per_breed_accuracy(csv_path: str, output_path: str):
    import csv
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Sort by accuracy ascending
    rows.sort(key=lambda x: float(x["accuracy"]))
    names = [r["breed"].replace("_", " ").title() for r in rows]
    accs = [float(r["accuracy"]) * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(10, 22))
    colors = ["#e94560" if a < 80 else "#ffd369" if a < 95 else "#4ecca3" for a in accs]
    ax.barh(names, accs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(accs), color="white", linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(accs):.1f}%")
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_ylabel("Breed", fontsize=12)
    ax.set_title("Per-Breed Accuracy (Sorted)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="paper/results")
    parser.add_argument("--figures-dir", default="paper/figures")
    parser.add_argument("--labels-path", default="data/unique_breeds.json")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    pred_path = os.path.join(args.results_dir, "predictions.npz")
    if not os.path.exists(pred_path):
        print(f"ERROR: {pred_path} not found. Run exp1_validation_suite.py first.")
        return

    with open(args.labels_path) as f:
        breeds = json.load(f)

    data = np.load(pred_path)
    confusion = np.load(os.path.join(args.results_dir, "confusion_matrix.npy"))

    plot_confusion_matrix_top20(confusion, breeds,
                                os.path.join(args.figures_dir, "fig2_confusion_matrix_top20.png"))
    plot_agreement_heatmap(data["per_model_cal_probs"], data["labels"], breeds,
                           os.path.join(args.figures_dir, "fig7_agreement_heatmap.png"))
    plot_per_breed_accuracy(os.path.join(args.results_dir, "per_breed_accuracy.csv"),
                            os.path.join(args.figures_dir, "fig8_per_breed_accuracy.png"))


if __name__ == "__main__":
    main()
