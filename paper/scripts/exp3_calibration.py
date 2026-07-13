"""
Calibration analysis: reliability diagrams and ECE tables.

Input: paper/results/predictions.npz  (produced by exp1_validation_suite.py)
Output:
  paper/results/calibration_metrics.csv
  paper/figures/fig4_reliability_diagram.png
  paper/figures/fig9_confidence_distribution.png

Usage:
  python paper/scripts/exp3_calibration.py --predictions paper/results/predictions.npz
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _reliability_data(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Return bin_centers, bin_accuracies, bin_confidences, bin_counts."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    accs = np.zeros(n_bins)
    confs = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        if i == 0:
            mask = (confidences >= edges[i]) & (confidences <= edges[i + 1])
        else:
            mask = (confidences > edges[i]) & (confidences <= edges[i + 1])
        counts[i] = mask.sum()
        if counts[i] > 0:
            accs[i] = correct[mask].mean()
            confs[i] = confidences[mask].mean()

    return centers, accs, confs, counts


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == 0:
            mask = (confidences >= edges[i]) & (confidences <= edges[i + 1])
        else:
            mask = (confidences > edges[i]) & (confidences <= edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(correct[mask].mean() - confidences[mask].mean())
    return ece / len(labels)


def _mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    max_gap = 0.0
    for i in range(n_bins):
        if i == 0:
            mask = (confidences >= edges[i]) & (confidences <= edges[i + 1])
        else:
            mask = (confidences > edges[i]) & (confidences <= edges[i + 1])
        if mask.sum() == 0:
            continue
        max_gap = max(max_gap, abs(correct[mask].mean() - confidences[mask].mean()))
    return max_gap


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    p_true = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(p_true + 1e-10)))


def plot_reliability(centers, accs_before, confs_before, counts_before,
                     accs_after, confs_after, counts_after,
                     ece_before, ece_after, model_name, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="Perfectly calibrated")

    # Before calibration
    ax.plot(confs_before, accs_before, "o-", color="#e94560", linewidth=2.2,
            markersize=7, label=f"Before calibration (ECE={ece_before:.3f})")

    # After calibration
    ax.plot(confs_after, accs_after, "s-", color="#4ecca3", linewidth=2.2,
            markersize=7, label=f"After temperature scaling (ECE={ece_after:.3f})")

    # Highlight gaps
    for i in range(len(centers)):
        if counts_before[i] > 0:
            ax.plot([confs_before[i], confs_before[i]],
                    [accs_before[i], confs_before[i]],
                    color="#e94560", alpha=0.4, linewidth=1.5)
        if counts_after[i] > 0:
            ax.plot([confs_after[i], confs_after[i]],
                    [accs_after[i], confs_after[i]],
                    color="#4ecca3", alpha=0.4, linewidth=1.5)

    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Reliability Diagram — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved reliability diagram: {output_path}")


def plot_confidence_distribution(confidences, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confidences, bins=30, range=(0, 1), color="#6c5ce7", edgecolor="white", alpha=0.85)
    ax.axvline(confidences.mean(), color="#ffd369", linestyle="--", linewidth=2,
               label=f"Mean = {confidences.mean():.3f}")
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Ensemble Confidence", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confidence histogram: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="paper/results/predictions.npz")
    parser.add_argument("--output-dir", default="paper/results")
    parser.add_argument("--figures-dir", default="paper/figures")
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    if not os.path.exists(args.predictions):
        print(f"ERROR: predictions file not found: {args.predictions}")
        print("Run exp1_validation_suite.py first.")
        sys.exit(1)

    print(f"Loading predictions from {args.predictions}...")
    data = np.load(args.predictions)
    labels = data["labels"]
    raw_probs = data["raw_probs"]          # uncalibrated ensemble
    cal_probs = data["cal_probs"]          # calibrated ensemble
    per_model_raw = data["per_model_raw_probs"]
    per_model_cal = data["per_model_cal_probs"]
    n_models = per_model_raw.shape[1]

    # Ensemble calibration
    centers, accs_raw, confs_raw, counts_raw = _reliability_data(raw_probs, labels, args.n_bins)
    _, accs_cal, confs_cal, counts_cal = _reliability_data(cal_probs, labels, args.n_bins)
    ece_raw = _ece(raw_probs, labels, args.n_bins)
    ece_cal = _ece(cal_probs, labels, args.n_bins)
    mce_raw = _mce(raw_probs, labels, args.n_bins)
    mce_cal = _mce(cal_probs, labels, args.n_bins)
    nll_raw = _nll(raw_probs, labels)
    nll_cal = _nll(cal_probs, labels)

    plot_reliability(centers, accs_raw, confs_raw, counts_raw,
                     accs_cal, confs_cal, counts_cal,
                     ece_raw, ece_cal, "Ensemble (EfficientNetV2S + ConvNeXtTiny)",
                     os.path.join(args.figures_dir, "fig4_reliability_diagram.png"))

    plot_confidence_distribution(np.max(cal_probs, axis=1),
                                 os.path.join(args.figures_dir, "fig9_confidence_distribution.png"))

    # Calibration metrics CSV
    rows = [
        ["ensemble", "raw", f"{ece_raw:.4f}", f"{mce_raw:.4f}", f"{nll_raw:.4f}"],
        ["ensemble", "calibrated", f"{ece_cal:.4f}", f"{mce_cal:.4f}", f"{nll_cal:.4f}"],
    ]
    for m in range(n_models):
        ece_raw_m = _ece(per_model_raw[:, m, :], labels, args.n_bins)
        ece_cal_m = _ece(per_model_cal[:, m, :], labels, args.n_bins)
        mce_raw_m = _mce(per_model_raw[:, m, :], labels, args.n_bins)
        mce_cal_m = _mce(per_model_cal[:, m, :], labels, args.n_bins)
        nll_raw_m = _nll(per_model_raw[:, m, :], labels)
        nll_cal_m = _nll(per_model_cal[:, m, :], labels)
        rows.append([f"model_{m}", "raw", f"{ece_raw_m:.4f}", f"{mce_raw_m:.4f}", f"{nll_raw_m:.4f}"])
        rows.append([f"model_{m}", "calibrated", f"{ece_cal_m:.4f}", f"{mce_cal_m:.4f}", f"{nll_cal_m:.4f}"])

    csv_path = os.path.join(args.output_dir, "calibration_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mode", "ece_15", "mce_15", "nll"])
        writer.writerows(rows)
    print(f"  Saved metrics: {csv_path}")

    print("\nCalibration summary:")
    print(f"  Ensemble ECE-15  raw={ece_raw:.4f}  calibrated={ece_cal:.4f}")
    print(f"  Ensemble MCE-15  raw={mce_raw:.4f}  calibrated={mce_cal:.4f}")
    print(f"  Ensemble NLL     raw={nll_raw:.4f}  calibrated={nll_cal:.4f}")


if __name__ == "__main__":
    main()
