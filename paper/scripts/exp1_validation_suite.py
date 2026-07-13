"""
Full validation-set evaluation for the research paper.

Designed to run on Kaggle, Hugging Face Spaces, or any machine with:
  - the trained models in models/
  - the Stanford Dogs images in data/Images/
  - data/unique_breeds.json

What it produces (all under paper/results/ by default):
  confusion_matrix.npy          120x120 confusion matrix (ensemble, calibrated)
  confusion_matrix_raw.npy      120x120 confusion matrix (ensemble, uncalibrated)
  per_breed_accuracy.csv        per-breed correct/total/accuracy
  predictions.npz               arrays: labels, cal_probs, raw_probs, per_model_cal_probs
  confidence_histogram.csv      confidence bin counts for fig 9
  metrics.json                  top-1/3/5, agreement rate, ECE estimates

Usage:
  # Full run (slow on CPU, ~20-60 min on M1 Mac, faster on Kaggle GPU/TPU)
  python paper/scripts/exp1_validation_suite.py --models-dir models --images-dir data/Images

  # Quick smoke test on 100 images
  python paper/scripts/exp1_validation_suite.py --models-dir models --images-dir data/Images --max-images 100
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.predictor import DogBreedPredictor


def _ece_from_probs(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error from softmax probabilities."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if i == 0:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / len(labels))


def _top_k_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Top-k accuracy from softmax probabilities."""
    top_k = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
    correct = np.any(top_k == labels[:, None], axis=1)
    return float(correct.mean())


def _hist_counts(confidences: np.ndarray, n_bins: int = 20) -> tuple:
    """Return bin edges and counts for a confidence histogram."""
    counts, edges = np.histogram(confidences, bins=n_bins, range=(0.0, 1.0))
    return edges, counts


def main():
    parser = argparse.ArgumentParser(description="Research paper validation suite")
    parser.add_argument("--models-dir", default="models", help="Directory containing .keras/.h5 models")
    parser.add_argument("--labels-path", default="data/unique_breeds.json", help="Breed labels JSON")
    parser.add_argument("--images-dir", default="data/Images", help="Stanford Dogs Images/ folder")
    parser.add_argument("--output-dir", default="paper/results", help="Where to write results")
    parser.add_argument("--max-images", type=int, default=0, help="0 = all images; else random sample")
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation (2-8x slower)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Dog Vision Research Paper — Validation Suite")
    print("=" * 70)

    # Load labels
    print("\n[1/4] Loading labels...")
    with open(args.labels_path) as f:
        breeds = json.load(f)
    num_breeds = len(breeds)
    breed_to_idx = {b: i for i, b in enumerate(breeds)}
    print(f"      {num_breeds} breeds loaded")

    # Load models
    print("\n[2/4] Loading predictor...")
    model_files = sorted(glob.glob(os.path.join(args.models_dir, "*.keras")))
    if not model_files:
        model_files = sorted(glob.glob(os.path.join(args.models_dir, "*.h5")))
    if not model_files:
        raise FileNotFoundError(f"No models found in {args.models_dir}")

    predictor = DogBreedPredictor(model_paths=model_files, labels_path=args.labels_path)
    num_models = predictor.num_models
    print(f"      {num_models} model(s) loaded")
    print(f"      Temperatures: {predictor.temperatures}")

    # Index images
    print("\n[3/4] Indexing validation images...")
    image_paths = []
    true_labels = []
    for breed_folder in sorted(os.listdir(args.images_dir)):
        breed_path = os.path.join(args.images_dir, breed_folder)
        if not os.path.isdir(breed_path):
            continue
        if breed_folder not in breed_to_idx:
            print(f"      Warning: skipping unknown breed folder '{breed_folder}'")
            continue
        for img_name in os.listdir(breed_path):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(breed_path, img_name))
                true_labels.append(breed_to_idx[breed_folder])

    n_total = len(image_paths)
    print(f"      {n_total} images found")

    if args.max_images > 0 and args.max_images < n_total:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_total, args.max_images, replace=False)
        image_paths = [image_paths[i] for i in indices]
        true_labels = [true_labels[i] for i in indices]
        print(f"      Using random sample of {args.max_images} images")

    n = len(image_paths)
    labels = np.array(true_labels, dtype=np.int32)

    # Storage arrays
    cal_probs = np.zeros((n, num_breeds), dtype=np.float32)      # ensemble calibrated
    raw_probs = np.zeros((n, num_breeds), dtype=np.float32)      # ensemble uncalibrated
    per_model_cal_probs = np.zeros((n, num_models, num_breeds), dtype=np.float32)
    per_model_raw_probs = np.zeros((n, num_models, num_breeds), dtype=np.float32)

    confusion_cal = np.zeros((num_breeds, num_breeds), dtype=np.int32)
    confusion_raw = np.zeros((num_breeds, num_breeds), dtype=np.int32)
    agreement_count = 0
    total_evaluated = 0

    # Evaluate
    print("\n[4/4] Evaluating images...")
    t0 = time.time()
    for i, (path, true_idx) in enumerate(zip(image_paths, true_labels)):
        with open(path, "rb") as f:
            img_bytes = f.read()

        try:
            result = predictor.predict(img_bytes, use_tta=args.tta, return_raw=True)
        except Exception as e:
            print(f"      Error on {path}: {e}")
            continue

        total_evaluated += 1

        # Reconstruct ensemble probability vectors from result
        cal_vec = np.zeros(num_breeds, dtype=np.float32)
        for tp in result["top_k"]:
            cal_vec[breed_to_idx[tp["breed"]]] = tp["confidence"]
        cal_probs[i] = cal_vec
        cal_pred = np.argmax(cal_vec)
        confusion_cal[true_idx, cal_pred] += 1

        raw_vec = np.zeros(num_breeds, dtype=np.float32)
        if "raw" in result:
            for tp in result["raw"]["top_k"]:
                raw_vec[breed_to_idx[tp["breed"]]] = tp["confidence"]
        else:
            raw_vec = cal_vec.copy()
        raw_probs[i] = raw_vec
        raw_pred = np.argmax(raw_vec)
        confusion_raw[true_idx, raw_pred] += 1

        # Per-model predictions
        for m_idx, ip in enumerate(result["ensemble"]["individual_predictions"]):
            per_model_cal_probs[i, m_idx, breed_to_idx[ip["breed"]]] = ip["confidence"]
        if "raw" in result:
            for m_idx, ip in enumerate(result["raw"]["individual_predictions"]):
                per_model_raw_probs[i, m_idx, breed_to_idx[ip["breed"]]] = ip["confidence"]

        # Agreement
        if result["ensemble"]["all_agree"]:
            agreement_count += 1

        if (i + 1) % 500 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"      {i+1:5d}/{n}  {100*(i+1)/n:5.1f}%  {rate:5.1f} img/s  ETA {eta:5.0f}s")

    elapsed = time.time() - t0
    print(f"\n      Done: {total_evaluated}/{n} images in {elapsed:.0f}s ({total_evaluated/elapsed:.1f} img/s)")

    # Trim arrays if some images failed
    if total_evaluated < n:
        labels = labels[:total_evaluated]
        cal_probs = cal_probs[:total_evaluated]
        raw_probs = raw_probs[:total_evaluated]
        per_model_cal_probs = per_model_cal_probs[:total_evaluated]
        per_model_raw_probs = per_model_raw_probs[:total_evaluated]

    # Compute metrics
    print("\n[5/4] Computing metrics...")
    metrics = {
        "num_images": total_evaluated,
        "num_models": num_models,
        "temperatures": predictor.temperatures,
        "ensemble_calibrated": {
            "top1": round(_top_k_accuracy(cal_probs, labels, 1) * 100, 2),
            "top3": round(_top_k_accuracy(cal_probs, labels, 3) * 100, 2),
            "top5": round(_top_k_accuracy(cal_probs, labels, 5) * 100, 2),
            "ece_15": round(_ece_from_probs(cal_probs, labels, 15), 4),
        },
        "ensemble_uncalibrated": {
            "top1": round(_top_k_accuracy(raw_probs, labels, 1) * 100, 2),
            "top3": round(_top_k_accuracy(raw_probs, labels, 3) * 100, 2),
            "top5": round(_top_k_accuracy(raw_probs, labels, 5) * 100, 2),
            "ece_15": round(_ece_from_probs(raw_probs, labels, 15), 4),
        },
        "model_agreement_pct": round(100.0 * agreement_count / total_evaluated, 2) if total_evaluated else 0,
    }

    for m_idx in range(num_models):
        m_cal = per_model_cal_probs[:, m_idx, :]
        m_raw = per_model_raw_probs[:, m_idx, :]
        metrics[f"model_{m_idx}_calibrated"] = {
            "top1": round(_top_k_accuracy(m_cal, labels, 1) * 100, 2),
            "top3": round(_top_k_accuracy(m_cal, labels, 3) * 100, 2),
            "top5": round(_top_k_accuracy(m_cal, labels, 5) * 100, 2),
            "ece_15": round(_ece_from_probs(m_cal, labels, 15), 4),
        }
        metrics[f"model_{m_idx}_uncalibrated"] = {
            "top1": round(_top_k_accuracy(m_raw, labels, 1) * 100, 2),
            "top3": round(_top_k_accuracy(m_raw, labels, 3) * 100, 2),
            "top5": round(_top_k_accuracy(m_raw, labels, 5) * 100, 2),
            "ece_15": round(_ece_from_probs(m_raw, labels, 15), 4),
        }

    # Per-breed accuracy
    per_breed = []
    for i, breed in enumerate(breeds):
        total = confusion_cal[i].sum()
        correct = confusion_cal[i, i]
        per_breed.append({
            "breed": breed,
            "total": int(total),
            "correct": int(correct),
            "accuracy": round(float(correct / total) if total > 0 else 0.0, 4),
        })
    per_breed_sorted = sorted(per_breed, key=lambda x: x["accuracy"])

    # Confidence histogram
    confidences = np.max(cal_probs, axis=1)
    edges, counts = _hist_counts(confidences, n_bins=20)

    # Save everything
    np.save(os.path.join(args.output_dir, "confusion_matrix.npy"), confusion_cal)
    np.save(os.path.join(args.output_dir, "confusion_matrix_raw.npy"), confusion_raw)
    np.savez_compressed(
        os.path.join(args.output_dir, "predictions.npz"),
        labels=labels,
        cal_probs=cal_probs,
        raw_probs=raw_probs,
        per_model_cal_probs=per_model_cal_probs,
        per_model_raw_probs=per_model_raw_probs,
    )

    with open(os.path.join(args.output_dir, "per_breed_accuracy.csv"), "w") as f:
        f.write("breed,total,correct,accuracy\n")
        for pb in per_breed_sorted:
            f.write(f"{pb['breed']},{pb['total']},{pb['correct']},{pb['accuracy']:.6f}\n")

    with open(os.path.join(args.output_dir, "confidence_histogram.csv"), "w") as f:
        f.write("bin_start,bin_end,count\n")
        for i in range(len(counts)):
            f.write(f"{edges[i]:.4f},{edges[i+1]:.4f},{counts[i]}\n")

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Images evaluated:     {total_evaluated}")
    print(f"  Ensemble calibrated:  {metrics['ensemble_calibrated']['top1']:.2f}% top-1")
    print(f"                        {metrics['ensemble_calibrated']['top3']:.2f}% top-3")
    print(f"                        {metrics['ensemble_calibrated']['top5']:.2f}% top-5")
    print(f"                        ECE-15 = {metrics['ensemble_calibrated']['ece_15']:.4f}")
    print(f"  Ensemble uncalibrated: {metrics['ensemble_uncalibrated']['top1']:.2f}% top-1")
    print(f"                         ECE-15 = {metrics['ensemble_uncalibrated']['ece_15']:.4f}")
    print(f"  Model agreement:      {metrics['model_agreement_pct']:.2f}%")
    print("\n  Files written:")
    for fn in ["confusion_matrix.npy", "confusion_matrix_raw.npy", "predictions.npz",
               "per_breed_accuracy.csv", "confidence_histogram.csv", "metrics.json"]:
        print(f"    {args.output_dir}/{fn}")


if __name__ == "__main__":
    main()
