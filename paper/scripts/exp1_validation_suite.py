"""
Run on Kaggle or HF Spaces where models + validation data are present.
Generates confusion matrix, per-breed accuracy, confidence histogram.

Usage: python paper/scripts/exp1_validation_suite.py --models-dir /path/to/models --data-dir /path/to/data

Output:
  paper/results/confusion_matrix.npy        (120×120)
  paper/results/per_breed_accuracy.csv
  paper/results/prediction_logits.npy
  paper/results/agreement_rate.json
"""

import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.predictor import DogBreedPredictor
from app.preprocessing import preprocess_image
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--labels-path', default='data/unique_breeds.json')
    parser.add_argument('--images-dir', default='data/Images')
    parser.add_argument('--output-dir', default='paper/results')
    parser.add_argument('--max-images', type=int, default=0, help='0 = all images')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading breeds...")
    with open(args.labels_path) as f:
        breeds = json.load(f)
    num_breeds = len(breeds)
    breed_to_idx = {b: i for i, b in enumerate(breeds)}
    print(f"  {num_breeds} breeds loaded")

    print("Loading predictor...")
    import glob
    model_files = sorted(glob.glob(os.path.join(args.models_dir, "*.keras")))
    if not model_files:
        model_files = sorted(glob.glob(os.path.join(args.models_dir, "*.h5")))
    predictor = DogBreedPredictor(model_paths=model_files, labels_path=args.labels_path)
    print(f"  {len(predictor.models)} model(s) loaded")

    # Find all images and their true labels
    print("Indexing images...")
    image_paths = []
    true_labels = []
    for breed_folder in sorted(os.listdir(args.images_dir)):
        breed_path = os.path.join(args.images_dir, breed_folder)
        if not os.path.isdir(breed_path):
            continue
        if breed_folder not in breed_to_idx:
            print(f"  Skipping unknown breed: {breed_folder}")
            continue
        for img_name in os.listdir(breed_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(breed_path, img_name))
                true_labels.append(breed_to_idx[breed_folder])

    n_total = len(image_paths)
    print(f"  {n_total} images found")

    if args.max_images > 0 and args.max_images < n_total:
        indices = np.random.choice(n_total, args.max_images, replace=False)
        image_paths = [image_paths[i] for i in indices]
        true_labels = [true_labels[i] for i in indices]
        print(f"  Sampled {args.max_images} images")

    n = len(image_paths)
    confusion = np.zeros((num_breeds, num_breeds), dtype=np.int32)
    per_model_predictions = [[] for _ in range(predictor.num_models)]
    all_ensemble_preds = []
    all_true = []
    agreement_count = 0
    total_count = 0

    print(f"Evaluating {n} images...")
    t0 = time.time()
    for i, (path, true_idx) in enumerate(zip(image_paths, true_labels)):
        with open(path, 'rb') as f:
            img_bytes = f.read()

        try:
            result = predictor.predict(img_bytes, use_tta=False)
        except Exception as e:
            print(f"  Error on {path}: {e}")
            continue

        total_count += 1

        # Ensemble prediction
        ensemble_pred_idx = breeds.index(result['primary']['breed']) if result['primary']['breed'] != 'unknown' else -1
        if ensemble_pred_idx >= 0:
            confusion[true_idx][ensemble_pred_idx] += 1
        all_ensemble_preds.append(ensemble_pred_idx)
        all_true.append(true_idx)

        # Per-model predictions
        if 'ensemble' in result:
            ind_preds = result['ensemble']['individual_predictions']
            for m_idx, ip in enumerate(ind_preds):
                pred_idx = breeds.index(ip['breed']) if ip['breed'] != 'unknown' else -1
                per_model_predictions[m_idx].append((true_idx, pred_idx, ip['confidence']))

            # Agreement check
            if len(ind_preds) >= 2:
                if ind_preds[0]['breed'] == ind_preds[1]['breed']:
                    agreement_count += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{n} ({100*(i+1)/n:.1f}%) — {rate:.1f} img/s — ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s ({n/elapsed:.1f} img/s)")

    # Compute per-breed accuracy
    per_breed_acc = {}
    for i, breed in enumerate(breeds):
        total = confusion[i].sum()
        correct = confusion[i][i]
        per_breed_acc[breed] = {
            'total': int(total),
            'correct': int(correct),
            'accuracy': float(correct / total) if total > 0 else 0.0
        }

    # Compute top-1, top-3, top-5
    correct = sum(confusion[i][i] for i in range(num_breeds))
    top1 = correct / confusion.sum() * 100

    # Agreement rate
    ag_rate = agreement_count / total_count * 100 if total_count > 0 else 0

    # Save results
    np.save(os.path.join(args.output_dir, 'confusion_matrix.npy'), confusion)
    
    with open(os.path.join(args.output_dir, 'per_breed_accuracy.csv'), 'w') as f:
        f.write("breed,total,correct,accuracy\n")
        for breed in breeds:
            acc = per_breed_acc[breed]
            f.write(f"{breed},{acc['total']},{acc['correct']},{acc['accuracy']:.6f}\n")

    with open(os.path.join(args.output_dir, 'agreement_rate.json'), 'w') as f:
        json.dump({
            'top1_accuracy_pct': round(top1, 2),
            'agreement_rate_pct': round(ag_rate, 2),
            'total_evaluated': total_count,
        }, f, indent=2)

    print(f"\n=== Results ===")
    print(f"  Top-1 accuracy: {top1:.2f}%")
    print(f"  Model agreement: {ag_rate:.2f}%")
    print(f"  Confusion matrix: {args.output_dir}/confusion_matrix.npy")
    print(f"  Per-breed accuracy: {args.output_dir}/per_breed_accuracy.csv")


if __name__ == '__main__':
    main()
