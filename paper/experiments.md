# Experiments Execution Plan

> **Target platforms:** Kaggle (recommended, GPU/TPU) or Hugging Face Spaces (8 vCPU, 32 GB RAM CPU-only).  
> **Not recommended:** full inference on Apple Silicon Macs — use `--max-images 100` for smoke tests only.
>
> This branch currently implements experiments 1, 3, and 7. Experiments 5 and 6 (ablation study and multi-dog composite evaluation) are listed as future work and are **not yet implemented**.

---

## Experiment 1: Validation Suite — Confusion Matrix, Per-Breed Accuracy & Agreement

**Goal:** Evaluate the ensemble on the full Stanford Dogs validation split, then generate the confusion matrix, per-breed accuracy ranking, confidence histogram, and model-agreement metrics.

**Data needed:**
- Trained models in `models/` (`.keras` or `.h5`)
- `data/unique_breeds.json`
- `data/Images/` (Stanford Dogs images, organised by breed folder)

**Steps:**
1. Load all `.keras`/`.h5` models via `app.predictor.DogBreedPredictor`
2. Iterate over every image under `data/Images/`
3. Record true label, calibrated ensemble probabilities, raw probabilities, and per-model probabilities
4. Compute top-1/3/5 accuracy, ECE-15, and model agreement
5. Save raw arrays and CSV summaries

**Output files (under `paper/results/` by default):**
- `confusion_matrix.npy` — 120×120 calibrated ensemble confusion matrix
- `confusion_matrix_raw.npy` — 120×120 uncalibrated confusion matrix
- `per_breed_accuracy.csv` — per-breed correct/total/accuracy
- `confidence_histogram.csv` — confidence bin counts for Figure 9
- `predictions.npz` — labels, `cal_probs`, `raw_probs`, `per_model_cal_probs`, `per_model_raw_probs`
- `metrics.json` — top-1/3/5, ECE-15, agreement percentage

**Script:** `paper/scripts/exp1_validation_suite.py`

**Quick run:**
```bash
# Smoke test on 100 images (Mac-safe)
python paper/scripts/exp1_validation_suite.py \
  --models-dir models \
  --images-dir data/Images \
  --labels-path data/unique_breeds.json \
  --output-dir paper/results \
  --max-images 100

# Full validation run (Kaggle / HF Spaces)
python paper/scripts/exp1_validation_suite.py \
  --models-dir models \
  --images-dir data/Images \
  --labels-path data/unique_breeds.json \
  --output-dir paper/results
```

---

## Experiment 3: Temperature Calibration — Reliability Diagrams

**Goal:** Generate before/after reliability diagrams and calibration metrics (ECE, MCE, NLL) from the validation probabilities produced in Experiment 1.

**Data needed:**
- `paper/results/predictions.npz` (created by `exp1_validation_suite.py`)

**Steps:**
1. Load raw and calibrated probabilities for the ensemble and each individual model
2. Bin confidences into 15 equal-width bins
3. Compute ECE-15, MCE-15, and NLL
4. Plot reliability diagram and confidence distribution

**Output files:**
- `paper/results/calibration_metrics.csv`
- `paper/figures/fig4_reliability_diagram.png`
- `paper/figures/fig9_confidence_distribution.png`

**Script:** `paper/scripts/exp3_calibration.py`

**Quick run:**
```bash
python paper/scripts/exp3_calibration.py \
  --predictions paper/results/predictions.npz \
  --output-dir paper/results \
  --figures-dir paper/figures
```

---

## Experiment 7: Training Curves

**Goal:** Plot training/validation loss and accuracy from a CSV training log.

**Data needed:**
- A CSV log with columns: `epoch`, `loss`, `val_loss`, `accuracy`, `val_accuracy`

**Output file:**
- `paper/figures/fig10_training_curves.png`

**Script:** `paper/scripts/exp7_training_curves.py`

**Quick run:**
```bash
python paper/scripts/exp7_training_curves.py \
  --log training_log.csv \
  --output paper/figures/fig10_training_curves.png \
  --title "Training and Validation Curves"
```

---

## Future Work / Not Yet Implemented

### Experiment 5: Ablation Study

**Goal:** Quantify the contribution of temperature scaling, MixUp/CutMix, label smoothing, dropout, progressive resizing, and fine-tuning.

**Status:** Not implemented. Some configurations require retraining models without specific components, which is out of scope for the current branch.

**Planned output:** `paper/results/ablation_study.csv` and Table 7 in `paper/outline.md`.

### Experiment 6: Multi-Dog Composite Accuracy

**Goal:** Evaluate detection recall and classification accuracy on synthetic multi-dog frames.

**Status:** Not implemented. Requires constructing a controlled composite test set with ground-truth labels.

**Planned output:** `paper/results/multi_dog_accuracy.csv`.

---

## Figure Generation Scripts

Several figures do not require model inference and can be regenerated at any time from known values or from Experiment 1 outputs.

| Figure | Script | Depends on |
|--------|--------|------------|
| fig2_confusion_matrix_top20.png | `paper/scripts/exp1_figures.py` | `paper/results/predictions.npz`, `paper/results/confusion_matrix.npy` |
| fig3_accuracy_vs_params.png | `paper/scripts/fig3_accuracy.py` | Hard-coded accuracy values |
| fig4_reliability_diagram.png | `paper/scripts/exp3_calibration.py` | `paper/results/predictions.npz` |
| fig5_fps_vs_dogs.png | `paper/scripts/generate_all_figures.py` | Hard-coded timing values |
| fig7_agreement_heatmap.png | `paper/scripts/exp1_figures.py` | `paper/results/predictions.npz` |
| fig8_per_breed_accuracy.png | `paper/scripts/exp1_figures.py` | `paper/results/per_breed_accuracy.csv` |
| fig9_confidence_distribution.png | `paper/scripts/exp3_calibration.py` | `paper/results/predictions.npz` |
| fig10_training_curves.png | `paper/scripts/exp7_training_curves.py` | Training CSV log |
| fig11_error_reduction.png | `paper/scripts/generate_all_figures.py` | Hard-coded accuracy values |
| fig12_latency_breakdown.png | `paper/scripts/generate_all_figures.py` | Hard-coded timing values |

---

## Files to Generate

```
paper/
├── outline.md              ✅ Comprehensive paper outline
├── references.bib          ✅ 55 curated references (14 categories)
├── experiments.md          ✅ This file — experiment execution plan
├── RUNBOOK.md              ✅ Step-by-step execution guide
├── scripts/
│   ├── exp1_validation_suite.py
│   ├── exp1_figures.py
│   ├── exp3_calibration.py
│   ├── exp7_training_curves.py
│   ├── fig3_accuracy.py
│   ├── generate_all_figures.py
│   ├── run_all_experiments.py        (created by another agent)
│   ├── run_on_huggingface.py         (created by another agent)
│   └── kaggle_paper_experiments.ipynb (created by another agent)
├── results/
│   ├── confusion_matrix.npy
│   ├── confusion_matrix_raw.npy
│   ├── per_breed_accuracy.csv
│   ├── confidence_histogram.csv
│   ├── predictions.npz
│   ├── metrics.json
│   ├── calibration_metrics.csv
│   ├── backbone_comparison.csv
│   └── temperature_calibration.csv
└── figures/
    ├── fig1_architecture.png
    ├── fig2_confusion_matrix_top20.png
    ├── fig3_accuracy_vs_params.png
    ├── fig3_accuracy_vs_params_dark.png
    ├── fig4_reliability_diagram.png
    ├── fig5_fps_vs_dogs.png
    ├── fig6_sample_detections.png
    ├── fig7_agreement_heatmap.png
    ├── fig8_per_breed_accuracy.png
    ├── fig9_confidence_distribution.png
    ├── fig10_training_curves.png
    ├── fig11_error_reduction.png
    └── fig12_latency_breakdown.png
```

---

## Quick Start: Run All Experiments

### Option A — One-command runner (recommended)

If `paper/scripts/run_all_experiments.py` (created by another agent) is present:

```bash
# On Kaggle or HF Spaces, from repo root:
python paper/scripts/run_all_experiments.py \
  --models-dir models \
  --images-dir data/Images \
  --labels-path data/unique_breeds.json \
  --output-dir paper/results \
  --figures-dir paper/figures
```

### Option B — Manual step-by-step

```bash
# 1. Validation suite (full run)
python paper/scripts/exp1_validation_suite.py \
  --models-dir models \
  --images-dir data/Images \
  --labels-path data/unique_breeds.json \
  --output-dir paper/results

# 2. Figures from validation results
python paper/scripts/exp1_figures.py \
  --results-dir paper/results \
  --figures-dir paper/figures \
  --labels-path data/unique_breeds.json

# 3. Calibration / reliability diagrams
python paper/scripts/exp3_calibration.py \
  --predictions paper/results/predictions.npz \
  --output-dir paper/results \
  --figures-dir paper/figures

# 4. Non-data-dependent figures
python paper/scripts/generate_all_figures.py

# 5. Training curves (if training_log.csv exists)
python paper/scripts/exp7_training_curves.py \
  --log training_log.csv \
  --output paper/figures/fig10_training_curves.png
```

For detailed platform-specific instructions (Kaggle, HF Spaces, local Mac), see `paper/RUNBOOK.md`.
