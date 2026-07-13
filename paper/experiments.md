# Experiments Execution Plan

> Run order: datasets and dependencies are listed per experiment.
> All experiments designed for execution on Hugging Face Spaces (8 vCPU, 32 GB RAM)
> or local Mac with TensorFlow 2.21.

---

## Experiment 1: Confusion Matrix & Per-Breed Accuracy

**Goal:** Generate a confusion matrix for the top-20 most confused breed pairs and a per-breed accuracy ranking.

**Data needed:** Validation set predictions (model output on 4,116 test images)

**Steps:**
1. Load the ensemble predictor
2. Iterate over all validation images (4,116 images)
3. Record: true breed, predicted breed, confidence for each model and the ensemble
4. Build 120×120 confusion matrix
5. Extract top-20 most confused pairs
6. Plot: seaborn heatmap (confusion matrix) + matplotlib sorted bar chart (per-breed accuracy)

**Output files:**
- `results/confusion_matrix.csv` (120×120)
- `results/per_breed_accuracy.csv`
- `figures/fig2_confusion_matrix.png`
- `figures/fig8_per_breed_accuracy.png`

**Script:** `paper/scripts/exp1_confusion_matrix.py`

---

## Experiment 2: Single-Model Inference Timing

**Goal:** Measure per-image inference time for each backbone independently.

**Data needed:** 100 representative test images (varying aspect ratios)

**Steps:**
1. Load each model separately
2. Warm-up: 10 inference passes (discard timing)
3. Measure: 100 inference passes, record wall-clock time per pass
4. Compute: mean, median, std, p95 latency
5. Repeat for batch sizes 1, 2, 3, 5

**Models to benchmark:**
- MobileNetV2 (2.4M params)
- EfficientNetV2S (21.0M params) ← live mode
- ConvNeXtTiny (28.3M params)
- Ensemble (both models, averaged)

**Output:**
- `results/inference_timing.csv`
- `figures/fig3_accuracy_vs_params.png`

**Script:** `paper/scripts/exp2_inference_timing.py`

---

## Experiment 3: Temperature Calibration (Reliability Diagrams)

**Goal:** Generate before/after reliability diagrams showing calibration improvement.

**Data needed:** Validation set logits (pre-softmax outputs) for both models

**Steps:**
1. Collect raw logits for all 4,116 validation images from each model
2. Compute softmax probabilities (before + after temperature scaling)
3. Compute confidence (max probability) and accuracy (is prediction correct?)
4. Bin confidences into 15 equal-width bins [0,1]
5. Per bin: compute mean confidence, mean accuracy, count
6. Compute ECE = Σ (|B_b|/N) · |acc(B_b) − conf(B_b)|
7. Plot: reliability diagram (confidence vs accuracy) with identity line

**Metrics:**
- ECE (Expected Calibration Error) for 15 bins
- MCE (Maximum Calibration Error)
- NLL (Negative Log-Likelihood)

**Output:**
- `results/calibration_metrics.csv`
- `figures/fig4_reliability_diagrams.png`

**Script:** `paper/scripts/exp3_calibration.py`

---

## Experiment 4: Pipeline FPS vs Number of Dogs

**Goal:** Measure end-to-end pipeline latency as a function of dogs per frame.

**Data needed:** Composite images with 1, 2, 3, 4, 5 dogs

**Steps:**
1. Create composite images: place N dogs in one frame (N = 1..5)
2. Each composite tested 20 times (discard first 3 for warmup)
3. Record: total pipeline latency, detector-only latency, classifier-only latency
4. Measure for both: ensemble mode and single-model fast mode

**Metrics:**
- FPS per configuration
- FPS degradation per additional dog
- Detector contribution to total latency
- Classifier contribution to total latency

**Output:**
- `results/pipeline_fps.csv`
- `figures/fig5_fps_vs_dogs.png`
- `figures/fig12_latency_breakdown.png`

**Script:** `paper/scripts/exp4_pipeline_benchmark.py`

---

## Experiment 5: Ablation Study

**Goal:** Quantify the contribution of each pipeline component.

**Data needed:** Validation set predictions under ablated configurations

**Configurations to test:**

| # | Configuration | Models Used | MixUp/CutMix | Label Smoothing | Temp. Scaling |
|---|--------------|-------------|:---:|:---:|:---:|
| A | Full pipeline (baseline) | EffV2S + ConvNeXt | ✓ | ✓ | ✓ |
| B | No temperature scaling | EffV2S + ConvNeXt | ✓ | ✓ | ✗ |
| C | No MixUp/CutMix | EffV2S + ConvNeXt | ✗ | ✓ | ✓ |
| D | No label smoothing | EffV2S + ConvNeXt | ✓ | ✗ | ✓ |
| E | Single model (fast) | EffV2S only | ✓ | ✓ | ✓ |
| F | MobileNetV2 baseline | MobileNetV2 | ✗ | ✗ | ✗ |

*Note: Configurations C and D require RETRAINING models without those components. If retraining is not feasible, report these as "expected" based on literature rather than measured.*

**Output:**
- `results/ablation_study.csv`
- Table 7 (in main outline)

**Script:** `paper/scripts/exp5_ablation.py`

---

## Experiment 6: Multi-Dog Detection Accuracy

**Goal:** Evaluate detection recall and classification accuracy on multi-dog frames.

**Data needed:** Composite images with known ground-truth breed + location

**Steps:**
1. Build a test set of 20 composite images (2–3 dogs each) with ground-truth labels
2. Run full pipeline on each image
3. Score: detection recall (dogs found / total dogs), classification accuracy (correctly classified / found)

**Metrics:**
- Detection recall @ confidence 0.4
- Classification accuracy on detected dogs
- Average precision (AP) for dog detection

**Output:**
- `results/multi_dog_accuracy.csv`

**Script:** `paper/scripts/exp6_multi_dog.py`

---

## Experiment 7: Training Curves

**Goal:** Plot accuracy and loss curves from training logs.

**Data needed:** Training CSV logs from `train.py` or `kaggle_train.py`

**Steps:**
1. Parse training log CSVs (or scrape from Kaggle training output)
2. Plot: training loss, validation loss, training accuracy, validation accuracy over epochs
3. Mark phase transitions (A→B, B→C)
4. Annotate key events (early stopping, learning rate reductions)

**Output:**
- `figures/fig10_training_curves.png`

**Script:** `paper/scripts/exp7_training_curves.py`

---

## Experiment 8: Error Analysis (Top Confusions)

**Goal:** Identify and visualise the most commonly confused breed pairs.

**Steps:**
1. From confusion matrix (Exp 1), extract top-10 confused pairs
2. For each pair: find 3 example images where misclassification occurred
3. Analyse common visual features causing confusion
4. Plot: grid of misclassified examples with true/predicted labels

**Output:**
- `results/top_confusions.csv`
- `figures/fig_error_examples.png`

---

## Files to Generate

```
paper/
├── outline.md              ✅ Comprehensive paper outline
├── references.bib           ✅ 55 curated references (14 categories)
├── experiments.md           ✅ This file — experiment execution plan
├── scripts/
│   ├── exp1_confusion_matrix.py
│   ├── exp2_inference_timing.py
│   ├── exp3_calibration.py
│   ├── exp4_pipeline_benchmark.py
│   ├── exp5_ablation.py
│   ├── exp6_multi_dog.py
│   └── exp7_training_curves.py
├── results/
│   ├── confusion_matrix.csv
│   ├── per_breed_accuracy.csv
│   ├── inference_timing.csv
│   ├── calibration_metrics.csv
│   ├── pipeline_fps.csv
│   ├── ablation_study.csv
│   ├── multi_dog_accuracy.csv
│   └── top_confusions.csv
└── figures/
    ├── fig1_architecture.png
    ├── fig2_confusion_matrix.png
    ├── fig3_accuracy_vs_params.png
    ├── fig4_reliability_diagrams.png
    ├── fig5_fps_vs_dogs.png
    ├── fig6_sample_detections.png
    ├── fig7_agreement_heatmap.png
    ├── fig8_per_breed_accuracy.png
    ├── fig9_confidence_histogram.png
    ├── fig10_training_curves.png
    ├── fig11_error_reduction.png
    └── fig12_latency_breakdown.png
```

---

## Quick Start: Run All Experiments

```bash
# On HF Spaces or local machine with models loaded:
cd paper/scripts

# 1. Confusion matrix + per-breed accuracy
python exp1_confusion_matrix.py --models-dir ../../models --data-dir ../../data --output ../results

# 2. Inference timing
python exp2_inference_timing.py --models-dir ../../models --output ../results

# 3. Calibration (reliability diagrams)
python exp3_calibration.py --models-dir ../../models --data-dir ../../data --output ../results

# 4. Pipeline FPS benchmark
python exp4_pipeline_benchmark.py --output ../results

# 5. Ablation study
python exp5_ablation.py --models-dir ../../models --data-dir ../../data --output ../results

# 6. Multi-dog detection
python exp6_multi_dog.py --output ../results

# 7. Training curves
python exp7_training_curves.py --logs-dir ../../ --output ../figures

# Generate all plots
python plot_all.py --results-dir ../results --figures-dir ../figures
```
