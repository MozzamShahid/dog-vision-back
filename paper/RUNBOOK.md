# Research Paper Experiments — Runbook

How to reproduce every figure and metric in the paper on a Mac (smoke-test only), Kaggle (recommended), or Hugging Face Spaces.

> **Hardware note:** The full Stanford Dogs validation set is 4,116 images. Running the full ensemble on a Mac CPU is slow and may run out of memory. Use Kaggle or HF Spaces for the real results, and only use a Mac for quick smoke tests.

---

## Prerequisites

1. Clone or download the repository:
   ```bash
   git clone https://github.com/<user>/dog-vision-back.git
   cd dog-vision-back
   git checkout research-paper
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # Or, if you use uv:
   uv pip install -r requirements.txt
   ```

3. Obtain the trained weights and the dataset:
   - **Models:** download the release assets and place all `.keras` (or `.h5`) files in `models/`
   - **Labels:** ensure `data/unique_breeds.json` exists
   - **Images:** ensure `data/Images/` contains the Stanford Dogs images, organised by breed folder (e.g. `data/Images/n02085620_Chihuahua/n02085620_10074.jpg`)

   Expected layout:
   ```
   dog-vision-back/
   ├── models/
   │   ├── efficientnetv2s.keras
   │   └── convnexttiny.keras
   ├── data/
   │   ├── unique_breeds.json
   │   └── Images/
   │       ├── n02085620_Chihuahua/
   │       ├── n02085782_Japanese_spaniel/
   │       └── ...
   ```

---

## 1. Run everything locally (Mac, small test only)

Use this only to verify that the scripts execute. It will **not** produce paper-quality numbers because the sample size is tiny.

```bash
# From the repo root
cd /path/to/dog-vision-back

# Smoke test: 100 random validation images
python paper/scripts/exp1_validation_suite.py \
  --models-dir models \
  --images-dir data/Images \
  --labels-path data/unique_breeds.json \
  --output-dir paper/results \
  --max-images 100

# Generate figures from the smoke-test results
python paper/scripts/exp1_figures.py \
  --results-dir paper/results \
  --figures-dir paper/figures \
  --labels-path data/unique_breeds.json

# Calibration diagrams
python paper/scripts/exp3_calibration.py \
  --predictions paper/results/predictions.npz \
  --output-dir paper/results \
  --figures-dir paper/figures

# Figures that need no model inference
python paper/scripts/generate_all_figures.py

# Training curves, if you have a training log
python paper/scripts/exp7_training_curves.py \
  --log training_log.csv \
  --output paper/figures/fig10_training_curves.png
```

**Expected time on Apple M1/M2:** ~5–20 minutes for 100 images, depending on model size.  
**Expected output:** `paper/results/metrics.json`, `paper/figures/fig2_confusion_matrix_top20.png`, etc.

---

## 2. Run on Kaggle (recommended)

Kaggle gives you a free NVIDIA T4 GPU, which makes the full validation run fast.

### 2.1 Create a new Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **New Notebook**
3. In the notebook settings (right-hand panel):
   - **Accelerator:** GPU T4 x2
   - **Internet:** On
   - **Environment:** Latest TensorFlow

### 2.2 Upload the repository

Option A — Git clone inside the notebook:

```python
!git clone https://github.com/<user>/dog-vision-back.git
%cd dog-vision-back
!git checkout research-paper
```

Option B — Upload `dog-vision-back.zip` via the **Add Data** button and unzip:

```python
!unzip -q /kaggle/input/dog-vision-back/dog-vision-back.zip -d /kaggle/working/
%cd /kaggle/working/dog-vision-back
```

### 2.3 Upload models and data

Add these as Kaggle Datasets (via **Add Data**):

- `dog-vision-models/` containing `*.keras` or `*.h5`
- `stanford-dogs/` containing `Images/` and `unique_breeds.json`

Then create symlinks so the paths match the scripts:

```python
import os
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Adjust the source paths to match your Kaggle dataset names
!ln -sf /kaggle/input/dog-vision-models/* models/
!ln -sf /kaggle/input/stanford-dogs/Images data/Images
!ln -sf /kaggle/input/stanford-dogs/unique_breeds.json data/unique_breeds.json
```

### 2.4 Install dependencies

```python
!pip install -q -r requirements.txt
```

### 2.5 Run the experiments

If `paper/scripts/kaggle_paper_experiments.ipynb` or `paper/scripts/run_all_experiments.py` (created by another agent) is available, run that single notebook/script. Otherwise run manually:

```python
# Full validation suite
!python paper/scripts/exp1_validation_suite.py \
  --models-dir models \
  --images-dir data/Images \
  --labels-path data/unique_breeds.json \
  --output-dir paper/results

# Figures from validation results
!python paper/scripts/exp1_figures.py \
  --results-dir paper/results \
  --figures-dir paper/figures \
  --labels-path data/unique_breeds.json

# Calibration diagrams
!python paper/scripts/exp3_calibration.py \
  --predictions paper/results/predictions.npz \
  --output-dir paper/results \
  --figures-dir paper/figures

# Non-data-dependent figures
!python paper/scripts/generate_all_figures.py
```

### 2.6 Download the outputs

```python
import shutil
shutil.make_archive('/kaggle/working/paper_outputs', 'zip', 'paper')
print("Download: /kaggle/working/paper_outputs.zip")
```

Click the download icon next to `paper_outputs.zip` in the Kaggle output panel.

**Expected time:** 5–15 minutes for the full 4,116-image validation suite on T4.  
**Expected artifacts:** `paper/results/predictions.npz`, `paper/figures/fig4_reliability_diagram.png`, etc.

---

## 3. Run on Hugging Face Spaces

HF Spaces is useful if you already deploy the app there, because the models and data are already present. The default Space uses CPU, so the full run is slower than Kaggle.

### 3.1 Create / open a Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create a **Gradio** or **Blank** Space
3. Link it to this GitHub repo, or upload files manually

### 3.2 Upload files

Upload via the Space file manager or `git`:

```bash
git clone https://huggingface.co/spaces/<your-username>/<your-space>
cd <your-space>

# Copy repo contents, models, and data
cp -r /path/to/dog-vision-back/* .
mkdir -p models data
cp /path/to/models/*.keras models/
cp /path/to/unique_breeds.json data/
cp -r /path/to/Images data/Images

git add .
git commit -m "Add paper experiment scripts, models, and data"
git push
```

### 3.3 Run with the provided HF helper

If `paper/scripts/run_on_huggingface.py` (created by another agent) exists:

```bash
python paper/scripts/run_on_huggingface.py \
  --models-dir models \
  --images-dir data/Images \
  --labels-path data/unique_breeds.json \
  --output-dir paper/results \
  --figures-dir paper/figures
```

Otherwise run the same manual steps as on Kaggle (see §2.5).

### 3.4 Retrieve outputs

In the Space file manager, navigate to `paper/results/` and `paper/figures/`, then download each file. Alternatively, from a local terminal:

```bash
huggingface-cli download <your-username>/<your-space> paper/results/metrics.json --local-dir ./downloads
huggingface-cli download <your-username>/<your-space> paper/figures/fig4_reliability_diagram.png --local-dir ./downloads
```

**Expected time:** 20–60 minutes for the full validation suite on 8 vCPU CPU.  
**Tip:** Run the smoke test first (`--max-images 100`) to verify paths before committing to the full run.

---

## 4. What each output file means

| File | Produced by | Meaning |
|------|-------------|---------|
| `paper/results/confusion_matrix.npy` | `exp1_validation_suite.py` | 120×120 calibrated ensemble confusion matrix. Row = true breed, column = predicted breed. |
| `paper/results/confusion_matrix_raw.npy` | `exp1_validation_suite.py` | Same as above, but before temperature scaling. |
| `paper/results/per_breed_accuracy.csv` | `exp1_validation_suite.py` | For each breed: total images, correct predictions, accuracy. |
| `paper/results/predictions.npz` | `exp1_validation_suite.py` | Compressed NumPy archive with `labels`, `cal_probs`, `raw_probs`, `per_model_cal_probs`, `per_model_raw_probs`. |
| `paper/results/confidence_histogram.csv` | `exp1_validation_suite.py` | Bin edges and counts for the calibrated confidence distribution. |
| `paper/results/metrics.json` | `exp1_validation_suite.py` | Top-1/3/5 accuracy, ECE-15, model agreement, per-model metrics. |
| `paper/results/calibration_metrics.csv` | `exp3_calibration.py` | ECE, MCE, NLL for ensemble and each model, raw vs calibrated. |
| `paper/results/backbone_comparison.csv` | `generate_all_figures.py` | Accuracy and inference-time table used for Table 2. |
| `paper/results/temperature_calibration.csv` | `generate_all_figures.py` | Per-model optimal temperature and interpretation. |
| `paper/figures/fig2_confusion_matrix_top20.png` | `exp1_figures.py` | Heatmap of the 20 most confused breeds. |
| `paper/figures/fig3_accuracy_vs_params.png` | `fig3_accuracy.py` | Accuracy vs parameter count for the three backbones + ensemble. |
| `paper/figures/fig4_reliability_diagram.png` | `exp3_calibration.py` | Reliability diagram before/after temperature scaling. |
| `paper/figures/fig5_fps_vs_dogs.png` | `generate_all_figures.py` | FPS as a function of dogs per frame. |
| `paper/figures/fig7_agreement_heatmap.png` | `exp1_figures.py` | Pairwise model agreement matrix. |
| `paper/figures/fig8_per_breed_accuracy.png` | `exp1_figures.py` | Sorted per-breed accuracy bar chart. |
| `paper/figures/fig9_confidence_distribution.png` | `exp3_calibration.py` | Histogram of calibrated confidence scores. |
| `paper/figures/fig10_training_curves.png` | `exp7_training_curves.py` | Training/validation loss and accuracy over epochs. |
| `paper/figures/fig11_error_reduction.png` | `generate_all_figures.py` | Ensemble error reduction vs single models. |
| `paper/figures/fig12_latency_breakdown.png` | `generate_all_figures.py` | Pipeline latency breakdown chart. |

---

## 5. Troubleshooting

### 5.1 Models not found

**Error:**
```
FileNotFoundError: No models found in models
```

**Fix:**
- Confirm model files are in `models/` and end with `.keras` or `.h5`
- Pass the correct directory:
  ```bash
  python paper/scripts/exp1_validation_suite.py --models-dir /path/to/models
  ```
- If you only have SavedModel directories, convert them to `.keras` first or point `app/predictor.py` at the SavedModel path.

### 5.2 Data not found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/unique_breeds.json'
```

or

```
0 images found
```

**Fix:**
- Verify `data/unique_breeds.json` exists
- Verify `data/Images/` contains breed subfolders with `.jpg`/`.jpeg`/`.png` images
- On Kaggle/HF, double-check that symlinks point to the correct uploaded dataset paths
- Run from the repository root, not from inside `paper/scripts/`

### 5.3 Uvicorn install issues

**Error:**
```
ModuleNotFoundError: No module named 'uvicorn'
```

or

```
ERROR: Could not find a version that satisfies the requirement uvicorn[standard]==0.49.0
```

**Fix:**
- The paper scripts do **not** require Uvicorn; it is only needed to run the FastAPI app. If you only want to generate figures, skip Uvicorn:
  ```bash
  pip install tensorflow==2.21.0 numpy==2.4.6 matplotlib seaborn pandas pillow
  ```
- If you need Uvicorn for the app, try installing without the pinned version:
  ```bash
  pip install uvicorn[standard]
  ```

### 5.4 Out of memory

**Error:**
```
tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor
```

or the process is killed on a Mac.

**Fix:**
- Use `--max-images N` for a smaller run
- Run on Kaggle/HF Spaces instead of Mac
- Load one model at a time instead of the full ensemble (requires editing `app/predictor.py`)
- Close other applications to free RAM
- On Kaggle, make sure the notebook accelerator is set to GPU, not CPU

### 5.5 `predictions.npz` not found

**Error:**
```
ERROR: predictions file not found: paper/results/predictions.npz
Run exp1_validation_suite.py first.
```

**Fix:**
Run `exp1_validation_suite.py` before `exp3_calibration.py` or `exp1_figures.py`.

### 5.6 Training log missing columns

**Error:**
```
ERROR: log missing columns: {'val_accuracy'}
```

**Fix:**
Ensure your training CSV has exactly the columns: `epoch`, `loss`, `val_loss`, `accuracy`, `val_accuracy`. Rename or add missing columns before running `exp7_training_curves.py`.

### 5.7 Figures look blank or have wrong colours

**Fix:**
- All plotting scripts use `matplotlib.use('Agg')`, so they are designed to run headlessly. Do not run them from an interactive notebook cell expecting inline plots unless you remove that line.
- If colours look inverted, check whether you opened the `_dark.png` variant of Figure 3 on a light background.

---

## Quick reference command chain

```bash
# 1. Full validation suite
python paper/scripts/exp1_validation_suite.py \
  --models-dir models --images-dir data/Images \
  --labels-path data/unique_breeds.json --output-dir paper/results

# 2. Data-dependent figures
python paper/scripts/exp1_figures.py \
  --results-dir paper/results --figures-dir paper/figures \
  --labels-path data/unique_breeds.json

# 3. Calibration diagrams
python paper/scripts/exp3_calibration.py \
  --predictions paper/results/predictions.npz \
  --output-dir paper/results --figures-dir paper/figures

# 4. Static figures
python paper/scripts/generate_all_figures.py

# 5. Training curves (optional)
python paper/scripts/exp7_training_curves.py \
  --log training_log.csv --output paper/figures/fig10_training_curves.png
```
