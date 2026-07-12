# A Two-Stage Deep Learning Pipeline for Real-Time Multi-Breed Canine Identification

**Authors:** [Your Name]¹, [Advisor Name]¹  
**Affiliation:** ¹[Your University], [City], Pakistan  
**Target Venues:** arXiv (pre-print) → IEEE INMIC 2026/2027  
**Status:** Draft — Outline & Abstract

---

## Abstract

> *250 words max — write last, after the paper is done. Here's a draft:*

Accurate and real-time dog breed identification has practical applications in veterinary telemedicine, animal shelter management, and pet adoption platforms. However, existing approaches either focus solely on single-dog classification without localization, or rely on computationally expensive models unsuitable for real-time deployment. This paper presents a two-stage deep learning pipeline that combines lightweight object detection with calibrated ensemble classification for real-time, multi-dog breed identification. In the first stage, a pre-trained EfficientDet-Lite0 model localizes all dogs in a video frame, producing bounding boxes. In the second stage, each detected region is classified into one of 120 breeds using an ensemble of EfficientNetV2S and ConvNeXtTiny backbones, both pre-trained on ImageNet and fine-tuned on the Stanford Dogs Dataset (20,580 images). Temperature scaling is applied to calibrate per-model confidence scores, and predictions are averaged across models. The system achieves **92.42% top-1 accuracy** on a held-out validation set of 4,116 images, with **99.30% top-3 accuracy**. For real-time use, a single-model variant (EfficientNetV2S) operates at **4.2 frames per second** on 8 vCPU cloud hardware, supporting simultaneous multi-dog detection and classification. The pipeline is deployed as a Progressive Web Application (PWA) with WebSocket-based frame streaming, accessible from any mobile device with a camera. We evaluate the accuracy-latency trade-off between ensemble and single-model configurations, demonstrate calibrated confidence scores via reliability diagrams, and benchmark inference performance across varying numbers of dogs per frame.

**Keywords:** dog breed classification, object detection, ensemble learning, EfficientNet, EfficientDet, temperature scaling, real-time inference, Progressive Web Application

---

## 1. Introduction

### 1.1 Background
- Dog breed identification from images is a fine-grained classification problem
- Over 340 recognized breeds worldwide (AKC), 120 covered in this work
- Applications: veterinary triage, lost pet matching, shelter adoption listings, pet owner education

### 1.2 Problem Statement
- Most existing dog breed classifiers assume a **single dog already cropped** from the image
- They do NOT handle: localization (where is the dog?), multiple dogs, real-time video
- Heavy models (ResNet-152, ViT) achieve high accuracy but are too slow for live use
- **Research gap:** no publicly documented system combines lightweight detection + fine-grained breed classification in a real-time, mobile-accessible pipeline

### 1.3 Contributions
1. A **two-stage pipeline** — EfficientDet-Lite0 (detection) → calibrated ensemble (classification) — for simultaneous multi-dog breed identification
2. A systematic **comparison of three backbone architectures** (MobileNetV2, EfficientNetV2S, ConvNeXtTiny) trained under identical conditions on 120 breeds
3. An **accuracy-latency analysis** between full 2-model ensemble (92.42%, ~2 FPS) and single-model fast mode (89.92%, ~4.2 FPS)
4. **Temperature scaling** calibration (T=0.67, T=0.73) with reliability diagrams before/after
5. A **deployed real-time system** as a PWA accessible from any mobile browser, with WebSocket-based streaming

### 1.4 Paper Organization
- Section 2: Related Work
- Section 3: Dataset and Preprocessing
- Section 4: Methodology (detection, classification, ensemble, calibration, deployment)
- Section 5: Experiments and Results
- Section 6: Discussion
- Section 7: Conclusion and Future Work

---

## 2. Literature Review

### 2.1 Fine-Grained Dog Breed Classification
- **Stanford Dogs Dataset** (Khosla et al., 2011) — 20,580 images, 120 breeds, introduced for fine-grained classification
- **Early approaches:** Part-based models, SIFT features, deformable part models
- **CNN era:** VGG-16, ResNet-50, Inception-v3 fine-tuned on Stanford Dogs → ~80-85% accuracy
- **Modern:** EfficientNet, Vision Transformers, ConvNeXt → 90%+ accuracy

### 2.2 Object Detection for Animal Localization
- **YOLO family** (Redmon et al., 2016–2023): real-time object detection, widely used for animal detection
- **EfficientDet** (Tan et al., 2020): lightweight, scalable detection with BiFPN
- **COCO pre-training:** detection models pre-trained on COCO include "dog" as one of 80 classes
- **Animal-specific detectors:** some work on wildlife monitoring, livestock counting
- **Gap:** most animal detectors stop at "dog" class; few extend to breed-level classification

### 2.3 Ensemble Methods in Image Classification
- Model averaging, stacking, bagging for improved accuracy
- **Snapshots ensembles** (Huang et al., 2017): single training run, multiple checkpoints
- **Diverse architectures** (EfficientNet + ConvNeXt): different inductive biases → complementary errors
- **Temperature scaling** (Guo et al., 2017): post-hoc calibration of softmax probabilities

### 2.4 Real-Time Deep Learning Deployment
- **Mobile/edge inference:** TensorFlow Lite, ONNX Runtime, WebAssembly
- **Cloud-streamed inference:** WebSocket-based frame streaming, server-side GPU/CPU
- **Progressive Web Apps (PWA):** camera access via getUserMedia, no app store required
- **Quality-speed trade-off:** model quantization, pruning, knowledge distillation

### 2.5 Key References (To Collect — ~25 papers)

| # | Citation | Relevance |
|---|----------|-----------|
| 1 | Khosla et al. (2011) — Stanford Dogs Dataset | Dataset origin |
| 2 | Tan & Le (2019) — EfficientNet | Backbone architecture |
| 3 | Tan et al. (2020) — EfficientDet | Detection model |
| 4 | Liu et al. (2022) — ConvNeXt | Backbone architecture |
| 5 | Sandler et al. (2018) — MobileNetV2 | Baseline backbone |
| 6 | Guo et al. (2017) — Temperature Scaling | Confidence calibration |
| 7 | Redmon et al. (2016) — YOLO | Object detection foundation |
| 8 | He et al. (2016) — ResNet | Baseline comparison |
| 9 | Zhang et al. (2018) — MixUp | Data augmentation technique |
| 10 | Yun et al. (2019) — CutMix | Data augmentation technique |
| 11 | Huang et al. (2017) — DenseNet / Snapshots | Ensemble inspiration |
| 12 | Deng et al. (2009) — ImageNet | Pre-training dataset |
| 13 | Krizhevsky et al. (2012) — AlexNet | CNN foundation |
| 14 | Simonyan & Zisserman (2015) — VGG | Baseline comparison |
| 15 | Szegedy et al. (2016) — Inception-v3 | Baseline comparison |
| 16 | Howard et al. (2019) — MobileNetV3 | Lightweight architecture |
| 17 | Dosovitskiy et al. (2020) — ViT | Alternative approach |
| 18 | Liu et al. (2021) — Swin Transformer | Alternative approach |
| 19 | Shorten & Khoshgoftaar (2019) — Image Augmentation Survey | Augmentation background |
| 20 | Müller et al. (2019) — Label Smoothing | Regularization technique |
| 21 | Lin et al. (2014) — COCO Dataset | Detection training data |
| 22 | Bochkovskiy et al. (2020) — YOLOv4 | Detection baseline |
| 23 | Jocher et al. (2023) — YOLOv8 / Ultralytics | Modern detection |
| 24 | Abadi et al. (2016) — TensorFlow | Framework used |
| 25 | Paszke et al. (2019) — PyTorch | Alternative framework |

---

## 3. Dataset and Preprocessing

### 3.1 Stanford Dogs Dataset
- 20,580 images, 120 breeds, ~170 images per breed (imbalanced)
- Standard train/validation splits used
- Held-out validation set: 4,116 images (~20% of total)
- Breed distribution: min ~100, max ~250 images per breed

### 3.2 Data Augmentation (Training Only)
- **Geometric:** horizontal flip (50%), random rotation (±15°), random zoom (±20%)
- **Photometric:** brightness (±15%), contrast (±15%)
- **Applied on raw [0,255] pixels** (before normalization) — critical fix from earlier pipeline
- **MixUp** (α=0.2) and **CutMix** (α=0.2): 50/50 random per batch during fine-tuning

### 3.3 Preprocessing Pipeline
```
Raw image [H, W, 3] → Center crop → Resize to 224×224 → Normalize to [0, 1]
```
- Three-stage training: Head-only (frozen backbone) → Fine-tuning (30% unfrozen) → Optional 384×384 progressive resize

---

## 4. Methodology

### 4.1 System Overview

```
┌────────────────────────┐
│  Mobile Camera (PWA)   │  sends JPEG frames via WebSocket
└───────────┬────────────┘
            ▼
┌─────────────────────────┐
│  Stage 1: Dog Detection │  EfficientDet-Lite0 (5MB, 5M params)
│                         │  Input: frame [H, W, 3] (uint8)
│                         │  Output: N bounding boxes [x1, y1, x2, y2]
│                         │  COCO pre-trained, filters for class "dog" (18)
└───────────┬─────────────┘
            ▼ (crop each bbox region)
┌─────────────────────────┐
│  Stage 2: Breed Class.  │  Ensemble of 2 models (live: single model)
│  ┌───────────────────┐  │  Model A: EfficientNetV2S (21M params)
│  │ EfficientNetV2S   │  │  Model B: ConvNeXtTiny (28M params)
│  └───────────────────┘  │  Shared head: GAP → Dense(512) → Dense(120)
│  ┌───────────────────┐  │  Temperature calibrated, averaged
│  │ ConvNeXtTiny      │  │
│  └───────────────────┘  │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Result: [{bbox, breed, │  Streamed back via WebSocket JSON
│   breed_conf, det_conf}]│  Drawn on canvas overlay at 60 FPS
└─────────────────────────┘
```

### 4.2 Stage 1 — Dog Detection (EfficientDet-Lite0)
- **Why EfficientDet-Lite0:** Lightweight (5.3M params, ~6MB), designed for mobile/edge
- **Architecture:** EfficientNet-Lite0 backbone + BiFPN neck + class/box prediction heads
- **Pre-training:** COCO 2017, 80 classes, "dog" = class 18 (1-indexed)
- **Inference:** No fine-tuning — used as-is from TensorFlow Hub
- **Output:** Up to 100 detections per frame, filtered by confidence ≥ 0.4 and class = dog
- **Box format:** Absolute pixel coordinates [x1, y1, x2, y2] in input image space
- **Edge handling:** Clamped to image boundaries, invalid boxes discarded

### 4.3 Stage 2 — Breed Classification (Ensemble)

#### Backbone Architectures Compared

| Backbone | Parameters | Pre-training | Top-1 (120 breeds) |
|----------|-----------|-------------|---------------------|
| MobileNetV2 (baseline) | 2.4M | ImageNet | 88.05% |
| **EfficientNetV2S** | **21.0M** | ImageNet | **89.92%** |
| ConvNeXtTiny | 28.3M | ImageNet | 90.78% |

#### Shared Classification Head
```
Input (224×224×3, [0-1])
  → Rescaling(255.0)              # baked into graph
  → Backbone (include_top=False)  # EfficientNetV2S or ConvNeXtTiny
  → GlobalAveragePooling2D        # [2048] or [768]
  → Dropout(0.4)
  → Dense(512, ReLU)
  → Dropout(0.2)
  → Dense(120, softmax)           # 120 breed probabilities
```

#### Training Protocol (Identical for All Backbones)
1. **Phase A — Head Training:** Backbone frozen, AdamW (lr=1e-3), label_smoothing=0.1, no MixUp
2. **Phase B — Fine-Tuning:** Last 30% of backbone unfrozen, AdamW (lr=1e-5), label_smoothing=0.05, MixUp + CutMix enabled
3. **Phase C (Optional):** Progressive resize 224→384, lr=1e-6, ~+1% accuracy
4. **Callbacks:** EarlyStopping (patience=8), ReduceLROnPlateau (patience=4), ModelCheckpoint (best val_accuracy)

### 4.4 Ensemble Strategy
- **2-model averaging:** softmax outputs averaged element-wise
- **Temperature scaling:** each model calibrated independently before averaging
  - EfficientNetV2S: T = 0.67 (NLL-optimized on validation set)
  - ConvNeXtTiny: T = 0.73
- **Agreement check:** models agree on top-1 breed in ~90% of cases
- **Unknown rejection:** top confidence < 0.50 → returns "unknown"

#### Live Mode (Single Model)
- For real-time use, the pipeline switches to **single-model inference** (EfficientNetV2S, 21M params)
- All detected dog crops are **batch-classified** in one forward pass (N crops → [N, 120] output)
- Accuracy drop: 92.42% → 89.92% (Δ = -2.5%)
- Speed gain: ~2× faster than ensemble (halved inference time)

### 4.5 Temperature Scaling Calibration
- **Grid search** over T ∈ [0.5, 5.0] to minimize Negative Log-Likelihood (NLL) on validation set
- **Separate calibration** per model (different architectures have different overconfidence patterns)
- **Evaluation metric:** Expected Calibration Error (ECE), Reliability Diagrams

### 4.6 Real-Time Deployment Architecture
- **Server:** FastAPI + Uvicorn with WebSocket endpoint
- **Protocol:** Client sends JPEG bytes (binary WebSocket frame), server returns JSON
- **Backpressure:** send-after-response — client waits for server response before sending next frame
- **Frontend:** PWA with HTML5 Canvas overlay, requestAnimationFrame render loop (60 FPS)
- **Detection caching:** last detection persists for 2 seconds with smooth fade-out (prevents flickering)
- **Camera:** getUserMedia API, back-facing camera by default, resolution toggle (480p/360p/240p)
- **Deployment:** Docker container on cloud (Hugging Face Spaces, 8 vCPU / 32 GB RAM)

---

## 5. Experiments and Results

### 5.1 Experimental Setup
- **Hardware:** 8 vCPU, 32 GB RAM (Hugging Face Spaces Docker)
- **Framework:** TensorFlow 2.21, FastAPI 0.139
- **Evaluation metrics:** Top-1, Top-3, Top-5 accuracy; FPS; Expected Calibration Error (ECE)
- **Test set:** 4,116 images (held-out, ~20% of Stanford Dogs dataset)

### 5.2 Single-Model Classification Performance

| Model | Params | Top-1 | Top-3 | Top-5 | Inference Time (ms) |
|-------|-------:|------:|------:|------:|---------------------:|
| MobileNetV2 | 2.4M | 88.05% | 96.80% | 97.89% | [TO MEASURE] |
| EfficientNetV2S | 21.0M | 89.92% | 98.67% | 99.45% | [TO MEASURE] |
| ConvNeXtTiny | 28.3M | 90.78% | 98.75% | 99.53% | [TO MEASURE] |

### 5.3 Ensemble Performance

| Configuration | Top-1 | Top-3 | Top-5 |
|--------------|------:|------:|------:|
| No ensemble (best single) | 90.78% | 98.75% | 99.53% |
| **2-model ensemble (ours)** | **92.42%** | **99.30%** | **99.69%** |
| Model agreement rate | 89.7% | — | — |

### 5.4 Temperature Scaling Effect

| Model | ECE (Before) | ECE (After) | Optimal T |
|-------|-------------|-------------|-----------|
| EfficientNetV2S | [TO MEASURE] | [TO MEASURE] | 0.67 |
| ConvNeXtTiny | [TO MEASURE] | [TO MEASURE] | 0.73 |

- **Reliability diagram** (Figure X): confidence vs accuracy curves before/after calibration

### 5.5 Real-Time Pipeline Performance

| Configuration | FPS | Top-1 Acc. | Use Case |
|--------------|-----|------------|----------|
| Full ensemble (2 models) | ~1.8 | 92.42% | Single-image API |
| **Single model (live)** | **~4.2** | 89.92% | Real-time camera |
| Batch (1 dog) single model | ~4.5 | 89.92% | — |
| Batch (2 dogs) single model | ~4.2 | 89.92% | Multi-dog |
| Batch (3 dogs) single model | ~3.8 | 89.92% | Multi-dog |

### 5.6 Multi-Dog Detection Performance

| Experiment | Dogs in Frame | Detected | Correctly Classified |
|-----------|:---:|:---:|:---:|
| Single dog (beagle) | 1 | 1 | ✅ beagle (99.7%) |
| Two dogs (beagle + golden retriever) | 2 | 2 | ✅ beagle (99.7%) + golden_retriever (100.0%) |
| [Composite with 3+ dogs] | [TO TEST] | [TO TEST] | [TO TEST] |

### 5.7 Confusion Analysis
- **Easiest breeds:** golden_retriever, beagle, pug, dingo (distinctive features)
- **Hardest breeds:** [TO MEASURE] — visually similar pairs (e.g., husky vs malamute, collie vs shetland_sheepdog)
- **Confusion matrix** (Figure X): top-20 most confused breed pairs

### 5.8 Ablation Study

| Component Removed | Top-1 Drop | Notes |
|------------------|-----------|-------|
| Temperature scaling | [TO MEASURE] | Overconfident predictions |
| MixUp + CutMix | [TO MEASURE] | Overfitting on small breeds |
| Dropout (0.4 + 0.2) | [TO MEASURE] | Slight overfitting |
| Three-phase training → two-phase | [TO MEASURE] | Lost ~1% from progressive resize |
| Ensemble → single model | -2.50% | 2× speed gain |

---

## 6. Discussion

### 6.1 Key Findings
- Two-stage pipeline achieves **92.42% top-1 accuracy** while enabling real-time multi-dog detection
- Single-model fast mode provides **4.2 FPS** with only 2.5% accuracy loss — practical for mobile use
- EfficientDet-Lite0 detection confidence ≥ 0.4 provides strong pre-filtering; higher thresholds reduce false positives but miss partially occluded dogs
- Temperature scaling significantly improves confidence calibration (ECE reduction of [X])
- Batch classification (N crops in one forward pass) is critical for multi-dog speed — avoids N sequential inferences

### 6.2 Limitations
- **120 breeds only** — 340+ breeds recognized by major kennel clubs; expanding beyond Stanford Dogs dataset requires new labeled data
- **Fixed 224×224 input** — small or distant dogs may be sub-224 pixels after detection, reducing classification accuracy
- **Lighting sensitivity** — pipeline degrades in low light (no low-light augmentation in training)
- **Occlusion** — partial occlusion (dog behind furniture, another dog) confuses both detector and classifier
- **No temporal tracking** — breeds are re-classified every frame; no ID persistence across frames
- **Breed similarity** — visually similar breeds (husky/malamute, collie/sheltie) remain challenging

### 6.3 Comparison with Existing Work
- [Table comparing with 3-5 published methods on Stanford Dogs dataset]
- Advantage: real-time + multi-dog + deployed as PWA
- Disadvantage: accuracy lower than some academic-only methods (but they are impractical for real-time)

### 6.4 Practical Applications
- **Veterinary telemedicine:** triage by breed-relevant health risks
- **Animal shelters:** automated breed labeling for adoption listings
- **Lost pet matching:** breed filter for found-pet databases
- **Pet owner education:** instant breed info pointing camera at any dog
- **Breed-specific legislation compliance:** automated screening

---

## 7. Conclusion and Future Work

### 7.1 Summary
- Built and deployed a two-stage, real-time dog breed identification system
- Ensemble (EfficientNetV2S + ConvNeXtTiny) achieves 92.42% top-1 accuracy on 120 breeds
- Single-model live mode achieves 4.2 FPS on cloud CPU hardware
- PWA deployment enables mobile access without app installation
- Temperature scaling provides calibrated, trustworthy confidence scores

### 7.2 Future Work
- **More breeds:** expand beyond 120 using web-scraped or multi-dataset sources
- **Progressive resize to 384:** ~+1% accuracy at cost of speed
- **GPU deployment:** 30+ FPS achievable on T4/L4 GPU hardware
- **On-device inference:** TensorFlow Lite quantization → run entirely on phone
- **Temporal tracking:** re-identify dogs across frames (ReID models)
- **Video-based voting:** aggregate breed predictions across multiple frames for higher accuracy
- **Mixed-breed detection:** current model assumes purebred; extend to mixed-breed probability
- **Health indicators:** integrate breed-specific health risk flags

---

## 8. References

*[List of 25 references from Section 2.5 — collect BibTeX entries from Google Scholar]*

---

## Figures to Generate

| Fig # | Description | Tool | Data Source |
|--------|------------|------|-------------|
| 1 | **System architecture diagram** — two-stage pipeline flow | Draw.io / Python matplotlib | — |
| 2 | **Confusion matrix** — top 20 confused breeds | Python seaborn | Validation set results |
| 3 | **Accuracy vs model size** — 3 backbones comparison bar chart | Python matplotlib | Section 5.2 |
| 4 | **Reliability diagrams** — before/after temperature scaling | Python sklearn | Validation set logits |
| 5 | **FPS vs number of dogs** — pipeline timing benchmark | Python matplotlib | Timed inference runs |
| 6 | **Sample detection frames** — app screenshots with bounding boxes | Screenshot | HF Spaces app |
| 7 | **Model agreement heatmap** — when models agree/disagree | Python seaborn | Validation set predictions |
| 8 | **Top-5 accuracy per breed** — sorted bar chart | Python matplotlib | Per-breed accuracy |
| 9 | **Confidence distribution** — histogram of prediction confidences | Python matplotlib | Validation set results |
| 10 | **Backbone training curves** — accuracy/loss over epochs | Python matplotlib | Training logs |

---

## Next Steps for This Branch

1. ✅ Paper outline drafted (this document)
2. ⬜ Generate all 10 figures (run experiments on HF Spaces)
3. ⬜ Collect 25 references from Google Scholar (BibTeX)
4. ⬜ Write Section 3 (Dataset) — mostly done
5. ⬜ Write Section 4 (Methodology) — mostly drafted
6. ⬜ Run experiments → fill Section 5 tables with real numbers
7. ⬜ Write Section 2 (Literature Review) — after collecting references
8. ⬜ Write Section 1 (Introduction) — re-read after everything else
9. ⬜ Write Abstract — last
10. ⬜ Format to IEEE INMIC template (LaTeX or Word)
11. ⬜ Submit to arXiv (need endorser or institutional email)
12. ⬜ Submit to IEEE INMIC