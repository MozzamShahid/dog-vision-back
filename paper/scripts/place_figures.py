"""Move the figure gallery in outline.md to per-section placements.

A trailing gallery makes every LaTeX float pile up at the end of the document.
This relocates each figure to the end of the section that discusses it, adds
Figure 6 (in-app detections), and closes the numbering gap left by the dropped
training-curve figure (old 11->10, old 12->11).

Run: python paper/scripts/place_figures.py [--write]
"""

import re
import sys
from pathlib import Path

SRC = Path("paper/outline.md")

FIGS = {
    1: "![Figure 1: The two-stage pipeline. A camera frame is streamed over WebSocket, EfficientDet-Lite0 localises every dog, each crop is classified by an ensemble of EfficientNetV2S and ConvNeXtTiny, and the temperature-scaled softmax outputs are averaged before rendering.](figures/fig1_architecture.png)",
    2: "![Figure 2: Confusion matrix restricted to the 20 most-confused breeds. Off-diagonal mass concentrates on visually near-identical pairs.](figures/fig2_confusion_matrix_top20.png)",
    3: "![Figure 3: Classification accuracy and parameter efficiency across the three backbones and the ensemble. Left: top-1/3/5 accuracy per model. Right: top-1 accuracy versus parameter count.](figures/fig3_accuracy_vs_params.png)",
    4: "![Figure 4: Reliability diagrams before and after temperature scaling. The calibrated curve tracks the diagonal far more closely, reducing ensemble ECE from 0.1273 to 0.0240.](figures/fig4_reliability_diagram.png)",
    5: "![Figure 5: Pipeline throughput (FPS) as a function of the number of dogs per frame.](figures/fig5_fps_vs_dogs.png)",
    6: "![Figure 6: Representative in-the-wild detections captured from the deployed PWA on a mobile browser, spanning varied breeds, lighting, poses and backgrounds. Reported confidences are temperature-calibrated.](figures/fig6_sample_detections.png)",
    7: "![Figure 7: Pairwise model-agreement matrix. The two backbones agree on 92.41% of images.](figures/fig7_agreement_heatmap.png)",
    8: "![Figure 8: Per-breed top-1 accuracy, sorted. The long tail corresponds to visually similar breed pairs.](figures/fig8_per_breed_accuracy.png)",
    9: "![Figure 9: Distribution of calibrated confidence scores across the validation predictions.](figures/fig9_confidence_distribution.png)",
    10: "![Figure 10: Ensemble top-1 error reduction relative to each single model.](figures/fig11_error_reduction.png)",
    11: "![Figure 11: Per-stage latency breakdown of the inference pipeline.](figures/fig12_latency_breakdown.png)",
}

# figures inserted immediately BEFORE each heading (i.e. at end of prior section)
PLACEMENT = {
    "### 4.2 Stage 1: Dog Detection": [1],
    "### 5.3 Ensemble Performance": [3],
    "### 5.4 Temperature Calibration Analysis": [7, 10],
    "### 5.5 Real-Time Pipeline Latency Analysis": [4, 9],
    "### 5.6 Per-Breed Analysis": [5, 11, 6],
    "### 5.7 Comparison with Published Methods": [2, 8],
}

QUALITATIVE = (
    "\n**Qualitative in-the-wild behaviour.** Figure 6 shows representative frames "
    "captured from the deployed PWA on a mobile browser. The pipeline localises and "
    "identifies dogs across varied lighting, poses, backgrounds and breeds, and the "
    "calibrated confidences degrade gracefully on harder subjects rather than "
    "remaining spuriously high.\n"
)


def main():
    text = SRC.read_text()

    # 1. drop the trailing gallery section (from '## Figures' up to the following '---')
    text = re.sub(r"\n---\n\n## Figures\n.*?\n---\n", "\n---\n", text, flags=re.S)
    if "## Figures" in text:
        print("!! gallery section not removed cleanly")
        return

    # 2. add the qualitative paragraph at the end of 5.5, before the 5.6 heading
    text = text.replace("\n### 5.6 Per-Breed Analysis", QUALITATIVE + "\n### 5.6 Per-Breed Analysis")

    # 3. insert figures before their anchor headings
    for heading, nums in PLACEMENT.items():
        block = "\n".join(FIGS[n] + "\n" for n in nums)
        if heading not in text:
            print("!! anchor missing:", heading)
            return
        text = text.replace("\n" + heading, "\n" + block + "\n" + heading, 1)

    placed = len(re.findall(r"^!\[Figure ", text, flags=re.M))
    print("figures placed:", placed)
    for n in sorted(FIGS):
        if f"Figure {n}:" not in text:
            print("  !! missing Figure", n)

    if "--write" in sys.argv:
        SRC.write_text(text)
        print("wrote", SRC)
    else:
        print("(dry run; pass --write to apply)")


if __name__ == "__main__":
    main()
