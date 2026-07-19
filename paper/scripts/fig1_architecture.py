"""Fig 1: Two-stage pipeline architecture diagram (self-contained, no mermaid CLI).

Renders the detection -> crop -> ensemble classification -> calibration flow
as a labelled block diagram. Run: python paper/scripts/fig1_architecture.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, x, y, w, h, text, fc, tc="white", fs=10, bold=True):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.5, edgecolor="white", facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
        linewidth=1.8, color="#ffd369", zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    fig.patch.set_facecolor("#0f0f1a")

    C_IN = "#3a3a5a"
    C_DET = "#4ecca3"
    C_CLS = "#e94560"
    C_CAL = "#ffd369"
    C_OUT = "#3a3a5a"

    # Row layout (y)
    ymid = 3.9
    ylow = 1.6

    box(ax, 0.2, ymid, 1.8, 1.1, "Camera frame\n(WebSocket)", C_IN)
    box(ax, 2.4, ymid, 2.1, 1.1, "Stage 1\nEfficientDet-Lite0\ndog detection", C_DET, tc="#0f0f1a")
    box(ax, 4.9, ymid, 1.7, 1.1, "Crop + filter\n(dogs, conf>0.3)", C_IN)

    # Stage 2 ensemble (two parallel classifiers)
    box(ax, 7.0, 4.6, 2.2, 0.95, "EfficientNetV2S\n(T=0.67)", C_CLS)
    box(ax, 7.0, 3.35, 2.2, 0.95, "ConvNeXtTiny\n(T=0.73)", C_CLS)
    box(ax, 9.5, ymid, 2.2, 1.1, "Temperature scale\n+ softmax average", C_CAL, tc="#0f0f1a")

    box(ax, 4.9, ylow, 2.1, 1.0, "Top-K breeds\n+ calibrated conf.", C_OUT)
    box(ax, 7.4, ylow, 2.3, 1.0, "Overlay render\n60 FPS PWA", C_OUT)

    # Arrows
    arrow(ax, 2.0, ymid + 0.55, 2.4, ymid + 0.55)
    arrow(ax, 4.5, ymid + 0.55, 4.9, ymid + 0.55)
    arrow(ax, 6.6, ymid + 0.55, 7.0, 5.05)
    arrow(ax, 6.6, ymid + 0.55, 7.0, 3.82)
    arrow(ax, 9.2, 5.05, 9.5, ymid + 0.75)
    arrow(ax, 9.2, 3.82, 9.5, ymid + 0.4)
    # calibration -> output (down and back left)
    arrow(ax, 10.6, ymid, 10.6, ylow + 1.4)
    arrow(ax, 10.6, ylow + 1.4, 9.7, ylow + 0.5)
    arrow(ax, 7.4, ylow + 0.5, 7.0, ylow + 0.5)

    # Stage brackets / labels
    ax.text(3.45, ymid + 1.35, "STAGE 1 — DETECTION", ha="center", fontsize=10,
            color=C_DET, fontweight="bold")
    ax.text(8.6, 5.75, "STAGE 2 — ENSEMBLE CLASSIFICATION", ha="center", fontsize=10,
            color=C_CLS, fontweight="bold")

    ax.set_title("Two-Stage Real-Time Multi-Dog Breed Identification Pipeline",
                 fontsize=14, color="white", fontweight="bold", pad=14)

    plt.tight_layout()
    out = "paper/figures/fig1_architecture.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    print("Fig 1 saved:", out)


if __name__ == "__main__":
    main()
