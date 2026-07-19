"""Fig 6: Sample in-the-wild detection frames captured from the deployed PWA.

Builds a 2x2 montage from paper/camera_testing/*.png.
Run: python paper/scripts/fig6_sample_detections.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

SRC = Path("paper/camera_testing")
OUT = Path("paper/figures/fig6_sample_detections.png")

PANELS = [
    ("image_1.png", "(a) Golden Retriever, 88%"),
    ("image_3.png", "(b) Walker Hound, 93%"),
    ("image_2.png", "(c) Walker Hound, 71%"),
    ("image_4.png", "(d) Chihuahua, 52%"),
]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, (fname, caption) in zip(axes.ravel(), PANELS):
        path = SRC / fname
        ax.imshow(mpimg.imread(path))
        ax.set_title(caption, fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print("Fig 6 saved:", OUT)


if __name__ == "__main__":
    main()
