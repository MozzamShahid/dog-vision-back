"""
Run the research-paper experiment pipeline inside a Hugging Face Spaces CPU container.

This script executes the same steps as the Kaggle notebook:
  1. exp1_validation_suite.py  — full Stanford Dogs validation evaluation
  2. exp3_calibration.py       — reliability diagrams and calibration metrics
  3. exp1_figures.py           — data-dependent figures

Expected layout in the Space root:
  models/                 EfficientNetV2S + ConvNeXtTiny .keras files
  data/Images/            Stanford Dogs images
  data/unique_breeds.json breed labels
  paper/scripts/          experiment scripts

If models/ or data/Images/ are missing, the script prints instructions and exits
without modifying anything.

Usage:
  python paper/scripts/run_on_huggingface.py
"""

import os
import subprocess
import sys
from pathlib import Path


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_script(script: str, *args: str) -> int:
    """Run a script from the project root with subprocess."""
    cmd = [sys.executable, script, *args]
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def list_outputs():
    """Print a summary of generated results and figures."""
    print("\nGenerated outputs:")
    for subdir in ["paper/results", "paper/figures"]:
        path = PROJECT_ROOT / subdir
        print(f"\n  {subdir}/")
        if not path.exists():
            print("    (empty)")
            continue
        for p in sorted(path.rglob("*")):
            if p.is_file():
                size_mb = p.stat().st_size / 1e6
                print(f"    {p.relative_to(PROJECT_ROOT)} ({size_mb:.2f} MB)")


def main():
    global PROJECT_ROOT
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    print_header("Dog Vision Research Paper — Hugging Face Spaces Runner")
    print(f"Project root: {PROJECT_ROOT}")

    # Check required assets
    models_dir = PROJECT_ROOT / "models"
    images_dir = PROJECT_ROOT / "data" / "Images"
    labels_file = PROJECT_ROOT / "data" / "unique_breeds.json"

    missing = []
    if not models_dir.exists() or not list(models_dir.glob("*.keras")):
        missing.append("models/ (should contain *.keras files)")
    if not images_dir.exists():
        missing.append("data/Images/ (Stanford Dogs images)")
    if not labels_file.exists():
        missing.append("data/unique_breeds.json")

    if missing:
        print("\nERROR: Required files are missing:")
        for item in missing:
            print(f"  - {item}")
        print("\nPlease upload or mount the following assets into this Space:")
        print("  1. models/               — both .keras model files + temp_scale JSONs")
        print("  2. data/Images/          — Stanford Dogs image folders")
        print("  3. data/unique_breeds.json — breed label list (already in repo)")
        print("\nIf you are using a persistent Space, you can add them as:")
        print("  - Files uploaded through the Files tab")
        print("  - A mounted dataset (hf.co/datasets/<user>/<dataset>)")
        sys.exit(1)

    print("\nAssets found:")
    print(f"  models: {models_dir} ({len(list(models_dir.glob('*.keras')))} .keras files)")
    print(f"  images: {images_dir}")
    print(f"  labels: {labels_file}")

    # Ensure output directories exist
    (PROJECT_ROOT / "paper" / "results").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "paper" / "figures").mkdir(parents=True, exist_ok=True)

    # Run experiments
    scripts_dir = PROJECT_ROOT / "paper" / "scripts"

    print_header("Experiment 1/3: Validation Suite")
    rc = run_script(
        str(scripts_dir / "exp1_validation_suite.py"),
        "--models-dir", str(models_dir),
        "--images-dir", str(images_dir),
        "--labels-path", str(labels_file),
        "--output-dir", "paper/results",
    )
    if rc != 0:
        print("\nValidation suite failed. Stopping pipeline.")
        sys.exit(rc)

    print_header("Experiment 2/3: Calibration Analysis")
    rc = run_script(
        str(scripts_dir / "exp3_calibration.py"),
        "--predictions", "paper/results/predictions.npz",
        "--output-dir", "paper/results",
        "--figures-dir", "paper/figures",
    )
    if rc != 0:
        print("\nCalibration analysis failed. Stopping pipeline.")
        sys.exit(rc)

    print_header("Experiment 3/3: Generate Figures")
    rc = run_script(
        str(scripts_dir / "exp1_figures.py"),
        "--results-dir", "paper/results",
        "--figures-dir", "paper/figures",
        "--labels-path", str(labels_file),
    )
    if rc != 0:
        print("\nFigure generation failed.")
        sys.exit(rc)

    print_header("Pipeline Complete")
    list_outputs()

    print("\nNext steps:")
    print("  - Download paper/results/ and paper/figures/ from the Space Files tab.")
    print("  - Or run paper/scripts/generate_all_figures.py for additional synthetic figures.")


if __name__ == "__main__":
    main()
