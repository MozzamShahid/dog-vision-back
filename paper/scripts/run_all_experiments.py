"""
Local/cloud orchestrator for the research-paper experiment pipeline.

Runs, in order:
  1. exp1_validation_suite.py   — full validation evaluation
  2. exp3_calibration.py        — reliability diagrams & calibration metrics
  3. exp1_figures.py            — data-dependent figures
  4. generate_all_figures.py    — remaining synthetic/constant figures

The orchestrator reports which results and figures were produced and exits with
a non-zero code if any step fails.

Usage:
  # Run everything with default paths
  python paper/scripts/run_all_experiments.py

  # Override asset locations
  python paper/scripts/run_all_experiments.py \
      --models-dir models \
      --images-dir data/Images \
      --labels-path data/unique_breeds.json
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_script(script_path: Path, *args: str) -> int:
    """Run a script via subprocess from the project root."""
    cmd = [sys.executable, str(script_path), *args]
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def list_files(directory: Path, pattern: str = "*") -> list:
    """Return sorted list of file paths matching a glob."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob(pattern) if p.is_file())


def summarize():
    """Print summary of produced results and figures."""
    results_dir = PROJECT_ROOT / "paper" / "results"
    figures_dir = PROJECT_ROOT / "paper" / "figures"

    result_files = list_files(results_dir)
    figure_files = list_files(figures_dir)

    print_header("Produced Results & Figures")

    print(f"\nResults ({len(result_files)} files in paper/results/):")
    for p in result_files:
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.relative_to(PROJECT_ROOT)} ({size_mb:.2f} MB)")

    print(f"\nFigures ({len(figure_files)} files in paper/figures/):")
    for p in figure_files:
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.relative_to(PROJECT_ROOT)} ({size_mb:.2f} MB)")

    expected_results = [
        "metrics.json",
        "predictions.npz",
        "confusion_matrix.npy",
        "confusion_matrix_raw.npy",
        "per_breed_accuracy.csv",
        "confidence_histogram.csv",
        "calibration_metrics.csv",
        "backbone_comparison.csv",
        "temperature_calibration.csv",
    ]
    expected_figures = [
        "fig2_confusion_matrix_top20.png",
        "fig4_reliability_diagram.png",
        "fig5_fps_vs_dogs.png",
        "fig7_agreement_heatmap.png",
        "fig8_per_breed_accuracy.png",
        "fig9_confidence_distribution.png",
        "fig11_error_reduction.png",
        "fig12_latency_breakdown.png",
    ]

    print("\nKey outputs checklist:")
    for name in expected_results:
        present = "✓" if (results_dir / name).exists() else "✗"
        print(f"  [{present}] paper/results/{name}")
    for name in expected_figures:
        present = "✓" if (figures_dir / name).exists() else "✗"
        print(f"  [{present}] paper/figures/{name}")


def main():
    global PROJECT_ROOT
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Run the full paper experiment pipeline")
    parser.add_argument("--models-dir", default="models", help="Directory containing .keras models")
    parser.add_argument("--images-dir", default="data/Images", help="Stanford Dogs Images/ folder")
    parser.add_argument("--labels-path", default="data/unique_breeds.json", help="Breed labels JSON")
    parser.add_argument("--output-dir", default="paper/results", help="Where to write numerical results")
    parser.add_argument("--figures-dir", default="paper/figures", help="Where to write figures")
    parser.add_argument("--max-images", type=int, default=0, help="0 = all images; else random sample")
    args = parser.parse_args()

    print_header("Dog Vision Research Paper — Full Experiment Orchestrator")
    print(f"Project root: {PROJECT_ROOT}")

    scripts_dir = PROJECT_ROOT / "paper" / "scripts"
    results_dir = PROJECT_ROOT / args.output_dir
    figures_dir = PROJECT_ROOT / args.figures_dir

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validation suite
    print_header("Step 1/4: Validation Suite (exp1_validation_suite.py)")
    exp1_args = [
        "--models-dir", args.models_dir,
        "--images-dir", args.images_dir,
        "--labels-path", args.labels_path,
        "--output-dir", args.output_dir,
    ]
    if args.max_images > 0:
        exp1_args += ["--max-images", str(args.max_images)]

    rc = run_script(scripts_dir / "exp1_validation_suite.py", *exp1_args)
    if rc != 0:
        print("\nStep 1 failed. Aborting pipeline.")
        sys.exit(rc)

    # 2. Calibration
    print_header("Step 2/4: Calibration Analysis (exp3_calibration.py)")
    rc = run_script(
        scripts_dir / "exp3_calibration.py",
        "--predictions", os.path.join(args.output_dir, "predictions.npz"),
        "--output-dir", args.output_dir,
        "--figures-dir", args.figures_dir,
    )
    if rc != 0:
        print("\nStep 2 failed. Aborting pipeline.")
        sys.exit(rc)

    # 3. Data-dependent figures
    print_header("Step 3/4: Data-Dependent Figures (exp1_figures.py)")
    rc = run_script(
        scripts_dir / "exp1_figures.py",
        "--results-dir", args.output_dir,
        "--figures-dir", args.figures_dir,
        "--labels-path", args.labels_path,
    )
    if rc != 0:
        print("\nStep 3 failed. Aborting pipeline.")
        sys.exit(rc)

    # 4. Synthetic/constant figures
    print_header("Step 4/4: Additional Figures (generate_all_figures.py)")
    rc = run_script(scripts_dir / "generate_all_figures.py")
    if rc != 0:
        print("\nStep 4 failed.")
        sys.exit(rc)

    # Summary
    summarize()
    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
