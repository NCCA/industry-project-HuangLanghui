#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyuv2_scc.advanced_visualization import plot_experiment_comparison


def main():
    parser = argparse.ArgumentParser(description="Plot IoU/F1 comparison for experiment result JSON files.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default="outputs/figures/experiment_comparison.png")
    parser.add_argument("--title", default="Experiment Comparison")
    args = parser.parse_args()
    out = PROJECT_ROOT / args.out
    plot_experiment_comparison(PROJECT_ROOT / args.results, out, title=args.title)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
