#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyuv2_scc.advanced_visualization import plot_training_curves


def main():
    parser = argparse.ArgumentParser(description="Plot report-ready training curves from train_history.json.")
    parser.add_argument("--history", default="outputs/metrics/train_history.json")
    parser.add_argument("--out", default="outputs/figures/training_curves.png")
    args = parser.parse_args()
    out = PROJECT_ROOT / args.out
    plot_training_curves(PROJECT_ROOT / args.history, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
