#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyuv2_scc.advanced_visualization import plot_metrics_bar


def main():
    parser = argparse.ArgumentParser(description="Plot a report-ready bar chart for IoU/Precision/Recall/F1.")
    parser.add_argument("--metrics", default="outputs/metrics/test_metrics.json")
    parser.add_argument("--out", default="outputs/figures/test_metrics_bar.png")
    parser.add_argument("--title", default="NYUv2 Proxy Occupancy Test Metrics")
    args = parser.parse_args()
    out = PROJECT_ROOT / args.out
    plot_metrics_bar(PROJECT_ROOT / args.metrics, out, title=args.title)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
