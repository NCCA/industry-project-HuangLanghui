#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the custom ablation study and PLY export scripts.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--indices", default="15,23,137,316")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional quick run limit for ablation metrics.")
    args = parser.parse_args()

    py = sys.executable
    shared = ["--config", args.config, "--checkpoint", args.checkpoint, "--split", args.split]
    max_args = [] if args.max_samples is None else ["--max-samples", str(args.max_samples)]

    run([py, "scripts/export_occupancy_ply.py", *shared, "--indices", args.indices])
    run([py, "scripts/noise_filtering_ablation.py", *shared, *max_args])
    run([py, "scripts/partial_generation_ablation.py", *shared, *max_args])
    run([py, "scripts/point_resolution_ablation.py", *shared, *max_args])
    run([py, "scripts/voxel_resolution_ablation.py", *shared, *max_args])
    run([py, "scripts/label_assisted_ablation.py", *shared, *max_args])

    print("Custom ablation study complete.")
    print("Key outputs:")
    print("  outputs/exports/*.ply")
    print("  outputs/metrics/noise_filtering_ablation.json")
    print("  outputs/metrics/partial_generation_ablation.json")
    print("  outputs/metrics/point_resolution_ablation.json")
    print("  outputs/metrics/voxel_resolution_ablation.json")
    print("  outputs/metrics/label_assisted_ablation.json")
    print("  outputs/figures/noise_filtering_ablation.png")
    print("  outputs/figures/partial_generation_ablation.png")
    print("  outputs/figures/point_resolution_ablation.png")
    print("  outputs/figures/voxel_resolution_ablation.png")
    print("  outputs/figures/label_assisted_ablation.png")


if __name__ == "__main__":
    main()
