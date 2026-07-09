#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], allow_fail: bool = False) -> None:
    print("+", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError:
        if allow_fail:
            print("[WARN] command failed, skipping")
        else:
            raise


def main():
    parser = argparse.ArgumentParser(description="Generate the main advanced figures for the report.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--indices", default=None, help="Optional NYUv2 sample indices for gallery, e.g. 15,23,137,316")
    parser.add_argument("--error-indices", default=None, help="Optional challenging sample indices for error maps, e.g. 109,272")
    parser.add_argument("--skip-model-eval", action="store_true", help="Only make charts from existing JSON files.")
    args = parser.parse_args()

    py = sys.executable
    run([py, "scripts/plot_training_curves.py"], allow_fail=True)
    run([py, "scripts/plot_metrics_bar.py"], allow_fail=True)

    if not args.skip_model_eval:
        run([py, "scripts/threshold_sweep.py", "--config", args.config, "--checkpoint", args.checkpoint, "--split", args.split])
        gallery_cmd = [py, "scripts/make_qualitative_gallery.py", "--config", args.config, "--checkpoint", args.checkpoint, "--split", args.split, "--num", str(args.num), "--save-triplets"]
        if args.indices:
            gallery_cmd.extend(["--indices", args.indices])
        run(gallery_cmd)
        err_cmd = [py, "scripts/plot_error_map.py", "--config", args.config, "--checkpoint", args.checkpoint, "--split", args.split, "--num", "2"]
        if args.error_indices:
            err_cmd.extend(["--indices", args.error_indices])
        run(err_cmd)
        run([py, "scripts/missingness_experiment.py", "--config", args.config, "--checkpoint", args.checkpoint, "--split", args.split])

    print("Advanced figures are saved in outputs/figures/.")


if __name__ == "__main__":
    main()
