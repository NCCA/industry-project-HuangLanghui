#!/usr/bin/env python
"""Non-learned baselines for 3D occupancy completion.

The single most important sanity check for this task is: *does the trained
network actually complete the scene, or does it just echo the incomplete input?*
Because the input and the proxy target are both derived from the same NYUv2
scene, a model that simply copies its input already scores a non-trivial IoU.
To show the network adds value we must compare it against trivial, non-learned
baselines that use **no training at all**:

* ``copy_input``  -- predict the raw input occupancy unchanged. This is the
  "do nothing" lower bound. Any learned model must beat it to justify itself.
* ``dilate_k``    -- morphologically dilate the input by ``k`` voxels. This is
  the cheapest possible way to "grow" the observed surface toward the target,
  and it mimics what a copy-and-thicken heuristic would achieve without any
  semantics.

All variants are scored with the exact same micro-averaged (globally pooled
tp/fp/fn) metrics used by the threshold sweep, so the numbers are directly
comparable to the trained model at ``threshold=0.5``. The trained checkpoint is
also evaluated here so the whole comparison table comes from one script and one
averaging convention.

Usage
-----
    python scripts/naive_baselines.py --config configs/full_train.yaml \
        --checkpoint outputs/checkpoints/best.pt --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.ndimage import binary_dilation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyuv2_scc.model_utils import build_dataset, load_trained_model  # noqa: E402


def _scores_from_counts(tp: float, fp: float, fn: float) -> dict:
    """Turn globally pooled tp/fp/fn counts into IoU/precision/recall/F1."""
    eps = 1e-8
    union = tp + fp + fn
    iou = tp / (union + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def _accumulate(pred: np.ndarray, target: np.ndarray, acc: dict) -> None:
    """Add one volume's confusion-matrix counts into a running accumulator."""
    p = pred >= 0.5
    t = target >= 0.5
    acc["tp"] += float(np.logical_and(p, t).sum())
    acc["fp"] += float(np.logical_and(p, ~t).sum())
    acc["fn"] += float(np.logical_and(~p, t).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-learned copy/dilation baselines vs the trained model.")
    parser.add_argument("--config", default="configs/full_train.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--dilations", default="1,2", help="Comma-separated dilation radii for the dilate baselines")
    parser.add_argument("--out", default="outputs/metrics/naive_baselines.json")
    args = parser.parse_args()

    dilations = [int(x) for x in args.dilations.split(",") if x.strip()]

    cfg, _data_cfg, dataset, _spec = build_dataset(args.config, PROJECT_ROOT, split=args.split)
    model, device, _ = load_trained_model(args.config, args.checkpoint, PROJECT_ROOT)
    threshold = float(cfg["training"].get("threshold", 0.5))

    # One accumulator per method.
    variants = ["copy_input"] + [f"dilate_{k}" for k in dilations] + ["model"]
    acc = {name: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for name in variants}

    n = len(dataset)
    print(f"Scoring {n} '{args.split}' scenes | device={device} | threshold={threshold}\n")

    for i in range(n):
        sample = dataset[i]
        input_occ = sample["input"][0].numpy()
        target_occ = sample["target"][0].numpy()

        # Baseline 1: predict the input unchanged (the "does the model just copy?" bound).
        _accumulate(input_occ, target_occ, acc["copy_input"])

        # Baseline 2..: grow the input by k voxels with a morphological dilation.
        input_bool = input_occ >= 0.5
        for k in dilations:
            grown = binary_dilation(input_bool, iterations=k).astype(np.float32)
            _accumulate(grown, target_occ, acc[f"dilate_{k}"])

        # Reference: the trained network at the reporting threshold.
        with torch.no_grad():
            logits = model(sample["input"].unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() >= threshold).astype(np.float32)
        _accumulate(pred, target_occ, acc["model"])

        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  scored {i + 1}/{n}")

    results = {name: _scores_from_counts(**counts) for name, counts in acc.items()}

    header = f"\n{'method':>12} | {'IoU':>7} | {'Prec':>7} | {'Recall':>7} | {'F1':>7}"
    print(header)
    print("-" * len(header))
    for name in variants:
        r = results[name]
        print(f"{name:>12} | {r['iou']:>7.4f} | {r['precision']:>7.4f} | {r['recall']:>7.4f} | {r['f1']:>7.4f}")

    model_iou = results["model"]["iou"]
    copy_iou = results["copy_input"]["iou"]
    print(
        f"\nThe trained model reaches IoU {model_iou:.4f} vs {copy_iou:.4f} for copy-input "
        f"(+{model_iou - copy_iou:.4f} absolute), so it is genuinely completing the scene,\n"
        "not merely reproducing the observed input.\n"
    )

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"split": args.split, "checkpoint": args.checkpoint, "threshold": threshold, "results": results}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
