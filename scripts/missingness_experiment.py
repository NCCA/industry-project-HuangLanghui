#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyuv2_scc.advanced_visualization import plot_missingness_results
from nyuv2_scc.geometry import make_incomplete_input
from nyuv2_scc.model_utils import build_dataset, load_trained_model


def parse_levels(text: str):
    """Parse name:dropout:cuboids entries."""
    levels = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, dropout, cuboids = item.split(":")
        levels.append({"name": name, "dropout": float(dropout), "cuboids": int(cuboids)})
    return levels


def counts_to_metrics(tp: int, fp: int, fn: int):
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main():
    parser = argparse.ArgumentParser(description="Evaluate robustness under additional input occupancy missingness without retraining.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--levels", default="clean:0.0:0,light:0.05:1,medium:0.15:2,heavy:0.30:3")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-fig", default=None)
    args = parser.parse_args()

    cfg, data_cfg, dataset, spec = build_dataset(args.config, PROJECT_ROOT, split=args.split)
    model, device, _ = load_trained_model(args.config, args.checkpoint, PROJECT_ROOT)
    threshold = args.threshold if args.threshold is not None else float(cfg["training"].get("threshold", 0.5))
    levels = parse_levels(args.levels)
    counts = {level["name"]: {"tp": 0, "fp": 0, "fn": 0} for level in levels}
    seed = int(data_cfg.get("seed", 42))

    with torch.no_grad():
        for local_i in tqdm(range(len(dataset)), desc="missingness experiment"):
            sample = dataset[local_i]
            nyu_index = int(sample["index"].item())
            base_input = sample["input"][0].numpy()
            target = sample["target"].unsqueeze(0).to(device) >= 0.5
            for level in levels:
                rng = np.random.default_rng(seed + nyu_index + int(level["dropout"] * 1000) + 17 * level["cuboids"])
                degraded = make_incomplete_input(
                    base_input,
                    rng,
                    voxel_dropout=level["dropout"],
                    cuboid_masks=level["cuboids"],
                    cuboid_min_size=int(data_cfg.get("cuboid_min_size", 5)),
                    cuboid_max_size=int(data_cfg.get("cuboid_max_size", 14)),
                )
                x = torch.from_numpy(degraded[None, None, ...]).float().to(device)
                pred = torch.sigmoid(model(x)) >= threshold
                name = level["name"]
                counts[name]["tp"] += int(torch.logical_and(pred, target).sum().item())
                counts[name]["fp"] += int(torch.logical_and(pred, ~target).sum().item())
                counts[name]["fn"] += int(torch.logical_and(~pred, target).sum().item())

    results = []
    for level in levels:
        name = level["name"]
        row = dict(level)
        row.update(counts_to_metrics(**counts[name]))
        results.append(row)

    out_json = PROJECT_ROOT / (args.out_json or f"outputs/metrics/missingness_{args.split}.json")
    out_fig = PROJECT_ROOT / (args.out_fig or f"outputs/figures/missingness_{args.split}.png")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"split": args.split, "checkpoint": args.checkpoint, "results": results}, f, indent=2)
    plot_missingness_results(out_json, out_fig)
    print(json.dumps(results, indent=2))
    print(f"Saved {out_json}")
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()
