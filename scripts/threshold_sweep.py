#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyuv2_scc.advanced_visualization import plot_threshold_sweep
from nyuv2_scc.config import load_config, resolve_path
from nyuv2_scc.dataset import NYUv2OccupancyDataset, prepare_splits_from_config
from nyuv2_scc.model_utils import load_trained_model


def parse_thresholds(text: str):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def metrics_from_counts(tp: int, fp: int, fn: int):
    eps = 1e-8
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main():
    parser = argparse.ArgumentParser(description="Evaluate IoU/Precision/Recall/F1 across occupancy thresholds.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-fig", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = dict(cfg["data"])
    data_cfg["cache_dir"] = str(resolve_path(data_cfg["cache_dir"], PROJECT_ROOT))
    mat_path = resolve_path(data_cfg["mat_path"], PROJECT_ROOT)
    splits = prepare_splits_from_config(mat_path, data_cfg)
    dataset = NYUv2OccupancyDataset(mat_path, splits[args.split], data_cfg, split=args.split)
    batch_size = args.batch_size or int(cfg["training"].get("batch_size", 2))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=int(cfg["training"].get("num_workers", 0)))

    model, device, _ = load_trained_model(args.config, args.checkpoint, PROJECT_ROOT)
    thresholds = parse_thresholds(args.thresholds)
    counts = {t: {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}

    with torch.no_grad():
        for batch in tqdm(loader, desc="threshold sweep"):
            x = batch["input"].to(device)
            target = (batch["target"].to(device) >= 0.5)
            probs = torch.sigmoid(model(x))
            for threshold in thresholds:
                pred = probs >= threshold
                counts[threshold]["tp"] += int(torch.logical_and(pred, target).sum().item())
                counts[threshold]["fp"] += int(torch.logical_and(pred, ~target).sum().item())
                counts[threshold]["fn"] += int(torch.logical_and(~pred, target).sum().item())

    results = []
    for threshold in thresholds:
        row = {"threshold": threshold}
        row.update(metrics_from_counts(**counts[threshold]))
        results.append(row)

    out_json = PROJECT_ROOT / (args.out_json or f"outputs/metrics/threshold_sweep_{args.split}.json")
    out_fig = PROJECT_ROOT / (args.out_fig or f"outputs/figures/threshold_sweep_{args.split}.png")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"split": args.split, "checkpoint": args.checkpoint, "results": results}, f, indent=2)
    plot_threshold_sweep(out_json, out_fig)
    print(json.dumps(results, indent=2))
    print(f"Saved {out_json}")
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()
