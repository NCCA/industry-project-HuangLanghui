#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyuv2_scc.advanced_visualization import save_occupancy_triplet_advanced, set_publication_style
from nyuv2_scc.config import resolve_path
from nyuv2_scc.custom_ablation import counts_to_metrics, parse_grid_variants, update_counts
from nyuv2_scc.dataset import NYUv2OccupancyDataset, build_specs, prepare_splits_from_config
from nyuv2_scc.losses import BCEDiceLoss
from nyuv2_scc.model_utils import load_project_config, load_trained_model


def plot_results(results_path: Path, out_path: Path) -> None:
    set_publication_style()
    with results_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["results"]
    labels = [row["name"] for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    for metric, marker in [("iou", "o"), ("f1", "s"), ("precision", "^"), ("recall", "D")]:
        ax.plot(x, [float(row[metric]) for row in rows], marker=marker, linewidth=2.2, label=metric.upper() if metric == "f1" else metric.capitalize())
    ax.set_xticks(x, labels, rotation=10)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Voxel grid resolution")
    ax.set_ylabel("Score")
    ax.set_title("Voxel Resolution Ablation")
    ax.legend(frameon=True, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Ablation study for voxel grid resolution using the same trained convolutional model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--grids", default="32,48,64")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional quick test limit.")
    parser.add_argument("--example-indices", default="15,23", help="Comma-separated local split indices for example figures.")
    parser.add_argument("--out-json", default="outputs/metrics/voxel_resolution_ablation.json")
    parser.add_argument("--out-fig", default="outputs/figures/voxel_resolution_ablation.png")
    args = parser.parse_args()

    cfg, base_data_cfg, mat_path = load_project_config(args.config, PROJECT_ROOT)
    model, device, _ = load_trained_model(args.config, args.checkpoint, PROJECT_ROOT)
    threshold = args.threshold if args.threshold is not None else float(cfg["training"].get("threshold", 0.5))
    variants = parse_grid_variants(args.grids)
    criterion = BCEDiceLoss(
        bce_weight=float(cfg["training"].get("bce_weight", 1.0)),
        dice_weight=float(cfg["training"].get("dice_weight", 0.5)),
        pos_weight=float(cfg["training"].get("pos_weight", 1.0)),
    )

    results = []
    example_indices = [int(x.strip()) for x in args.example_indices.split(",") if x.strip()]
    example_dir = PROJECT_ROOT / "outputs/figures/voxel_resolution_examples"
    example_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        data_cfg = copy.deepcopy(base_data_cfg)
        data_cfg["grid_size"] = list(variant["grid_size"])
        # Isolate caches by grid size to avoid mixing different volume shapes.
        base_cache = Path(str(base_data_cfg.get("cache_dir", PROJECT_ROOT / "data/cache")))
        data_cfg["cache_dir"] = str(base_cache.parent / f"{base_cache.name}_grid_{variant['name']}")
        splits = prepare_splits_from_config(mat_path, data_cfg)
        dataset = NYUv2OccupancyDataset(mat_path, splits[args.split], data_cfg, split=args.split)
        _, spec = build_specs(data_cfg)
        total = len(dataset) if args.max_samples is None else min(len(dataset), int(args.max_samples))
        counts = {"tp": 0, "fp": 0, "fn": 0, "loss_sum": 0.0, "n": 0}
        with torch.no_grad():
            for local_i in tqdm(range(total), desc=f"voxel grid {variant['name']}"):
                sample = dataset[local_i]
                x = sample["input"].unsqueeze(0).to(device).float()
                target = sample["target"].unsqueeze(0).to(device).float()
                logits = model(x)
                pred = torch.sigmoid(logits) >= threshold
                update_counts(counts, pred, target >= 0.5)
                counts["loss_sum"] += float(criterion(logits, target).item())
                counts["n"] += 1
        row = {"name": variant["name"], "grid_size": list(variant["grid_size"])}
        row.update(counts_to_metrics(counts["tp"], counts["fp"], counts["fn"]))
        row["loss"] = float(counts["loss_sum"] / max(1, counts["n"]))
        row["num_samples"] = int(counts["n"])
        row["note"] = "Inference-time voxel representation ablation using the same trained convolutional checkpoint."
        results.append(row)

        with torch.no_grad():
            for local_i in example_indices:
                if local_i < 0 or local_i >= len(dataset):
                    continue
                sample = dataset[local_i]
                x = sample["input"].unsqueeze(0).to(device).float()
                pred = (torch.sigmoid(model(x))[0, 0].detach().cpu().numpy() >= threshold).astype(np.float32)
                nyu_index = int(sample["index"].item())
                save_occupancy_triplet_advanced(
                    sample["input"][0].numpy().astype(np.float32), pred, sample["target"][0].numpy().astype(np.float32), spec,
                    example_dir / f"voxel_{variant['name']}_local{local_i:04d}_nyu{nyu_index:04d}.png",
                    title=f"Voxel resolution: {variant['name']} | local {local_i}, NYUv2 {nyu_index}",
                )

    out_json = PROJECT_ROOT / args.out_json
    out_fig = PROJECT_ROOT / args.out_fig
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"split": args.split, "checkpoint": args.checkpoint, "threshold": threshold, "results": results}, f, indent=2)
    plot_results(out_json, out_fig)
    print(json.dumps(results, indent=2))
    print(f"Saved {out_json}")
    print(f"Saved {out_fig}")
    print(f"Saved example figures to {example_dir}")


if __name__ == "__main__":
    main()
