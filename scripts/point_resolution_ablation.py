#!/usr/bin/env python
from __future__ import annotations

import argparse
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
from nyuv2_scc.custom_ablation import counts_to_metrics, parse_point_variants, subsample_points, update_counts
from nyuv2_scc.geometry import depth_to_pointcloud, points_to_occupancy
from nyuv2_scc.losses import BCEDiceLoss
from nyuv2_scc.model_utils import build_dataset, load_trained_model
from nyuv2_scc.nyuv2_io import NYUv2MatFile


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
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Maximum back-projected input points")
    ax.set_ylabel("Score")
    ax.set_title("Point Resolution Ablation")
    ax.legend(frameon=True, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Ablation study for point-cloud sampling resolution before voxelization.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--variants", default="1k:1000,2k:2000,5k:5000,10k:10000,all:all")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional quick test limit.")
    parser.add_argument("--example-indices", default="15,23", help="Comma-separated local split indices for example figures.")
    parser.add_argument("--out-json", default="outputs/metrics/point_resolution_ablation.json")
    parser.add_argument("--out-fig", default="outputs/figures/point_resolution_ablation.png")
    args = parser.parse_args()

    cfg, data_cfg, dataset, spec = build_dataset(args.config, PROJECT_ROOT, split=args.split)
    model, device, _ = load_trained_model(args.config, args.checkpoint, PROJECT_ROOT)
    threshold = args.threshold if args.threshold is not None else float(cfg["training"].get("threshold", 0.5))
    variants = parse_point_variants(args.variants)
    criterion = BCEDiceLoss(
        bce_weight=float(cfg["training"].get("bce_weight", 1.0)),
        dice_weight=float(cfg["training"].get("dice_weight", 0.5)),
        pos_weight=float(cfg["training"].get("pos_weight", 1.0)),
    )

    mat_path = resolve_path(data_cfg["mat_path"], PROJECT_ROOT)
    stats = {v["name"]: {"tp": 0, "fp": 0, "fn": 0, "loss_sum": 0.0, "n": 0, "points_sum": 0} for v in variants}
    total = len(dataset) if args.max_samples is None else min(len(dataset), int(args.max_samples))

    with NYUv2MatFile(mat_path) as reader, torch.no_grad():
        for local_i in tqdm(range(total), desc="point resolution ablation"):
            sample = dataset[local_i]
            nyu_index = int(sample["index"].item())
            raw_depth = reader.read_depth(data_cfg.get("input_key", "rawDepths"), nyu_index)
            full_points = depth_to_pointcloud(
                raw_depth,
                dataset.camera,
                depth_min_m=float(data_cfg.get("depth_min_m", 0.4)),
                depth_max_m=float(data_cfg.get("depth_max_m", 8.0)),
                pixel_stride=int(data_cfg.get("pixel_stride", 2)),
            )
            target = sample["target"].unsqueeze(0).to(device).float()
            target_bool = target >= 0.5
            for v in variants:
                pts = subsample_points(full_points, v["max_points"], seed=nyu_index + 1307)
                input_occ = points_to_occupancy(pts, spec, dilate_iterations=int(data_cfg.get("dilate_iterations", 1)))
                x = torch.from_numpy(input_occ[None, None, ...]).float().to(device)
                logits = model(x)
                pred = torch.sigmoid(logits) >= threshold
                name = v["name"]
                update_counts(stats[name], pred, target_bool)
                stats[name]["loss_sum"] += float(criterion(logits, target).item())
                stats[name]["n"] += 1
                stats[name]["points_sum"] += int(pts.shape[0])

    results = []
    for v in variants:
        name = v["name"]
        row = {"name": name, "max_points": v["max_points"] if v["max_points"] is not None else "all"}
        row.update(counts_to_metrics(stats[name]["tp"], stats[name]["fp"], stats[name]["fn"]))
        row["loss"] = float(stats[name]["loss_sum"] / max(1, stats[name]["n"]))
        row["mean_used_points"] = float(stats[name]["points_sum"] / max(1, stats[name]["n"]))
        row["num_samples"] = int(stats[name]["n"])
        results.append(row)

    out_json = PROJECT_ROOT / args.out_json
    out_fig = PROJECT_ROOT / args.out_fig
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"split": args.split, "checkpoint": args.checkpoint, "threshold": threshold, "results": results}, f, indent=2)
    plot_results(out_json, out_fig)

    example_dir = PROJECT_ROOT / "outputs/figures/point_resolution_examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    example_indices = [int(x.strip()) for x in args.example_indices.split(",") if x.strip()]
    with NYUv2MatFile(mat_path) as reader, torch.no_grad():
        for local_i in example_indices:
            if local_i < 0 or local_i >= len(dataset):
                continue
            sample = dataset[local_i]
            nyu_index = int(sample["index"].item())
            target_occ = sample["target"][0].numpy().astype(np.float32)
            raw_depth = reader.read_depth(data_cfg.get("input_key", "rawDepths"), nyu_index)
            full_points = depth_to_pointcloud(raw_depth, dataset.camera, float(data_cfg.get("depth_min_m", 0.4)), float(data_cfg.get("depth_max_m", 8.0)), int(data_cfg.get("pixel_stride", 2)))
            for v in variants:
                pts = subsample_points(full_points, v["max_points"], seed=nyu_index + 1307)
                input_occ = points_to_occupancy(pts, spec, dilate_iterations=int(data_cfg.get("dilate_iterations", 1)))
                x = torch.from_numpy(input_occ[None, None, ...]).float().to(device)
                pred = (torch.sigmoid(model(x))[0, 0].detach().cpu().numpy() >= threshold).astype(np.float32)
                save_occupancy_triplet_advanced(
                    input_occ, pred, target_occ, spec,
                    example_dir / f"point_{v['name']}_local{local_i:04d}_nyu{nyu_index:04d}.png",
                    title=f"Point resolution: {v['name']} | local {local_i}, NYUv2 {nyu_index}",
                )

    print(json.dumps(results, indent=2))
    print(f"Saved {out_json}")
    print(f"Saved {out_fig}")
    print(f"Saved example figures to {example_dir}")


if __name__ == "__main__":
    main()
