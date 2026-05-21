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
from nyuv2_scc.custom_ablation import apply_noise_filter, counts_to_metrics, parse_noise_variants, update_counts
from nyuv2_scc.losses import BCEDiceLoss
from nyuv2_scc.model_utils import build_dataset, load_trained_model


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
    ax.set_xlabel("Input occupancy filtering setting")
    ax.set_ylabel("Score")
    ax.set_title("Noise Filtering Ablation")
    ax.legend(frameon=True, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Ablation study for simple noise filtering on input occupancy volumes.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--variants", default="baseline,remove_isolated,median,median_plus_isolated")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional quick test limit.")
    parser.add_argument("--example-indices", default="15,23", help="Comma-separated local split indices for example figures.")
    parser.add_argument("--out-json", default="outputs/metrics/noise_filtering_ablation.json")
    parser.add_argument("--out-fig", default="outputs/figures/noise_filtering_ablation.png")
    args = parser.parse_args()

    cfg, data_cfg, dataset, spec = build_dataset(args.config, PROJECT_ROOT, split=args.split)
    model, device, _ = load_trained_model(args.config, args.checkpoint, PROJECT_ROOT)
    threshold = args.threshold if args.threshold is not None else float(cfg["training"].get("threshold", 0.5))
    variants = parse_noise_variants(args.variants)
    criterion = BCEDiceLoss(
        bce_weight=float(cfg["training"].get("bce_weight", 1.0)),
        dice_weight=float(cfg["training"].get("dice_weight", 0.5)),
        pos_weight=float(cfg["training"].get("pos_weight", 1.0)),
    )

    stats = {name: {"tp": 0, "fp": 0, "fn": 0, "loss_sum": 0.0, "n": 0} for name in variants}
    total = len(dataset) if args.max_samples is None else min(len(dataset), int(args.max_samples))

    with torch.no_grad():
        for local_i in tqdm(range(total), desc="noise filtering ablation"):
            sample = dataset[local_i]
            base_input = sample["input"][0].numpy().astype(np.float32)
            target = sample["target"].unsqueeze(0).to(device).float()
            target_bool = target >= 0.5
            for name in variants:
                filtered = apply_noise_filter(base_input, name)
                x = torch.from_numpy(filtered[None, None, ...]).float().to(device)
                logits = model(x)
                pred = torch.sigmoid(logits) >= threshold
                update_counts(stats[name], pred, target_bool)
                stats[name]["loss_sum"] += float(criterion(logits, target).item())
                stats[name]["n"] += 1

    results = []
    for name in variants:
        row = {"name": name}
        row.update(counts_to_metrics(stats[name]["tp"], stats[name]["fp"], stats[name]["fn"]))
        row["loss"] = float(stats[name]["loss_sum"] / max(1, stats[name]["n"]))
        row["num_samples"] = int(stats[name]["n"])
        results.append(row)

    out_json = PROJECT_ROOT / args.out_json
    out_fig = PROJECT_ROOT / args.out_fig
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"split": args.split, "checkpoint": args.checkpoint, "threshold": threshold, "results": results}, f, indent=2)
    plot_results(out_json, out_fig)

    # Save a few example triplet figures for visual comparison.
    example_dir = PROJECT_ROOT / "outputs/figures/noise_filtering_examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    example_indices = [int(x.strip()) for x in args.example_indices.split(",") if x.strip()]
    with torch.no_grad():
        for local_i in example_indices:
            if local_i < 0 or local_i >= len(dataset):
                continue
            sample = dataset[local_i]
            target_occ = sample["target"][0].numpy().astype(np.float32)
            for name in variants:
                filtered = apply_noise_filter(sample["input"][0].numpy().astype(np.float32), name)
                x = torch.from_numpy(filtered[None, None, ...]).float().to(device)
                pred = (torch.sigmoid(model(x))[0, 0].detach().cpu().numpy() >= threshold).astype(np.float32)
                nyu_index = int(sample["index"].item())
                save_occupancy_triplet_advanced(
                    filtered, pred, target_occ, spec,
                    example_dir / f"noise_{name}_local{local_i:04d}_nyu{nyu_index:04d}.png",
                    title=f"Noise filtering: {name} | local {local_i}, NYUv2 {nyu_index}",
                )

    print(json.dumps(results, indent=2))
    print(f"Saved {out_json}")
    print(f"Saved {out_fig}")
    print(f"Saved example figures to {example_dir}")


if __name__ == "__main__":
    main()
