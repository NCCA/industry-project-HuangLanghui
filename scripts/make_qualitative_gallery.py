#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyuv2_scc.advanced_visualization import save_projection_gallery, save_qualitative_gallery, save_occupancy_triplet_advanced
from nyuv2_scc.dataset import build_specs
from nyuv2_scc.model_utils import build_dataset, load_trained_model


def parse_indices(text: str | None):
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Create a polished qualitative occupancy completion gallery.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--indices", default=None, help="Optional NYUv2 absolute indices, e.g. 15,23,137")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--out", default="outputs/figures/qualitative_gallery.png")
    parser.add_argument("--out-projections", default="outputs/figures/qualitative_projection_gallery.png")
    parser.add_argument("--save-triplets", action="store_true", help="Also save one advanced triplet per selected sample.")
    args = parser.parse_args()

    cfg, data_cfg, dataset, spec = build_dataset(args.config, PROJECT_ROOT, split=args.split)
    model, device, _ = load_trained_model(args.config, args.checkpoint, PROJECT_ROOT)
    threshold = args.threshold if args.threshold is not None else float(cfg["training"].get("threshold", 0.5))
    requested = parse_indices(args.indices)
    rows = []

    for local_i in range(len(dataset)):
        sample = dataset[local_i]
        nyu_index = int(sample["index"].item())
        if requested is not None and nyu_index not in requested:
            continue
        x = sample["input"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = (torch.sigmoid(model(x))[0, 0].cpu().numpy() >= threshold).astype(np.float32)
        row = {
            "index": nyu_index,
            "input": sample["input"][0].numpy(),
            "pred": pred,
            "target": sample["target"][0].numpy(),
        }
        rows.append(row)
        if len(rows) >= args.num and requested is None:
            break
        if requested is not None and len(rows) >= len(requested):
            break

    if not rows:
        raise RuntimeError("No samples selected. Check --indices and --split.")

    out = PROJECT_ROOT / args.out
    out_proj = PROJECT_ROOT / args.out_projections
    save_qualitative_gallery(rows, spec, out)
    save_projection_gallery(rows, out_proj)
    print(f"Saved {out}")
    print(f"Saved {out_proj}")
    if args.save_triplets:
        for row in rows:
            triplet_path = PROJECT_ROOT / f"outputs/figures/advanced_triplet_{args.split}_{int(row['index']):04d}.png"
            save_occupancy_triplet_advanced(row["input"], row["pred"], row["target"], spec, triplet_path, title=f"NYUv2 sample {int(row['index'])}")
            print(f"Saved {triplet_path}")


if __name__ == "__main__":
    main()
