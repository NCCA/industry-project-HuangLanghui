#!/usr/bin/env python
"""Render input / prediction / target occupancy triplets for trained checkpoints.

Usage
-----
    python visualize.py --config configs/default.yaml \
        --checkpoint outputs/checkpoints/best.pt --split test --num 6

For each of the first ``--num`` scenes in the split, this runs the model, binarises
the prediction at the configured threshold, and saves a report-ready figure
(3D scatter + top-view projection of input, prediction and target) to
``outputs/visualizations/``. This is the qualitative evidence that the model
actually completes the scene, not just the summary metrics.

For a quick one-command end-to-end demonstration on a few samples, see ``demo.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyuv2_scc.config import get_device, load_config, resolve_path
from nyuv2_scc.dataset import NYUv2OccupancyDataset, build_specs, prepare_splits_from_config
from nyuv2_scc.model import ResidualAttentionUNet3D
from nyuv2_scc.advanced_visualization import save_occupancy_triplet_advanced


def main():
    """Parse CLI arguments, run the model on a few scenes, and save triplet figures."""
    parser = argparse.ArgumentParser(description="Visualize input / prediction / target occupancy volumes.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num", type=int, default=6)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = dict(cfg["data"])
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    mat_path = resolve_path(data_cfg["mat_path"], PROJECT_ROOT)
    data_cfg["cache_dir"] = str(resolve_path(data_cfg["cache_dir"], PROJECT_ROOT))
    vis_dir = resolve_path(cfg["outputs"]["vis_dir"], PROJECT_ROOT)
    vis_dir.mkdir(parents=True, exist_ok=True)
    _, spec = build_specs(data_cfg)

    splits = prepare_splits_from_config(mat_path, data_cfg)
    dataset = NYUv2OccupancyDataset(mat_path, splits[args.split], data_cfg, split=args.split)

    device = get_device(train_cfg.get("device", "auto"))
    model = ResidualAttentionUNet3D(
        in_channels=1,
        base_channels=int(model_cfg.get("base_channels", 8)),
        use_attention=bool(model_cfg.get("use_attention", True)),
    ).to(device)
    ckpt = torch.load(resolve_path(args.checkpoint, PROJECT_ROOT), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    threshold = float(train_cfg.get("threshold", 0.5))
    for local_i in range(min(args.num, len(dataset))):
        sample = dataset[local_i]
        x = sample["input"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() >= threshold).astype(np.float32)
        input_occ = sample["input"][0].numpy()
        target_occ = sample["target"][0].numpy()
        nyu_index = int(sample["index"].item())
        out_path = vis_dir / f"completion_{args.split}_{nyu_index:04d}.png"
        save_occupancy_triplet_advanced(input_occ, pred, target_occ, spec, out_path, title=f"NYUv2 sample {nyu_index}")
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
