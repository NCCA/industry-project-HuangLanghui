#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyuv2_scc.advanced_visualization import save_error_map
from nyuv2_scc.model_utils import build_dataset, load_trained_model


def parse_indices(text: str | None):
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Create TP/FP/FN error maps for selected samples.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num", type=int, default=2)
    parser.add_argument("--indices", default=None, help="Optional NYUv2 absolute indices, e.g. 109,272")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    cfg, data_cfg, dataset, spec = build_dataset(args.config, PROJECT_ROOT, split=args.split)
    model, device, _ = load_trained_model(args.config, args.checkpoint, PROJECT_ROOT)
    threshold = args.threshold if args.threshold is not None else float(cfg["training"].get("threshold", 0.5))
    requested = parse_indices(args.indices)
    saved = 0

    for local_i in range(len(dataset)):
        sample = dataset[local_i]
        nyu_index = int(sample["index"].item())
        if requested is not None and nyu_index not in requested:
            continue
        x = sample["input"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = (torch.sigmoid(model(x))[0, 0].cpu().numpy() >= threshold).astype(np.float32)
        input_occ = sample["input"][0].numpy()
        target_occ = sample["target"][0].numpy()
        out = PROJECT_ROOT / f"outputs/figures/error_map_{args.split}_{nyu_index:04d}.png"
        save_error_map(input_occ, pred, target_occ, spec, out, title=f"TP/FP/FN Error Map - NYUv2 sample {nyu_index}")
        print(f"Saved {out}")
        saved += 1
        if requested is None and saved >= args.num:
            break
        if requested is not None and saved >= len(requested):
            break

    if saved == 0:
        raise RuntimeError("No error maps saved. Check --indices and --split.")


if __name__ == "__main__":
    main()
