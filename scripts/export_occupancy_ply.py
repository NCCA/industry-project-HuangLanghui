#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyuv2_scc.geometry import occupancy_to_points
from nyuv2_scc.model_utils import build_dataset, load_trained_model
from nyuv2_scc.ply_utils import write_ascii_ply


def parse_indices(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def export_error_ply(pred: np.ndarray, target: np.ndarray, spec, path: Path) -> Path:
    pred_b = pred > 0.5
    target_b = target > 0.5
    tp = pred_b & target_b
    fp = pred_b & ~target_b
    fn = ~pred_b & target_b

    pts_all = []
    cols_all = []
    for vol, color in [
        (tp.astype(np.float32), (46, 160, 67)),     # true positive: green
        (fp.astype(np.float32), (220, 50, 47)),     # false positive: red
        (fn.astype(np.float32), (245, 166, 35)),    # false negative: orange
    ]:
        pts = occupancy_to_points(vol, spec, max_points=None)
        if pts.size:
            pts_all.append(pts)
            cols_all.append(np.tile(np.asarray(color, dtype=np.uint8), (pts.shape[0], 1)))
    if pts_all:
        points = np.concatenate(pts_all, axis=0)
        colors = np.concatenate(cols_all, axis=0)
    else:
        points = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.uint8)
    return write_ascii_ply(points, path, colors=colors)


def main():
    parser = argparse.ArgumentParser(description="Export input, predicted, target, and error occupancy volumes as PLY point clouds.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--indices", default="15,23,137,316", help="Comma-separated local dataset indices within the selected split.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-points", type=int, default=None, help="Optional point subsampling for each exported PLY.")
    parser.add_argument("--out-dir", default="outputs/exports")
    args = parser.parse_args()

    cfg, data_cfg, dataset, spec = build_dataset(args.config, PROJECT_ROOT, split=args.split)
    model, device, _ = load_trained_model(args.config, args.checkpoint, PROJECT_ROOT)
    threshold = args.threshold if args.threshold is not None else float(cfg["training"].get("threshold", 0.5))
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    with torch.no_grad():
        for local_idx in parse_indices(args.indices):
            if local_idx < 0 or local_idx >= len(dataset):
                print(f"[WARN] local index {local_idx} is outside split length {len(dataset)}; skipping")
                continue
            sample = dataset[local_idx]
            nyu_index = int(sample["index"].item())
            input_occ = sample["input"][0].numpy().astype(np.float32)
            target_occ = sample["target"][0].numpy().astype(np.float32)
            x = sample["input"].unsqueeze(0).to(device)
            probs = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()
            pred_occ = (probs >= threshold).astype(np.float32)

            prefix = f"{args.split}_local{local_idx:04d}_nyu{nyu_index:04d}"
            paths = {
                "input": out_dir / f"{prefix}_input.ply",
                "prediction": out_dir / f"{prefix}_prediction.ply",
                "target_proxy": out_dir / f"{prefix}_target_proxy.ply",
                "error_tp_fp_fn": out_dir / f"{prefix}_error_tp_fp_fn.ply",
            }
            write_ascii_ply(occupancy_to_points(input_occ, spec, max_points=args.max_points), paths["input"], color=(52, 120, 246))
            write_ascii_ply(occupancy_to_points(pred_occ, spec, max_points=args.max_points), paths["prediction"], color=(39, 174, 96))
            write_ascii_ply(occupancy_to_points(target_occ, spec, max_points=args.max_points), paths["target_proxy"], color=(245, 166, 35))
            export_error_ply(pred_occ, target_occ, spec, paths["error_tp_fp_fn"])

            manifest.append({
                "split": args.split,
                "local_index": local_idx,
                "nyu_index": nyu_index,
                "threshold": threshold,
                "files": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in paths.items()},
            })
            print(f"Exported PLY files for local index {local_idx} / NYUv2 index {nyu_index}")

    manifest_path = out_dir / "export_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")
    print("Open the .ply files in MeshLab, CloudCompare, Blender, or Open3D to rotate the 3D point clouds.")


if __name__ == "__main__":
    main()
