#!/usr/bin/env python
"""One-command demonstration of 3D scene occupancy completion.

This script is the shortest path from "a trained checkpoint" to "see the model
actually completing a scene". For a handful of held-out NYUv2 test scenes it:

1. loads the incomplete **input** occupancy (built from raw NYUv2 depth),
2. runs the trained network to predict a **completed** occupancy volume,
3. compares both against the **target/proxy** occupancy (built from NYUv2's
   in-painted depth),
4. prints, per scene, how many voxels the model recovered and how much the
   completion improves IoU / recall over the raw input, and
5. saves a side-by-side figure (input | prediction | target) and, optionally,
   ``.ply`` point clouds you can open in MeshLab / CloudCompare.

The console table makes the key point explicit: the prediction is a *more
complete* volume than the input, and it agrees better with the target. That is
exactly the "3D scene occupancy completion" claim, demonstrated on concrete
examples rather than summary metrics alone.

Usage
-----
    python demo.py                       # 4 test scenes, default config + best.pt
    python demo.py --num 6 --export-ply  # more scenes, also write .ply clouds
    python demo.py --config configs/default.yaml \
                   --checkpoint outputs/checkpoints/best.pt --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyuv2_scc.advanced_visualization import save_error_map, save_occupancy_triplet_advanced
from nyuv2_scc.geometry import occupancy_to_points
from nyuv2_scc.metrics import occupancy_metrics
from nyuv2_scc.model_utils import build_dataset, load_trained_model
from nyuv2_scc.ply_utils import write_ascii_ply


def _binarize_prediction(model, input_tensor: torch.Tensor, device, threshold: float) -> np.ndarray:
    """Run the model on one input volume and return the thresholded occupancy grid."""
    x = input_tensor.unsqueeze(0).to(device)          # (1, 1, D, H, W)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (prob >= threshold).astype(np.float32)


def main():
    """Load a checkpoint, complete a few test scenes, and report + visualise the result."""
    parser = argparse.ArgumentParser(description="Demonstrate 3D scene occupancy completion on a few NYUv2 test scenes.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num", type=int, default=4, help="Number of scenes to demonstrate")
    parser.add_argument("--out-dir", default="outputs/demo", help="Where to write demo figures / PLY files")
    parser.add_argument("--export-ply", action="store_true", help="Also export input/prediction/target point clouds as .ply")
    args = parser.parse_args()

    # Rebuild the dataset (for this split) and the trained model from the config.
    cfg, _data_cfg, dataset, spec = build_dataset(args.config, PROJECT_ROOT, split=args.split)
    model, device, _cfg = load_trained_model(args.config, args.checkpoint, PROJECT_ROOT)
    threshold = float(cfg["training"].get("threshold", 0.5))

    out_dir = (PROJECT_ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(args.num, len(dataset))
    print(f"\nDemonstrating occupancy completion on {n} '{args.split}' scene(s)")
    print(f"Device: {device} | probability threshold: {threshold}\n")

    # Column header for the per-scene summary table.
    header = f"{'scene':>7} | {'input vox':>9} | {'pred vox':>8} | {'target vox':>10} | {'IoU in->tgt':>11} | {'IoU pred->tgt':>13} | {'recall gain':>11}"
    print(header)
    print("-" * len(header))

    for i in range(n):
        sample = dataset[i]
        nyu_index = int(sample["index"].item())
        input_occ = sample["input"][0].numpy()
        target_occ = sample["target"][0].numpy()
        pred_occ = _binarize_prediction(model, sample["input"], device, threshold)

        # Torch views for the metric helper (which expects boolean-ish tensors).
        t_input = torch.from_numpy(input_occ)
        t_pred = torch.from_numpy(pred_occ)
        t_target = torch.from_numpy(target_occ)

        # "How complete is the raw input already?" vs "how complete is the prediction?"
        input_vs_target = occupancy_metrics(t_input >= 0.5, t_target >= 0.5)
        pred_vs_target = occupancy_metrics(t_pred >= 0.5, t_target >= 0.5)

        n_input = int((input_occ >= 0.5).sum())
        n_pred = int((pred_occ >= 0.5).sum())
        n_target = int((target_occ >= 0.5).sum())
        recall_gain = pred_vs_target["recall"] - input_vs_target["recall"]

        print(
            f"{nyu_index:>7} | {n_input:>9} | {n_pred:>8} | {n_target:>10} | "
            f"{input_vs_target['iou']:>11.3f} | {pred_vs_target['iou']:>13.3f} | {recall_gain:>+11.3f}"
        )

        # Qualitative figures: the completion triplet and a TP/FP/FN error map.
        triplet_path = out_dir / f"demo_{args.split}_{nyu_index:04d}_completion.png"
        save_occupancy_triplet_advanced(
            input_occ, pred_occ, target_occ, spec, triplet_path,
            title=f"NYUv2 scene {nyu_index}: incomplete input -> completed prediction -> target",
        )
        error_path = out_dir / f"demo_{args.split}_{nyu_index:04d}_error.png"
        save_error_map(input_occ, pred_occ, target_occ, spec, error_path,
                       title=f"NYUv2 scene {nyu_index}: completion error map (TP/FP/FN)")

        if args.export_ply:
            # Blue = observed input, orange = model completion, green = target.
            write_ascii_ply(occupancy_to_points(input_occ, spec), out_dir / f"demo_{nyu_index:04d}_input.ply", color=(70, 130, 180))
            write_ascii_ply(occupancy_to_points(pred_occ, spec), out_dir / f"demo_{nyu_index:04d}_prediction.ply", color=(255, 140, 0))
            write_ascii_ply(occupancy_to_points(target_occ, spec), out_dir / f"demo_{nyu_index:04d}_target.ply", color=(60, 180, 75))

    print(
        "\nReading the table:\n"
        "  * 'pred vox' > 'input vox' means the model filled in voxels the raw depth was missing.\n"
        "  * 'IoU pred->tgt' > 'IoU in->tgt' and a positive 'recall gain' mean the completed\n"
        "    volume matches the target scene better than the incomplete input did.\n"
        f"\nFigures (and any .ply clouds) saved to: {out_dir}\n"
    )


if __name__ == "__main__":
    main()
