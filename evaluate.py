#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyuv2_scc.config import get_device, load_config, resolve_path
from nyuv2_scc.dataset import NYUv2OccupancyDataset, prepare_splits_from_config
from nyuv2_scc.losses import BCEDiceLoss
from nyuv2_scc.model import ResidualAttentionUNet3D
from nyuv2_scc.train_eval import run_one_epoch


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the NYUv2 proxy occupancy test split.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = dict(cfg["data"])
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    mat_path = resolve_path(data_cfg["mat_path"], PROJECT_ROOT)
    data_cfg["cache_dir"] = str(resolve_path(data_cfg["cache_dir"], PROJECT_ROOT))
    metrics_dir = resolve_path(cfg["outputs"]["metrics_dir"], PROJECT_ROOT)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    splits = prepare_splits_from_config(mat_path, data_cfg)
    dataset = NYUv2OccupancyDataset(mat_path, splits[args.split], data_cfg, split=args.split)
    loader = DataLoader(dataset, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)))

    device = get_device(train_cfg.get("device", "auto"))
    model = ResidualAttentionUNet3D(
        in_channels=1,
        base_channels=int(model_cfg.get("base_channels", 8)),
        use_attention=bool(model_cfg.get("use_attention", True)),
    ).to(device)
    ckpt_path = resolve_path(args.checkpoint, PROJECT_ROOT)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    criterion = BCEDiceLoss(
        bce_weight=float(train_cfg.get("bce_weight", 1.0)),
        dice_weight=float(train_cfg.get("dice_weight", 0.5)),
        pos_weight=float(train_cfg.get("pos_weight", 1.0)),
    )
    metrics = run_one_epoch(
        model, loader, optimizer=None, criterion=criterion, device=device,
        threshold=float(train_cfg.get("threshold", 0.5)), train=False,
    )
    out_path = metrics_dir / f"{args.split}_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
