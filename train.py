#!/usr/bin/env python
"""Train the residual-attention 3D U-Net on real NYUv2 depth occupancy.

Usage
-----
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume outputs/checkpoints/last.pt

What it does
------------
1. Builds reproducible train/val/test splits from the NYUv2 ``.mat`` file and
   saves them to ``outputs/metrics/splits.json``.
2. Trains the model with the weighted BCE + Dice loss (see
   :mod:`nyuv2_scc.losses`) using AdamW.
3. After every epoch, writes ``last.pt`` and, whenever validation IoU improves,
   ``best.pt``; the full per-epoch history goes to
   ``outputs/metrics/train_history.json`` for the training-curve figure.

All hyper-parameters (grid size, epochs, batch size, loss weights, ...) come
from the YAML config, so this file contains no magic numbers.
"""
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
from nyuv2_scc.dataset import NYUv2OccupancyDataset, prepare_splits_from_config, save_splits
from nyuv2_scc.losses import BCEDiceLoss
from nyuv2_scc.model import ResidualAttentionUNet3D, count_parameters
from nyuv2_scc.train_eval import run_one_epoch


def main():
    """Parse CLI arguments, run the training loop, and checkpoint the best model."""
    parser = argparse.ArgumentParser(description="Train residual-attention 3D U-Net on real NYUv2 depth occupancy.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    mat_path = resolve_path(data_cfg["mat_path"], PROJECT_ROOT)
    data_cfg = dict(data_cfg)
    data_cfg["cache_dir"] = str(resolve_path(data_cfg["cache_dir"], PROJECT_ROOT))
    save_dir = resolve_path(train_cfg["save_dir"], PROJECT_ROOT)
    metrics_dir = resolve_path(cfg["outputs"]["metrics_dir"], PROJECT_ROOT)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    splits = prepare_splits_from_config(mat_path, data_cfg)
    save_splits(splits, metrics_dir / "splits.json")

    train_set = NYUv2OccupancyDataset(mat_path, splits["train"], data_cfg, split="train")
    val_set = NYUv2OccupancyDataset(mat_path, splits["val"], data_cfg, split="val")
    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=True,
    )

    device = get_device(train_cfg.get("device", "auto"))
    model = ResidualAttentionUNet3D(
        in_channels=1,
        base_channels=int(model_cfg.get("base_channels", 8)),
        use_attention=bool(model_cfg.get("use_attention", True)),
    ).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Device: {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    criterion = BCEDiceLoss(
        bce_weight=float(train_cfg.get("bce_weight", 1.0)),
        dice_weight=float(train_cfg.get("dice_weight", 0.5)),
        pos_weight=float(train_cfg.get("pos_weight", 1.0)),
    )

    start_epoch = 0
    best_iou = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_iou = float(ckpt.get("best_iou", -1.0))

    history = []
    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        print(f"Epoch {epoch + 1}/{train_cfg['epochs']}")
        train_metrics = run_one_epoch(
            model, train_loader, optimizer, criterion, device,
            threshold=float(train_cfg.get("threshold", 0.5)), train=True,
        )
        val_metrics = run_one_epoch(
            model, val_loader, optimizer, criterion, device,
            threshold=float(train_cfg.get("threshold", 0.5)), train=False,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, indent=2))

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "best_iou": best_iou,
        }
        torch.save(ckpt, save_dir / "last.pt")
        if val_metrics.get("iou", 0.0) > best_iou:
            best_iou = val_metrics["iou"]
            ckpt["best_iou"] = best_iou
            torch.save(ckpt, save_dir / "best.pt")

        with (metrics_dir / "train_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Best validation IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
