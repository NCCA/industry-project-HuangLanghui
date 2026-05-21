from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import BCEDiceLoss
from .metrics import average_metric_dicts, occupancy_metrics_from_logits


def run_one_epoch(model, loader: DataLoader, optimizer, criterion, device, threshold: float, train: bool) -> Dict[str, float]:
    model.train(train)
    losses = []
    metric_items = []
    iterator = tqdm(loader, desc="train" if train else "eval", leave=False)
    for batch in iterator:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        metric_items.append(occupancy_metrics_from_logits(logits.detach(), y, threshold=threshold))
        avg_loss = sum(losses) / len(losses)
        iterator.set_postfix(loss=f"{avg_loss:.4f}")
    out = average_metric_dicts(metric_items)
    out["loss"] = float(sum(losses) / max(1, len(losses)))
    return out
