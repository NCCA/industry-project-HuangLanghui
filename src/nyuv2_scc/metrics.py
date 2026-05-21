from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def occupancy_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    t = targets >= 0.5
    return occupancy_metrics(preds, t)


@torch.no_grad()
def occupancy_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    preds = preds.bool()
    targets = targets.bool()
    tp = torch.logical_and(preds, targets).sum().item()
    fp = torch.logical_and(preds, ~targets).sum().item()
    fn = torch.logical_and(~preds, targets).sum().item()
    union = torch.logical_or(preds, targets).sum().item()
    eps = 1e-8
    iou = tp / (union + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def average_metric_dicts(items):
    if not items:
        return {}
    keys = [k for k in items[0].keys() if k not in {"tp", "fp", "fn"}]
    return {k: float(sum(d[k] for d in items) / len(items)) for k in keys}
