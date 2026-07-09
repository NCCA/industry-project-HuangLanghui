"""Occupancy evaluation metrics.

All metrics compare a *predicted* binary occupancy volume against the *target*
(proxy) occupancy volume, voxel by voxel. They are computed from the standard
confusion-matrix counts:

* ``tp`` (true positive)  -- voxels correctly predicted as occupied,
* ``fp`` (false positive) -- empty voxels wrongly predicted as occupied,
* ``fn`` (false negative) -- occupied voxels the model missed.

From these we derive the four scores reported throughout the project:

* **IoU**       = tp / (tp + fp + fn)     -- overall region overlap,
* **Precision** = tp / (tp + fp)          -- how clean the predicted surface is,
* **Recall**    = tp / (tp + fn)          -- how much of the scene was completed,
* **F1**        = harmonic mean of the two.

These are the same quantities the Dice term in :mod:`nyuv2_scc.losses`
optimises, so training and evaluation are measuring consistent things.
"""
from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def occupancy_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Threshold raw logits and compute occupancy metrics.

    Parameters
    ----------
    logits:
        Raw model outputs of shape ``(B, 1, D, H, W)``.
    targets:
        Binary occupancy targets of the same shape.
    threshold:
        Probability above which a voxel is declared occupied (default ``0.5``).

    Returns
    -------
    dict
        See :func:`occupancy_metrics`.
    """
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    t = targets >= 0.5
    return occupancy_metrics(preds, t)


@torch.no_grad()
def occupancy_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute IoU / precision / recall / F1 from two binary volumes.

    Parameters
    ----------
    preds, targets:
        Boolean (or 0/1) tensors of identical shape. Any leading batch/channel
        dimensions are pooled together, so this returns *micro-averaged* scores
        over every voxel in the input.

    Returns
    -------
    dict
        Keys ``iou``, ``precision``, ``recall``, ``f1`` (floats in ``[0, 1]``)
        plus the raw ``tp``, ``fp``, ``fn`` counts for downstream aggregation.
    """
    preds = preds.bool()
    targets = targets.bool()
    tp = torch.logical_and(preds, targets).sum().item()
    fp = torch.logical_and(preds, ~targets).sum().item()
    fn = torch.logical_and(~preds, targets).sum().item()
    union = torch.logical_or(preds, targets).sum().item()
    eps = 1e-8  # guard against division by zero for empty volumes
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
    """Average a list of metric dicts, ignoring the raw ``tp/fp/fn`` counts.

    Used to reduce the per-batch metric dictionaries produced during an epoch
    into a single mean score per metric. Raw counts are excluded because a plain
    mean of ratios is what the project reports; the counts are kept per-batch
    only for optional finer-grained analysis.
    """
    if not items:
        return {}
    keys = [k for k in items[0].keys() if k not in {"tp", "fp", "fn"}]
    return {k: float(sum(d[k] for d in items) / len(items)) for k in keys}
