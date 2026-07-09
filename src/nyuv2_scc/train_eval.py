"""Shared training / evaluation epoch loop.

Both ``train.py`` and ``evaluate.py`` drive the model through
:func:`run_one_epoch`. Keeping a single implementation guarantees that the
numbers reported at evaluation time are produced by exactly the same forward
pass, thresholding and metric code used during training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import BCEDiceLoss
from .metrics import average_metric_dicts, occupancy_metrics_from_logits


def run_one_epoch(model, loader: DataLoader, optimizer, criterion, device, threshold: float, train: bool) -> Dict[str, float]:
    """Run a single pass over ``loader`` and return averaged metrics.

    The same function performs training and evaluation; the ``train`` flag
    switches the behaviour:

    * ``train=True``  -- puts the model in train mode, enables autograd, and
      runs the backward pass + optimiser step for every batch.
    * ``train=False`` -- puts the model in eval mode (frozen BatchNorm stats),
      disables autograd, and only records loss and metrics. ``optimizer`` may
      be ``None`` in this case (as done by ``evaluate.py``).

    Parameters
    ----------
    model:
        The occupancy-completion network.
    loader:
        DataLoader yielding ``{"input", "target", "index"}`` batches.
    optimizer:
        Torch optimiser (ignored when ``train=False``).
    criterion:
        Loss module, e.g. :class:`nyuv2_scc.losses.BCEDiceLoss`.
    device:
        Device the batches and model live on.
    threshold:
        Probability threshold used to binarise predictions for metrics.
    train:
        Whether to update weights (see above).

    Returns
    -------
    dict
        Mean ``iou``, ``precision``, ``recall``, ``f1`` and ``loss`` over the
        whole epoch.
    """
    model.train(train)
    losses = []
    metric_items = []
    iterator = tqdm(loader, desc="train" if train else "eval", leave=False)
    for batch in iterator:
        x = batch["input"].to(device)      # incomplete input occupancy  (B,1,D,H,W)
        y = batch["target"].to(device)     # proxy target occupancy       (B,1,D,H,W)
        if train:
            optimizer.zero_grad(set_to_none=True)
        # ``set_grad_enabled`` builds the autograd graph only when training,
        # which makes the evaluation pass both faster and memory-light.
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
