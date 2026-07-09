"""Training loss for binary 3D occupancy completion.

The network predicts, for every voxel in the grid, a single logit that says
"is this voxel occupied?". This is a dense binary classification problem with a
strong class imbalance: in a NYUv2 room only a small fraction of the 64x64x64
voxels are actually occupied, so a naive loss collapses to predicting "empty"
everywhere.

We therefore combine two complementary terms (see :class:`BCEDiceLoss`):

* **Weighted binary cross-entropy (BCE)** -- a per-voxel classification loss.
  The ``pos_weight`` factor multiplies the loss contribution of the rare
  *occupied* voxels so the optimiser cannot ignore them.
* **Soft Dice loss** -- an overlap loss operating on the whole volume at once.
  It directly optimises the intersection-over-union-like agreement between the
  predicted and target occupied regions, which is exactly the shape-completion
  quality we report as IoU / F1 at evaluation time.

The final objective is a weighted sum::

    loss = bce_weight * BCE(logits, target)
         + dice_weight * (1 - soft_dice(sigmoid(logits), target))

Default weights come from ``configs/default.yaml``
(``bce_weight=1.0``, ``dice_weight=0.5``, ``pos_weight=8.0``).
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """Weighted BCE + soft Dice loss for sparse binary occupancy grids.

    Parameters
    ----------
    bce_weight:
        Scalar multiplier on the binary cross-entropy term. Controls how much
        the loss cares about correct *per-voxel* labels.
    dice_weight:
        Scalar multiplier on the soft Dice term. Controls how much the loss
        cares about *region overlap* between prediction and target.
    pos_weight:
        Positive-class weight passed to BCE. Values > 1 up-weight the rare
        occupied voxels to counteract the empty/occupied class imbalance
        (``8.0`` by default, i.e. one occupied voxel counts as much as eight
        empty ones).

    Notes
    -----
    ``pos_weight`` is stored as a registered buffer so it is moved to the
    correct device together with the module (``.to(device)``) and is saved in
    the checkpoint, without being treated as a trainable parameter.
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 0.5, pos_weight: float = 1.0):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.register_buffer("pos_weight_tensor", torch.tensor([float(pos_weight)]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the combined loss.

        Parameters
        ----------
        logits:
            Raw network outputs of shape ``(B, 1, D, H, W)`` (no sigmoid
            applied). These are the model's un-normalised occupancy scores.
        targets:
            Binary occupancy targets of the same shape, with values in
            ``{0, 1}`` (built from the real NYUv2 ``depths`` proxy).

        Returns
        -------
        torch.Tensor
            A single scalar loss ready for ``.backward()``.
        """
        # --- Term 1: weighted binary cross-entropy (per-voxel classification) ---
        # ``binary_cross_entropy_with_logits`` fuses the sigmoid and the BCE in a
        # numerically stable way, so we pass raw logits (not probabilities).
        pos_weight = self.pos_weight_tensor.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

        # --- Term 2: soft Dice (region overlap over the whole volume) ---
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))                       # reduce over D,H,W (keep batch)
        intersection = torch.sum(probs * targets, dim=dims)      # soft overlap per sample
        denom = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
        # ``+ 1.0`` on numerator and denominator is Laplace smoothing: it keeps
        # the score defined (and gradient stable) for near-empty volumes.
        dice = 1.0 - torch.mean((2.0 * intersection + 1.0) / (denom + 1.0))

        return self.bce_weight * bce + self.dice_weight * dice
