"""Lightweight visualization entry points.

Thin, stable wrappers kept for backward compatibility. The report-quality
plotting actually lives in :mod:`nyuv2_scc.advanced_visualization`; this module
just re-exposes the two figures used most often: the input/prediction/target
triplet and the RGB + rawDepths + depths preview.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .advanced_visualization import save_occupancy_triplet_advanced
from .geometry import VoxelSpec


def save_occupancy_triplet(
    input_occ: np.ndarray,
    pred_occ: np.ndarray,
    target_occ: np.ndarray,
    spec: VoxelSpec,
    out_path: str | Path,
    title: str = "NYUv2 proxy scene completion",
    max_points: int = 7000,
) -> None:
    """Backward-compatible wrapper using the upgraded report-ready visualization."""
    save_occupancy_triplet_advanced(input_occ, pred_occ, target_occ, spec, out_path, title=title, max_points=max_points)


def save_depth_preview(rgb: np.ndarray | None, raw_depth: np.ndarray, depth: np.ndarray, out_path: str | Path) -> None:
    """Save a polished preview of RGB, rawDepths and in-painted depths."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = 3 if rgb is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(5.5 * cols, 4.4))
    if cols == 1:
        axes = [axes]
    p = 0
    if rgb is not None:
        axes[p].imshow(rgb)
        axes[p].set_title("RGB image", fontweight="bold")
        axes[p].axis("off")
        p += 1
    im0 = axes[p].imshow(raw_depth, cmap="viridis")
    axes[p].set_title("rawDepths input source", fontweight="bold")
    axes[p].axis("off")
    fig.colorbar(im0, ax=axes[p], fraction=0.046, label="meters")
    p += 1
    im1 = axes[p].imshow(depth, cmap="viridis")
    axes[p].set_title("depths target/proxy source", fontweight="bold")
    axes[p].axis("off")
    fig.colorbar(im1, ax=axes[p], fraction=0.046, label="meters")
    fig.suptitle("Real NYUv2 RGB-D Input and Proxy Target Construction", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
