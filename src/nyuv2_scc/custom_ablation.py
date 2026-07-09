"""Building blocks for the custom ablation studies.

Helpers shared by the ``scripts/*_ablation.py`` experiments: confusion-count
metric accumulation, and post-processing / input-perturbation operators
(isolated-voxel removal, median filtering, partial-observation generation, ...)
whose effect on completion quality the ablations measure.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from scipy import ndimage

from .geometry import make_incomplete_input


def counts_to_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Convert accumulated TP/FP/FN counts into IoU / precision / recall / F1."""
    eps = 1e-8
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def update_counts(counts: Dict[str, int], pred: torch.Tensor, target: torch.Tensor) -> None:
    """Accumulate TP/FP/FN counts for one batch into the running ``counts`` dict (in place)."""
    pred = pred.bool()
    target = target.bool()
    counts["tp"] += int(torch.logical_and(pred, target).sum().item())
    counts["fp"] += int(torch.logical_and(pred, ~target).sum().item())
    counts["fn"] += int(torch.logical_and(~pred, target).sum().item())


def remove_isolated_voxels(volume: np.ndarray, min_neighbors: int = 2) -> np.ndarray:
    """Remove occupied voxels that have too few occupied 26-neighborhood voxels."""
    occ = volume > 0.5
    kernel = np.ones((3, 3, 3), dtype=np.int16)
    kernel[1, 1, 1] = 0
    neighbor_count = ndimage.convolve(occ.astype(np.int16), kernel, mode="constant", cval=0)
    cleaned = occ & (neighbor_count >= int(min_neighbors))
    return cleaned.astype(np.float32)


def median_filter_occupancy(volume: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply a small 3D median filter to a binary occupancy volume."""
    filtered = ndimage.median_filter((volume > 0.5).astype(np.float32), size=int(size), mode="nearest")
    return (filtered > 0.5).astype(np.float32)


def closing_filter_occupancy(volume: np.ndarray) -> np.ndarray:
    """Light binary closing to connect small gaps after voxelization."""
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    closed = ndimage.binary_closing(volume > 0.5, structure=structure, iterations=1)
    return closed.astype(np.float32)


def apply_noise_filter(volume: np.ndarray, variant: str) -> np.ndarray:
    """Apply an input occupancy cleanup variant for the noise filtering ablation."""
    variant = variant.lower().strip()
    if variant in {"none", "baseline", "no_filter"}:
        return volume.astype(np.float32, copy=True)
    if variant in {"isolated", "remove_isolated"}:
        return remove_isolated_voxels(volume, min_neighbors=2)
    if variant in {"median", "median3d"}:
        return median_filter_occupancy(volume, size=3)
    if variant in {"closing", "binary_closing"}:
        return closing_filter_occupancy(volume)
    if variant in {"median_isolated", "median+isolated", "median_plus_isolated"}:
        return remove_isolated_voxels(median_filter_occupancy(volume, size=3), min_neighbors=2)
    raise ValueError(f"Unknown noise filter variant: {variant}")


def parse_noise_variants(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_partial_variants(text: str) -> List[Dict]:
    """Parse name:dropout:cuboids:min_size:max_size entries."""
    variants: List[Dict] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) not in {3, 5}:
            raise ValueError(
                "Partial variants must use name:dropout:cuboids or name:dropout:cuboids:min_size:max_size"
            )
        name, dropout, cuboids = parts[:3]
        row = {"name": name, "dropout": float(dropout), "cuboids": int(cuboids)}
        if len(parts) == 5:
            row["cuboid_min_size"] = int(parts[3])
            row["cuboid_max_size"] = int(parts[4])
        variants.append(row)
    return variants


def apply_partial_variant(
    volume: np.ndarray,
    variant: Dict,
    seed: int,
    fallback_min_size: int = 5,
    fallback_max_size: int = 14,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return make_incomplete_input(
        volume,
        rng,
        voxel_dropout=float(variant.get("dropout", 0.0)),
        cuboid_masks=int(variant.get("cuboids", 0)),
        cuboid_min_size=int(variant.get("cuboid_min_size", fallback_min_size)),
        cuboid_max_size=int(variant.get("cuboid_max_size", fallback_max_size)),
    )


def parse_point_variants(text: str) -> List[Dict]:
    """Parse point-count variants such as '1k:1000,5k:5000,all:all'."""
    variants: List[Dict] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, value = item.split(":", 1)
        else:
            name, value = item, item
        value = value.strip().lower()
        max_points = None if value in {"all", "none", "full"} else int(float(value.replace("k", "000")))
        variants.append({"name": name.strip(), "max_points": max_points})
    return variants


def parse_grid_variants(text: str) -> List[Dict]:
    """Parse voxel grid variants such as '32,48,64' or '32x32x32,64x64x64'."""
    variants: List[Dict] = []
    for item in text.split(","):
        raw = item.strip().lower()
        if not raw:
            continue
        if "x" in raw:
            dims = tuple(int(v) for v in raw.split("x"))
            if len(dims) != 3:
                raise ValueError("Grid variant must be a single integer or D x H x W, e.g. 64 or 64x64x64")
        else:
            g = int(raw)
            dims = (g, g, g)
        variants.append({"name": "x".join(map(str, dims)), "grid_size": dims})
    return variants


def subsample_points(points: np.ndarray, max_points: int | None, seed: int) -> np.ndarray:
    """Deterministically subsample a point cloud to max_points for resolution ablation."""
    if max_points is None or points.shape[0] <= int(max_points):
        return points.astype(np.float32, copy=False)
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(points.shape[0], size=int(max_points), replace=False)
    return points[chosen].astype(np.float32, copy=False)


def parse_label_variants(text: str) -> List[Dict]:
    """Parse semantic-prior variants.

    Supported built-ins:
      geometry_only, valid_labels, top5, top10, boundary_clean, large_components
    Optional forms:
      topK, large_components:minimum_pixel_count
    """
    out: List[Dict] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name = item
        param = None
        if ":" in item:
            name, raw_param = item.split(":", 1)
            param = int(raw_param)
        key = name.strip().lower()
        if key.startswith("top") and key[3:].isdigit():
            out.append({"name": key, "mode": "topk", "k": int(key[3:])})
        elif key in {"geometry_only", "none", "baseline"}:
            out.append({"name": "geometry_only", "mode": "none"})
        elif key in {"valid", "valid_labels", "label_valid"}:
            out.append({"name": "valid_labels", "mode": "valid"})
        elif key in {"boundary", "boundary_clean", "interior"}:
            out.append({"name": "boundary_clean", "mode": "boundary_clean"})
        elif key in {"large", "large_components", "connected"}:
            out.append({"name": f"large_components_{param or 250}", "mode": "large_components", "min_pixels": param or 250})
        else:
            raise ValueError(f"Unknown label-assisted variant: {item}")
    return out


def _topk_label_mask(labels: np.ndarray, k: int) -> np.ndarray:
    labels = labels.astype(np.int32, copy=False)
    valid = labels > 0
    if not np.any(valid):
        return valid
    values, counts = np.unique(labels[valid], return_counts=True)
    keep_values = values[np.argsort(counts)[::-1][:int(k)]]
    return np.isin(labels, keep_values)


def _boundary_clean_mask(labels: np.ndarray) -> np.ndarray:
    """Keep semantic interior pixels whose 3x3 neighborhood has one label.

    This is a weak 2D semantic prior that removes jagged/noisy class boundaries
    before back-projecting depth into 3D. It is not a 3D semantic ground truth.
    """
    labels = labels.astype(np.int32, copy=False)
    valid = labels > 0
    local_min = ndimage.minimum_filter(labels, size=3, mode="nearest")
    local_max = ndimage.maximum_filter(labels, size=3, mode="nearest")
    return valid & (local_min == local_max)


def _large_components_mask(labels: np.ndarray, min_pixels: int = 250) -> np.ndarray:
    """Keep only larger connected visible semantic regions to reduce clutter."""
    labels = labels.astype(np.int32, copy=False)
    out = np.zeros(labels.shape, dtype=bool)
    for value in np.unique(labels):
        if value <= 0:
            continue
        labeled, num = ndimage.label(labels == value)
        if num == 0:
            continue
        counts = np.bincount(labeled.ravel())
        keep = np.where(counts >= int(min_pixels))[0]
        keep = keep[keep != 0]
        if keep.size:
            out |= np.isin(labeled, keep)
    return out


def make_label_mask(labels: np.ndarray, variant: Dict) -> np.ndarray:
    """Create a 2D visible-region semantic prior mask from NYUv2 labels."""
    mode = variant.get("mode", "none")
    if mode == "none":
        return np.ones(labels.shape, dtype=bool)
    if mode == "valid":
        return labels > 0
    if mode == "topk":
        return _topk_label_mask(labels, int(variant.get("k", 5)))
    if mode == "boundary_clean":
        return _boundary_clean_mask(labels)
    if mode == "large_components":
        return _large_components_mask(labels, int(variant.get("min_pixels", 250)))
    raise ValueError(f"Unsupported label mask mode: {mode}")


def apply_label_mask_to_depth(depth: np.ndarray, labels: np.ndarray, variant: Dict) -> np.ndarray:
    """Apply a visible 2D semantic-prior mask to a depth map.

    The labels are used only to filter the input observation. The target/proxy
    occupancy remains geometry-derived from NYUv2 depths.
    """
    if labels.shape != depth.shape:
        # Try the common transposed layout before failing.
        if labels.T.shape == depth.shape:
            labels = labels.T
        else:
            raise ValueError(f"Label shape {labels.shape} does not match depth shape {depth.shape}")
    mask = make_label_mask(labels, variant)
    out = depth.astype(np.float32, copy=True)
    out[~mask] = np.nan
    return out
