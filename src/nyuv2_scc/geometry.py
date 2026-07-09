"""Geometry core: depth maps <-> 3D point clouds <-> voxel occupancy grids.

This module holds the pure-geometry building blocks the dataset relies on:

* :func:`depth_to_pointcloud` -- back-project a 2D depth map into a 3D point
  cloud using the pinhole camera model and the NYUv2 intrinsics.
* :func:`points_to_occupancy` -- bin those points into a binary voxel grid
  (with optional morphological dilation to close pinholes between points).
* :func:`make_incomplete_input` -- degrade an input volume (dropout + cuboid
  cut-outs) to simulate the missing observations a completion model must fill.
* :func:`occupancy_to_points` -- the inverse of voxelization, used only to turn
  a grid back into points for visualization / PLY export.

Grid convention throughout the project: axis order is ``[D, H, W] == [z, y, x]``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics (focal lengths ``fx, fy`` and principal point ``cx, cy``)."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class VoxelSpec:
    """Voxel grid definition: resolution and the metric bounds it covers.

    ``grid_size`` is ``(D, H, W) == (z, y, x)`` and ``bounds`` maps each axis
    name ``"x"/"y"/"z"`` to its ``(min, max)`` extent in metres.
    """

    grid_size: Tuple[int, int, int]  # D, H, W == z, y, x
    bounds: Dict[str, Tuple[float, float]]


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_min_m: float = 0.4,
    depth_max_m: float = 8.0,
    pixel_stride: int = 1,
) -> np.ndarray:
    """Project a real depth map into a 3D point cloud.

    Returns points in camera coordinates with columns [x, y, z].
    Invalid, zero, NaN, or out-of-range depth values are removed.
    """
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got shape {depth.shape}")
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be >= 1")

    depth = depth.astype(np.float32, copy=False)
    depth = depth[::pixel_stride, ::pixel_stride]
    h, w = depth.shape

    vv, uu = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    # Coordinates still refer to the original image lattice after striding.
    u = uu.astype(np.float32) * pixel_stride
    v = vv.astype(np.float32) * pixel_stride
    z = depth

    valid = np.isfinite(z) & (z > depth_min_m) & (z < depth_max_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    z_valid = z[valid]
    x = (u[valid] - intrinsics.cx) * z_valid / intrinsics.fx
    y = (v[valid] - intrinsics.cy) * z_valid / intrinsics.fy
    return np.stack([x, y, z_valid], axis=1).astype(np.float32)


def points_to_occupancy(
    points: np.ndarray,
    spec: VoxelSpec,
    dilate_iterations: int = 0,
) -> np.ndarray:
    """Voxelize a point cloud into a binary occupancy volume.

    Grid order is [D, H, W] = [z, y, x].
    """
    d, h, w = spec.grid_size
    volume = np.zeros((d, h, w), dtype=np.bool_)
    if points.size == 0:
        return volume.astype(np.float32)

    bx = spec.bounds["x"]
    by = spec.bounds["y"]
    bz = spec.bounds["z"]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    in_bounds = (
        (x >= bx[0]) & (x <= bx[1]) &
        (y >= by[0]) & (y <= by[1]) &
        (z >= bz[0]) & (z <= bz[1])
    )
    if not np.any(in_bounds):
        return volume.astype(np.float32)

    x = x[in_bounds]
    y = y[in_bounds]
    z = z[in_bounds]

    ix = np.floor((x - bx[0]) / (bx[1] - bx[0]) * w).astype(np.int64)
    iy = np.floor((y - by[0]) / (by[1] - by[0]) * h).astype(np.int64)
    iz = np.floor((z - bz[0]) / (bz[1] - bz[0]) * d).astype(np.int64)
    ix = np.clip(ix, 0, w - 1)
    iy = np.clip(iy, 0, h - 1)
    iz = np.clip(iz, 0, d - 1)

    volume[iz, iy, ix] = True
    if dilate_iterations > 0:
        structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
        volume = ndimage.binary_dilation(volume, structure=structure, iterations=dilate_iterations)
    return volume.astype(np.float32)


def make_incomplete_input(
    occupancy: np.ndarray,
    rng: np.random.Generator,
    voxel_dropout: float = 0.0,
    cuboid_masks: int = 0,
    cuboid_min_size: int = 4,
    cuboid_max_size: int = 12,
) -> np.ndarray:
    """Remove observed occupied voxels from an input volume.

    This is used only as an input degradation / augmentation step. It does not
    create target data; target occupancy still comes from real NYUv2 depth.
    """
    out = occupancy.copy()
    if voxel_dropout > 0:
        occupied = out > 0.5
        drop = rng.random(out.shape) < float(voxel_dropout)
        out[occupied & drop] = 0.0

    if cuboid_masks > 0:
        d, h, w = out.shape
        for _ in range(cuboid_masks):
            size_d = int(rng.integers(cuboid_min_size, cuboid_max_size + 1))
            size_h = int(rng.integers(cuboid_min_size, cuboid_max_size + 1))
            size_w = int(rng.integers(cuboid_min_size, cuboid_max_size + 1))
            z0 = int(rng.integers(0, max(1, d - size_d + 1)))
            y0 = int(rng.integers(0, max(1, h - size_h + 1)))
            x0 = int(rng.integers(0, max(1, w - size_w + 1)))
            out[z0:z0 + size_d, y0:y0 + size_h, x0:x0 + size_w] = 0.0
    return out.astype(np.float32)


def occupancy_to_points(volume: np.ndarray, spec: VoxelSpec, max_points: int | None = None) -> np.ndarray:
    """Convert occupied voxel centers back to approximate 3D points for visualization."""
    idx = np.argwhere(volume > 0.5)
    if max_points is not None and idx.shape[0] > max_points:
        step = int(np.ceil(idx.shape[0] / max_points))
        idx = idx[::step]
    if idx.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    d, h, w = volume.shape
    bx, by, bz = spec.bounds["x"], spec.bounds["y"], spec.bounds["z"]
    z_idx, y_idx, x_idx = idx[:, 0], idx[:, 1], idx[:, 2]
    x = bx[0] + (x_idx + 0.5) / w * (bx[1] - bx[0])
    y = by[0] + (y_idx + 0.5) / h * (by[1] - by[0])
    z = bz[0] + (z_idx + 0.5) / d * (bz[1] - bz[0])
    return np.stack([x, y, z], axis=1).astype(np.float32)
