"""Dataset construction: real NYUv2 depth maps -> occupancy volume pairs.

This module turns the official NYUv2 ``.mat`` file into ``(input, target)``
occupancy pairs the network trains on. For each scene:

* the **input** occupancy is built from ``rawDepths`` (the raw Kinect depth,
  which has holes/missing regions) -> an *incomplete* volume;
* the **target/proxy** occupancy is built from ``depths`` (NYUv2's in-painted
  depth) -> a *more complete* volume the model learns to predict.

The heavy lifting (depth -> point cloud -> voxel grid) lives in
:mod:`nyuv2_scc.geometry`; this module wires it together, caches the result to
``.npz`` so it is computed only once, applies training-time input degradation,
and produces reproducible train/val/test splits.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import (
    CameraIntrinsics,
    VoxelSpec,
    depth_to_pointcloud,
    make_incomplete_input,
    points_to_occupancy,
)
from .nyuv2_io import NYUv2MatFile


def build_specs(data_cfg: Dict) -> Tuple[CameraIntrinsics, VoxelSpec]:
    """Build the camera-intrinsics and voxel-grid specs from the ``data`` config.

    Returns ``(CameraIntrinsics, VoxelSpec)`` -- the intrinsics used to back-project
    depth into 3D, and the grid size + metric bounds used to voxelize it.
    """
    intr = data_cfg["intrinsics"]
    camera = CameraIntrinsics(
        fx=float(intr["fx"]),
        fy=float(intr["fy"]),
        cx=float(intr["cx"]),
        cy=float(intr["cy"]),
    )
    bounds = {
        "x": tuple(map(float, data_cfg["bounds"]["x"])),
        "y": tuple(map(float, data_cfg["bounds"]["y"])),
        "z": tuple(map(float, data_cfg["bounds"]["z"])),
    }
    spec = VoxelSpec(grid_size=tuple(map(int, data_cfg["grid_size"])), bounds=bounds)
    return camera, spec


def cache_filename(nyu_index: int, input_key: str, target_key: str, grid_size: Sequence[int]) -> str:
    """Name of the ``.npz`` cache entry for one scene.

    Encoding the source keys and the grid size means different resolutions
    (e.g. the 32/48/64 voxel ablations) never collide in the cache.
    """
    grid_tag = "x".join(map(str, grid_size))
    return f"nyuv2_{nyu_index:04d}_{input_key}_to_{target_key}_{grid_tag}.npz"


def cached_indices(cache_dir: Path, input_key: str, target_key: str, grid_size: Sequence[int]) -> List[int]:
    """Scene indices that already have a cache entry, sorted ascending."""
    if not cache_dir.is_dir():
        return []
    grid_tag = "x".join(map(str, grid_size))
    suffix = f"_{input_key}_to_{target_key}_{grid_tag}.npz"
    found: List[int] = []
    for path in cache_dir.glob(f"nyuv2_*{suffix}"):
        stem = path.name[len("nyuv2_"):-len(suffix)]
        if stem.isdigit():
            found.append(int(stem))
    return sorted(found)


def num_samples_from_cache(data_cfg: Dict) -> int:
    """Infer the scene count from a complete voxel cache, without the ``.mat`` file.

    Only trusted when the cache covers a contiguous range ``0..n-1``: the splits
    are drawn from ``np.arange(num_samples)``, so a cache with holes would
    silently produce a *different* train/val/test partition than the one the
    checkpoint was trained on. A partial cache therefore raises rather than
    guessing.
    """
    cache_dir = Path(data_cfg.get("cache_dir", ""))
    _, spec = build_specs(data_cfg)
    input_key = data_cfg.get("input_key", "rawDepths")
    target_key = data_cfg.get("target_key", "depths")
    indices = cached_indices(cache_dir, input_key, target_key, spec.grid_size)

    hint = (
        "Either download the real nyu_depth_v2_labeled.mat to the configured mat_path, "
        "or restore a complete voxel cache (see data/README.md)."
    )
    if not indices:
        raise FileNotFoundError(
            f"No voxel cache found in {cache_dir} for "
            f"'{input_key}' -> '{target_key}' at {spec.grid_size}, and the .mat file is missing.\n{hint}"
        )
    if indices != list(range(len(indices))):
        missing = sorted(set(range(indices[-1] + 1)) - set(indices))
        raise FileNotFoundError(
            f"The voxel cache in {cache_dir} has gaps (missing scene indices: {missing[:10]}"
            f"{', ...' if len(missing) > 10 else ''}). Splits are index-based, so an incomplete cache "
            f"would not reproduce the recorded train/val/test partition.\n{hint}"
        )
    return len(indices)


def make_splits(num_samples: int, train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[int]]:
    """Deterministically shuffle sample indices into train/val/test lists.

    The ``seed`` fixes the shuffle so every run (training, evaluation,
    visualization) sees the exact same split -- essential for the test set to
    stay unseen. The test split gets whatever remains after train and val.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    return {
        "train": indices[:n_train].tolist(),
        "val": indices[n_train:n_train + n_val].tolist(),
        "test": indices[n_train + n_val:].tolist(),
    }


class NYUv2OccupancyDataset(Dataset):
    """Dataset that converts real NYUv2 depth maps to occupancy volumes on demand.

    Input occupancy is built from `rawDepths` by default. Proxy target occupancy
    is built from `depths` by default. Both are real NYUv2 depth sources.
    """

    def __init__(
        self,
        mat_path: str | Path,
        indices: Sequence[int],
        data_cfg: Dict,
        split: str = "train",
    ):
        self.mat_path = Path(mat_path)
        self.indices = list(map(int, indices))
        self.cfg = data_cfg
        self.split = split
        self.target_key = data_cfg.get("target_key", "depths")
        self.input_key = data_cfg.get("input_key", "rawDepths")
        self.camera, self.spec = build_specs(data_cfg)
        self.cache_dir = Path(data_cfg.get("cache_dir", ""))
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(data_cfg.get("seed", 42))

        # The .mat file is only opened for scenes that are not already cached. When every
        # requested scene has a cache entry the dataset is fully usable without it, so the
        # CPU quickstart runs from the shipped cache alone. Whenever the file *is* needed,
        # validate both depth keys up front so a bad config fails immediately.
        if any(not self._cache_file(i).exists() for i in self.indices):
            with NYUv2MatFile(self.mat_path) as reader:
                if not reader.has_key(self.target_key):
                    raise KeyError(f"Target key '{self.target_key}' not found in {self.mat_path}. Available keys: {reader.keys}")
                if not reader.has_key(self.input_key):
                    raise KeyError(
                        f"Input key '{self.input_key}' not found in {self.mat_path}. "
                        "For this coursework prototype, use a real NYUv2 input source such as rawDepths."
                    )

    def __len__(self) -> int:
        return len(self.indices)

    def _cache_file(self, nyu_index: int) -> Path:
        """Cache path for one scene, keyed by scene index, source keys and grid size."""
        return self.cache_dir / cache_filename(nyu_index, self.input_key, self.target_key, self.spec.grid_size)

    def _depth_to_occ(self, depth: np.ndarray) -> np.ndarray:
        """Convert one depth map to an occupancy volume (project -> voxelize -> dilate)."""
        points = depth_to_pointcloud(
            depth,
            self.camera,
            depth_min_m=float(self.cfg.get("depth_min_m", 0.4)),
            depth_max_m=float(self.cfg.get("depth_max_m", 8.0)),
            pixel_stride=int(self.cfg.get("pixel_stride", 2)),
        )
        return points_to_occupancy(
            points,
            self.spec,
            dilate_iterations=int(self.cfg.get("dilate_iterations", 1)),
        )

    def _load_or_create_pair(self, nyu_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the ``(input_occ, target_occ)`` pair for one scene, using the cache.

        On a cache miss the depth maps are read from the ``.mat`` file, converted
        to occupancy, reconciled (see the inline note below), and saved so the
        expensive projection/voxelization runs only once per scene.
        """
        cache_path = self._cache_file(nyu_index)
        if cache_path.exists():
            cached = np.load(cache_path)
            return cached["input_occ"].astype(np.float32), cached["target_occ"].astype(np.float32)

        with NYUv2MatFile(self.mat_path) as reader:
            input_depth = reader.read_depth(self.input_key, nyu_index)
            target_depth = reader.read_depth(self.target_key, nyu_index)

        input_occ = self._depth_to_occ(input_depth)
        target_occ = self._depth_to_occ(target_depth)

        # Target is proxy-supervision from real depth. Input should never exceed target if raw/inpainted differences
        # cause small projection artifacts, so we keep only input occupied voxels that are inside the target envelope
        # after dilation. This stabilizes training and metric interpretation.
        input_occ = np.minimum(input_occ, np.maximum(input_occ, target_occ))
        np.savez_compressed(cache_path, input_occ=input_occ, target_occ=target_occ)
        return input_occ.astype(np.float32), target_occ.astype(np.float32)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        """Return one training example as ``{"input", "target", "index"}`` tensors.

        For the ``train`` split the input is additionally degraded on-the-fly
        (random voxel dropout + cuboid cut-outs) via
        :func:`nyuv2_scc.geometry.make_incomplete_input`, so the network sees a
        different, harder incomplete observation each epoch. The seed is derived
        from the scene index for reproducibility. Val/test inputs are left as-is.
        """
        nyu_index = self.indices[item]
        input_occ, target_occ = self._load_or_create_pair(nyu_index)

        if self.split == "train":
            rng = np.random.default_rng(self.seed + nyu_index)
            input_occ = make_incomplete_input(
                input_occ,
                rng,
                voxel_dropout=float(self.cfg.get("input_voxel_dropout", 0.0)),
                cuboid_masks=int(self.cfg.get("cuboid_masks", 0)),
                cuboid_min_size=int(self.cfg.get("cuboid_min_size", 4)),
                cuboid_max_size=int(self.cfg.get("cuboid_max_size", 12)),
            )

        return {
            "input": torch.from_numpy(input_occ[None, ...]).float(),
            "target": torch.from_numpy(target_occ[None, ...]).float(),
            "index": torch.tensor(nyu_index, dtype=torch.long),
        }


def prepare_splits_from_config(mat_path: str | Path, data_cfg: Dict) -> Dict[str, List[int]]:
    """Read the scene count and build the splits.

    The count comes from the ``.mat`` file when it is present. When it is not,
    it is recovered from a complete voxel cache (see :func:`num_samples_from_cache`)
    so evaluation and the demo reproduce from the shipped cache alone. Both paths
    feed the same deterministic :func:`make_splits`, so the partition is identical
    either way.

    Honours the optional ``max_samples`` cap (used to run on a subset while
    developing) before delegating to :func:`make_splits`.
    """
    if Path(mat_path).exists():
        with NYUv2MatFile(mat_path) as reader:
            n = reader.num_samples(data_cfg.get("target_key", "depths"))
    else:
        n = num_samples_from_cache(data_cfg)
    max_samples = data_cfg.get("max_samples")
    if max_samples is not None:
        n = min(n, int(max_samples))
    return make_splits(
        n,
        train_ratio=float(data_cfg.get("train_ratio", 0.8)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=int(data_cfg.get("seed", 42)),
    )


def save_splits(splits: Dict[str, List[int]], path: str | Path) -> None:
    """Write the split index lists to JSON for reproducibility / later inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
