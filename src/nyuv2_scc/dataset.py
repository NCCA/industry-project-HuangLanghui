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


def make_splits(num_samples: int, train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[int]]:
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
        grid_tag = "x".join(map(str, self.spec.grid_size))
        return self.cache_dir / f"nyuv2_{nyu_index:04d}_{self.input_key}_to_{self.target_key}_{grid_tag}.npz"

    def _depth_to_occ(self, depth: np.ndarray) -> np.ndarray:
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
    with NYUv2MatFile(mat_path) as reader:
        n = reader.num_samples(data_cfg.get("target_key", "depths"))
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
