"""Reader for the official NYUv2 labeled ``.mat`` file.

The official ``nyu_depth_v2_labeled.mat`` is an HDF5/MATLAB v7.3 file. When read
through ``h5py`` the array axes come out transposed relative to their logical
MATLAB shape, and the sample (N) axis can be first or last depending on the
field. :class:`NYUv2MatFile` hides all of that so callers always get depth maps
and label maps in a consistent ``[H, W]`` layout and RGB in ``[H, W, C]``.

Only real NYUv2 data is read here -- the project never fabricates dummy depth.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np


class NYUv2MatFile:
    """Thin reader for the official NYUv2 labeled .mat file.

    The official file is distributed as a MATLAB/HDF5 .mat file. h5py exposes
    arrays with MATLAB dimensions often reversed, so this class standardizes
    depths to [H, W].
    """

    def __init__(self, mat_path: str | Path):
        """Open the NYUv2 ``.mat`` file for reading.

        Prefer using this class as a context manager (``with NYUv2MatFile(...)``)
        so the underlying HDF5 handle is always closed.

        Raises
        ------
        FileNotFoundError
            If the real dataset file is missing (with a hint on where to get it).
        """
        self.mat_path = Path(mat_path)
        if not self.mat_path.exists():
            raise FileNotFoundError(
                f"NYUv2 .mat file not found: {self.mat_path}\n"
                "Download the real nyu_depth_v2_labeled.mat file from the official NYUv2 page "
                "and place it at the configured path. This project does not create dummy data."
            )
        self.file = h5py.File(self.mat_path, "r")

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @property
    def keys(self):
        return list(self.file.keys())

    def has_key(self, key: str) -> bool:
        return key in self.file

    def num_samples(self, key: str = "depths") -> int:
        """Number of scenes stored under ``key``.

        The sample axis may be first (h5py's usual ``[N, W, H]``) or last
        (logical ``[H, W, N]``); we disambiguate with the heuristic that the
        1449-scene axis is the large one (> 100).
        """
        if key not in self.file:
            raise KeyError(f"Key '{key}' not found. Available keys: {self.keys}")
        shape = self.file[key].shape
        # Common h5py shape for NYUv2 depths: [N, W, H]. Official logical shape is [H, W, N].
        if len(shape) == 3:
            return int(shape[0] if shape[0] > 100 else shape[-1])
        if len(shape) == 4:
            return int(shape[0])
        raise ValueError(f"Cannot infer sample count from shape {shape} for key {key}")

    def read_depth(self, key: str, index: int) -> np.ndarray:
        """Read scene ``index`` from depth field ``key`` as a ``[H, W]`` float32 map.

        ``key`` is typically ``"rawDepths"`` (input source) or ``"depths"``
        (in-painted proxy target). The array is transposed as needed so the
        returned map is always ``[H, W]`` in metres.
        """
        if key not in self.file:
            raise KeyError(f"Key '{key}' not found. Available keys: {self.keys}")
        arr = self.file[key]
        if arr.ndim != 3:
            raise ValueError(f"Expected a 3D depth dataset for key '{key}', got {arr.shape}")

        shape = arr.shape
        if shape[0] > 100:
            # Typical h5py layout: [N, W, H]. Transpose to [H, W].
            depth = np.asarray(arr[index], dtype=np.float32).T
        else:
            # Logical MATLAB-like layout: [H, W, N].
            depth = np.asarray(arr[:, :, index], dtype=np.float32)
        return depth

    def read_rgb(self, index: int) -> Optional[np.ndarray]:
        """Read scene ``index``'s RGB image as ``[H, W, 3]`` uint8, or ``None`` if absent.

        RGB is used only for qualitative report figures, never for training.
        """
        if "images" not in self.file:
            return None
        arr = self.file["images"]
        if arr.ndim != 4:
            return None
        if arr.shape[0] > 100:
            # Typical h5py layout: [N, C, W, H].
            img = np.asarray(arr[index], dtype=np.uint8).transpose(2, 1, 0)
        else:
            # Possible logical layout: [H, W, C, N].
            img = np.asarray(arr[:, :, :, index], dtype=np.uint8)
        return img


    def read_labels(self, index: int) -> np.ndarray:
        """Read a NYUv2 dense 2D semantic label map as [H, W].

        These labels are visible-region 2D annotations, not complete 3D
        semantic ground truth. They are used only for weak semantic-prior
        analysis / label-assisted filtering in the coursework ablations.
        """
        if "labels" not in self.file:
            raise KeyError(f"Key 'labels' not found. Available keys: {self.keys}")
        arr = self.file["labels"]
        if arr.ndim != 3:
            raise ValueError(f"Expected a 3D label dataset, got {arr.shape}")
        shape = arr.shape
        if shape[0] > 100:
            labels = np.asarray(arr[index], dtype=np.int32).T
        else:
            labels = np.asarray(arr[:, :, index], dtype=np.int32)
        return labels
