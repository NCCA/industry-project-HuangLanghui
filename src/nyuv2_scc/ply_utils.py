"""ASCII PLY point-cloud export.

Occupancy grids are turned back into 3D points (see
:func:`nyuv2_scc.geometry.occupancy_to_points`) and written here as ``.ply``
files that open in MeshLab / CloudCompare / Blender, so the completed scenes can
be inspected as real 3D geometry rather than only as rendered figures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def write_ascii_ply(
    points: np.ndarray,
    path: str | Path,
    color: Tuple[int, int, int] | None = None,
    colors: np.ndarray | None = None,
) -> Path:
    """Write a point cloud to a simple ASCII PLY file.

    Parameters
    ----------
    points:
        Nx3 array in metric camera coordinates.
    path:
        Output PLY path.
    color:
        Optional single RGB color used for all points.
    colors:
        Optional Nx3 uint8 RGB colors. Takes precedence over ``color``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)

    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if colors.shape[0] != points.shape[0]:
            raise ValueError("colors must have the same number of rows as points")
    elif color is not None:
        colors = np.tile(np.asarray(color, dtype=np.uint8), (points.shape[0], 1))
    else:
        colors = np.tile(np.asarray((70, 130, 180), dtype=np.uint8), (points.shape[0], 1))

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment NYUv2 proxy 3D occupancy completion export\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    return path
