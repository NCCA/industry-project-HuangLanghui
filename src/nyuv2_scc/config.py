"""Configuration and path/device helpers.

The whole project is driven by YAML config files under ``configs/`` (see
``configs/default.yaml``). These small helpers load a config, turn config-relative
paths into absolute paths, and resolve the compute device.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML experiment config into a nested dictionary.

    Parameters
    ----------
    path:
        Path to a ``.yaml`` config file (e.g. ``configs/default.yaml``).

    Returns
    -------
    dict
        The parsed config with top-level ``data``, ``model``, ``training`` and
        ``outputs`` sections.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    """Resolve a possibly-relative config path against the project root.

    Absolute paths are returned unchanged. Relative paths (the common case in
    the YAML configs, e.g. ``data/nyu_depth_v2_labeled.mat``) are joined onto
    ``project_root`` so scripts work regardless of the current directory.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    if project_root is None:
        project_root = Path.cwd()
    return Path(project_root) / p


def get_device(device: str):
    """Return a ``torch.device`` from a config string.

    ``"auto"`` picks CUDA when a GPU is available and falls back to CPU;
    any other value (``"cpu"``, ``"cuda"``, ``"cuda:1"`` ...) is passed through
    to ``torch.device`` verbatim.
    """
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
