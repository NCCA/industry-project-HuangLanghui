from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch

from .config import get_device, load_config, resolve_path
from .dataset import NYUv2OccupancyDataset, build_specs, prepare_splits_from_config
from .model import ResidualAttentionUNet3D


def project_root_from_script(script_file: str | Path) -> Path:
    return Path(script_file).resolve().parents[1]


def load_project_config(config_path: str | Path, project_root: str | Path) -> Tuple[Dict, Dict, Path]:
    project_root = Path(project_root)
    cfg = load_config(config_path)
    data_cfg = dict(cfg["data"])
    data_cfg["cache_dir"] = str(resolve_path(data_cfg["cache_dir"], project_root))
    mat_path = resolve_path(data_cfg["mat_path"], project_root)
    return cfg, data_cfg, mat_path


def build_dataset(config_path: str | Path, project_root: str | Path, split: str = "test"):
    cfg, data_cfg, mat_path = load_project_config(config_path, project_root)
    splits = prepare_splits_from_config(mat_path, data_cfg)
    dataset = NYUv2OccupancyDataset(mat_path, splits[split], data_cfg, split=split)
    _, spec = build_specs(data_cfg)
    return cfg, data_cfg, dataset, spec


def build_model_from_config(cfg: Dict, device: torch.device):
    model_cfg = cfg["model"]
    return ResidualAttentionUNet3D(
        in_channels=1,
        base_channels=int(model_cfg.get("base_channels", 8)),
        use_attention=bool(model_cfg.get("use_attention", True)),
    ).to(device)


def load_trained_model(config_path: str | Path, checkpoint_path: str | Path, project_root: str | Path):
    cfg = load_config(config_path)
    device = get_device(cfg["training"].get("device", "auto"))
    model = build_model_from_config(cfg, device)
    ckpt_path = resolve_path(checkpoint_path, Path(project_root))
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model, device, cfg
