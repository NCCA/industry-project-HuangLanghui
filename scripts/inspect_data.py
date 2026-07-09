#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyuv2_scc.config import load_config, resolve_path
from nyuv2_scc.nyuv2_io import NYUv2MatFile
from nyuv2_scc.visualization import save_depth_preview


def main():
    parser = argparse.ArgumentParser(description="Inspect a real NYUv2 labeled .mat file.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mat_path = resolve_path(cfg["data"]["mat_path"], PROJECT_ROOT)
    out_path = PROJECT_ROOT / "outputs" / "visualizations" / f"depth_preview_{args.index:04d}.png"

    with NYUv2MatFile(mat_path) as reader:
        print("Available keys:", reader.keys)
        print("Number of samples:", reader.num_samples(cfg["data"]["target_key"]))
        raw = reader.read_depth(cfg["data"]["input_key"], args.index)
        depth = reader.read_depth(cfg["data"]["target_key"], args.index)
        rgb = reader.read_rgb(args.index)
        print("rawDepths shape/min/max:", raw.shape, float(raw[raw > 0].min()), float(raw.max()))
        print("depths shape/min/max:", depth.shape, float(depth[depth > 0].min()), float(depth.max()))
        save_depth_preview(rgb, raw, depth, out_path)
        print(f"Saved preview to {out_path}")


if __name__ == "__main__":
    main()
