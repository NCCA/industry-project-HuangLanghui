#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyuv2_scc.advanced_visualization import plot_experiment_comparison
from nyuv2_scc.config import load_config


def make_variant_config(base_cfg: dict, name: str, use_attention: bool, epochs: int | None) -> Path:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["model"]["use_attention"] = bool(use_attention)
    if epochs is not None:
        cfg["training"]["epochs"] = int(epochs)
    cfg["training"]["save_dir"] = f"outputs/experiments/{name}/checkpoints"
    cfg["outputs"]["metrics_dir"] = f"outputs/experiments/{name}/metrics"
    cfg["outputs"]["vis_dir"] = f"outputs/experiments/{name}/visualizations"
    out = PROJECT_ROOT / "configs" / "advanced" / f"{name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def read_metric(path: Path, name: str):
    with path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    row = {"name": name}
    for k in ["iou", "precision", "recall", "f1", "loss"]:
        if k in metrics:
            row[k] = float(metrics[k])
    return row


def main():
    parser = argparse.ArgumentParser(description="Run a small attention on/off ablation and plot the comparison.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=4, help="Short ablation by default. Use 12 for full training.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--skip-existing", action="store_true", help="Do not retrain a variant if best.pt already exists.")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    variants = [("attention_on", True), ("attention_off", False)]
    rows = []
    for name, use_attention in variants:
        cfg_path = make_variant_config(base_cfg, name, use_attention, args.epochs)
        ckpt = PROJECT_ROOT / "outputs" / "experiments" / name / "checkpoints" / "best.pt"
        if not (args.skip_existing and ckpt.exists()):
            run([sys.executable, "train.py", "--config", str(cfg_path.relative_to(PROJECT_ROOT))])
        run([sys.executable, "evaluate.py", "--config", str(cfg_path.relative_to(PROJECT_ROOT)), "--checkpoint", str(ckpt.relative_to(PROJECT_ROOT)), "--split", args.split])
        metrics_path = PROJECT_ROOT / "outputs" / "experiments" / name / "metrics" / f"{args.split}_metrics.json"
        rows.append(read_metric(metrics_path, name))

    out_json = PROJECT_ROOT / "outputs" / "experiments" / "attention_ablation_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"split": args.split, "epochs": args.epochs, "results": rows}, f, indent=2)
    out_fig = PROJECT_ROOT / "outputs" / "figures" / "attention_ablation.png"
    plot_experiment_comparison(out_json, out_fig, title="Attention Module Ablation")
    print(json.dumps(rows, indent=2))
    print(f"Saved {out_json}")
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()
