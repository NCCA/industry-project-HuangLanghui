"""Report-quality figures for the occupancy-completion results.

Every figure the coursework report uses is produced here from the JSON metrics
and the occupancy volumes. Two families of plots:

* **Quantitative** -- training curves, metric bars, threshold sweeps and
  experiment/ablation comparisons, read straight from the ``outputs/metrics``
  JSON files.
* **Qualitative** -- 3D scatter triplets (input / prediction / target), multi-view
  projection galleries and TP/FP/FN error maps that *show* the completion, which
  is what the examiner asked for beyond the summary numbers.

All functions share a common publication matplotlib style
(:func:`set_publication_style`) and save straight to disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .geometry import VoxelSpec, occupancy_to_points


def set_publication_style() -> None:
    """Apply a clean report-friendly matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 240,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
    })


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric_value(row: Dict, split: str, name: str) -> float:
    return float(row.get(split, {}).get(name, np.nan))


def plot_training_curves(history_path: str | Path, out_path: str | Path) -> None:
    """Create a two-panel figure for loss and validation metrics."""
    set_publication_style()
    history = load_json(history_path)
    if not history:
        raise ValueError(f"No history entries found in {history_path}")

    epochs = [int(row.get("epoch", i)) + 1 for i, row in enumerate(history)]
    train_loss = [_metric_value(row, "train", "loss") for row in history]
    val_loss = [_metric_value(row, "val", "loss") for row in history]
    train_iou = [_metric_value(row, "train", "iou") for row in history]
    val_iou = [_metric_value(row, "val", "iou") for row in history]
    train_f1 = [_metric_value(row, "train", "f1") for row in history]
    val_f1 = [_metric_value(row, "val", "f1") for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(epochs, train_loss, marker="o", linewidth=2.2, label="Train loss")
    ax.plot(epochs, val_loss, marker="s", linewidth=2.2, label="Validation loss")
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=True)

    ax = axes[1]
    ax.plot(epochs, train_iou, marker="o", linewidth=2.2, label="Train IoU")
    ax.plot(epochs, val_iou, marker="s", linewidth=2.2, label="Validation IoU")
    ax.plot(epochs, train_f1, marker="^", linewidth=2.2, label="Train F1")
    ax.plot(epochs, val_f1, marker="D", linewidth=2.2, label="Validation F1")
    ax.set_title("Occupancy Metrics across Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=True, ncol=2)

    fig.suptitle("Model Training Behaviour", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    out_path = ensure_parent(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bar(metrics_path: str | Path, out_path: str | Path, title: str = "Test Metrics") -> None:
    """Bar chart of the four headline test metrics (IoU / Precision / Recall / F1)."""
    set_publication_style()
    metrics = load_json(metrics_path)
    names = ["iou", "precision", "recall", "f1"]
    labels = ["IoU", "Precision", "Recall", "F1"]
    values = [float(metrics[k]) for k in names if k in metrics]
    shown_labels = [labels[i] for i, k in enumerate(names) if k in metrics]
    if not values:
        raise ValueError(f"No supported metrics found in {metrics_path}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(values))
    bars = ax.bar(x, values, width=0.62, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x, shown_labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontweight="bold")
    if "loss" in metrics:
        ax.text(0.99, 0.04, f"Loss: {float(metrics['loss']):.4f}", transform=ax.transAxes, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.9))
    fig.tight_layout()
    out_path = ensure_parent(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sweep(results_path: str | Path, out_path: str | Path) -> None:
    """Line plot of each metric vs. the occupancy probability threshold.

    Shows the precision/recall trade-off as the binarisation threshold moves,
    justifying the operating point used elsewhere.
    """
    set_publication_style()
    results = load_json(results_path)
    rows = results["results"] if isinstance(results, dict) and "results" in results else results
    thresholds = [float(r["threshold"]) for r in rows]
    metrics = ["iou", "precision", "recall", "f1"]
    labels = {"iou": "IoU", "precision": "Precision", "recall": "Recall", "f1": "F1"}

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for metric in metrics:
        ax.plot(thresholds, [float(r[metric]) for r in rows], marker="o", linewidth=2.2, label=labels[metric])
    ax.set_title("Threshold Sensitivity")
    ax.set_xlabel("Occupancy probability threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=True, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = ensure_parent(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_missingness_results(results_path: str | Path, out_path: str | Path) -> None:
    """Plot metrics as the extra input-missingness level increases (robustness study)."""
    set_publication_style()
    results = load_json(results_path)
    rows = results["results"] if isinstance(results, dict) and "results" in results else results
    labels = [str(r["name"]) for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(x, [float(r["iou"]) for r in rows], marker="o", linewidth=2.4, label="IoU")
    ax.plot(x, [float(r["f1"]) for r in rows], marker="s", linewidth=2.4, label="F1")
    ax.plot(x, [float(r["precision"]) for r in rows], marker="^", linewidth=2.0, label="Precision")
    ax.plot(x, [float(r["recall"]) for r in rows], marker="D", linewidth=2.0, label="Recall")
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Additional input missingness level")
    ax.set_ylabel("Score")
    ax.set_title("Robustness to Input Occupancy Missingness")
    ax.legend(frameon=True, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = ensure_parent(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _configure_3d_axis(ax, spec: VoxelSpec) -> None:
    """Set a consistent camera view, axis limits and labels for the 3D scatter plots.

    Plots use ``(x, z, -y)`` so the scene appears upright with depth going into
    the page, matching how the room is actually oriented.
    """
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("-y")
    ax.set_xlim(spec.bounds["x"])
    ax.set_ylim(spec.bounds["z"])
    ax.set_zlim((-spec.bounds["y"][1], -spec.bounds["y"][0]))
    ax.view_init(elev=22, azim=-65)
    try:
        ax.xaxis.pane.set_alpha(0.05)
        ax.yaxis.pane.set_alpha(0.05)
        ax.zaxis.pane.set_alpha(0.05)
    except Exception:
        pass


def _scatter_volume(ax, volume: np.ndarray, spec: VoxelSpec, max_points: int = 7000, s: float = 1.1, alpha: float = 0.72, color=None):
    pts = occupancy_to_points(volume.squeeze(), spec, max_points=max_points)
    if pts.size:
        ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], s=s, alpha=alpha, color=color)
    return pts.shape[0]


def _projection(volume: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """Collapse a 3D occupancy volume to a 2D silhouette by max-projecting over ``axes``."""
    return np.max(volume.squeeze() > 0.5, axis=axes).astype(np.float32)


def save_occupancy_triplet_advanced(
    input_occ: np.ndarray,
    pred_occ: np.ndarray,
    target_occ: np.ndarray,
    spec: VoxelSpec,
    out_path: str | Path,
    title: str = "NYUv2 proxy scene completion",
    max_points: int = 7000,
) -> None:
    """Save a report-ready triplet figure with 3D scatter and top-view projections."""
    set_publication_style()
    volumes = [input_occ, pred_occ, target_occ]
    names = ["Input occupancy", "Predicted occupancy", "Target / proxy occupancy"]
    fig = plt.figure(figsize=(15, 8.5))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    for i, (vol, name) in enumerate(zip(volumes, names), start=1):
        ax = fig.add_subplot(2, 3, i, projection="3d")
        shown = _scatter_volume(ax, vol, spec, max_points=max_points)
        ax.set_title(f"{name}\nshown voxels: {shown}")
        _configure_3d_axis(ax, spec)

    proj_titles = ["Input top projection", "Prediction top projection", "Target top projection"]
    for i, (vol, name) in enumerate(zip(volumes, proj_titles), start=4):
        ax = fig.add_subplot(2, 3, i)
        top = _projection(vol, axes=(1,))  # collapse y; output z-x
        ax.imshow(top, origin="lower", aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("x voxel")
        ax.set_ylabel("z voxel")
        ax.grid(False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = ensure_parent(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_qualitative_gallery(
    rows: Sequence[Dict],
    spec: VoxelSpec,
    out_path: str | Path,
    title: str = "Qualitative Occupancy Completion Gallery",
    max_points: int = 5000,
) -> None:
    """Rows must contain input, pred, target and index."""
    set_publication_style()
    n = len(rows)
    if n == 0:
        raise ValueError("No rows provided for qualitative gallery")
    names = ["Input", "Prediction", "Target / proxy"]
    fig = plt.figure(figsize=(14, max(3.8, 3.2 * n)))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    for r, row in enumerate(rows):
        vols = [row["input"], row["pred"], row["target"]]
        for c, (vol, name) in enumerate(zip(vols, names)):
            ax = fig.add_subplot(n, 3, r * 3 + c + 1, projection="3d")
            shown = _scatter_volume(ax, vol, spec, max_points=max_points)
            sample = int(row.get("index", r))
            ax.set_title(f"Sample {sample} - {name}\nshown voxels: {shown}")
            _configure_3d_axis(ax, spec)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out_path = ensure_parent(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_projection_gallery(
    rows: Sequence[Dict],
    out_path: str | Path,
    title: str = "Multi-view Occupancy Projections",
) -> None:
    set_publication_style()
    n = len(rows)
    if n == 0:
        raise ValueError("No rows provided for projection gallery")
    vol_names = ["Input", "Prediction", "Target"]
    view_names = ["Top x-z", "Front x-y", "Side z-y"]
    fig, axes = plt.subplots(n, 9, figsize=(18, max(3.0, 2.4 * n)))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    for r, row in enumerate(rows):
        volumes = [row["input"], row["pred"], row["target"]]
        sample = int(row.get("index", r))
        for vi, vol in enumerate(volumes):
            projections = [
                _projection(vol, axes=(1,)),       # z-x top view
                _projection(vol, axes=(0,)),       # y-x front view
                _projection(vol, axes=(2,)).T,     # y-z side view
            ]
            for pi, proj in enumerate(projections):
                ax = axes[r, vi * 3 + pi]
                ax.imshow(proj, origin="lower", aspect="auto")
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"{vol_names[vi]}\n{view_names[pi]}")
                if vi == 0 and pi == 0:
                    ax.set_ylabel(f"Sample {sample}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_path = ensure_parent(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_error_map(
    input_occ: np.ndarray,
    pred_occ: np.ndarray,
    target_occ: np.ndarray,
    spec: VoxelSpec,
    out_path: str | Path,
    title: str = "Prediction Error Map",
    max_points_each: int = 5000,
) -> None:
    """Save TP/FP/FN error visualization for one sample."""
    set_publication_style()
    pred = pred_occ.squeeze() > 0.5
    target = target_occ.squeeze() > 0.5
    tp = pred & target
    fp = pred & (~target)
    fn = (~pred) & target

    fig = plt.figure(figsize=(15, 8.5))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    vols = [input_occ, pred_occ, target_occ]
    names = ["Input occupancy", "Predicted occupancy", "Target / proxy occupancy"]
    for i, (vol, name) in enumerate(zip(vols, names), start=1):
        ax = fig.add_subplot(2, 3, i, projection="3d")
        _scatter_volume(ax, vol, spec, max_points=max_points_each)
        ax.set_title(name)
        _configure_3d_axis(ax, spec)

    ax = fig.add_subplot(2, 3, 4, projection="3d")
    categories = [
        (tp.astype(np.float32), "True positive", "#2ca02c"),
        (fp.astype(np.float32), "False positive", "#d62728"),
        (fn.astype(np.float32), "False negative", "#ff7f0e"),
    ]
    for vol, label, color in categories:
        pts = occupancy_to_points(vol, spec, max_points=max_points_each)
        if pts.size:
            ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], s=1.4, alpha=0.75, color=color, label=label)
    ax.set_title("3D error map")
    _configure_3d_axis(ax, spec)
    ax.legend(frameon=True, loc="upper left")

    err_rgb = np.zeros((*tp.shape, 3), dtype=np.float32)
    err_rgb[tp] = np.array([0.17, 0.63, 0.17])
    err_rgb[fp] = np.array([0.84, 0.15, 0.16])
    err_rgb[fn] = np.array([1.00, 0.50, 0.05])
    # top projection by category
    top_rgb = err_rgb.max(axis=1)
    front_rgb = err_rgb.max(axis=0)
    for slot, img, name in [(5, top_rgb, "Top-view error projection"), (6, front_rgb, "Front-view error projection")]:
        ax = fig.add_subplot(2, 3, slot)
        ax.imshow(img, origin="lower", aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("voxel")
        ax.set_ylabel("voxel")
        ax.grid(False)
    handles = [Line2D([0], [0], marker="o", linestyle="", label="TP", color="#2ca02c"),
               Line2D([0], [0], marker="o", linestyle="", label="FP", color="#d62728"),
               Line2D([0], [0], marker="o", linestyle="", label="FN", color="#ff7f0e")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out_path = ensure_parent(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_comparison(results_path: str | Path, out_path: str | Path, title: str = "Experiment Comparison") -> None:
    """Grouped IoU/F1 bars comparing named variants of an ablation study."""
    set_publication_style()
    payload = load_json(results_path)
    rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
    labels = [str(r.get("name", r.get("variant", i))) for i, r in enumerate(rows)]
    metrics = ["iou", "f1"]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for offset, metric in zip([-width/2, width/2], metrics):
        vals = [float(r[metric]) for r in rows]
        bars = ax.bar(x + offset, vals, width=width, label=metric.upper(), edgecolor="black", linewidth=0.7)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(frameon=True)
    fig.tight_layout()
    out_path = ensure_parent(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
