# SOTA-inspired 3D Scene Occupancy Completion on Real NYUv2 Depth Data

A complete, reproducible pipeline that takes a **single real NYUv2 depth map**, back-projects it into
a 3D point cloud, voxelizes it into a `64×64×64` occupancy grid, and trains a **residual-attention 3D
U-Net** to predict a *more complete* occupancy volume. The project reads real NYUv2 data (no synthetic
substitutes), evaluates with IoU / Precision / Recall / F1, exports figures and `.ply` point clouds,
and includes non-learned baselines and six ablation studies.

> The full report is in [`report/3D_Scene_Occupancy_Completion_Report.pdf`](report/3D_Scene_Occupancy_Completion_Report.pdf).

## Results (held-out test set, 146 scenes)

| Metric | Model | Copy-input baseline | Meaning |
|---|---|---|---|
| IoU | **0.726** | 0.633 | Region overlap with the proxy target |
| Precision | 0.771 | 0.970 | Correctness of predicted voxels |
| Recall | **0.923** | 0.646 | Fraction of the target scene recovered |
| F1 | **0.839** | 0.775 | Balance of precision and recall |

The model lifts **recall 0.65 → 0.92** and IoU **+0.076** over "copy the raw input", while a naive
dilation baseline makes things *worse* — i.e. it genuinely **completes** the scene rather than copying
it. See report §6.2 and `outputs/metrics/naive_baselines.json`.

---

## 1. Environment (Windows)

This project is set up to run on **Windows 11 / PowerShell** with **Python 3.10+**. A trained
checkpoint (`outputs/checkpoints/best.pt`) is included, so evaluation, the demo, visualization, and
all analysis run on **CPU** with no GPU required.

From the folder containing this README:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

On Windows the default package index serves CPU builds of PyTorch, so no CUDA install is needed to
reproduce the results below.

> **Note on training hardware.** The checkpoint was trained from scratch on Windows 11 with an
> NVIDIA RTX 4070 Laptop GPU (PyTorch 2.11 + CUDA 12.8). Re-*training* from scratch needs a GPU, but
> evaluation, the demo, visualization, and all analysis reproduce on Windows CPU using the trained
> checkpoint.

## 2. Dataset

**The quickstart in §3 needs no download.** Every NYUv2 scene has already been projected and
voxelized, and those `64x64x64` occupancy grids ship in `data/cache_64/` (see `data/README.md`), so
evaluation, the baselines, the demo and the visualizations reproduce straight from the cache.

Download the official NYUv2 labeled dataset only for the parts that need the raw pixels rather than
the cached voxels: `scripts/inspect_data.py`, the point-resolution / voxel-resolution /
label-assisted ablations, and therefore `scripts/run_custom_ablation_study.py`, which runs all six.
Place the `.mat` file here:

```text
data\nyu_depth_v2_labeled.mat
```

Source: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html

The file provides RGB, raw depth (`rawDepths`, incomplete input source), in-painted depth (`depths`,
proxy target source), and 2D labels (used only in the semantic-prior analysis). The cached grids were
built from it; nothing in this project fabricates synthetic depth.

## 3. Quickstart — reproduce the results (CPU, no dataset download)

Run these four commands to regenerate the headline evidence. They use the included checkpoint and
the cached voxel grids, so they need neither a GPU nor the `.mat` file. Step (b) reproduces the
test-set numbers in the table above exactly.

```powershell
# (a) One-command completion demo: per-scene table + input|prediction|target figures + error maps
python demo.py --config configs\full_train.yaml --checkpoint outputs\checkpoints\best.pt --num 6 --export-ply

# (b) Evaluate on the test set -> IoU / Precision / Recall / F1 / Loss
python evaluate.py --config configs\full_train.yaml --checkpoint outputs\checkpoints\best.pt --split test
type outputs\metrics\test_metrics.json

# (c) Non-learned baselines: proves the model completes rather than copies the input
python scripts\naive_baselines.py --config configs\full_train.yaml --checkpoint outputs\checkpoints\best.pt --split test

# (d) Side-by-side completion figures for several test scenes
python visualize.py --config configs\full_train.yaml --checkpoint outputs\checkpoints\best.pt --split test --num 12
```

Outputs land in `outputs\demo\`, `outputs\metrics\`, and `outputs\visualizations\`.

## 4. Full pipeline (optional — retrain and regenerate everything)

Steps 1 and 5 read the raw NYUv2 pixels and labels, so they need `data\nyu_depth_v2_labeled.mat`
(§2). Steps 2–4 run from the cached voxel grids.

```powershell
# 1. Sanity-check the dataset can be read (writes a depth preview PNG)
python scripts\inspect_data.py --config configs\full_train.yaml --index 0

# 2. Train (GPU recommended; produces best.pt / last.pt / train_history.json / splits.json)
python train.py --config configs\full_train.yaml

# 3. Evaluate + advanced report figures (training curves, metric bars, threshold sweep, galleries)
python evaluate.py --config configs\full_train.yaml --checkpoint outputs\checkpoints\best.pt --split test
python scripts\run_advanced_figures.py --config configs\full_train.yaml --checkpoint outputs\checkpoints\best.pt --split test --num 4

# 4. Export 3D PLY point clouds (input / prediction / target / TP-FP-FN error map)
python scripts\export_occupancy_ply.py --config configs\full_train.yaml --checkpoint outputs\checkpoints\best.pt --split test --indices 0,20,40,60,80,100,120,145

# 5. Run all six ablation studies
python scripts\run_custom_ablation_study.py --config configs\full_train.yaml --checkpoint outputs\checkpoints\best.pt --split test --indices 0,20,40,60,80,100,120
```

**Index note.** `run_advanced_figures.py` expects *original NYUv2 sample ids*, while
`export_occupancy_ply.py` and `run_custom_ablation_study.py` expect *test-local indices* (must be
`< len(test)` = 146). The split is recorded in `outputs\metrics\splits.json`; inspect it with:

```powershell
python -c "import json; s=json.load(open('outputs/metrics/splits.json')); print({k:len(v) for k,v in s.items()})"
```

## 5. Report

The finished report is in the `report/` folder:

- **`report/3D_Scene_Occupancy_Completion_Report.pdf`** — the report to read (figures and tables included)

Just open the PDF; nothing needs to be run.

## 6. Method summary

- **Depth → point cloud.** Pinhole back-projection with NYUv2 intrinsics
  (`fx=518.86, fy=519.47, cx=325.58, cy=253.74`); invalid / out-of-range depths dropped.
- **Voxelization.** Clip to metric bounds and hash into a `64×64×64` grid (voxel size ≈ 0.075–0.12 m),
  one dilation to close 1-voxel gaps.
- **Input / target.** Input from `rawDepths`; proxy target from in-painted `depths`. Training-only
  input degradation (dropout + cuboid cut-outs); test inputs are never augmented.
- **Model.** Residual-attention 3D U-Net (~1.4M params): residual blocks + squeeze-and-excitation
  attention, 4 encoder/decoder stages with skip connections, single occupancy-logit head.
- **Loss.** Weighted BCE (`pos_weight=8`) + `0.5 ×` soft Dice — per-voxel label supervision plus
  direct region-overlap optimisation under heavy class imbalance.
- **Metrics.** IoU / Precision / Recall / F1 from pooled TP/FP/FN.

Full detail, related work, and ablation interpretation are in
[`report/3D_Scene_Occupancy_Completion_Report.pdf`](report/3D_Scene_Occupancy_Completion_Report.pdf).

## 7. Repository structure

```text
nyuv2-scene-completion1/
├── README.md
├── requirements.txt
├── demo.py                     # one-command completion demonstration
├── train.py  evaluate.py  visualize.py
├── configs/                    # default.yaml, full_train.yaml
├── data/                       # place nyu_depth_v2_labeled.mat here (+ caches)
├── src/nyuv2_scc/              # library: geometry, dataset, model, losses, metrics, ...
├── scripts/
│   ├── naive_baselines.py      # copy-input / dilation baselines
│   ├── inspect_data.py  run_advanced_figures.py  export_occupancy_ply.py
│   └── run_custom_ablation_study.py  (+ per-ablation scripts)
├── outputs/                    # checkpoints, metrics, figures, exports, demo
└── report/                     # 3D_Scene_Occupancy_Completion_Report.pdf (the final report)
```

## 8. Output folders

| Folder | Contents |
|---|---|
| `outputs/checkpoints/` | `best.pt`, `last.pt` |
| `outputs/metrics/` | `test_metrics.json`, `train_history.json`, `splits.json`, `naive_baselines.json`, `*_ablation.json` |
| `outputs/figures/` | report-ready figures (training curves, metric bars, ablations, galleries) |
| `outputs/visualizations/` | depth previews and `completion_test_*.png` |
| `outputs/demo/` | `demo.py` completion triplets + error maps |
| `outputs/exports/` | `.ply` point clouds for CloudCompare / MeshLab / Blender / Open3D |
