# SOTA-inspired 3D Scene Occupancy Completion from Real NYUv2 Depth Data

This repository implements a coursework project for 3D scene occupancy completion using real NYUv2 RGB-D depth data.  
The project builds a complete experimental pipeline from real depth maps to 3D occupancy prediction, quantitative evaluation, visualization, PLY point-cloud export, and ablation analysis.

## 1. Project Overview

The main pipeline is:

1. Load real NYUv2 data from `nyu_depth_v2_labeled.mat`.
2. Use `rawDepths` to construct incomplete input occupancy.
3. Use processed `depths` to construct target/proxy occupancy.
4. Convert depth maps into 3D point clouds using camera intrinsics.
5. Voxelize point clouds into 3D occupancy volumes.
6. Train a residual-attention 3D U-Net.
7. Evaluate IoU, Precision, Recall, F1, and Loss.
8. Generate visualization figures and export `.ply` point clouds.
9. Run additional ablation studies on noise, partial observation, point resolution, voxel resolution, and semantic priors.

Recommended project title:

> A SOTA-inspired 3D Scene Completion Prototype Using Real NYUv2 Depth Data

## 2. Dataset

Download the official NYUv2 labeled dataset from:

https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html

Expected file path:

```text
data/nyu_depth_v2_labeled.mat
```

The project requires the real NYUv2 `.mat` file. It does not generate dummy depth maps or synthetic replacement data.

The NYUv2 labeled file contains RGB images, raw depth maps, processed depth maps, and labels. In this project:

- `rawDepths` are used to construct the input occupancy.
- `depths` are used to construct the target/proxy occupancy.
- RGB images are used for visual inspection and report figures.
- Labels are used only in the label-assisted / semantic-prior analysis.

## 3. Recommended Runtime Environment

The final large-sample experiment was run on a Linux GPU environment.

Recommended environment:

```text
Operating system: Ubuntu 22.04
Python: 3.12
PyTorch: 2.7.0
CUDA: 12.8
GPU: NVIDIA RTX 4090 24GB
CPU: 16 vCPU Intel Xeon Gold 6430
RAM: 120GB
```

GPU training is strongly recommended because 3D convolution is much slower on CPU.

Check GPU availability:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected output should include:

```text
True
NVIDIA GeForce RTX 4090
```

If PyTorch with CUDA 12.8 needs to be installed manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 4. Installation

### Linux / Ubuntu

```bash
cd /root/autodl-tmp/nyuv2-scene-completion
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
cd D:\nyuv2-scene-completion\nyuv2-scene-completion1
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If the virtual environment is already created, only activate it before running scripts.

## 5. Final Training Setting

The final report uses the large-sample setting:

```text
Config file: configs/full_train.yaml
Dataset: real nyu_depth_v2_labeled.mat
Max samples: 1449
Training epochs: 20
Main voxel grid: 64 x 64 x 64
Train / validation / test split: 1159 / 144 / 146
Main checkpoint: outputs/checkpoints/best.pt
```

The split is stored in:

```text
outputs/metrics/splits.json
```

This file is important because some scripts use test sample ids, while some scripts use test local indices.

## 6. Linux / Ubuntu Full Run Steps

### Step 1. Enter the project directory

```bash
cd /root/autodl-tmp/nyuv2-scene-completion
```

### Step 2. Inspect the real NYUv2 dataset

```bash
python scripts/inspect_data.py --config configs/full_train.yaml --index 0
```

Output:

```text
outputs/visualizations/depth_preview_0000.png
```

This figure shows the RGB image, `rawDepths`, and `depths`, and confirms that the real NYUv2 data can be read correctly.

### Step 3. Train the model

```bash
python train.py --config configs/full_train.yaml
```

Outputs:

```text
outputs/checkpoints/best.pt
outputs/checkpoints/last.pt
outputs/metrics/train_history.json
outputs/metrics/splits.json
```

Meaning:

- `best.pt`: best validation checkpoint, used for evaluation, visualization, PLY export, and ablation studies.
- `last.pt`: checkpoint from the final epoch.
- `train_history.json`: training and validation history.
- `splits.json`: train / validation / test split record.

### Step 4. Check train / validation / test split

```bash
python - <<'PY'
import json
s = json.load(open("outputs/metrics/splits.json"))
print({k: len(v) for k, v in s.items()})
print("first 20 test ids:", s["test"][:20])
PY
```

Expected large-sample split:

```text
{'train': 1159, 'val': 144, 'test': 146}
```

The printed test ids are original NYUv2 sample ids. They are used by `run_advanced_figures.py`.

### Step 5. Evaluate the model on the test set

```bash
python evaluate.py --config configs/full_train.yaml --checkpoint outputs/checkpoints/best.pt --split test
cat outputs/metrics/test_metrics.json
```

Output:

```text
outputs/metrics/test_metrics.json
```

This file contains IoU, Precision, Recall, F1, and Loss.

### Step 6. Generate standard completion visualizations

```bash
python visualize.py --config configs/full_train.yaml --checkpoint outputs/checkpoints/best.pt --split test --num 12
```

Output:

```text
outputs/visualizations/completion_test_xxxx.png
```

These images compare input occupancy, predicted occupancy, and target/proxy occupancy.

### Step 7. Generate advanced figures without hard-coded sample ids

This step automatically reads valid test ids from `splits.json`, so it remains valid after retraining.

```bash
TEST_IDS=$(python - <<'PY'
import json
s = json.load(open("outputs/metrics/splits.json"))
print(",".join(map(str, s["test"][:6])))
PY
)

ERR_IDS=$(python - <<'PY'
import json
s = json.load(open("outputs/metrics/splits.json"))
print(",".join(map(str, s["test"][:3])))
PY
)

echo "Using test ids: $TEST_IDS"
echo "Using error ids: $ERR_IDS"

python scripts/run_advanced_figures.py \
  --config configs/full_train.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split test \
  --indices "$TEST_IDS" \
  --error-indices "$ERR_IDS"
```

Outputs:

```text
outputs/figures/training_curves.png
outputs/figures/test_metrics_bar.png
outputs/figures/threshold_sweep_test.png
outputs/figures/missingness_test.png
outputs/figures/qualitative_gallery.png
outputs/figures/qualitative_projection_gallery.png
outputs/figures/error_map_test_xxxx.png
outputs/metrics/threshold_sweep_test.json
outputs/metrics/missingness_test.json
```

Important note:

- `run_advanced_figures.py` uses original NYUv2 test sample ids.
- Do not manually reuse local indices such as `15,23,40,80` unless they are confirmed to exist in `splits.json`.

### Step 8. Export real 3D PLY point clouds

`export_occupancy_ply.py` uses test local indices. The following command automatically keeps only valid local indices.

```bash
PLY_LOCAL_IDS=$(python - <<'PY'
import json
s = json.load(open("outputs/metrics/splits.json"))
n = len(s["test"])
candidates = [0, 20, 40, 60, 80, 100, 120, 145]
valid = [i for i in candidates if i < n]
print(",".join(map(str, valid)))
PY
)

echo "Using PLY local indices: $PLY_LOCAL_IDS"

python scripts/export_occupancy_ply.py \
  --config configs/full_train.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split test \
  --indices "$PLY_LOCAL_IDS"
```

Outputs:

```text
outputs/exports/*_input.ply
outputs/exports/*_prediction.ply
outputs/exports/*_target_proxy.ply
outputs/exports/*_error_tp_fp_fn.ply
outputs/exports/export_manifest.json
```

The `.ply` files can be opened in CloudCompare, MeshLab, Blender, or Open3D.

### Step 9. Run custom ablation studies

This step automatically selects valid test local indices.

```bash
ABLATION_LOCAL_IDS=$(python - <<'PY'
import json
s = json.load(open("outputs/metrics/splits.json"))
n = len(s["test"])
candidates = [0, 20, 40, 60, 80, 100, 120]
valid = [i for i in candidates if i < n]
print(",".join(map(str, valid)))
PY
)

echo "Using ablation local indices: $ABLATION_LOCAL_IDS"

python scripts/run_custom_ablation_study.py \
  --config configs/full_train.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split test \
  --indices "$ABLATION_LOCAL_IDS"
```

Outputs:

```text
outputs/metrics/noise_filtering_ablation.json
outputs/metrics/partial_generation_ablation.json
outputs/metrics/point_resolution_ablation.json
outputs/metrics/voxel_resolution_ablation.json
outputs/metrics/label_assisted_ablation.json

outputs/figures/noise_filtering_ablation.png
outputs/figures/partial_generation_ablation.png
outputs/figures/point_resolution_ablation.png
outputs/figures/voxel_resolution_ablation.png
outputs/figures/label_assisted_ablation.png

outputs/exports/*.ply
```

## 7. Windows PowerShell Command Notes

Most Linux commands use `\` for line continuation. Windows PowerShell does not use `\` for line continuation. Use either one-line commands or PowerShell backtick `` ` ``.

Example one-line command:

```powershell
python scripts/run_advanced_figures.py --config configs/full_train.yaml --checkpoint outputs/checkpoints/best.pt --split test --num 4
```

Example PowerShell multi-line command:

```powershell
python scripts/run_advanced_figures.py `
  --config configs/full_train.yaml `
  --checkpoint outputs/checkpoints/best.pt `
  --split test `
  --num 4
```

To view JSON files on Windows:

```powershell
type outputs\metrics\test_metrics.json
```

To inspect the split on Windows:

```powershell
python -c "import json; s=json.load(open('outputs/metrics/splits.json')); print({k:len(v) for k,v in s.items()}); print('first 20 test ids:', s['test'][:20])"
```

## 8. Manual Commands for Windows PowerShell

### Evaluate

```powershell
python evaluate.py --config configs/full_train.yaml --checkpoint outputs/checkpoints/best.pt --split test
type outputs\metrics\test_metrics.json
```

### Visualize

```powershell
python visualize.py --config configs/full_train.yaml --checkpoint outputs/checkpoints/best.pt --split test --num 12
```

### Advanced figures

The safest PowerShell command is:

```powershell
python scripts/run_advanced_figures.py --config configs/full_train.yaml --checkpoint outputs/checkpoints/best.pt --split test --num 4
```

If manual indices are used, make sure the ids exist in `outputs/metrics/splits.json`.

### Export PLY

Use local test indices that are smaller than the test split length:

```powershell
python scripts/export_occupancy_ply.py --config configs/full_train.yaml --checkpoint outputs/checkpoints/best.pt --split test --indices 0,20,40,60,80,100,120,145
```

### Run all custom ablations

```powershell
python scripts/run_custom_ablation_study.py --config configs/full_train.yaml --checkpoint outputs/checkpoints/best.pt --split test --indices 0,20,40,60,80,100,120
```

If the test split is smaller than the largest index, remove the out-of-range values.

## 9. Method Summary

### Depth to point cloud

For each valid depth pixel `(u, v, Z)`, the pinhole camera model is used:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth(u, v)
```

### Voxelization

The 3D points are clipped to fixed camera-space bounds and written to a binary occupancy grid.

Default grid:

```text
64 x 64 x 64
```

### Input and target/proxy construction

- Input occupancy: built from real NYUv2 `rawDepths`.
- Target/proxy occupancy: built from real NYUv2 `depths`.

### Model

The model is a residual-attention 3D U-Net with:

- 3D residual encoder blocks
- downsampling
- squeeze-and-excitation attention
- transpose-convolution decoder
- skip connections
- binary occupancy logits output

### Metrics

Evaluation metrics:

- IoU
- Precision
- Recall
- F1 score
- Loss

## 10. Custom Ablation Studies

The project includes six extra analysis functions.

### 1. Export occupancy to PLY

Exports input, prediction, target/proxy, and TP/FP/FN error maps as real 3D `.ply` point clouds.

Output:

```text
outputs/exports/
```

### 2. Noise filtering ablation

Compares baseline, isolated voxel removal, median filtering, and combined filtering.

Outputs:

```text
outputs/metrics/noise_filtering_ablation.json
outputs/figures/noise_filtering_ablation.png
```

### 3. Partial generation ablation

Compares clean input, random dropout, block occlusion, and mixed partial input.

Outputs:

```text
outputs/metrics/partial_generation_ablation.json
outputs/figures/partial_generation_ablation.png
```

### 4. Point resolution ablation

Compares different point counts such as 1k, 2k, 5k, 10k, and all points.

Outputs:

```text
outputs/metrics/point_resolution_ablation.json
outputs/figures/point_resolution_ablation.png
```

### 5. Voxel resolution ablation

Compares voxel grids such as 32, 48, and 64.

Outputs:

```text
outputs/metrics/voxel_resolution_ablation.json
outputs/figures/voxel_resolution_ablation.png
```

### 6. Label-assisted / semantic-prior analysis

Uses NYUv2 visible 2D labels as weak semantic priors for filtering and analysis.

Outputs:

```text
outputs/metrics/label_assisted_ablation.json
outputs/figures/label_assisted_ablation.png
```

## 11. Output Folder Explanation

```text
outputs/checkpoints/
```

Stores trained model checkpoints:

```text
best.pt
last.pt
```

```text
outputs/metrics/
```

Stores numeric results:

```text
test_metrics.json
train_history.json
splits.json
threshold_sweep_test.json
missingness_test.json
*_ablation.json
```

```text
outputs/figures/
```

Stores report-ready figures:

```text
training_curves.png
test_metrics_bar.png
threshold_sweep_test.png
missingness_test.png
qualitative_gallery.png
qualitative_projection_gallery.png
error_map_test_xxxx.png
*_ablation.png
```

```text
outputs/visualizations/
```

Stores basic visualizations:

```text
depth_preview_0000.png
completion_test_xxxx.png
```

```text
outputs/exports/
```

Stores real 3D point cloud exports:

```text
*_input.ply
*_prediction.ply
*_target_proxy.ply
*_error_tp_fp_fn.ply
export_manifest.json
```

## 12. Repository Structure

```text
nyuv2-scene-completion/
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml
│   └── full_train.yaml
├── data/
│   ├── README.md
│   └── nyu_depth_v2_labeled.mat
├── src/nyuv2_scc/
│   ├── config.py
│   ├── custom_ablation.py
│   ├── dataset.py
│   ├── geometry.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── nyuv2_io.py
│   ├── ply_utils.py
│   ├── train_eval.py
│   └── visualization.py
├── scripts/
│   ├── inspect_data.py
│   ├── run_advanced_figures.py
│   ├── export_occupancy_ply.py
│   ├── run_custom_ablation_study.py
│   ├── noise_filtering_ablation.py
│   ├── partial_generation_ablation.py
│   ├── point_resolution_ablation.py
│   ├── voxel_resolution_ablation.py
│   └── label_assisted_ablation.py
├── train.py
├── evaluate.py
├── visualize.py
├── outputs/
└── report/
```