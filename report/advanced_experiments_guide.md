# Advanced Experiments and Visualizations Guide

This guide adds report-ready figures and optional experiments to the NYUv2 3D Scene Completion coursework project.

## Recommended figure set for the final report

1. **Training curves**: `outputs/figures/training_curves.png`
   - Shows train/validation loss and IoU/F1 across epochs.
2. **Test metrics bar chart**: `outputs/figures/test_metrics_bar.png`
   - Shows IoU, Precision, Recall and F1 in a clean bar chart.
3. **Threshold sweep**: `outputs/figures/threshold_sweep_test.png`
   - Shows the trade-off between Precision and Recall when changing the occupancy threshold.
4. **Qualitative gallery**: `outputs/figures/qualitative_gallery.png`
   - Shows input / prediction / proxy target in a polished multi-sample layout.
5. **Projection gallery**: `outputs/figures/qualitative_projection_gallery.png`
   - Shows top/front/side projections that are often easier to read than 3D scatter plots.
6. **TP / FP / FN error maps**: `outputs/figures/error_map_test_XXXX.png`
   - Shows where the model is correct and where it over/under-predicts occupancy.
7. **Missingness robustness plot**: `outputs/figures/missingness_test.png`
   - Tests how performance changes when additional missingness is applied to the input occupancy.

## One-command advanced figure generation

After training and evaluating the model, run:

```bash
python scripts/run_advanced_figures.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --split test --num 4
```

For your known good examples, use:

```bash
python scripts/run_advanced_figures.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --split test --indices 15,23,137,316 --error-indices 109,272
```

## Individual commands

```bash
python scripts/plot_training_curves.py
python scripts/plot_metrics_bar.py
python scripts/threshold_sweep.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --split test
python scripts/make_qualitative_gallery.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --split test --indices 15,23,137,316 --save-triplets
python scripts/plot_error_map.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --split test --indices 109,272
python scripts/missingness_experiment.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --split test
```

## Optional attention ablation

This script trains two short variants: attention enabled and attention disabled. It is optional because it takes extra time.

```bash
python scripts/run_attention_ablation.py --config configs/default.yaml --epochs 4 --split test
```

For a full-length ablation, replace `--epochs 4` with `--epochs 12`.

## Suggested report text

The threshold sweep shows the precision-recall trade-off introduced by the occupancy probability threshold. Lower thresholds usually increase recall but may generate more false positive voxels, while higher thresholds usually improve precision but may miss true occupied regions. This supports the interpretation that the model has learned a meaningful completion prior but still tends to over-predict some occupied voxels in difficult scenes.

The TP/FP/FN visualization provides spatial error analysis. True positives show occupied voxels correctly recovered by the model, false positives show additional occupied voxels predicted by the model, and false negatives show proxy target voxels missed by the prediction. This error analysis is especially useful for explaining cases where recall is high but precision is lower.

The missingness experiment evaluates the robustness of the trained model under additional input degradation. Since the target/proxy occupancy still comes from real NYUv2 depth maps, this experiment remains consistent with the coursework requirement of using real data, while testing how completion quality changes when the observed input becomes more incomplete.

## Custom Ablation Study Added

This version adds three practical improvements/experiments to make the project look more like a complete 3D occupancy completion prototype rather than only a set of PNG figures.

### 1. Export Occupancy Volumes to PLY

Command:

```bash
python scripts/export_occupancy_ply.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --split test --indices 15,23,137,316
```

Outputs:

```text
outputs/exports/*_input.ply
outputs/exports/*_prediction.ply
outputs/exports/*_target_proxy.ply
outputs/exports/*_error_tp_fp_fn.ply
```

These files can be opened in MeshLab, CloudCompare, Blender, or Open3D. This makes the result a real 3D point cloud export instead of only a static 2D PNG screenshot.

### 2. Noise Filtering Ablation

Command:

```bash
python scripts/noise_filtering_ablation.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --split test
```

Outputs:

```text
outputs/metrics/noise_filtering_ablation.json
outputs/figures/noise_filtering_ablation.png
outputs/figures/noise_filtering_examples/
```

This experiment compares baseline input occupancy with simple cleanup variants such as isolated voxel removal and 3D median filtering. It analyzes whether simple filtering helps reduce NYUv2 depth/voxel noise before completion.

### 3. Partial Generation Ablation

Command:

```bash
python scripts/partial_generation_ablation.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --split test
```

Outputs:

```text
outputs/metrics/partial_generation_ablation.json
outputs/figures/partial_generation_ablation.png
outputs/figures/partial_generation_examples/
```

This experiment compares different partial input generation strategies, such as random dropout, block occlusion, and mixed corruption. It tests whether the model is more sensitive to scattered missing voxels or larger continuous missing regions.

### One-command Custom Study

```bash
python scripts/run_custom_ablation_study.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --split test --indices 15,23,137,316
```

For a quick smoke test:

```bash
python scripts/run_custom_ablation_study.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --split test --indices 15,23 --max-samples 20
```

### Suggested Report Wording

The custom ablation study analyzes three input-side factors: whether the 3D output can be exported as real point cloud files, whether simple noise filtering improves proxy occupancy completion, and how different partial observation patterns affect the model. These experiments do not change the task into full 3D reconstruction. They still evaluate proxy 3D occupancy completion using real NYUv2 depth-derived inputs and targets.

### 4. Point Resolution Ablation

This ablation changes the maximum number of input points used before voxelization, for example 1k, 2k, 5k, 10k, and all available points. The target/proxy occupancy is kept fixed. The purpose is to test whether the model is sensitive to sparse point clouds. If performance drops at low point counts, it means the input geometry is too sparse to support stable completion.

### 5. Voxel Resolution Ablation

This ablation evaluates 32^3, 48^3, and 64^3 voxel grids. The same trained convolutional checkpoint is used because the 3D U-Net is fully convolutional. This should be explained as an inference-time representation-resolution analysis, not a full retrained model comparison. It tests whether coarser or finer occupancy grids change IoU, F1, and visual quality.

### 6. Label-assisted / Semantic-prior Analysis

This ablation uses NYUv2 2D dense labels as weak visible-region priors. It compares geometry-only input against label-filtered input variants such as valid-label masking, top-k dominant visible labels, boundary-clean filtering, and large connected semantic components. The labels are not complete 3D ground truth and are not used as target occupancy. This experiment only studies whether simple visible semantic priors can reduce clutter or noisy regions before point-cloud generation.

Recommended wording for the report:

```text
To make the project more than a baseline implementation, I added custom ablation studies for point-cloud resolution, voxel-grid resolution, and visible semantic-prior filtering. These experiments analyze how input sparsity, 3D representation resolution, and NYUv2 2D labels affect proxy occupancy completion. The label-assisted setting uses visible 2D labels only as weak priors and does not provide complete 3D semantic supervision.
```
