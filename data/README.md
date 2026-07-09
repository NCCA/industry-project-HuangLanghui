# Data folder

## The dataset itself is not included — download it first

Download the real NYUv2 labeled dataset from the official page and place it here:

```text
data/nyu_depth_v2_labeled.mat
```

Source: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html

This file is required by **every** entry point, including evaluation and the demo. The code will
fail with a clear error if this real dataset file is missing. It does not generate dummy depth maps
or synthetic replacement data.

## Precomputed voxel caches (included)

Converting a depth map to a `64x64x64` occupancy grid is deterministic, so each result is cached to
`.npz` and reused. These caches ship alongside the code so the first run does not have to
re-voxelize the whole dataset:

| Folder | Files | Contents |
|---|---|---|
| `cache_64/` | 1449 | All NYUv2 scenes at `64x64x64` — used by `configs/default.yaml` and `configs/full_train.yaml` |
| `cache_64_grid_32x32x32/` | 174 | Voxel-resolution ablation subset (`scripts/voxel_resolution_ablation.py`) |
| `cache_64_grid_48x48x48/` | 174 | Voxel-resolution ablation subset |
| `cache_64_grid_64x64x64/` | 174 | Voxel-resolution ablation subset |

Cache files are named `nyuv2_<index>_<input_key>_to_<target_key>_<grid>.npz`, so different source
keys and grid resolutions never collide.

`cache_64/` is committed to the repository so a fresh `git clone` reproduces the reported results on
CPU without downloading anything. The three ablation caches are not: they cover a subset of scenes,
and the splits are index-based, so a partial cache cannot reconstruct the recorded partition. The
scripts that need them rebuild them from the `.mat` file.

The caches are a convenience, not a dependency: deleting any of them is safe, and the missing entries
are regenerated from `nyu_depth_v2_labeled.mat` on the next run.
