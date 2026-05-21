# 3D PLY Exports

This folder is used by `scripts/export_occupancy_ply.py`.

The exported files are real 3D point cloud files generated from occupancy volumes:

- `*_input.ply`: input occupancy generated from NYUv2 rawDepths.
- `*_prediction.ply`: model-predicted completed proxy occupancy.
- `*_target_proxy.ply`: target/proxy occupancy generated from NYUv2 depths.
- `*_error_tp_fp_fn.ply`: color-coded error map.

Color convention for error PLY files:

- Green: true positive occupied voxels.
- Red: false positive extra occupied voxels.
- Orange: false negative missed occupied voxels.

Open these files with MeshLab, CloudCompare, Blender, or Open3D to rotate the 3D result interactively.
