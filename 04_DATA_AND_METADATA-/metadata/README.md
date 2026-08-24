# Dataset Metadata

This directory contains dataset manifests, preparation logs and supporting metadata used to document the data-processing workflow for the project.

The metadata are retained to improve traceability and reproducibility of the APS and YCB-28 experiments.

## APS Metadata

### `dataset_manifest_v2.csv`

Contains the dataset manifest used to document the APS dataset structure and experimental data organisation.

## YCB-28 Metadata

### Final preparation records

The following files describe the final YCB-28 preparation used for the reported evaluation:

- `ycb28_download_log_final.csv`
- `ycb28_extract_log_final.csv`
- `ycb28_mesh_manifest.csv`
- `ycb28_pointcloud_manifest_bbox_norm.csv`
- `ycb28_pointcloud_generation_failures_bbox_norm.csv`

The `bbox_norm` files correspond to the final YCB-28 point-cloud preparation used in the project.

### Download and recovery records

The following files document intermediate download checks and recovery operations:

- `ycb28_download_availability_check.csv`
- `ycb28_download_log.csv`
- `ycb28_retry_download_log.csv`
- `ycb28_001_chips_can_fallback_log.csv`

These records are retained as provenance information showing how the required YCB object models were obtained.

### Earlier preparation records

The following files correspond to an earlier YCB point-cloud preparation stage:

- `ycb28_pointcloud_manifest.csv`
- `ycb28_pointcloud_generation_failures.csv`

They are retained for experimental traceability. The final reported evaluation uses the corresponding `bbox_norm` versions.

## Final YCB-28 Dataset

The manually curated YCB-28 subset contains:

| Primitive class | Objects | Samples |
|---|---:|---:|
| Box | 9 | 180 |
| Cylinder | 8 | 160 |
| Sphere | 11 | 220 |
| **Total** | **28** | **560** |

Each object contributes 20 point-cloud samples with 1000 points per sample.

## Notes

Notebook-specific temporary status JSON files are intentionally excluded because they are not required to reproduce the experiments.

The experiment notebooks, source code, final model checkpoints and result tables are stored elsewhere in the repository.
