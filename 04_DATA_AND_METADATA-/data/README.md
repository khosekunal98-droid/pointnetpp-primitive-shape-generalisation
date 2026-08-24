# Project Data

This directory contains the datasets and prepared point-cloud data used in the MSc Robotics project:

**Primitive Shape Recognition from 3D Point Clouds Using PointNet++ Based Deep Learning**

The project investigates synthetic-to-real generalisation of a PointNet++ classifier for three primitive shape classes:

- Box
- Cylinder
- Sphere

All experiments use point clouds containing 1000 points per sample.

## Directory Structure

### `raw/`

Contains the Artificial Primitive Shape (APS) data used as the primary synthetic training and evaluation source.

The APS data are organised into:

- `clean`
- `error`
- `error_double`

with separate folders for boxes, cylinders and spheres.

### `generated/`

Contains the generated APS-style evaluation datasets used during the experiments:

- `aps_clean_test_style`
- `aps_clean_val_style`
- `aps_error_test_style`
- `aps_error_double_test_style`

These datasets were generated separately from the main APS training data and were used for controlled validation and robustness evaluation.

### `ycb28/`

Contains the YCB-28 real-object dataset preparation used for external synthetic-to-real evaluation.

The manually curated YCB-28 subset contains:

- 9 box-like objects
- 8 cylinder-like objects
- 11 sphere-like objects

Each object contributes 20 generated point-cloud samples, giving:

**28 objects × 20 samples = 560 point clouds**

The final prepared dataset used for the reported experiments is stored in:

`generated_pointclouds_bbox_norm/`

The directory also retains the downloaded YCB resources, extracted models and supporting object images used during dataset preparation.

### `modelnet40/`

Contains the ModelNet40 data used for the additional cross-dataset evaluation.

A manually selected primitive-like subset was evaluated using objects mapped to the three project classes. The subset contains box-like, cylinder-like and bowl objects, with bowls used as a weak proxy for the sphere class.

## Reproducibility

Dataset preparation and evaluation procedures are documented in the notebooks contained in `02_NOTEBOOKS-`.

Important preparation stages include:

- APS setup and dataset generation
- YCB-28 object selection and point-cloud generation
- ModelNet40 primitive-like subset construction
- 1000-point sampling
- point-cloud normalisation
- external evaluation using saved PointNet++ checkpoints

Associated manifests and preparation records are stored in:

`../metadata/`

## Dataset Attribution

APS, YCB and ModelNet40 are external datasets used for academic research. Their original publications and sources are cited in the dissertation and project documentation.

Users wishing to reuse or redistribute these datasets should consult the original dataset sources and applicable licence or usage conditions.
