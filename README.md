# Primitive Shape Recognition from 3D Point Clouds Using PointNet++

MSc Robotics Advanced Project  
University of Birmingham

This repository contains the implementation, experiment notebooks, metadata,
selected results and supporting material for an MSc dissertation investigating
primitive-shape recognition from 3D point clouds using PointNet++.

The study evaluates whether a PointNet++ classifier trained only on artificial
primitive-shape (APS) data can generalise to real and external object datasets.

## Target Classes

- Box
- Cylinder
- Sphere

## Main Experiments

The project investigates:

- APS-clean, APS-error and mixed APS training
- XYZ + surface-normal versus XYZ-only input
- PointNet++ classification using AdamW
- Synthetic-to-real evaluation on YCB-28
- External evaluation on a primitive-like ModelNet40 subset
- Repeated-run statistical analysis
- Supplementary RProp optimiser comparison

## Final Configuration

The final model uses:

- PointNet++ classifier
- 1000 XYZ points per sample
- Mixed APS-clean + APS-error training
- AdamW optimisation
- Nine independent training runs

## Main Results

| Evaluation | Result |
|---|---:|
| YCB-28 mixed XYZ-only mean accuracy | 89.19% |
| YCB-28 median accuracy | 90.00% |
| Best YCB-28 run | 91.25% |
| APS-error-double mixed accuracy | 98.89% |
| ModelNet40 primitive-like subset, 10-vote mean | 81.00% |
| Supplementary RProp YCB-28 mean | 71.20% |

The ModelNet40 result refers only to a manually selected primitive-like subset
and should not be interpreted as performance on the official 40-class
ModelNet40 benchmark.

## Repository Structure

- `notebooks/` – experiment and analysis notebooks
- `src/` – reusable PointNet++ and utility code
- `data_metadata/` – dataset mappings and metadata
- `results/` – final experimental outputs
- `models/` – selected trained-model information/checkpoints
- `figures/` – figures used for analysis and reporting
- `supplementary/` – additional supporting outputs
- `docs/` – final project documentation

## Datasets

The project uses:

- Artificial Primitive Shape (APS) dataset
- YCB Object and Model Set
- ModelNet40

Third-party datasets are not redistributed in this repository. They should be
obtained from their original sources.

## Reproducibility

The notebooks are numbered in the approximate order in which the experimental
pipeline should be followed.

The main final training and YCB-28 evaluation pipeline is contained in:

`notebooks/05_xyz_only_adamw_training_ycb28.ipynb`

The final result analysis is contained in:

`notebooks/06_final_aps_ycb28_analysis.ipynb`

## Author

Kunal Prakash Khose  
MSc Robotics  
University of Birmingham

## Academic Use

This repository accompanies an MSc dissertation and is provided for academic
reproducibility and reference.
