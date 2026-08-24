# Primitive Shape Recognition from 3D Point Clouds Using PointNet++

**MSc Robotics Advanced Project**  
**University of Birmingham**

This repository contains the implementation, experiment notebooks, dataset metadata, selected results, model information and supporting material for an MSc dissertation on primitive-shape recognition from 3D point clouds using PointNet++.

The study investigates whether a PointNet++ classifier trained only on Artificial Primitive Shape (APS) data can generalise to real-object and external CAD datasets.

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
- External evaluation on a manually curated primitive-like ModelNet40 subset
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
| Mixed-model APS-error-double mean accuracy | 98.89% |
| ModelNet40 primitive-like subset, 10-vote mean accuracy | 81.00% |
| Supplementary RProp YCB-28 mean accuracy | 71.20% |

The ModelNet40 result refers only to a manually selected primitive-like subset and must not be interpreted as performance on the official 40-class ModelNet40 benchmark. The bowl category is used only as a weak sphere-like proxy because ModelNet40 does not contain a true sphere or ball category.

## Repository Structure

- `01_FINAL_REPORT/` – final report and related documentation
- `02_NOTEBOOKS-/` – experiment and analysis notebooks
- `03_SOURCE_CODE-/` – reusable source code
- `04_DATA_AND_METADATA-/` – dataset metadata, mappings and reconstruction information
- `05_FINAL_RESULTS-/` – selected final experimental results
- `06_FINAL_MODELS-/` – selected model checkpoints and model information
- `07_FIGURES-/` – final figures and plots
- `08_SUPPLEMENTARY-/` – supplementary supporting material
- `requirements.txt` – Python package requirements
- `.gitignore` – files and directories excluded from version control

## Datasets

The project uses:

- Artificial Primitive Shape (APS) dataset
- YCB Object and Model Set
- ModelNet40

Third-party datasets are not redistributed in this repository and should be obtained from their original sources.

The metadata directory is intended to preserve the exact object/category mappings, sample counts, split definitions and preprocessing information required to reconstruct the experimental datasets.

## Reproducibility

The notebooks are arranged to follow the main experimental workflow.

The main final AdamW training and YCB-28 evaluation pipeline is contained in:

`05_xyz_only_adamw_training_ycb28.ipynb`

The final APS and YCB-28 results analysis is contained in:

`06_final_aps_ycb28_analysis.ipynb`

Additional notebooks cover the XYZ + normals baseline, YCB-28 preparation, ModelNet40 primitive-like subset evaluation and the supplementary RProp comparison.

The project reports repeated-run results rather than relying on a single training run. Final numerical outputs and statistical summaries are retained in the results directory to support verification of the dissertation findings.

## Notes on External Data

YCB-28 is used as the primary real-object evaluation dataset.

ModelNet40 is used only as an additional external CAD evaluation through a manually curated primitive-like subset. The reported ModelNet40 results are therefore not directly comparable with official 40-class ModelNet40 benchmark results.

## Author

**Kunal Prakash Khose**  
MSc Robotics  
University of Birmingham

## Academic Use

This repository accompanies an MSc Robotics dissertation and is provided for academic reproducibility, verification and future research use.

## Citation and Use

This repository is provided for academic review, research transparency,
reproducibility, and portfolio purposes.

If this work contributes to academic or research activity, please provide
appropriate attribution. Citation information is available in
[`CITATION.cff`](CITATION.cff).

Copyright © 2026 Kunal Prakash Khose. All rights reserved.

Public availability of this repository does not grant permission to reproduce,
redistribute, republish, modify, or commercially exploit the original project
material without prior permission.

See [`LICENSE`](LICENSE) for the applicable terms.

Third-party datasets and externally sourced materials remain subject to their
respective original licences and terms.
