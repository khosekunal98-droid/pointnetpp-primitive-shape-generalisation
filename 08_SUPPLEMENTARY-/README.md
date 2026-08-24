# Supplementary Results

This directory contains supporting experimental outputs for the MSc Robotics project:

**Primitive Shape Recognition from 3D Point Clouds Using PointNet++ Based Deep Learning**

The material supplements the main results presented in the dissertation and preserves additional run-level, object-level and statistical information for reproducibility.

## Directory Structure

### `appendix/`

Contains supporting material used for, or associated with, the dissertation appendices, including:

- run-level ModelNet40 results
- ModelNet40 voting results and predictions
- ModelNet40 class-wise summaries
- YCB-28 object-level accuracy results
- individual YCB-28 confusion matrices
- combined statistical-test outputs
- exact experimental-setting records
- YCB-28 point-cloud visualisation material

These files provide more detailed experimental evidence than could be included in the main report.

### `tables/`

Contains CSV tables used to construct the numerical summaries presented in the report.

The files cover:

- YCB-28 AdamW evaluation
- APS test performance
- XYZ + normals versus XYZ-only comparison
- ModelNet40 evaluation
- AdamW versus RProp comparison
- Kruskal-Wallis tests
- Mann-Whitney U tests
- run-level statistical data

Where both preliminary and corrected statistical outputs were produced during analysis, the corrected version is retained as the authoritative result.

## Notes

The main dissertation reports only selected summary statistics because of report length and table limits. These supplementary files retain additional experimental detail for transparency and reproducibility.

Final model checkpoints are stored in `06_FINAL_MODELS-`, while the principal experiment outputs are stored in `05_FINAL_RESULTS-`.
