# Final Model Checkpoints

This directory contains the trained PointNet++ checkpoints used in the final dissertation experiments.

## AdamW Models

The main XYZ-only AdamW experiment contains nine independently trained checkpoints for each APS training condition:

- Clean-trained: 9 checkpoints
- Error-trained: 9 checkpoints
- Mixed-trained: 9 checkpoints

Total AdamW checkpoints: 27.

These checkpoints correspond to the repeated-run APS and YCB-28 results reported in the dissertation.

## RProp Models

The supplementary RProp comparison contains 10 independently trained checkpoints using the mixed APS training condition.

Total RProp checkpoints: 10.

## Model Input

All final checkpoints use:

- PointNet++ architecture
- XYZ-only point-cloud input
- 1000 points per sample
- Three output classes: box, cylinder and sphere

The AdamW mixed-trained checkpoints represent the final selected model configuration used for the main external evaluation.
