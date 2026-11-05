# Analysis pipeline

The stage registry is `code/stages.json`; `python code/run.py list` prints the available stage names.

| Stage group | Purpose | Primary scripts |
|---|---|---|
| Splits | Five outer held-out-block folds and four inner folds | `01_training_validation/make_splits.py` |
| CAMBER and matched controls | Anchor pretraining, full model, direct branches, interaction and weighting controls | `01_training_validation/train_models.py` |
| EEGNet baselines | Band-restricted and broadband unimodal, decision, and frozen-feature fusion | `02_model_comparison/` |
| Additional controls | Static temporal/branch policies and no-Anchor model | `03_ablation/` |
| Parameter analyses | Single-factor sensitivity and nested selection | `04_parameter_sensitivity/`, `05_nested_selection/` |
| Sensor and spectral analyses | Local-power probes, channel occlusion, regional matched-null tests | `06_spatial_interpretability/` |
| Frozen-model interventions | Reassignment of fitted selection states and within-trial token-order controls | `07_frozen_interventions/` |
| Complementarity | Paired correctness summaries across fitted classifiers | `08_complementarity/` |
| Statistics and tables | Participant bootstrap intervals, exact sign-flip tests, Holm adjustment, result tables | `09_summary/` |

## Core model entry points

- `code/camber.py`: stable public alias and constructor.
- `code/00_shared/models/hierarchical_pair_anchor_models_v25.py`: exact model implementation.
- `code/00_shared/models/feature_coefficients.py`: coefficient registry loading and validation.
- `code/00_shared/models/run_nested_hierarchical_pair_anchor_v25.py`: Anchor and main-model fitting.
- `code/00_shared/config/main_config.json`: fixed main configuration.

## Frozen-model interventions

The intervention code changes fitted selection states or the order of within-trial feature tokens while holding the saved model parameters fixed. The task array is used only to match donor trials within the same held-out block and class. Task codes themselves are not modified.

The internal result key `selection_shuffle` therefore refers to selection-state reassignment. It is retained so that newly generated summaries remain compatible with the manuscript figure inputs.

## Reproducibility records

Training scripts save the configuration, fit indices, evaluation indices, deterministic seed, calibration state, checkpoints, training history, and out-of-fold predictions. Fitted scaling and calibration use development trials only.
