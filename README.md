# CAMBER: simultaneous EEG–MEG motor-imagery decoding

Official analysis code for **CAMBER: Input-conditioned multi-branch fusion of simultaneous EEG and MEG for motor imagery decoding**.

CAMBER (Conditioned Aggregation of Multi-Branch EEG–MEG Representations) combines modality-specific branches, explicit paired interactions, bidirectional computational paths, input-conditioned temporal and branch aggregation, and a separately pretrained Anchor predictor.

## Scope and data boundary

This release begins with **preprocessed, paired four-second EEG and MEG epochs sampled at 250 Hz**. It does not implement the acquisition-to-epoch preprocessing described in the article. Raw participant recordings are not distributed here.

## What is included

- the complete CAMBER feature bank, model, Anchor pretraining, and final mixture;
- five-block within-participant training and evaluation;
- matched structural controls and static-weight controls;
- band-restricted and broadband EEGNet comparison pipelines;
- parameter sensitivity and nested selection analyses;
- frozen-model selection/content interventions;
- local-power probes, channel occlusion, regional tests, and sensor projection;
- trial-level complementarity and participant-level statistical summaries;
- the Python source used to generate all quantitative manuscript figures.

Participant recordings, trained checkpoints, and derived result artifacts are not stored in this repository. The expected data layout is documented in [`docs/data_format.md`](docs/data_format.md).

## Label convention

The analysis reads the event-derived task array `y` unchanged:

| Value | Motor-imagery task |
|---:|---|
| 0 | Left hand |
| 1 | Right hand |
| 2 | Both feet |

`y` is treated as a read-only input throughout training, evaluation, frozen-model interventions, and figure calculations. The loss option `label_smoothing` is ordinary cross-entropy regularization: it changes the loss target distribution used during optimization and does not rewrite, exchange, or reassign entries of `y`.

The internal intervention key `selection_shuffle` is retained for compatibility with the saved result schema. It denotes within-class reassignment of fitted controller and branch-marker states. Donor trials are constrained to have the same task code, and the task array itself remains unchanged.

The value `0.12` in the CAMBER implementation bounds learned residual contributions to temporal weighting, branch aggregation, and final prediction mixing. It is a model amplitude and has no relationship to the task codes.

The fixed spatial-projection dictionary is generated directly in [`feature_coefficients.py`](code/00_shared/models/feature_coefficients.py) from seeded Walsh--Hadamard bases and explicit channel-order permutations. Its generation reads no participant signal, task label, sensor position, or performance result. The projections are fixed before model fitting and are never learned or updated. They provide distinct, reproducible mathematical coordinates to the model routes; they are not source-localization estimates or anatomical maps. The embedded permutations reproduce the channel order of the retained analysis dictionary; the requirement for other channel orders is described in [`docs/data_format.md`](docs/data_format.md).

## Repository layout

```text
code/
  camber.py                         Public model entry point
  00_shared/                        Exact feature, model, and training implementation
  01_training_validation/           Block splits and main experiments
  02_model_comparison/              EEGNet baselines
  03_ablation/                      No-Anchor and static-policy controls
  04_parameter_sensitivity/         Single-factor analyses
  05_nested_selection/              Nested parameter selection
  06_spatial_interpretability/      Power and sensor analyses
  07_frozen_interventions/          Saved-model interventions
  08_complementarity/               Paired prediction comparisons
  09_summary/                       Statistics and result tables
figures/make_manuscript_figures.py  Quantitative paper figures
projection/                         EEG/MEG display-coordinate construction
data/channel_positions.json         Template sensor-coordinate registry
```

The numeric analysis identifiers `29101`–`29112` map to the anonymized directories `sub01`–`sub12`; they are analysis codes, not personal identifiers. The internal `v25`, `r11`, and `r12` tags are retained in several implementation and artifact-schema names so that the released code remains compatible with saved experiment manifests and checkpoints. `code/camber.py` provides the stable public model name.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

For CUDA, install the PyTorch build appropriate for the local driver before installing the remaining packages.

## Quick checks

Parse every released Python file and scan the release manifest:

```bash
python -m unittest tests.test_release_integrity
```

With PyTorch installed, check the CAMBER forward contract:

```bash
python -m unittest tests.test_model_contract
```

## Running the analysis

Place each participant file at `data/subXX/paired_mi.npz` as described in [`docs/data_format.md`](docs/data_format.md). Then list or run the registered stages:

```bash
python code/run.py list
python code/run.py 00_train_validation_splits --device cpu
python code/run.py 01_full_and_structural_ablations --device cuda
```

`code/run.py all` executes the complete registered workflow. The default destination is a timestamped directory under `local_runs/`; pass `--results-root` to select another output directory. Several stages are computationally expensive.

The ordered workflow and the relationship between scripts and manuscript analyses are summarized in [`docs/pipeline.md`](docs/pipeline.md).

## Rebuilding quantitative figures

After producing the expected result artifacts:

```bash
python figures/make_manuscript_figures.py --source . --output figures/output
```

The script regenerates the performance, confusion, parameter, complementarity, local-power, spatial-sensitivity, task-power, and sensor-position PDF figures. See [`docs/figures.md`](docs/figures.md) for the input-to-figure mapping.

## Citation

If this code contributes to a publication, please cite the associated CAMBER article. Machine-readable author and title metadata are provided in [`CITATION.cff`](CITATION.cff); journal and DOI fields will be added after publication.

## Authors

Chengliang Yu, Yuanyuan Zhang, Zhaoying Li, Yongzhi Shan, Yumin Luo, and Guoguang Zhao
Xuanwu Hospital, Capital Medical University, Beijing 100053, China
