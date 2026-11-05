# Quantitative figure generation

Run `figures/make_manuscript_figures.py` after the analysis pipeline has created the required result artifacts.

| Output file | Content | Main inputs |
|---|---|---|
| `figure_3_performance.pdf` | Participant distributions and paired performance differences | summary tables and paired comparisons |
| `figure_4_confusion.pdf` | Four row-normalized confusion matrices | out-of-fold predictions |
| `figure_5_parameters.pdf` | Single-factor sensitivity and nested selection | parameter and nested-search summaries |
| `figure_6_complementarity.pdf` | Error recovery and paired correctness matrices | complementarity summaries |
| `figure_7_power.pdf` | Local-power probe accuracy and temporal concordance | local-power fold records |
| `figure_8_spatial.pdf` | Sensor sensitivity maps and regional excess effects | occlusion arrays and regional tests |
| `figure_S1_task_power.pdf` | Class-dependent task-power ranges | saved task-power arrays |
| `figure_S2_sensor_projection.pdf` | EEG and MEG display positions | `data/channel_positions.json` |

The script fixes typography, colors, panel dimensions, bootstrap displays, boxplots, shared color scales, and topographic interpolation. It verifies key numerical values before saving the figures.

The experimental-protocol illustration and neural-network architecture diagrams are static manuscript artwork. Their quantitative content is defined by the methods and model source rather than generated from result arrays.
