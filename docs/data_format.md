# Input data format

The analysis expects one directory per participant:

```text
data/
  sub01/paired_mi.npz
  sub02/paired_mi.npz
  ...
  sub12/paired_mi.npz
  channel_positions.json
```

## Required arrays in `paired_mi.npz`

| Key | Shape | Type | Meaning |
|---|---:|---|---|
| `eeg` | `(150, 64, 1000)` | `float32` | Four-second EEG epochs sampled at 250 Hz |
| `meg` | `(150, 64, 1000)` | `float32` | Corresponding four-second MEG epochs sampled at 250 Hz |
| `y` | `(150,)` | integer | Task code: 0 left hand, 1 right hand, 2 both feet |
| `block` | `(150,)` | integer | Five recording blocks, 30 trials per block |
| `sampling_frequency_hz` | scalar | numeric | Sampling frequency; expected value is 250 |

The rows of `eeg`, `meg`, `y`, and `block` must describe the same trials in the same order. Every participant must contain 50 trials for each task overall. Class counts may differ within an individual block.

Recommended additional metadata arrays include `trial_id`, EEG and MEG channel names, and an anonymized analysis identifier. The CAMBER training code uses the first four arrays; the full comparison pipeline also validates `sampling_frequency_hz`.

## Channel order

The fixed coefficient registry is generated transparently in `code/00_shared/models/feature_coefficients.py` from two seeded Walsh--Hadamard bases and saved channel-order permutations. The generator does not read participant signals, task labels, sensor positions, or performance results. Its output assumes the same 64-channel order used by the analysis arrays. If another channel order is used, apply the identical permutation to the input arrays and all corresponding coefficient vectors before training.

`data/channel_positions.json` contains the template-coordinate mapping used by the sensor displays. These coordinates are display metadata and are not passed to CAMBER during decoding.

## Splits

The outer test fold is one complete recording block. The other four blocks form the development partition. Each development partition is divided into four inner folds by holding out one of its blocks in turn.

Run:

```bash
python code/01_training_validation/make_splits.py
```

to create and verify `split.json` files from the `block` and `y` arrays.
