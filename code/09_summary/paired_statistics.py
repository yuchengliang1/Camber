from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
import csv
import itertools
import json
from pathlib import Path
import numpy as np
ROOT = _RESULTS / '09_summary'
FAMILIES = {'architecture_and_selection': ['Matched_concat', 'No_path_controller', 'Uniform_temporal', 'Fixed_branch_gate', 'Fixed_anchor_mixing', 'No_Anchor_from_scratch', 'Learned_static_temporal', 'Learned_static_branch', 'Learned_static_both'], 'eegnet_benchmarks': ['Band_EEGNet_EEG', 'Band_EEGNet_MEG', 'Band_EEGNet_decision', 'Band_EEGNet_concat', 'Broad_EEGNet_EEG', 'Broad_EEGNet_MEG', 'Broad_EEGNet_decision', 'Broad_EEGNet_concat']}

def exact_signflip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    permuted = np.abs((signs * values[None, :]).mean(axis=1))
    return float(np.count_nonzero(permuted >= observed - 1e-12) / len(permuted))

def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.zeros(len(pvalues))
    running = 0.0
    for (rank, idx) in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()

def main() -> None:
    with (ROOT / 'subject_accuracy.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12 and sorted((int(row['subject']) for row in rows)) == list(range(29101, 29113))
    full = np.asarray([float(row['Full']) for row in rows])
    rng = np.random.default_rng(20261025)
    boot = rng.integers(0, len(rows), (10000, len(rows)))
    result = {'status': 'PASS', 'unit': '12 subject-paired outer-fold OOF accuracies', 'families': {}}
    for (family, names) in FAMILIES.items():
        entries = []
        for name in names:
            values = np.asarray([float(row[name]) for row in rows])
            delta = full - values
            entries.append({'comparator': name, 'mean_accuracy_percent': float(values.mean()), 'full_minus_comparator_pp': float(delta.mean()), 'bootstrap_ci95_pp': np.quantile(delta[boot].mean(axis=1), [0.025, 0.975]).tolist(), 'subjects_full_higher': int(np.count_nonzero(delta > 0)), 'exact_signflip_two_sided_p': exact_signflip(delta)})
        corrected = holm([x['exact_signflip_two_sided_p'] for x in entries])
        for (entry, p) in zip(entries, corrected):
            entry['holm_p_within_family'] = p
        result['families'][family] = entries
    (ROOT / 'paired_comparisons.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: len(v) for (k, v) in result['families'].items()}))
if __name__ == '__main__':
    main()
