import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from postprocess_remote import exact, holm
ROOT = _PACKAGE
OUT = _RESULTS / '09_summary'
LABELS = {'Matched_concat': 'Matched concatenation', 'No_path_controller': 'No path controller', 'Uniform_temporal': 'Uniform temporal weights', 'Fixed_branch_gate': 'Fixed branch gate', 'Fixed_anchor_mixing': 'Fixed anchor mixing', 'No_Anchor_from_scratch': 'No anchor, retrained', 'Learned_static_temporal': 'Learned static temporal weights', 'Learned_static_branch': 'Learned static branch weights', 'Learned_static_both': 'Both learned static'}

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.results_dir
    summary = read(out / 'method_summary.json')
    tests = read(out / 'paired_comparisons.json')
    assert summary['status'] == tests['status'] == 'PASS'
    with (out / 'subject_accuracy.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert sorted((int(row['subject']) for row in rows)) == list(range(29101, 29113))
    full = np.array([float(row['Full']) for row in rows])
    for (name, record) in summary['methods'].items():
        values = np.array([float(row[name]) for row in rows])
        assert np.isclose(values.mean(), record['mean_subject_accuracy_percent'], atol=1e-10)
    for (family, records) in tests['families'].items():
        raw = []
        for record in records:
            values = np.array([float(row[record['comparator']]) for row in rows])
            delta = full - values
            raw.append(exact(delta))
            assert np.isclose(delta.mean(), record['full_minus_comparator_pp'], atol=1e-10)
            assert np.isclose(raw[-1], record['exact_signflip_two_sided_p'], atol=1e-12)
        assert np.allclose(holm(raw), [r['holm_p_within_family'] for r in records], rtol=0, atol=1e-12)
    checked = {'status': 'PASS', 'participants': 12, 'method_means_checked': len(summary['methods']), 'paired_tests_checked': sum((len(rows) for rows in tests['families'].values()))}
    records = tests['families']['architecture_and_selection']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    (fig, ax) = plt.subplots(figsize=(9.1, 5.1))
    for (i, row) in enumerate(records):
        mean = row['full_minus_comparator_pp']
        (lo, hi) = row['bootstrap_ci95_pp']
        p = row['holm_p_within_family']
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt='o', color='#245b84' if p < 0.05 else '#899096', capsize=3)
        ax.text(1.02, i, f'{p:.4f}', transform=ax.get_yaxis_transform(), va='center', fontsize=8)
    ax.set_yticks(range(len(records)), [LABELS[row['comparator']] for row in records])
    ax.invert_yaxis()
    ax.axvline(0, color='#888', ls='--')
    ax.set_xlim(-1, 26)
    ax.set_xlabel('Full minus comparator accuracy (percentage points)')
    ax.set_title('Paired component and selection controls | 12 participants', loc='left')
    ax.text(1.02, 1.02, 'Holm p', transform=ax.transAxes, fontsize=8)
    ax.grid(axis='x', color='#e6e6e6')
    ax.set_axisbelow(True)
    fig.text(0.02, 0.025, 'Intervals: pointwise 95% participant bootstrap. P values: exact paired sign flips, Holm correction across 9 controls.', fontsize=8)
    fig.tight_layout(rect=(0, 0.065, 1, 0.98))
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(out / 'figures' / ('paired_control_effects.' + ext), dpi=250)
    plt.close(fig)
    print(json.dumps(checked))
if __name__ == '__main__':
    main()
