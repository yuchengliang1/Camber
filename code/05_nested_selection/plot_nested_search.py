from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.result_root
    with (root / 'subject_comparison.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    audit = json.loads((root / 'independent_audit_and_comparison.json').read_text())
    subjects = [str(row['subject'])[-2:] for row in rows]
    fixed = np.array([float(row['fixed_accuracy_percent']) for row in rows])
    tuned = np.array([float(row['tuned_accuracy_percent']) for row in rows])
    config_counts = audit['selected_config_counts']
    epoch_counts = audit['selected_epoch_counts']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig = plt.figure(figsize=(12.2, 4.3), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.25, 0.65])
    ax = fig.add_subplot(grid[0, 0])
    x = np.arange(len(subjects))
    ax.plot(x, fixed, 'o-', color='#4B6475', lw=1.5, ms=4.5, label='Fixed configuration')
    ax.plot(x, tuned, 'o-', color='#B3564B', lw=1.5, ms=4.5, label='Inner-selected')
    ax.axhline(fixed.mean(), ls=':', lw=1, color='#4B6475', alpha=0.8)
    ax.axhline(tuned.mean(), ls=':', lw=1, color='#B3564B', alpha=0.8)
    ax.set_xticks(x, subjects)
    ax.set_xlabel('Group ID suffix')
    ax.set_ylabel('Outer-fold accuracy (%)')
    ax.set_ylim(max(0, min(fixed.min(), tuned.min()) - 6), min(100, max(fixed.max(), tuned.max()) + 6))
    ax.set_title('A  Paired accuracy across 12 groups', loc='left', fontweight='bold')
    ax.grid(axis='y', color='#e4e6e8', lw=0.7)
    ax.legend(frameon=False, loc='lower right')
    ax = fig.add_subplot(grid[0, 1])
    names = list(config_counts)
    counts = [config_counts[n] for n in names]
    short = [n.replace('lr', 'LR ').replace('_drop', ' / D ').replace('_wd', ' / WD ') for n in names]
    order = np.argsort(counts)
    ypos = np.arange(len(names))
    ax.barh(ypos, np.array(counts)[order], color='#7197A9')
    ax.set_yticks(ypos, np.array(short)[order])
    ax.set_xlim(0, max(counts) + 3)
    ax.set_xlabel('Outer folds selecting configuration')
    ax.set_title('B  Selected combinations', loc='left', fontweight='bold')
    for (i, j) in enumerate(order):
        ax.text(counts[j] + 0.2, i, str(counts[j]), va='center', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax = fig.add_subplot(grid[0, 2])
    epoch_labels = sorted(epoch_counts, key=int)
    epoch_values = [epoch_counts[e] for e in epoch_labels]
    ax.bar(epoch_labels, epoch_values, color='#A58B72')
    ax.set_xlabel('Selected epochs')
    ax.set_ylabel('Outer folds')
    ax.set_title('C  Training length', loc='left', fontweight='bold')
    for (i, value) in enumerate(epoch_values):
        ax.text(i, value + 0.5, str(value), ha='center', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    subtitle = f"Inner-selected {audit['tuned_mean_subject_accuracy_percent']:.2f}% vs fixed {audit['fixed_mean_subject_accuracy_percent']:.2f}%; paired difference {audit['tuned_minus_fixed_pp']:+.2f} pp"
    fig.suptitle(subtitle, fontsize=11, fontweight='bold')
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(root / f'nested_parameter_selection.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
if __name__ == '__main__':
    main()
