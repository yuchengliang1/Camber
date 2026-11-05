from pathlib import Path
import csv
import json
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PACKAGE, RESULTS
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def write_table(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(({key: row[key] for key in columns} for row in rows))

def main():
    final = RESULTS / '09_summary'
    with (final / 'subject_accuracy.csv').open(newline='', encoding='utf-8-sig') as stream:
        source = list(csv.DictReader(stream))
    rows = [{**row, 'subject': f"sub{int(row['subject']) - 29100:02d}"} for row in source]
    comparison = ['Full', 'Matched_concat', 'V25_EEG_only', 'V25_MEG_only', 'Band_EEGNet_EEG', 'Band_EEGNet_MEG', 'Band_EEGNet_decision', 'Band_EEGNet_concat', 'Broad_EEGNet_EEG', 'Broad_EEGNet_MEG', 'Broad_EEGNet_decision', 'Broad_EEGNet_concat']
    ablation = ['Full', 'No_path_controller', 'Uniform_temporal', 'Fixed_branch_gate', 'Fixed_anchor_mixing', 'No_Anchor_from_scratch', 'Learned_static_temporal', 'Learned_static_branch', 'Learned_static_both']
    methods = json.loads((final / 'method_summary.json').read_text())['methods']
    paired = json.loads((final / 'paired_comparisons.json').read_text())['families']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    for (category, names) in [('02_model_comparison', comparison), ('03_ablation', ablation)]:
        directory = RESULTS / category
        write_table(directory / 'subject_accuracy.csv', rows, ['subject'] + names)
        stats = []
        for name in names:
            record = methods[name]
            (lower, upper) = record['accuracy_ci95_percent']
            stats.append({'model': name, 'accuracy_percent': record['mean_subject_accuracy_percent'], 'ci95_lower_percent': lower, 'ci95_upper_percent': upper})
        write_table(directory / 'accuracy_summary.csv', stats, list(stats[0]))
        tests = [row for family in paired.values() for row in family if row['comparator'] in names]
        (directory / 'paired_comparisons.json').write_text(json.dumps(tests, indent=2) + '\n')
        (fig, ax) = plt.subplots(figsize=(8.8, 0.39 * len(names) + 1.8))
        means = np.array([s['accuracy_percent'] for s in stats])
        lower = np.array([s['ci95_lower_percent'] for s in stats])
        upper = np.array([s['ci95_upper_percent'] for s in stats])
        ax.barh(range(len(names)), means, color=['#b45545' if name == 'Full' else '#648798' for name in names], height=0.64)
        ax.errorbar(means, range(len(names)), xerr=np.vstack([means - lower, upper - means]), fmt='none', color='#333333', capsize=3)
        ax.set_yticks(range(len(names)), [name.replace('_', ' ') for name in names])
        ax.invert_yaxis()
        ax.set(xlabel='Accuracy (%)', xlim=(0, 100))
        ax.grid(axis='x', color='#e4e4e4', alpha=0.7)
        ax.set_axisbelow(True)
        for (i, mean) in enumerate(means):
            ax.text(max(mean, upper[i]) + 1.2, i, f'{mean:.2f}', va='center', fontsize=8)
        fig.tight_layout()
        (directory / 'figures').mkdir(exist_ok=True)
        for extension in ('png', 'pdf', 'svg'):
            fig.savefig(directory / 'figures' / f'accuracy_comparison.{extension}', dpi=240)
        plt.close(fig)
    for p in final.glob('parameter*'):
        if p.is_file():
            shutil.copy2(p, RESULTS / '04_parameter_sensitivity' / p.name)
    for p in (final / 'figures').iterdir():
        category = None
        if p.stem in ('sensor_overlay', 'smooth_spatial_maps', 'regional_effects'):
            category = '06_spatial_interpretability'
        elif 'frozen' in p.stem:
            category = '07_frozen_interventions'
        elif 'sensitivity' in p.stem or 'parameter' in p.stem:
            category = '04_parameter_sensitivity'
        elif p.stem == 'paired_control_effects':
            category = '03_ablation'
        if category:
            directory = RESULTS / category / 'figures'
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, directory / p.name)
    splits = []
    for i in range(1, 13):
        subject = f'sub{i:02d}'
        spec = json.loads((PACKAGE / 'data' / subject / 'split.json').read_text())
        for outer in spec['folds']:
            for inner in outer['inner']:
                splits.append({'subject': subject, 'outer_fold': outer['fold'], 'test_block': outer['held_out_block'], 'inner_fold': inner['inner_fold'], 'validation_block': inner['validation_block'], 'train_trials': len(inner['train_indices']), 'validation_trials': len(inner['validation_indices']), 'outer_train_trials': len(outer['train_indices']), 'test_trials': len(outer['test_indices'])})
    write_table(RESULTS / '01_training_validation' / 'fold_sizes.csv', splits, list(splits[0]))
    print(json.dumps({'status': 'PASS', 'subjects': len(rows), 'split_rows': len(splits)}))
if __name__ == '__main__':
    main()
