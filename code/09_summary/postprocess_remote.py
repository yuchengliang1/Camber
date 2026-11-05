import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
from pathlib import Path
import argparse
import csv
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
import tarfile
import numpy as np
ROOT = Path(str(_PACKAGE))
REF = Path(str(_PACKAGE))
MAIN = _RESULTS / '01_training_validation/full_and_ablations'
FINAL = _RESULTS / '09_summary'
SPATIAL = _RESULTS / '06_spatial_interpretability'
REGION = SPATIAL / 'common_orientation_projection_assumed/matched_null64'
FULL = 'hierarchical_pair_anchor_v25_full'

def read(p):
    return json.loads(Path(p).read_text())

def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2) + '\n')

def rows(p):
    with p.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def exact(v):
    v = np.asarray(v, float)
    signs = np.asarray(list(itertools.product((-1, 1), repeat=len(v))))
    return float(np.mean(np.abs((signs * v).mean(1)) >= abs(v.mean()) - 1e-12))

def holm(p):
    ix = np.argsort(p)
    out = np.zeros(len(p))
    best = 0
    for (rank, j) in enumerate(ix):
        best = max(best, (len(p) - rank) * p[j])
        out[j] = min(best, 1)
    return out.tolist()

def components():
    result = []
    checked = 0
    for sid in range(29101, 29113):
        g = f'sub{sid - 29100:02d}'
        with np.load(_PACKAGE / 'data' / g / 'paired_mi.npz', allow_pickle=False) as z:
            y = z['y']
            block = z['block']
        summary = read(MAIN / g / 'summary.json')
        for (variant, values) in summary['variants'].items():
            with np.load(MAIN / g / variant / 'oof_predictions.npz', allow_pickle=False) as z:
                assert np.array_equal(z['y'], y) and np.array_equal(z['block'], block) and z['covered'].all()
                assert abs(float(100 * np.mean(z['prediction'] == y)) - values['oof']['accuracy_percent']) < 1e-08
            for f in range(5):
                fit = MAIN / g / variant / f'fold_{f}' / 'outer'
                spec = read(fit / 'specification.json')
                assert not set(spec['train_indices']) & set(spec['evaluation_indices'])
                assert spec['evaluation_indices'] == np.flatnonzero(block == f).tolist()
                with np.load(fit / 'predictions.npz', allow_pickle=False) as z:
                    assert np.array_equal(z['y'], y[z['indices']])
                checked += 1
        anchor = np.zeros(150, dtype=int)
        structured = anchor.copy()
        for f in range(5):
            with np.load(MAIN / g / FULL / f'fold_{f}' / 'outer/predictions.npz', allow_pickle=False) as z:
                anchor[z['indices']] = z['anchor_probability'].argmax(1)
                structured[z['indices']] = z['structured_probability'].argmax(1)
        result.append({'subject': sid, 'anchor_only_accuracy_percent': float(100 * np.mean(anchor == y)), 'posthoc_structured_component_accuracy_percent': float(100 * np.mean(structured == y))})
    write(MAIN / 'anchor_component_summary_20261025.json', {'status': 'PASS', 'subject_results': result})

def regional_statistics():
    folds = rows(REGION / 'regional_occlusion_fold.csv')
    effects = {}
    for metric in ['predicted_logprob_drop', 'cross_entropy_increase', 'accuracy_drop']:
        entries = []
        for mod in ('eeg', 'meg'):
            for name in ['Anterior', 'Central AP', 'Posterior', 'Left', 'Midline', 'Right']:
                chosen = [r for r in folds if r['modality'] == mod and r['region'] == name]
                differences = []
                for sid in range(29101, 29113):
                    d = [r for r in chosen if int(r['subject']) == sid]
                    assert len([r for r in d if r['kind'] == 'region']) == 5
                    assert len([r for r in d if r['kind'] == 'matched_random']) == 5 * 64
                    differences.append(np.mean([float(r[metric]) for r in d if r['kind'] == 'region']) - np.mean([float(r[metric]) for r in d if r['kind'] == 'matched_random']))
                entries.append({'modality': mod, 'region': name, 'mean_excess': float(np.mean(differences)), 'subject_excess': differences, 'exact_two_sided_p': exact(differences)})
        for (row, p) in zip(entries, holm([r['exact_two_sided_p'] for r in entries])):
            row['holm_p_12_regions'] = p
        effects[metric] = entries
    write(REGION / 'regional_occlusion_multiplicity.json', {'matched_null_draws': 64, 'results': effects})

def frozen_statistics(source=None, destination=None):
    source = Path(source) if source is not None else _RESULTS / '07_frozen_interventions'
    destination = Path(destination) if destination is not None else source / 'paired_statistics.json'
    summary = read(source / 'summary.json')
    assert summary['status'] == 'PASS' and summary['subjects'] == 12
    groups = [read(source / f'sub{sid - 29100:02d}_summary.json') for sid in range(29101, 29113)]
    assert [int(row['subject']) for row in groups] == list(range(29101, 29113))
    boot = np.random.default_rng(20261026).integers(0, 12, size=(10000, 12))
    names = sorted(set(summary['mean_subject_scores']) - {'clean'})
    endpoints = {}
    for metric in ('accuracy_percent', 'cross_entropy'):
        clean = np.array([row['condition_scores']['clean'][metric] for row in groups])
        entries = []
        for name in names:
            values = np.array([row['condition_scores'][name][metric] for row in groups])
            assert np.isclose(values.mean(), summary['mean_subject_scores'][name][metric], atol=1e-08)
            damage = clean - values if metric == 'accuracy_percent' else values - clean
            entries.append({'condition': name, 'mean_score': float(values.mean()), 'mean_damage': float(damage.mean()), 'damage_ci95': np.quantile(damage[boot].mean(1), [0.025, 0.975]).tolist(), 'exact_two_sided_p': exact(damage), 'subject_damage': damage.tolist()})
        for (row, p) in zip(entries, holm([r['exact_two_sided_p'] for r in entries])):
            row['holm_p_8_conditions'] = p
        endpoints[metric] = entries
        assert np.allclose([row['condition_scores']['pair_joint_realigned'][metric] for row in groups], clean, rtol=0, atol=1e-07)
    report = {'status': 'PASS', 'unit': '12 paired participant means; repeats averaged within participant', 'repeats': summary['repeats'], 'damage_units': {'accuracy_percent': 'percentage points', 'cross_entropy': 'natural-log units'}, 'results': endpoints}
    write(destination, report)
    return report

def statistics():
    shutil.copy2(_RESULTS / '07_frozen_interventions/summary.json', FINAL / 'frozen_interventions_summary.json')
    subprocess.run([sys.executable, str(_PACKAGE / 'code/09_summary/paired_statistics.py')], check=True)
    groups = rows(FINAL / 'subject_accuracy.csv')
    parameters = rows(FINAL / 'parameter_sensitivity_subject.csv')
    by = {int(r['subject']): r for r in groups}
    names = sorted({r['config'] for r in parameters if r['config'] != 'fixed_global12_baseline'})
    results = []
    for name in names:
        d = np.array([float(r['accuracy_percent']) - float(by[int(r['subject'])]['Full']) for r in parameters if r['config'] == name])
        results.append({'configuration': name, 'mean_difference_pp': float(d.mean()), 'exact_p': exact(d)})
    for (row, p) in zip(results, holm([r['exact_p'] for r in results])):
        row['holm_p_8_settings'] = p
    write(FINAL / 'parameter_multiplicity.json', {'unit': 'paired group', 'results': results})
    regional_statistics()
    frozen_statistics()
    with np.load(SPATIAL / 'spatial_audit_arrays.npz', allow_pickle=False) as z:
        occlusion = z['test_occlusion']
    from scipy.stats import spearmanr
    stability = {}
    for (i, mod) in enumerate(('eeg', 'meg')):
        correlations = [float(spearmanr(occlusion[a, i], occlusion[b, i]).statistic) for a in range(12) for b in range(a + 1, 12)]
        stability[mod] = {'pair_count': 66, 'mean_pairwise_spearman': float(np.mean(correlations))}
    write(SPATIAL / 'spatial_stability_summary.json', stability)

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = FINAL / 'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name):
        for extension in ('png', 'svg', 'pdf'):
            fig.savefig(out / (name + '.' + extension), dpi=250, bbox_inches='tight')
        plt.close(fig)
    methods = read(FINAL / 'method_summary.json')['methods']
    params = read(FINAL / 'parameter_sensitivity_summary.json')['configs']
    (fig, axes) = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    lists = [['Full', 'Broad_EEGNet_EEG', 'Broad_EEGNet_MEG', 'Broad_EEGNet_decision', 'Broad_EEGNet_concat', 'Band_EEGNet_EEG', 'Band_EEGNet_MEG', 'Band_EEGNet_decision', 'Band_EEGNet_concat'], ['Full', 'Matched_concat', 'No_path_controller', 'Uniform_temporal', 'Fixed_branch_gate', 'Fixed_anchor_mixing', 'Learned_static_temporal', 'Learned_static_branch', 'Learned_static_both', 'No_Anchor_from_scratch']]
    for (ax, names, title) in zip(axes, lists, ['EEGNet and fusion comparisons', 'Components and selection controls']):
        means = np.array([methods[n]['mean_subject_accuracy_percent'] for n in names])
        ci = np.array([methods[n]['accuracy_ci95_percent'] for n in names])
        ax.barh(range(len(names)), means, color=['#245b84' if n == 'Full' else '#94a9b5' for n in names])
        ax.errorbar(means, range(len(names)), xerr=np.vstack([means - ci[:, 0], ci[:, 1] - means]), fmt='none', color='#333', capsize=2)
        ax.set_yticks(range(len(names)), [n.replace('_', ' ') for n in names])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel('Mean outer-fold accuracy (%)')
        ax.set_title(title)
    save(fig, 'model_comparisons')
    (fig, axes) = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    names = list(params)
    values = [params[n]['mean_accuracy_percent'] for n in names]
    axes[0].barh(range(len(names)), values, color='#7197a9')
    axes[0].set_yticks(range(len(names)), [n.replace('fixed_global12_', '') for n in names])
    axes[0].invert_yaxis()
    axes[0].axvline(methods['Full']['mean_subject_accuracy_percent'], color='#b3564b', ls='--')
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel('Accuracy (%)')
    axes[0].set_title('One-factor sensitivity')
    data = rows(FINAL / 'subject_accuracy.csv')
    axes[1].bar(range(12), [float(r['Full']) for r in data], color='#648798')
    axes[1].set_xticks(range(12), ['sub' + r['subject'][-2:] for r in data])
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Subject accuracy')
    save(fig, 'parameters_and_subjects')
    spatial_figures()
    frozen_figures()

def frozen_figures(source=None, output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source = Path(source) if source is not None else _RESULTS / '07_frozen_interventions'
    output = Path(output) if output is not None else FINAL / 'figures'
    output.mkdir(parents=True, exist_ok=True)
    summary = read(source / 'summary.json')
    tests = read(source / 'paired_statistics.json')
    labels = {'clean': 'Clean Full', 'selection_shuffle': 'Selection shuffle', 'controller_shuffle': 'Controller shuffle', 'branch_shuffle': 'Branch shuffle', 'pair_meg_shuffle': 'MEG pair-content shuffle', 'pair_eeg_shuffle': 'EEG pair-content shuffle', 'both_shuffle': 'Selection + pair shuffle', 'pair_joint_unaligned': 'Joint pair shuffle', 'pair_joint_realigned': 'Joint shuffle + realignment'}
    names = list(labels)
    results = {r['condition']: r for r in tests['results']['accuracy_percent']}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    (fig, axes) = plt.subplots(1, 2, figsize=(12.4, 5.4), layout='constrained')
    means = [summary['mean_subject_scores'][name]['accuracy_percent'] for name in names]
    colors = ['#245b84' if name == 'clean' else '#488574' if name == 'pair_joint_realigned' else '#8b9ca8' for name in names]
    axes[0].barh(range(len(names)), means, color=colors)
    for (i, mean) in enumerate(means):
        axes[0].text(mean + 0.6, i, f'{mean:.2f}', va='center', fontsize=8)
    axes[0].set(yticks=range(len(names)), yticklabels=[labels[n] for n in names], xlim=(0, 90), xlabel='Mean accuracy (%)', title='Frozen model: 100 repeats')
    axes[0].invert_yaxis()
    for (i, name) in enumerate(names[1:]):
        row = results[name]
        mean = row['mean_damage']
        (lo, hi) = row['damage_ci95']
        axes[1].errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt='o', color=colors[i + 1], capsize=3)
    axes[1].axvline(0, color='#888', ls='--')
    axes[1].set(yticks=range(len(names) - 1), yticklabels=[labels[n] + (' *' if results[n]['holm_p_8_conditions'] < 0.05 else '') for n in names[1:]], xlabel='Clean minus intervention (percentage points)', title='Paired damage across 12 participants')
    axes[1].invert_yaxis()
    fig.supxlabel('Intervals: pointwise 95% participant bootstrap. * Holm p < 0.05 across all 8 comparisons, including realignment.', fontsize=8)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(output / ('frozen_interventions.' + ext), dpi=250, bbox_inches='tight')
    plt.close(fig)

def spatial_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from scipy.interpolate import CloughTocher2DInterpolator
    out = FINAL / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name):
        for extension in ('png', 'svg', 'pdf'):
            fig.savefig(out / (name + '.' + extension), dpi=250, bbox_inches='tight')
        plt.close(fig)
    channels = read(_PACKAGE / 'data/channel_positions.json')['channels']
    positions = {}
    for mod in ('eeg', 'meg'):
        pos = sorted([r for r in channels if r['modality'] == mod], key=lambda r: r['array_channel_index0'])
        ids = np.array([r['array_channel_index0'] for r in pos if r['has_display_position']], int)
        xy = np.array([r['display_xy_head_radius'] for r in pos if r['has_display_position']], float)
        positions[mod] = (ids, xy)

    def head(ax):
        ax.add_patch(Circle((0, 0), 1, fill=False, color='#333', lw=1))
        ax.plot([-0.1, 0, 0.1], [1, 1.12, 1], color='#333', lw=1)
        for sign in (-1, 1):
            ax.plot(np.array([1, 1.07, 1.07, 1]) * sign, [0.12, 0.12, -0.12, -0.12], color='#333', lw=1)
        ax.set(xlim=(-1.12, 1.12), ylim=(-1.1, 1.16), aspect='equal')
        ax.axis('off')
    (fig, ax) = plt.subplots(figsize=(5, 5))
    head(ax)
    for (mod, marker, color) in [('eeg', 'o', '#2166ac'), ('meg', '^', '#b35806')]:
        (_, xy) = positions[mod]
        ax.scatter(xy[:, 0], xy[:, 1], marker=marker, s=23, facecolors='none', edgecolors=color, label=mod.upper(), lw=0.8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)
    ax.set_title('EEG and MEG sensor positions')
    save(fig, 'sensor_overlay')
    with np.load(SPATIAL / 'spatial_audit_arrays.npz', allow_pickle=False) as z:
        occ = z['test_occlusion'].mean(0)
        power = z['power_class'].mean(0)
    ranges = power.max(2) - power.min(2)
    (fig, axes) = plt.subplots(2, 3, figsize=(11, 7), layout='constrained')
    grid = np.linspace(-1, 1, 220)
    (gx, gy) = np.meshgrid(grid, grid)
    for (i, mod) in enumerate(('eeg', 'meg')):
        (ids, xy) = positions[mod]
        (unique, inverse) = np.unique(xy, axis=0, return_inverse=True)
        for (j, (values, title)) in enumerate([(occ[i], 'Channel sensitivity'), (ranges[i, 0], 'Task mu class range (dB)'), (ranges[i, 1], 'Task beta class range (dB)')]):
            ax = axes[i, j]
            head(ax)
            v = values[ids]
            means = np.array([v[inverse == k].mean() for k in range(len(unique))])
            zz = CloughTocher2DInterpolator(unique, means)(gx, gy)
            zz[gx * gx + gy * gy > 1] = np.nan
            if j == 0:
                limit = max(abs(occ).max(), 1e-06)
                (lo, hi) = (-limit, limit)
                cmap = 'RdBu_r'
            else:
                (lo, hi) = (0, max(ranges[:, j - 1].max(), 1e-06))
                cmap = 'YlOrRd'
            im = ax.contourf(gx, gy, zz, levels=np.linspace(lo, hi, 129), vmin=lo, vmax=hi, cmap=cmap, extend='both')
            ax.scatter(xy[:, 0], xy[:, 1], s=3, color='#333', alpha=0.6)
            ax.set_title(mod.upper() + ' | ' + title)
            fig.colorbar(im, ax=ax, shrink=0.65, pad=0.015, ticks=np.linspace(lo, hi, 5))
    fig.supxlabel('Smooth interpolation within sensor coverage. Channel sensitivity: predicted-class log-probability drop.', fontsize=8)
    save(fig, 'smooth_spatial_maps')
    records = rows(REGION / 'regional_occlusion_summary.csv')
    sig = read(REGION / 'regional_occlusion_multiplicity.json')['results']
    (fig, axes) = plt.subplots(1, 3, figsize=(15, 5.2), layout='constrained')
    for (ax, metric) in zip(axes, ['predicted_logprob_drop', 'cross_entropy_increase', 'accuracy_drop']):
        scale = 100 if metric == 'accuracy_drop' else 1
        means = scale * np.array([float(r[metric + '_excess']) for r in records])
        low = scale * np.array([float(r[metric + '_excess_ci_low']) for r in records])
        high = scale * np.array([float(r[metric + '_excess_ci_high']) for r in records])
        tests = {(s['modality'], s['region']): s for s in sig[metric]}
        assert all((np.isclose(float(r[metric + '_excess']), tests[r['modality'], r['region']]['mean_excess']) for r in records))
        ax.errorbar(means, range(12), xerr=np.vstack([means - low, high - means]), fmt='o', ms=4, color='#44697d', capsize=2)
        ax.axvline(0, color='#888', ls='--')
        ax.set_yticks(range(12), [r['modality'].upper() + ' ' + r['region'] + (' *' if tests[r['modality'], r['region']]['holm_p_12_regions'] < 0.05 else '') for r in records])
        ax.invert_yaxis()
        ax.set_title(metric.replace('_', ' '))
        ax.set_xlabel('Region minus matched random' + (' (pp)' if metric == 'accuracy_drop' else ' (log units)'))
    fig.suptitle('Regional effects: pointwise 95% participant-bootstrap intervals; * Holm p < 0.05 within endpoint')
    save(fig, 'regional_effects')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['components', 'statistics', 'figures'])
    a = p.parse_args()
    globals()[a.stage]()
    print(json.dumps({'status': 'PASS', 'stage': a.stage}), flush=True)
