import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
from pathlib import Path
import csv, json
import numpy as np
BASE = Path(str(_PACKAGE))
MAIN = _RESULTS / '01_training_validation/full_and_ablations'
NO = _RESULTS / '03_ablation/no_anchor'
STATIC = _RESULTS / '03_ablation/learned_static'
BAND = _RESULTS / '02_model_comparison/eegnet_band'
BROAD = _RESULTS / '02_model_comparison/eegnet_broad'
SENS = _RESULTS / '04_parameter_sensitivity/primary'
EXTRA = _RESULTS / '04_parameter_sensitivity/extended'
OUT = _RESULTS / '09_summary'
FULL = 'hierarchical_pair_anchor_v25_full'
MAIN_NAMES = {'Full': FULL, 'Matched_concat': 'hierarchical_pair_anchor_v25_matched_concat', 'No_path_controller': 'hierarchical_pair_anchor_v25_no_bidirectional_path_controller', 'Uniform_temporal': 'hierarchical_pair_anchor_v25_uniform_temporal', 'Fixed_branch_gate': 'hierarchical_pair_anchor_v25_fixed_branch_gate', 'Fixed_anchor_mixing': 'hierarchical_pair_anchor_v25_fixed_anchor_mixing', 'V25_EEG_only': 'hierarchical_pair_anchor_v25_eeg_only', 'V25_MEG_only': 'hierarchical_pair_anchor_v25_meg_only'}
STATIC_NAMES = {'Learned_static_temporal': 'r12_learned_static_temporal', 'Learned_static_branch': 'r12_learned_static_branch', 'Learned_static_both': 'r12_learned_static_both'}
BAND_NAMES = {'Band_EEGNet_EEG': 'eegnet_band_eeg', 'Band_EEGNet_MEG': 'eegnet_band_meg', 'Band_EEGNet_decision': 'eegnet_band_decision', 'Band_EEGNet_concat': 'eegnet_band_feature_concat'}
BROAD_NAMES = {'Broad_EEGNet_EEG': 'eegnet_eeg', 'Broad_EEGNet_MEG': 'eegnet_meg', 'Broad_EEGNet_decision': 'eegnet_decision', 'Broad_EEGNet_concat': 'eegnet_feature_concat'}

def read(p):
    return json.loads(Path(p).read_text())

def archive(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}

def require():
    for p in (NO / 'summary.json', STATIC / 'summary.json', _RESULTS / '07_frozen_interventions/summary.json', _RESULTS / '06_spatial_interpretability/local_power/summary.json'):
        if not p.exists() or read(p).get('status') != 'PASS':
            raise RuntimeError('Incomplete or failed ' + str(p))
    for sid in range(29101, 29113):
        for cfg in ('fixed_global12_lr018', 'fixed_global12_epochs160', 'fixed_global12_dropout030', 'fixed_global12_wd003'):
            if not (EXTRA / f'sub{sid - 29100:02d}' / cfg / 'summary.json').exists():
                raise RuntimeError('Missing extra sensitivity ' + str((sid, cfg)))

def main():
    require()
    anchor = {r['subject']: r for r in read(MAIN / 'anchor_component_summary_20261025.json')['subject_results']}
    rows = []
    parameters = []
    for sid in range(29101, 29113):
        group = f'sub{sid - 29100:02d}'
        data = archive(_PACKAGE / 'data' / f'{group}/paired_mi.npz')
        mf = read(MAIN / group / 'summary.json')
        sf = read(STATIC / group / 'summary.json')
        bf = read(BAND / group / 'summary.json')
        wf = read(BROAD / group / 'summary.json')
        row = {'subject': sid, 'n_trials': int(len(data['y']))}
        for (name, v) in MAIN_NAMES.items():
            row[name] = float(mf['variants'][v]['oof']['accuracy_percent'])
        row['No_Anchor_from_scratch'] = float(read(NO / group / 'summary.json')['accuracy_percent'])
        row['Saved_pretrained_Anchor_only'] = float(anchor[sid]['anchor_only_accuracy_percent'])
        row['Posthoc_structured_component'] = float(anchor[sid]['posthoc_structured_component_accuracy_percent'])
        for (name, v) in STATIC_NAMES.items():
            row[name] = float(sf['controls'][v]['accuracy_percent'])
        for (name, v) in BAND_NAMES.items():
            row[name] = float(bf['metrics'][v]['accuracy_percent'])
        for (name, v) in BROAD_NAMES.items():
            row[name] = float(wf['metrics'][v]['accuracy_percent'])
        for (roots, configs) in ((SENS, ('fixed_global12_baseline', 'fixed_global12_lr006', 'fixed_global12_epochs80', 'fixed_global12_dropout015', 'fixed_global12_wd0003')), (EXTRA, ('fixed_global12_lr018', 'fixed_global12_epochs160', 'fixed_global12_dropout030', 'fixed_global12_wd003'))):
            for cfg in configs:
                z = read(roots / group / cfg / 'summary.json')
                parameters.append({'subject': sid, 'config': cfg, 'accuracy_percent': float(z['metrics']['accuracy_percent']), 'learning_rate': float(z['config']['learning_rate']), 'epochs': int(z['config']['epochs']), 'dropout': float(z['config']['dropout']), 'weight_decay': float(z['config']['weight_decay'])})
        for oof_path in (NO / group / 'oof_predictions.npz', STATIC / group / 'r12_learned_static_both' / 'oof_predictions.npz', BAND / group / 'oof_predictions.npz', BROAD / group / 'oof_predictions.npz'):
            z = archive(oof_path)
            assert np.array_equal(z['y'], data['y']) and np.array_equal(z['block'], data['block']), oof_path
        assert abs(row['Full'] - next((p['accuracy_percent'] for p in parameters if p['subject'] == sid and p['config'] == 'fixed_global12_baseline'))) < 1e-08
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'subject_accuracy.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (OUT / 'parameter_sensitivity_subject.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(parameters[0]))
        w.writeheader()
        w.writerows(parameters)
    rng = np.random.default_rng(20261025)
    boot = rng.integers(0, 12, size=(10000, 12))
    method = {}
    for key in rows[0]:
        if key in ('subject', 'n_trials'):
            continue
        v = np.array([r[key] for r in rows])
        d = np.array([r['Full'] - r[key] for r in rows])
        method[key] = {'mean_subject_accuracy_percent': float(v.mean()), 'accuracy_ci95_percent': np.quantile(v[boot].mean(1), [0.025, 0.975]).tolist(), 'full_minus_method_pp': float(d.mean()), 'full_minus_ci95_pp': np.quantile(d[boot].mean(1), [0.025, 0.975]).tolist()}
    (OUT / 'method_summary.json').write_text(json.dumps({'status': 'PASS', 'n_subjects': 12, 'outer_test_trials': 1800, 'methods': method}, indent=2))
    names = sorted(set((p['config'] for p in parameters)))
    sensitivities = {}
    for cfg in names:
        p = np.array([next((x['accuracy_percent'] for x in parameters if x['subject'] == sid and x['config'] == cfg)) for sid in range(29101, 29113)])
        base = np.array([r['Full'] for r in rows])
        d = p - base
        conf = next((x for x in parameters if x['config'] == cfg))
        sensitivities[cfg] = {'learning_rate': conf['learning_rate'], 'epochs': conf['epochs'], 'dropout': conf['dropout'], 'weight_decay': conf['weight_decay'], 'mean_accuracy_percent': float(p.mean()), 'delta_vs_fixed_baseline_pp': float(d.mean()), 'delta_ci95_pp': np.quantile(d[boot].mean(1), [0.025, 0.975]).tolist()}
    (OUT / 'parameter_sensitivity_summary.json').write_text(json.dumps({'status': 'PASS', 'n_subjects': 12, 'configs': sensitivities}, indent=2))
    print(json.dumps({'status': 'PASS', 'output': str(OUT), 'methods': len(method), 'parameter_configs': len(sensitivities)}, indent=2), flush=True)
if __name__ == '__main__':
    main()
