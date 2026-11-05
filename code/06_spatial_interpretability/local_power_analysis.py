import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
from pathlib import Path
import json
import numpy as np
from scipy.stats import spearmanr
import local_power_probes as probe
DATA = Path(str(_PACKAGE / 'data'))
RUN = Path(str(_RESULTS / '01_training_validation/full_and_ablations'))
OUT = Path(str(_RESULTS / '06_spatial_interpretability/local_power'))
FULL = 'hierarchical_pair_anchor_v25_full'

def read(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}

def write(p, x):
    p.write_text(json.dumps(x, indent=2, sort_keys=True))

def main():
    OUT.mkdir(exist_ok=False)
    write(OUT / 'manifest.json', {'bands_hz': probe.BANDS_HZ, 'windows': 8, 'outer_folds': 5, 'inner_folds': 4, 'model': 'Full'})
    allrows = []
    for sid in range(29101, 29113):
        g = f'sub{sid - 29100:02d}'
        d = read(DATA / g / 'paired_mi.npz')
        feat = probe.extract_features(d['eeg'], d['meg'])
        foldrows = []
        for fold in range(5):
            q = probe.fit_fold(feat, d['y'], d['block'], fold)
            test = q['test_indices']
            fit = read(RUN / g / FULL / f'fold_{fold}' / 'outer' / 'predictions.npz')
            assert np.array_equal(test, fit['indices']) and np.array_equal(d['y'][test], fit['y'])
            weights = fit['local_weights']
            assert weights.shape == (30, 8, 4)
            for (modality, branch) in ((0, 0), (1, 1)):
                score = q['inner_score'][modality].mean(axis=0)
                selected = np.argsort(-score, kind='stable')[:2]
                model = weights[:, :, branch].mean(axis=0)
                rho = float(spearmanr(score, model).statistic)
                model_top = np.argsort(-model, kind='stable')[:2]
                overlap = int(len(set(selected) & set(model_top)))
                prob = np.take(q['prob_test'][:, modality], selected, axis=2).mean(axis=(1, 2))
                accuracy = float(100 * np.mean(prob.argmax(1) == d['y'][test]))
                foldrows.append({'subject': sid, 'fold': fold, 'modality': probe.MODALITIES[modality], 'spearman_time_score_vs_weight': rho, 'top2_overlap_n': overlap, 'probe_test_accuracy_percent': accuracy, 'probe_inner_score_by_window': score.tolist(), 'model_mean_weight_by_window': model.tolist(), 'probe_selected_windows': selected.tolist(), 'model_selected_windows': model_top.tolist(), 'fit_count': q['fit_count']})
            print(json.dumps({'subject': sid, 'fold': fold, 'status': 'PASS'}), flush=True)
        write(OUT / f'{g}.json', foldrows)
        allrows.extend(foldrows)
        print(json.dumps({'subject': sid, 'status': 'complete'}), flush=True)
    summary = {'status': 'PASS', 'subjects': 12, 'folds_per_subject': 5, 'rows': len(allrows), 'modalities': {}}
    rng = np.random.default_rng(20261025)
    for mod in probe.MODALITIES:
        rr = [r for r in allrows if r['modality'] == mod]
        by_subject = np.array([np.mean([r['spearman_time_score_vs_weight'] for r in rr if r['subject'] == sid]) for sid in range(29101, 29113)])
        boots = rng.integers(0, 12, size=(10000, 12))
        summary['modalities'][mod] = {'mean_subject_spearman': float(by_subject.mean()), 'ci95': np.quantile(by_subject[boots].mean(1), [0.025, 0.975]).tolist(), 'subjects_positive': int(np.sum(by_subject > 0)), 'mean_top2_overlap_n': float(np.mean([r['top2_overlap_n'] for r in rr])), 'mean_probe_test_accuracy_percent': float(np.mean([r['probe_test_accuracy_percent'] for r in rr]))}
    write(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
if __name__ == '__main__':
    main()
