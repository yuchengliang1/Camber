import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
import itertools, json
from pathlib import Path
import numpy as np
A = Path(str(_RESULTS / '01_training_validation/full_and_ablations'))
B = Path(str(_RESULTS / '02_model_comparison/eegnet_broad'))
O = Path(str(_RESULTS / '08_complementarity'))
O.mkdir(parents=True, exist_ok=True)
keys = ['eegnet_eeg', 'eegnet_meg', 'eegnet_decision', 'eegnet_feature_concat']
rows = []
fulls = []
bs = []
for sid in range(29101, 29113):
    with np.load(A / f'sub{sid - 29100:02d}/hierarchical_pair_anchor_v25_full/oof_predictions.npz') as z:
        y = z['y']
        full = z['prediction']
        block = z['block']
    with np.load(B / f'sub{sid - 29100:02d}/oof_predictions.npz') as z:
        assert np.array_equal(z['y'], y) and np.array_equal(z['block'], block)
        preds = {k: z[k + '_prediction'] for k in keys}
    rows.append({'subject': sid, 'full': 100 * np.mean(full == y), **{k: 100 * np.mean(preds[k] == y) for k in keys}})
    fulls.append(full)
    bs.append((y, preds))
rng = np.random.default_rng(20261025)
boot = rng.integers(0, 12, (10000, 12))
sign = np.array(list(itertools.product((-1, 1), repeat=12)))
results = []
for k in keys:
    vals = np.array([r[k] for r in rows])
    d = np.array([r['full'] - r[k] for r in rows])
    p = np.mean(np.abs((sign * d).mean(1)) >= abs(d.mean()) - 1e-12)
    results.append({'method': k, 'mean_accuracy_percent': float(vals.mean()), 'full_minus_pp': float(d.mean()), 'ci95_pp': np.quantile(d[boot].mean(1), [0.025, 0.975]).tolist(), 'subjects_full_higher': int((d > 0).sum()), 'exact_p': float(p)})
order = np.argsort([r['exact_p'] for r in results])
running = 0
for (rank, idx) in enumerate(order):
    running = max(running, (4 - rank) * results[idx]['exact_p'])
    results[idx]['holm_p_4'] = min(1.0, running)
both = rescued = fwin = bwin = 0
for (full, (y, p)) in zip(fulls, bs):
    wrong = (p['eegnet_eeg'] != y) & (p['eegnet_meg'] != y)
    both += int(wrong.sum())
    rescued += int((wrong & (full == y)).sum())
    fwin += int(((full == y) & (p['eegnet_feature_concat'] != y)).sum())
    bwin += int(((full != y) & (p['eegnet_feature_concat'] == y)).sum())
out = {'status': 'PASS', 'n_subjects': 12, 'n_trials': 1800, 'full_mean_accuracy_percent': float(np.mean([r['full'] for r in rows])), 'methods': results, 'both_broad_unimodal_wrong_total': both, 'full_rescues_both_broad_wrong_total': rescued, 'full_correct_broad_concat_wrong_total': fwin, 'broad_concat_correct_full_wrong_total': bwin, 'subject_results': rows}
(O / 'broad_comparison_summary.json').write_text(json.dumps(out, indent=2))
print(json.dumps({k: v for (k, v) in out.items() if k != 'subject_results'}, indent=2))
