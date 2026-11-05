import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
from pathlib import Path
import json
import numpy as np
ROOT = Path(str(_RESULTS / '01_training_validation/full_and_ablations'))
OUT = Path(str(_RESULTS / '08_complementarity'))
FULL = 'hierarchical_pair_anchor_v25_full'
VARIANTS = {'full': FULL, 'eeg': 'hierarchical_pair_anchor_v25_eeg_only', 'meg': 'hierarchical_pair_anchor_v25_meg_only', 'concat': 'hierarchical_pair_anchor_v25_matched_concat', 'fixed_branch': 'hierarchical_pair_anchor_v25_fixed_branch_gate', 'fixed_anchor': 'hierarchical_pair_anchor_v25_fixed_anchor_mixing'}

def read(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}

def stat(y, probs):
    return probs.argmax(1) == y

def pct(x):
    return float(100 * np.mean(x))

def main():
    OUT.mkdir(exist_ok=False)
    rows = []
    detailed = []
    for sid in range(29101, 29113):
        group = ROOT / f'sub{sid - 29100:02d}'
        records = {name: read(group / variant / 'oof_predictions.npz') for (name, variant) in VARIANTS.items()}
        y = records['full']['y']
        block = records['full']['block']
        assert len(y) == 150 and all((np.array_equal(z['y'], y) and np.array_equal(z['block'], block) and z['covered'].all() for z in records.values()))
        correct = {name: stat(y, z['probability']) for (name, z) in records.items()}
        both_wrong = ~correct['eeg'] & ~correct['meg']
        any_right = correct['eeg'] | correct['meg']
        row = {'subject': sid, 'n_trials': 150, 'accuracy_percent': {name: pct(v) for (name, v) in correct.items()}, 'both_unimodal_wrong_n': int(both_wrong.sum()), 'full_rescues_both_wrong_n': int((correct['full'] & both_wrong).sum()), 'full_rescues_both_wrong_among_both_wrong_percent': pct(correct['full'][both_wrong]) if both_wrong.any() else None, 'full_wrong_when_any_unimodal_right_n': int((~correct['full'] & any_right).sum()), 'full_correct_when_any_unimodal_right_n': int((correct['full'] & any_right).sum()), 'full_correct_concat_wrong_n': int((correct['full'] & ~correct['concat']).sum()), 'concat_correct_full_wrong_n': int((correct['concat'] & ~correct['full']).sum())}
        rows.append(row)
        for i in range(150):
            detailed.append({'subject': sid, 'trial_index0': i, 'block': int(block[i]), 'y': int(y[i]), **{name + '_prediction': int(records[name]['probability'][i].argmax()) for name in VARIANTS}})
    (OUT / 'subject_summary.json').write_text(json.dumps(rows, indent=2))
    (OUT / 'trial_predictions.json').write_text(json.dumps(detailed, separators=(',', ':')))
    pooled = {'subjects': 12, 'trials': 1800, 'mean_subject_accuracy_percent': {name: float(np.mean([r['accuracy_percent'][name] for r in rows])) for name in VARIANTS}, 'total_both_unimodal_wrong_n': sum((r['both_unimodal_wrong_n'] for r in rows)), 'total_full_rescues_both_wrong_n': sum((r['full_rescues_both_wrong_n'] for r in rows)), 'total_full_wrong_when_any_unimodal_right_n': sum((r['full_wrong_when_any_unimodal_right_n'] for r in rows)), 'total_full_correct_concat_wrong_n': sum((r['full_correct_concat_wrong_n'] for r in rows)), 'total_concat_correct_full_wrong_n': sum((r['concat_correct_full_wrong_n'] for r in rows))}
    rng = np.random.default_rng(20261025)
    ix = rng.integers(0, 12, size=(10000, 12))
    for key in ('full_rescues_both_wrong_n', 'full_wrong_when_any_unimodal_right_n'):
        values = np.array([r[key] / 150 * 100 for r in rows])
        pooled[key + '_mean_subject_trial_percent'] = float(values.mean())
        pooled[key + '_ci95_percent'] = np.quantile(values[ix].mean(1), [0.025, 0.975]).tolist()
    (OUT / 'summary.json').write_text(json.dumps(pooled, indent=2))
    print(json.dumps(pooled, indent=2))
if __name__ == '__main__':
    main()
