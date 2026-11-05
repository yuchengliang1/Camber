from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
BASE = Path(str(_RESULTS / '01_training_validation/full_and_ablations'))
DATA = Path(str(_PACKAGE / 'data'))
VARIANT = 'hierarchical_pair_anchor_v25_full'
EPOCHS = (80, 120, 160)

def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}

def score(probs: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    p = probs.reshape(-1, 3)
    target = y.reshape(-1)
    return (float(100 * np.mean(p.argmax(axis=1) == target)), float(-np.log(np.clip(p[np.arange(len(target)), target], 1e-12, 1)).mean()))

def exact_signflip_p(deltas: np.ndarray) -> float:
    n = len(deltas)
    observed = abs(float(deltas.mean()))
    null = np.array([abs(float(np.mean(deltas * np.array([1 if mask & 1 << i else -1 for i in range(n)])))) for mask in range(1 << n)])
    return float(np.mean(null >= observed - 1e-12))

def bootstrap_ci(deltas: np.ndarray, seed: int=291012) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(deltas), size=(20000, len(deltas)))
    return np.percentile(deltas[draws].mean(axis=1), [2.5, 97.5]).astype(float).tolist()

def classification_summary(y: np.ndarray, p: np.ndarray) -> dict:
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (y, p.argmax(axis=1)), 1)
    recall = np.divide(np.diag(cm), cm.sum(axis=1), out=np.zeros(3, float), where=cm.sum(axis=1) > 0)
    precision = np.divide(np.diag(cm), cm.sum(axis=0), out=np.zeros(3, float), where=cm.sum(axis=0) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3, float), where=precision + recall > 0)
    return {'confusion_matrix': cm.tolist(), 'class_recall_percent': (100 * recall).tolist(), 'macro_recall_percent': float(100 * recall.mean()), 'macro_f1': float(f1.mean())}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--result-root', type=Path, default=_RESULTS / '05_nested_selection')
    args = ap.parse_args()
    root = args.result_root
    manifest = json.loads((root / 'manifest.json').read_text())
    configs = manifest['candidates']
    rows = []
    subject_rows = []
    selected_names = Counter()
    selected_epochs = Counter()
    (all_y, all_tuned, all_fixed) = ([], [], [])
    for subject in range(29101, 29113):
        group = f'sub{subject - 29100:02d}'
        dataset = read_npz(DATA / group / 'paired_mi.npz')
        split = json.loads((_PACKAGE / 'data' / group / 'split.json').read_text())
        tuned = read_npz(root / group / 'oof_predictions.npz')
        fixed = read_npz(BASE / group / VARIANT / 'oof_predictions.npz')
        y = dataset['y'].astype(np.int64)
        if not np.array_equal(tuned['y'], y) or not np.array_equal(fixed['y'], y):
            raise AssertionError(f'Label mismatch: {group}')
        if not np.array_equal(tuned['block'], dataset['block']):
            raise AssertionError(f'Block mismatch: {group}')
        (tuned_p, fixed_p) = (tuned['probability'], fixed['probability'])
        if tuned_p.shape != (150, 3) or not np.isfinite(tuned_p).all() or (not np.allclose(tuned_p.sum(axis=1), 1, atol=1e-05)):
            raise AssertionError(f'Malformed tuned probabilities: {group}')
        coverage = np.zeros(150, dtype=int)
        for fold in range(5):
            split_fold = split['folds'][fold]
            outer_idx = np.asarray(split_fold['test_indices'], dtype=np.int64)
            inner_file = read_npz(root / group / f'fold_{fold}' / 'inner_predictions.npz')
            inner_p = inner_file['probabilities']
            inner_idx = inner_file['validation_indices']
            inner_y = inner_file['labels']
            if inner_p.shape != (len(configs), len(EPOCHS), 4, 30, 3):
                raise AssertionError(f'Inner probability shape: {group} fold {fold}')
            if not np.isfinite(inner_p).all() or not np.allclose(inner_p.sum(axis=-1), 1, atol=1e-05):
                raise AssertionError(f'Inner probability malformed: {group} fold {fold}')
            for (inner_fold, record) in enumerate(split_fold['inner']):
                idx = np.asarray(record['validation_indices'], dtype=np.int64)
                train = np.asarray(record['train_indices'], dtype=np.int64)
                if not np.array_equal(inner_idx[inner_fold], idx) or not np.array_equal(inner_y[inner_fold], y[idx]):
                    raise AssertionError(f'Inner indices/labels changed: {group} fold {fold}')
                if np.intersect1d(outer_idx, np.union1d(train, idx)).size:
                    raise AssertionError(f'Outer leakage: {group} fold {fold}')
            selection = json.loads((root / group / f'fold_{fold}' / 'selection.json').read_text())
            rebuilt = []
            for (ci, cfg) in enumerate(configs):
                for (ei, epoch) in enumerate(EPOCHS):
                    (acc, ce) = score(inner_p[ci, ei], inner_y)
                    rebuilt.append((ci, epoch, acc, ce))
            if len(selection['scores']) != len(rebuilt):
                raise AssertionError(f'Score count: {group} fold {fold}')
            for (item, (ci, epoch, acc, ce)) in zip(selection['scores'], rebuilt):
                if item['candidate_index'] != ci or item['epochs'] != epoch or abs(item['inner_accuracy_percent'] - acc) > 1e-06 or (abs(item['inner_cross_entropy'] - ce) > 1e-06):
                    raise AssertionError(f'Inner score mismatch: {group} fold {fold}')
            best = min(rebuilt, key=lambda x: (-x[2], x[3], x[1], x[0]))
            chosen = selection['selected']
            if chosen['candidate_index'] != best[0] or chosen['epochs'] != best[1]:
                raise AssertionError(f'Selection mismatch: {group} fold {fold}')
            outer = read_npz(root / group / f'fold_{fold}' / 'outer' / 'predictions.npz')
            spec = json.loads((root / group / f'fold_{fold}' / 'outer' / 'specification.json').read_text())
            if not np.array_equal(outer['indices'], outer_idx) or not np.array_equal(outer['y'], y[outer_idx]):
                raise AssertionError(f'Outer indices/labels changed: {group} fold {fold}')
            if not np.allclose(outer['probability'], tuned_p[outer_idx], atol=0, rtol=0):
                raise AssertionError(f'OOF replay mismatch: {group} fold {fold}')
            if spec['config']['epochs'] != chosen['epochs'] or spec['config']['name'] != chosen['name']:
                raise AssertionError(f'Outer config mismatch: {group} fold {fold}')
            if np.intersect1d(spec['train_indices'], outer_idx).size:
                raise AssertionError(f'Outer train/test overlap: {group} fold {fold}')
            coverage[outer_idx] += 1
            selected_names[chosen['name']] += 1
            selected_epochs[chosen['epochs']] += 1
            rows.append({'subject': subject, 'fold': fold, 'held_out_block': split_fold['held_out_block'], 'selected_config': chosen['name'], 'selected_epochs': chosen['epochs'], 'inner_accuracy_percent': chosen['inner_accuracy_percent'], 'outer_accuracy_percent': score(outer['probability'], y[outer_idx])[0], 'fixed_outer_accuracy_percent': score(fixed_p[outer_idx], y[outer_idx])[0]})
        if not np.all(coverage == 1):
            raise AssertionError(f'Outer coverage: {group}')
        subject_rows.append({'subject': subject, 'tuned_accuracy_percent': score(tuned_p, y)[0], 'fixed_accuracy_percent': score(fixed_p, y)[0]})
        all_y.append(y)
        all_tuned.append(tuned_p)
        all_fixed.append(fixed_p)
    diff = np.array([r['tuned_accuracy_percent'] - r['fixed_accuracy_percent'] for r in subject_rows])
    combined_y = np.concatenate(all_y)
    combined_tuned = np.concatenate(all_tuned)
    combined_fixed = np.concatenate(all_fixed)
    result = {'audit_status': 'PASS', 'subjects': len(subject_rows), 'outer_folds': len(rows), 'outer_predictions': len(combined_y), 'candidate_count': len(configs), 'checkpoint_epochs': list(EPOCHS), 'tuned_mean_subject_accuracy_percent': float(np.mean([r['tuned_accuracy_percent'] for r in subject_rows])), 'fixed_mean_subject_accuracy_percent': float(np.mean([r['fixed_accuracy_percent'] for r in subject_rows])), 'tuned_minus_fixed_pp': float(diff.mean()), 'paired_subject_bootstrap_ci95_pp': bootstrap_ci(diff), 'paired_subject_exact_signflip_p_two_sided': exact_signflip_p(diff), 'tuned_classification': classification_summary(combined_y, combined_tuned), 'fixed_classification': classification_summary(combined_y, combined_fixed), 'tuned_correct_fixed_wrong': int(np.sum((combined_tuned.argmax(1) == combined_y) & (combined_fixed.argmax(1) != combined_y))), 'fixed_correct_tuned_wrong': int(np.sum((combined_fixed.argmax(1) == combined_y) & (combined_tuned.argmax(1) != combined_y))), 'selected_config_counts': dict(selected_names), 'selected_epoch_counts': {str(k): int(v) for (k, v) in selected_epochs.items()}}
    (root / 'independent_audit_and_comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    for (name, data) in (('fold_comparison.csv', rows), ('subject_comparison.csv', subject_rows)):
        with (root / name).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
