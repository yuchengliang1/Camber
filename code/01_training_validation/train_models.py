import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
from runtime import resolve_device
from pathlib import Path
import argparse, json, sys, hashlib, importlib.util
import numpy as np, torch
CODE = Path(str(_PACKAGE / 'code/00_shared/models'))
sys.path.insert(0, str(CODE))
import run_nested_hierarchical_pair_anchor_v25 as ex
import hierarchical_pair_anchor_models_v25 as base_model
from make_splits import make_splits
SYMS = ex._symbols(base_model)
VARIANTS = SYMS['VARIANTS']
FULL = SYMS['FULL_VARIANT']
FIXED = SYMS['PRIMARY_ABLATIONS']['adaptive_anchor_probability_mixing']

def load_data(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}

def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')

def custom_oof(root, variant, data, folds):
    probs = np.full((len(data['y']), 3), np.nan, np.float32)
    preds = np.full(len(data['y']), -1, np.int64)
    covered = np.zeros(len(data['y']), bool)
    fold_rows = []
    for (f, _) in enumerate(folds):
        p = root / variant / f'fold_{f}' / 'outer' / 'predictions.npz'
        with np.load(p, allow_pickle=False) as z:
            idx = np.asarray(z['indices'], np.int64)
            prob = np.asarray(z['probability'], np.float32)
            pred = np.asarray(z['prediction'], np.int64)
        if covered[idx].any():
            raise RuntimeError('overlap ' + str(p))
        covered[idx] = True
        probs[idx] = prob
        preds[idx] = pred
        fold_rows.append({'fold': f, 'n_test': int(len(idx)), 'accuracy_percent': float(100 * np.mean(pred == data['y'][idx]))})
    if not covered.all():
        raise RuntimeError('coverage incomplete ' + variant)
    np.savez_compressed(root / variant / 'oof_predictions.npz', probability=probs, prediction=preds, y=data['y'], block=data['block'], covered=covered)
    return {'oof': ex.metrics(data['y'], probs), 'outer_folds': fold_rows}

def run_subject(subject, data_root, out_root, cfg, base_seed, device, model_module):
    g = f'sub{subject - 29100:02d}'
    data = load_data(data_root / g / 'paired_mi.npz')
    split_path = data_root / g / 'split.json'
    split = (json.loads(split_path.read_text()) if split_path.exists()
             else make_splits(data['block'], data['y']))
    folds = split['folds']
    root = out_root / g
    root.mkdir(parents=True, exist_ok=False)
    provenance = {}
    write(root / 'protocol.json', {'subject': subject, 'configuration': cfg, 'variants': list(VARIANTS), 'outer_folds': len(folds), 'inner_folds_per_outer': 4})
    for (f, outer) in enumerate(folds):
        inner_paths = []
        inner_records = []
        for inner in outer['inner']:
            j = int(inner['inner_fold'])
            tr = np.asarray(inner['train_indices'], np.int64)
            val = np.asarray(inner['validation_indices'], np.int64)
            path = root / FULL / f'fold_{f}' / f'inner_{j}'
            seed = int(base_seed + subject * 100000 + f * 1000 + int(inner['validation_block']) * 10)
            result = ex._fit_once(path, variant=FULL, config=cfg, train_indices=tr, evaluation_indices=val, seed=seed, eeg_raw=data['eeg'], meg_raw=data['meg'], y=data['y'], device=device, provenance=provenance, model_mod=model_module)
            inner_paths.append(path)
            inner_records.append({'inner_fold': j, 'validation_block': int(inner['validation_block']), 'seed': seed, 'outcome': result['outcome']})
        selected = ex._select_fixed_anchor_lambda(inner_paths, cfg)
        chosen = float(selected['selected_lambda'])
        write(root / FULL / f'fold_{f}' / 'selection.json', {'fold': f, 'selector_variant': FULL, 'fixed_anchor_control_selection': selected, 'inner_records': inner_records})
        for variant in VARIANTS:
            tr = np.asarray(outer['train_indices'], np.int64)
            te = np.asarray(outer['test_indices'], np.int64)
            path = root / variant / f'fold_{f}' / 'outer'
            seed = int(base_seed + subject * 100000 + f * 1000 + 999)
            result = ex._fit_once(path, variant=variant, config=cfg, train_indices=tr, evaluation_indices=te, seed=seed, eeg_raw=data['eeg'], meg_raw=data['meg'], y=data['y'], device=device, provenance=provenance, model_mod=model_module, fixed_anchor_lambda=chosen if variant == FIXED else None)
            if variant != FULL:
                write(root / variant / f'fold_{f}' / 'selection.json', {'fold': f, 'selector_variant': FULL, 'fixed_anchor_control_lambda': chosen})
        print(json.dumps({'subject': subject, 'fold': f, 'status': 'outer_complete'}), flush=True)
    summary = {'subject': subject, 'variants': {}}
    for variant in VARIANTS:
        summary['variants'][variant] = custom_oof(root, variant, data, folds)
    full_acc = summary['variants'][FULL]['oof']['accuracy_percent']
    summary['full_minus_variant_percent'] = {v: float(full_acc - summary['variants'][v]['oof']['accuracy_percent']) for v in VARIANTS if v != FULL}
    write(root / 'summary.json', summary)
    print(json.dumps({'subject': subject, 'status': 'complete', 'full_accuracy_percent': full_acc}), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--out-root', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--subjects', required=True)
    ap.add_argument('--base-seed', type=int, default=27029001)
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    cfg = ex._read_training_config(a.config)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model_module = base_model
    subjects = [int(x) for x in a.subjects.split(',')]
    a.out_root.mkdir(parents=True, exist_ok=False)
    write(a.out_root / 'experiment_manifest.json', {'subjects': subjects, 'data_root': str(a.data_root), 'config': cfg, 'base_seed': a.base_seed, 'variants': list(VARIANTS)})
    device = resolve_device(a.device)
    for subject in subjects:
        run_subject(subject, a.data_root, a.out_root, cfg, a.base_seed, device, model_module)
    print(json.dumps({'status': 'PASS', 'subjects': subjects}), flush=True)
if __name__ == '__main__':
    main()
