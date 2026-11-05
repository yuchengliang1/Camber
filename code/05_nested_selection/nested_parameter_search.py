from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
from runtime import resolve_device
import argparse
import hashlib
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
CODE = Path(str(_PACKAGE / 'code/00_shared/models'))
REPRO = Path(str(_PACKAGE / 'code/00_shared/config'))
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(REPRO))
import train_models as training
import run_nested_hierarchical_pair_anchor_v25 as ex
DATA = Path(str(_PACKAGE / 'data'))
DEFAULT_OUT = Path(str(_RESULTS / '05_nested_selection'))
LEARNING_RATES = (0.012, 0.018)
DROPOUTS = (0.0, 0.15)
WEIGHT_DECAYS = (0.003, 0.03)
CHECKPOINT_EPOCHS = (80, 120, 160)
BASE_SEED = 27029001

def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')

def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def candidates() -> list[dict]:
    base = json.loads((_PACKAGE / 'code/00_shared/config/main_config.json').read_text())['configs'][0]
    out = []
    for (lr, dropout, wd) in itertools.product(LEARNING_RATES, DROPOUTS, WEIGHT_DECAYS):
        cfg = dict(base)
        cfg.update(name=f'lr{lr:g}_drop{dropout:g}_wd{wd:g}', learning_rate=float(lr), dropout=float(dropout), weight_decay=float(wd), epochs=max(CHECKPOINT_EPOCHS))
        out.append(cfg)
    return out

def fit_inner(*, data: dict[str, np.ndarray], train_indices: np.ndarray, validation_indices: np.ndarray, config: dict, seed: int, model_module: object, device: torch.device) -> np.ndarray:
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError('Inner train/validation overlap')
    ex._seed(seed)
    eeg_scale = ex._fit_scale_raw(data['eeg'], train_indices)
    meg_scale = ex._fit_scale_raw(data['meg'], train_indices)
    eeg = torch.as_tensor(data['eeg'] / eeg_scale[None, :, None], dtype=torch.float32, device=device)
    meg = torch.as_tensor(data['meg'] / meg_scale[None, :, None], dtype=torch.float32, device=device)
    target = torch.as_tensor(data['y'], dtype=torch.long, device=device)
    symbols = ex._symbols(model_module)
    model = symbols['make_model'](symbols['FULL_VARIANT'], dropout=float(config['dropout'])).to(device)
    output = []
    with tempfile.TemporaryDirectory(prefix='g12_inner_', dir=None) as td:
        temp = Path(td)
        ex._fit_calibrations(model, eeg=eeg, meg=meg, train_indices=train_indices, eval_indices=validation_indices, fit_dir=temp, fixed_anchor_lambda=None, symbols=symbols)
        ex._pretrain_anchor(model, eeg=eeg, meg=meg, target=target, train_indices=train_indices, config=config, seed=seed, fit_dir=temp)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))
        order_generator = torch.Generator(device='cpu').manual_seed(seed + 31)
        for epoch in range(1, max(CHECKPOINT_EPOCHS) + 1):
            model.train()
            order = train_indices[torch.randperm(len(train_indices), generator=order_generator).numpy()]
            for start in range(0, len(order), int(config['batch_size'])):
                batch = torch.as_tensor(order[start:start + int(config['batch_size'])], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                log_probability = model(eeg[batch], meg[batch])
                loss = ex._smoothed_nll(log_probability, target[batch], float(config['label_smoothing'])).mean()
                loss.backward()
                optimizer.step()
            if epoch in CHECKPOINT_EPOCHS:
                (detail, expected) = ex._evaluate(model, eeg, meg, data['y'], validation_indices, tuple(symbols['DETAIL_KEYS']))
                if not np.array_equal(expected, data['y'][validation_indices]):
                    raise AssertionError('Validation labels changed')
                output.append(detail['probability'])
    return np.stack(output).astype(np.float32)

def score_grid(probabilities: np.ndarray, labels: np.ndarray) -> list[dict]:
    scores = []
    for (ci, config) in enumerate(candidates()):
        for (ei, epochs) in enumerate(CHECKPOINT_EPOCHS):
            p = probabilities[ci, ei].reshape(-1, 3)
            y = labels.reshape(-1)
            accuracy = float(np.mean(p.argmax(axis=1) == y) * 100)
            cross_entropy = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
            scores.append({'candidate_index': ci, 'name': config['name'], 'learning_rate': config['learning_rate'], 'dropout': config['dropout'], 'weight_decay': config['weight_decay'], 'epochs': epochs, 'inner_accuracy_percent': accuracy, 'inner_cross_entropy': cross_entropy, 'n_inner_predictions': len(y)})
    return scores

def one_subject(subject: int, out_root: Path, model_module: object, device: torch.device) -> None:
    group = f'sub{subject - 29100:02d}'
    root = out_root / group
    root.mkdir(parents=True, exist_ok=True)
    data = training.load_data(DATA / group / 'paired_mi.npz')
    splits = json.loads((_PACKAGE / 'data' / group / 'split.json').read_text())['folds']
    configs = candidates()
    full = ex._symbols(model_module)['FULL_VARIANT']
    provenance = {}
    all_probabilities = np.full((len(data['y']), 3), np.nan, dtype=np.float32)
    selected_rows = []
    for (fold, split) in enumerate(splits):
        fold_root = root / f'fold_{fold}'
        outer_dir = fold_root / 'outer'
        if (fold_root / 'selection.json').exists() and (outer_dir / 'result.json').exists():
            with np.load(outer_dir / 'predictions.npz', allow_pickle=False) as z:
                all_probabilities[z['indices']] = z['probability']
            selected_rows.append(json.loads((fold_root / 'selection.json').read_text())['selected'])
            continue
        n_inner = len(split['inner'])
        probabilities = np.empty((len(configs), len(CHECKPOINT_EPOCHS), n_inner, 30, 3), np.float32)
        validation_indices = np.empty((n_inner, 30), np.int64)
        labels = np.empty((n_inner, 30), np.int64)
        for (ci, cfg) in enumerate(configs):
            for (ji, inner) in enumerate(split['inner']):
                train = np.asarray(inner['train_indices'], dtype=np.int64)
                val = np.asarray(inner['validation_indices'], dtype=np.int64)
                if len(train) != 90 or len(val) != 30:
                    raise AssertionError('Unexpected inner fold size')
                seed = BASE_SEED + subject * 100000 + fold * 1000 + int(inner['validation_block']) * 10
                probabilities[ci, :, ji] = fit_inner(data=data, train_indices=train, validation_indices=val, config=cfg, seed=seed, model_module=model_module, device=device)
                validation_indices[ji] = val
                labels[ji] = data['y'][val]
            print(json.dumps({'subject': subject, 'fold': fold, 'candidate': cfg['name'], 'status': 'inner_complete'}), flush=True)
        scores = score_grid(probabilities, labels)
        selected = sorted(scores, key=lambda row: (-row['inner_accuracy_percent'], row['inner_cross_entropy'], row['epochs'], row['candidate_index']))[0]
        cfg = dict(configs[selected['candidate_index']])
        cfg['epochs'] = int(selected['epochs'])
        if fold_root.exists() and any(fold_root.iterdir()):
            raise FileExistsError(f'Incomplete existing fold; inspect before resuming: {fold_root}')
        fold_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fold_root / 'inner_predictions.npz', probabilities=probabilities, validation_indices=validation_indices, labels=labels)
        write_json(fold_root / 'selection.json', {'subject': subject, 'fold': fold, 'outer_test_block': int(split['held_out_block']), 'selection_source': 'four inner validation blocks only', 'scores': scores, 'selected': selected, 'inner_validation_blocks': [int(x['validation_block']) for x in split['inner']], 'inner_train_sizes': [len(x['train_indices']) for x in split['inner']]})
        train = np.asarray(split['train_indices'], dtype=np.int64)
        test = np.asarray(split['test_indices'], dtype=np.int64)
        seed = BASE_SEED + subject * 100000 + fold * 1000 + 999
        result = ex._fit_once(outer_dir, variant=full, config=cfg, train_indices=train, evaluation_indices=test, seed=seed, eeg_raw=data['eeg'], meg_raw=data['meg'], y=data['y'], device=device, provenance=provenance, model_mod=model_module)
        with np.load(outer_dir / 'predictions.npz', allow_pickle=False) as z:
            if not np.array_equal(z['indices'], test):
                raise AssertionError('Outer predictions do not match held-out block')
            all_probabilities[test] = z['probability']
        selected_rows.append(selected)
        print(json.dumps({'subject': subject, 'fold': fold, 'selected': cfg['name'], 'epochs': cfg['epochs'], 'outer_accuracy_percent': result['outcome']['accuracy_percent']}), flush=True)
    if not np.isfinite(all_probabilities).all():
        raise AssertionError('Incomplete outer predictions')
    np.savez_compressed(root / 'oof_predictions.npz', probability=all_probabilities, y=data['y'], block=data['block'])
    summary = {'subject': subject, 'n_outer_predictions': len(data['y']), 'metrics': ex.metrics(data['y'], all_probabilities), 'selected': selected_rows}
    write_json(root / 'summary.json', summary)
    print(json.dumps({'subject': subject, 'status': 'complete', 'accuracy_percent': summary['metrics']['accuracy_percent']}), flush=True)

def aggregate(out_root: Path) -> None:
    subjects = list(range(29101, 29113))
    summaries = [json.loads((out_root / f'sub{s - 29100:02d}' / 'summary.json').read_text()) for s in subjects]
    values = np.array([s['metrics']['accuracy_percent'] for s in summaries])
    selected = [row for summary in summaries for row in summary['selected']]
    aggregate_record = {'status': 'PASS', 'n_subjects': len(summaries), 'n_outer_folds': len(selected), 'mean_subject_accuracy_percent': float(values.mean()), 'subject_accuracy_percent': {str(s): float(v) for (s, v) in zip(subjects, values)}, 'selected_epoch_counts': {str(e): sum((row['epochs'] == e for row in selected)) for e in CHECKPOINT_EPOCHS}, 'selected_candidate_counts': {cfg['name']: sum((row['name'] == cfg['name'] for row in selected)) for cfg in candidates()}}
    write_json(out_root / 'aggregate.json', aggregate_record)
    print(json.dumps(aggregate_record), flush=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-root', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--subjects', default=','.join((str(s) for s in range(29101, 29113))))
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    if args.aggregate:
        aggregate(args.out_root)
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest = args.out_root / 'manifest.json'
    if not manifest.exists():
        write_json(manifest, {'data_root': str(DATA), 'base_seed': BASE_SEED, 'candidates': candidates(), 'checkpoint_epochs': list(CHECKPOINT_EPOCHS), 'outer': 'five leave-one-block-out folds per group', 'inner': 'four leave-one-block-out folds within each outer training set'})
    model_module = training.base_model
    subjects = [int(x) for x in args.subjects.split(',') if x]
    for subject in subjects:
        one_subject(subject, args.out_root, model_module, resolve_device())
if __name__ == '__main__':
    main()
