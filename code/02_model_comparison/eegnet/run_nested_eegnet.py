from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
import argparse
import platform
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import eegnet_baselines as model_module
import run_eegnet_blocks as fixed_module
from eegnet_baselines import LEGACY_SOURCE_SHA256, METHOD_NAMES, MODEL_SETTINGS, FrozenConcatFeatureFusion, array_sha256, block_splits, classification_metrics, file_sha256, load_public_data, make_eegnet, normalize_trials, parameter_counts, self_test as model_self_test, state_sha256
from run_eegnet_blocks import _json, _npz, _resolve_device, _save_checkpoint, _seed, _train_classifier
CANDIDATE_EPOCHS = (10, 30, 60)
FUSION_HEAD_EPOCHS = 30

def make_nested_splits(block: np.ndarray) -> list[dict[str, Any]]:
    plan = []
    for (outer_block, (outer_train, outer_test)) in enumerate(block_splits(block)):
        inner = []
        for validation_block in range(5):
            if validation_block == outer_block:
                continue
            train = np.flatnonzero((block != outer_block) & (block != validation_block))
            validation = np.flatnonzero(block == validation_block)
            if (len(train), len(validation)) != (90, 30):
                raise ValueError('Inner split must contain90 training and30 validation trials')
            if np.intersect1d(train, outer_test).size or np.intersect1d(validation, outer_test).size or np.intersect1d(train, validation).size:
                raise ValueError('Outer test leakage / overlapping inner split')
            if not np.array_equal(np.sort(np.r_[train, validation]), outer_train):
                raise ValueError('Inner split must cover only the120 outer-training trials')
            inner.append({'validation_block': validation_block, 'train_indices': train, 'validation_indices': validation})
        validation_coverage = np.bincount(np.concatenate([x['validation_indices'] for x in inner]), minlength=len(block))
        expected = np.zeros(len(block), dtype=np.int64)
        expected[outer_train] = 1
        if not np.array_equal(validation_coverage, expected):
            raise ValueError('Inner validation must cover each outer-training trial once')
        plan.append({'outer_block': outer_block, 'train_indices': outer_train, 'test_indices': outer_test, 'inner': inner})
    return plan

def _serializable_plan(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{'outer_block': x['outer_block'], 'fold': x['outer_block'] + 1, 'train_indices': x['train_indices'].tolist(), 'test_indices': x['test_indices'].tolist(), 'inner': [{'validation_block': item['validation_block'], 'train_indices': item['train_indices'].tolist(), 'validation_indices': item['validation_indices'].tolist()} for item in x['inner']]} for x in plan]

def _fit_seed(base: int, subject: int, outer_block: int, modality: str, validation_block: int | None=None) -> int:
    modality_offset = {'eeg': 1, 'meg': 2, 'fusion': 3}[modality]
    if validation_block is None:
        return base + 1000000 * subject + 500000 + 1000 * outer_block + modality_offset
    return base + 1000000 * subject + 10000 * outer_block + 100 * validation_block + modality_offset

def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    result = classification_metrics(y, probability)
    result['cross_entropy'] = float(-np.log(np.maximum(probability[np.arange(len(y)), y.astype(np.int64)], 1e-12)).mean())
    return result

def select_budget(records: list[dict[str, Any]], expected_validation_blocks: list[int]) -> dict[str, Any]:
    expected = set(expected_validation_blocks)
    if len(expected) != 4:
        raise ValueError('Exactly four distinct inner validation blocks are required')
    if len(records) != len(CANDIDATE_EPOCHS) * 4:
        raise ValueError('Missing or duplicate inner candidate records')
    scores = []
    for epochs in CANDIDATE_EPOCHS:
        selected = [r for r in records if r['epochs'] == epochs]
        blocks = [r['validation_block'] for r in selected]
        if len(selected) != 4 or set(blocks) != expected or len(set(blocks)) != 4:
            raise ValueError(f'Candidate{epochs} requires all four distinct validation blocks')
        if any((r['validation_metrics']['n'] != 30 for r in selected)):
            raise ValueError('Each validation score must refer to30 trials')
        correct = sum((int(r['validation_metrics']['correct']) for r in selected))
        ce = float(np.mean([r['validation_metrics']['cross_entropy'] for r in selected]))
        if not np.isfinite(ce) or not 0 <= correct <= 120:
            raise ValueError('Invalid validation metrics')
        scores.append({'epochs': epochs, 'correct_total': correct, 'validation_total': 120, 'mean_validation_accuracy_percent': 100 * correct / 120, 'mean_validation_cross_entropy': ce, 'validation_blocks': sorted(blocks)})
    best = min(scores, key=lambda r: (-r['correct_total'], r['mean_validation_cross_entropy'], r['epochs']))
    return {'selected_epochs': best['epochs'], 'selected_score': best, 'candidate_scores': scores, 'outer_test_used': False}

def _fit_unimodal(normalized: np.ndarray, y: np.ndarray, train: np.ndarray, evaluate: np.ndarray, modality: str, epochs: int, seed: int, device: torch.device, destination: Path, context: dict[str, Any], evaluation_role: str) -> tuple[torch.nn.Module, np.ndarray, dict[str, Any]]:
    if evaluation_role not in ('inner_validation', 'outer_test'):
        raise ValueError('Invalid evaluation role')
    if np.intersect1d(train, evaluate).size:
        raise ValueError('Training/evaluation index overlap')
    destination.mkdir(parents=True, exist_ok=False)
    _seed(seed)
    model = make_eegnet(modality).to(device)
    initial_state = state_sha256(model)
    x_train = torch.from_numpy(normalized[train, None]).to(device)
    y_train = torch.as_tensor(y[train], dtype=torch.long, device=device)
    (history, order) = _train_classifier(model, x_train, y_train, epochs=epochs, seed=seed, lr=0.0008, weight_decay=0.001)
    with torch.no_grad():
        training_probability = torch.softmax(model(x_train), dim=1).cpu().numpy()
        x_eval = torch.from_numpy(normalized[evaluate, None]).to(device)
        probability = torch.softmax(model(x_eval), dim=1).cpu().numpy()
    checkpoint = destination / 'model.pt'
    _save_checkpoint(checkpoint, model, {'class': 'EEGNet', 'modality': modality, 'settings': MODEL_SETTINGS[modality]}, {**context, 'seed': seed, 'epochs': epochs, 'evaluation_role': evaluation_role, 'train_indices': train.tolist(), 'evaluation_indices': evaluate.tolist()})
    _npz(destination / 'indices.npz', train=train, evaluation=evaluate)
    _npz(destination / 'training_order.npz', relative_order=order, global_indices=train[order])
    _npz(destination / 'predictions.npz', y=y[evaluate], probability=probability, prediction=probability.argmax(axis=1), indices=evaluate)
    _npz(destination / 'training_predictions.npz', y=y[train], probability=training_probability, prediction=training_probability.argmax(axis=1), indices=train)
    result = {'modality': modality, 'epochs': epochs, 'seed': seed, 'n_train': len(train), 'n_evaluation': len(evaluate), 'evaluation_role': evaluation_role, 'training_metrics': _metrics(y[train], training_probability), 'evaluation_metrics': _metrics(y[evaluate], probability), 'initial_state_sha256': initial_state, 'final_state_sha256': state_sha256(model), 'parameter_counts': parameter_counts(model), 'batch_order_sha256': array_sha256(order), 'checkpoint_sha256': file_sha256(checkpoint), 'predictions_sha256': file_sha256(destination / 'predictions.npz')}
    _json(destination / 'training.json', {'seed': seed, 'epochs': epochs, 'history': history, 'fixed_final_epoch': True, 'early_stopping': False, 'test_used_for_training_or_selection': False})
    _json(destination / 'result.json', result)
    del x_train, y_train, x_eval
    return (model, probability, result)

def _fit_feature_head(models: dict[str, torch.nn.Module], normalized: dict[str, np.ndarray], y: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int, device: torch.device, destination: Path, context: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=False)
    _seed(seed)
    model = FrozenConcatFeatureFusion(models['eeg'], models['meg']).to(device)
    frozen_before = {key: state_sha256(models[key]) for key in ('eeg', 'meg')}
    x_train = {key: torch.from_numpy(normalized[key][train, None]).to(device) for key in ('eeg', 'meg')}
    with torch.no_grad():
        train_features = model.concatenate(x_train['eeg'], x_train['meg'])
        model.fit_feature_scaler(train_features)
    initial_hash = state_sha256(model)
    y_train = torch.as_tensor(y[train], dtype=torch.long, device=device)
    (history, order) = _train_classifier(model, train_features, y_train, epochs=FUSION_HEAD_EPOCHS, seed=seed, lr=0.0005, weight_decay=0.002, forward=model.logits_from_features, parameters=model.classifier.parameters())
    for key in ('eeg', 'meg'):
        if frozen_before[key] != state_sha256(models[key]):
            raise RuntimeError('Frozen backbone or BatchNorm changed during fusion-head training')
    with torch.no_grad():
        training_probability = torch.softmax(model.logits_from_features(train_features), dim=1).cpu().numpy()
        x_test = {key: torch.from_numpy(normalized[key][test, None]).to(device) for key in ('eeg', 'meg')}
        probability = torch.softmax(model(x_test['eeg'], x_test['meg']), dim=1).cpu().numpy()
    checkpoint = destination / 'model.pt'
    _save_checkpoint(checkpoint, model, {'class': 'FrozenConcatFeatureFusion', 'feature_dim': 360, 'encoder_settings': MODEL_SETTINGS}, {**context, 'seed': seed, 'epochs_additional_head': FUSION_HEAD_EPOCHS, 'train_indices': train.tolist(), 'test_indices': test.tolist()})
    _npz(destination / 'feature_scaler.npz', mean=model.feature_mean.cpu().numpy(), scale=model.feature_scale.cpu().numpy(), fit_indices=train)
    _npz(destination / 'training_order.npz', relative_order=order, global_indices=train[order])
    _npz(destination / 'predictions.npz', y=y[test], probability=probability, prediction=probability.argmax(axis=1), indices=test)
    _npz(destination / 'training_predictions.npz', y=y[train], probability=training_probability, prediction=training_probability.argmax(axis=1), indices=train)
    result = {'epochs_additional_head': FUSION_HEAD_EPOCHS, 'seed': seed, 'training_metrics': _metrics(y[train], training_probability), 'evaluation_metrics': _metrics(y[test], probability), 'initial_state_sha256': initial_hash, 'final_state_sha256': state_sha256(model), 'parameter_counts': parameter_counts(model), 'batch_order_sha256': array_sha256(order), 'frozen_backbone_state_sha256': frozen_before, 'frozen_backbones_unchanged': True, 'feature_scaler_fit_indices': train.tolist(), 'checkpoint_sha256': file_sha256(checkpoint), 'predictions_sha256': file_sha256(destination / 'predictions.npz')}
    _json(destination / 'training.json', {'seed': seed, 'history': history, 'head_epochs_predeclared': FUSION_HEAD_EPOCHS, 'head_epoch_selection': 'none', 'backbone_selection': 'separate inner-validation unimodal epoch selection', 'test_used_for_training_or_selection': False})
    _json(destination / 'result.json', result)
    del model, x_train, x_test, train_features, y_train
    return (probability, result)

def run(data_path: Path, subject: int, out_dir: Path, seed: int, device_name: str=None) -> dict[str, Any]:
    if seed < 0 or subject < 0:
        raise ValueError('seed and subject must be nonnegative')
    if out_dir.exists():
        raise FileExistsError(f'Refusing to overwrite an existing output path: {out_dir}')
    device = _resolve_device(device_name)
    data_hash = file_sha256(data_path)
    public = load_public_data(data_path, subject)
    plan = make_nested_splits(public['block'])
    (normalized, scales) = ({}, {})
    for modality in ('eeg', 'meg'):
        (normalized[modality], scales[modality]) = normalize_trials(public[modality])
    source_paths = {'run_nested_eegnet.py': Path(__file__), 'run_eegnet_blocks.py': Path(fixed_module.__file__), 'eegnet_baselines.py': Path(model_module.__file__)}
    source_hashes = {name: file_sha256(path) for (name, path) in source_paths.items()}
    seed_plan = [{'outer_block': outer['outer_block'], 'inner': [{'validation_block': inner['validation_block'], 'seeds_same_across_epoch_candidates': {m: _fit_seed(seed, subject, outer['outer_block'], m, inner['validation_block']) for m in ('eeg', 'meg')}} for inner in outer['inner']]} for outer in plan]
    protocol = {'subject': subject, 'data_path': str(data_path.resolve()), 'data_sha256': data_hash, 'public_array_sha256': {k: array_sha256(v) for (k, v) in public.items()}, 'candidate_epochs': list(CANDIDATE_EPOCHS), 'fusion_head_epochs': FUSION_HEAD_EPOCHS, 'splits': _serializable_plan(plan), 'seed': seed, 'seed_plan': seed_plan, 'inner_feature_head_fits': 0, 'training_fit_counts': {'inner_unimodal': 120, 'outer_unimodal_refits': 10, 'outer_feature_heads': 5, 'total': 135}, 'inner_budget_epochs_per_modality_per_outer': 4 * sum(CANDIDATE_EPOCHS), 'optimization_unimodal': {'optimizer': 'AdamW', 'lr': 0.0008, 'weight_decay': 0.001, 'batch_size': 30, 'label_smoothing': 0.04, 'scheduler': None}, 'optimization_head': {'optimizer': 'AdamW', 'lr': 0.0005, 'weight_decay': 0.002, 'batch_size': 30, 'label_smoothing': 0.04, 'scheduler': None}, 'encoder_settings': MODEL_SETTINGS, 'public_inputs_only': ['eeg', 'meg', 'y', 'block', 'fs', 'trial_id', 'optional subject_id'], 'outer_test_used_for_selection': False, 'early_stopping': False, 'last_epoch_checkpoint_per_candidate': True, 'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda, 'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(), 'cudnn': torch.backends.cudnn.version(), 'tf32': False, 'deterministic_algorithms': True}, 'created_utc': datetime.now(timezone.utc).isoformat()}
    out_dir.mkdir(parents=True, exist_ok=False)
    _json(out_dir / 'protocol.json', protocol)
    _npz(out_dir / 'trial_local_normalization_scales.npz', eeg=scales['eeg'], meg=scales['meg'], trial_id=public['trial_id'], block=public['block'])
    oof = {name: np.full((150, 3), np.nan, dtype=np.float32) for name in METHOD_NAMES}
    coverage = np.zeros(150, dtype=np.int64)
    all_results = []
    started = time.monotonic()
    for outer in plan:
        outer_block = outer['outer_block']
        (train, test) = (outer['train_indices'], outer['test_indices'])
        fold_dir = out_dir / f'fold_{outer_block + 1}'
        fold_dir.mkdir(exist_ok=False)
        _npz(fold_dir / 'outer_indices.npz', train=train, test=test, train_trial_id=public['trial_id'][train], test_trial_id=public['trial_id'][test])
        context = {'subject': subject, 'outer_block': outer_block, 'fold': outer_block + 1, 'data_sha256': data_hash}
        inner_records = {'eeg': [], 'meg': []}
        initial_hashes = {}
        for inner in outer['inner']:
            validation_block = inner['validation_block']
            for epochs in CANDIDATE_EPOCHS:
                for modality in ('eeg', 'meg'):
                    fit_seed = _fit_seed(seed, subject, outer_block, modality, validation_block)
                    destination = fold_dir / f'inner_validation_block_{validation_block}' / f'epochs_{epochs}' / modality
                    (model, probability, result) = _fit_unimodal(normalized[modality], public['y'], inner['train_indices'], inner['validation_indices'], modality, epochs, fit_seed, device, destination, {**context, 'validation_block': validation_block}, 'inner_validation')
                    key = (validation_block, modality)
                    if key in initial_hashes and initial_hashes[key] != result['initial_state_sha256']:
                        raise RuntimeError('Paired epoch candidates did not share the same initialization')
                    initial_hashes[key] = result['initial_state_sha256']
                    inner_records[modality].append({'epochs': epochs, 'validation_block': validation_block, 'validation_metrics': result['evaluation_metrics'], 'fit_result': str((destination / 'result.json').relative_to(fold_dir)), 'seed': fit_seed, 'initial_state_sha256': result['initial_state_sha256']})
                    print(json.dumps({'event': 'inner_complete', 'subject': subject, 'outer_block': outer_block, 'validation_block': validation_block, 'modality': modality, 'epochs': epochs, 'validation_accuracy_percent': result['evaluation_metrics']['accuracy_percent']}), flush=True)
                    del model, probability
                    torch.cuda.empty_cache()
        validation_blocks = [inner['validation_block'] for inner in outer['inner']]
        selection = {modality: select_budget(inner_records[modality], validation_blocks) for modality in ('eeg', 'meg')}
        _json(fold_dir / 'selection.json', {'outer_block': outer_block, 'selection': selection, 'inner_records': inner_records, 'outer_test_used': False, 'feature_head_epochs_fixed': FUSION_HEAD_EPOCHS})
        (models, probabilities, refit_results) = ({}, {}, {})
        for modality in ('eeg', 'meg'):
            selected_epochs = selection[modality]['selected_epochs']
            fit_seed = _fit_seed(seed, subject, outer_block, modality)
            (model, probability, result) = _fit_unimodal(normalized[modality], public['y'], train, test, modality, selected_epochs, fit_seed, device, fold_dir / f'outer_refit_{modality}', {**context, 'selection_sha256': file_sha256(fold_dir / 'selection.json')}, 'outer_test')
            models[modality] = model
            probabilities[f'eegnet_{modality}'] = probability
            refit_results[modality] = result
        (feature_probability, feature_result) = _fit_feature_head(models, normalized, public['y'], train, test, _fit_seed(seed, subject, outer_block, 'fusion'), device, fold_dir / 'outer_feature_concat', {**context, 'selected_backbone_epochs': {m: selection[m]['selected_epochs'] for m in ('eeg', 'meg')}, 'backbone_checkpoint_sha256': {m: refit_results[m]['checkpoint_sha256'] for m in ('eeg', 'meg')}, 'selection_sha256': file_sha256(fold_dir / 'selection.json')})
        probabilities['eegnet_feature_concat'] = feature_probability
        probabilities['eegnet_decision'] = 0.5 * probabilities['eegnet_eeg'] + 0.5 * probabilities['eegnet_meg']
        _json(fold_dir / 'decision_fusion.json', {'weights': {'eeg': 0.5, 'meg': 0.5}, 'trainable_parameters': 0, 'backbone_checkpoint_sha256': {m: refit_results[m]['checkpoint_sha256'] for m in ('eeg', 'meg')}})
        fold_metrics = {method: _metrics(public['y'][test], probability) for (method, probability) in probabilities.items()}
        for (method, probability) in probabilities.items():
            oof[method][test] = probability
        coverage[test] += 1
        _npz(fold_dir / 'outer_predictions.npz', y=public['y'][test], indices=test, trial_id=public['trial_id'][test], **probabilities)
        result = {'outer_block': outer_block, 'fold': outer_block + 1, 'selected_epochs': {m: selection[m]['selected_epochs'] for m in ('eeg', 'meg')}, 'metrics': fold_metrics, 'refit_results': refit_results, 'feature_head_result': feature_result, 'selection_sha256': file_sha256(fold_dir / 'selection.json'), 'outer_predictions_sha256': file_sha256(fold_dir / 'outer_predictions.npz')}
        _json(fold_dir / 'result.json', result)
        all_results.append(result)
        print(json.dumps({'event': 'outer_complete', 'subject': subject, 'outer_block': outer_block, 'selected_epochs': result['selected_epochs'], 'accuracy_percent': {name: value['accuracy_percent'] for (name, value) in fold_metrics.items()}}, ensure_ascii=False), flush=True)
        del models, model, feature_probability, probability
        torch.cuda.empty_cache()
    if not np.array_equal(coverage, np.ones(150, dtype=np.int64)):
        raise RuntimeError('Invalid outer prediction coverage')
    for probability in oof.values():
        _metrics(public['y'], probability)
    if file_sha256(data_path) != data_hash:
        raise RuntimeError('Input file changed during the run')
    if any((file_sha256(path) != source_hashes[name] for (name, path) in source_paths.items())):
        raise RuntimeError('Runner or dependency source changed during the run')
    arrays = {'y': public['y'], 'block': public['block'], 'trial_id': public['trial_id'], 'fold': public['block'] + 1, 'test_coverage': coverage, **oof}
    arrays.update({name + '_prediction': probability.argmax(axis=1) for (name, probability) in oof.items()})
    _npz(out_dir / 'oof_predictions.npz', **arrays)
    summary = {'complete': True, 'subject': subject, 'n': 150, 'outer_folds': 5, 'inner_validation_blocks_per_outer': 4, 'candidate_epochs': list(CANDIDATE_EPOCHS), 'feature_head_epochs_fixed': FUSION_HEAD_EPOCHS, 'protocol_sha256': file_sha256(out_dir / 'protocol.json'), 'data_sha256': data_hash, 'oof_sha256': file_sha256(out_dir / 'oof_predictions.npz'), 'elapsed_seconds': time.monotonic() - started, 'metrics': {name: _metrics(public['y'], probability) for (name, probability) in oof.items()}, 'fold_results': all_results, 'fit_counts': protocol['training_fit_counts'], 'outer_test_used_for_selection': False, 'completed_utc': datetime.now(timezone.utc).isoformat()}
    _json(out_dir / 'summary.json', summary)
    print(json.dumps({'event': 'complete', 'subject': subject, 'metrics': summary['metrics'], 'output': str(out_dir)}, ensure_ascii=False), flush=True)
    return summary

def self_test() -> dict[str, Any]:
    architecture = model_self_test()
    block = np.repeat(np.arange(5), 30)
    plan = make_nested_splits(block)
    assert len(plan) == 5
    all_test = []
    for outer in plan:
        assert len(outer['train_indices']) == 120 and len(outer['test_indices']) == 30
        all_test.extend(outer['test_indices'].tolist())
        assert len(outer['inner']) == 4
        for inner in outer['inner']:
            assert len(inner['train_indices']) == 90 and len(inner['validation_indices']) == 30
            assert len(set(block[inner['train_indices']])) == 3
            assert set(block[inner['validation_indices']]) == {inner['validation_block']}
            assert not np.intersect1d(outer['test_indices'], np.r_[inner['train_indices'], inner['validation_indices']]).size
    assert sorted(all_test) == list(range(150))
    permutation = np.random.default_rng(55).permutation(150)
    shuffled = make_nested_splits(block[permutation])
    for outer in shuffled:
        assert np.all(block[permutation][outer['test_indices']] == outer['outer_block'])
    records = [{'epochs': epochs, 'validation_block': b, 'validation_metrics': {'n': 30, 'correct': 18 if epochs == 10 else 20, 'cross_entropy': 0.7 if epochs == 30 else 0.6}} for epochs in CANDIDATE_EPOCHS for b in [1, 2, 3, 4]]
    assert select_budget(records, [1, 2, 3, 4])['selected_epochs'] == 60
    for record in records:
        record['validation_metrics']['cross_entropy'] = 0.7
    assert select_budget(records, [1, 2, 3, 4])['selected_epochs'] == 30
    for record in records:
        if record['epochs'] == 10:
            record['validation_metrics']['correct'] = 21
    assert select_budget(records, [1, 2, 3, 4])['selected_epochs'] == 10
    try:
        select_budget(records[:-1], [1, 2, 3, 4])
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete candidate results were accepted')
    seeds = []
    for outer in plan:
        for inner in outer['inner']:
            for modality in ('eeg', 'meg'):
                seeds.append(_fit_seed(20261020, 9001, outer['outer_block'], modality, inner['validation_block']))
        for modality in ('eeg', 'meg', 'fusion'):
            seeds.append(_fit_seed(20261020, 9001, outer['outer_block'], modality))
    assert len(seeds) == len(set(seeds))
    assert _resolve_device('cpu').type == 'cpu'
    json.dumps(_serializable_plan(plan))
    return {'status': 'PASS', 'device': 'CPU QA only', 'architecture': architecture, 'outer_folds': 5, 'inner_folds_per_outer': 4, 'outer_sizes': [120, 30], 'inner_sizes': [90, 30], 'candidate_epochs': list(CANDIDATE_EPOCHS), 'fixed_fusion_head_epochs': FUSION_HEAD_EPOCHS, 'fit_counts_per_subject': {'inner_unimodal': 120, 'outer_unimodal': 10, 'feature_heads': 5}}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--subject', type=int)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--device', default=None)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, ensure_ascii=False))
        return
    if args.data is None or args.subject is None or args.out_dir is None:
        parser.error('--data, --subject, --out-dir required unless --self-test')
    run(args.data, args.subject, args.out_dir, args.seed, args.device)
if __name__ == '__main__':
    main()
