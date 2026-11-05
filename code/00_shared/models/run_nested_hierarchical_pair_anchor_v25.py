from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
import argparse
import ast
import hashlib
import importlib
import json
import math
import platform
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from training_utils import array_hash, metrics, seed_everything, sha256, state_hash
RUNNER_FAMILY = 'nested_lobo_v25_coordinate_isolated_r1'
PROTOCOL_SCHEMA_VERSION = 'nested_lobo_v25.1'
DETAIL_SCHEMA_VERSION = 'v25.1'
CONTROLLER_CALIBRATION_FILENAME = 'controller_calibration.npz'
ANCHOR_MIXING_CALIBRATION_FILENAME = 'anchor_mixing_calibration.npz'
ANCHOR_PRETRAIN_WEIGHT_FILENAME = 'anchor_pretraining_weights.npz'
PUBLIC_FIELDS = ('eeg', 'meg', 'y', 'block')

def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return payload

def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()

def _write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite immutable V25 artifact: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

def _save_npz_new(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite immutable V25 artifact: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)

def _symbols(model: Any) -> dict[str, Any]:
    required = ('MODEL_FAMILY', 'GRAPH_CONTRACT', 'FULL_VARIANT', 'VARIANTS', 'STRUCTURED_VARIANTS', 'SINGLE_VARIANTS', 'DETAIL_KEYS', 'make_model', 'parameter_count', 'trainable_parameter_count', 'self_test', 'PRIMARY_ABLATIONS', 'CONTROLLER_STATE_KEYS', 'ANCHOR_MIXING_STATE_KEYS', 'VARIANT_POLICY', 'REGISTRY_SHA256')
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise AttributeError(f'V25 model lacks runner contract: {missing}')
    result = {name: getattr(model, name) for name in required}
    expected_primary = {'bidirectional_path_controller', 'dual_expert_pair_interaction', 'dynamic_four_route_temporal_pooling', 'adaptive_four_head_aggregation', 'adaptive_anchor_probability_mixing'}
    if set(result['PRIMARY_ABLATIONS']) != expected_primary:
        raise ValueError('V25 must declare exactly five pre-specified primary ablations')
    (variants, structured, single) = (tuple(result['VARIANTS']), tuple(result['STRUCTURED_VARIANTS']), tuple(result['SINGLE_VARIANTS']))
    if not variants or result['FULL_VARIANT'] not in structured or set(variants) != set(structured) | set(single):
        raise ValueError('Invalid V25 variant partition')
    if len(set(result['PRIMARY_ABLATIONS'].values())) != 5:
        raise ValueError('V25 primary controls must name five distinct variants')
    if set(result['PRIMARY_ABLATIONS'].values()) - set(structured):
        raise ValueError('V25 primary controls must be structured variants')
    if set(result['VARIANT_POLICY']) != set(structured):
        raise ValueError('V25 policy must cover exactly all structured variants')
    return result

def _read_training_config(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    configs = payload.get('configs')
    if not isinstance(configs, list) or len(configs) != 1 or (not isinstance(configs[0], dict)):
        raise ValueError('V25 needs exactly one predeclared fixed configuration')
    config = dict(configs[0])
    required = {'name', 'learning_rate', 'weight_decay', 'dropout', 'epochs', 'batch_size', 'label_smoothing', 'anchor_epochs', 'anchor_learning_rate', 'anchor_weight_decay', 'fixed_anchor_lambda_grid', 'fixed_anchor_lambda_selection'}
    if set(config) != required:
        raise ValueError(f'Unexpected V25 training configuration fields: {sorted(set(config) ^ required)}')
    if not isinstance(config['name'], str) or not config['name'].startswith('fixed_'):
        raise ValueError('V25 configuration must be fixed_*')
    for field in ('epochs', 'batch_size', 'anchor_epochs'):
        if int(config[field]) < 1:
            raise ValueError(f'V25 {field} must be positive')
    for field in ('learning_rate', 'weight_decay', 'dropout', 'label_smoothing', 'anchor_learning_rate', 'anchor_weight_decay'):
        if not np.isfinite(float(config[field])) or float(config[field]) < 0.0:
            raise ValueError(f'V25 invalid {field}')
    if not 0.0 <= float(config['label_smoothing']) < 1.0:
        raise ValueError('V25 label smoothing must lie in [0,1)')
    grid = np.asarray(config['fixed_anchor_lambda_grid'], dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2 or (not np.isfinite(grid).all()) or np.any((grid <= 0.0) | (grid >= 1.0)):
        raise ValueError('V25 fixed Anchor grid must contain finite values strictly inside (0,1)')
    if len(np.unique(grid)) != len(grid) or not isinstance(config['fixed_anchor_lambda_selection'], str):
        raise ValueError('V25 fixed Anchor selection declaration invalid')
    config['fixed_anchor_lambda_grid'] = [float(value) for value in grid.tolist()]
    return config

def _seed(seed: int) -> None:
    seed_everything(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _fit_scale_raw(values: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.mean(np.square(values[np.asarray(train_indices, dtype=np.int64)], dtype=np.float64), axis=(0, 2)))
    return np.maximum(scale, 1e-06).astype(np.float32)

def _save_calibration(path: Path, values: dict[str, Any], indices: np.ndarray) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {'fit_indices': np.asarray(indices, dtype=np.int64)}
    for (name, value) in values.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array.astype(np.float64, copy=False)).all():
            raise ValueError(f'Non-finite/non-numeric V25 calibration field: {name}')
        arrays[name] = array
    _save_npz_new(path, **arrays)
    return arrays

def _fit_calibrations(model: torch.nn.Module, *, eeg: torch.Tensor, meg: torch.Tensor, train_indices: np.ndarray, eval_indices: np.ndarray, fit_dir: Path, fixed_anchor_lambda: float | None, symbols: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    if np.intersect1d(train_indices, eval_indices).size:
        raise ValueError('V25 calibration sees evaluation trials')
    pre = state_hash(model.state_dict())
    if bool(getattr(model, 'requires_controller_calibration', False)):
        train = torch.as_tensor(train_indices, dtype=torch.long, device=eeg.device)
        model.fit_controller_calibration(eeg[train], meg[train])
        saved = model.controller_calibration()
        arrays = _save_calibration(fit_dir / CONTROLLER_CALIBRATION_FILENAME, saved, train_indices)
        if arrays.get('centers', np.empty(0)).shape != (3, 3) or arrays.get('spreads', np.empty(0)).shape != (3,) or np.any(np.asarray(arrays['spreads']) < 0.04):
            raise ValueError('V25 controller calibration shape/range invalid')
        controller = {'applicable': True, 'source': 'scaled outer/inner train partition only', 'labels_consumed': False, 'fit_indices': train_indices.astype(int).tolist(), 'fit_indices_sha256': array_hash(train_indices), 'artifact': CONTROLLER_CALIBRATION_FILENAME, 'values': {name: np.asarray(value).astype(float).tolist() for (name, value) in arrays.items() if name != 'fit_indices'}, 'state_sha256': state_hash({name: model.state_dict()[name] for name in symbols['CONTROLLER_STATE_KEYS']})}
    else:
        controller = {'applicable': False, 'reason': 'descriptive_mono_control'}
    post_controller = state_hash(model.state_dict())
    if bool(getattr(model, 'requires_anchor_mixing_calibration', False)):
        train = torch.as_tensor(train_indices, dtype=torch.long, device=eeg.device)
        model.fit_anchor_mixing_calibration(eeg[train], meg[train])
        if fixed_anchor_lambda is not None:
            model.set_fixed_anchor_lambda(float(fixed_anchor_lambda))
        saved = model.anchor_mixing_calibration()
        arrays = _save_calibration(fit_dir / ANCHOR_MIXING_CALIBRATION_FILENAME, saved, train_indices)
        if any((np.asarray(arrays[name]).size != 1 for name in ('center', 'scale', 'fixed_structured_mean'))) or float(np.asarray(arrays['scale']).item()) <= 0.0:
            raise ValueError('V25 Anchor calibration shape/range invalid')
        mixing = {'applicable': True, 'source': 'scaled outer/inner train partition only', 'labels_consumed': False, 'fit_indices': train_indices.astype(int).tolist(), 'fit_indices_sha256': array_hash(train_indices), 'artifact': ANCHOR_MIXING_CALIBRATION_FILENAME, 'fixed_outer_train_lambda': None if fixed_anchor_lambda is None else float(fixed_anchor_lambda), 'values': {name: float(np.asarray(value).item()) for (name, value) in arrays.items() if name != 'fit_indices'}, 'state_sha256': state_hash({name: model.state_dict()[name] for name in symbols['ANCHOR_MIXING_STATE_KEYS']})}
    else:
        mixing = {'applicable': False, 'reason': 'descriptive_mono_control'}
    return (controller, mixing, post_controller, state_hash(model.state_dict()))

def _smoothed_nll(log_probability: torch.Tensor, target: torch.Tensor, smoothing: float) -> torch.Tensor:
    selected = -log_probability.gather(1, target[:, None]).squeeze(1)
    if smoothing == 0.0:
        return selected
    return (1.0 - float(smoothing)) * selected + float(smoothing) * -log_probability.mean(dim=-1)

def _batch_digest(order: np.ndarray, digest: Any) -> None:
    digest.update(np.asarray(order, dtype='<i8').tobytes())

def _pretrain_anchor(model: torch.nn.Module, *, eeg: torch.Tensor, meg: torch.Tensor, target: torch.Tensor, train_indices: np.ndarray, config: dict[str, Any], seed: int, fit_dir: Path) -> dict[str, Any]:
    if not bool(getattr(model, 'requires_anchor_pretraining', False)):
        return {'applicable': False, 'reason': 'descriptive_mono_control'}
    index = torch.as_tensor(train_indices, dtype=torch.long, device=eeg.device)
    model.eval()
    with torch.no_grad():
        raw_weights = model.anchor_pretraining_weight(eeg[index], meg[index])
    weights = np.asarray(raw_weights.detach().cpu(), dtype=np.float32).reshape(-1)
    if weights.shape != (len(train_indices),) or not np.isfinite(weights).all() or np.any(weights < 0.0) or (float(weights.sum()) <= 0.0):
        raise RuntimeError('V25 Anchor pretraining requires positive finite label-free weights')
    weights /= float(weights.mean())
    _save_npz_new(fit_dir / ANCHOR_PRETRAIN_WEIGHT_FILENAME, fit_indices=train_indices, weights=weights)
    start = state_hash({name: value for (name, value) in model.state_dict().items() if name.startswith('anchor.')})
    optimizer = torch.optim.AdamW(model.anchor.parameters(), lr=float(config['anchor_learning_rate']), weight_decay=float(config['anchor_weight_decay']))
    order_generator = torch.Generator(device='cpu').manual_seed(int(seed) + 17)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=eeg.device)
    positions = {int(value): location for (location, value) in enumerate(train_indices.tolist())}
    (digest, history, steps) = (hashlib.sha256(), [], 0)
    for epoch in range(int(config['anchor_epochs'])):
        model.anchor.train()
        order = train_indices[torch.randperm(len(train_indices), generator=order_generator).numpy()]
        _batch_digest(order, digest)
        (losses, correct) = ([], 0)
        for start_at in range(0, len(order), int(config['batch_size'])):
            batch = order[start_at:start_at + int(config['batch_size'])]
            batch_index = torch.as_tensor(batch, dtype=torch.long, device=eeg.device)
            loc = torch.as_tensor([positions[int(value)] for value in batch], dtype=torch.long, device=eeg.device)
            optimizer.zero_grad(set_to_none=True)
            log_probability = model.anchor_log_probability(eeg[batch_index], meg[batch_index])
            per_item = _smoothed_nll(log_probability, target[batch_index], float(config['label_smoothing']))
            loss = (per_item * weights_t[loc]).sum() / weights_t[loc].sum().clamp_min(1e-08)
            loss.backward()
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach().cpu()))
            correct += int((log_probability.argmax(-1) == target[batch_index]).sum().item())
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'unweighted_accuracy_percent': float(100.0 * correct / len(order))})
    model.freeze_anchor()
    final = state_hash({name: value for (name, value) in model.state_dict().items() if name.startswith('anchor.')})
    return {'applicable': True, 'weight_source': 'label-free Anchor-route cue on scaled train partition', 'labels_consumed_for_weights': False, 'labels_consumed_for_weighted_anchor_nll': True, 'fit_indices': train_indices.astype(int).tolist(), 'fit_indices_sha256': array_hash(train_indices), 'weight_artifact': ANCHOR_PRETRAIN_WEIGHT_FILENAME, 'weight_summary': {'n': int(len(weights)), 'mean': float(weights.mean()), 'min': float(weights.min()), 'max': float(weights.max()), 'positive_count': int((weights > 0.0).sum())}, 'epochs': int(config['anchor_epochs']), 'batch_size': int(config['batch_size']), 'optimizer_steps': int(steps), 'expected_optimizer_steps': int(config['anchor_epochs']) * math.ceil(len(train_indices) / int(config['batch_size'])), 'batch_order_sha256': digest.hexdigest(), 'history': history, 'initial_anchor_state_sha256': start, 'final_anchor_state_sha256': final, 'status': 'trained_on_train_partition_then_frozen'}

def _evaluate(model: torch.nn.Module, eeg: torch.Tensor, meg: torch.Tensor, y: np.ndarray, indices: np.ndarray, detail_keys: tuple[str, ...]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    model.eval()
    with torch.no_grad():
        index = torch.as_tensor(indices, dtype=torch.long, device=eeg.device)
        detail = model.forward_with_details(eeg[index], meg[index])
    required = {'log_probability', 'local_weights', 'branch_weights', 'mix_weight', 'structured_probability', 'anchor_probability'}
    if required - set(detail):
        raise KeyError(f'V25 forward detail misses {sorted(required - set(detail))}')
    arrays = {'probability': detail['log_probability'].exp().detach().cpu().numpy().astype(np.float32), 'structured_probability': detail['structured_probability'].detach().cpu().numpy().astype(np.float32), 'anchor_probability': detail['anchor_probability'].detach().cpu().numpy().astype(np.float32), 'local_weights': detail['local_weights'].detach().cpu().numpy().astype(np.float32), 'branch_weights': detail['branch_weights'].detach().cpu().numpy().astype(np.float32), 'mix_weight': detail['mix_weight'].detach().cpu().numpy().astype(np.float32)}
    for key in detail_keys:
        if key not in detail or not isinstance(detail[key], torch.Tensor):
            raise KeyError(f'V25 declared detail unavailable: {key}')
        arrays[key] = detail[key].detach().cpu().numpy().astype(np.float32)
    return (arrays, np.asarray(y[indices], dtype=np.int64))

def _fit_once(fit_dir: Path, *, variant: str, config: dict[str, Any], train_indices: np.ndarray, evaluation_indices: np.ndarray, seed: int, eeg_raw: np.ndarray, meg_raw: np.ndarray, y: np.ndarray, device: torch.device, provenance: dict[str, Any], model_mod: Any, fixed_anchor_lambda: float | None=None) -> dict[str, Any]:
    if fit_dir.exists():
        raise FileExistsError(f'Refusing existing V25 fit directory: {fit_dir}')
    (train_indices, evaluation_indices) = (np.asarray(train_indices, dtype=np.int64), np.asarray(evaluation_indices, dtype=np.int64))
    if np.intersect1d(train_indices, evaluation_indices).size:
        raise ValueError('V25 train/evaluation overlap')
    fit_dir.mkdir(parents=True, exist_ok=False)
    symbols = _symbols(model_mod)
    specification = {'variant': variant, 'config': config, 'train_indices': train_indices.astype(int).tolist(), 'evaluation_indices': evaluation_indices.astype(int).tolist(), 'seed': int(seed), 'fixed_anchor_lambda': None if fixed_anchor_lambda is None else float(fixed_anchor_lambda), 'model_family': symbols['MODEL_FAMILY']}
    specification['specification_sha256'] = _canonical_hash(specification)
    _write_json_new(fit_dir / 'specification.json', specification)
    _seed(int(seed))
    (eeg_scale, meg_scale) = (_fit_scale_raw(eeg_raw, train_indices), _fit_scale_raw(meg_raw, train_indices))
    _save_npz_new(fit_dir / 'scalers.npz', eeg_scale=eeg_scale, meg_scale=meg_scale, fit_indices=train_indices)
    eeg = torch.as_tensor(eeg_raw / eeg_scale[None, :, None], dtype=torch.float32, device=device)
    meg = torch.as_tensor(meg_raw / meg_scale[None, :, None], dtype=torch.float32, device=device)
    target = torch.as_tensor(y, dtype=torch.long, device=device)
    model = symbols['make_model'](variant, dropout=float(config['dropout'])).to(device)
    initial = state_hash(model.state_dict())
    (controller, anchor_mixing, post_controller, post_mixing) = _fit_calibrations(model, eeg=eeg, meg=meg, train_indices=train_indices, eval_indices=evaluation_indices, fit_dir=fit_dir, fixed_anchor_lambda=fixed_anchor_lambda, symbols=symbols)
    anchor = _pretrain_anchor(model, eeg=eeg, meg=meg, target=target, train_indices=train_indices, config=config, seed=int(seed), fit_dir=fit_dir)
    if anchor.get('applicable') and int(anchor['optimizer_steps']) != int(anchor['expected_optimizer_steps']):
        raise RuntimeError('V25 unexpected Anchor pretraining step count')
    _write_json_new(fit_dir / 'anchor_pretraining.json', anchor)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))
    order_generator = torch.Generator(device='cpu').manual_seed(int(seed) + 31)
    (digest, history, steps) = (hashlib.sha256(), [], 0)
    for epoch in range(int(config['epochs'])):
        model.train()
        order = train_indices[torch.randperm(len(train_indices), generator=order_generator).numpy()]
        _batch_digest(order, digest)
        (losses, correct) = ([], 0)
        for start_at in range(0, len(order), int(config['batch_size'])):
            batch = order[start_at:start_at + int(config['batch_size'])]
            batch_index = torch.as_tensor(batch, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            log_probability = model(eeg[batch_index], meg[batch_index])
            loss = _smoothed_nll(log_probability, target[batch_index], float(config['label_smoothing'])).mean()
            loss.backward()
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach().cpu()))
            correct += int((log_probability.argmax(-1) == target[batch_index]).sum().item())
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'train_accuracy_percent': float(100.0 * correct / len(order))})
    expected_steps = int(config['epochs']) * math.ceil(len(train_indices) / int(config['batch_size']))
    if steps != expected_steps:
        raise RuntimeError('V25 main training step count mismatch')
    (details, expected) = _evaluate(model, eeg, meg, y, evaluation_indices, tuple(symbols['DETAIL_KEYS']) if variant in symbols['STRUCTURED_VARIANTS'] else ())
    state = {name: value.detach().cpu() for (name, value) in model.state_dict().items()}
    torch.save(state, fit_dir / 'model.pt')
    predicted = details['probability'].argmax(axis=1).astype(np.int64)
    _save_npz_new(fit_dir / 'predictions.npz', indices=evaluation_indices, y=expected, prediction=predicted, **details)
    training = {'epochs': int(config['epochs']), 'batch_size': int(config['batch_size']), 'optimizer_steps': int(steps), 'expected_optimizer_steps': int(expected_steps), 'batch_order_sha256': digest.hexdigest(), 'history': history, 'initial_state_sha256': initial, 'post_controller_calibration_state_sha256': post_controller, 'post_anchor_mixing_calibration_state_sha256': post_mixing, 'final_state_sha256': state_hash(state), 'parameter_count': int(symbols['parameter_count'](model)), 'trainable_parameter_count_after_anchor_freeze': int(symbols['trainable_parameter_count'](model)), 'device': str(device), 'controller_calibration': controller, 'anchor_mixing_calibration': anchor_mixing}
    _write_json_new(fit_dir / 'training.json', training)
    artifacts = ['specification.json', 'scalers.npz', 'anchor_pretraining.json', 'predictions.npz', 'training.json', 'model.pt']
    if anchor.get('applicable'):
        artifacts.append(ANCHOR_PRETRAIN_WEIGHT_FILENAME)
    if controller.get('applicable'):
        artifacts.append(CONTROLLER_CALIBRATION_FILENAME)
    if anchor_mixing.get('applicable'):
        artifacts.append(ANCHOR_MIXING_CALIBRATION_FILENAME)
    outcome = metrics(expected, details['probability'])
    result = {'variant': variant, 'specification_sha256': specification['specification_sha256'], 'n_train': int(len(train_indices)), 'n_evaluation': int(len(evaluation_indices)), 'outcome': outcome, 'artifacts': {name: sha256(fit_dir / name) for name in artifacts}, 'initial_state_sha256': initial, 'final_state_sha256': training['final_state_sha256']}
    _write_json_new(fit_dir / 'result.json', result)
    del model, optimizer, eeg, meg, target, state
    torch.cuda.empty_cache()
    return result

def _select_fixed_anchor_lambda(full_inner_paths: list[Path], config: dict[str, Any]) -> dict[str, Any]:
    (target, structured, anchor) = ([], [], [])
    for path in full_inner_paths:
        with np.load(path / 'predictions.npz', allow_pickle=False) as archive:
            target.append(np.asarray(archive['y'], dtype=np.int64))
            structured.append(np.asarray(archive['structured_probability'], dtype=np.float64))
            anchor.append(np.asarray(archive['anchor_probability'], dtype=np.float64))
    (y, ps, pa) = (np.concatenate(target), np.concatenate(structured), np.concatenate(anchor))
    candidates = []
    for value in config['fixed_anchor_lambda_grid']:
        probability = float(value) * ps + (1.0 - float(value)) * pa
        candidates.append({'lambda': float(value), 'pooled_inner_accuracy_percent': float(100.0 * np.mean(probability.argmax(axis=1) == y))})
    ranked = sorted(candidates, key=lambda item: (-item['pooled_inner_accuracy_percent'], abs(item['lambda'] - 0.5), item['lambda']))
    return {'n_inner_validation_predictions': int(len(y)), 'candidate_scores': candidates, 'selected_lambda': float(ranked[0]['lambda']), 'selected_pooled_inner_accuracy_percent': float(ranked[0]['pooled_inner_accuracy_percent'])}
