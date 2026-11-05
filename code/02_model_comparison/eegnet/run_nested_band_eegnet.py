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
import scipy
from scipy.signal import butter, sosfiltfilt
import torch
import eegnet_baselines as model_module
import run_eegnet_blocks as fixed_module
import run_nested_eegnet as nested_module
from eegnet_baselines import LEGACY_SOURCE_SHA256, MODEL_SETTINGS, array_sha256, file_sha256, load_public_data, normalize_trials
from run_eegnet_blocks import _json, _npz, _resolve_device
from run_nested_eegnet import CANDIDATE_EPOCHS, FUSION_HEAD_EPOCHS, _fit_feature_head, _fit_seed, _fit_unimodal, _metrics, _serializable_plan, make_nested_splits, select_budget
PREPROCESSING_ID = 'eegnet_mu_beta_bandpass_sum_trialstd_v1'
METHOD_NAMES = ('eegnet_band_eeg', 'eegnet_band_meg', 'eegnet_band_decision', 'eegnet_band_feature_concat')
METHOD_LABELS = {'eegnet_band_eeg': 'EEGNet μ/β band-pass: EEG', 'eegnet_band_meg': 'EEGNet μ/β band-pass: MEG', 'eegnet_band_decision': 'EEGNet μ/β band-pass: decision fusion', 'eegnet_band_feature_concat': 'EEGNet μ/β band-pass: frozen-feature concatenation'}
BANDS = ((8.0, 13.0), (15.0, 28.0))

def filter_spec(fs: float) -> dict[str, Any]:
    if not np.isfinite(fs) or fs != 250.0:
        raise ValueError('This diagnostic expects fs=250 Hz')
    filters = []
    for (low, high) in BANDS:
        sos = butter(4, (low, high), btype='bandpass', fs=fs, output='sos')
        padlen = 250
        filters.append({'band_hz': [low, high], 'sos': sos.tolist(), 'sos_sha256': array_sha256(sos), 'padlen': padlen})
    return {'id': PREPROCESSING_ID, 'fs_hz': fs, 'design': "scipy.signal.butter(N=4,btype='bandpass',output='sos')", 'butterworth_order_argument': 4, 'single_pass_bandpass_transfer_order': 8, 'application': 'scipy.signal.sosfiltfilt, axis=-1, per trial/channel', 'phase': 'zero phase, forward-backward; magnitude response squared', 'padtype': 'odd', 'padlen': 250, 'boundary_reference': "Matches Proposed's fixed250-sample odd extension; not SciPy default padding.", 'filters': filters, 'combination': 'sum two filtered waveforms, not concatenation', 'arithmetic': 'float64 filtering and summation; cast summed signal to float32', 'normalization': 'archived normalize_trials: x/(x.std(channels,time)+1e-6), float32', 'normalization_order': 'after band summation', 'trial_boundary': 'never filter across trial boundaries', 'dataset_fitting': False, 'labels_used': False, 'scipy_version': scipy.__version__}

def band_sum_trials(x: np.ndarray, fs: float=250.0) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != (64, 1000) or len(x) == 0:
        raise ValueError('Expected a nonempty (N,64,1000) input')
    if not np.isfinite(x).all():
        raise ValueError('Input contains nonfinite values')
    spec = filter_spec(float(fs))
    coefficients = [(np.asarray(item['sos'], dtype=np.float64), item['padlen']) for item in spec['filters']]
    filtered = np.empty(x.shape, dtype=np.float32)
    for index in range(len(x)):
        trial = np.asarray(x[index], dtype=np.float64)
        combined = np.zeros(trial.shape, dtype=np.float64)
        for (sos, padlen) in coefficients:
            combined += sosfiltfilt(sos, trial, axis=-1, padtype='odd', padlen=padlen)
        filtered[index] = combined.astype(np.float32)
    if not np.isfinite(filtered).all():
        raise FloatingPointError('Nonfinite band-filtered input')
    return (filtered, spec)

def prepare_inputs(public: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    (normalized, scales, filtered_hashes) = ({}, {}, {})
    fs = float(np.asarray(public['fs']).reshape(-1)[0])
    spec = None
    for modality in ('eeg', 'meg'):
        (filtered, current_spec) = band_sum_trials(public[modality], fs)
        if spec is not None and current_spec != spec:
            raise RuntimeError('Modality filter specifications differ')
        spec = current_spec
        filtered_hashes[modality] = array_sha256(filtered)
        (normalized[modality], scales[modality]) = normalize_trials(filtered)
    provenance = {'preprocessing_id': PREPROCESSING_ID, 'filter_spec': spec, 'filtered_sum_array_sha256': filtered_hashes, 'normalized_input_array_sha256': {m: array_sha256(x) for (m, x) in normalized.items()}, 'normalization_scale_array_sha256': {m: array_sha256(x) for (m, x) in scales.items()}, 'normalized_input_shape': {m: list(x.shape) for (m, x) in normalized.items()}, 'normalized_input_dtype': 'float32'}
    return (normalized, scales, provenance)

def run(data_path: Path, subject: int, out_dir: Path, seed: int, device_name: str=None) -> dict[str, Any]:
    if seed < 0 or subject < 0:
        raise ValueError('seed and subject must be nonnegative')
    if out_dir.exists():
        raise FileExistsError(f'Refusing to overwrite an existing output path: {out_dir}')
    device = _resolve_device(device_name)
    data_hash = file_sha256(data_path)
    public = load_public_data(data_path, subject)
    plan = make_nested_splits(public['block'])
    (normalized, scales, input_provenance) = prepare_inputs(public)
    source_paths = {'run_nested_band_eegnet.py': Path(__file__), 'run_nested_eegnet.py': Path(nested_module.__file__), 'run_eegnet_blocks.py': Path(fixed_module.__file__), 'eegnet_baselines.py': Path(model_module.__file__)}
    source_hashes = {name: file_sha256(path) for (name, path) in source_paths.items()}
    seed_plan = [{'outer_block': outer['outer_block'], 'inner': [{'validation_block': inner['validation_block'], 'seeds_same_across_epoch_candidates': {m: _fit_seed(seed, subject, outer['outer_block'], m, inner['validation_block']) for m in ('eeg', 'meg')}} for inner in outer['inner']]} for outer in plan]
    protocol = {'model_family': 'EEGNet mu/beta band-pass', 'method_names': list(METHOD_NAMES), 'method_labels': METHOD_LABELS, 'subject': subject, 'data_path': str(data_path.resolve()), 'data_sha256': data_hash, 'public_array_sha256': {k: array_sha256(v) for (k, v) in public.items()}, **input_provenance, 'candidate_epochs': list(CANDIDATE_EPOCHS), 'fusion_head_epochs': FUSION_HEAD_EPOCHS, 'splits': _serializable_plan(plan), 'seed': seed, 'seed_plan': seed_plan, 'inner_feature_head_fits': 0, 'training_fit_counts': {'inner_unimodal': 120, 'outer_unimodal_refits': 10, 'outer_feature_heads': 5, 'total': 135}, 'inner_budget_epochs_per_modality_per_outer': 4 * sum(CANDIDATE_EPOCHS), 'optimization_unimodal': {'optimizer': 'AdamW', 'lr': 0.0008, 'weight_decay': 0.001, 'batch_size': 30, 'label_smoothing': 0.04, 'scheduler': None}, 'optimization_head': {'optimizer': 'AdamW', 'lr': 0.0005, 'weight_decay': 0.002, 'batch_size': 30, 'label_smoothing': 0.04, 'scheduler': None}, 'encoder_settings': MODEL_SETTINGS, 'public_inputs_only': ['eeg', 'meg', 'y', 'block', 'fs', 'trial_id', 'optional subject_id'], 'outer_test_used_for_selection': False, 'early_stopping': False, 'last_epoch_checkpoint_per_candidate': True, 'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__, 'cuda': torch.version.cuda, 'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(), 'cudnn': torch.backends.cudnn.version(), 'tf32': False, 'deterministic_algorithms': True}, 'created_utc': datetime.now(timezone.utc).isoformat()}
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
        context = {'subject': subject, 'outer_block': outer_block, 'fold': outer_block + 1, 'data_sha256': data_hash, 'model_family': protocol['model_family'], 'preprocessing_id': PREPROCESSING_ID, 'normalized_input_array_sha256': input_provenance['normalized_input_array_sha256']}
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
                    print(json.dumps({'event': 'inner_complete', 'preprocessing_id': PREPROCESSING_ID, 'subject': subject, 'outer_block': outer_block, 'validation_block': validation_block, 'modality': modality, 'epochs': epochs, 'validation_accuracy_percent': result['evaluation_metrics']['accuracy_percent']}), flush=True)
                    del model, probability
                    torch.cuda.empty_cache()
        validation_blocks = [inner['validation_block'] for inner in outer['inner']]
        selection = {m: select_budget(inner_records[m], validation_blocks) for m in ('eeg', 'meg')}
        _json(fold_dir / 'selection.json', {'outer_block': outer_block, 'selection': selection, 'inner_records': inner_records, 'outer_test_used': False, 'feature_head_epochs_fixed': FUSION_HEAD_EPOCHS})
        (models, probabilities, refit_results) = ({}, {}, {})
        for modality in ('eeg', 'meg'):
            selected_epochs = selection[modality]['selected_epochs']
            fit_seed = _fit_seed(seed, subject, outer_block, modality)
            (model, probability, result) = _fit_unimodal(normalized[modality], public['y'], train, test, modality, selected_epochs, fit_seed, device, fold_dir / f'outer_refit_{modality}', {**context, 'selection_sha256': file_sha256(fold_dir / 'selection.json')}, 'outer_test')
            models[modality] = model
            probabilities[f'eegnet_band_{modality}'] = probability
            refit_results[modality] = result
        (feature_probability, feature_result) = _fit_feature_head(models, normalized, public['y'], train, test, _fit_seed(seed, subject, outer_block, 'fusion'), device, fold_dir / 'outer_feature_concat', {**context, 'selected_backbone_epochs': {m: selection[m]['selected_epochs'] for m in ('eeg', 'meg')}, 'backbone_checkpoint_sha256': {m: refit_results[m]['checkpoint_sha256'] for m in ('eeg', 'meg')}, 'selection_sha256': file_sha256(fold_dir / 'selection.json')})
        probabilities['eegnet_band_feature_concat'] = feature_probability
        probabilities['eegnet_band_decision'] = 0.5 * probabilities['eegnet_band_eeg'] + 0.5 * probabilities['eegnet_band_meg']
        _json(fold_dir / 'decision_fusion.json', {'weights': {'eeg': 0.5, 'meg': 0.5}, 'trainable_parameters': 0, 'backbone_checkpoint_sha256': {m: refit_results[m]['checkpoint_sha256'] for m in ('eeg', 'meg')}})
        fold_metrics = {method: _metrics(public['y'][test], probability) for (method, probability) in probabilities.items()}
        for (method, probability) in probabilities.items():
            oof[method][test] = probability
        coverage[test] += 1
        _npz(fold_dir / 'outer_predictions.npz', y=public['y'][test], indices=test, trial_id=public['trial_id'][test], **probabilities)
        result = {'outer_block': outer_block, 'fold': outer_block + 1, 'selected_epochs': {m: selection[m]['selected_epochs'] for m in ('eeg', 'meg')}, 'metrics': fold_metrics, 'refit_results': refit_results, 'feature_head_result': feature_result, 'selection_sha256': file_sha256(fold_dir / 'selection.json'), 'outer_predictions_sha256': file_sha256(fold_dir / 'outer_predictions.npz')}
        _json(fold_dir / 'result.json', result)
        all_results.append(result)
        print(json.dumps({'event': 'outer_complete', 'preprocessing_id': PREPROCESSING_ID, 'subject': subject, 'outer_block': outer_block, 'selected_epochs': result['selected_epochs'], 'accuracy_percent': {name: value['accuracy_percent'] for (name, value) in fold_metrics.items()}}, ensure_ascii=False), flush=True)
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
    summary = {'complete': True, 'subject': subject, 'n': 150, 'outer_folds': 5, 'model_family': protocol['model_family'], 'method_labels': METHOD_LABELS, 'preprocessing_id': PREPROCESSING_ID, 'normalized_input_array_sha256': input_provenance['normalized_input_array_sha256'], 'inner_validation_blocks_per_outer': 4, 'candidate_epochs': list(CANDIDATE_EPOCHS), 'feature_head_epochs_fixed': FUSION_HEAD_EPOCHS, 'protocol_sha256': file_sha256(out_dir / 'protocol.json'), 'data_sha256': data_hash, 'oof_sha256': file_sha256(out_dir / 'oof_predictions.npz'), 'elapsed_seconds': time.monotonic() - started, 'metrics': {name: _metrics(public['y'], probability) for (name, probability) in oof.items()}, 'fold_results': all_results, 'fit_counts': protocol['training_fit_counts'], 'outer_test_used_for_selection': False, 'completed_utc': datetime.now(timezone.utc).isoformat()}
    _json(out_dir / 'summary.json', summary)
    print(json.dumps({'event': 'complete', 'preprocessing_id': PREPROCESSING_ID, 'subject': subject, 'metrics': summary['metrics'], 'output': str(out_dir)}, ensure_ascii=False), flush=True)
    return summary

def self_test() -> dict[str, Any]:
    inherited = nested_module.self_test()
    raw = np.random.default_rng(73418).normal(size=(3, 64, 1000)).astype(np.float32)
    raw_before = raw.copy()
    (combined, spec) = band_sum_trials(raw)
    assert combined.shape == raw.shape and combined.dtype == np.float32
    assert np.isfinite(combined).all()
    np.testing.assert_array_equal(raw, raw_before)
    (normalized, scales) = normalize_trials(combined)
    assert normalized.shape == raw.shape and scales.shape == (3,)
    for trial in range(len(raw)):
        (single, same_spec) = band_sum_trials(raw[trial:trial + 1])
        (single_normalized, single_scale) = normalize_trials(single)
        assert same_spec == spec
        np.testing.assert_array_equal(single[0], combined[trial])
        np.testing.assert_array_equal(single_normalized[0], normalized[trial])
        np.testing.assert_array_equal(single_scale, scales[trial:trial + 1])
    changed = raw.copy()
    changed[1:] *= 800.0
    (changed_filtered, _) = band_sum_trials(changed)
    np.testing.assert_array_equal(changed_filtered[0], combined[0])
    permutation = np.array([2, 0, 1])
    (permuted, _) = band_sum_trials(raw[permutation])
    np.testing.assert_array_equal(permuted, combined[permutation])
    reference = np.zeros_like(raw[0], dtype=np.float64)
    for item in spec['filters']:
        sos = butter(4, item['band_hz'], btype='bandpass', fs=250, output='sos')
        np.testing.assert_array_equal(sos, np.asarray(item['sos']))
        assert item['sos_sha256'] == array_sha256(sos)
        assert item['padlen'] == spec['padlen'] == 250
        reference += sosfiltfilt(sos, raw[0].astype(np.float64), axis=-1, padtype='odd', padlen=250)
    np.testing.assert_array_equal(reference.astype(np.float32), combined[0])
    default_reference = sum((sosfiltfilt(butter(4, band, btype='bandpass', fs=250, output='sos'), raw[0].astype(np.float64), axis=-1) for band in BANDS))
    assert not np.array_equal(default_reference.astype(np.float32), combined[0])
    (prepared, prepared_scales, provenance) = prepare_inputs({'eeg': raw, 'meg': raw[::-1], 'fs': np.asarray(250)})
    np.testing.assert_array_equal(prepared['eeg'], normalized)
    np.testing.assert_array_equal(prepared_scales['eeg'], scales)
    assert provenance['normalized_input_array_sha256']['eeg'] == array_sha256(normalized)
    assert provenance['filtered_sum_array_sha256']['eeg'] == array_sha256(combined)
    assert _fit_unimodal is nested_module._fit_unimodal
    assert _fit_feature_head is nested_module._fit_feature_head
    assert select_budget is nested_module.select_budget
    for (invalid, fs) in [(raw[:, :, :-1], 250), (raw, 200), (np.full_like(raw, np.nan), 250)]:
        try:
            band_sum_trials(invalid, fs)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid filter input unexpectedly accepted')
    json.dumps(provenance)
    return {'status': 'PASS', 'device': 'CPU QA only', 'inherited_nested_qa': inherited, 'preprocessing_id': PREPROCESSING_ID, 'scipy_version': scipy.__version__, 'filter_spec': spec, 'method_names': list(METHOD_NAMES)}

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
