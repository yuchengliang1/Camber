from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
import hashlib
import warnings
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
SCHEMA = 'r12_independent_local_power_probes_v1'
MODALITIES = ('EEG', 'MEG')
BANDS_HZ = ((8.0, 13.0), (18.0, 30.0))
SAMPLE_RATE_HZ = 250.0
(CHANNELS, SAMPLES, WINDOWS, WINDOW_SAMPLES) = (64, 1000, 8, 125)
VARIANCE_FLOOR = 1e-12
SCORE_PROBABILITY_FLOOR = 1e-12
CLASSES = np.arange(3, dtype=np.int64)
PROBE_PARAMETERS = {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000, 'tol': 0.0001, 'fit_intercept': True}

def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(array.dtype).encode('ascii'))
    digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()

def extract_features(eeg, meg) -> np.ndarray:
    inputs = (np.asarray(eeg), np.asarray(meg))
    if inputs[0].shape != inputs[1].shape:
        raise ValueError('EEG and MEG must have the same array shape')
    if inputs[0].ndim != 3 or inputs[0].shape[1:] != (CHANNELS, SAMPLES) or (not len(inputs[0])):
        raise ValueError('Expected nonempty arrays shaped (N,64,1000)')
    if any((not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x) for x in inputs)):
        raise ValueError('Input signals must be real numeric arrays')
    frequencies = np.fft.rfftfreq(SAMPLES, d=1.0 / SAMPLE_RATE_HZ)
    result = np.empty((len(inputs[0]), 2, 2, WINDOWS, CHANNELS), dtype=np.float64)
    for (modality, original) in enumerate(inputs):
        values = np.asarray(original, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite signal values are not supported')
        spectrum = np.fft.rfft(values, axis=-1)
        for (band, (low, high)) in enumerate(BANDS_HZ):
            mask = (frequencies >= low) & (frequencies <= high)
            filtered = np.fft.irfft(spectrum * mask, n=SAMPLES, axis=-1)
            variance = filtered.reshape(len(values), CHANNELS, WINDOWS, WINDOW_SAMPLES).var(axis=-1, ddof=0)
            result[:, modality, band] = np.log(np.maximum(variance, VARIANCE_FLOOR)).transpose(0, 2, 1)
    if not np.isfinite(result).all():
        raise RuntimeError('Extracted local power features are non-finite')
    return result

def rank_top_windows(inner_score, top_k=2) -> np.ndarray:
    score = np.asarray(inner_score, dtype=np.float64)
    if score.shape != (2, 2, WINDOWS) or not np.isfinite(score).all():
        raise ValueError('Expected finite evidence scores shaped (2,2,8)')
    if not isinstance(top_k, (int, np.integer)) or not 1 <= top_k <= WINDOWS:
        raise ValueError('top_k must be between one and eight')
    return np.argsort(-score, axis=-1, kind='stable')[..., :top_k]

def _fit_one(x_train, y_train, x_evaluate, *, label):
    if not np.array_equal(np.unique(y_train), CLASSES):
        raise ValueError(f'All three classes must occur in probe training: {label}')
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    train_z = scaler.fit_transform(x_train)
    evaluation_z = scaler.transform(x_evaluate)
    if not np.isfinite(train_z).all() or not np.isfinite(evaluation_z).all():
        raise RuntimeError(f'Non-finite standardized probe features: {label}')
    model = LogisticRegression(**PROBE_PARAMETERS)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model.fit(train_z, y_train)
    convergence_warnings = [str(item.message) for item in captured if issubclass(item.category, ConvergenceWarning)]
    iterations = np.asarray(getattr(model, 'n_iter_', []), dtype=np.int64)
    if convergence_warnings or not iterations.size or np.max(iterations) >= PROBE_PARAMETERS['max_iter']:
        raise RuntimeError(f'Probe did not converge: {label}; iterations={iterations.tolist()}; {convergence_warnings}')
    if not np.array_equal(model.classes_, CLASSES):
        raise RuntimeError(f'Unexpected probability class order: {label}')
    probability = model.predict_proba(evaluation_z)
    if probability.shape != (len(x_evaluate), 3) or not np.isfinite(probability).all():
        raise RuntimeError(f'Invalid fitted probability shape/values: {label}')
    if np.any(probability < 0) or np.any(probability > 1) or (not np.allclose(probability.sum(axis=1), 1.0, atol=1e-12, rtol=0)):
        raise RuntimeError(f'Invalid probability distribution: {label}')
    result = {'scaler_mean': scaler.mean_.copy(), 'scaler_scale': scaler.scale_.copy(), 'scaler_var': scaler.var_.copy(), 'constant_mask': scaler.var_ == 0, 'coef': model.coef_.copy(), 'intercept': model.intercept_.copy(), 'n_iter': int(np.max(model.n_iter_))}
    audit = {'label': label, 'converged': True, 'n_iter': result['n_iter'], 'n_fit': len(x_train), 'n_evaluate': len(x_evaluate), 'constant_features': int(result['constant_mask'].sum()), 'constant_indices': np.flatnonzero(result['constant_mask']).tolist(), 'non_convergence_warnings': [str(item.message) for item in captured if not issubclass(item.category, ConvergenceWarning)], 'training_feature_sha256': _array_hash(np.asarray(x_train)), 'training_label_sha256': _array_hash(np.asarray(y_train))}
    return (probability, result, audit)

def _allocate_models(prefix_shape):
    shape = tuple(prefix_shape) + (2, 2, WINDOWS)
    return {'scaler_mean': np.empty(shape + (CHANNELS,), dtype=np.float64), 'scaler_scale': np.empty(shape + (CHANNELS,), dtype=np.float64), 'scaler_var': np.empty(shape + (CHANNELS,), dtype=np.float64), 'constant_mask': np.empty(shape + (CHANNELS,), dtype=bool), 'coef': np.empty(shape + (3, CHANNELS), dtype=np.float64), 'intercept': np.empty(shape + (3,), dtype=np.float64), 'n_iter': np.empty(shape, dtype=np.int64)}

def fit_fold(features, y, block, outer_fold) -> dict:
    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(y)
    blocks = np.asarray(block)
    if values.shape != (150, 2, 2, WINDOWS, CHANNELS) or not np.isfinite(values).all():
        raise ValueError('Expected finite features shaped (150,2,2,8,64)')
    if labels.shape != (150,) or blocks.shape != (150,):
        raise ValueError('Labels and blocks must both contain 150 rows')
    (unique_blocks, counts) = np.unique(blocks, return_counts=True)
    if len(unique_blocks) != 5 or not np.all(counts == 30) or outer_fold not in unique_blocks:
        raise ValueError('Expected five blocks of 30 trials and a valid outer_fold')
    train_indices = np.flatnonzero(blocks != outer_fold)
    test_indices = np.flatnonzero(blocks == outer_fold)
    train_y = labels[train_indices].copy()
    if not np.issubdtype(train_y.dtype, np.integer) or not np.array_equal(np.unique(train_y), CLASSES):
        raise ValueError('Outer-training labels must be integers 0, 1, 2')
    if np.any(np.bincount(train_y, minlength=3) == 0):
        raise ValueError('Outer training requires all three classes')
    inner_blocks = unique_blocks[unique_blocks != outer_fold]
    inner_prob = np.empty((120, 2, 2, WINDOWS, 3), dtype=np.float64)
    prob_test = np.empty((30, 2, 2, WINDOWS, 3), dtype=np.float64)
    inner_models = _allocate_models((4,))
    outer_models = _allocate_models(())
    inner_fit_indices = np.empty((4, 90), dtype=np.int64)
    inner_validation_indices = np.empty((4, 30), dtype=np.int64)
    inner_coverage = np.zeros(120, dtype=np.int64)
    audits = []
    for (inner_fold, validation_block) in enumerate(inner_blocks):
        train_local = np.flatnonzero(blocks[train_indices] != validation_block)
        validation_local = np.flatnonzero(blocks[train_indices] == validation_block)
        fit_indices = train_indices[train_local]
        validation_indices = train_indices[validation_local]
        inner_fit_indices[inner_fold] = fit_indices
        inner_validation_indices[inner_fold] = validation_indices
        inner_coverage[validation_local] += 1
        for modality in range(2):
            for band in range(2):
                for window in range(WINDOWS):
                    unit = (modality, band, window)
                    (probability, parameters, audit) = _fit_one(values[fit_indices, modality, band, window], train_y[train_local], values[validation_indices, modality, band, window], label=f'inner{inner_fold}/m{modality}/b{band}/w{window}')
                    inner_prob[validation_local, modality, band, window] = probability
                    for (key, parameter) in parameters.items():
                        inner_models[key][(inner_fold,) + unit] = parameter
                    audit.update(phase='inner', inner_fold=inner_fold, validation_block=int(validation_block), modality=modality, band=band, window=window)
                    audits.append(audit)
    if not np.array_equal(inner_coverage, np.ones(120, dtype=np.int64)):
        raise RuntimeError('Inner cross-fitting did not cover each training row exactly once')
    true_probability = np.take_along_axis(inner_prob, train_y[:, None, None, None, None], axis=-1)[..., 0]
    inner_nll = -np.log(np.clip(true_probability, SCORE_PROBABILITY_FLOOR, 1.0)).mean(axis=0)
    inner_score = np.log(3.0) - inner_nll
    top2_windows = rank_top_windows(inner_score, top_k=2)
    for modality in range(2):
        for band in range(2):
            for window in range(WINDOWS):
                unit = (modality, band, window)
                (probability, parameters, audit) = _fit_one(values[train_indices, modality, band, window], train_y, values[test_indices, modality, band, window], label=f'outer/m{modality}/b{band}/w{window}')
                prob_test[:, modality, band, window] = probability
                for (key, parameter) in parameters.items():
                    outer_models[key][unit] = parameter
                audit.update(phase='outer', outer_fold=int(outer_fold), modality=modality, band=band, window=window)
                audits.append(audit)
    return {'prob_test': prob_test, 'inner_prob': inner_prob, 'inner_nll': inner_nll, 'inner_score': inner_score, 'top2_windows': top2_windows, 'train_indices': train_indices, 'test_indices': test_indices, 'inner_blocks': inner_blocks.copy(), 'inner_train_indices': inner_fit_indices, 'inner_validation_indices': inner_validation_indices, 'inner_coverage': inner_coverage, 'classes': CLASSES.copy(), 'outer_models': outer_models, 'inner_models': inner_models, 'fit_audit': audits, 'fit_count': len(audits), 'configuration': {'modalities': list(MODALITIES), 'bands_hz': [list(band) for band in BANDS_HZ], 'windows': WINDOWS, 'channels_per_local_probe': CHANNELS, 'sample_rate_hz': SAMPLE_RATE_HZ, 'samples_per_window': WINDOW_SAMPLES, 'variance_ddof': 0, 'variance_floor': VARIANCE_FLOOR, 'score_probability_floor': SCORE_PROBABILITY_FLOOR, 'probe_parameters': dict(PROBE_PARAMETERS), 'outer_test_labels_used': False, 'neural_model_used': False, 'numpy_version': np.__version__, 'sklearn_version': sklearn.__version__}}
