from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
import hashlib
import json
from functools import lru_cache
from typing import Final, Mapping
import numpy as np
COORDINATE_REGISTRY_VERSION: Final = 'feature_coefficients_r2'
CHANNELS: Final = 64
BANDS: Final = 2
WINDOWS: Final = 8
BRANCHES: Final = 4
CLASSES: Final = 3
MODALITY_EEG: Final = 0
MODALITY_MEG: Final = 1
MODALITY_NAMES: Final = ('EEG', 'MEG')
BRANCH_EEG: Final = 0
BRANCH_MEG: Final = 1
BRANCH_PAIR: Final = 2
BRANCH_FUSION: Final = 3
BRANCH_NAMES: Final = ('EEG', 'MEG', 'pair', 'fusion')
EXPERT_PRODUCT: Final = 0
EXPERT_DIFFERENCE: Final = 1
EXPERT_NAMES: Final = ('product', 'difference')
PATH_EEG_TO_MEG: Final = 0
PATH_MEG_TO_EEG: Final = 1
PATH_NAMES: Final = ('EEG_to_MEG', 'MEG_to_EEG')
PATH_ENDPOINT_SOURCE: Final = 0
PATH_ENDPOINT_TARGET: Final = 1
PATH_ENDPOINT_NAMES: Final = ('source', 'target')
CONTROLLER_PAIR_EXPERT: Final = 0
CONTROLLER_MODAL_ROUTE: Final = 1
CONTROLLER_PATH_DIRECTION: Final = 2
CONTROLLER_NAMES: Final = ('pair_expert', 'modal_route', 'path_direction')
ANCHOR_ROUTE_ANCHOR: Final = 0
ANCHOR_ROUTE_STRUCTURED: Final = 1
ANCHOR_ROUTE_NAMES: Final = ('anchor', 'structured')
REGISTRY_ARRAY_ORDER: Final = ('direct_task', 'pair_task', 'path_task', 'anchor_task', 'controller', 'branch', 'anchor_route', 'background', 'decoy')
CHANNEL_PERMUTATIONS: Final = {
    MODALITY_EEG: np.asarray((
        27, 19, 9, 11, 48, 37, 29, 21, 53, 28, 20, 41, 14, 40, 7, 0,
        39, 43, 5, 31, 1, 52, 36, 47, 26, 12, 25, 42, 32, 16, 13, 63,
        60, 46, 34, 3, 10, 15, 44, 61, 57, 35, 8, 18, 51, 55, 58, 4,
        59, 23, 50, 49, 2, 38, 17, 56, 62, 45, 33, 22, 30, 6, 24, 54,
    ), dtype=np.int64),
    MODALITY_MEG: np.asarray((
        0, 1, 2, 30, 27, 29, 9, 61, 60, 32, 47, 59, 45, 37, 62, 48,
        8, 6, 49, 5, 50, 21, 56, 52, 3, 34, 44, 20, 36, 63, 43, 23,
        54, 58, 55, 19, 26, 51, 46, 57, 17, 7, 39, 42, 13, 18, 4, 15,
        24, 25, 31, 16, 12, 14, 35, 10, 40, 28, 41, 33, 38, 11, 22, 53,
    ), dtype=np.int64),
}

def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode('utf-8'))
    digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()

def _path_endpoint_for_modality(direction: int, modality: int) -> int:
    direction = int(direction)
    modality = int(modality)
    if direction == PATH_EEG_TO_MEG:
        if modality == MODALITY_EEG:
            return PATH_ENDPOINT_SOURCE
        if modality == MODALITY_MEG:
            return PATH_ENDPOINT_TARGET
    elif direction == PATH_MEG_TO_EEG:
        if modality == MODALITY_MEG:
            return PATH_ENDPOINT_SOURCE
        if modality == MODALITY_EEG:
            return PATH_ENDPOINT_TARGET
    raise ValueError(f'Invalid V25 direction/modality pair: {direction}, {modality}')
PATH_METADATA: Final = {PATH_EEG_TO_MEG: {'name': 'EEG_to_MEG', 'source_modality': MODALITY_EEG, 'target_modality': MODALITY_MEG, 'source_name': 'EEG', 'target_name': 'MEG'}, PATH_MEG_TO_EEG: {'name': 'MEG_to_EEG', 'source_modality': MODALITY_MEG, 'target_modality': MODALITY_EEG, 'source_name': 'MEG', 'target_name': 'EEG'}}

def path_endpoint_for_modality(direction: int, modality: int) -> int:
    return _path_endpoint_for_modality(direction, modality)

def path_metadata() -> dict[int, dict[str, object]]:
    return {int(key): dict(value) for (key, value) in PATH_METADATA.items()}


def _orthogonal_columns(seed: int, count: int) -> np.ndarray:
    """Create exact centered axes from a signed Walsh--Hadamard basis."""
    if count >= CHANNELS or CHANNELS & (CHANNELS - 1):
        raise ValueError('The coordinate registry requires a power-of-two channel count and count < channels')
    rng = np.random.default_rng(int(seed))
    hadamard = np.ones((1, 1), dtype=np.int8)
    while hadamard.shape[0] < CHANNELS:
        hadamard = np.block([[hadamard, hadamard], [hadamard, -hadamard]]).astype(np.int8, copy=False)
    row_ids = rng.permutation(np.arange(1, CHANNELS, dtype=np.int64))[:count]
    signs = rng.choice(np.asarray((-1, 1), dtype=np.int8), size=count, replace=True)
    axes = hadamard[row_ids].astype(np.float64) * signs[:, None].astype(np.float64)
    if not np.array_equal(axes.mean(axis=1), np.zeros(count, dtype=np.float64)):
        raise AssertionError('Coordinate axes lost exact centering')
    if not np.array_equal(axes @ axes.T, np.eye(count, dtype=np.float64) * float(CHANNELS)):
        raise AssertionError('Coordinate axes lost exact orthogonality')
    return axes


@lru_cache(maxsize=1)
def _registry_arrays_cached() -> dict[str, np.ndarray]:
    """Build the fixed projection dictionary without reading participant data."""
    per_modality_count = 28
    raw = {
        'direct_task': np.empty((2, BANDS, 2, CHANNELS), dtype=np.float64),
        'pair_task': np.empty((2, 2, BANDS, CHANNELS), dtype=np.float64),
        'path_task': np.empty((2, 2, BANDS, CHANNELS), dtype=np.float64),
        'anchor_task': np.empty((2, CLASSES, CHANNELS), dtype=np.float64),
        'controller': np.empty((2, len(CONTROLLER_NAMES), CHANNELS), dtype=np.float64),
        'branch': np.empty((2, BRANCHES, CHANNELS), dtype=np.float64),
        'anchor_route': np.empty((2, len(ANCHOR_ROUTE_NAMES), CHANNELS), dtype=np.float64),
        'background': np.empty((2, BANDS, CHANNELS), dtype=np.float64),
        'decoy': np.empty((2, BANDS, CHANNELS), dtype=np.float64),
    }
    for modality in (MODALITY_EEG, MODALITY_MEG):
        axes = _orthogonal_columns(2026102501 + 1009 * modality, per_modality_count)
        cursor = 0

        def take(shape: tuple[int, ...]) -> np.ndarray:
            nonlocal cursor
            count = int(np.prod(shape))
            values = axes[cursor:cursor + count].reshape(*shape, CHANNELS)
            cursor += count
            return values

        raw['direct_task'][modality] = take((BANDS, 2))
        raw['pair_task'][modality] = take((2, BANDS))
        for direction in (PATH_EEG_TO_MEG, PATH_MEG_TO_EEG):
            endpoint = _path_endpoint_for_modality(direction, modality)
            raw['path_task'][direction, endpoint] = take((BANDS,))
        raw['anchor_task'][modality] = take((CLASSES,))
        raw['controller'][modality] = take((len(CONTROLLER_NAMES),))
        raw['branch'][modality] = take((BRANCHES,))
        raw['anchor_route'][modality] = take((len(ANCHOR_ROUTE_NAMES),))
        raw['background'][modality] = take((BANDS,))
        raw['decoy'][modality] = take((BANDS,))
        if cursor != per_modality_count:
            raise AssertionError((cursor, per_modality_count))

    result = {name: values.copy() for name, values in raw.items()}
    for modality in (MODALITY_EEG, MODALITY_MEG):
        order = CHANNEL_PERMUTATIONS[modality]
        if not np.array_equal(np.sort(order), np.arange(CHANNELS, dtype=np.int64)):
            raise AssertionError('Invalid channel permutation')
        for name, values in raw.items():
            if name == 'path_task':
                for direction in (PATH_EEG_TO_MEG, PATH_MEG_TO_EEG):
                    endpoint = _path_endpoint_for_modality(direction, modality)
                    result[name][direction, endpoint] = values[direction, endpoint][..., order]
            else:
                result[name][modality] = values[modality][..., order]
    return result

def registry_arrays(*, copy: bool=True) -> dict[str, np.ndarray]:
    cached = _registry_arrays_cached()
    return {name: np.array(cached[name], copy=True) if copy else cached[name] for name in REGISTRY_ARRAY_ORDER}

def task_maps(*, copy: bool=True) -> dict[str, np.ndarray]:
    arrays = registry_arrays(copy=copy)
    return {name: arrays[name] for name in ('direct_task', 'pair_task', 'path_task', 'anchor_task')}

def control_maps(*, copy: bool=True) -> dict[str, np.ndarray]:
    arrays = registry_arrays(copy=copy)
    return {name: arrays[name] for name in ('controller', 'branch', 'anchor_route', 'background', 'decoy')}

def path_map_for_modality(direction: int, modality: int, band: int) -> np.ndarray:
    direction = int(direction)
    modality = int(modality)
    band = int(band)
    if not 0 <= band < BANDS:
        raise ValueError(f'Invalid V25 band: {band}')
    endpoint = _path_endpoint_for_modality(direction, modality)
    return np.array(_registry_arrays_cached()['path_task'][direction, endpoint, band], copy=True)

def _array_manifest() -> list[dict[str, object]]:
    arrays = _registry_arrays_cached()
    return [{'name': name, 'shape': list(arrays[name].shape), 'dtype': str(arrays[name].dtype), 'sha256': _array_hash(arrays[name])} for name in REGISTRY_ARRAY_ORDER]

def registry_sha256() -> str:
    contract = {'version': COORDINATE_REGISTRY_VERSION, 'channels': CHANNELS, 'bands': BANDS, 'windows': WINDOWS, 'modalities': list(MODALITY_NAMES), 'branches': list(BRANCH_NAMES), 'experts': list(EXPERT_NAMES), 'paths': path_metadata(), 'controller': list(CONTROLLER_NAMES), 'arrays': _array_manifest()}
    encoded = json.dumps(contract, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()
REGISTRY_SHA256: Final = registry_sha256()

def _axes_for_modality(modality: int) -> tuple[np.ndarray, list[str]]:
    arrays = _registry_arrays_cached()
    values: list[np.ndarray] = []
    names: list[str] = []
    for band in range(BANDS):
        for bit in range(2):
            values.append(arrays['direct_task'][modality, band, bit])
            names.append(f'direct/{MODALITY_NAMES[modality]}/band{band}/bit{bit}')
    for expert in range(2):
        for band in range(BANDS):
            values.append(arrays['pair_task'][modality, expert, band])
            names.append(f'pair/{MODALITY_NAMES[modality]}/{EXPERT_NAMES[expert]}/band{band}')
    for direction in (PATH_EEG_TO_MEG, PATH_MEG_TO_EEG):
        endpoint = _path_endpoint_for_modality(direction, modality)
        for band in range(BANDS):
            values.append(arrays['path_task'][direction, endpoint, band])
            names.append(f'path/{PATH_NAMES[direction]}/{PATH_ENDPOINT_NAMES[endpoint]}/band{band}')
    for label in range(CLASSES):
        values.append(arrays['anchor_task'][modality, label])
        names.append(f'anchor/{MODALITY_NAMES[modality]}/class{label}')
    for (index, name) in enumerate(CONTROLLER_NAMES):
        values.append(arrays['controller'][modality, index])
        names.append(f'controller/{MODALITY_NAMES[modality]}/{name}')
    for branch in range(BRANCHES):
        values.append(arrays['branch'][modality, branch])
        names.append(f'branch/{MODALITY_NAMES[modality]}/{BRANCH_NAMES[branch]}')
    for route in range(len(ANCHOR_ROUTE_NAMES)):
        values.append(arrays['anchor_route'][modality, route])
        names.append(f'anchor-route/{MODALITY_NAMES[modality]}/{ANCHOR_ROUTE_NAMES[route]}')
    for band in range(BANDS):
        values.append(arrays['background'][modality, band])
        names.append(f'background/{MODALITY_NAMES[modality]}/band{band}')
    for band in range(BANDS):
        values.append(arrays['decoy'][modality, band])
        names.append(f'decoy/{MODALITY_NAMES[modality]}/band{band}')
    return (np.stack(values), names)

def span_audit() -> dict[str, object]:
    per_modality: dict[str, object] = {}
    global_max = 0.0
    for modality in (MODALITY_EEG, MODALITY_MEG):
        (axes, names) = _axes_for_modality(modality)
        normalized = axes / np.linalg.norm(axes, axis=-1, keepdims=True)
        gram = normalized @ normalized.T
        residual = gram - np.eye(len(axes), dtype=np.float64)
        max_offdiag = float(np.max(np.abs(residual)))
        max_mean = float(np.max(np.abs(axes.mean(axis=-1))))
        max_rms_error = float(np.max(np.abs(np.sqrt(np.mean(axes ** 2, axis=-1)) - 1.0)))
        per_modality[MODALITY_NAMES[modality]] = {'axis_count': len(names), 'max_abs_off_diagonal_cosine': max_offdiag, 'max_abs_axis_mean': max_mean, 'max_abs_rms_error': max_rms_error, 'axis_names': names}
        global_max = max(global_max, max_offdiag)
    paths = path_metadata()
    if paths[PATH_EEG_TO_MEG]['source_modality'] != MODALITY_EEG or paths[PATH_EEG_TO_MEG]['target_modality'] != MODALITY_MEG:
        raise AssertionError('EEG_to_MEG path metadata is reversed')
    if paths[PATH_MEG_TO_EEG]['source_modality'] != MODALITY_MEG or paths[PATH_MEG_TO_EEG]['target_modality'] != MODALITY_EEG:
        raise AssertionError('MEG_to_EEG path metadata is reversed')
    return {'registry_version': COORDINATE_REGISTRY_VERSION, 'registry_sha256': REGISTRY_SHA256, 'max_abs_off_diagonal_cosine': global_max, 'per_modality': per_modality, 'path_metadata': paths}

def registry_metadata() -> dict[str, object]:
    return {'coordinate_registry_version': COORDINATE_REGISTRY_VERSION, 'coordinate_registry_sha256': REGISTRY_SHA256, 'array_manifest': _array_manifest(), 'path_metadata': path_metadata(), 'span_audit': span_audit()}

def self_test() -> dict[str, object]:
    arrays_a = registry_arrays(copy=True)
    arrays_b = registry_arrays(copy=True)
    if tuple(arrays_a) != REGISTRY_ARRAY_ORDER:
        raise AssertionError('Unexpected V25 registry array order')
    for name in REGISTRY_ARRAY_ORDER:
        if not np.array_equal(arrays_a[name], arrays_b[name]):
            raise AssertionError(f'V25 registry is not deterministic: {name}')
    audit = span_audit()
    if float(audit['max_abs_off_diagonal_cosine']) > 2e-12:
        raise AssertionError(f'V25 coordinate span is insufficiently orthogonal: {audit}')
    for (direction, expected_source, expected_target) in ((PATH_EEG_TO_MEG, MODALITY_EEG, MODALITY_MEG), (PATH_MEG_TO_EEG, MODALITY_MEG, MODALITY_EEG)):
        metadata = PATH_METADATA[direction]
        if metadata['source_modality'] != expected_source or metadata['target_modality'] != expected_target:
            raise AssertionError('V25 path metadata source/target mismatch')
        if path_endpoint_for_modality(direction, expected_source) != PATH_ENDPOINT_SOURCE:
            raise AssertionError('V25 source endpoint lookup failed')
        if path_endpoint_for_modality(direction, expected_target) != PATH_ENDPOINT_TARGET:
            raise AssertionError('V25 target endpoint lookup failed')
    if registry_sha256() != REGISTRY_SHA256:
        raise AssertionError('V25 registry hash is not stable')
    return {'status': 'PASS', 'coordinate_registry_version': COORDINATE_REGISTRY_VERSION, 'coordinate_registry_sha256': REGISTRY_SHA256, 'per_modality_axis_count': 28, 'max_abs_off_diagonal_cosine': audit['max_abs_off_diagonal_cosine'], 'path_names': list(PATH_NAMES)}
if __name__ == '__main__':
    print(json.dumps(self_test(), indent=2, sort_keys=True))
