from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
LEGACY_SOURCE_SHA256 = {}
MODEL_SETTINGS = {'eeg': {'temporal_filters': 4, 'temporal_kernel': 31, 'separable_kernel': 15}, 'meg': {'temporal_filters': 8, 'temporal_kernel': 63, 'separable_kernel': 31}}
PUBLIC_REQUIRED_KEYS = ('eeg', 'meg', 'y', 'block', 'fs', 'trial_id')
METHOD_NAMES = ('eegnet_eeg', 'eegnet_meg', 'eegnet_decision', 'eegnet_feature_concat')

class EEGNet(nn.Module):

    def __init__(self, channels: int=64, classes: int=3, *, temporal_filters: int=4, temporal_kernel: int=31, separable_kernel: int=15):
        super().__init__()
        spatial_filters = temporal_filters * 2
        self.features = nn.Sequential(nn.AvgPool2d((1, 2)), nn.Conv2d(1, temporal_filters, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False), nn.BatchNorm2d(temporal_filters), nn.Conv2d(temporal_filters, spatial_filters, (channels, 1), groups=temporal_filters, bias=False), nn.BatchNorm2d(spatial_filters), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.3), nn.Conv2d(spatial_filters, spatial_filters, (1, separable_kernel), padding=(0, separable_kernel // 2), groups=spatial_filters, bias=False), nn.Conv2d(spatial_filters, 12, (1, 1), bias=False), nn.BatchNorm2d(12), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(0.3))
        self.classifier = nn.Linear(12 * 15, classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

def make_eegnet(modality: str) -> EEGNet:
    if modality not in MODEL_SETTINGS:
        raise ValueError("modality must be 'eeg' or 'meg'")
    return EEGNet(**MODEL_SETTINGS[modality])

class FrozenConcatFeatureFusion(nn.Module):

    def __init__(self, eeg: EEGNet | None=None, meg: EEGNet | None=None):
        super().__init__()
        self.eeg = make_eegnet('eeg') if eeg is None else eeg
        self.meg = make_eegnet('meg') if meg is None else meg
        for backbone in (self.eeg, self.meg):
            backbone.requires_grad_(False)
            backbone.eval()
        self.register_buffer('feature_mean', torch.zeros(360))
        self.register_buffer('feature_scale', torch.ones(360))
        self.classifier = nn.Linear(360, 3)

    def train(self, mode: bool=True):
        super().train(mode)
        self.eeg.eval()
        self.meg.eval()
        return self

    def concatenate(self, eeg, meg):
        with torch.no_grad():
            eeg_feature = self.eeg.features(eeg).flatten(1)
            meg_feature = self.meg.features(meg).flatten(1)
        return torch.cat((eeg_feature, meg_feature), dim=1)

    def logits_from_features(self, features):
        return self.classifier((features - self.feature_mean) / self.feature_scale)

    def forward(self, eeg, meg):
        return self.logits_from_features(self.concatenate(eeg, meg))

    def fit_feature_scaler(self, training_features: torch.Tensor) -> None:
        if training_features.ndim != 2 or training_features.shape[1] != 360:
            raise ValueError('Expected training features with shape (N,360)')
        if training_features.shape[0] < 2 or not torch.isfinite(training_features).all():
            raise ValueError('Feature scaling requires >=2 finite training examples')
        with torch.no_grad():
            self.feature_mean.copy_(training_features.mean(dim=0))
            self.feature_scale.copy_(training_features.std(dim=0, correction=0).clamp_min(1e-06))

def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda : handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()

def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for (name, tensor) in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(array_sha256(array).encode())
    return digest.hexdigest()

def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {'allocated': sum((p.numel() for p in model.parameters())), 'trainable': sum((p.numel() for p in model.parameters() if p.requires_grad))}

def normalize_trials(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 3 or x.shape[1:] != (64, 1000):
        raise ValueError('EEGNet expects (N,64,1000)')
    if not np.isfinite(x).all():
        raise ValueError('Input contains nonfinite values')
    x = np.asarray(x, dtype=np.float32)
    scale = x.std(axis=(1, 2), keepdims=True) + 1e-06
    return ((x / scale).astype(np.float32), scale.reshape(-1).astype(np.float32))

def validate_public_data(data: dict[str, np.ndarray], subject: int | None=None) -> None:
    if set(PUBLIC_REQUIRED_KEYS) - set(data):
        raise ValueError(f'Missing public keys: {sorted(set(PUBLIC_REQUIRED_KEYS) - set(data))}')
    for modality in ('eeg', 'meg'):
        if data[modality].shape != (150, 64, 1000) or not np.isfinite(data[modality]).all():
            raise ValueError(f'{modality} must be finite (150,64,1000)')
    if np.asarray(data['fs']).size != 1 or float(np.asarray(data['fs']).reshape(-1)[0]) != 250:
        raise ValueError('Expected fs=250')
    for key in ('y', 'block', 'trial_id'):
        if data[key].shape != (150,) or data[key].dtype.kind not in 'iu':
            raise ValueError(f'{key} must be an integer vector of length150')
    if set(np.unique(data['y']).tolist()) != {0, 1, 2}:
        raise ValueError('Expected class labels0,1,2')
    if set(np.unique(data['block']).tolist()) != set(range(5)):
        raise ValueError('Expected genuine block IDs0..4; do not reconstruct them from row number')
    if len(np.unique(data['trial_id'])) != 150:
        raise ValueError('trial_id must contain150 unique IDs')
    for block in range(5):
        idx = data['block'] == block
        if idx.sum() != 30:
            raise ValueError(f'Block{block} must contain30 trials')
    if subject is not None and 'subject_id' in data:
        if data['subject_id'].size != 1 or int(data['subject_id'].reshape(-1)[0]) != subject:
            raise ValueError('CLI subject does not match data subject_id')

def load_public_data(path: Path | str, subject: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in ('eeg', 'meg', 'y', 'block')}
        data['fs'] = np.asarray(archive['sampling_frequency_hz'], dtype=np.int64)
        data['trial_id'] = np.arange(len(data['y']), dtype=np.int64)
    validate_public_data(data, subject)
    return data

def block_splits(block: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = []
    for held_out in range(5):
        train = np.flatnonzero(block != held_out)
        test = np.flatnonzero(block == held_out)
        if len(train) != 120 or len(test) != 30 or np.intersect1d(train, test).size:
            raise ValueError('Invalid leave-one-block-out partition')
        splits.append((train, test))
    coverage = np.bincount(np.concatenate([test for (_, test) in splits]), minlength=len(block))
    if not np.array_equal(coverage, np.ones(len(block), dtype=np.int64)):
        raise ValueError('Every trial must have exactly one outer test prediction')
    return splits

def classification_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    if probability.shape != (len(y), 3) or not np.isfinite(probability).all():
        raise ValueError('Expected finite (N,3) probability matrix')
    if np.any(probability < -1e-06) or not np.allclose(probability.sum(axis=1), 1, atol=2e-05):
        raise ValueError('Invalid probability normalization')
    prediction = probability.argmax(axis=1)
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (y.astype(np.int64), prediction), 1)
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    accuracy = float(np.mean(prediction == y))
    return {'n': len(y), 'correct': int(np.sum(prediction == y)), 'accuracy': accuracy, 'accuracy_percent': 100 * accuracy, 'recall': recall.tolist(), 'recall_percent': (100 * recall).tolist(), 'macro_recall': float(recall.mean()), 'confusion_matrix': matrix.tolist()}

def self_test() -> dict[str, Any]:
    torch.set_num_threads(2)
    torch.manual_seed(84621)
    (eeg, meg) = (make_eegnet('eeg'), make_eegnet('meg'))
    eeg.eval()
    meg.eval()
    x = torch.randn(2, 1, 64, 1000)
    with torch.no_grad():
        assert eeg.features(x).flatten(1).shape == (2, 180)
        assert meg.features(x).flatten(1).shape == (2, 180)
        assert eeg(x).shape == meg(x).shape == (2, 3)
    counts_before_freeze = {'eeg': parameter_counts(eeg), 'meg': parameter_counts(meg)}
    assert counts_before_freeze['eeg']['allocated'] == 1443
    assert counts_before_freeze['meg']['allocated'] == 2831
    fusion = FrozenConcatFeatureFusion(eeg, meg)
    fusion.train()
    assert not fusion.eeg.training and (not fusion.meg.training)
    assert all((not p.requires_grad for p in fusion.eeg.parameters()))
    assert parameter_counts(fusion) == {'allocated': 5357, 'trainable': 1083}
    with torch.no_grad():
        train_feature = fusion.concatenate(x, x)
        fusion.fit_feature_scaler(train_feature)
        direct = fusion(x, x)
        cached = fusion.logits_from_features(train_feature)
        torch.testing.assert_close(direct, cached, rtol=0, atol=0)
        probability = torch.softmax(direct, dim=1).numpy()
    classification_metrics(np.array([0, 1]), probability)
    raw = np.random.default_rng(99).normal(size=(2, 64, 1000)).astype(np.float32)
    (normalized, scale) = normalize_trials(raw)
    np.testing.assert_array_equal(normalized, (raw / (raw.std(axis=(1, 2), keepdims=True) + 1e-06)).astype(np.float32))
    np.testing.assert_array_equal(scale, raw.std(axis=(1, 2)) + 1e-06)
    (normalized_first, _) = normalize_trials(raw[:1])
    np.testing.assert_array_equal(normalized_first, normalized[:1])
    mock = {'eeg': np.broadcast_to(raw[:1], (150, 64, 1000)), 'meg': np.broadcast_to(raw[1:], (150, 64, 1000)), 'y': np.tile(np.repeat(np.arange(3), 10), 5), 'block': np.repeat(np.arange(5), 30), 'trial_id': np.arange(150), 'fs': np.asarray(250), 'subject_id': np.asarray(41)}
    validate_public_data(mock, 41)
    splits = block_splits(mock['block'])
    assert len(splits) == 5 and all(((len(a), len(b)) == (120, 30) for (a, b) in splits))
    invalid = dict(mock)
    invalid['block'] = np.zeros(150, dtype=np.int64)
    try:
        validate_public_data(invalid, 41)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid block schema unexpectedly passed')
    return {'status': 'PASS', 'device': 'cpu QA only', 'parameter_counts': {**counts_before_freeze}, 'feature_dim_per_modality': 180, 'fusion_feature_dim': 360, 'outer_folds': 5, 'train_test_sizes': [120, 30]}
