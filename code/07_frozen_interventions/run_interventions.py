from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
import frozen_helpers as H
import numpy as np
import torch
# ``selection_shuffle`` is a compatibility key for within-class reassignment of
# fitted selection states. The task-code array is never modified.
CONDITIONS = ('clean', 'selection_shuffle', 'pair_meg_shuffle', 'both_shuffle', 'controller_shuffle', 'branch_shuffle', 'pair_eeg_shuffle', 'pair_joint_unaligned', 'pair_joint_realigned')
SEED = 2026102211
REPEATS = 100
TOL = 2e-05

def derangement(n, rng):
    if n < 2:
        raise ValueError('A derangement needs at least two elements')
    identity = np.arange(n)
    while True:
        candidate = rng.permutation(n)
        if np.all(candidate != identity):
            return candidate

def make_plan(y, group, fold, repeats=REPEATS):
    y = np.asarray(y)
    donors = np.empty((repeats, len(y)), dtype=np.int64)
    windows = np.empty((repeats, len(y), 8), dtype=np.int64)
    for r in range(repeats):
        rng_s = np.random.default_rng(np.random.SeedSequence([SEED, group, fold, r, 0]))
        rng_p = np.random.default_rng(np.random.SeedSequence([SEED, group, fold, r, 1]))
        for label in np.unique(y):
            ids = np.flatnonzero(y == label)
            donors[r, ids] = ids[derangement(len(ids), rng_s)]
        for trial in range(len(y)):
            windows[r, trial] = derangement(8, rng_p)
    assert np.all(y[donors] == y[None])
    assert np.all(donors != np.arange(len(y))[None])
    assert np.all(np.sort(donors, axis=1) == np.arange(len(y)))
    assert np.all(np.sort(windows, axis=-1) == np.arange(8))
    assert np.all(windows != np.arange(8))
    return (donors, windows)

def window_take(value, order):
    return torch.gather(value, 1, order[..., None].expand(-1, -1, value.shape[-1]))

@torch.no_grad()
def extract_bank(model, eeg, meg, clean):
    bank = {'e': clean['direct_eeg_token'], 'm': clean['direct_meg_token'], 'f': clean['fusion_token'], 'state': clean['controller_state'], 'marker': clean['branch_marker'], 'pa': clean['anchor_probability'], 'g': clean['mix_weight'], 'pe': model.coordinates.pair_features(eeg, 0), 'pm': model.coordinates.pair_features(meg, 1)}
    return {k: v.detach().clone() for (k, v) in bank.items()}

@torch.no_grad()
def forward_bank(model, bank, condition='clean', donors=None, windows=None):
    (state, marker, pe, pm) = (bank[k] for k in ('state', 'marker', 'pe', 'pm'))
    if condition in ('selection_shuffle', 'both_shuffle', 'controller_shuffle'):
        state = state[donors]
    if condition in ('selection_shuffle', 'both_shuffle', 'branch_shuffle'):
        marker = marker[donors]
    membership = model._controller_membership(state)
    role = model._role_support(membership)
    expert = model._interaction_weights(membership, state)
    (local, _, _, _) = model._temporal_weights(role, state)
    (beta, _, _) = model._branch_weights(marker)
    if condition in ('pair_meg_shuffle', 'both_shuffle', 'pair_joint_unaligned', 'pair_joint_realigned'):
        pm = window_take(pm, windows)
    if condition in ('pair_eeg_shuffle', 'pair_joint_unaligned', 'pair_joint_realigned'):
        pe = window_take(pe, windows)
    if condition == 'pair_joint_realigned':
        expert = window_take(expert, windows)
        local = local.clone()
        local[..., 2:3] = window_take(local[..., 2:3], windows)
    pair = model._pair_token(pe, pm, expert)
    tokens = (bank['e'], bank['m'], pair, bank['f'])
    heads = (model.eeg_head, model.meg_head, model.pair_head, model.fusion_head)
    z = torch.stack([head(model._weighted_mean(token, local[..., k])) for (k, (token, head)) in enumerate(zip(tokens, heads))], 1)
    ps = torch.softmax((beta[..., None] * z).sum(1), -1)
    p = bank['g'] * ps + (1 - bank['g']) * bank['pa']
    assert torch.isfinite(p).all()
    return {'probability': p, 'structured_probability': ps, 'branch_logits': z, 'beta': beta, 'expert': expert, 'local': local, 'pair_token': pair}

def maximum_error(a, b):
    return float((a - b).abs().max().item())

def assert_close(a, b, label, tolerance=TOL):
    error = maximum_error(a, b)
    if error > tolerance:
        raise AssertionError(f'{label}: {error}')
    return error

@torch.no_grad()
def evaluate_fold(model, bank, clean, y, group, fold, repeats):
    n = len(y)
    (donors, windows) = make_plan(y, group, fold, repeats)
    base = forward_bank(model, bank)
    errors = {'bank_forward_probability': assert_close(base['probability'], clean['log_probability'].exp(), 'bank reconstruction'), 'bank_forward_branch_logits': assert_close(base['branch_logits'], clean['branch_logits'], 'head reconstruction'), 'bank_forward_beta': assert_close(base['beta'], clean['branch_weights'], 'beta reconstruction'), 'bank_forward_expert': assert_close(base['expert'], clean['interaction_weights'], 'expert reconstruction'), 'bank_forward_temporal': assert_close(base['local'], clean['local_weights'], 'window reconstruction')}
    assert torch.equal(base['probability'].argmax(-1), clean['log_probability'].argmax(-1))
    expanded = {k: v[None].expand(repeats, *v.shape).reshape(repeats * n, *v.shape[1:]) for (k, v) in bank.items()}
    donor_t = torch.as_tensor((donors + np.arange(repeats)[:, None] * n).reshape(-1), device=bank['e'].device)
    window_t = torch.as_tensor(windows.reshape(-1, 8), device=bank['e'].device)
    outputs = [forward_bank(model, expanded, c, donor_t, window_t) for c in CONDITIONS]
    reference = outputs[0]
    errors['expanded_batch_replay'] = assert_close(reference['probability'], base['probability'].repeat(repeats, 1), 'expanded replay')
    assert torch.equal(reference['probability'].argmax(-1), base['probability'].argmax(-1).repeat(repeats))
    errors['joint_realigned_probability'] = assert_close(outputs[8]['probability'], reference['probability'], 'negative control')
    errors['joint_realigned_pair_logits'] = assert_close(outputs[8]['branch_logits'][:, 2], reference['branch_logits'][:, 2], 'negative pair head')
    assert torch.equal(outputs[8]['probability'].argmax(-1), reference['probability'].argmax(-1))
    for j in (2, 6, 7, 8):
        errors[f'{CONDITIONS[j]}_other_heads'] = assert_close(outputs[j]['branch_logits'][:, [0, 1, 3]], reference['branch_logits'][:, [0, 1, 3]], 'isolated non-Pair heads', 0.0)
        errors[f'{CONDITIONS[j]}_beta'] = assert_close(outputs[j]['beta'], reference['beta'], 'isolated beta', 0.0)
        if j != 8:
            for k in ('expert', 'local'):
                assert_close(outputs[j][k], reference[k], f'isolated {k}', 0.0)
    assert_close(outputs[5]['branch_logits'], reference['branch_logits'], 'branch-only content', 0.0)
    assert_close(outputs[4]['beta'], reference['beta'], 'controller-only beta', 0.0)
    for k in ('beta', 'expert', 'local'):
        assert_close(outputs[1][k], outputs[3][k], f'factorial {k}', 0.0)
    arrays = {}
    for key in ('probability', 'structured_probability', 'branch_logits', 'beta'):
        value = torch.stack([row[key] for row in outputs], 1)
        value = value.reshape(repeats, n, len(CONDITIONS), *value.shape[2:]).transpose(0, 1)
        arrays[key] = value.cpu().numpy()
    arrays['selection_donor_local'] = donors.T
    arrays['window_permutation'] = windows.transpose(1, 0, 2)
    arrays['pa'] = bank['pa'].cpu().numpy()
    arrays['g'] = bank['g'].cpu().numpy()
    bank_hashes = {k: H.array_sha(v.cpu().numpy()) for (k, v) in bank.items()}
    errors['probability_normalization'] = float(np.max(np.abs(arrays['probability'].sum(-1) - 1)))
    assert errors['probability_normalization'] < 2e-06
    return (arrays, {'bank_sha256': bank_hashes, 'fixed_anchor_and_g': True, 'isolation_assertions': 'PASS'})
