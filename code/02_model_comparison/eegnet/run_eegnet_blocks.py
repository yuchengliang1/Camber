from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
from runtime import resolve_device
import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
import eegnet_baselines as baseline_module
from eegnet_baselines import LEGACY_SOURCE_SHA256, METHOD_NAMES, MODEL_SETTINGS, FrozenConcatFeatureFusion, array_sha256, block_splits, classification_metrics, file_sha256, load_public_data, make_eegnet, normalize_trials, parameter_counts, self_test, state_sha256

def _json(path: Path, payload: dict[str, Any]) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write('\n')

def _npz(path: Path, **arrays) -> None:
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)

def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _resolve_device(name: str) -> torch.device:
    device = resolve_device(name)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    sample = torch.ones((2, 2), device=device)
    if not torch.isfinite(sample @ sample).all():
        raise RuntimeError('Device computation check failed')
    return device

def _train_classifier(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, epochs: int, seed: int, lr: float, weight_decay: float, forward=None, parameters=None) -> tuple[list[dict[str, Any]], np.ndarray]:
    optimizer = torch.optim.AdamW(model.parameters() if parameters is None else parameters, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.04)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    (records, orders) = ([], [])
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(y), generator=generator)
        orders.append(order.numpy())
        total_loss = 0.0
        for start in range(0, len(order), 30):
            idx = order[start:start + 30].to(x.device)
            (xb, target) = (x[idx], y[idx])
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb) if forward is None else forward(xb)
            loss = loss_fn(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)
        records.append({'epoch': epoch + 1, 'training_loss': total_loss / len(y)})
    model.eval()
    return (records, np.stack(orders).astype(np.int64))

def _save_checkpoint(path: Path, model: nn.Module, spec: dict[str, Any], context: dict[str, Any]) -> None:
    payload = {'model_state_dict': {k: v.detach().cpu() for (k, v) in model.state_dict().items()}, 'model_spec': spec, 'context': context, 'state_sha256': state_sha256(model)}
    with path.open('xb') as handle:
        torch.save(payload, handle)
