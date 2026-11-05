from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


def sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    """Hash an array together with its dtype and shape."""
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    """Return a stable digest for a PyTorch state mapping."""
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(array_hash(value.detach().cpu().numpy()).encode())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    """Compute the classification summaries saved by the experiment runner."""
    from sklearn.metrics import confusion_matrix, f1_score

    prediction = probability.argmax(1)
    confusion = confusion_matrix(y, prediction, labels=[0, 1, 2])
    return {
        "n": len(y),
        "correct": int((prediction == y).sum()),
        "accuracy_percent": float(100 * np.mean(prediction == y)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "confusion_matrix": confusion.tolist(),
        "recall_percent": (100 * np.diag(confusion) / confusion.sum(1)).tolist(),
        "cross_entropy": float(
            -np.log(np.maximum(probability[np.arange(len(y)), y], 1e-12)).mean()
        ),
    }
