"""Stable public entry point for the CAMBER model implementation.

The underlying module name retains the experiment-version tag used by the
saved checkpoints.  New code should import the aliases exposed here.
"""

from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent
for directory in (CODE, CODE / "00_shared" / "models"):
    sys.path.insert(0, str(directory))

from hierarchical_pair_anchor_models_v25 import (  # noqa: E402
    FULL_VARIANT,
    HierarchicalPairAnchorModelV25,
    make_model,
)

CAMBERModel = HierarchicalPairAnchorModelV25
CAMBER_VARIANT = FULL_VARIANT


def build_camber(dropout: float = 0.0) -> CAMBERModel:
    """Construct the full CAMBER architecture used in the main analysis."""

    return make_model(CAMBER_VARIANT, dropout=dropout)


__all__ = ["CAMBERModel", "CAMBER_VARIANT", "build_camber"]
