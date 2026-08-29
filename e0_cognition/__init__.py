"""ESOES E0: causal cognition benchmark research infrastructure.

This package is intentionally model-agnostic. It contains no V5 model,
trainer, optimizer, checkpoint loader, or dependency on VNext.
"""

from .contracts import CausalCase, CausalPair, PairKind, Split
from .evaluation_generators import build_evaluation_suite

__all__ = [
    "CausalCase",
    "CausalPair",
    "PairKind",
    "Split",
    "build_evaluation_suite",
]
