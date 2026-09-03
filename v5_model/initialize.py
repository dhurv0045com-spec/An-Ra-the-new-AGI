"""Deterministic initialization for the V5 core.

Embedding, Q/K/V, gate, and up tensors draw ``Normal(0, 0.02)``.
Attention-output and FFN-down draw ``Normal(0, 0.02/sqrt(2L))``. RMSNorm and
affine QK scales start at 1. There are no bias tensors. Tied embedding/output
storage is asserted after initialization.
"""

from __future__ import annotations

import math
from typing import Any

NORMAL_STD = 0.02


def residual_output_std(*, layers: int) -> float:
    """Return the exact 1/sqrt(2L)-scaled residual output standard deviation."""

    if layers <= 0:
        raise ValueError("layer count must be positive")
    return NORMAL_STD / math.sqrt(2 * layers)


def initialize_module(module: Any, *, layers: int, torch_module: Any) -> None:
    """Apply the frozen initialization in place under a seeded generator."""

    torch = torch_module
    nn = torch.nn
    std = residual_output_std(layers=layers)
    for child in module.modules():
        if isinstance(child, (nn.Linear, nn.Embedding)):
            nn.init.normal_(child.weight, std=NORMAL_STD)
    for block in module.blocks:
        nn.init.normal_(block.attention.output.weight, std=std)
        nn.init.normal_(block.down.weight, std=std)


__all__ = ["NORMAL_STD", "initialize_module", "residual_output_std"]
