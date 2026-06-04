"""
anra/core/model.py
Re-exports CausalTransformerV2 from the root anra_brain.py into the anra package.
This is the canonical import path: from anra.core.model import CausalTransformerV2
anra_brain.py at root is kept for backward compatibility only.
"""

from __future__ import annotations

from anra_brain import (  # noqa: F401
    BlockV2,
    CausalTransformerV2,
    MoDRouter,
    MultiHeadAttentionV2,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
)

__all__ = [
    "CausalTransformerV2",
    "BlockV2",
    "MultiHeadAttentionV2",
    "RotaryEmbedding",
    "RMSNorm",
    "SwiGLU",
    "MoDRouter",
]
