"""
anra/core/model.py
Re-exports CausalTransformerV2 from anra_brain.py into the anra package.
"""

from __future__ import annotations

from anra_brain import (  # noqa: F401
    BlockV2,
    CausalTransformerV2,
    CausalTransformerV3,
    MetacognitiveRouter,
    MoDRouter,
    MultiHeadAttentionV2,
    ResidualIdentityModulator,
    RMSNorm,
    RotaryEmbedding,
    RouterContext,
    SwiGLU,
)

__all__ = [
    "CausalTransformerV2",
    "CausalTransformerV3",
    "BlockV2",
    "MultiHeadAttentionV2",
    "RotaryEmbedding",
    "RMSNorm",
    "SwiGLU",
    "MoDRouter",
    "MetacognitiveRouter",
    "ResidualIdentityModulator",
    "RouterContext",
]
