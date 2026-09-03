"""Validated model configuration accepted from a frozen ModelSpec."""

from __future__ import annotations

from dataclasses import dataclass

from v5_contracts.model_spec import ModelSpec


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocabulary_size: int
    width: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dimension: int
    ffn_width: int
    context_length: int
    rope_base: float
    norm_epsilon: float
    qk_norm: bool
    qk_norm_affine: bool
    qk_norm_epsilon: float


def from_spec(spec: ModelSpec, *, qk_norm_epsilon: float) -> ModelConfig:
    """Accept only a validated, bias-free, dropout-free dense-decoder spec."""

    spec.assert_valid()
    if spec.linear_bias or spec.dropout != 0.0 or not spec.tied_embeddings:
        raise ValueError("V5 accepts only bias-free, dropout-free, tied-embedding specs")
    if spec.head_dimension % 2:
        raise ValueError("V5 requires an even head dimension for pairwise RoPE")
    if not qk_norm_epsilon > 0:
        raise ValueError("QK normalization epsilon must be positive")
    return ModelConfig(
        vocabulary_size=spec.vocabulary_size,
        width=spec.width,
        layers=spec.layers,
        query_heads=spec.query_heads,
        kv_heads=spec.kv_heads,
        head_dimension=spec.head_dimension,
        ffn_width=spec.ffn_width,
        context_length=spec.context_length,
        rope_base=spec.rope_base,
        norm_epsilon=spec.norm_epsilon,
        qk_norm=spec.qk_norm,
        qk_norm_affine=spec.qk_norm_affine,
        qk_norm_epsilon=qk_norm_epsilon,
    )


__all__ = ["ModelConfig", "from_spec"]
