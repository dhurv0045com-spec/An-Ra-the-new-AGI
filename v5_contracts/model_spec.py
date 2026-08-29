"""Pure model configuration and exact parameter accounting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ParameterReceipt:
    embedding: int
    attention_per_layer: int
    ffn_per_layer: int
    block_norms_per_layer: int
    qk_norms_per_layer: int
    block_total: int
    all_blocks: int
    final_norm: int
    total: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    schema: str
    family: str
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
    tied_embeddings: bool
    qk_norm: bool
    qk_norm_affine: bool
    linear_bias: bool
    dropout: float

    def assert_valid(self) -> None:
        integers = (
            self.vocabulary_size,
            self.width,
            self.layers,
            self.query_heads,
            self.kv_heads,
            self.head_dimension,
            self.ffn_width,
            self.context_length,
        )
        if any(value <= 0 for value in integers):
            raise ValueError("all dimensions must be positive")
        if self.width != self.query_heads * self.head_dimension:
            raise ValueError("width must equal query_heads * head_dimension")
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        if not self.tied_embeddings:
            raise ValueError("this receipt currently supports tied embeddings only")
        if self.qk_norm_affine and not self.qk_norm:
            raise ValueError("affine QK norm requires QK normalization")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def parameter_receipt(self) -> ParameterReceipt:
        self.assert_valid()
        kv_width = self.kv_heads * self.head_dimension
        q_width = self.query_heads * self.head_dimension
        embedding = self.vocabulary_size * self.width
        attention = self.width * q_width + 2 * self.width * kv_width + q_width * self.width
        ffn = 3 * self.width * self.ffn_width
        block_norms = 2 * self.width
        qk_norms = q_width + kv_width if self.qk_norm and self.qk_norm_affine else 0
        block = attention + ffn + block_norms + qk_norms
        all_blocks = self.layers * block
        final_norm = self.width
        return ParameterReceipt(
            embedding=embedding,
            attention_per_layer=attention,
            ffn_per_layer=ffn,
            block_norms_per_layer=block_norms,
            qk_norms_per_layer=qk_norms,
            block_total=block,
            all_blocks=all_blocks,
            final_norm=final_norm,
            total=embedding + all_blocks + final_norm,
        )

    def canonical(self) -> dict[str, object]:
        self.assert_valid()
        return asdict(self)

    def sha256(self) -> str:
        payload = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


V5A_250M = ModelSpec(
    schema="anra-v5-model-spec/v1",
    family="dense-decoder-transformer",
    vocabulary_size=24_576,
    width=896,
    layers=26,
    query_heads=14,
    kv_heads=7,
    head_dimension=64,
    ffn_width=2_368,
    context_length=4_096,
    rope_base=10_000.0,
    norm_epsilon=1e-5,
    tied_embeddings=True,
    qk_norm=True,
    qk_norm_affine=True,
    linear_bias=False,
    dropout=0.0,
)
