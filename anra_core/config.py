from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class CoreConfig:
    architecture_version: str = "anra_v4_rope_interleaved_v1"
    vocab_size: int = 32_768
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3
    d_model: int = 896
    n_layers: int = 18
    n_heads: int = 14
    n_kv_heads: int = 2
    head_dim: int = 64
    d_ff: int = 2_432
    block_size: int = 2_048
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    rope_base: float = 10_000.0
    base_seq_len: int = 2_048
    target_seq_len: int = 2_048
    sliding_window: int = 1_024
    full_attention_every: int = 4
    qk_norm: bool = True
    use_mtp: bool = False
    use_moe: bool = False
    initialization_scheme: str = "depth_scaled_residual_v1"

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "d_ff": self.d_ff,
            "block_size": self.block_size,
            "sliding_window": self.sliding_window,
            "full_attention_every": self.full_attention_every,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"Core dimensions must be positive: {invalid}")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError("n_heads * head_dim must equal d_model")
        if self.sliding_window > self.block_size:
            raise ValueError("sliding_window cannot exceed block_size")
        if self.dropout != 0.0:
            raise ValueError("standalone V4 Core requires dropout=0.0")
        if self.use_mtp or self.use_moe:
            raise ValueError("standalone dense V4 Core cannot enable MTP or MoE")

    @property
    def dense_parameter_count(self) -> int:
        embedding = self.vocab_size * self.d_model
        attention = (
            self.d_model * self.n_heads * self.head_dim
            + 2 * self.d_model * self.n_kv_heads * self.head_dim
            + self.n_heads * self.head_dim * self.d_model
        )
        feed_forward = 3 * self.d_model * self.d_ff
        norms = 2 * self.d_model
        return embedding + self.n_layers * (attention + feed_forward + norms) + self.d_model

    def immutable_fields(self) -> dict[str, object]:
        return asdict(self)

    @property
    def architecture_sha256(self) -> str:
        payload = {
            "schema": "anra-core-architecture/v1",
            **self.immutable_fields(),
            "dense_parameter_count": self.dense_parameter_count,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


CANONICAL_CONFIG = CoreConfig()
