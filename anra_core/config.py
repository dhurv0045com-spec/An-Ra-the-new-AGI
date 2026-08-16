from __future__ import annotations

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
    sliding_window: int = 1_024
    full_attention_every: int = 4
    qk_norm: bool = True

    @property
    def dense_parameter_count(self) -> int:
        return 180_093_312

    def immutable_fields(self) -> dict[str, object]:
        return asdict(self)


CANONICAL_CONFIG = CoreConfig()
