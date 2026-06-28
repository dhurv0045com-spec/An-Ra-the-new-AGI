"""Canonical AN-RA V3 architecture contracts and exact parameter accounting."""

from __future__ import annotations

from dataclasses import dataclass

CANONICAL_VOCAB_SIZE = 8209


@dataclass(frozen=True)
class ArchitectureContract:
    name: str
    vocab_size: int
    d_model: int
    n_layers: int
    n_query_heads: int
    n_kv_heads: int
    d_ff: int
    context_length: int = 2048
    esv_dim: int = 64
    mod_layers: tuple[int, ...] = ()

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_query_heads

    def transformer_parameters(self) -> int:
        attention = (
            self.d_model * self.n_query_heads * self.head_dim
            + 2 * self.d_model * self.n_kv_heads * self.head_dim
            + self.d_model * self.d_model
        )
        ffn = 3 * self.d_model * self.d_ff
        norms = 2 * self.d_model
        blocks = self.n_layers * (attention + ffn + norms)
        return self.vocab_size * self.d_model + blocks + self.d_model

    def full_system_parameters(self) -> int:
        esv_predictor = self.esv_dim * 3 + 3
        rims = self.n_layers * (self.esv_dim * self.d_model + 1)
        mcr = len(self.mod_layers) * (self.d_model + 4)
        residual_depth = self.n_layers
        dstp_temperatures = self.n_layers
        return (
            self.transformer_parameters()
            + esv_predictor
            + rims
            + mcr
            + residual_depth
            + dstp_temperatures
        )


FRONTIER = ArchitectureContract(
    name="anra-frontier-500m",
    vocab_size=CANONICAL_VOCAB_SIZE,
    d_model=1280,
    n_layers=28,
    n_query_heads=16,
    n_kv_heads=4,
    d_ff=3456,
    context_length=1024,
    mod_layers=(4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26),
)

DRAFT = ArchitectureContract(
    name="anra-draft",
    vocab_size=CANONICAL_VOCAB_SIZE,
    d_model=256,
    n_layers=8,
    n_query_heads=4,
    n_kv_heads=2,
    d_ff=704,
)


def verify_canonical_counts() -> dict[str, int]:
    counts = {
        "frontier_transformer": FRONTIER.transformer_parameters(),
        "frontier_full": FRONTIER.full_system_parameters(),
        "draft_transformer": DRAFT.transformer_parameters(),
        "draft_full": DRAFT.transformer_parameters() + DRAFT.esv_dim * 3 + 3,
    }
    expected = {
        "frontier_transformer": 496_857_600,
        "frontier_full": 499_167_047,
        "draft_transformer": 8_004_096,
        "draft_full": 8_004_291,
    }
    if counts != expected:
        raise AssertionError(f"Canonical parameter contract mismatch: {counts} != {expected}")
    return counts
