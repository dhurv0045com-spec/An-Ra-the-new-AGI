"""Canonical AN-RA V4 architecture contract and exact parameter accounting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

CANONICAL_VOCAB_SIZE = 32_768


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

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("architecture name must not be empty")
        if min(self.vocab_size, self.d_model, self.n_layers, self.d_ff) <= 0:
            raise ValueError("architecture dimensions must be positive")
        if self.n_query_heads <= 0 or self.d_model % self.n_query_heads:
            raise ValueError("d_model must be divisible by positive n_query_heads")
        if self.n_kv_heads <= 0 or self.n_query_heads % self.n_kv_heads:
            raise ValueError("n_query_heads must be divisible by positive n_kv_heads")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if len(set(self.mod_layers)) != len(self.mod_layers) or any(
            layer < 0 or layer >= self.n_layers for layer in self.mod_layers
        ):
            raise ValueError("mod_layers must be unique zero-based layer indices")

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
        layer_temperature_biases = self.n_layers
        return (
            self.transformer_parameters()
            + esv_predictor
            + rims
            + mcr
            + residual_depth
            + dstp_temperatures
            + layer_temperature_biases
        )

    def identity_payload(self) -> dict[str, object]:
        """Return the stable architecture identity used by lineage manifests."""
        return {
            "schema_version": 1,
            "name": self.name,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_query_heads": self.n_query_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "d_ff": self.d_ff,
            "context_length": self.context_length,
            "esv_dim": self.esv_dim,
            "mod_layers": list(self.mod_layers),
            "transformer_parameters": self.transformer_parameters(),
            "full_system_parameters": self.full_system_parameters(),
        }

    def sha256(self) -> str:
        material = json.dumps(
            self.identity_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(material).hexdigest()


FRONTIER = ArchitectureContract(
    name="anra-v4-180m",
    vocab_size=CANONICAL_VOCAB_SIZE,
    d_model=896,
    n_layers=18,
    n_query_heads=14,
    n_kv_heads=2,
    d_ff=2432,
    context_length=2048,
    mod_layers=(4, 6, 8, 10, 12, 14, 16),
)

# This is a registered growth target, not a second canonical scratch model.
# It inherits the V4 vocabulary and learned state from FRONTIER through a
# verified growth manifest before a fresh optimizer is created.
GROWTH_500M = ArchitectureContract(
    name="anra-v4-500m-growth",
    vocab_size=CANONICAL_VOCAB_SIZE,
    d_model=1280,
    n_layers=27,
    n_query_heads=20,
    n_kv_heads=2,
    d_ff=3456,
    context_length=2048,
    mod_layers=tuple(range(4, 27, 2)),
)

GROWTH_500M_PARAMETER_COUNT = 499_880_031


def verify_canonical_counts() -> dict[str, int]:
    counts = {
        "frontier_transformer": FRONTIER.transformer_parameters(),
        "frontier_full": FRONTIER.full_system_parameters(),
    }
    expected = {
        "frontier_transformer": 180_093_312,
        "frontier_full": 181_132_071,
    }
    if counts != expected:
        raise AssertionError(f"Canonical parameter contract mismatch: {counts} != {expected}")
    return counts


def verify_growth_counts() -> dict[str, int]:
    counts = {
        "growth_500m_transformer": GROWTH_500M.transformer_parameters(),
        "growth_500m_full": GROWTH_500M.full_system_parameters(),
    }
    expected = {
        "growth_500m_transformer": 497_652_480,
        "growth_500m_full": GROWTH_500M_PARAMETER_COUNT,
    }
    if counts != expected:
        raise AssertionError(f"Growth parameter contract mismatch: {counts} != {expected}")
    return counts
