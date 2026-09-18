"""Bounded hormonal-to-attention-scale projection (HORM-001 sibling)."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .hormonal_state import HORMONES

SCALE_BOUNDS = (0.8, 1.2)


@dataclass(frozen=True, slots=True)
class HormonalProjection:
    weights: tuple[float, ...]
    bound: float
    raw_alpha: float

    def __post_init__(self) -> None:
        if len(self.weights) != len(HORMONES):
            raise ValueError("projection needs exactly seven hormone weights")
        if not 0.0 < self.bound <= 0.2:
            raise ValueError("bound must be in (0, 0.2] so scale stays in [0.8, 1.2]")
        if not 0.0 <= self.raw_alpha <= 1.0:
            raise ValueError("raw_alpha must lie in [0, 1]; frozen overlay starts inert at 0")

    def scale(self, hormone_vector: tuple[float, ...]) -> float:
        """Return 1 + B*tanh(alpha * w . h), clamped to SCALE_BOUNDS."""

        if len(hormone_vector) != len(HORMONES):
            raise ValueError("hormone vector length mismatch")
        raw = sum(w * h for w, h in zip(self.weights, hormone_vector)) * self.raw_alpha
        value = 1.0 + self.bound * __import__("math").tanh(raw)
        return min(SCALE_BOUNDS[1], max(SCALE_BOUNDS[0], value))

    def receipt(self) -> dict[str, object]:
        import hashlib

        payload = json.dumps(
            {"weights": list(self.weights), "bound": self.bound,
             "raw_alpha": self.raw_alpha, "scale_bounds": list(SCALE_BOUNDS)},
            sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {"schema": "anra-horm-projection/v1",
                "sha256": hashlib.sha256(payload).hexdigest()}
