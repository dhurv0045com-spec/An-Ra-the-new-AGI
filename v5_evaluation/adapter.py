"""Model-scoring adapter contract for V5 evaluation.

The adapter exposes three calls over an immutable checkpoint: candidate
scoring (candidate-suffix token log-probabilities only), free generation,
and constrained generation. It never trains, never mutates checkpoints, and
never sees sealed fixtures. scoring_certification audits the aggregation;
production mode stays null until a preregistered bias-resistant policy wins.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable


ADAPTER_SCHEMA = "anra-v5-model-adapter/v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


class ModelAdapter:
    """Thin validated facade over caller-supplied scoring/generation callables."""

    def __init__(
        self,
        *,
        adapter_id: str,
        checkpoint_sha256: str,
        score_candidates: Callable[[str, str, list[str]], list[float]],
        generate_free: Callable[[str, int], str],
        generate_constrained: Callable[[str, list[str]], str],
    ) -> None:
        if not adapter_id or any(c.isspace() for c in adapter_id):
            raise ValueError("adapter id must be a compact nonempty identity")
        _assert_sha256("checkpoint", checkpoint_sha256)
        self._identity = {
            "schema": ADAPTER_SCHEMA,
            "adapter_id": adapter_id,
            "checkpoint_sha256": checkpoint_sha256,
        }
        self._score = score_candidates
        self._free = generate_free
        self._constrained = generate_constrained

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self._identity)).hexdigest()

    def score_candidates(self, context: str, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            raise ValueError("candidate sets cannot be empty")
        scores = [float(value) for value in self._score(context, query, list(candidates))]
        if len(scores) != len(candidates):
            raise ValueError("adapter must return one score per candidate")
        if any(not math.isfinite(value) for value in scores):
            raise ValueError("candidate scores must be finite log-probabilities")
        return scores

    def generate_free(self, prompt: str, max_new_tokens: int = 64) -> str:
        if max_new_tokens <= 0 or max_new_tokens > 64:
            raise ValueError("free generation is capped at 64 new tokens")
        return str(self._free(prompt, max_new_tokens))

    def generate_constrained(self, prompt: str, candidates: list[str]) -> str:
        if not candidates:
            raise ValueError("constrained generation needs candidates")
        return str(self._constrained(prompt, list(candidates)))


def _assert_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


__all__ = ["ADAPTER_SCHEMA", "ModelAdapter"]
