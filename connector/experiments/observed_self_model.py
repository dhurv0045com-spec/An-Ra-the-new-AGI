"""Observed-only self-model: features, policy, and structural leakage guards.

DESIGN (v2, after the gi-inference and evaluator-read bugs):

ObservedArmState  -- runtime-visible fields ONLY. Built explicitly by the
                     arm runner from actual observations; NEVER inferred
                     from fixture index; NEVER reads evaluator fields.
EvaluationOutcome -- gold/verifier fields ONLY. Produced after arms run.
AdaptivePolicy    -- accepts ObservedArmState ONLY (type-enforced).

The arm runner stores per target:
    n_candidates, format_name, raw_pick_code, norm_pick_code, free_out_code,
    constrained_pick_code, raw_scores[], norm_scores[]
and the feature builder consumes those explicit fields — no `gi` arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Optional

# Keys that must never appear in runtime-visible state.
FORBIDDEN_KEYS = frozenset({
    "gold", "gold_code", "RAW_ok", "NORMALIZED_ok", "CONSTRAINED_ok",
    "NORM_EXACT_ok", "FREE_ok", "raw_rank_of_gold", "adj_rank_of_gold",
    "verifier_result", "is_correct",
})


@dataclass
class ObservedArmState:
    """Runtime-visible observations for one target. No gold. No verifier."""
    n_candidates: int
    format_name: str                    # 'prose' | 'table' | 'list' | ...
    raw_pick_code: str                  # RAW arm's chosen code
    norm_pick_code: str                 # normalized arm's chosen code
    free_out_code: Optional[str]        # code extracted from free decode
    constrained_pick_code: Optional[str]
    raw_scores: list[float]
    norm_scores: list[float]

    def __post_init__(self):
        present = set(self.__dict__) & FORBIDDEN_KEYS
        if present:
            raise ValueError(
                f"ObservedArmState cannot contain evaluator keys: {present}")

    # ---- derived observed geometry -------------------------------
    @property
    def raw_top2_margin(self) -> float:
        s = sorted(self.raw_scores)
        return s[-1] - s[-2] if len(s) >= 2 else 0.0

    @property
    def norm_top2_margin(self) -> float:
        s = sorted(self.norm_scores)
        return s[-1] - s[-2] if len(s) >= 2 else 0.0

    @property
    def raw_spread_std(self) -> float:
        m = sum(self.raw_scores) / len(self.raw_scores)
        return (sum((x - m) ** 2 for x in self.raw_scores)
                / len(self.raw_scores)) ** 0.5

    @property
    def norm_spread_std(self) -> float:
        m = sum(self.norm_scores) / len(self.norm_scores)
        return (sum((x - m) ** 2 for x in self.norm_scores)
                / len(self.norm_scores)) ** 0.5

    FORMAT_CODES = {"prose": 0.0, "table": 1.0, "list": 2.0}

    def feature_vector(self) -> list[float]:
        return [
            float(self.n_candidates),
            self.FORMAT_CODES.get(self.format_name, 3.0),
            self.raw_top2_margin,
            self.norm_top2_margin,
            self.raw_spread_std,
            self.norm_spread_std,
            float(self.raw_pick_code == self.norm_pick_code),
            float(self.free_out_code is not None),
            float(self.free_out_code == self.raw_pick_code),
            float(self.free_out_code == self.norm_pick_code),
        ]

    FEATURE_NAMES = ["n_candidates", "format_code", "raw_top2_margin",
                     "norm_top2_margin", "raw_spread_std", "norm_spread_std",
                     "raw_norm_same_pick", "free_in_candidates",
                     "free_matches_raw_pick", "free_matches_norm_pick"]


@dataclass
class EvaluationOutcome:
    """Gold/verifier fields ONLY — produced by the evaluator post-hoc."""
    gold_code: str
    raw_ok: bool
    normalized_ok: bool
    constrained_ok: bool
    free_ok: bool
    raw_rank_of_gold: int
    adj_rank_of_gold: int


@dataclass(frozen=True)
class AdaptivePolicy:
    """Logistic policy over ObservedArmState features.

    Type discipline: decide()/prob_normalize() take ObservedArmState;
    an EvaluationOutcome has no feature_vector() and fails loudly.
    """
    weights: tuple[float, ...]
    bias: float
    threshold: float = 0.5

    def prob_normalize(self, state: ObservedArmState) -> float:
        if not isinstance(state, ObservedArmState):
            raise TypeError(
                f"policy requires ObservedArmState, got {type(state).__name__} "
                "(EvaluationOutcome is not a decision input)")
        z = sum(w * x for w, x in zip(self.weights, state.feature_vector())) \
            + self.bias
        return 1.0 / (1.0 + math.exp(-z))

    def decide(self, state: ObservedArmState) -> str:
        return "NORMALIZE" if self.prob_normalize(state) >= self.threshold \
            else "KEEP_RAW"

    def to_json(self) -> dict:
        return {"schema": "anra-observed-policy/v2",
                "feature_names": ObservedArmState.FEATURE_NAMES,
                "weights": list(self.weights), "bias": self.bias,
                "threshold": self.threshold}


def build_state_from_row(row: dict) -> ObservedArmState:
    """Build runtime state from the canonical runner's row.

    Reads ONLY explicit observed fields. Raises if any forbidden key is
    consulted (defensive: they should not even be in the row).
    """
    leaked = FORBIDDEN_KEYS & set(row)
    if leaked:
        # evaluator fields may coexist in a receipt row, but must not be read
        pass
    return ObservedArmState(
        n_candidates=int(row["n_candidates"]),
        format_name=str(row["format_name"]),
        raw_pick_code=str(row["raw_pick_code"]),
        norm_pick_code=str(row["norm_pick_code"]),
        free_out_code=row.get("free_out_code"),
        constrained_pick_code=row.get("constrained_pick_code"),
        raw_scores=[float(x) for x in row["raw_scores"]],
        norm_scores=[float(x) for x in row["norm_scores"]],
    )
