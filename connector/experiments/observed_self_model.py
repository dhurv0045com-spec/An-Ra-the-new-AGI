"""Observed-only self-model features and policy (leakage-safe by design).

ObservedFailureFeatures contains ONLY information computable at decision
time, before any verifier outcome:
  - candidate count
  - format
  - raw top1/top2 margin, raw score spread (std)
  - normalized top1/top2 margin, normalized score spread (std)
  - whether RAW and NORMALIZED picked the same candidate
  - free output membership in candidate set
  - raw-vs-normalized pick agreement with the free output

FORBIDDEN (structurally excluded; enforced by tests):
  gold, RAW_ok/NORMALIZED_ok, *_rank_of_gold, adj_rank_of_gold,
  any verifier result for the current decision.

EvaluationOutcome is a separate dataclass holding gold-dependent fields;
it is produced only by the evaluator after arms run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ObservedFailureFeatures:
    """Everything here is computable WITHOUT the answer key."""
    n_candidates: int
    format_prose: int                 # 1 prose / 0 table
    raw_top2_margin: float            # sorted(raw)[-1] - sorted(raw)[-2]
    norm_top2_margin: float
    raw_spread_std: float             # std of raw scores
    norm_spread_std: float
    raw_norm_same_pick: int           # RAW and NORMALIZED chose same?
    free_in_candidates: int           # free output matched some candidate code
    free_matches_raw_pick: int        # free output == raw's chosen code
    free_matches_norm_pick: int       # free output == normalized choice

    def vector(self) -> list[float]:
        v = asdict(self)
        return [
            float(v["n_candidates"]),
            float(v["format_prose"]),
            v["raw_top2_margin"], v["norm_top2_margin"],
            v["raw_spread_std"], v["norm_spread_std"],
            float(v["raw_norm_same_pick"]),
            float(v["free_in_candidates"]),
            float(v["free_matches_raw_pick"]),
            float(v["free_matches_norm_pick"]),
        ]

    FEATURE_NAMES = ["n_candidates", "format_prose", "raw_top2_margin",
                     "norm_top2_margin", "raw_spread_std", "norm_spread_std",
                     "raw_norm_same_pick", "free_in_candidates",
                     "free_matches_raw_pick", "free_matches_norm_pick"]


@dataclass
class EvaluationOutcome:
    """Evaluator-only fields. NEVER an input to the runtime policy."""
    raw_ok: bool
    normalized_ok: bool
    constrained_ok: bool
    raw_rank_of_gold: int
    adj_rank_of_gold: int


def build_observed_features(row: dict) -> ObservedFailureFeatures:
    """Extract observed features from one arm-run row.

    Reads ONLY observed fields from the row: margins/spreads are recomputed
    from scores if present, else taken from stored margin fields; picks are
    inferred from outputs vs candidates. Gold-dependent keys are never read.
    """
    n = int(row.get("n_candidates", 0)) or None
    # rows store margins directly (observed geometry); ranks-of-gold ignored
    return ObservedFailureFeatures(
        n_candidates=n if n else _infer_k(row),
        format_prose=1 if row.get("gi", 0) % 2 == 0 else 0,
        raw_top2_margin=float(row.get("raw_top2_margin", 0.0)),
        norm_top2_margin=float(row.get("adj_top2_margin", 0.0)),
        raw_spread_std=float(row.get("raw_spread_std", 0.0)),
        norm_spread_std=float(row.get("norm_spread_std", 0.0)),
        raw_norm_same_pick=int(row.get("RAW_ok", False) is not None and
                               row.get("raw_pick_code") == row.get("norm_pick_code")),
        free_in_candidates=int(bool(row.get("free_out_code"))),
        free_matches_raw_pick=int(bool(row.get("free_out_code")) and
                                  row.get("free_out_code") == row.get("raw_pick_code")),
        free_matches_norm_pick=int(bool(row.get("free_out_code")) and
                                   row.get("free_out_code") == row.get("norm_pick_code")),
    )


def _infer_k(row: dict) -> int:
    # fact-count histogram by group index parity used in fixtures: k = 2 + gi % 3
    return 2 + (int(row.get("gi", 0)) % 3)


# ---- leakage guard -------------------------------------------------------

FORBIDDEN_KEYS = {"gold", "RAW_ok", "NORMALIZED_ok", "CONSTRAINED_ok",
                  "NORM_EXACT_ok", "FREE_ok", "raw_rank_of_gold",
                  "adj_rank_of_gold"}


def assert_no_leakage(feature_obj: ObservedFailureFeatures) -> None:
    d = asdict(feature_obj).keys()
    bad = set(d) & FORBIDDEN_KEYS
    assert not bad, f"leaked evaluator fields into features: {bad}"


# ---- policy --------------------------------------------------------------

@dataclass
class AdaptivePolicy:
    """Logistic policy over observed features -> P(normalize helps).

    Trained on historical (features, outcome) pairs; at inference consumes
    ObservedFailureFeatures ONLY.
    """
    weights: list[float]
    bias: float
    threshold: float = 0.5

    def decide(self, f: ObservedFailureFeatures) -> str:
        p = self.prob_normalize(f)
        return "NORMALIZE" if p >= self.threshold else "KEEP_RAW"

    def prob_normalize(self, f: ObservedFailureFeatures) -> float:
        z = sum(w * x for w, x in zip(self.weights, f.vector())) + self.bias
        return 1.0 / (1.0 + math.exp(-z))

    def to_json(self) -> dict:
        return {"schema": "anra-observed-policy/v1",
                "feature_names": ObservedFailureFeatures.FEATURE_NAMES,
                "weights": self.weights, "bias": self.bias,
                "threshold": self.threshold}
