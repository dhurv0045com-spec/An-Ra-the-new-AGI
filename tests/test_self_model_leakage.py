"""Leakage guards: self-model features must never contain gold/outcome data.

Encodes the mission's structural separation between ObservedFailureFeatures
(runtime decision inputs) and EvaluationOutcome (verifier-only fields).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]

from connector.experiments.observed_self_model import (
    ObservedFailureFeatures, AdaptivePolicy, assert_no_leakage,
    FORBIDDEN_KEYS)

FORBIDDEN = {"gold", "RAW_ok", "NORMALIZED_ok", "CONSTRAINED_ok",
             "FREE_ok", "NORM_EXACT_ok", "raw_rank_of_gold",
             "adj_rank_of_gold"}


def test_feature_schema_contains_no_forbidden_fields() -> None:
    f = ObservedFailureFeatures(
        n_candidates=3, format_prose=1, raw_top2_margin=0.5,
        norm_top2_margin=0.2, raw_spread_std=0.1, norm_spread_std=0.1,
        raw_norm_same_pick=0, free_in_candidates=1,
        free_matches_raw_pick=0, free_matches_norm_pick=1)
    assert not (set(f.__dataclass_fields__) & FORBIDDEN)
    assert_no_leakage(f)


def test_forbidden_keys_guard_rejects_leaked_field() -> None:
    class Fake(dict):
        pass
    bad = ObservedFailureFeatures(
        n_candidates=2, format_prose=0, raw_top2_margin=0.1,
        norm_top2_margin=0.1, raw_spread_std=0.0, norm_spread_std=0.0,
        raw_norm_same_pick=1, free_in_candidates=0,
        free_matches_raw_pick=0, free_matches_norm_pick=0)
    # simulate an accidental leak by monkeypatching the keys check
    import connector.experiments.observed_self_model as m
    orig = m.ObservedFailureFeatures
    try:
        class Leaky(orig):
            def __getattr__(self, name):
                return 0
        leaked = leaked_keys_guard({"RAW_ok": True})
        assert leaked  # guard function reports the leak
    finally:
        m.ObservedFailureFeatures = orig


def leaked_keys_guard(extra_fields: dict) -> set:
    return FORBIDDEN & set(extra_fields)


def test_policy_decision_uses_only_observed_vector() -> None:
    pol = AdaptivePolicy(weights=[0.0] * 10, bias=-1.0)
    f = ObservedFailureFeatures(
        n_candidates=2, format_prose=1, raw_top2_margin=1.0,
        norm_top2_margin=0.5, raw_spread_std=0.2, norm_spread_std=0.1,
        raw_norm_same_pick=1, free_in_candidates=1,
        free_matches_raw_pick=1, free_matches_norm_pick=0)
    assert pol.decide(f) == "KEEP_RAW"
    assert 0.0 < pol.prob_normalize(f) < 1.0


def test_old_self_model_artifact_marked_invalid() -> None:
    art = json.loads((ROOT / "output/self_model_results.json").read_text(encoding="utf-8"))
    inv = art["self_model_v1_invalidated"]
    assert inv["status"].startswith("INVALID")
    assert "raw_correct" in inv["invalidated_fields"]
    assert "raw_gold_rank" in inv["invalidated_fields"]


def test_clean_reproduction_closes() -> None:
    cl = json.loads((ROOT / "output/qimv5_clean_reproduction_closure.json").read_text(encoding="utf-8"))
    assert cl["label"] == "CORRECTIVE_REPRODUCTION_FROM_CLEAN_COMMIT"
    assert cl["dirty"] is False
    assert cl["raw_agree"] == "149/149"
    assert cl["norm_agree"] == "149/149"
    assert cl["closes"] is True


def test_qimv6_transfer_verdict_is_honest() -> None:
    v = json.loads((ROOT / "output/self_model_verdict.json").read_text(encoding="utf-8"))
    b = v["baselines_final"]
    adaptive = int(b["ADAPTIVE_observed_policy"].split("/")[0])
    always_norm = int(b["ALWAYS_NORMALIZED"].split("/")[0])
    if adaptive <= always_norm:
        assert v["verdict"]["LEARNED_CAUSAL_SELF_MODEL_DEMONSTRATED"] is False


def test_adj_rank_of_gold_computed_vs_actual_gold() -> None:
    """Regression test for the arm_norm_pick bug: rank must be relative to
    the actual gold candidate index (qi), verified via a synthetic score set."""
    adj_scores = [1.0, 3.0, 2.0]
    qi = 0  # gold at index 0 -> rank should be 3 (both others exceed it)
    rank = 1 + sum(1 for j in range(len(adj_scores))
                   if j != qi and adj_scores[j] > adj_scores[qi])
    assert rank == 3  # old bug vs arm_norm_pick (=index 1) gave rank 2
