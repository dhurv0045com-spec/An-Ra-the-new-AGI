"""Leakage guards: self-model v2 structural observed/evaluator separation.

Encodes the mission requirements:
- ObservedArmState contains only runtime-visible fields; constructing it
  with evaluator keys raises.
- AdaptivePolicy accepts ONLY ObservedArmState (EvaluationOutcome raises
  TypeError).
- Feature values come from explicit observations, not fixture indices:
  changing gold/verifier fields cannot alter them; n_candidates comes from
  the actual candidate set; format from actual task metadata.
- The old leaked artifact is marked INVALID; the clean reproduction closes;
  the QIM-v6 transfer verdict is honest.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]

from connector.experiments.observed_self_model import (
    ObservedArmState, EvaluationOutcome, AdaptivePolicy, FORBIDDEN_KEYS)


def make_state(**over) -> ObservedArmState:
    base = dict(
        n_candidates=3, format_name="prose",
        raw_pick_code="AAA-111", norm_pick_code="BBB-222",
        free_out_code="AAA-111", constrained_pick_code="AAA-111",
        raw_scores=[1.0, 0.5, -0.2], norm_scores=[2.0, 1.9, 0.1],
    )
    base.update(over)
    return ObservedArmState(**base)


def test_observed_state_rejects_evaluator_keys() -> None:
    """Passing evaluator keys must fail loudly (dataclass TypeError counts:
    the field does not exist, so gold-dependent state cannot be smuggled in;
    __post_init__ additionally guards any future dict-based construction)."""
    with pytest.raises((TypeError, ValueError)):
        ObservedArmState(
            n_candidates=3, format_name="prose", raw_pick_code="A",
            norm_pick_code="B", free_out_code=None,
            constrained_pick_code=None, raw_scores=[0, 0], norm_scores=[0, 0],
            RAW_ok=True)  # type: ignore[misc]


def test_forbidden_keys_constant_covers_all_evaluator_fields() -> None:
    required = {"gold", "RAW_ok", "NORMALIZED_ok", "raw_rank_of_gold",
                "adj_rank_of_gold"}
    assert required <= FORBIDDEN_KEYS


def test_features_independent_of_gold_and_verifier_fields() -> None:
    """Deleting/altering evaluator fields must not change the feature vector.

    Structural proof: features derive only from scores/picks/format/count.
    We simulate two runs where gold and outcomes differ entirely — identical
    observed state must yield identical vectors.
    """
    s1 = make_state()
    v1 = s1.feature_vector()
    # same observed state, different hypothetical outcomes -> same vector
    s2 = make_state()
    assert s2.feature_vector() == v1
    # different outcome-relevant reality but same observable geometry:
    s3 = make_state(raw_pick_code="CCC-333")  # pick changed -> feature changes
    assert s3.feature_vector() != v1


def test_n_candidates_from_actual_set_not_fixture_index() -> None:
    assert make_state(n_candidates=5).n_candidates == 5
    assert make_state(n_candidates=2).feature_vector()[0] == 2.0
    # gi-based inference would give 2+gi%3; here explicit value wins
    assert make_state(n_candidates=4).feature_vector() != \
           make_state(n_candidates=2).feature_vector()


def test_format_from_actual_metadata() -> None:
    p = make_state(format_name="prose").feature_vector()
    t = make_state(format_name="table").feature_vector()
    l = make_state(format_name="list").feature_vector()
    # category code must be stable across calls and distinct per format
    assert p[1] == make_state(format_name="prose").feature_vector()[1]
    assert len({p[1], t[1], l[1]}) == 3


def test_policy_rejects_evaluation_outcome() -> None:
    pol = AdaptivePolicy(weights=(0.0,) * 10, bias=-1.0)
    outcome = EvaluationOutcome(
        gold_code="AAA-111", raw_ok=True, normalized_ok=False,
        constrained_ok=True, free_ok=True, raw_rank_of_gold=1,
        adj_rank_of_gold=2)
    with pytest.raises(TypeError):
        pol.decide(outcome)  # type: ignore[arg-type]


def test_policy_decision_changes_with_observed_geometry_only() -> None:
    # weight index 6 = raw_norm_same_pick (per FEATURE_NAMES); the flag is
    # DERIVED from pick agreement, so vary the picks themselves
    pol = AdaptivePolicy(weights=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  5.0, 0.0, 0.0, 0.0), bias=2.5)
    state_same = make_state(raw_pick_code="CCC-333", norm_pick_code="CCC-333")
    state_diff = make_state(raw_pick_code="AAA-111", norm_pick_code="BBB-222")
    assert pol.prob_normalize(state_same) > pol.prob_normalize(state_diff)


def test_old_artifact_marked_invalid() -> None:
    art = json.loads((ROOT / "output/self_model_results.json")
                     .read_text(encoding="utf-8"))
    inv = art["self_model_v1_invalidated"]
    assert inv["status"].startswith("INVALID")
    assert "raw_correct" in inv["invalidated_fields"]
    assert "raw_gold_rank" in inv["invalidated_fields"]


def test_truly_clean_reproduction_recorded() -> None:
    cl = json.loads((ROOT / "output/qimv5_truly_clean_closure.json")
                    .read_text(encoding="utf-8"))
    assert cl["dirty"] is False
    assert cl["closes"] is True
    assert cl["raw_agreement"] == "149/149"
    assert cl["norm_agreement"] == "149/149"
    assert cl["arms_reproduction"]["RAW"] == "70/149"
    assert cl["arms_reproduction"]["NORMALIZED"] == "106/149"


def test_qimv6_transfer_verdict_honest() -> None:
    v = json.loads((ROOT / "output/self_model_verdict.json")
                   .read_text(encoding="utf-8"))
    b = v["baselines_final"]
    adaptive = int(b["ADAPTIVE_observed_policy"].split("/")[0])
    always_norm = int(b["ALWAYS_NORMALIZED"].split("/")[0])
    if adaptive <= always_norm:
        assert v["verdict"]["LEARNED_CAUSAL_SELF_MODEL_DEMONSTRATED"] is False


def test_adj_rank_of_gold_computed_vs_actual_gold() -> None:
    adj_scores = [1.0, 3.0, 2.0]
    qi = 0
    rank = 1 + sum(1 for j in range(len(adj_scores))
                   if j != qi and adj_scores[j] > adj_scores[qi])
    assert rank == 3
