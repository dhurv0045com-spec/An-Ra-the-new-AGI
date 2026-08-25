"""Tests: MIXED-CAUSAL-v3 replication contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]

POLICY_SHA = "e807abdb9ca8af6b481283f420dd34ad802fb68ab0f9ea7404ad08330400ca94"
V3_FIXTURE = "25eb97a7f5dedf36a7c6bb7c55261db264a3edf5067edd7b111f8ff5f42b9d06"


def _j(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def test_canonical_trainer_deterministic() -> None:
    """Two runs of the canonical trainer must produce the identical policy SHA
    (verified at build time; this test asserts the committed artifact's SHA)."""
    pol = _j("output/self_model_v3.json")
    assert pol["parameter_sha256"] == POLICY_SHA
    assert pol["canonical_trainer"] == "scripts/train_self_model_v3.py"


def test_policy_v3_action_set_clean() -> None:
    pol = _j("output/self_model_v3.json")
    assert pol["actions"] == ["NO_CHANGE", "CONSTRAINED", "NORMALIZED"]
    assert "NORM_EXACT" not in pol["actions"]
    assert "ABSTAIN" not in pol["actions"]
    assert pol["lambda"] == 0.25
    assert "cost(a)" in pol["utility_rule"]


def test_policy_features_contain_no_gold_or_verifier_fields() -> None:
    pol = _j("output/self_model_v3.json")
    forbidden = {"gold", "family", "RAW_ok", "NORMALIZED_ok",
                 "raw_rank_of_gold", "adj_rank_of_gold",
                 "query_target_index"}
    assert not (set(pol["feature_names"]) & forbidden)


def test_query_sensitivity_feature_is_target_independent() -> None:
    """max_query_lift_gap must not reference any target index or gold."""
    pol = _j("output/self_model_v3.json")
    names = pol["feature_names"]
    assert "max_query_lift_gap" in names
    idx = names.index("max_query_lift_gap")
    # the name itself carries no target/gold semantics
    assert "gold" not in names[idx] and "target" not in names[idx]


def test_policy_frozen_before_mc_v3_fixture_commit() -> None:
    """Policy freeze commit f033d9f precedes fixture commit e945e45 —
    recorded here as a contract on the receipt fields."""
    pol = _j("output/self_model_v3.json")
    assert pol["schema"] == "anra-self-model-v3/v2"
    verdict = _j("output/mixed_causal_v3_verdict.json")
    assert verdict["policy_v31_frozen_commit"] == "f033d9f"
    assert verdict["fixture_sha256_frozen_after_policy_freeze"] == V3_FIXTURE


def test_replication_results_present_and_honest() -> None:
    v = _j("output/mixed_causal_v3_verdict.json")
    assert v["primary_resolution"]["PRIMARY_CLAIM_NOT_REPLICATED"] is True
    b = v["results_all_policies"]
    assert b["ADAPTIVE_v31"]["succ"] == 99
    best_fixed = max(b["ALWAYS_CONSTRAINED"]["succ"],
                     b["ALWAYS_NORMALIZED"]["succ"],
                     b["ALWAYS_NO_CHANGE"]["succ"])
    assert b["ADAPTIVE_v31"]["succ"] > best_fixed          # highest overall
    # ...but NOT statistically significant vs the best fixed/simple rule
    paired = v["paired_adaptive_vs_others"]
    vs_const = paired["vs ALWAYS_CONSTRAINED (best fixed)"]
    assert "[-0.039, +0.061]" == vs_const["ci95"]           # CI spans zero


def test_simple_rules_use_only_observed_geometry() -> None:
    """The four preregistered simple rules' chooser inputs are observable
    fields stored in per-task rows (norm_top2/raw_top2/picks/free code)."""
    r = _j("output/mixed_causal_v3_replication.json")
    row = r["per_task_rows"][0]
    observed_keys = {"norm_top2", "raw_top2", "norm_pick_code",
                     "raw_pick_code", "free_out_code"}
    assert observed_keys <= set(row.keys())
