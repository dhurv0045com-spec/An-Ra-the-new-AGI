"""Tests: MIXED-CAUSAL-v6 promotion contracts (marginal-result guard)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]

V6_FIXTURE = "765736449472e714e6083eb7aea91ad77f19d75dfdb919ab1330a3bd5b3b76e7"
V6_POLICY = "1dba9c9de5c1de030dc39fbb00f96ee2378391221ad963ee431ebc202af2e764"


def _j(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def test_v6_policy_frozen_before_fixture() -> None:
    pol = _j("output/self_model_v6.json")
    assert pol["parameter_sha256"] == V6_POLICY
    verdict = _j("output/mixed_causal_v6_verdict.json")
    assert verdict["policy_v6_frozen_commit"] == "f3eef84"
    assert verdict["fixture_sha256_frozen_after_policy_freeze"] == V6_FIXTURE


def test_v6_lambda_selected_by_lofo_not_final_fixture() -> None:
    pol = _j("output/self_model_v6.json")
    assert pol["lambda"] == 0.0
    assert "LOFO" in pol["lambda_selection"]


def test_v6_new_feature_recorded() -> None:
    pol = _j("output/self_model_v6.json")
    assert "norm_margin_dominance" in pol["feature_names"]
    assert len(pol["feature_names"]) == 13


def test_v6_result_honestly_marginal() -> None:
    v = _j("output/mixed_causal_v6_verdict.json")
    b = v["results_all_policies"]
    # adaptive has the highest point estimate everywhere
    adaptive = b["ADAPTIVE_v6"]["succ"]
    others = [p["succ"] for k, p in b.items()
              if k not in ("ADAPTIVE_v6", "ORACLE_evaluator_only")]
    assert adaptive > max(others)
    # but the strict criterion was honestly recorded as NOT met
    assert v["primary_resolution"]["PRIMARY_CLAIM_NOT_SUPPORTED_AT_STRICT_ALPHA"] is True
    assert v["verdict"]["LEARNED_INTERVENTION_SELECTION_REPLICATED"] is False


def test_vie_zero_and_no_training() -> None:
    v = _j("output/mixed_causal_v6_verdict.json")
    assert v["verified_intervention_experiences"] == 0
    assert v["training_decision"] == "DO_NOT_TRAIN_CORE"
