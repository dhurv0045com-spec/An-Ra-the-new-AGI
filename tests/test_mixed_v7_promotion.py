"""Tests: MIXED-CAUSAL-v7 PROMOTION contracts (strict pass guard)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]

V7_FIXTURE = "56be1755c03aee0e53ef672ad4354ee36ee9354a0138debf7714d9324af73ee9"
V7_POLICY = "ae9ad72b9ba56f218e6d3c1b5bdc0cd7c3785b3f7e17c5c1ebc27020701bcffe"


def _j(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def test_v7_policy_frozen_before_powered_fixture() -> None:
    pol = _j("output/self_model_v7.json")
    assert pol["parameter_sha256"] == V7_POLICY
    verdict = _j("output/mixed_causal_v7_verdict.json")
    assert verdict["policy_v7_frozen_commit"] == "f3b55cf"
    assert verdict["fixture_sha256_frozen_after_policy_freeze"] == V7_FIXTURE


def test_v7_trained_on_full_pooled_dev_set() -> None:
    pol = _j("output/self_model_v7.json")
    assert len(pol["training_sources"]) == 3
    harvest = _j("output/harvest_v7_pool.json")
    assert harvest["n_rows"] == 482
    # per-action training sizes must reflect the pool (NO_CHANGE sees all)
    assert pol["models"]["NO_CHANGE"]["train_examples"] == 722


def test_v7_strict_pass_is_real() -> None:
    """The promoted claim must match the recorded numbers."""
    v = _j("output/mixed_causal_v7_verdict.json")
    b = v["results_all_policies"]
    adaptive = b["ADAPTIVE_v7"]["succ"]
    fixed = [p["succ"] for k, p in b.items() if k.startswith(("ALWAYS", "SIMPLE"))]
    assert adaptive == 310 and max(fixed) == 274
    paired = v["paired_adaptive_vs_others_ALL_SIGNIFICANT"]
    vs_best = paired["vs ALWAYS_NORMALIZED (best fixed)"]
    # CI [+5.2,+10.0] excludes zero — the strict bar
    assert "CI [+5.2,+10.0]" in vs_best and "p<1e-6" in vs_best
    assert v["primary_resolution"]["PRIMARY_CLAIM_SUPPORTED"] is True
    assert v["verdict"]["LEARNED_INTERVENTION_SELECTION_REPLICATED"] is True


def test_scope_limits_recorded_not_hidden() -> None:
    v = _j("output/mixed_causal_v7_verdict.json")
    lims = " ".join(v["verdict"]["scope_limits"])
    assert "composition" in lims and "single checkpoint" in lims


def test_vie_still_zero_pending_separate_audit() -> None:
    """Promotion of the POLICY claim does not auto-promote VIE."""
    v = _j("output/mixed_causal_v7_verdict.json")
    assert v["verified_intervention_experiences"] == 0
    assert v["training_decision"] == "DO_NOT_TRAIN_CORE"
