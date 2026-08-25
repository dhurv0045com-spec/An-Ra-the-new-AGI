"""Tests: MIXED-CAUSAL-v4 replication contracts (honest-negative guard)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]

V4_FIXTURE = "a46ab2fbf774afb80523df4a0c6d9bcaf7948573b5524df832216384c4fef8e2"


def _j(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def test_v4_policy_frozen_before_fixture() -> None:
    pol = _j("output/self_model_v4.json")
    assert pol["parameter_sha256"].startswith("937ded8c")
    verdict = _j("output/mixed_causal_v4_verdict.json")
    assert verdict["policy_v4_frozen_commit"] == "1861e0f"
    assert verdict["fixture_sha256_frozen_after_policy_freeze"] == V4_FIXTURE


def test_v4_policy_trained_on_both_dev_matrices() -> None:
    man = _j("output/v4_training_manifest.json")
    assert man["tasks"] == 240 and man["outcomes"] == 520
    assert len(man["sources"]) == 2


def test_v4_standardization_recorded() -> None:
    pol = _j("output/self_model_v4.json")
    assert "standardization" in pol
    assert len(pol["standardization"]["means"]) == 12
    assert len(pol["standardization"]["stds"]) == 12


def test_v4_result_is_honest_negative() -> None:
    v = _j("output/mixed_causal_v4_verdict.json")
    b = v["results_all_policies"]
    # adaptive did NOT beat the best constant on MC-v4
    assert b["ALWAYS_NORMALIZED"]["succ"] > b["ADAPTIVE_v4"]["succ"]
    # adaptive DID significantly beat two constants (recorded in paired stats)
    r = _j("output/mixed_causal_v3_replication.json")  # noqa: F841
    assert v["primary_resolution"]["PRIMARY_CLAIM_NOT_SUPPORTED"] is True
    assert v["verdict"]["LEARNED_INTERVENTION_SELECTION_REPLICATED"] is False


def test_vie_stays_zero_and_no_core_training() -> None:
    v = _j("output/mixed_causal_v4_verdict.json")
    assert v["verified_intervention_experiences"] == 0
    assert v["training_decision"] == "DO_NOT_TRAIN_CORE"
