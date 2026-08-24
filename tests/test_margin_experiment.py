"""Tests: margin experiment receipts + cross-fixture greedy replication."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]

C_SHA = "36c87e5be671ff37951b4b433e21ec14bb5e59f6f0770d7f81e50e99ead9e001"
FIX_V4 = "5effd87acdd5d865462334d0455bd3f07082b3f0c2061b80a5c45b276bc9ddd4"
FIX_V3 = "27713accb3aa6825de23cf09540497943a0e56ca3ac177ba94f0f164f740a614"


def _j(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def test_preregistration_committed_with_all_predictions() -> None:
    prop = _j("data/proposal_margin_queryswap_003.json")
    preds = prop["pre_registered_predictions"]
    assert {k.split("_")[0] for k in preds} == {f"PR{i}" for i in range(1, 8)}
    assert prop["parent"]["parameter_sha256"] == C_SHA
    assert prop["replication_instrument"]["fixture_sha256"] == FIX_V4


def test_margin_receipts_resolve_every_prediction() -> None:
    final = _j("output/margin_receipt_final.json")
    resolved = final["preregistered_predictions_resolved"]
    assert {k.split("_")[0] for k in resolved} == {f"PR{i}" for i in range(1, 8)}
    # primary prediction honestly recorded as failed
    assert resolved["PR1_selection_rank_v4"].startswith("FAIL")


def test_qim_v4_artifacts_consistent() -> None:
    par = _j("output/qim4_parent_baseline.json")
    chi = _j("output/qim4_sft7_margin.json")
    assert par["fixture_sha256"] == chi["fixture_sha256"] == FIX_V4
    assert par["parameter_sha256"] == C_SHA
    assert chi["parameter_sha256"] == \
        "c01cbbeea470e0c9faa47a31a6c3cc80a3b353343f1fc36c722e945402f8ae53"
    # paired delta recomputes from per-group lifts
    deltas = [c - p for c, p in zip(chi["per_group_query_lift"],
                                    par["per_group_query_lift"])]
    mean_d = sum(deltas) / len(deltas)
    assert abs(mean_d - chi["paired_vs_parent"]["mean_paired_delta"]) < 5e-4


def test_greedy_gain_replicated_on_second_frozen_fixture() -> None:
    repl = _j("output/margin_greedy_replication.json")
    assert repl["verdict"] == "REPLICATED"
    g = repl["result_cross_fixture"]["greedy_corresponding_accuracy"]
    assert g["SFT6_on_QIM_v3"] == "44/119"
    assert g["SFT7_on_QIM_v3_replication"] == "49/119 (+5)"
    assert g["SFT6_on_QIM_v4"] == "57/119"
    assert g["SFT7_on_QIM_v4_original"] == "62/119 (+5)"


def test_replication_used_independent_fixture_for_sft7() -> None:
    """QIM-v3 was never consulted by SFT7's training/design (SFT7 used
    QIM-v4), so using it here is a legitimate independent replication —
    this must stay true or the replication claim collapses."""
    src = (ROOT / "training/sft_margin_queryswap.py").read_text(encoding="utf-8")
    assert "query_influence_v3" not in src, \
        "margin trainer must not import QIM-v3 (fixture independence)"
    repl = _j("output/qim3_sft7_margin_replication.json")
    assert repl["fixture_sha256"] == FIX_V3


def test_no_verified_intervention_claim_from_margin_run() -> None:
    final = _j("output/margin_receipt_final.json")
    assert final["verified_intervention_experiences_earned"] == 0
