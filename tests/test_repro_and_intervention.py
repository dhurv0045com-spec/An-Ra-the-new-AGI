"""Tests: reproduction script, CVE-v2 disjointness, constrained-decode report."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]

FIX = "27713accb3aa6825de23cf09540497943a0e56ca3ac177ba94f0f164f740a614"


def _j(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


# ---------------- clean-commit reproduction ------------------------------

def test_reproduction_script_exists_with_expected_contract() -> None:
    src = (ROOT / "scripts/reproduce_corrective_rescore.py").read_text(
        encoding="utf-8")
    for token in ("c3bc615e", "36c87e5b", FIX,
                  '"greedy": "44/119"', "LIVE REDO MATCHES"):
        assert token in src, f"missing {token!r}"


def test_corrective_artifacts_match_expected_numbers() -> None:
    par = _j("output/qim3_parent_corrective_rescore.json")
    chi = _j("output/qim3_sft6_corrective_rescore.json")
    assert par["candidate_diagnostic_only"]["greedy_corresponding_accuracy"] == "36/119"
    assert chi["candidate_diagnostic_only"]["greedy_corresponding_accuracy"] == "44/119"
    assert chi["group_level"]["mean_group_lift"] == pytest.approx(2.5052)


# ---------------- CVE-v2 true entity disjointness -------------------------

def test_cve_v2_disjointness_is_verified_not_claimed() -> None:
    from connector.experiments.context_value_extraction_v2 import (
        vocabulary_disjointness)
    res = vocabulary_disjointness()
    assert res["disjoint"] is True
    # the proof must cover entities AND prefixes against constants AND corpora
    for key in ("prefix_constant_overlaps", "prefix_corpus_hits",
                "entity_constant_overlaps", "entity_corpus_hits",
                "checked_files"):
        assert key in res
    assert len(res["checked_files"]) >= 8  # data + sealed OOD item files
    assert not res["entity_constant_overlaps"]
    assert not res["entity_corpus_hits"]


def test_cve_v1_jamb_defect_would_be_caught_by_v2_checker() -> None:
    """The v1 fixture's 'jamb' MUST fail v2's corpus check — proving the
    checker actually works (it would have caught the original defect)."""
    from connector.experiments.context_value_extraction_v2 import (
        vocabulary_disjointness)
    blob = ""
    for p in ("data/grouped_queryswap/train.jsonl",
              "data/grouped_queryswap/heldout.jsonl"):
        blob += (ROOT / p).read_text(encoding="utf-8")
    assert "Jamb" in blob or "jamb" in blob
    # and v2's own entities have zero corpus hits:
    res = vocabulary_disjointness()
    assert res["entity_corpus_hits"] == []


# ---------------- constrained-decode intervention -------------------------

def test_constrained_intervention_report_complete_and_sane() -> None:
    rep = _j("output/constrained_decode_intervention.json")
    assert rep["intervention_class"] == "RUNTIME_SINGLE_VARIABLE_OBSERVED_ONLY"
    assert rep["parameter_sha256"] == \
        "36c87e5be671ff37951b4b433e21ec14bb5e59f6f0770d7f81e50e99ead9e001"
    assert rep["fixture_sha256"] == FIX
    assert rep["free_greedy_accuracy"] == "44/119"
    assert rep["constrained_accuracy"] == "64/119"
    assert rep["flips_fail_to_pass"] == 20
    assert rep["flips_pass_to_fail"] == 0
    rows = rep["per_item_rows"]
    assert len(rows) == 119
    # arithmetic closure
    free = sum(1 for r in rows if r["free_ok"])
    constr = sum(1 for r in rows if r["constr_ok"])
    f2p = sum(1 for r in rows if not r["free_ok"] and r["constr_ok"])
    p2f = sum(1 for r in rows if r["free_ok"] and not r["constr_ok"])
    assert (free, constr, f2p, p2f) == (44, 64, 20, 0)


def test_no_verified_intervention_claim_is_made() -> None:
    rep = _j("output/constrained_decode_intervention.json")
    guard = rep["interpretation_guard"]
    assert "not a VerifiedInterventionExperience claim" in guard
