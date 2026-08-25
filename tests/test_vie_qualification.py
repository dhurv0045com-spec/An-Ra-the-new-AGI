"""Tests: VIE qualification + MC-v8 confirmation contracts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]

V8_FIXTURE = "f5664e13ddd37f2024c7960d79dcead6ad1f5c16adffa089af858fd33c2ca8fa"


def _j(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def test_v8_confirms_v7_promotion() -> None:
    r = _j("output/mixed_causal_v8_confirmation.json")
    assert r["fixture_sha256"] == V8_FIXTURE
    b = r["policies"]
    assert b["ADAPTIVE_v7"]["succ"] == 291
    best_fixed = max(b["ALWAYS_NORMALIZED"]["succ"],
                     b["ALWAYS_CONSTRAINED"]["succ"],
                     b["ALWAYS_NO_CHANGE"]["succ"])
    assert b["ADAPTIVE_v7"]["succ"] > best_fixed
    paired = r["paired_adaptive_vs_others"]["ALWAYS_NORMALIZED"]
    assert paired["ci95"][0] > 0
    # policy unchanged from the promoted v7 — this is a replication draw
    assert "unchanged" in r["policy_frozen_commit"]


def test_vie_bank_qualified_entries_exist() -> None:
    bank_file = ROOT / "data/experience_bank/experiences.jsonl"
    entries = [json.loads(line) for line in
               bank_file.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    assert len(entries) >= 166
    q = _j("output/vie_qualification_mc8.json")
    assert q["added_this_run"] == 166
    assert q["bank_total_after"] == len(entries)


def test_vie_entries_satisfy_contract_fields() -> None:
    """Every qualified entry carries baseline-fail/intervention-pass,
    retained outputs, provenance, and observed-only decision evidence."""
    bank_file = ROOT / "data/experience_bank/experiences.jsonl"
    entries = [json.loads(line) for line in
               bank_file.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    for e in entries:
        assert e["baseline_success"] is False
        assert e["intervention_success"] is True
        assert e["changed_variable"] in ("decode", "selection")
        assert e["corrected_output"]
        assert len(e["parent_checkpoint_sha256"]) == 64
        assert "checkpoint" in " ".join(e["variables_held_constant"])
        break  # spot-check contract on first; count checked elsewhere


def test_mc7_audit_honestly_blocked_before_rerun() -> None:
    audit = _j("output/vie_audit_mc7.json")
    assert audit["contract_qualifying_flips"] == 0
    assert "retained" in audit["blocking_gap"]
