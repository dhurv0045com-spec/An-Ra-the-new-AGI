"""Focused tests: experience contracts, deterministic proposer, causal eligibility.

The X-factor under test: counterfactual-paired dev scoring — a positional or
loss-driven shortcut cannot pass it, because the twin swaps the value while
holding every other byte identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from connector.experience import (
    CapabilityContract,
    ExperienceBank,
    ObservedFailure,
    TrainingProposal,
    VerifiedExperience,
    propose_from_experiences,
)
from training.sft_accumulate import _cf_twin, _strict


def _experience(variable: str, salt: str) -> VerifiedExperience:
    return VerifiedExperience(
        experience_id=f"ve-{salt}",
        task=ObservedFailure(f"task-{salt}", "input", "verifier", "bad output"),
        parent_checkpoint_sha256="abc",
        changed_variable=variable,
        intervention_cost=3,
        corrected_output="good output",
        variables_held_constant=("question", "decode"),
        baseline_success=False,
        intervention_success=True,
        diagnosis_hypothesis=f"{variable} flips",
        diagnosis_confidence=1.0,
        source_commit=None,
        timestamp="2026-08-22 00:00:00",
    )


def test_bank_roundtrip_and_queries(tmp_path: Path) -> None:
    bank = ExperienceBank(tmp_path / "bank.jsonl")
    for i in range(4):
        bank.add(_experience("knowledge", f"k{i}"))
    bank.add(_experience("decode", "d1"))
    assert len(bank.all()) == 5
    assert len(bank.fixed_by("knowledge")) == 4
    assert len(bank.fixed_by("decode")) == 1
    rows = bank.all()
    assert all(r["baseline_success"] is False and r["intervention_success"] is True
               for r in rows), "only verified repairs may inhabit the bank"


def test_proposer_requires_min_support_and_recommends_no_training_for_decode(tmp_path: Path) -> None:
    bank = ExperienceBank(tmp_path / "b.jsonl")
    contract = CapabilityContract("sha", {
        "context_binding": "PROMOTED", "tool_result_use": "PROMOTED",
        "selective_binding": "EXPERIMENTAL"})
    # Below support: 2 experiences -> no proposal.
    bank.add(_experience("knowledge", "a"))
    bank.add(_experience("knowledge", "b"))
    assert propose_from_experiences(bank, contract, min_support=3) is None
    # Third experience triggers a capability-training proposal.
    bank.add(_experience("knowledge", "c"))
    proposal = propose_from_experiences(bank, contract, min_support=3)
    assert isinstance(proposal, TrainingProposal)
    assert proposal.recommendation == "CAPABILITY_TRAINING"
    assert proposal.target_capability == "context_binding"
    assert set(proposal.protected_capabilities) == {"context_binding", "tool_result_use"}
    assert len(proposal.source_experience_ids) == 3
    # Decode-only evidence must recommend NO_TRAINING (policy, not weights).
    decode_bank = ExperienceBank(tmp_path / "d.jsonl")
    decode_bank.add(_experience("decode", "x"))
    decode_bank.add(_experience("decode", "y"))
    dp = propose_from_experiences(decode_bank, contract, min_support=1)
    assert dp.recommendation == "NO_TRAINING"


def test_cf_twin_is_byte_pure_and_strict_parser_rejects_dumping() -> None:
    item = {"prompt": "Object Kettle holds code AVR-123.\nReturn ONLY the code for Kettle.\nAnswer:",
            "answer": "AVR-123", "family": "selective"}
    twin = _cf_twin(item)
    assert twin is not None
    # Byte purity: replacing the twin's value back yields the original.
    assert twin["prompt"].replace(twin["answer"], item["answer"]) == item["prompt"]
    assert twin["prompt"].count(twin["answer"]) == 1
    # Strict parsing: multiple codes in output FAIL even if gold is present.
    assert _strict("AVR-123.", "AVR-123") is True
    assert _strict("AVR-123 or maybe BQW-456", "AVR-123") is False
    assert _strict("BQW-456", "AVR-123") is False


def test_cf_twin_rejects_ambiguous_gold() -> None:
    item = {"prompt": "code X-1 and code X-1 again", "answer": "X-1"}
    assert _cf_twin(item) is None  # gold not a code / not unique -> no twin


def test_lineage_links_child_to_proposal(tmp_path: Path, monkeypatch) -> None:
    from connector.experience import link_child
    bank = ExperienceBank(tmp_path / "b.jsonl")
    for i in range(3):
        bank.add(_experience("tool", f"t{i}"))
    contract = CapabilityContract("sha", {"tool_result_use": "EXPERIMENTAL"})
    proposal = propose_from_experiences(bank, contract, min_support=3)
    assert proposal is not None
    out = tmp_path / "lineage.json"
    lineage = link_child("checkpoints/child.pt", proposal, str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["proposal_id"] == proposal.proposal_id
    assert data["source_evidence_count"] == 3
    assert data["support_type"] == "OBSERVATIONAL"
    assert lineage["checkpoint"] == "checkpoints/child.pt"


def test_bank_loader_skips_comment_lines(tmp_path: Path) -> None:
    """The migrated experiences file carries explanatory comment lines;
    the loader must tolerate them, not crash."""
    path = tmp_path / "experiences.jsonl"
    path.write_text(
        "# VerifiedInterventionExperience entries only.\n"
        "# Historical SFT wins live in observations.jsonl.\n"
        + _experience("knowledge", "k9").to_json() + "\n",
        encoding="utf-8")
    bank = ExperienceBank(path)
    assert len(bank.all()) == 1
    assert bank.all()[0]["changed_variable"] == "knowledge"


def test_proposal_from_dict_tolerates_field_rename(tmp_path: Path) -> None:
    """Proposal JSON persisted before the taxonomy split must still load."""
    from connector.experience import TrainingProposal
    legacy = {"proposal_id": "tp-x", "source_observation_ids": ["obs-1", "obs-2"],
              "recommendation": "CAPABILITY_TRAINING", "target_capability": "selective_binding",
              "unknown_future_field": 123}
    p = TrainingProposal.from_dict(legacy)
    assert p.proposal_id == "tp-x"
    assert p.source_observation_ids == ("obs-1", "obs-2")
    assert p.source_experience_ids == ()


def test_experience_from_runtime_rejects_unverified() -> None:
    from connector.experience import experience_from_runtime

    class Fake:
        status = "failed"

    with pytest.raises(ValueError, match="only verified repairs"):
        experience_from_runtime("case-1", Fake())
