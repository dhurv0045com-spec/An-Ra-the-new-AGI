from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from anra.extensions import (
    adapter_state_dict,
    attach_candidate_adapters,
    detach_candidate_adapters,
    save_capability_adapter,
)
from inference.adapters import AdapterRegistry
from training.dpo import audited_preference_loss
from training.posttraining_contract import (
    REQUIRED_SFT_CATEGORIES,
    audit_preference_pairs,
    audit_verifiable_outcomes,
    verify_posttraining_gate_manifest,
    verify_sft_lineage_manifest,
    write_posttraining_gate_manifest,
    write_sft_lineage_manifest,
)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sft_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    shard = tmp_path / "sft.jsonl"
    shard.write_text('{"instruction":"answer carefully"}\n', encoding="utf-8")
    manifest = tmp_path / "dataset.json"
    manifest.write_text(
        json.dumps(
            {
                "quality_gate_passed": True,
                "licenses_audited": True,
                "split": "train",
                "accepted_examples": len(REQUIRED_SFT_CATEGORIES),
                "category_counts": dict.fromkeys(REQUIRED_SFT_CATEGORIES, 1),
                "artifacts": [
                    {
                        "path": shard.name,
                        "sha256": _hash_bytes(shard.read_bytes()),
                        "size_bytes": shard.stat().st_size,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"immutable-base")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(b"v4-32768")
    return manifest, checkpoint, tokenizer


def test_sft_lineage_is_signed_complete_and_immutable(tmp_path: Path) -> None:
    dataset, checkpoint, tokenizer = _sft_inputs(tmp_path)
    output = tmp_path / "sft-lineage.json"
    manifest = write_sft_lineage_manifest(
        output,
        lineage_id="sft-001",
        dataset_manifest_path=dataset,
        base_checkpoint_path=checkpoint,
        tokenizer_path=tokenizer,
        source_commit="abc123",
        signing_key="owner-secret",
    )
    verified = verify_sft_lineage_manifest(output, signing_key="owner-secret")
    assert verified["manifest_sha256"] == manifest["manifest_sha256"]
    assert verified["dataset"]["accepted_examples"] == 8

    with pytest.raises(FileExistsError, match="immutable"):
        write_sft_lineage_manifest(
            output,
            lineage_id="different-lineage",
            dataset_manifest_path=dataset,
            base_checkpoint_path=checkpoint,
            tokenizer_path=tokenizer,
            source_commit="abc123",
            signing_key="owner-secret",
        )

    tokenizer.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        verify_sft_lineage_manifest(output, signing_key="owner-secret")


def test_sft_lineage_rejects_missing_capability_category(tmp_path: Path) -> None:
    dataset, checkpoint, tokenizer = _sft_inputs(tmp_path)
    raw = json.loads(dataset.read_text(encoding="utf-8"))
    raw["category_counts"]["uncertainty"] = 0
    raw["accepted_examples"] -= 1
    dataset.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="uncertainty"):
        write_sft_lineage_manifest(
            tmp_path / "blocked.json",
            lineage_id="sft-blocked",
            dataset_manifest_path=dataset,
            base_checkpoint_path=checkpoint,
            tokenizer_path=tokenizer,
            source_commit="abc123",
            signing_key="owner-secret",
        )


def _outcome(*, outcome: str = "pass", included: bool = True) -> dict[str, object]:
    return {
        "task_id": "math-1",
        "domain": "mathematics",
        "candidate_sha256": "a" * 64,
        "verifier_id": "exact-math",
        "verifier_version": "1",
        "verifier_evidence_sha256": "b" * 64,
        "outcome": outcome,
        "reward": 1.0 if outcome == "pass" else 0.0,
        "included": included,
    }


def test_rlvr_star_and_dpo_have_explicit_evidence_gates(tmp_path: Path) -> None:
    rlvr = audit_verifiable_outcomes("rlvr", [_outcome()])
    assert rlvr["passed"] is True
    star_bad = audit_verifiable_outcomes("star", [_outcome(outcome="fail")])
    assert star_bad["passed"] is False
    heuristic = audit_preference_pairs(
        [
            {
                "pair_id": "p1",
                "source_kind": "synthetic_unverified",
                "source_id": "length-heuristic",
                "auditor_id": "reviewer",
                "audit_decision": "approved",
                "prompt_sha256": "1" * 64,
                "chosen_sha256": "2" * 64,
                "rejected_sha256": "3" * 64,
                "audit_evidence_sha256": "4" * 64,
            }
        ]
    )
    assert heuristic["passed"] is False
    scores = torch.tensor([0.2])
    with pytest.raises(PermissionError, match="has not passed"):
        audited_preference_loss(
            scores,
            scores - 0.1,
            scores,
            scores - 0.05,
            audit_report=heuristic,
        )

    gate_path = tmp_path / "rlvr-gate.json"
    gate = write_posttraining_gate_manifest(
        gate_path,
        stage="rlvr",
        parent_manifest_sha256="c" * 64,
        source_commit="abc123",
        gate_report=rlvr,
        signing_key="owner-secret",
    )
    assert gate["optimizer_restart_required"] is True
    assert (
        verify_posttraining_gate_manifest(
            gate_path, signing_key="owner-secret", expected_stage="rlvr"
        )["stage"]
        == "rlvr"
    )


def _model() -> nn.Sequential:
    torch.manual_seed(7)
    return nn.Sequential(nn.Linear(4, 4, bias=False))


def test_adapter_production_activation_requires_eval_and_rollback(tmp_path: Path) -> None:
    model = _model()
    baseline = model(torch.ones(1, 4)).detach().clone()
    attach_candidate_adapters(model, rank=2, target_modules=("0",))
    artifact = tmp_path / "adapter.pt"
    checkpoint_hash = "d" * 64
    tokenizer_hash = "e" * 64
    save_capability_adapter(
        model,
        artifact,
        capability_id="math-v1",
        base_model_profile="anra-v4-181m",
        base_checkpoint_sha256=checkpoint_hash,
        tokenizer_sha256=tokenizer_hash,
        source_commit="abc123",
    )
    detach_candidate_adapters(model)
    registry = AdapterRegistry()
    registered = registry.register(
        adapter_id="math-v1",
        path=artifact,
        base_checkpoint_hash=checkpoint_hash,
        tokenizer_hash=tokenizer_hash,
    )
    with pytest.raises(PermissionError, match="promotion evidence"):
        registry.activate_promoted_on_model(
            "math-v1",
            model,
            base_model_profile="anra-v4-181m",
            base_checkpoint_hash=checkpoint_hash,
            tokenizer_hash=tokenizer_hash,
        )

    promotion = registry.promote(
        "math-v1",
        evaluation={
            "passed": True,
            "baseline_score": 0.50,
            "candidate_score": 0.62,
            "protected_regression": 0.01,
            "suite_sha256": "1" * 64,
            "adapter_sha256": registered.sha256,
            "base_checkpoint_hash": checkpoint_hash,
        },
        rollback={
            "passed": True,
            "detach_restores_base": True,
            "rehearsal_sha256": "2" * 64,
            "adapter_sha256": registered.sha256,
            "base_checkpoint_hash": checkpoint_hash,
            "rollback_target_adapter_id": None,
        },
    )
    assert promotion.adapter_id == "math-v1"
    registry.activate_promoted_on_model(
        "math-v1",
        model,
        base_model_profile="anra-v4-181m",
        base_checkpoint_hash=checkpoint_hash,
        tokenizer_hash=tokenizer_hash,
    )
    assert adapter_state_dict(model)
    assert registry.rollback_on_model(
        model,
        base_model_profile="anra-v4-181m",
        base_checkpoint_hash=checkpoint_hash,
        tokenizer_hash=tokenizer_hash,
    ) is None
    assert not adapter_state_dict(model)
    torch.testing.assert_close(model(torch.ones(1, 4)), baseline)
