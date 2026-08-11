from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch

from training.posttraining_contract import REQUIRED_SFT_CATEGORIES
from training.sft_dataset_v4 import (
    SFT_SOURCE_RECEIPTS_SCHEMA,
    _normalise_text,
    build_sft_dataset_v4,
    sha256_file,
)
from training.sft_v4 import (
    SFTConversationDataset,
    _behavior_smoke_report,
    _behavior_smoke_verdict,
    _prune_sft_checkpoint_copies,
    _sft_recipe,
    _validate_sft_sampler_position,
    _validate_frozen_parent_kl_resume,
    _verify_resume_checkpoint_binding,
    _verify_full_sft_approval,
    _write_full_sft_approval,
    _resume_parent_validation_loss,
    assistant_only_loss,
    frozen_parent_kl_loss,
    load_sft_examples,
    load_sft_validation_examples,
)


def test_sft_sampler_position_is_bound_to_optimizer_progress() -> None:
    assert (
        _validate_sft_sampler_position(
            dataset_size=100,
            global_step=5,
            batch_size=1,
            accumulation=8,
            epoch=0,
            cursor=40,
        )
        == 40
    )
    with pytest.raises(ValueError, match="step/recipe imply 40 examples"):
        _validate_sft_sampler_position(
            dataset_size=100,
            global_step=5,
            batch_size=1,
            accumulation=8,
            epoch=0,
            cursor=39,
        )


def test_sft_checkpoint_pruning_keeps_one_canonical_payload(tmp_path: Path) -> None:
    vault = tmp_path / "training-home"
    sft_root = vault / "sft-v4"
    current = sft_root / "anra-v4-current-full-resume.pt"
    archived = sft_root / "archive" / "old-lineage" / "anra-v4-current-full-resume.pt"
    legacy = sft_root / "anra-v4-step-000000000200-full-resume.pt"
    current.parent.mkdir(parents=True)
    current.write_bytes(b"current")
    archived.parent.mkdir(parents=True)
    archived.write_bytes(b"archived")
    legacy.write_bytes(b"legacy")

    removed = _prune_sft_checkpoint_copies(vault)

    assert current.read_bytes() == b"current"
    assert not archived.exists()
    assert not legacy.exists()
    assert str(archived.resolve()) in removed
    assert str(legacy.resolve()) in removed


def _records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category in REQUIRED_SFT_CATEGORIES:
        for index in range(12):
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": "Answer directly and honestly."},
                        {"role": "user", "content": f"{category} request {index}"},
                        {"role": "assistant", "content": f"{category} answer {index}"},
                    ],
                    "category": category,
                    "source_id": f"audited-fixture-{index}",
                    "split_group": f"fixture-group-{index}",
                    "license": "Apache-2.0",
                }
            )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_sft_dataset_builds_immutable_disjoint_audited_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [*_records(), _records()[0]])
    result = build_sft_dataset_v4(
        [source],
        tmp_path / "prepared",
        quality_gate_passed=True,
        licenses_audited=True,
        allow_unregistered_inputs=True,
    )
    assert result.rejected_examples == 1
    train_examples, manifest = load_sft_examples(result.manifests["train"])
    assert len(train_examples) == manifest["accepted_examples"]
    assert set(manifest["category_counts"]) == set(REQUIRED_SFT_CATEGORIES)
    validation_examples, validation_manifest = load_sft_validation_examples(
        result.manifests["validation"]
    )
    assert len(validation_examples) == validation_manifest["accepted_examples"]
    assert set(validation_manifest["category_counts"]) == set(REQUIRED_SFT_CATEGORIES)
    assert not {
        example.split_group for example in train_examples
    } & {example.split_group for example in validation_examples}
    all_ids: set[str] = set()
    for split in ("train", "validation", "test"):
        artifact = result.output_dir / f"sft-v4-{split}.jsonl"
        ids = {
            json.loads(line)["conversation_sha256"]
            for line in artifact.read_text(encoding="utf-8").splitlines()
        }
        assert not (all_ids & ids)
        all_ids |= ids
    _write_jsonl(source, _records())
    with pytest.raises(FileExistsError, match="immutable"):
        build_sft_dataset_v4(
            [source],
            result.output_dir,
            quality_gate_passed=True,
            licenses_audited=True,
            allow_unregistered_inputs=True,
        )


def test_canonical_sft_builder_requires_a_hash_verified_source_receipt(tmp_path: Path) -> None:
    source = tmp_path / "verified-source.jsonl"
    rows = _records()
    for row in rows:
        row["source_id"] = "verified-fixture"
    _write_jsonl(source, rows)
    receipt = tmp_path / "sft-v4-source-receipts.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": SFT_SOURCE_RECEIPTS_SCHEMA,
                "sources": [
                    {
                        "source_id": "verified-fixture",
                        "license": "Apache-2.0",
                        "path": str(source),
                        "sha256": sha256_file(source),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = build_sft_dataset_v4(
        [source],
        tmp_path / "canonical",
        quality_gate_passed=True,
        licenses_audited=True,
        source_receipts_path=receipt,
    )
    manifest = json.loads(result.manifests["train"].read_text(encoding="utf-8"))
    assert manifest["source_receipt_sha256"] == sha256_file(receipt)
    assert manifest["unregistered_local_pilot"] is False

    with pytest.raises(PermissionError, match="verified source receipts"):
        build_sft_dataset_v4(
            [source],
            tmp_path / "blocked",
            quality_gate_passed=True,
            licenses_audited=True,
        )


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [3 + (ord(character) % 19) for character in text]

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(token) for token in ids)


def test_sft_supervision_masks_prompt_and_keeps_answer_and_eos() -> None:
    dataset = SFTConversationDataset(
        [
            type("Example", (), {"prompt": "user words", "answer": "assistant words"})(),
        ],
        _Tokenizer(),
        block_size=128,
    )
    row = dataset[0]
    weights = row["weights"]
    assert torch.any(weights == 0)
    assert torch.any(weights == 1)
    first_answer = int(torch.nonzero(weights, as_tuple=False)[0].item())
    assert torch.all(weights[:first_answer] == 0)
    assert torch.all(weights[first_answer:] == 1)

    logits = torch.randn(1, 4, 7)
    targets = torch.tensor([[1, 2, 3, 4]])
    answer_weights = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    baseline = assistant_only_loss(logits, targets, answer_weights)
    altered_prompt = logits.clone()
    altered_prompt[:, :2, :] += 100.0
    torch.testing.assert_close(
        baseline, assistant_only_loss(altered_prompt, targets, answer_weights)
    )


def test_frozen_parent_kl_is_zero_for_matching_logits_and_trains_only_student() -> None:
    student = torch.randn(1, 3, 7, requires_grad=True)
    parent = student.detach().clone()
    ids = torch.tensor([[2, 3, 0]])

    identical = frozen_parent_kl_loss(
        student, parent, ids, pad_token_id=0, temperature=1.0
    )
    assert float(identical.detach()) == pytest.approx(0.0, abs=1e-7)

    changed_parent = parent.clone()
    changed_parent[:, 0, 1] += 2.0
    anchored = frozen_parent_kl_loss(
        student, changed_parent, ids, pad_token_id=0, temperature=1.0
    )
    assert float(anchored.detach()) > 0.0
    anchored.backward()
    assert student.grad is not None
    assert changed_parent.grad is None


def test_frozen_parent_kl_resume_policy_is_explicit_and_reversible() -> None:
    disabled = _sft_recipe(
        seed=1301,
        batch_size=1,
        accumulation=8,
        total_steps=5_000,
        base_kl_weight=0.0,
        base_kl_interval=4,
        base_kl_temperature=1.0,
    )
    # Pre-anchor checkpoints can resume only with the explicit default-off path.
    _validate_frozen_parent_kl_resume({"training_recipe": {}}, disabled)
    anchored = _sft_recipe(
        seed=1301,
        batch_size=1,
        accumulation=8,
        total_steps=5_000,
        base_kl_weight=0.02,
        base_kl_interval=4,
        base_kl_temperature=1.0,
    )
    with pytest.raises(RuntimeError, match="recipe changed"):
        _validate_frozen_parent_kl_resume({"training_recipe": {}}, anchored)
    _validate_frozen_parent_kl_resume({"training_recipe": anchored}, anchored)


def test_behavior_probe_restores_training_mode() -> None:
    from anra_brain import CausalTransformerV2

    model = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=128,
    ).train()
    report = _behavior_smoke_report(
        model,
        _Tokenizer(),
        device=torch.device("cpu"),
        max_new_tokens=2,
    )
    assert report["prompt_count"] == 8
    assert model.training is True


def test_behavior_smoke_rejects_one_generic_answer_for_every_prompt() -> None:
    verdict = _behavior_smoke_verdict(
        [
            {
                "output": "A generic keyword-stuffed response.",
                "behavior_pass": True,
            }
            for _ in range(8)
        ]
    )
    assert verdict["unique_output_count"] == 1
    assert verdict["minimum_unique_outputs"] == 6
    assert verdict["collapse_detected"] is True
    assert verdict["passed"] is False


def test_sft_text_normalisation_preserves_code_indentation_and_newlines() -> None:
    code = "\r\n  def add(a, b):\r\n      return a + b\r\n"
    assert _normalise_text(code, field="answer") == "def add(a, b):\n      return a + b"


def test_full_sft_needs_signed_approval_for_the_current_pilot_checkpoint(tmp_path: Path) -> None:
    vault = tmp_path / "training-home"
    sft_root = vault / "sft-v4"
    sft_root.mkdir(parents=True)
    checkpoint = sft_root / "anra-v4-current-full-resume.pt"
    checkpoint.write_bytes(b"protected pilot checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (sft_root / "latest_sft_report.json").write_text(
        json.dumps(
            {
                "global_step": 12,
                "parent_validation_loss": 1.50,
                "best_validation_loss": 1.25,
                "lineage_id": "sft-fixture",
                "base_checkpoint_sha256": "b" * 64,
                "train_manifest_sha256": "c" * 64,
                    "validation_manifest_sha256": "d" * 64,
                    "checkpoint_sha256": checkpoint_sha256,
                    "behavior_smoke": {
                        "schema": "anra-sft-behavior-smoke/v1",
                        "passed": True,
                        "prompt_count": 8,
                        "unique_output_count": 8,
                        "minimum_unique_outputs": 6,
                        "collapse_detected": False,
                    },
                }
        ),
        encoding="utf-8",
    )
    (sft_root / "ready_to_sft.json").write_text(
        json.dumps(
            {
                "schema": "anra-sft-readiness/v1",
                "lineage_id": "sft-fixture",
                "checkpoint_sha256": checkpoint_sha256,
                "train_manifest_sha256": "c" * 64,
                "validation_manifest_sha256": "d" * 64,
                "global_step": 12,
                "validation_improved": True,
                "behavior_smoke_passed": True,
                "full_sft_ready": True,
            }
        ),
        encoding="utf-8",
    )
    lineage = {
        "manifest_sha256": "a" * 64,
        "parent": {"base_checkpoint_sha256": "b" * 64},
        "dataset": {"manifest_sha256": "c" * 64},
        "evaluation": {"manifest_sha256": "d" * 64},
        "lineage_id": "sft-fixture",
    }
    approval = _write_full_sft_approval(
        vault_root=vault,
        lineage=lineage,
        signing_key="test-signing-key",
        owner_approval="I reviewed the protected SFT pilot result.",
    )
    assert approval.is_file()
    verified = _verify_full_sft_approval(
        vault_root=vault,
        lineage=lineage,
        signing_key="test-signing-key",
    )
    assert verified["pilot_global_step"] == 12

    # Once full SFT starts, the canonical file advances beyond the approved
    # pilot hash.  The approval must remain valid for a lineage-bound child,
    # while still rejecting a checkpoint from another lineage.
    torch.save(
        {
            "global_step": 13,
            "sft_checkpoint_schema": "anra-v4-sft-checkpoint/v1",
            "sft": {
                "stage": "sft",
                "lineage_manifest_sha256": "a" * 64,
                "base_checkpoint_sha256": "b" * 64,
                "dataset_manifest_sha256": "c" * 64,
                "validation_manifest_sha256": "d" * 64,
                "assistant_only_loss": True,
            },
        },
        checkpoint,
    )
    resumed = _verify_full_sft_approval(
        vault_root=vault,
        lineage=lineage,
        signing_key="test-signing-key",
    )
    assert resumed["pilot_global_step"] == 12

    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(PermissionError, match="current protected pilot checkpoint"):
        _verify_full_sft_approval(
            vault_root=vault,
            lineage=lineage,
            signing_key="test-signing-key",
        )


def test_sft_resume_checkpoint_must_match_the_signed_lineage(tmp_path: Path) -> None:
    lineage = {
        "manifest_sha256": "a" * 64,
        "parent": {"base_checkpoint_sha256": "b" * 64},
        "dataset": {"manifest_sha256": "c" * 64},
        "evaluation": {"manifest_sha256": "d" * 64},
    }
    checkpoint = tmp_path / "sft.pt"
    torch.save(
        {
            "sft_checkpoint_schema": "anra-v4-sft-checkpoint/v1",
            "sft": {
                "stage": "sft",
                "lineage_manifest_sha256": "a" * 64,
                "base_checkpoint_sha256": "b" * 64,
                "dataset_manifest_sha256": "c" * 64,
                "validation_manifest_sha256": "d" * 64,
                "assistant_only_loss": True,
            },
        },
        checkpoint,
    )
    _verify_resume_checkpoint_binding(checkpoint, lineage)
    lineage["dataset"] = {"manifest_sha256": "e" * 64}
    with pytest.raises(ValueError, match="dataset_manifest_sha256"):
        _verify_resume_checkpoint_binding(checkpoint, lineage)


def test_legacy_resume_baseline_migrates_only_from_verified_full_approval() -> None:
    loaded = {"global_step": 12}
    baseline, migrated = _resume_parent_validation_loss(
        loaded,
        resuming=True,
        mode="full",
        signed_approval={"parent_validation_loss": 1.5},
    )
    assert baseline == 1.5
    assert migrated is True

    baseline, migrated = _resume_parent_validation_loss(
        loaded,
        resuming=True,
        mode="pilot",
        signed_approval={"parent_validation_loss": 1.5},
    )
    assert baseline == float("inf")
    assert migrated is False

    baseline, migrated = _resume_parent_validation_loss(
        {"parent_validation_loss": 1.25},
        resuming=True,
        mode="full",
        signed_approval={"parent_validation_loss": 1.5},
    )
    assert baseline == 1.25
    assert migrated is False
