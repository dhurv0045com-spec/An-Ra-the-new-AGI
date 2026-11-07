"""Read-only operator checks on tiny committed correctness checkpoints."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from anra_v5.elendril.action_learning import action_fixture
from anra_v5.elendril.checkpoint_tools import (
    _current_git_head, compare_checkpoints, inspect_checkpoint, main,
    resume_dry_run, verify_checkpoint,
)
from anra_v5.elendril.codec import canonical
from anra_v5.elendril.foundation_training import FoundationTrainer, Recipe, TokenSample, legacy_tokenizer_manifest
from anra_v5.elendril.training import MODEL, configure_cpu
from v5_training.checkpoint import CheckpointStore


SPEC = replace(MODEL, width=16, layers=1, query_heads=2, kv_heads=1,
               head_dimension=8, ffn_width=32, context_length=64)


@pytest.fixture
def checkpoints(tmp_path):
    configure_cpu()
    data = action_fixture("train", worlds=5, seed=1729)
    samples = tuple(TokenSample(row.task_id, row.tokens, row.eligible, row.world_id, row.kind) for row in data.examples)
    trainer = FoundationTrainer(samples, spec=SPEC, recipe=Recipe(total_updates=3, batch_size=4),
        source_commit=_current_git_head(), root=tmp_path, lineage="operator", data_provenance=data.manifest,
        tokenizer_manifest=legacy_tokenizer_manifest())
    trainer.step()
    first = trainer.save()
    trainer.step()
    second = trainer.save()
    return trainer, first, second


def snapshot(root):
    return {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_inspection_verification_comparison_cli_are_read_only(checkpoints, capsys):
    trainer, first, second = checkpoints
    before = snapshot(trainer.store.root)
    report = inspect_checkpoint(trainer.store.root, "operator")
    assert report["checkpoint_sha256"] == second and report["byte_verification"] == "PASS"
    assert report["state"]["global_update"] == 2
    assert all(row["finite"] for row in report["model_tensors"].values())
    assert set(report["optimizer_steps"].values()) == {2.}
    checked = verify_checkpoint(trainer.store.root, "operator", first)
    assert checked["semantic_validation"].startswith("MODEL_TOKENIZER_RECIPE_COUNTERS_PASS")
    same = compare_checkpoints(trainer.store.root, "operator", first, first)
    different = compare_checkpoints(trainer.store.root, "operator", first, second)
    assert same["max_abs_parameter_difference"] == 0
    assert different["max_abs_parameter_difference"] > 0
    main(["inspect-checkpoint", "--root", str(trainer.store.root), "--lineage", "operator", "--sha", first])
    assert json.loads(capsys.readouterr().out)["checkpoint_sha256"] == first
    assert snapshot(trainer.store.root) == before


def test_dry_run_restores_head_without_update_or_write(checkpoints):
    trainer, first, second = checkpoints
    before = snapshot(trainer.store.root)
    # The source-check boundary is mocked for this unit test while implementation
    # is uncommitted. Production source checks remain enabled by default.
    with patch.object(FoundationTrainer, "step", side_effect=AssertionError("dry-run attempted optimizer update")), \
            patch("anra_v5.elendril.checkpoint_tools._source_tree_clean", return_value=True):
        result = resume_dry_run(trainer.store.root, "operator")
    assert result["validation"] == "PASS" and result["source_binding"] == "matching-git-head"
    assert result["next_update"] == 3 and result["next_indices"] == list(trainer.plan[2])
    assert result["optimizer_updates_executed"] == 0 and result["checkpoint_writes"] == 0
    assert snapshot(trainer.store.root) == before
    with pytest.raises(ValueError, match="LATEST only"):
        resume_dry_run(trainer.store.root, "operator", first)
    with patch("anra_v5.elendril.checkpoint_tools._current_git_head", return_value=None):
        with pytest.raises(ValueError, match="source mismatch"):
            resume_dry_run(trainer.store.root, "operator")
        diagnostic = resume_dry_run(trainer.store.root, "operator", allow_source_mismatch=True)
        assert diagnostic["validation"] == "DIAGNOSTIC_PASS_SOURCE_UNVERIFIED"
    with patch("anra_v5.elendril.checkpoint_tools._source_tree_clean", return_value=False):
        with pytest.raises(ValueError, match="dirty Python tree"):
            resume_dry_run(trainer.store.root, "operator")


def test_corruption_paths_and_payload_allocation_bound_fail_closed(checkpoints, tmp_path):
    trainer, _, second = checkpoints
    missing = tmp_path / "not-created"
    with pytest.raises(FileNotFoundError):
        inspect_checkpoint(missing, "operator")
    assert not missing.exists()
    with pytest.raises(ValueError, match="safe single path"):
        inspect_checkpoint(trainer.store.root, "../operator")
    with pytest.raises(ValueError, match="read limit"):
        inspect_checkpoint(trainer.store.root, "operator", max_bytes=1)
    path = trainer.store.objects / second / "model.bin"
    path.write_bytes(path.read_bytes()[:10])
    with pytest.raises(ValueError, match="corrupt"):
        verify_checkpoint(trainer.store.root, "operator")
    assert trainer.store.latest_sha256() == second


def generic_copy(trainer, first, destination, *, scheduler=None, model_sha=None, tokenizer_sha=None):
    state, payloads = trainer.store.restore(first)
    if scheduler is not None:
        payloads["scheduler.json"] = canonical(scheduler)
    if model_sha is not None or tokenizer_sha is not None:
        identities = replace(state.identities,
            model_spec_sha256=model_sha or state.identities.model_spec_sha256,
            tokenizer_sha256=tokenizer_sha or state.identities.tokenizer_sha256)
        state = replace(state, identities=identities)
        payloads["training_state.json"] = canonical(state.canonical())
    store = CheckpointStore(destination, "operator")
    copied = store.publish(state=state, payloads=payloads, expected_parent_sha256=None)
    return store, copied


def test_legacy_is_honestly_byte_tensor_verified_without_fake_resume(checkpoints, tmp_path):
    trainer, first, _ = checkpoints
    store, identity = generic_copy(trainer, first, tmp_path / "legacy", scheduler={"legacy": "control"})
    result = verify_checkpoint(store.root, "operator")
    assert result["training_format"] == "legacy-or-external"
    assert result["semantic_validation"].startswith("FORMAT_SPECIFIC_NOT_RUN")
    with pytest.raises(ValueError, match="original trainer"):
        resume_dry_run(store.root, "operator")


def test_incompatible_tokenizer_or_architecture_never_subtracted(checkpoints, tmp_path):
    trainer, first, _ = checkpoints
    # A second valid transaction in the same lineage can bind another model;
    # the comparison command must detect it before tensor subtraction.
    state, payloads = trainer.store.restore()
    state = replace(state, generation=3, global_update=3, optimizer_step_max=3,
        parent_checkpoint_sha256=trainer.parent, identities=replace(state.identities, tokenizer_sha256="b" * 64))
    payloads["training_state.json"] = canonical(state.canonical())
    other = CheckpointStore(trainer.store.root, "operator").publish(state=state, payloads=payloads,
        expected_parent_sha256=trainer.parent)
    with pytest.raises(ValueError, match="incompatible model/tokenizer"):
        compare_checkpoints(trainer.store.root, "operator", first, other)
    with pytest.raises(ValueError, match="tokenizer/vocabulary mismatch"):
        verify_checkpoint(trainer.store.root, "operator", other)


def test_resume_runtime_mismatch_and_nonreconstructible_data_are_explicit(checkpoints, tmp_path):
    trainer, first, _ = checkpoints
    state, payloads = trainer.store.restore(first)
    scheduler = json.loads(payloads["scheduler.json"])
    scheduler["metadata"]["runtime"]["torch"] = "not-this-runtime"
    store, _ = generic_copy(trainer, first, tmp_path / "runtime", scheduler=scheduler)
    with pytest.raises(ValueError, match="mismatch"):
        resume_dry_run(store.root, "operator", allow_source_mismatch=True)
    scheduler["metadata"]["provenance"] = {"split": "train", "license": "repository LICENSE"}
    store, _ = generic_copy(trainer, first, tmp_path / "custom", scheduler=scheduler)
    with pytest.raises(ValueError, match="reconstructible"):
        resume_dry_run(store.root, "operator")
