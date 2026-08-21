from __future__ import annotations

from pathlib import Path
import hashlib

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.build_brain import (
    _active_training_data_layout,
    _assert_resume_data_layout_compatible,
    _assert_resume_data_profile_compatible,
    _assert_training_loader_dataset,
    _collect_data_manifest_payloads,
    _configure_continuation_phase,
    _freeze_training_lineage,
    _masked_logit_z_loss,
    _tokenizer_checkpoint_contract,
    _weighted_loss,
)
from anra_brain import CausalTransformerV2
from training.v2_runtime import (
    CheckpointCompatibilityError,
    assert_checkpoint_optimizer_boundary,
    load_checkpoint,
)


def _phase_model() -> CausalTransformerV2:
    return CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_layer=2,
        block_size=16,
        mod_layers={1},
    )


def test_phase_a_is_a_true_dense_baseline() -> None:
    model = _phase_model()
    report = _configure_continuation_phase(model, "A")
    assert report["enabled_subsystems"] == []
    assert report["policy_source"] == "dense_foundation_contract"
    assert not any(report["subsystem_activation"].values())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith(
            (
                "mod_routers.",
                "rim_modules.",
                "esv_module.",
                "residual_depth_logits",
                "dstp_temperature_log",
                "layer_temperature_bias_log",
            )
        )
    )


def test_phase_b_activates_only_the_declared_ablation(monkeypatch) -> None:
    model = _phase_model()
    monkeypatch.setenv("ANRA_PHASE_B_SUBSYSTEM", "mod")
    report = _configure_continuation_phase(model, "B")
    assert report["enabled_subsystems"] == ["mod"]
    assert report["mod_capacity"] == 0.5
    assert model.use_mod is True
    assert model.use_rim is False
    assert model.use_dstp is False


def test_later_phase_rejects_an_implicit_all_on_recipe(monkeypatch) -> None:
    monkeypatch.delenv("ANRA_ENABLED_SUBSYSTEMS", raising=False)
    with pytest.raises(RuntimeError, match="explicit ANRA_ENABLED_SUBSYSTEMS"):
        _configure_continuation_phase(_phase_model(), "C")


def test_schema7_checkpoint_must_be_on_an_optimizer_boundary(tmp_path: Path) -> None:
    checkpoint = tmp_path / "partial.pt"
    with pytest.raises(CheckpointCompatibilityError, match="complete optimizer boundary"):
        assert_checkpoint_optimizer_boundary(
            {
                "checkpoint_schema_version": 7,
                "completed_optimizer_boundary": False,
                "accum_micro_steps": 3,
            },
            checkpoint,
        )
    with pytest.raises(CheckpointCompatibilityError, match="accum_micro_steps=-1"):
        assert_checkpoint_optimizer_boundary(
            {"checkpoint_schema_version": 7}, checkpoint
        )
    assert_checkpoint_optimizer_boundary(
        {
            "checkpoint_schema_version": 7,
            "completed_optimizer_boundary": True,
            "accum_micro_steps": 0,
        },
        checkpoint,
    )


def test_checkpoint_cannot_silently_add_mtp_or_moe(tmp_path: Path) -> None:
    source = _phase_model()
    checkpoint = tmp_path / "dense.pt"
    torch.save(
        {
            "checkpoint_schema_version": 7,
            "completed_optimizer_boundary": True,
            "accum_micro_steps": 0,
            "model": source.state_dict(),
            "model_config": source.model_config(),
        },
        checkpoint,
    )
    mtp_target = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_layer=2,
        block_size=16,
        mod_layers={1},
        use_mtp=True,
    )
    with pytest.raises(CheckpointCompatibilityError, match="use_mtp=False"):
        load_checkpoint(
            mtp_target,
            None,
            None,
            None,
            checkpoint,
            device=torch.device("cpu"),
        )


def test_signed_sampler_transition_requires_exact_checkpoint_token_boundary(
    tmp_path: Path,
) -> None:
    source = _phase_model()
    source.training_recipe = {
        "sampler_algorithm": "counter_based_sha256_v1",
    }
    checkpoint = tmp_path / "sampler-transition.pt"
    torch.save(
        {
            "checkpoint_schema_version": 7,
            "completed_optimizer_boundary": True,
            "accum_micro_steps": 0,
            "model": source.state_dict(),
            "model_config": source.model_config(),
            "training_recipe": dict(source.training_recipe),
            "continuation_token_counts": {"A": 196_608},
        },
        checkpoint,
    )
    target = _phase_model()
    target.training_recipe = {
        "sampler_algorithm": "global_affine_permutation_v1",
    }

    with pytest.raises(CheckpointCompatibilityError, match="sampler_algorithm"):
        load_checkpoint(
            target,
            None,
            None,
            None,
            checkpoint,
            device=torch.device("cpu"),
        )
    with pytest.raises(CheckpointCompatibilityError, match="boundary"):
        load_checkpoint(
            target,
            None,
            None,
            None,
            checkpoint,
            device=torch.device("cpu"),
            sampler_reset_token=200_000,
            continuation_phase="A",
        )

    state = load_checkpoint(
        target,
        None,
        None,
        None,
        checkpoint,
        device=torch.device("cpu"),
        sampler_reset_token=196_608,
        continuation_phase="A",
    )
    assert state["training_recipe"] == source.training_recipe

    assert state["loaded"] is True
    assert state["training_recipe_migrations"] == [
        {
            "field": "sampler_algorithm",
            "saved": "counter_based_sha256_v1",
            "active": "global_affine_permutation_v1",
            "token_boundary": 196_608,
        }
    ]


def test_weighted_training_loss_reports_explicit_answer_and_scaffold_tokens() -> None:
    logits = torch.zeros((1, 3, 4), dtype=torch.float32)
    logits[0, 1, 2] = 4.0
    targets = torch.tensor([[1, 2, 0]])
    weights = torch.tensor([[1.0, 2.0, 0.0]])
    answer_mask = torch.tensor([[False, True, False]])

    loss, sample_losses, breakdown = _weighted_loss(
        logits,
        targets,
        weights,
        answer_mask,
        pad_id=0,
    )

    assert torch.isfinite(loss)
    assert sample_losses.shape == (1,)
    assert int(breakdown["answer_tokens"].item()) == 1
    assert int(breakdown["scaffold_tokens"].item()) == 1
    assert breakdown["answer_nll_sum"] < breakdown["scaffold_nll_sum"]


def test_masked_logit_z_loss_penalizes_overconfidence_and_ignores_padding() -> None:
    targets = torch.tensor([[1, 2, 0]])
    calm = torch.zeros((1, 3, 4))
    extreme = calm.clone()
    extreme[:, :2, 1] = 100.0
    calm_loss = _masked_logit_z_loss(calm, targets, pad_id=0, weight=1e-4)
    extreme_loss = _masked_logit_z_loss(extreme, targets, pad_id=0, weight=1e-4)
    padded_extreme = extreme.clone()
    padded_extreme[:, 2] = 1_000.0
    padded_loss = _masked_logit_z_loss(
        padded_extreme, targets, pad_id=0, weight=1e-4
    )
    assert extreme_loss > calm_loss
    torch.testing.assert_close(padded_loss, extreme_loss)


def test_training_loader_cannot_select_validation_dataset() -> None:
    training = TensorDataset(torch.arange(4))
    validation = TensorDataset(torch.arange(2))
    _assert_training_loader_dataset(DataLoader(training), training, validation)

    with pytest.raises(RuntimeError, match="validation dataset selected"):
        _assert_training_loader_dataset(DataLoader(validation), training, validation)


def test_resume_accepts_matching_data_profile(monkeypatch) -> None:
    monkeypatch.delenv("ANRA_ALLOW_DATA_PROFILE_CHANGE", raising=False)
    assert not _assert_resume_data_profile_compatible("t4-15gb", "t4-15gb")


def test_resume_rejects_changed_data_profile(monkeypatch) -> None:
    monkeypatch.delenv("ANRA_ALLOW_DATA_PROFILE_CHANGE", raising=False)

    with pytest.raises(RuntimeError, match="different data profile"):
        _assert_resume_data_profile_compatible("t4-15gb", "t4-cached")


def test_resume_allows_explicit_profile_experiment(monkeypatch) -> None:
    monkeypatch.setenv("ANRA_ALLOW_DATA_PROFILE_CHANGE", "1")
    assert _assert_resume_data_profile_compatible("t4-15gb", "t4-cached")


def test_resume_rejects_changed_data_layout() -> None:
    with pytest.raises(RuntimeError, match="different training data layout"):
        _assert_resume_data_layout_compatible("legacy_padded_v0", "bucket_packed_v1")


@pytest.mark.parametrize(
    ("saved", "active", "phase"),
    [
        ("bucket_packed_v1", "raw_causal_shards_v1", "A"),
        ("raw_causal_shards_v1", "bucket_packed_v1", "D"),
    ],
)
def test_resume_allows_planned_curriculum_layout_transition(
    saved: str,
    active: str,
    phase: str,
) -> None:
    _assert_resume_data_layout_compatible(saved, active, phase)


def test_current_trainer_enforces_packed_layout(monkeypatch) -> None:
    monkeypatch.delenv("ANRA_TRAINING_DATA_LAYOUT", raising=False)
    assert _active_training_data_layout() == "bucket_packed_v1"

    monkeypatch.setenv("ANRA_TRAINING_DATA_LAYOUT", "legacy_padded_v0")
    with pytest.raises(RuntimeError, match="only supports"):
        _active_training_data_layout()


def test_training_lineage_freezes_checkpoint_tokenizer_and_manifests(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    tokenizer = tmp_path / "tokenizer.json"
    manifest = tmp_path / "manifest.json"
    checkpoint.write_bytes(b"checkpoint-v1")
    tokenizer.write_text('{"token_to_id": {"<pad>": 0}}', encoding="utf-8")
    manifest.write_text('{"shards": []}', encoding="utf-8")
    monkeypatch.setattr("scripts.build_brain.OUTPUT_V2_DIR", tmp_path / "output")

    frozen = _freeze_training_lineage(
        checkpoint_path=checkpoint,
        tokenizer_path=tokenizer,
        model_config={"vocab_size": 8209},
        data_manifests=[manifest],
    )
    checkpoint.write_bytes(b"checkpoint-v2")

    archived = frozen["checkpoint_archive"]
    assert archived is not None
    assert Path(archived).read_bytes() == b"checkpoint-v1"
    assert frozen["data_manifest_sha256"]


def test_checkpoint_collects_complete_manifest_bytes(tmp_path: Path) -> None:
    root = tmp_path / "manifests"
    nested = root / "native" / "manifest.json"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b'{"shards":[{"sha256":"abc"}]}')

    hashes, payloads = _collect_data_manifest_payloads(root)

    assert payloads == {"native/manifest.json": nested.read_bytes()}
    assert hashes["native/manifest.json"] == hashlib.sha256(nested.read_bytes()).hexdigest()


def test_checkpoint_tokenizer_contract_uses_live_active_identity(monkeypatch) -> None:
    identity = {
        "available": True,
        "schema_version": 4,
        "sha256": "v4-file-hash",
        "vocabulary_sha256": "v4-vocabulary-hash",
        "vocab_size": 32_768,
        "special_token_ids": {"<pad>": 0, "<unk>": 1},
        "probe_count": 500,
        "probe_sha256": "v4-probe-hash",
    }
    monkeypatch.setattr(
        "scripts.build_brain.active_tokenizer_identity", lambda: identity
    )

    contract = _tokenizer_checkpoint_contract()

    assert contract["schema_version"] == 4
    assert contract["vocab_size"] == 32_768
    assert contract["sha256"] == "v4-file-hash"
    assert contract["probe_sha256"] == "v4-probe-hash"
