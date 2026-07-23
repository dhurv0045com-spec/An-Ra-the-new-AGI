from __future__ import annotations

import json
from dataclasses import asdict
import hashlib

import pytest
import torch

from anra.architecture import (
    FRONTIER,
    GROWTH_500M,
    GROWTH_500M_PARAMETER_COUNT,
    verify_growth_counts,
)
from anra_brain import CausalTransformerV2
from training.csii import (
    CrossScaleIdentityInheritance,
    model_architecture_sha256,
)
from training.growth_runtime import load_growth_training_pair
from training.v2_config import (
    ANRA_V4_GROWTH_MODEL,
    ANRA_V4_GROWTH_MODEL_PARAMETER_COUNT,
    ANRA_V4_GROWTH_MODEL_PROFILE,
    CANONICAL_MODEL_PROFILE,
    CHECKPOINT_SCHEMA_VERSION,
    EXPERIMENTAL_MODEL_PROFILES,
    MODEL_SIZES,
    model_parameter_count,
    model_profile_registration,
    resolve_model_profile,
)
from training.v2_runtime import build_model_for_profile


def test_500m_growth_profile_has_exact_registered_contract() -> None:
    assert verify_growth_counts() == {
        "growth_500m_transformer": 497_652_480,
        "growth_500m_full": GROWTH_500M_PARAMETER_COUNT,
    }
    assert GROWTH_500M_PARAMETER_COUNT == 499_880_031
    assert model_parameter_count(ANRA_V4_GROWTH_MODEL) == 499_880_031
    assert ANRA_V4_GROWTH_MODEL_PARAMETER_COUNT == 499_880_031
    assert GROWTH_500M.vocab_size == FRONTIER.vocab_size == 32_768
    assert GROWTH_500M.context_length == FRONTIER.context_length == 2_048
    assert GROWTH_500M.mod_layers == tuple(range(4, 27, 2))

    # Meta construction validates the real module graph without allocating a
    # roughly 2 GB fp32 model during a unit test.
    with torch.device("meta"):
        model = build_model_for_profile(
            ANRA_V4_GROWTH_MODEL_PROFILE,
            allow_experimental=True,
        )
    assert sum(parameter.numel() for parameter in model.parameters()) == 499_880_031


def test_growth_profile_requires_explicit_experimental_resolution() -> None:
    assert set(MODEL_SIZES) == {CANONICAL_MODEL_PROFILE}
    assert set(EXPERIMENTAL_MODEL_PROFILES) == {ANRA_V4_GROWTH_MODEL_PROFILE}
    with pytest.raises(ValueError, match="experimental"):
        resolve_model_profile(ANRA_V4_GROWTH_MODEL_PROFILE)
    with pytest.raises(ValueError, match="experimental"):
        build_model_for_profile(ANRA_V4_GROWTH_MODEL_PROFILE)
    model, _training = resolve_model_profile(
        ANRA_V4_GROWTH_MODEL_PROFILE,
        allow_experimental=True,
    )
    assert model is ANRA_V4_GROWTH_MODEL
    registration = model_profile_registration(ANRA_V4_GROWTH_MODEL_PROFILE)
    assert registration.parent_profile == CANONICAL_MODEL_PROFILE
    assert registration.requires_growth_manifest is True
    assert registration.scratch_training_allowed is False


def _small_model(*, width: int, heads: int, kv_heads: int, layers: int) -> CausalTransformerV2:
    return CausalTransformerV2(
        vocab_size=64,
        n_embd=width,
        n_head=heads,
        n_kv_head=kv_heads,
        n_layer=layers,
        block_size=8,
        d_ff=256,
        mod_layers=set(),
        use_hal=False,
        sliding_window=4,
        full_attention_every=2,
    )


def test_growth_preserves_interleaved_attention_and_binds_real_logits(
    tmp_path,
) -> None:
    torch.manual_seed(9)
    source = _small_model(width=128, heads=16, kv_heads=4, layers=4)
    target = _small_model(width=160, heads=20, kv_heads=5, layers=6)
    source_checkpoint = tmp_path / "parent.pt"
    source_checkpoint.write_bytes(b"hash-bound-parent")

    report = CrossScaleIdentityInheritance.grow(
        source,
        target,
        source_checkpoint=source_checkpoint,
        source_profile="small-parent",
        target_profile="small-child",
    )

    mapping = dict(report.layer_mapping)
    assert mapping == {0: 0, 2: 1, 4: 2, 5: 3}
    # Target layer 2 would be sliding under its own periodic pattern, but it
    # inherits source layer 1 and therefore must remain full attention.
    assert target.blocks[2].attn.sliding_window is None
    assert report.attention_mode_mapping[2] == (2, 1, "full", None)
    assert len(report.source_checkpoint_sha256) == 64
    assert report.optimizer_restart_required is True
    assert report.optimizer_state_inherited is False

    token_ids = torch.randint(0, 64, (1, 8))
    parity = CrossScaleIdentityInheritance.verify_parity(source, target, token_ids)
    assert parity["parity_semantics"] == "real_logits_same_token_ids_v1"
    report = CrossScaleIdentityInheritance.bind_parity(
        report,
        parity,
        minimum_cosine=0.99,
    )
    payload = CrossScaleIdentityInheritance.validate_growth_report(report)
    assert payload["parity_passed"] is True
    assert float(payload["parity_cosine"]) > 0.999

    report_path = CrossScaleIdentityInheritance.write_report(
        report,
        tmp_path / "growth.json",
    )
    stored = json.loads(report_path.read_text(encoding="utf-8"))
    assert stored["source_architecture_sha256"] == report.source_architecture_sha256
    assert stored["target_architecture_sha256"] == report.target_architecture_sha256
    assert stored["optimizer_restart_required"] is True

    reconstructed = _small_model(width=160, heads=20, kv_heads=5, layers=6)
    CrossScaleIdentityInheritance.apply_attention_mode_mapping(reconstructed, stored)
    assert model_architecture_sha256(reconstructed) == report.target_architecture_sha256


def test_architecture_hash_includes_effective_layer_attention_modes() -> None:
    model = _small_model(width=128, heads=16, kv_heads=4, layers=4)
    before = model_architecture_sha256(model)
    model.blocks[0].attn.sliding_window = None
    assert model_architecture_sha256(model) != before


def test_growth_runtime_loads_child_and_teacher_without_optimizer_inheritance(
    tmp_path,
    monkeypatch,
) -> None:
    torch.manual_seed(17)
    source = _small_model(width=128, heads=16, kv_heads=4, layers=4)
    parent_progress = {
        "tokens_seen": 2048,
        "continuation_token_counts": {"A": 2048},
        "raw_window_consumption": {"visited": [0]},
        "data_sampler_state": {"position": 1},
        "data_profile": "test-profile",
        "training_data_layout": "raw_causal_shards_v1",
        "seed_contract": {"seed": 1301},
        "data_manifests": {"train.json": "a" * 64},
        "best_validation_loss": 3.0,
        "best_answer_validation_loss": float("inf"),
        "validation_history": [],
    }
    parent_path = tmp_path / "parent.pt"
    torch.save(
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_artifact_class": "full_resume",
            "completed_optimizer_boundary": True,
            "model": source.state_dict(),
            **parent_progress,
        },
        parent_path,
    )
    target = _small_model(width=160, heads=20, kv_heads=5, layers=6)
    report = CrossScaleIdentityInheritance.grow(
        source,
        target,
        source_checkpoint=parent_path,
        source_profile=CANONICAL_MODEL_PROFILE,
        target_profile=ANRA_V4_GROWTH_MODEL_PROFILE,
    )
    parity = CrossScaleIdentityInheritance.verify_parity(
        source,
        target,
        torch.randint(0, 64, (1, 8)),
    )
    report = CrossScaleIdentityInheritance.bind_parity(report, parity)
    report_path = CrossScaleIdentityInheritance.write_report(
        report,
        tmp_path / "growth.json",
    )
    initialization_path = tmp_path / "growth-init.pt"
    torch.save(
        {
            "artifact_class": "growth_initialization",
            "training_resume_allowed": False,
            "optimizer_restart_required": True,
            "optimizer_state_inherited": False,
            "model_profile": ANRA_V4_GROWTH_MODEL_PROFILE,
            "model": target.state_dict(),
            "growth_manifest": asdict(report),
            "parent_progress": parent_progress,
        },
        initialization_path,
    )

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    initialization_path.with_suffix(initialization_path.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "schema": "anra-growth-initialization/v1",
                "artifact_class": "growth_initialization",
                "artifact_sha256": digest(initialization_path),
                "growth_manifest_sha256": digest(report_path),
                "source_checkpoint_sha256": digest(parent_path),
                "source_profile": CANONICAL_MODEL_PROFILE,
                "target_profile": ANRA_V4_GROWTH_MODEL_PROFILE,
                "optimizer_restart_required": True,
                "optimizer_state_inherited": False,
                "training_resume_allowed": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "training.growth_runtime.build_model_for_profile",
        lambda _profile: _small_model(width=128, heads=16, kv_heads=4, layers=4),
    )
    reconstructed = _small_model(width=160, heads=20, kv_heads=5, layers=6)
    teacher, controller, provenance = load_growth_training_pair(
        reconstructed,
        initialization_path=initialization_path,
        growth_manifest_path=report_path,
        parent_checkpoint_path=parent_path,
    )

    assert model_architecture_sha256(reconstructed) == report.target_architecture_sha256
    assert model_architecture_sha256(teacher) == report.source_architecture_sha256
    assert controller.identity_layers == report.identity_layers
    assert provenance["parent_progress"] == parent_progress
    assert provenance["optimizer_restart_required"] is True
