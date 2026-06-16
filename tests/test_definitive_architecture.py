from __future__ import annotations

import json

import numpy as np
import torch

from evaluation.ibs import IBSBenchmark
from robotics.world_model import PredictiveWorldModel, WorldModelCodec, train_world_model_offline
from training.continual import (
    assess_continual_readiness,
    proposal_auto_application_allowed,
)
from training.data_ledger import DataQuality
from training.data_pipeline_v3 import SourceRecord, TokenShardPublisher
from training.csii import CrossScaleIdentityInheritance
from training.ssg import SovereignScalingGovernor
from training.v2_runtime import migrate_checkpoint_state
from anra_brain import CausalTransformerV2
from runtime.technology_registry import validate_technology_reachability


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


def _quality() -> DataQuality:
    return DataQuality(0.5, 1.0, 1.0, 1.0, 1.0, 1.0)


def test_token_shards_are_uint16_hashed_and_immutable(tmp_path) -> None:
    output = tmp_path / "tokens"
    publisher = TokenShardPublisher(
        output,
        tokenizer_version="v3-test",
        tokens_per_shard=4,
    )
    manifest = publisher.publish(
        [
            SourceRecord(
                "abcdefgh",
                "FineWeb-Edu",
                "ODC-By",
                "foundation",
                _quality(),
                source_revision="unit",
            )
        ],
        _Tokenizer(),
    )
    assert manifest["total_tokens"] == 8
    assert all(item["tokens"] == 4 for item in manifest["shards"])
    assert np.load(output / manifest["shards"][0]["path"]).dtype == np.uint16
    try:
        publisher.publish([], _Tokenizer())
    except FileExistsError:
        pass
    else:
        raise AssertionError("published token shards must be immutable")


def test_checkpoint_migration_preserves_legacy_rows_and_initializes_dstp() -> None:
    legacy_embedding = torch.randn(8192, 4)
    legacy_depth = torch.randn(2)
    source = {
        "token_embedding_table.weight": legacy_embedding,
        "lm_head.weight": legacy_embedding.clone(),
        "dstp_logits": legacy_depth,
    }
    target = {
        "token_embedding_table.weight": torch.zeros(8209, 4),
        "lm_head.weight": torch.zeros(8209, 4),
        "residual_depth_logits": torch.zeros(2),
        "dstp_temperature_log": torch.ones(2),
    }
    migrated, report = migrate_checkpoint_state(source, target)
    torch.testing.assert_close(
        migrated["token_embedding_table.weight"][:8192],
        legacy_embedding,
    )
    torch.testing.assert_close(migrated["residual_depth_logits"], legacy_depth)
    torch.testing.assert_close(migrated["dstp_temperature_log"], torch.ones(2))
    assert report["source_vocab_size"] == 8192
    assert report["target_vocab_size"] == 8209
    assert report["appended_token_rows"] == 17


def test_checkpoint_vocabulary_migration_is_deterministic() -> None:
    legacy = torch.randn(8192, 4)
    source = {
        "token_embedding_table.weight": legacy,
        "lm_head.weight": legacy.clone(),
    }
    first, _ = migrate_checkpoint_state(
        source,
        {
            "token_embedding_table.weight": torch.randn(8209, 4),
            "lm_head.weight": torch.randn(8209, 4),
        },
    )
    second, _ = migrate_checkpoint_state(
        source,
        {
            "token_embedding_table.weight": torch.randn(8209, 4),
            "lm_head.weight": torch.randn(8209, 4),
        },
    )
    torch.testing.assert_close(
        first["token_embedding_table.weight"],
        second["token_embedding_table.weight"],
    )
    torch.testing.assert_close(
        first["token_embedding_table.weight"],
        first["lm_head.weight"],
    )


def test_ssg_growth_phase_defers_only_parity(tmp_path) -> None:
    governor = SovereignScalingGovernor()
    result = governor.check(
        phase="growth",
        checkpoint_path=tmp_path / "missing.pt",
        ibs_path=tmp_path / "ibs.json",
        civ_path=tmp_path / "civ.json",
        memory_profile_path=tmp_path / "profile.json",
        growth_report_path=tmp_path / "growth.json",
        token_manifest_path=tmp_path / "tokens.json",
        tokenizer_manifest_path=tmp_path / "tokenizer.json",
    )
    assert "growth parity deferred until candidate construction" in result.passed
    assert not any("model-growth parity report" in item for item in result.blockers)


def test_continual_threshold_and_proposal_policy() -> None:
    assert assess_continual_readiness(99)["action"] == "skip"
    assert assess_continual_readiness(100)["ready"] is True
    assert proposal_auto_application_allowed([True, False, False, False, False])
    assert not proposal_auto_application_allowed([False] * 5)


def test_three_seed_ibs_writes_aggregate(tmp_path) -> None:
    report = IBSBenchmark().run_three_seed(
        lambda prompt, seed: f"{prompt}:{seed}",
        lambda task, response: (1.0, ""),
        label="unit",
        output_path=tmp_path / "ibs.json",
    )
    assert report["seed_count"] == 3
    assert report["overall"] == 1.0
    assert json.loads((tmp_path / "ibs.json").read_text())["seed_count"] == 3


def test_world_model_requires_full_activation_evidence() -> None:
    assert not PredictiveWorldModel.activation_allowed(
        simulation_transitions=99_999,
        held_out_accuracy=0.90,
        planning_improvement=0.20,
    )
    assert PredictiveWorldModel.activation_allowed(
        simulation_transitions=100_000,
        held_out_accuracy=0.70,
        planning_improvement=0.10,
    )


def test_world_model_offline_training_reports_boundary(tmp_path) -> None:
    transitions = tmp_path / "transitions.jsonl"
    transitions.write_text(
        json.dumps(
            {
                "state": {"x": 0},
                "action": {"skill": "move"},
                "next_state": {"x": 1},
                "reward": 1.0,
                "terminal": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    model = PredictiveWorldModel(state_dim=8, action_dim=4, hidden_dim=16)
    report = train_world_model_offline(
        model,
        transitions,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        codec=WorldModelCodec(state_dim=8, action_dim=4),
    )
    assert report["transition_count"] == 1
    assert report["offline_only"] is True


def test_function_preserving_growth_handles_gqa_head_change() -> None:
    torch.manual_seed(9)
    source = CausalTransformerV2(
        vocab_size=64,
        n_embd=128,
        n_head=16,
        n_kv_head=4,
        n_layer=1,
        block_size=8,
        mod_layers=set(),
        use_hal=False,
    )
    target = CausalTransformerV2(
        vocab_size=64,
        n_embd=160,
        n_head=20,
        n_kv_head=5,
        n_layer=2,
        block_size=8,
        mod_layers=set(),
        use_hal=False,
    )
    CrossScaleIdentityInheritance.grow(source, target)
    report = CrossScaleIdentityInheritance.verify_parity(
        source,
        target,
        torch.randint(0, 64, (1, 8)),
    )
    assert report["parity_cosine"] > 0.999


def test_all_t01_through_t26_entrypoints_are_reachable() -> None:
    reachable = validate_technology_reachability()
    assert list(reachable) == [f"T-{index:02d}" for index in range(1, 27)]


def test_all_c01_through_c07_entrypoints_are_reachable() -> None:
    from runtime.cognition_registry import (
        COGNITIVE_CAPABILITIES,
        validate_cognition_reachability,
    )

    reachable = validate_cognition_reachability()
    expected = [
        f"C-{index:02d}"
        for index in range(1, len(COGNITIVE_CAPABILITIES) + 1)
    ]
    assert list(reachable) == expected, (
        f"Cognitive contracts {set(expected) - set(reachable)} are not reachable. "
        "Update runtime/cognition_registry.py if the entrypoint changed."
    )
