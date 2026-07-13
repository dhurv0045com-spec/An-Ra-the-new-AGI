from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.ibs import IBSBenchmark
from robotics.world_model import PredictiveWorldModel, WorldModelCodec, train_world_model_offline
from training.continual import (
    assess_continual_readiness,
    proposal_auto_application_allowed,
)
from training.anra_optimizer import build_append_only_row_learning_rate
from training.data_ledger import DataQuality
from training.data_pipeline_v3 import SourceRecord, TokenShardPublisher
from training.v2_data_mix import (
    RawCausalShardDataset,
    TrainingExample,
    WindowConsumptionTracker,
    build_post_training_mix,
)
from training.csii import CrossScaleIdentityInheritance
from training.ssg import SovereignScalingGovernor
from training.v2_runtime import migrate_checkpoint_state
from training.v2_config import frontier_parameter_count
from anra_brain import CausalTransformerV2
from runtime.technology_registry import validate_technology_reachability
from scripts.download_training_data import (
    MinHashDeduplicator,
    _code_syntax_valid,
    _detect_content_language,
    _math_text_valid,
)
from tokenizer.validate_tokenizer_v3 import build_append_only_v4


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
    assert manifest["source_token_mix"] == {"FineWeb-Edu": 8}
    assert all(item["tokens"] == 4 for item in manifest["shards"])
    assert np.load(output / manifest["shards"][0]["path"]).dtype == np.uint16
    try:
        publisher.publish([], _Tokenizer())
    except FileExistsError:
        pass
    else:
        raise AssertionError("published token shards must be immutable")


def test_raw_causal_dataset_trains_every_next_token(tmp_path) -> None:
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 3

        @staticmethod
        def encode(text: str) -> list[int]:
            return [4 + ord(character) % 50 for character in text]

        @staticmethod
        def decode(ids: list[int]) -> str:
            return " ".join(str(value) for value in ids)

    output = tmp_path / "raw"
    publisher = TokenShardPublisher(
        output,
        tokenizer_version="v3-test",
        tokenizer_sha256="tokenizer-hash",
        tokens_per_shard=16,
    )
    publisher.publish(
        [
            SourceRecord(
                "abcdefghijklmnopqrstuvwxyz",
                "unit",
                "MIT",
                "foundation",
                _quality(),
            )
        ],
        Tokenizer(),
        allow_partial_final=True,
    )
    dataset = RawCausalShardDataset(
        output / "manifest.json",
        Tokenizer(),
        block_size=8,
        expected_tokenizer_sha256="tokenizer-hash",
    )
    x, y, weights, _, answer_mask = dataset[0]
    torch.testing.assert_close(x[1:], y[:-1])
    assert torch.all(weights == 1)
    assert not answer_mask.any()
    assert dataset.bucket_for_window(0) == "foundation"
    assert dataset.bucket_for_sample(dataset[0][3]) == "foundation"


def test_token_shards_preserve_campaign_source_class_boundaries(tmp_path) -> None:
    output = tmp_path / "source-pure"
    manifest = TokenShardPublisher(
        output,
        tokenizer_version="v3-test",
        tokenizer_sha256="hash",
        tokens_per_shard=64,
    ).publish(
        [
            SourceRecord(
                "abcdefghijk",
                "FineWeb-Edu",
                "ODC-By",
                "foundation",
                _quality(),
                source_class="fineweb_edu",
            ),
            SourceRecord(
                "def valid_code(): return 1",
                "Common Pile Stack v2 open code",
                "MIT AND Apache-2.0",
                "foundation",
                _quality(),
                source_class="permissive_code",
            ),
        ],
        _Tokenizer(),
        allow_partial_final=True,
    )

    assert [shard["source_class"] for shard in manifest["shards"]] == [
        "fineweb_edu",
        "permissive_code",
    ]
    assert manifest["source_class_token_mix"] == {
        "fineweb_edu": 11,
        "permissive_code": 26,
    }
    dataset = RawCausalShardDataset(
        output / "manifest.json",
        type(
            "Tokenizer",
            (),
            {"pad_token_id": 0, "encode": staticmethod(_Tokenizer().encode)},
        )(),
        block_size=4,
        expected_tokenizer_sha256="hash",
    )
    assert set(dataset.source_window_ranges()) == {"fineweb_edu", "permissive_code"}


def test_short_identity_source_is_explicitly_replayed_to_trainable_windows(
    tmp_path,
) -> None:
    output = tmp_path / "identity-replay"
    manifest = TokenShardPublisher(
        output,
        tokenizer_version="v3-test",
        tokenizer_sha256="hash",
        tokens_per_shard=10_000,
    ).publish(
        [
            SourceRecord(
                "H: Who are you?\nANRA: I am An-Ra.",
                "An-Ra identity replay",
                "owner",
                "identity",
                _quality(),
                source_class="identity_replay",
            )
        ],
        _Tokenizer(),
        allow_partial_final=True,
        minimum_replay_tokens={"identity_replay": 4097},
    )

    assert manifest["source_class_token_mix"]["identity_replay"] == 4097
    assert manifest["source_class_replayed_tokens"]["identity_replay"] > 0
    dataset = RawCausalShardDataset(
        output / "manifest.json",
        type(
            "Tokenizer",
            (),
            {"pad_token_id": 0, "encode": staticmethod(_Tokenizer().encode)},
        )(),
        block_size=2048,
        expected_tokenizer_sha256="hash",
    )
    assert len(dataset) == 2
    assert set(dataset.source_window_ranges()) == {"identity_replay"}


def test_raw_window_ids_are_stable_across_shard_rotation(tmp_path) -> None:
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = -1

        @staticmethod
        def encode(text: str) -> list[int]:
            return [ord(character) for character in text]

        @staticmethod
        def decode(ids: list[int]) -> str:
            return "".join(chr(value) for value in ids)

    output = tmp_path / "rotated"
    TokenShardPublisher(
        output,
        tokenizer_version="test",
        tokenizer_sha256="hash",
        tokens_per_shard=8,
    ).publish(
        [SourceRecord("abcdefghijklmnopqrstuvwxyz", "unit", "MIT", "foundation", _quality())],
        Tokenizer(),
        allow_partial_final=True,
    )
    first = RawCausalShardDataset(
        output / "manifest.json",
        Tokenizer(),
        block_size=4,
        rotation_seed=0,
        expected_tokenizer_sha256="hash",
    )
    rotated = RawCausalShardDataset(
        output / "manifest.json",
        Tokenizer(),
        block_size=4,
        rotation_seed=1,
        expected_tokenizer_sha256="hash",
    )
    assert {first[index][3] for index in range(len(first))} == {
        rotated[index][3] for index in range(len(rotated))
    }


def test_window_consumption_tracker_persists_unique_and_repeat_counts() -> None:
    tracker = WindowConsumptionTracker(16, 1024)
    tracker.mark([1, 2, 2, 3])
    restored = WindowConsumptionTracker(16, 1024, state=tracker.state_dict())
    report = restored.report(phase_target_tokens=10_000)

    assert report["unique_tokens_consumed"] == 3 * 1024
    assert report["repeated_windows"] == 1
    assert report["repeated_token_percentage"] == 25.0
    assert report["remaining_phase_tokens"] == 10_000 - 3 * 1024


def test_native_corpus_quality_filters_cover_near_duplicates_and_domains() -> None:
    deduplicator = MinHashDeduplicator()
    base = (
        "The model learns from verified technical documents and preserves the "
        "complete implementation context for every training example. "
    ) * 12
    near_duplicate = base + "The final sentence is metadata only."

    assert deduplicator.seen_or_add(base) is False
    assert deduplicator.seen_or_add(near_duplicate) is True
    assert _detect_content_language(base, source="FineWeb-Edu") == "en"
    assert _detect_content_language("数学模型数据", source="FineWeb-Edu") == "unknown"
    assert _code_syntax_valid(
        "def add(a, b):\n    return a + b\n",
        source="The Stack v2 dedup",
        language_hint="python",
    )
    assert not _code_syntax_valid(
        "def broken(:\n    pass\n",
        source="The Stack v2 dedup",
        language_hint="python",
    )
    assert _math_text_valid("Final answer: 2 + 3 = 5", source="FineMath-4+")
    assert not _math_text_valid("Final answer: 2 + 3 = 9", source="FineMath-4+")


def test_post_training_mix_is_proportional_and_without_replacement() -> None:
    category_examples = {
        "instruction": ("teacher", "instruction"),
        "code": ("teacher", "code"),
        "math_logic": ("teacher", "math"),
        "tools_actions": ("frontier_dfc", "tool"),
        "failure_replay": ("replay", "reasoning"),
        "identity": ("identity", "identity"),
    }
    examples = [
        TrainingExample(
            bucket=bucket,
            prompt=f"{category} prompt {index}",
            answer=f"{category} answer {index}",
            source=f"{category}-{index}",
            metadata={
                "task_type": task_type,
                "verified": category == "failure_replay",
            },
        )
        for category, (bucket, task_type) in category_examples.items()
        for index in range(50)
    ]

    mixed, requested, realized = build_post_training_mix(
        examples,
        seed=11,
        max_examples=100,
    )

    assert len(mixed) == 100
    assert len({(item.source, item.prompt, item.answer) for item in mixed}) == 100
    assert requested == {
        "instruction": 35,
        "code": 25,
        "math_logic": 15,
        "tools_actions": 10,
        "failure_replay": 10,
        "identity": 5,
    }
    assert realized == requested


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
        "layer_temperature_bias_log": torch.zeros(2),
    }
    migrated, report = migrate_checkpoint_state(source, target)
    torch.testing.assert_close(
        migrated["token_embedding_table.weight"][:8192],
        legacy_embedding,
    )
    torch.testing.assert_close(migrated["residual_depth_logits"], legacy_depth)
    torch.testing.assert_close(migrated["dstp_temperature_log"], torch.ones(2))
    torch.testing.assert_close(migrated["layer_temperature_bias_log"], torch.zeros(2))
    assert report["source_vocab_size"] == 8192
    assert report["target_vocab_size"] == 8209
    assert report["appended_token_rows"] == 17


def test_checkpoint_migration_converts_legacy_temperature_scale_to_log_parameter() -> None:
    source = {
        "token_embedding_table.weight": torch.ones(8, 4),
        "layer_temperature_bias": torch.tensor([0.5, 1.0, 2.0]),
    }
    target = {
        "token_embedding_table.weight": torch.ones(8, 4),
        "layer_temperature_bias_log": torch.zeros(3),
    }

    migrated, report = migrate_checkpoint_state(source, target)

    assert "layer_temperature_bias" not in migrated
    torch.testing.assert_close(
        migrated["layer_temperature_bias_log"],
        torch.tensor([0.5, 1.0, 2.0]).log(),
    )
    assert "layer_temperature_bias->layer_temperature_bias_log" in report["changes"]
    assert report["schema_version"] == 7


def test_checkpoint_migration_rejects_invalid_legacy_temperature_scale() -> None:
    source = {
        "token_embedding_table.weight": torch.ones(8, 4),
        "layer_temperature_bias": torch.tensor([1.0, 0.0]),
    }
    target = {
        "token_embedding_table.weight": torch.ones(8, 4),
        "layer_temperature_bias_log": torch.zeros(2),
    }

    with pytest.raises(ValueError, match="finite positive"):
        migrate_checkpoint_state(source, target)


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


def test_checkpoint_migration_names_and_neutralizes_new_native_tensors() -> None:
    source = {"token_embedding_table.weight": torch.ones(8, 4)}
    target = {
        "token_embedding_table.weight": torch.ones(8, 4),
        "mod_routers.0.gate.weight": torch.randn(1, 4),
        "mod_routers.0.context_weights": torch.zeros(3),
        "rim_modules.0.raw_alpha": torch.zeros(()),
        "esv_module.predictor.0.weight": torch.zeros(3, 4),
        "residual_depth_logits": torch.zeros(2),
    }
    migrated, report = migrate_checkpoint_state(source, target)

    torch.testing.assert_close(
        migrated["mod_routers.0.gate.weight"],
        torch.zeros(1, 4),
    )
    assert set(migrated) == set(target)
    assert "initialize_native:residual_depth_logits" in report["changes"]


def test_append_only_rows_receive_three_times_realized_update() -> None:
    class ExpandedEmbedding(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding_table = torch.nn.Embedding(6, 2)

    model = ExpandedEmbedding()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    controller = build_append_only_row_learning_rate(
        model,
        base_rows=4,
        multiplier=3.0,
        max_steps=2,
    )
    assert controller is not None
    before_all = model.token_embedding_table.weight.detach().clone()
    model.token_embedding_table.weight.grad = torch.ones_like(model.token_embedding_table.weight)
    appended_before = controller.capture()
    optimizer.step()
    controller.apply(appended_before)
    delta = before_all - model.token_embedding_table.weight.detach()
    torch.testing.assert_close(delta[:4], torch.full((4, 2), 0.1))
    torch.testing.assert_close(delta[4:], torch.full((2, 2), 0.3))
    assert controller.report()["steps_completed"] == 1


def test_append_only_v4_preserves_every_v3_token_id(tmp_path) -> None:
    source = Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer_v3.json"
    source_meta = source.with_suffix(source.suffix + ".meta.json")
    tokenizer_path = tmp_path / "tokenizer_v3.json"
    tokenizer_path.write_bytes(source.read_bytes())
    tokenizer_path.with_suffix(tokenizer_path.suffix + ".meta.json").write_bytes(
        source_meta.read_bytes()
    )
    original = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    output = tmp_path / "tokenizer_v4.json"
    build_append_only_v4(
        tokenizer_path,
        output,
        {
            "eligible_for_schema_v4": True,
            "sampled_units": 1_000_000,
            "projected_reduction": 0.20,
            "candidate_tokens": ["native_extension_token"],
        },
    )
    grown = json.loads(output.read_text(encoding="utf-8"))
    assert grown["id_to_token"][:8209] == original["id_to_token"]
    assert len(grown["id_to_token"]) == 16_384
    assert (
        frontier_parameter_count(16_384) - frontier_parameter_count(8_209)
        == (16_384 - 8_209) * 1_280
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
    expected = [f"C-{index:02d}" for index in range(1, len(COGNITIVE_CAPABILITIES) + 1)]
    assert list(reachable) == expected, (
        f"Cognitive contracts {set(expected) - set(reachable)} are not reachable. "
        "Update runtime/cognition_registry.py if the entrypoint changed."
    )
