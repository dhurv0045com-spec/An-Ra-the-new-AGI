from pathlib import Path

import pytest
import torch

from anra_core.config import CoreConfig
from anra_core.model import AnRaCore
from training.state import (
    CosineSchedule,
    DataPosition,
    ResumableDistributedSampler,
    build_training_state,
    dataset_fingerprint,
    validate_full_resume,
    validate_training_state,
)
from training.train_xla import _clip_global_grad_norm, parse_args


def _tiny_config() -> CoreConfig:
    return CoreConfig(
        vocab_size=64,
        d_model=16,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=8,
        d_ff=32,
        block_size=16,
        base_seq_len=16,
        target_seq_len=16,
        sliding_window=4,
        full_attention_every=2,
    )


def test_tiled_attention_and_rematerialization_preserve_model_contract() -> None:
    torch.manual_seed(7)
    reference = AnRaCore(_tiny_config()).train()
    optimized = AnRaCore(_tiny_config()).train()
    optimized.load_state_dict(reference.state_dict())
    state_keys = tuple(optimized.state_dict())
    optimized.enable_memory_efficient_attention(3)
    optimized.enable_gradient_checkpointing(True)
    assert tuple(optimized.state_dict()) == state_keys

    tokens = torch.randint(0, 64, (2, 9))
    expected = reference(tokens)
    actual = optimized(tokens)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    actual.square().mean().backward()
    assert all(parameter.grad is not None for parameter in optimized.parameters())


def test_resumable_distributed_sampler_continues_at_exact_rank_offset() -> None:
    dataset = list(range(41))
    full = ResumableDistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=True, seed=1301, drop_last=True
    )
    full.set_epoch(5)
    expected = list(full)

    resumed = ResumableDistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=True, seed=1301, drop_last=True
    )
    resumed.set_epoch(5)
    resumed.set_start_index(7)
    assert list(resumed) == expected[7:]
    assert len(resumed) == len(expected) - 7


def test_data_position_and_schedule_are_stable_across_resume() -> None:
    position = DataPosition.from_microbatches(23, batches_per_epoch=10)
    assert position == DataPosition(epoch=2, batch_in_epoch=3, microbatches_consumed=23)

    schedule = CosineSchedule(base_lr=2e-4, min_lr=2e-5, origin_step=20_000, decay_steps=100)
    payload = {"lr_schedule": schedule.to_dict()}
    restored = CosineSchedule.from_checkpoint(
        payload,
        start_step=20_050,
        checkpoint_lr=9e-4,
        decay_steps=999,
        min_lr_ratio=0.5,
    )
    assert restored == schedule
    assert restored.lr_at(20_000) == pytest.approx(2e-4)
    assert restored.lr_at(20_100) == pytest.approx(2e-5)


def test_full_resume_and_recipe_drift_fail_closed() -> None:
    payload = {
        "checkpoint_artifact_class": "full_resume",
        "checkpoint_schema_version": 1,
        "global_step": 20_000,
        "optimizer_state_dict": {"state": {}, "param_groups": []},
    }
    assert validate_full_resume(payload, minimum_step=20_000) == 20_000
    with pytest.raises(ValueError, match="full_resume"):
        validate_full_resume({**payload, "checkpoint_artifact_class": "model_only"}, minimum_step=0)
    with pytest.raises(ValueError, match="trainer_state"):
        validate_full_resume({**payload, "checkpoint_schema_version": 2}, minimum_step=0)

    position = DataPosition(0, 0, 0)
    current = build_training_state(
        step=20_000,
        optimizer_updates=0,
        position=position,
        dataset_sha256="a" * 64,
        dataset_windows=100,
        batch_size=1,
        grad_accum_steps=1,
        world_size=8,
        sequence_length=2048,
        seed=1301,
        attention_chunk_size=128,
        gradient_checkpointing=True,
        gradient_clip_norm=1.0,
    )
    validate_training_state(current, current, allow_legacy=False)
    with pytest.raises(ValueError, match="world_size"):
        validate_training_state({**current, "world_size": 4}, current, allow_legacy=False)
    with pytest.raises(ValueError, match="predates"):
        validate_training_state({}, current, allow_legacy=False)
    validate_training_state({}, current, allow_legacy=True)


def test_dataset_fingerprint_is_path_independent_and_content_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "train.txt").write_text("same", encoding="utf-8")
    (second / "train.txt").write_text("same", encoding="utf-8")
    assert dataset_fingerprint(first / "train.txt") == dataset_fingerprint(second / "train.txt")
    (second / "train.txt").write_text("changed", encoding="utf-8")
    assert dataset_fingerprint(first / "train.txt") != dataset_fingerprint(second / "train.txt")


def test_global_gradient_clipping_and_lineage_defaults() -> None:
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    norm = _clip_global_grad_norm(iter([parameter]), 1.0)
    assert float(norm) == pytest.approx(5.0)
    assert float(parameter.grad.norm()) == pytest.approx(1.0, rel=1e-5)

    args = parse_args(["--dataset-path", "data", "--output-checkpoint", "out.pt"])
    assert args.expected_resume_step == 20_000
    assert args.grad_accum_steps == 1
    assert args.require_world_size == 8
    assert args.gradient_checkpointing is True
