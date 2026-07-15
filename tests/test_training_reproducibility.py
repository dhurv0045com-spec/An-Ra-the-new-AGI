from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from training.curriculum_sampler import (
    SAMPLER_ALGORITHM,
    ScheduledCurriculumSampler,
)
from training.reproducibility import (
    DETERMINISM_MODE,
    CANONICAL_TRAINING_SEED,
    capture_rng_states,
    make_data_generator,
    restore_rng_states,
    seed_everything,
)
from training.mixed_precision import MixedPrecisionTrainer
from training.scheduler import get_cosine_schedule_with_warmup
from training.v2_config import (
    ANRA_V4_TRAINING,
    CANONICAL_FOUNDATION_OPTIMIZER,
)
from training.verified_process import VERIFIED_PROCESS_OBJECTIVE
from training.v2_runtime import load_checkpoint
from anra_brain import CausalTransformerV2


def test_seed_is_a_replay_address_not_a_quality_parameter() -> None:
    assert CANONICAL_TRAINING_SEED == 1301
    assert ANRA_V4_TRAINING.seed == CANONICAL_TRAINING_SEED
    assert ANRA_V4_TRAINING.optimizer == CANONICAL_FOUNDATION_OPTIMIZER == "adamw"
    assert ANRA_V4_TRAINING.max_grad_norm == 1.0


def test_complete_rng_snapshot_replays_all_foundation_generators() -> None:
    previous = capture_rng_states()
    try:
        report = seed_everything(CANONICAL_TRAINING_SEED)
        data_generator = make_data_generator(CANONICAL_TRAINING_SEED)
        snapshot = capture_rng_states(data_generator=data_generator)
        expected = (
            random.random(),
            float(np.random.random()),
            torch.rand(4),
            torch.randint(0, 10_000, (4,), generator=data_generator),
        )

        restore_rng_states(snapshot, data_generator=data_generator)
        replayed = (
            random.random(),
            float(np.random.random()),
            torch.rand(4),
            torch.randint(0, 10_000, (4,), generator=data_generator),
        )

        assert expected[0] == replayed[0]
        assert expected[1] == replayed[1]
        torch.testing.assert_close(expected[2], replayed[2])
        torch.testing.assert_close(expected[3], replayed[3])
        assert report.deterministic_algorithms
        assert not report.cudnn_benchmark
    finally:
        restore_rng_states(previous)


def test_counter_sampler_resume_is_exact_suffix() -> None:
    ranges = {"code": ((0, 100),), "prose": ((100, 300),)}
    full = list(
        ScheduledCurriculumSampler(
            ranges,
            curriculum="none",
            num_samples=500,
            seed=CANONICAL_TRAINING_SEED,
        )
    )
    resumed = ScheduledCurriculumSampler(
        ranges,
        curriculum="none",
        num_samples=500,
        seed=CANONICAL_TRAINING_SEED,
        start_position=173,
    )

    assert list(resumed) == full[173:]
    assert resumed.state_dict()["algorithm"] == SAMPLER_ALGORITHM


def test_different_seed_changes_run_without_being_ranked_better() -> None:
    first = list(
        ScheduledCurriculumSampler(
            {"foundation": ((0, 10_000),)},
            curriculum="none",
            num_samples=64,
            seed=1301,
        )
    )
    second = list(
        ScheduledCurriculumSampler(
            {"foundation": ((0, 10_000),)},
            curriculum="none",
            num_samples=64,
            seed=1302,
        )
    )

    assert first != second


def test_schema9_training_resume_restores_optimizer_scheduler_scaler_and_rng(
    tmp_path: Path,
) -> None:
    previous = capture_rng_states()
    try:
        seed_everything(CANONICAL_TRAINING_SEED)
        generator = make_data_generator(CANONICAL_TRAINING_SEED)
        model = CausalTransformerV2(
            vocab_size=64,
            n_embd=32,
            n_head=4,
            n_kv_head=2,
            n_layer=2,
            block_size=16,
            use_rim=False,
            use_dstp=False,
        )
        model.configure_subsystems(())
        model.training_recipe = {
            "model_profile": "tiny-test",
            "training_layout": "raw_causal_shards_v1",
            "curriculum": "none",
            "optimizer": "adamw",
            "seed": CANONICAL_TRAINING_SEED,
            "schedule": "cosine_with_warmup_v1",
            "gradient_clip_norm": 1.0,
            "verified_process_objective": VERIFIED_PROCESS_OBJECTIVE,
            "verified_process_multiplier": 1.25,
            "sampler_algorithm": SAMPLER_ALGORITHM,
            "determinism_mode": DETERMINISM_MODE,
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps=1, total_steps=10
        )
        mixed_precision = MixedPrecisionTrainer(device=torch.device("cpu"))

        x = torch.randint(0, 64, (1, 8))
        _, loss = model(x, x)
        assert loss is not None
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        checkpoint = tmp_path / "resume.pt"
        torch.save(
            {
                "checkpoint_schema_version": 9,
                "completed_optimizer_boundary": True,
                "accum_micro_steps": 0,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": mixed_precision.state_dict(),
                "model_config": model.model_config(),
                "training_recipe": dict(model.training_recipe),
                "rng_states": capture_rng_states(data_generator=generator),
            },
            checkpoint,
        )
        expected_global = torch.rand(4)
        expected_data = torch.randint(0, 10_000, (4,), generator=generator)
        with torch.no_grad():
            model.token_embedding_table.weight.zero_()

        state = load_checkpoint(
            model,
            optimizer,
            scheduler,
            mixed_precision,
            checkpoint,
            device=torch.device("cpu"),
            resume_training=True,
            data_generator=generator,
        )

        assert state["loaded"] is True
        assert state["rng_restore"]["data_generator"] is True
        torch.testing.assert_close(torch.rand(4), expected_global)
        torch.testing.assert_close(
            torch.randint(0, 10_000, (4,), generator=generator), expected_data
        )
        assert not torch.equal(
            model.token_embedding_table.weight,
            torch.zeros_like(model.token_embedding_table.weight),
        )
    finally:
        restore_rng_states(previous)
