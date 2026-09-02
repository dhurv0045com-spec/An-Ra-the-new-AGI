"""Unit tests for senora.trainer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from senora.experiment_design import P35_MODEL_SPEC
from senora.trainer import (
    LocalScientificComputeConstraintError,
    P35Trainer,
    P35TrainerConfig,
    WSDSchedule,
)
from v5_training.state import CURSOR_SCHEMA, CursorState, IdentityBindings


class TestSenoraTrainer(unittest.TestCase):
    def setUp(self) -> None:
        self.sha = "d" * 64
        self.config = P35TrainerConfig(
            model_spec=P35_MODEL_SPEC,
            token_budget=10_000,
            tokens_per_update=1_000,
            learning_rate=3e-4,
            weight_decay=0.1,
            gradient_clip_norm=1.0,
            query_swap_lambda=0.0,
            remote_authorized=False,
        )
        self.identities = IdentityBindings(
            schema="anra-v5-identity-bindings/v1",
            source_commit="e" * 40,
            model_spec_sha256=self.sha,
            tokenizer_sha256=self.sha,
            data_manifest_sha256=self.sha,
            pack_manifest_sha256=self.sha,
            run_spec_sha256=self.sha,
            optimizer_spec_sha256=self.sha,
            schedule_spec_sha256=self.sha,
            curriculum_spec_sha256=self.sha,
        )

    def test_wsd_schedule_curve(self) -> None:
        schedule = WSDSchedule.from_budget(
            token_budget=100_000,
            peak_lr=3e-4,
            warmup_fraction=0.10,  # 10,000 tokens
            decay_fraction=0.20,   # 20,000 tokens
            min_lr_ratio=0.10,     # 3e-5
        )
        self.assertEqual(schedule.warmup_tokens, 10_000)
        self.assertEqual(schedule.stable_tokens, 70_000)
        self.assertEqual(schedule.decay_tokens, 20_000)

        # 1. Warmup
        self.assertEqual(schedule.get_lr(0), 0.0)
        self.assertAlmostEqual(schedule.get_lr(5_000), 1.5e-4)

        # 2. Stable
        self.assertEqual(schedule.get_lr(10_000), 3e-4)
        self.assertEqual(schedule.get_lr(50_000), 3e-4)
        self.assertAlmostEqual(schedule.get_lr(80_000), 3e-4)

        # 3. Decay & Final
        self.assertLess(schedule.get_lr(90_000), 3e-4)
        self.assertAlmostEqual(schedule.get_lr(100_000), 3e-5)

    def test_hard_compute_constraint_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = P35Trainer(
                self.config,
                identity_bindings=self.identities,
                checkpoint_directory=Path(temp_dir),
            )
            with self.assertRaises(LocalScientificComputeConstraintError):
                trainer.verify_remote_execution_guard()

    def test_step_advancement_and_abort_guards(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = P35Trainer(
                self.config,
                identity_bindings=self.identities,
                checkpoint_directory=Path(temp_dir),
            )
            cursor = CursorState(
                schema=CURSOR_SCHEMA,
                pack_manifest_sha256=self.sha,
                shard_ordinal=0,
                sequence_ordinal=0,
                token_offset=0,
            )
            state = trainer.initialize_training_state(
                initial_cursor=cursor,
                rng_state_sha256=self.sha,
            )
            self.assertEqual(state.global_update, 0)
            self.assertEqual(state.cumulative_tokens, 0)

            # Advance one valid step
            new_cursor = CursorState(
                schema=CURSOR_SCHEMA,
                pack_manifest_sha256=self.sha,
                shard_ordinal=0,
                sequence_ordinal=1,
                token_offset=1000,
            )
            advanced_state = trainer.advance_step(
                state,
                tokens_by_source={"natural": 800, "cognition": 200},
                new_cursor=new_cursor,
                new_rng_state_sha256="f" * 64,
                loss_value=2.45,
                gradient_norm=0.85,
            )
            self.assertEqual(advanced_state.global_update, 1)
            self.assertEqual(advanced_state.cumulative_tokens, 1000)

            # Abort on NaN loss
            with self.assertRaises(ValueError):
                trainer.advance_step(
                    advanced_state,
                    tokens_by_source={"natural": 800, "cognition": 200},
                    new_cursor=new_cursor,
                    new_rng_state_sha256="f" * 64,
                    loss_value=float("nan"),
                    gradient_norm=0.85,
                )


if __name__ == "__main__":
    unittest.main()