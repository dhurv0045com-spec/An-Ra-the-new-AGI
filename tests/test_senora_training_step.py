"""Unit tests for senora.training_step."""

from __future__ import annotations

import math
import unittest

from senora.data_pipeline import CursorState
from senora.trainer import WSDSchedule
from v5_training.state import CURSOR_SCHEMA, IdentityBindings, TrainingState

try:
    import torch
    from senora.model import P35_MODEL_SPEC, P35Model
    from senora.optimizer import build_p35_optimizer
    from senora.training_step import (
        RealBatch,
        SilentParameterFailureError,
        execute_real_training_step,
    )
except ImportError:
    torch = None
    P35Model = None
    build_p35_optimizer = None
    RealBatch = None
    execute_real_training_step = None
    SilentParameterFailureError = None


class TestSenoraTrainingStep(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch required for real training step test")
    def test_execute_real_training_step(self) -> None:
        # Construct tiny 2-layer model for fast unit testing
        mini_spec = P35_MODEL_SPEC.__class__(
            schema="anra-v5-mini-spec/v1",
            family="dense-decoder-transformer",
            vocabulary_size=256,
            width=64,
            layers=2,
            query_heads=4,
            kv_heads=2,
            head_dimension=16,
            ffn_width=128,
            context_length=64,
            rope_base=10000.0,
            norm_epsilon=1e-5,
            tied_embeddings=True,
            qk_norm=True,
            qk_norm_affine=True,
            linear_bias=False,
            dropout=0.0,
        )
        model = P35Model(mini_spec)
        optimizer, manifest = build_p35_optimizer(model, learning_rate=1e-3)
        scheduler = WSDSchedule.from_budget(token_budget=1000, peak_lr=1e-3)

        dummy_sha = "0" * 64
        cursor = CursorState(
            schema=CURSOR_SCHEMA,
            pack_manifest_sha256=dummy_sha,
            shard_ordinal=0,
            sequence_ordinal=0,
            token_offset=0,
        )
        identities = IdentityBindings(
            schema="anra-v5-identity-bindings/v1",
            source_commit="a" * 40,
            model_spec_sha256=dummy_sha,
            tokenizer_sha256=dummy_sha,
            data_manifest_sha256=dummy_sha,
            pack_manifest_sha256=dummy_sha,
            run_spec_sha256=dummy_sha,
            optimizer_spec_sha256=dummy_sha,
            schedule_spec_sha256=dummy_sha,
            curriculum_spec_sha256=dummy_sha,
        )
        init_state = TrainingState.initial(
            lineage_id="lineage-test-step",
            token_budget=1000,
            tokens_per_update=16,
            cursor=cursor,
            rng_state_sha256=dummy_sha,
            curriculum_phase="phase-1-test",
            identities=identities,
        )

        batch_size = 2
        seq_len = 8
        batch_tokens = batch_size * seq_len
        input_ids = torch.randint(0, mini_spec.vocabulary_size, (batch_size, seq_len))
        targets = torch.randint(0, mini_spec.vocabulary_size, (batch_size, seq_len))

        next_cursor = CursorState(
            schema=CURSOR_SCHEMA,
            pack_manifest_sha256=dummy_sha,
            shard_ordinal=0,
            sequence_ordinal=1,
            token_offset=batch_tokens,
        )
        batch = RealBatch(
            input_ids=input_ids,
            targets=targets,
            tokens_by_source={"natural": batch_tokens},
            batch_token_count=batch_tokens,
            new_cursor=next_cursor,
        )

        next_state, receipt = execute_real_training_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            batch=batch,
            state=init_state,
        )

        # Invariant checks
        self.assertEqual(next_state.global_update, 1)
        self.assertEqual(next_state.cumulative_tokens, batch_tokens)
        self.assertEqual(receipt.global_update, 1)
        self.assertNotEqual(receipt.initial_parameter_sha256, receipt.updated_parameter_sha256)
        self.assertGreater(receipt.parameters_moved_count, 0)
        self.assertTrue(receipt.adam_moments_active)
        self.assertTrue(math.isfinite(receipt.loss.total_loss))
        self.assertTrue(math.isfinite(receipt.gradient_norm))

    @unittest.skipIf(torch is None, "PyTorch required for silent failure test")
    def test_silent_parameter_failure_raises_error(self) -> None:
        mini_spec = P35_MODEL_SPEC.__class__(
            schema="anra-v5-mini-spec/v1",
            family="dense-decoder-transformer",
            vocabulary_size=256,
            width=64,
            layers=2,
            query_heads=4,
            kv_heads=2,
            head_dimension=16,
            ffn_width=128,
            context_length=64,
            rope_base=10000.0,
            norm_epsilon=1e-5,
            tied_embeddings=True,
            qk_norm=True,
            qk_norm_affine=True,
            linear_bias=False,
            dropout=0.0,
        )
        model = P35Model(mini_spec)
        optimizer, _ = build_p35_optimizer(model, learning_rate=0.0)  # zero learning rate prevents parameter movement
        scheduler = WSDSchedule.from_budget(token_budget=1000, peak_lr=0.0, min_lr_ratio=0.0)

        dummy_sha = "0" * 64
        cursor = CursorState(schema=CURSOR_SCHEMA, pack_manifest_sha256=dummy_sha, shard_ordinal=0, sequence_ordinal=0, token_offset=0)
        identities = IdentityBindings(
            schema="anra-v5-identity-bindings/v1",
            source_commit="a" * 40,
            model_spec_sha256=dummy_sha,
            tokenizer_sha256=dummy_sha,
            data_manifest_sha256=dummy_sha,
            pack_manifest_sha256=dummy_sha,
            run_spec_sha256=dummy_sha,
            optimizer_spec_sha256=dummy_sha,
            schedule_spec_sha256=dummy_sha,
            curriculum_spec_sha256=dummy_sha,
        )
        init_state = TrainingState.initial(
            lineage_id="lineage-test-zero-lr",
            token_budget=1000,
            tokens_per_update=16,
            cursor=cursor,
            rng_state_sha256=dummy_sha,
            curriculum_phase="phase-1-test",
            identities=identities,
        )

        batch = RealBatch(
            input_ids=torch.randint(0, mini_spec.vocabulary_size, (2, 8)),
            targets=torch.randint(0, mini_spec.vocabulary_size, (2, 8)),
            tokens_by_source={"natural": 16},
            batch_token_count=16,
            new_cursor=cursor,
        )

        with self.assertRaises(SilentParameterFailureError):
            execute_real_training_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                batch=batch,
                state=init_state,
            )


if __name__ == "__main__":
    unittest.main()