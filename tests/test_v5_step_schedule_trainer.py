from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from v5_training.checkpoint import CheckpointStore, _canonical_json
from v5_training.runner import RunController
from v5_training.schedule import (
    FINAL_LEARNING_RATE,
    PEAK_LEARNING_RATE,
    STABLE_END_TOKENS,
    TOKEN_BUDGET,
    WARMUP_END_TOKENS,
    lr_at,
    schedule_receipt,
)
from v5_training.state import CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState, IdentityBindings, TrainingState
from v5_training.step import certify_update
from v5_training.trainer import BackendReport, train


def _hash(character: str) -> str:
    return character * 64


def _initial(budget: int = 12, per_update: int = 4) -> TrainingState:
    identities = IdentityBindings(
        IDENTITY_SCHEMA,
        "a" * 40,
        _hash("1"), _hash("2"), _hash("3"), _hash("4"),
        _hash("5"), _hash("6"), _hash("7"), _hash("8"),
    )
    return TrainingState.initial(
        lineage_id="trainer-test",
        token_budget=budget,
        tokens_per_update=per_update,
        cursor=CursorState(CURSOR_SCHEMA, identities.pack_manifest_sha256, 0, 0, 0),
        rng_state_sha256=_hash("9"),
        curriculum_phase="uniform",
        identities=identities,
    )


def _payloads(state: TrainingState) -> dict[str, bytes]:
    generation = state.generation
    return {
        "model.bin": f"model-generation-{generation}".encode(),
        "optimizer.bin": f"adam-generation-{generation}-step-{state.optimizer_step_max}".encode(),
        "scheduler.json": _canonical_json({"schedule_tokens": state.schedule_tokens}),
        "rng.bin": state.rng_state_sha256.encode(),
        "cursor.json": _canonical_json(
            {
                "schema": state.cursor.schema,
                "pack_manifest_sha256": state.cursor.pack_manifest_sha256,
                "shard_ordinal": state.cursor.shard_ordinal,
                "sequence_ordinal": state.cursor.sequence_ordinal,
                "token_offset": state.cursor.token_offset,
            }
        ),
        "ledger.json": _canonical_json(dict(state.tokens_by_source)),
        "training_state.json": _canonical_json(state.canonical()),
    }


class ScheduleTests(unittest.TestCase):
    def test_warmup_stable_decay_boundaries(self) -> None:
        self.assertEqual(lr_at(cumulative_tokens=0), 0.0)
        self.assertAlmostEqual(lr_at(cumulative_tokens=25_000_000), PEAK_LEARNING_RATE / 2)
        self.assertEqual(lr_at(cumulative_tokens=WARMUP_END_TOKENS), PEAK_LEARNING_RATE)
        self.assertEqual(lr_at(cumulative_tokens=STABLE_END_TOKENS), PEAK_LEARNING_RATE)
        midpoint = (STABLE_END_TOKENS + TOKEN_BUDGET) // 2
        self.assertAlmostEqual(
            lr_at(cumulative_tokens=midpoint), (PEAK_LEARNING_RATE + FINAL_LEARNING_RATE) / 2
        )
        self.assertAlmostEqual(lr_at(cumulative_tokens=TOKEN_BUDGET), FINAL_LEARNING_RATE)

    def test_schedule_is_pure_and_fail_closed(self) -> None:
        self.assertEqual(lr_at(cumulative_tokens=1000), lr_at(cumulative_tokens=1000))
        receipt = schedule_receipt()
        self.assertFalse(receipt["rewarm_on_resume_or_pack_change"])
        with self.assertRaises(ValueError):
            lr_at(cumulative_tokens=-1)
        with self.assertRaises(ValueError):
            lr_at(cumulative_tokens=TOKEN_BUDGET + 1)
        with self.assertRaises(ValueError):
            lr_at(cumulative_tokens=True)


class StepTests(unittest.TestCase):
    def _advance(self, state: TrainingState):
        cursor = CursorState(
            CURSOR_SCHEMA, state.identities.pack_manifest_sha256,
            state.cursor.shard_ordinal, state.cursor.sequence_ordinal + 1,
            state.cursor.token_offset + 1,
        )
        tokens = {"natural": 3, "verified_cognition": 1}
        after = state.advance(
            tokens_by_source=tokens, cursor=cursor,
            rng_state_sha256=f"{state.global_update + 1:064x}",
            parent_checkpoint_sha256=None,
        )
        return after, tokens

    def test_happy_path_and_determinism(self) -> None:
        before = _initial()
        after, tokens = self._advance(before)
        first = certify_update(
            before=before, after=after, tokens_by_source=tokens,
            loss_finite=True, grad_finite=True,
            grad_norm_post_clip=0.7, tied_preserved=True,
        )
        self.assertEqual(first["update"], 1)
        self.assertEqual(
            first,
            certify_update(
                before=before, after=after, tokens_by_source=tokens,
                loss_finite=True, grad_finite=True,
                grad_norm_post_clip=0.7, tied_preserved=True,
            ),
        )

    def test_aborts(self) -> None:
        before = _initial()
        after, tokens = self._advance(before)
        base = dict(loss_finite=True, grad_finite=True, grad_norm_post_clip=0.7, tied_preserved=True)
        for key, value in [
            ("loss_finite", False), ("grad_finite", False), ("tied_preserved", False),
            ("grad_norm_post_clip", 1.5), ("grad_norm_post_clip", float("nan")),
        ]:
            case = dict(base, **{key: value})
            with self.assertRaises(ValueError, msg=key):
                certify_update(before=before, after=after, tokens_by_source=tokens, **case)
        with self.assertRaises(ValueError):
            certify_update(
                before=before, after=after,
                tokens_by_source={"natural": 4, "verified_cognition": 0}, **base
            )
        with self.assertRaises(ValueError):
            certify_update(
                before=before, after=before,
                tokens_by_source={"natural": 4}, **base
            )


class TrainerTests(unittest.TestCase):
    def _backend(self, finite: bool = True):
        def step(state: TrainingState) -> BackendReport:
            cursor = CursorState(
                CURSOR_SCHEMA, state.identities.pack_manifest_sha256,
                state.cursor.shard_ordinal, state.cursor.sequence_ordinal + 1,
                state.cursor.token_offset + 1,
            )
            return BackendReport(
                tokens_by_source={"natural": 3, "verified_cognition": 1},
                cursor=cursor,
                rng_state_sha256=f"{state.global_update + 1:064x}",
                loss_finite=finite,
                grad_finite=True,
                grad_norm_post_clip=0.5,
                tied_preserved=True,
            )
        return step

    def test_train_advances_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(Path(directory), "trainer-test")
            controller = RunController(target_update=3)
            controller.start()
            final = train(
                state=_initial(), controller=controller, store=store,
                payload_builder=_payloads, backend_step=self._backend(),
                updates=3, checkpoint_every=2,
            )
            self.assertEqual((final.global_update, final.cumulative_tokens), (3, 12))
            self.assertTrue(final.complete)
            self.assertIsNotNone(store.latest_sha256())
            restored, _ = store.restore(store.latest_sha256())
            self.assertEqual(restored, final)

    def test_train_aborts_and_marks_failed_on_nonfinite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(Path(directory), "trainer-test")
            controller = RunController(target_update=3)
            controller.start()
            with self.assertRaises(ValueError):
                train(
                    state=_initial(), controller=controller, store=store,
                    payload_builder=_payloads, backend_step=self._backend(finite=False),
                    updates=3, checkpoint_every=2,
                )
            from v5_training.runner import RunStatus
            self.assertEqual(controller.state.status, RunStatus.FAILED)


if __name__ == "__main__":
    unittest.main()
