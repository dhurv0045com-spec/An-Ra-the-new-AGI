from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from v5_training.checkpoint import CheckpointStore, _canonical_json
from v5_training.state import CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState, IdentityBindings, TrainingState, next_update_tokens
from v5_training.transaction_canary import (
    _advance,
    _initial,
    _payloads,
    implementation_sha256,
    run_canary,
)


class V5TrainingTests(unittest.TestCase):
    def test_nondivisible_token_budget_uses_final_partial_update(self) -> None:
        self.assertEqual(
            [next_update_tokens(token_budget=10, cumulative_tokens=n, tokens_per_update=4)
             for n in (0, 4, 8, 10)],
            [4, 4, 2, 0],
        )

    def test_state_starts_at_zero_and_advances_exactly_once(self) -> None:
        state = _initial()
        self.assertEqual((state.global_update, state.cumulative_tokens, state.schedule_tokens), (0, 0, 0))
        updated = _advance(state, None)
        self.assertEqual(updated.global_update, 1)
        self.assertEqual(updated.cumulative_tokens, 4)
        self.assertEqual(sum(updated.tokens_by_source.values()), 4)
        with self.assertRaises(ValueError):
            updated.advance(
                tokens_by_source={"natural": 3},
                cursor=updated.cursor,
                rng_state_sha256="a" * 64,
                parent_checkpoint_sha256=None,
            )

    def test_unknown_schemas_and_ledger_drift_fail_closed(self) -> None:
        state = _initial()
        with self.assertRaises(ValueError):
            replace(state, schema="future/v9").assert_valid()
        with self.assertRaises(ValueError):
            replace(state, cumulative_tokens=1).assert_valid()
        with self.assertRaises(ValueError):
            replace(state, schedule_tokens=1).assert_valid()

    def test_checkpoint_inventory_and_stale_writer_fail_closed(self) -> None:
        state = _advance(_initial(), None)
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(Path(directory), "canary")
            payloads = _payloads(state)
            missing = dict(payloads)
            missing.pop("rng.bin")
            with self.assertRaises(ValueError):
                store.publish(state=state, payloads=missing, expected_parent_sha256=None)
            identity = store.publish(state=state, payloads=payloads, expected_parent_sha256=None)
            with self.assertRaises(ValueError):
                store.publish(state=state, payloads=payloads, expected_parent_sha256=None)
            restored, restored_payloads = store.restore(identity)
            self.assertEqual(restored, state)
            self.assertEqual(restored_payloads, payloads)
            (store.objects / identity / "untracked.bin").write_bytes(b"not-in-manifest")
            with self.assertRaises(ValueError):
                store.restore(identity)

    def test_training_state_payload_is_bound(self) -> None:
        state = _advance(_initial(), None)
        payloads = _payloads(state)
        payloads["training_state.json"] = _canonical_json(_initial().canonical())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                CheckpointStore(Path(directory), "canary").publish(
                    state=state, payloads=payloads, expected_parent_sha256=None
                )

    def test_cursor_and_ledger_components_are_bound_to_state(self) -> None:
        state = _advance(_initial(), None)
        payloads = _payloads(state)
        payloads["cursor.json"] = _canonical_json(
            asdict(replace(state.cursor, token_offset=state.cursor.token_offset + 1))
        )
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(Path(directory), "canary")
            with self.assertRaises(ValueError):
                store.publish(state=state, payloads=payloads, expected_parent_sha256=None)

    def test_transaction_canary_and_committed_receipt_pass(self) -> None:
        live = run_canary()
        self.assertEqual(live["status"], "PASS")
        root = Path(__file__).parents[1]
        committed = json.loads(
            (root / "artifacts/v5/training_transaction_canary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(live, committed)
        self.assertEqual(committed["implementation_sha256"], implementation_sha256())


if __name__ == "__main__":
    unittest.main()
