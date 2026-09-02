"""Tests for Senora Real Checkpoint Serialization and Restoration."""

from __future__ import annotations

import unittest
from pathlib import Path

from senora.checkpoint import restore_real_checkpoint_payloads, serialize_real_checkpoint_payloads
from v5_training.state import CursorState, IdentityBindings, TrainingState

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = object


class TinyModel(nn.Module if torch is not None else object):  # type: ignore
    def __init__(self) -> None:
        if torch is not None:
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

    def forward(self, x: Any) -> Any:
        return self.linear(x)


class TestSenoraCheckpoint(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch required for checkpoint serialization test")
    def test_real_serialization_and_exact_parameter_restore(self) -> None:
        model1 = TinyModel()
        model2 = TinyModel()

        # Verify models start with different random weights
        params1 = list(model1.parameters())
        params2 = list(model2.parameters())
        self.assertFalse(torch.equal(params1[0], params2[0]))

        opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

        cursor = CursorState(
            schema="anra-v5-pack-cursor/v1",
            pack_manifest_sha256="f" * 64,
            shard_ordinal=0,
            sequence_ordinal=0,
            token_offset=0,
        )
        identities = IdentityBindings(
            schema="anra-v5-identity-bindings/v1",
            source_commit="a" * 40,
            model_spec_sha256="b" * 64,
            tokenizer_sha256="c" * 64,
            data_manifest_sha256="e" * 64,
            pack_manifest_sha256="f" * 64,
            run_spec_sha256="1" * 64,
            optimizer_spec_sha256="2" * 64,
            schedule_spec_sha256="3" * 64,
            curriculum_spec_sha256="4" * 64,
        )
        state = TrainingState.initial(
            lineage_id="lineage-001",
            token_budget=50_000_000,
            tokens_per_update=131_072,
            cursor=cursor,
            rng_state_sha256="0" * 64,
            curriculum_phase="main",
            identities=identities,
        )

        payloads = serialize_real_checkpoint_payloads(model1, opt1, state, device="cpu")
        self.assertIn("model.bin", payloads)
        self.assertIn("optimizer.bin", payloads)
        self.assertIn("rng.bin", payloads)
        self.assertIn("scheduler.json", payloads)
        self.assertIn("cursor.json", payloads)
        self.assertIn("ledger.json", payloads)
        self.assertIn("training_state.json", payloads)

        restore_real_checkpoint_payloads(model2, opt2, payloads, device="cpu")

        # Verify bitwise parameter equality after restore
        params1_restored = list(model1.parameters())
        params2_restored = list(model2.parameters())
        for p1, p2 in zip(params1_restored, params2_restored):
            self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()