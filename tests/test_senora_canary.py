"""Unit tests for senora.canary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from senora.canary import execute_preflight_canary

try:
    import torch
except ImportError:
    torch = None


class TestSenoraCanary(unittest.TestCase):
    def test_unauthorized_execution_fails_closed(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            execute_preflight_canary(device="cpu", remote_authorized=False)
        self.assertIn("requires explicit target authorization", str(ctx.exception))

    @unittest.skipIf(torch is None, "PyTorch required for canary test")
    def test_preflight_canary_mini_model_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_file = Path(temp_dir) / "test_canary.json"
            receipt = execute_preflight_canary(
                device="cpu",
                remote_authorized=True,
                output_receipt=out_file,
                use_mini_model_for_test=True,
            )

            self.assertEqual(receipt.status, "PASS_CANARY_CERTIFIED")
            self.assertTrue(receipt.single_step_finite_loss)
            self.assertTrue(receipt.gradients_finite)
            self.assertTrue(receipt.parameter_sha_changed)
            self.assertGreater(receipt.parameters_moved_count, 0)
            self.assertTrue(receipt.adam_moments_active)
            self.assertTrue(receipt.tied_embeddings_preserved)
            self.assertTrue(receipt.twenty_five_step_stability)
            self.assertTrue(receipt.checkpoint_restore_reproduced)
            self.assertTrue(out_file.exists())


if __name__ == "__main__":
    unittest.main()