"""Unit tests for senora.dry_run."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from senora.dry_run import execute_dry_run


class TestSenoraDryRun(unittest.TestCase):
    def test_execute_dry_run_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            receipt_path = Path(temp_dir) / "test_receipt.json"
            receipt = execute_dry_run(output_receipt=receipt_path)

            self.assertEqual(receipt["status"], "PASS_PLUMBING_CERTIFIED")
            self.assertEqual(receipt["experiment_id"], "P35-CMS-1")
            self.assertTrue(receipt["checkpoint_restored_clean"])
            self.assertTrue(receipt["scorer_firewall_gate_enforced"])
            self.assertTrue(receipt_path.exists())


if __name__ == "__main__":
    unittest.main()