from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


class V5TorchCanaryReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.receipt = json.loads(
            (cls.root / "artifacts/v5/local_p35_checkpoint_canary.json").read_text(encoding="utf-8")
        )

    def test_committed_receipt_is_a_p35_resume_pass(self) -> None:
        receipt = self.receipt
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["schema"], "esoes-v5-p35-checkpoint-canary/v1")
        self.assertEqual(receipt["config"]["arm"], "middle")
        self.assertEqual(receipt["global_update"], receipt["optimizer_step_max"])
        self.assertEqual(receipt["cumulative_tokens"], 16)
        self.assertEqual(receipt["resume"]["parameter_max_abs_error"], 0.0)
        self.assertEqual(receipt["resume"]["optimizer_state_max_abs_error"], 0.0)
        self.assertTrue(all(receipt["checks"].values()))
        self.assertNotEqual(receipt["first_checkpoint_sha256"], receipt["final_checkpoint_sha256"])

    def test_receipt_binds_current_canary_and_constructor(self) -> None:
        receipt = self.receipt
        source = hashlib.sha256((self.root / "v5_training/torch_canary.py").read_bytes()).hexdigest()
        constructor = hashlib.sha256((self.root / "e2_architecture/block_benchmark.py").read_bytes()).hexdigest()
        self.assertEqual(receipt["implementation_sha256"], source)
        self.assertEqual(receipt["model_constructor_sha256"], constructor)
        self.assertTrue(receipt["resume"]["clean_copy_restore"])


if __name__ == "__main__":
    unittest.main()
