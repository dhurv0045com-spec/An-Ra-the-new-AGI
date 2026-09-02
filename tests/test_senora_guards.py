"""Unit tests for senora.guards."""

from __future__ import annotations

import unittest

from senora.guards import ScientificExecutionGuard, ScientificIntegrityViolationError


class TestSenoraGuards(unittest.TestCase):
    def test_current_git_head_is_sha40(self) -> None:
        head = ScientificExecutionGuard.get_current_git_head()
        self.assertEqual(len(head), 40)

    def test_checkpoint_payload_rejection(self) -> None:
        # Mock byte string should be rejected
        fake_payloads = {
            "model.bin": b"remote_model_state",
            "optimizer.bin": b"remote_optimizer_state",
            "rng.bin": b"remote_rng_state",
            "training_state.json": b"{}",
        }
        with self.assertRaises(ScientificIntegrityViolationError):
            ScientificExecutionGuard.assert_real_checkpoint_payloads(fake_payloads)

        # Real size payloads should pass
        real_payloads = {
            "model.bin": b"x" * 1024,
            "optimizer.bin": b"y" * 1024,
            "rng.bin": b"z" * 1024,
            "training_state.json": b"w" * 1024,
        }
        ScientificExecutionGuard.assert_real_checkpoint_payloads(real_payloads)

    def test_gold_answer_leakage_detection(self) -> None:
        prompt = "Question: What is X? The answer: 42"
        with self.assertRaises(ScientificIntegrityViolationError):
            ScientificExecutionGuard.assert_no_gold_in_policy_input(prompt, "42")

        clean_prompt = "Question: What is X? Answer:"
        ScientificExecutionGuard.assert_no_gold_in_policy_input(clean_prompt, "42")


if __name__ == "__main__":
    unittest.main()