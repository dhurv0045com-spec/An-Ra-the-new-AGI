"""Unit tests for senora.objectives."""

from __future__ import annotations

import math
import unittest

from senora.objectives import NonFiniteLossError

try:
    import torch
    from senora.objectives import (
        causal_cross_entropy,
        compute_composite_training_loss,
        query_swap_contrastive_loss,
    )
except ImportError:
    torch = None
    causal_cross_entropy = None
    compute_composite_training_loss = None
    query_swap_contrastive_loss = None


class TestSenoraObjectives(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch required for objectives tests")
    def test_causal_cross_entropy_basic(self) -> None:
        batch_size = 2
        seq_len = 4
        vocab_size = 8

        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        targets = torch.tensor([[1, 2, 3, -100], [0, 5, -100, -100]], dtype=torch.long)

        loss, valid_tokens = causal_cross_entropy(logits, targets, ignore_index=-100)
        self.assertEqual(valid_tokens, 5)
        self.assertTrue(torch.isfinite(loss))

        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    @unittest.skipIf(torch is None, "PyTorch required for non-finite loss abort test")
    def test_causal_cross_entropy_abort_on_nan(self) -> None:
        logits = torch.tensor([[[float("nan"), 1.0], [0.0, 1.0]]])
        targets = torch.tensor([[0, 1]], dtype=torch.long)
        with self.assertRaises(NonFiniteLossError):
            causal_cross_entropy(logits, targets)

    @unittest.skipIf(torch is None, "PyTorch required for query swap test")
    def test_query_swap_contrastive_loss_properties(self) -> None:
        # Case A: Model strongly discriminates in favor of query-dependent targets
        # factual target >> distractor, counterfactual target >> distractor
        f_target = torch.tensor([5.0])
        cf_distractor = torch.tensor([-5.0])
        cf_target = torch.tensor([5.0])
        f_distractor = torch.tensor([-5.0])

        loss_good = query_swap_contrastive_loss(f_target, cf_distractor, cf_target, f_distractor)
        # Should be very close to 0
        self.assertLess(loss_good.item(), 0.01)

        # Case B: Model fails to discriminate (blind candidate prior: favors target 1 on both queries)
        # under q1: target 1 favored (5.0 vs -5.0) -> good
        # under q2: target 1 still favored (-5.0 vs 5.0) -> bad
        f_target_bad = torch.tensor([5.0])
        cf_distractor_bad = torch.tensor([-5.0])
        cf_target_bad = torch.tensor([-5.0])
        f_distractor_bad = torch.tensor([5.0])

        loss_bad = query_swap_contrastive_loss(f_target_bad, cf_distractor_bad, cf_target_bad, f_distractor_bad)
        # Should heavily penalize the error
        self.assertGreater(loss_bad.item(), 4.5)

    @unittest.skipIf(torch is None, "PyTorch required for composite loss test")
    def test_compute_composite_training_loss(self) -> None:
        logits = torch.randn(2, 4, 8)
        targets = torch.tensor([[1, 2, 3, 4], [0, 1, 2, 3]], dtype=torch.long)

        # 1. Pure CE
        loss_ce_only, receipt_ce = compute_composite_training_loss(
            logits, targets, query_swap_lambda=0.0
        )
        self.assertEqual(receipt_ce.query_swap_loss, 0.0)
        self.assertEqual(receipt_ce.query_swap_pairs_count, 0)
        self.assertEqual(receipt_ce.total_loss, receipt_ce.ce_loss)

        # 2. CE + Query-Swap
        payload = {
            "factual_target_logprob": torch.tensor([1.0]),
            "counterfactual_distractor_logprob": torch.tensor([0.0]),
            "counterfactual_target_logprob": torch.tensor([1.0]),
            "factual_distractor_logprob": torch.tensor([0.0]),
        }
        loss_composite, receipt_composite = compute_composite_training_loss(
            logits, targets, query_swap_lambda=0.10, query_swap_payload=payload
        )
        self.assertGreater(receipt_composite.query_swap_loss, 0.0)
        self.assertEqual(receipt_composite.query_swap_pairs_count, 1)
        expected_total = receipt_composite.ce_loss + 0.10 * receipt_composite.query_swap_loss
        self.assertAlmostEqual(receipt_composite.total_loss, expected_total, places=5)


if __name__ == "__main__":
    unittest.main()