from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - environment-dependent
    torch = None

from v5_objectives.causal_lm import causal_lm_loss
from v5_objectives.query_swap import query_swap_loss


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ObjectiveTests(unittest.TestCase):
    def test_causal_loss_masks_bos_pad_and_transitions(self) -> None:
        torch.manual_seed(0)
        vocab, batch, length = 32, 2, 10
        logits = torch.randn(batch, length, vocab)
        tokens = torch.randint(4, vocab, (batch, length))
        tokens[:, 0] = 2
        segments = torch.zeros(batch, length, dtype=torch.int64)
        segments[1, 5:] = 1
        loss, count = causal_lm_loss(logits, tokens, segments)
        self.assertGreater(count, 0)
        self.assertTrue(torch.isfinite(loss).item())
        # Single segment, no BOS/PAD targets: every shifted position counts.
        clean_tokens = torch.randint(4, vocab, (1, 8))
        clean_logits = torch.randn(1, 8, vocab)
        clean_segments = torch.zeros(1, 8, dtype=torch.int64)
        _, clean_count = causal_lm_loss(clean_logits, clean_tokens, clean_segments)
        self.assertEqual(clean_count, 7)

    def test_causal_loss_requires_supervised_targets(self) -> None:
        logits = torch.zeros(1, 4, 8)
        tokens = torch.full((1, 4), 2)
        segments = torch.zeros(1, 4, dtype=torch.int64)
        with self.assertRaises(ValueError):
            causal_lm_loss(logits, tokens, segments)

    def test_query_swap_refuses_when_disabled(self) -> None:
        gold = torch.zeros(4)
        negatives = torch.zeros(4, 3)
        with self.assertRaises(ValueError):
            query_swap_loss(gold, negatives, enabled=False)
        with self.assertRaises(ValueError):
            query_swap_loss(gold, negatives, enabled=True, margin=0.1)
        with self.assertRaises(ValueError):
            query_swap_loss(gold, torch.zeros(4, 2), enabled=True)

    def test_query_swap_is_zero_when_gold_wins(self) -> None:
        gold = torch.tensor([2.0, 2.0])
        negatives = torch.tensor([[0.5, 0.1, -1.0], [1.9, 0.0, 0.2]])
        loss = query_swap_loss(gold, negatives, enabled=True)
        self.assertAlmostEqual(float(loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
