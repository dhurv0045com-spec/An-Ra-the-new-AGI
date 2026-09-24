from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:  # pragma: no cover - environment-dependent
    torch = None

from v5_objectives.causal_lm import causal_lm_loss, causal_lm_loss_from_hidden
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
        mean_with_numerator, count_with_numerator, numerator = causal_lm_loss(
            logits, tokens, segments, return_numerator=True,
        )
        self.assertEqual(count_with_numerator, count)
        torch.testing.assert_close(mean_with_numerator, loss)
        torch.testing.assert_close(numerator, loss * count)
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

    def test_causal_loss_can_return_graph_connected_zero_for_empty_replica(self) -> None:
        logits = torch.full((2, 4, 8), 1e38, requires_grad=True)
        tokens = torch.zeros((2, 4), dtype=torch.long)
        segments = torch.full((2, 4), -1, dtype=torch.int64)
        eligible = torch.zeros((2, 4), dtype=torch.bool)

        loss, count, numerator = causal_lm_loss(
            logits,
            tokens,
            segments,
            eligible=eligible,
            return_numerator=True,
            allow_empty=True,
        )
        self.assertEqual(count, 0)
        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual(float(numerator.item()), 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

        with self.assertRaisesRegex(ValueError, "no supervised targets"):
            causal_lm_loss(logits.detach(), tokens, segments, eligible=eligible)

    def test_chunked_hidden_loss_matches_full_logits_and_gradients(self) -> None:
        torch.manual_seed(91)
        tokens = torch.tensor([
            [2, 4, 5, 6, 7, 8, 0, 0],
            [2, 9, 10, 11, 12, 13, 14, 15],
        ])
        segments = torch.tensor([
            [0, 0, 0, 1, 1, 1, -1, -1],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ])
        eligible = torch.ones_like(tokens, dtype=torch.bool)
        eligible[0, 2] = False
        eligible[1, 6] = False
        hidden_reference = torch.randn(2, 8, 6, requires_grad=True)
        weight_reference = torch.randn(19, 6, requires_grad=True)
        reference, reference_count, reference_numerator = causal_lm_loss(
            torch.nn.functional.linear(hidden_reference, weight_reference),
            tokens, segments, eligible=eligible, return_numerator=True,
        )
        reference_gradients = torch.autograd.grad(
            reference, (hidden_reference, weight_reference),
        )

        hidden_chunked = hidden_reference.detach().clone().requires_grad_(True)
        weight_chunked = weight_reference.detach().clone().requires_grad_(True)
        chunked, chunked_count, chunked_numerator = causal_lm_loss_from_hidden(
            hidden_chunked, weight_chunked, tokens, segments, eligible=eligible,
            return_numerator=True, chunk_tokens=3,
        )
        chunked_gradients = torch.autograd.grad(
            chunked, (hidden_chunked, weight_chunked),
        )

        self.assertEqual(chunked_count, reference_count)
        torch.testing.assert_close(chunked, reference, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            chunked_numerator, reference_numerator, rtol=1e-6, atol=1e-6,
        )
        for actual, expected in zip(chunked_gradients, reference_gradients):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_chunked_hidden_loss_bounds_each_vocabulary_projection(self) -> None:
        hidden = torch.randn(2, 9, 6, requires_grad=True)
        weight = torch.randn(19, 6, requires_grad=True)
        tokens = torch.randint(4, 19, (2, 9))
        tokens[:, 0] = 2
        segments = torch.zeros_like(tokens)
        projection_lengths: list[int] = []
        linear = torch.nn.functional.linear

        def record_projection(inputs, output_weight, bias=None):
            if output_weight.shape[0] == weight.shape[0]:
                projection_lengths.append(int(inputs.shape[1]))
            return linear(inputs, output_weight, bias)

        with patch.object(torch.nn.functional, "linear", side_effect=record_projection):
            loss, _ = causal_lm_loss_from_hidden(
                hidden, weight, tokens, segments, chunk_tokens=3,
            )
            loss.backward()

        self.assertTrue(projection_lengths)
        self.assertEqual(max(projection_lengths), 3)
        self.assertTrue(all(length <= 3 for length in projection_lengths))

    def test_chunked_hidden_loss_preserves_bfloat16_autocast_gradients(self) -> None:
        torch.manual_seed(92)
        tokens = torch.tensor([[2, 4, 5, 6, 7, 8]])
        segments = torch.zeros_like(tokens)
        hidden_reference = torch.randn(1, 6, 5, requires_grad=True)
        weight_reference = torch.randn(13, 5, requires_grad=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            reference, reference_count = causal_lm_loss(
                torch.nn.functional.linear(hidden_reference, weight_reference),
                tokens, segments,
            )
        reference_gradients = torch.autograd.grad(
            reference, (hidden_reference, weight_reference),
        )

        hidden_chunked = hidden_reference.detach().clone().requires_grad_(True)
        weight_chunked = weight_reference.detach().clone().requires_grad_(True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            chunked, chunked_count = causal_lm_loss_from_hidden(
                hidden_chunked, weight_chunked, tokens, segments, chunk_tokens=2,
            )
        chunked_gradients = torch.autograd.grad(
            chunked, (hidden_chunked, weight_chunked),
        )

        self.assertEqual(chunked_count, reference_count)
        torch.testing.assert_close(chunked, reference, rtol=1e-2, atol=1e-3)
        for actual, expected in zip(chunked_gradients, reference_gradients):
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)

    def test_chunked_hidden_loss_materializes_zero_gradients_for_empty_replica(self) -> None:
        hidden = torch.full((2, 4, 5), 1e38, requires_grad=True)
        weight = torch.full((11, 5), 1e38, requires_grad=True)
        tokens = torch.zeros((2, 4), dtype=torch.long)
        segments = torch.full((2, 4), -1, dtype=torch.long)
        eligible = torch.zeros_like(tokens, dtype=torch.bool)

        loss, count, numerator = causal_lm_loss_from_hidden(
            hidden, weight, tokens, segments, eligible=eligible,
            return_numerator=True, allow_empty=True,
        )
        self.assertEqual(count, 0)
        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual(float(numerator.item()), 0.0)
        loss.backward()
        torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))
        torch.testing.assert_close(weight.grad, torch.zeros_like(weight))

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
