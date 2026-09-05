import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("BRAMASTRA model checks require the bramastra extra")

from bramastra_lab import BramastraModel, ModelConfig, parameter_count


class BramastraModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = ModelConfig(vocab=32, width=16, layers=2, heads=4, ffn=24, max_seq=12)
        self.model = BramastraModel(self.config)

    def test_parameter_count_is_exact_and_output_is_tied(self) -> None:
        actual = sum(parameter.numel() for parameter in self.model.parameters())
        self.assertEqual(actual, parameter_count(self.config))
        self.assertFalse(hasattr(self.model, "lm_head"))

    def test_future_tokens_do_not_change_past_logits(self) -> None:
        first = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        second = torch.tensor([[1, 2, 3, 9, 8]], dtype=torch.long)
        with torch.no_grad():
            first_logits = self.model(first)
            second_logits = self.model(second)
        torch.testing.assert_close(first_logits[:, :3], second_logits[:, :3])

    def test_forward_backward_are_finite_with_trailing_padding(self) -> None:
        tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
        mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
        logits = self.model(tokens, padding_mask=mask)
        self.assertEqual(tuple(logits.shape), (2, 4, self.config.vocab))
        loss = logits[mask].float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in self.model.parameters()))

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, self.config.max_seq + 1, dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model(torch.zeros(3, dtype=torch.long))
        with self.assertRaises((IndexError, RuntimeError)):
            self.model(torch.tensor([[self.config.vocab]], dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model(
                torch.tensor([[1, 0, 2]], dtype=torch.long),
                padding_mask=torch.tensor([[True, False, True]]),
            )


if __name__ == "__main__":
    unittest.main()
