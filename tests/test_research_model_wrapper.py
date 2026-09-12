"""B01 focused tests: the integrated wrapper around the accepted decoder.

These tests construct tiny models and run forward/backward correctness
checks only. No optimizer is created and no update is performed; learned
smoke remains separately gated and is not exercised here.
"""
import unittest

import torch

from bramastra_lab.research.config import BuildConfig, seed_everything
from bramastra_lab.research.models import IntegratedModel


def tiny_build_config() -> BuildConfig:
    return BuildConfig.from_dict({"model": {"profile": "tiny"}})


class WrapperStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        seed_everything(11)
        self.config = tiny_build_config()
        self.model = IntegratedModel(self.config)

    def test_instantiated_count_matches_analytic_count(self) -> None:
        actual = sum(parameter.numel() for parameter in self.model.parameters())
        self.assertEqual(actual, self.config.parameter_count())
        self.assertEqual(self.model.base_parameter_count(),
                         self.config.base_decoder_parameter_count())
        self.assertEqual(self.model.head_parameter_count(), 2 * self.config.model.width)

    def test_logits_cover_full_physical_vocabulary(self) -> None:
        tokens = torch.randint(0, self.config.model.vocab, (2, 16))
        output = self.model(tokens)
        self.assertEqual(tuple(output.logits.shape), (2, 16, self.config.model.vocab))
        self.assertIsNone(output.hidden)
        self.assertIsNone(output.action_scores)
        self.assertIsNone(output.value)

    def test_wrapper_shares_decoder_representation(self) -> None:
        tokens = torch.randint(0, self.config.model.vocab, (1, 9))
        self.model.eval()
        with torch.no_grad():
            through_wrapper = self.model(tokens).logits
            through_decoder = self.model.decoder(tokens)
        torch.testing.assert_close(through_wrapper, through_decoder)

    def test_hidden_states_are_returned_when_requested(self) -> None:
        tokens = torch.randint(0, self.config.model.vocab, (2, 7))
        output = self.model(tokens, return_hidden=True)
        self.assertEqual(tuple(output.hidden.shape), (2, 7, self.config.model.width))


class ActionAndValueHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        seed_everything(13)
        self.config = tiny_build_config()
        self.model = IntegratedModel(self.config)

    def test_action_scores_shape_and_masking(self) -> None:
        tokens = torch.randint(0, self.config.model.vocab, (2, 12))
        spans = torch.tensor([[3, 7, 11], [4, 8, 9]])
        mask = torch.tensor([[True, True, False], [True, False, False]])
        output = self.model(tokens, action_span_ends=spans, action_mask=mask)
        self.assertEqual(tuple(output.action_scores.shape), (2, 3))
        self.assertTrue(torch.isinf(output.action_scores[0, 2]))
        self.assertTrue(torch.isinf(output.action_scores[1, 1]))
        self.assertTrue(torch.isinf(output.action_scores[1, 2]))
        self.assertTrue(torch.isfinite(output.action_scores[0, :2]).all())

    def test_all_illegal_row_rejects(self) -> None:
        tokens = torch.randint(0, self.config.model.vocab, (1, 6))
        spans = torch.tensor([[1, 2]])
        mask = torch.tensor([[False, False]])
        with self.assertRaises(ValueError):
            self.model(tokens, action_span_ends=spans, action_mask=mask)

    def test_out_of_range_span_rejects(self) -> None:
        tokens = torch.randint(0, self.config.model.vocab, (1, 6))
        spans = torch.tensor([[6]])
        with self.assertRaises(ValueError):
            self.model(tokens, action_span_ends=spans)

    def test_value_uses_last_real_position_under_padding(self) -> None:
        tokens = torch.randint(0, self.config.model.vocab, (2, 8))
        padding = torch.tensor([[True] * 8, [True, True, True, True, False, False, False, False]])
        with_value = self.model(tokens, padding_mask=padding, return_value=True)
        self.assertEqual(tuple(with_value.value.shape), (2,))
        # Changing padded suffix content must not change the value estimate.
        tokens_perturbed = tokens.clone()
        tokens_perturbed[1, 5:] = (tokens_perturbed[1, 5:] + 7) % self.config.model.vocab
        perturbed = self.model(tokens_perturbed, padding_mask=padding, return_value=True)
        torch.testing.assert_close(with_value.value[1], perturbed.value[1])

    def test_mask_without_spans_rejects(self) -> None:
        tokens = torch.randint(0, self.config.model.vocab, (1, 6))
        with self.assertRaises(ValueError):
            self.model(tokens, action_mask=torch.tensor([[True, False]]))


class InitializationTests(unittest.TestCase):
    def test_repeated_seed_initialization_matches(self) -> None:
        config = tiny_build_config()
        seed_everything(77)
        first = IntegratedModel(config)
        seed_everything(77)
        second = IntegratedModel(config)
        for (name_a, tensor_a), (name_b, tensor_b) in zip(
                first.state_dict().items(), second.state_dict().items()):
            self.assertEqual(name_a, name_b)
            torch.testing.assert_close(tensor_a, tensor_b)

    def test_different_seeds_differ(self) -> None:
        config = tiny_build_config()
        seed_everything(1)
        first = IntegratedModel(config)
        seed_everything(2)
        second = IntegratedModel(config)
        weights_a = first.decoder.embedding.weight.detach()
        weights_b = second.decoder.embedding.weight.detach()
        self.assertFalse(torch.equal(weights_a, weights_b))


class GradientPathTests(unittest.TestCase):
    """Backward reachability only; no optimizer updates are performed here."""

    def test_loss_backward_reaches_decoder_and_heads(self) -> None:
        seed_everything(21)
        config = tiny_build_config()
        model = IntegratedModel(config)
        tokens = torch.randint(0, config.model.vocab, (2, 10))
        output = model(tokens, return_value=True)
        answer_loss = torch.nn.functional.cross_entropy(
            output.logits[:, :-1].reshape(-1, config.model.vocab),
            tokens[:, 1:].reshape(-1))
        value_loss = output.value.square().mean()
        (answer_loss + value_loss).backward()
        self.assertIsNotNone(model.decoder.embedding.weight.grad)
        self.assertIsNotNone(model.value_head.weight.grad)
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
