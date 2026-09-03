from __future__ import annotations

import dataclasses
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - environment-dependent
    torch = None

from v5_contracts.model_spec import V5A_250M
from v5_model.config import from_spec
from v5_model.core import assert_receipt, initialize, packed_layout, parameter_receipt


def _tiny_spec():
    return dataclasses.replace(
        V5A_250M, layers=2, width=64, query_heads=2, kv_heads=1,
        head_dimension=32, ffn_width=128, vocabulary_size=256, context_length=64,
    )


class ModelConfigTests(unittest.TestCase):
    def test_from_spec_rejects_nonconforming_specs(self) -> None:
        with self.assertRaises(ValueError):
            from_spec(dataclasses.replace(V5A_250M, linear_bias=True), qk_norm_epsilon=1e-6)
        with self.assertRaises(ValueError):
            from_spec(dataclasses.replace(V5A_250M, dropout=0.1), qk_norm_epsilon=1e-6)
        with self.assertRaises(ValueError):
            from_spec(
                dataclasses.replace(V5A_250M, width=130, head_dimension=65),
                qk_norm_epsilon=1e-6,
            )
        config = from_spec(V5A_250M, qk_norm_epsilon=1e-6)
        self.assertEqual(config.width, 896)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ModelCoreTests(unittest.TestCase):
    def test_tiny_model_inventory_matches_spec_and_runs(self) -> None:
        torch.manual_seed(3)
        spec = _tiny_spec()
        model = initialize(spec, 3)
        assert_receipt(model, spec)
        self.assertEqual(
            sum(parameter_receipt(model).values()), spec.parameter_receipt().total
        )
        model.eval()
        tokens = torch.randint(4, spec.vocabulary_size, (2, 16))
        segments = torch.zeros(2, 16, dtype=torch.int64)
        positions, mask = packed_layout(segments, torch_module=torch)
        with torch.no_grad():
            logits = model(tokens, positions, mask)
        self.assertEqual(tuple(logits.shape), (2, 16, spec.vocabulary_size))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_initialization_is_deterministic_per_seed(self) -> None:
        first = initialize(_tiny_spec(), 11)
        second = initialize(_tiny_spec(), 11)
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_packed_layout_blocks_cross_segment_attention(self) -> None:
        segments = torch.tensor([[0, 0, 1, 1]])
        positions, mask = packed_layout(segments, torch_module=torch)
        self.assertEqual(positions[0].tolist(), [0, 1, 0, 1])
        self.assertFalse(mask[0, 0, 2, 1].item())
        self.assertTrue(mask[0, 0, 1, 0].item())

    def test_v5a_center_inventory_matches_contract(self) -> None:
        torch.manual_seed(5)
        model = initialize(V5A_250M, 5)
        assert_receipt(model, V5A_250M)


if __name__ == "__main__":
    unittest.main()
