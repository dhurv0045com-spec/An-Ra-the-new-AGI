from __future__ import annotations

import dataclasses
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:  # pragma: no cover - environment-dependent
    torch = None

from v5_contracts.model_spec import V5A_250M
import v5_model.core as core_module
from v5_model.attention import apply_rope, build_rope_cache
from v5_model.config import from_spec
from v5_model.core import (
    _packed_layout_from_validated_segments,
    assert_receipt,
    initialize,
    packed_layout,
    parameter_receipt,
)


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

    def test_prevalidated_layout_matches_strict_path_without_scalar_readback(self) -> None:
        from unittest.mock import patch

        segments = torch.tensor([[0, 0, 1, 1, -1, -1]])
        expected_positions, expected_mask = packed_layout(segments, torch_module=torch)
        with patch.object(
            torch.Tensor,
            "item",
            side_effect=AssertionError("prevalidated layout must not read scalars to the host"),
        ):
            positions, mask = _packed_layout_from_validated_segments(
                segments, torch_module=torch,
            )
        self.assertTrue(torch.equal(positions, expected_positions))
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_shared_rope_cache_matches_original_outputs_and_gradients(self) -> None:
        positions = torch.tensor([[0, 1, 2, 0, 1], [0, 1, 2, 3, 4]])
        cosine, sine = build_rope_cache(
            positions, head_dimension=8, rope_base=10_000.0,
            torch_module=torch,
        )
        inverse = 10_000.0 ** (
            -torch.arange(0, 8, 2, dtype=torch.float32) / 8
        )
        phase = positions.float()[:, None, :, None] * inverse[None, None, None, :]
        for dtype in (torch.float32, torch.bfloat16):
            value = torch.randn(2, 3, 5, 8).to(dtype).requires_grad_(True)
            probe = torch.randn(2, 3, 5, 8).to(dtype)
            actual = apply_rope(value, cosine, sine, torch_module=torch)
            old_cosine = phase.cos().to(value.dtype)
            old_sine = phase.sin().to(value.dtype)
            even, odd = value[..., 0::2], value[..., 1::2]
            expected = torch.stack((
                even * old_cosine - odd * old_sine,
                even * old_sine + odd * old_cosine,
            ), -1).flatten(-2)
            actual_gradient = torch.autograd.grad(
                (actual.float() * probe.float()).sum(), value,
            )[0]
            expected_gradient = torch.autograd.grad(
                (expected.float() * probe.float()).sum(), value,
            )[0]

            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(actual_gradient, expected_gradient))

    def test_model_builds_rope_angles_once_and_checkpointing_preserves_gradients(self) -> None:
        spec = _tiny_spec()
        plain = initialize(spec, 47).train()
        checkpointed = initialize(spec, 47).train()
        tokens = torch.randint(4, spec.vocabulary_size, (2, 16))
        segments = torch.zeros(2, 16, dtype=torch.int64)
        positions, mask = packed_layout(segments, torch_module=torch)
        cache_builder = core_module.build_rope_cache

        with patch.object(core_module, "build_rope_cache", wraps=cache_builder) as counted:
            plain_logits = plain(tokens, positions, mask)
            checkpoint_logits = checkpointed(
                tokens, positions, mask, use_activation_checkpointing=True,
            )
        self.assertEqual(counted.call_count, 2)
        self.assertTrue(torch.equal(plain_logits, checkpoint_logits))
        plain_logits.float().square().mean().backward()
        checkpoint_logits.float().square().mean().backward()
        for plain_parameter, checkpoint_parameter in zip(
            plain.parameters(), checkpointed.parameters(),
        ):
            self.assertTrue(torch.equal(
                plain_parameter.grad, checkpoint_parameter.grad,
            ))

    def test_v5a_center_inventory_matches_contract(self) -> None:
        torch.manual_seed(5)
        model = initialize(V5A_250M, 5)
        assert_receipt(model, V5A_250M)


if __name__ == "__main__":
    unittest.main()
