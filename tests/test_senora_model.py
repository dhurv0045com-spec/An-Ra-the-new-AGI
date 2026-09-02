"""Unit tests for senora.model."""

from __future__ import annotations

import unittest

from senora.model import (
    EXPECTED_P35_PARAMETER_COUNT,
    P35_MODEL_SPEC,
    get_p35_parameter_receipt,
    p35_constructor_sha256,
)

try:
    import torch
    from senora.model import P35Model, build_p35_model
except ImportError:
    torch = None
    P35Model = None
    build_p35_model = None


class TestSenoraModel(unittest.TestCase):
    def test_parameter_receipt(self) -> None:
        receipt = get_p35_parameter_receipt()
        self.assertEqual(receipt.total_parameters, EXPECTED_P35_PARAMETER_COUNT)
        self.assertEqual(receipt.total_parameters, 35_411_328)
        self.assertTrue(receipt.weight_tying_verified)
        self.assertTrue(receipt.qk_norm_scales_verified)
        self.assertEqual(receipt.dormant_parameter_count, 0)

    def test_constructor_sha256_deterministic(self) -> None:
        sha1 = p35_constructor_sha256()
        sha2 = p35_constructor_sha256()
        self.assertEqual(sha1, sha2)
        self.assertEqual(len(sha1), 64)

    @unittest.skipIf(torch is None, "PyTorch required for live neural model tests")
    def test_live_model_construction_and_invariants(self) -> None:
        model = build_p35_model(device="cpu")
        self.assertEqual(model.parameter_count(), EXPECTED_P35_PARAMETER_COUNT)
        self.assertTrue(model.verify_weight_tying())

        # Test forward pass with tiny batch
        batch_size = 2
        seq_len = 8
        tokens = torch.randint(0, P35_MODEL_SPEC.vocabulary_size, (batch_size, seq_len))
        logits = model(tokens)
        self.assertEqual(logits.shape, (batch_size, seq_len, P35_MODEL_SPEC.vocabulary_size))
        self.assertTrue(torch.isfinite(logits).all())

    @unittest.skipIf(torch is None, "PyTorch required for causal mask test")
    def test_causal_mask_invariance(self) -> None:
        # Modifying token at t=4 should NOT affect logits at t < 4
        model = build_p35_model(device="cpu")
        tokens1 = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
        tokens2 = torch.tensor([[10, 20, 30, 999, 50]], dtype=torch.long)

        logits1 = model(tokens1)
        logits2 = model(tokens2)

        # Logits at positions 0, 1, 2 must be identical
        self.assertTrue(torch.allclose(logits1[:, :3], logits2[:, :3], atol=1e-5))

    @unittest.skipIf(torch is None, "PyTorch required for trace hook test")
    def test_trace_hooks_triquetra_readiness(self) -> None:
        model = build_p35_model(device="cpu")
        captured_layers: list[str] = []

        def trace_hook(name: str, tensor: torch.Tensor) -> None:
            captured_layers.append(name)

        model.register_trace_hook(trace_hook)
        tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
        _ = model(tokens)

        # 16 transformer blocks + 1 final norm = 17 hook invocations
        self.assertEqual(len(captured_layers), 17)
        self.assertEqual(captured_layers[0], "block_0")
        self.assertEqual(captured_layers[-1], "final_norm")


if __name__ == "__main__":
    unittest.main()