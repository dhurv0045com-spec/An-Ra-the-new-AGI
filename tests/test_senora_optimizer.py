"""Unit tests for senora.optimizer."""

from __future__ import annotations

import unittest

from senora.optimizer import classify_parameter_decay

try:
    import torch
    from senora.model import build_p35_model
    from senora.optimizer import build_p35_optimizer
except ImportError:
    torch = None
    build_p35_model = None
    build_p35_optimizer = None


class TestSenoraOptimizer(unittest.TestCase):
    def test_classify_parameter_decay(self) -> None:
        self.assertEqual(classify_parameter_decay("embedding.weight", None), "no_decay_0.0")
        self.assertEqual(classify_parameter_decay("blocks.0.attention_norm.weight", None), "no_decay_0.0")
        self.assertEqual(classify_parameter_decay("blocks.0.attention.query_scale", None), "no_decay_0.0")
        self.assertEqual(classify_parameter_decay("final_norm.weight", None), "no_decay_0.0")
        self.assertEqual(classify_parameter_decay("blocks.0.attention.query.weight", None), "decay_0.1")
        self.assertEqual(classify_parameter_decay("blocks.0.ffn.gate.weight", None), "decay_0.1")

    @unittest.skipIf(torch is None, "PyTorch required for optimizer live tests")
    def test_build_p35_optimizer(self) -> None:
        model = build_p35_model(device="cpu")
        optimizer, manifest = build_p35_optimizer(model, learning_rate=3e-4)

        self.assertEqual(manifest.total_trainable_parameters, 35_411_328)
        self.assertEqual(manifest.decayed_parameters_count, 25_952_256)
        self.assertEqual(manifest.non_decayed_parameters_count, 9_459_072)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.1)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)

        # Confirm optimizer owns identical live parameters as model
        model_params = {p for p in model.parameters() if p.requires_grad}
        opt_params = {p for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(model_params, opt_params)


if __name__ == "__main__":
    unittest.main()