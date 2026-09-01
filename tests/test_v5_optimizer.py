from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - environment-dependent
    torch = None
    nn = type("_MissingNN", (), {"Module": object})()

from v5_training.optimizer import (
    BETA1,
    BETA2,
    EPSILON,
    WEIGHT_DECAY,
    build_adamw_optimizer,
    optimizer_group_receipt,
    validate_parameter_ownership,
)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class V5OptimizerTests(unittest.TestCase):
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(4, 3)
            self.norm = nn.LayerNorm(3)
            self.qk_scale = nn.Parameter(torch.ones(3))
            self.output = nn.Linear(3, 4, bias=False)
            self.output.weight = self.embedding.weight

    def test_groups_follow_rank_and_tied_embedding_is_owned_once(self) -> None:
        model = self.TinyModel()
        optimizer = build_adamw_optimizer(model)
        receipt = optimizer_group_receipt(model, optimizer)
        groups = {group["name"]: group for group in receipt["groups"]}
        self.assertIn("embedding.weight", groups["decay"]["parameter_names"])
        self.assertNotIn("output.weight", groups["decay"]["parameter_names"])
        self.assertIn("norm.weight", groups["no_decay"]["parameter_names"])
        self.assertIn("qk_scale", groups["no_decay"]["parameter_names"])
        self.assertEqual(receipt["parameter_count"], len(list(model.parameters())))
        self.assertEqual(sum(group["parameter_count"] for group in groups.values()), receipt["parameter_count"])
        self.assertEqual(optimizer.defaults["betas"], (BETA1, BETA2))
        self.assertEqual(optimizer.defaults["eps"], EPSILON)
        self.assertEqual(optimizer.defaults["weight_decay"], WEIGHT_DECAY)

    def test_receipt_is_deterministic_for_same_parameter_layout(self) -> None:
        first = self.TinyModel()
        second = self.TinyModel()
        first_receipt = optimizer_group_receipt(first, build_adamw_optimizer(first))
        second_receipt = optimizer_group_receipt(second, build_adamw_optimizer(second))
        self.assertEqual(first_receipt, second_receipt)

    def test_ownership_rejects_duplicate_optimizer_parameter(self) -> None:
        model = self.TinyModel()
        optimizer = torch.optim.AdamW([model.embedding.weight, model.embedding.weight])
        with self.assertRaises(ValueError):
            validate_parameter_ownership(model, optimizer)


if __name__ == "__main__":
    unittest.main()
