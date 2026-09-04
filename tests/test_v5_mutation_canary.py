"""Mutation evidence and target canary tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - environment-dependent
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MutationTests(unittest.TestCase):
    def _model_opt(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return model, optimizer

    def test_change_detection(self) -> None:
        from v5_training.mutation import (
            assert_mutation,
            global_grad_norm,
            moment_fingerprint,
            optimizer_step,
            parameter_sha,
        )

        model, optimizer = self._model_opt()
        before_sha = parameter_sha(model, torch_module=torch)
        before_fp = moment_fingerprint(optimizer, torch_module=torch)
        self.assertEqual(optimizer_step(optimizer), 0)
        model(torch.randn(2, 8)).sum().backward()
        self.assertGreater(global_grad_norm(model, torch_module=torch), 0.0)
        optimizer.step()
        assert_mutation(
            before_sha=before_sha, after_sha=parameter_sha(model, torch_module=torch),
            before_moments=before_fp,
            after_moments=moment_fingerprint(optimizer, torch_module=torch),
            before_step=0, after_step=optimizer_step(optimizer),
            learning_rate=1e-3,
        )

    def test_zero_lr_requires_stillness_but_live_moments(self) -> None:
        from v5_training.mutation import (
            assert_mutation,
            moment_fingerprint,
            optimizer_step,
            parameter_sha,
        )

        model, optimizer = self._model_opt()
        sha = parameter_sha(model, torch_module=torch)
        fp = moment_fingerprint(optimizer, torch_module=torch)
        for group in optimizer.param_groups:
            group["lr"] = 0.0
        model(torch.randn(2, 8)).sum().backward()
        optimizer.step()
        assert_mutation(
            before_sha=sha, after_sha=parameter_sha(model, torch_module=torch),
            before_moments=fp,
            after_moments=moment_fingerprint(optimizer, torch_module=torch),
            before_step=0, after_step=optimizer_step(optimizer),
            learning_rate=0.0,
        )

    def test_unchanged_model_refused(self) -> None:
        from v5_training.mutation import (
            assert_mutation,
            moment_fingerprint,
            parameter_sha,
        )

        model, optimizer = self._model_opt()
        sha = parameter_sha(model, torch_module=torch)
        fp = moment_fingerprint(optimizer, torch_module=torch)
        with self.assertRaises(ValueError):
            assert_mutation(
                before_sha=sha, after_sha=sha, before_moments=fp,
                after_moments=fp, before_step=0, after_step=0,
                learning_rate=1e-3,
            )

    def test_all_none_gradients_refused(self) -> None:
        from v5_training.mutation import global_grad_norm

        model, _ = self._model_opt()
        with self.assertRaises(ValueError):
            global_grad_norm(model, torch_module=torch)

    def test_frozen_parameter_refused(self) -> None:
        from v5_training.mutation import parameter_sha

        model, _ = self._model_opt()
        for parameter in model.parameters():
            parameter.requires_grad = False
        with self.assertRaises(ValueError):
            parameter_sha(model, torch_module=torch)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TargetCanaryTests(unittest.TestCase):
    def test_cpu_canary_passes_all_stages(self) -> None:
        import dataclasses

        from v5_contracts.model_spec import V5A_250M
        from v5_training.target_canary import run_target_canary

        spec = dataclasses.replace(
            V5A_250M, layers=1, width=32, query_heads=2, kv_heads=1,
            head_dimension=16, ffn_width=64, vocabulary_size=256, context_length=64,
        )
        with tempfile.TemporaryDirectory() as directory:
            receipt = run_target_canary(
                model_spec=spec, device="cpu", workdir=Path(directory), seed=3,
                torch_module=torch,
            )
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue(all(value == "PASS" for value in receipt["stages"].values()))


if __name__ == "__main__":
    unittest.main()
