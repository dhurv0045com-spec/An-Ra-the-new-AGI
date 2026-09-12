"""Integrity checks for isolated receipts, budgets, metrics and failure packaging."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments/COLAB"))
import discovery_v6_common as common
import run_discovery_v6 as entry


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def context(self, name="run", **kw):
        return common.RunContext(torch.device("cpu"), "test-head", time.time(), 2,
                                 self.root / name, **kw)

    def writer(self, ctx):
        return common.ReceiptWriter(ctx, experiment_id="TEST", plan_sha="plan", runner_path=Path(__file__))

    def test_existing_run_is_never_reused(self):
        self.context()
        with self.assertRaises(FileExistsError):
            self.context()

    def test_receipts_keep_revisions_bind_shared_source_and_detect_tampering(self):
        ctx = self.context()
        writer = self.writer(ctx)
        writer.save("PARTIAL.json", {"completed": 1})
        path = writer.save("PARTIAL.json", {"completed": 2})
        receipt = json.loads(path.read_text())
        self.assertTrue(common.verify_receipt(receipt))
        self.assertIn("experiments/COLAB/discovery_v6_common.py", receipt["source_sha256"])
        for name, digest in receipt["source_sha256"].items():
            self.assertEqual(common.file_sha256(ctx.output_dir / "sources" / name), digest)
        revisions = [json.loads(p.read_text()) for p in (ctx.output_dir / "revisions").glob("*.json")]
        self.assertEqual({r["completed"] for r in revisions}, {1, 2})
        receipt["completed"] = 99
        self.assertFalse(common.verify_receipt(receipt))
        with self.assertRaises(ValueError):
            writer.save("../escape.json", {})
        with self.assertRaises(ValueError):
            writer.save("NAN.json", {"score": float("nan")})

    def test_failed_atomic_replace_keeps_prior_receipt(self):
        path = self.root / "atomic.json"
        common.atomic_write(path, b"previous")
        with patch.object(common.os, "replace", side_effect=OSError("injected disk error")):
            with self.assertRaises(OSError):
                common.atomic_write(path, b"next")
        self.assertEqual(path.read_bytes(), b"previous")
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_archive_excludes_other_runs(self):
        a, b = self.context("a"), self.context("b")
        self.writer(a).save("A.json", {})
        self.writer(b).save("B.json", {})
        archive = common.package_all(False, ctx=a)
        with zipfile.ZipFile(archive) as zf:
            self.assertIn("A.json", zf.namelist())
            self.assertNotIn("B.json", zf.namelist())
            self.assertIn("sources/experiments/COLAB/discovery_v6_common.py", zf.namelist())
            for name in zf.namelist():
                if name.endswith(".json") and not name.startswith("sources/"):
                    self.assertTrue(common.verify_receipt(json.loads(zf.read(name))))

    def test_wall_clock_changes_cannot_extend_budget(self):
        ctx = self.context()
        with patch.object(common.time, "time", return_value=-1e12):
            self.assertLessEqual(ctx.minutes_left, 2)
        ctx._monotonic_start -= 121
        with self.assertRaises(common.BudgetExhausted):
            common.ensure_budget(ctx)

    def test_nonpositive_budget_rejected_before_directory_creation(self):
        for budget in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                common.RunContext(torch.device("cpu"), "head", time.time(), budget, self.root / "invalid")
        self.assertFalse((self.root / "invalid").exists())

    def test_recurrent_drop_excludes_initial_unrecovered_period(self):
        values = [.1, .2, .3, .95, .95, .95, .4, .4, .4]
        metrics = common.trajectory_metrics(
            [{"step": i * 200, "exact": v} for i, v in enumerate(values, 1)], "exact")
        self.assertEqual(metrics["G90_CONFIRM"], 1200)
        self.assertEqual(metrics["DROP90_ONSET"], 1400)
        self.assertEqual(metrics["DROP90_CONFIRM"], 1800)
        metrics = common.trajectory_metrics([{"step": i, "exact": .2} for i in range(4)], "exact")
        self.assertIsNone(metrics["DROP90_CONFIRM"])

    def test_failure_and_keyboard_interrupt_still_package_receipt(self):
        for i, error in enumerate([RuntimeError("injected"), KeyboardInterrupt()]):
            output = self.root / f"failure{i}"
            with patch.object(entry, "preflight", side_effect=error):
                self.assertEqual(entry.main(["--output-dir", str(output)]), 1)
            with zipfile.ZipFile(output / "ARKENSTONE_DISCOVERY_V6_RESULTS.zip") as zf:
                data = json.loads(zf.read("FAILURE_RECEIPT.json"))
                self.assertEqual(data["status"], "FAILED")
                self.assertTrue(common.verify_receipt(data))

    def test_cuda_initialization_failure_can_still_write_receipt(self):
        output = self.root / "no-cuda"
        with patch.object(entry, "current_device", side_effect=RuntimeError("no GPU")), \
             patch.object(common.torch.cuda, "get_device_name", side_effect=RuntimeError("no GPU")):
            result = entry.main(["--device", "cuda", "--output-dir", str(output)])
        self.assertEqual(result, 1)
        receipt = json.loads((output / "FAILURE_RECEIPT.json").read_text())
        self.assertEqual(receipt["device_name"], "unavailable: RuntimeError")
        self.assertTrue(common.verify_receipt(receipt))
        self.assertTrue((output / "ARKENSTONE_DISCOVERY_V6_RESULTS.zip").exists())

    def test_shared_training_boundary_enforces_budget(self):
        ctx = self.context()
        calls = []
        fake = SimpleNamespace(__file__=str(Path(__file__)),
                               loss_and_positions=lambda: calls.append("forward"))
        common.bind_ark11_runtime(fake, ctx.device, ctx.head, ctx=ctx)
        self.assertEqual(fake.RESULTS_DIR, ctx.output_dir / "ARK-011")
        fake.loss_and_positions()
        self.assertEqual(calls, ["forward"])
        ctx._monotonic_start -= 121
        with self.assertRaises(common.BudgetExhausted):
            fake.loss_and_positions()
        self.assertEqual(calls, ["forward"])

    def test_invalid_trajectories_fail_closed(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                common.trajectory_metrics([{"step": 1, "exact": value}], "exact")
        for steps in ((1, 1), (2, 1)):
            with self.assertRaises(ValueError):
                common.trajectory_metrics([{"step": step, "exact": .5} for step in steps], "exact")

    def test_runtime_imports_have_no_output_directory_side_effect(self):
        with patch.object(Path, "mkdir", side_effect=AssertionError("import attempted a directory write")):
            common.load_ark11()
            common.import_experiment_module(12)
            common.import_experiment_module(13)

    def test_sampler_preserves_multiple_legacy_streams(self):
        for seed, size, pool in [(0, 1, 1), (6801, 64, 500), (2702, 7, 997)]:
            g = torch.Generator().manual_seed(seed)
            expected = [torch.randint(0, pool, (size,), generator=g).tolist() for _ in range(19)]
            self.assertEqual(common.generate_indices(seed, 19, size, pool), expected)
        self.assertEqual(common.generate_indices(1, 0, 4, 5), [])


if __name__ == "__main__":
    unittest.main()
