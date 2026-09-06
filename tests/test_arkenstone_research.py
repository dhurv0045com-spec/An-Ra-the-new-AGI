"""Arkenstone research-safety tests: metrics, generators, leakage, binding."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from lib import ark_metrics as m  # noqa: E402
from lib import ark_tasks as t  # noqa: E402


class SustainedThresholdTest(unittest.TestCase):
    def test_three_consecutive_required(self) -> None:
        traj = [
            {"step": 100, "test_exact": 0.95, "train_exact": 1.0, "tokens": 1, "exposures": 1},
            {"step": 200, "test_exact": 0.20, "train_exact": 1.0, "tokens": 2, "exposures": 2},
            {"step": 300, "test_exact": 0.91, "train_exact": 1.0, "tokens": 3, "exposures": 3},
        ]
        result = m.sustained_threshold(traj, "test_exact", 0.90)
        self.assertIsNone(result["step"], "two isolated spikes must not count")

    def test_three_consecutive_met(self) -> None:
        traj = [
            {"step": 100, "test_exact": 0.0, "train_exact": 0.2, "tokens": 1, "exposures": 1},
            {"step": 200, "test_exact": 0.91, "train_exact": 1.0, "tokens": 2, "exposures": 2},
            {"step": 300, "test_exact": 0.93, "train_exact": 1.0, "tokens": 3, "exposures": 3},
            {"step": 400, "test_exact": 0.95, "train_exact": 1.0, "tokens": 4, "exposures": 4},
        ]
        result = m.sustained_threshold(traj, "test_exact", 0.90)
        self.assertEqual(result["step"], 200)
        self.assertEqual(result["status"], "DEMONSTRATED")

    def test_run_ending_early_is_not_demonstrated(self) -> None:
        traj = [{"step": 100, "test_exact": 0.95, "train_exact": 1.0, "tokens": 1, "exposures": 1}]
        result = m.sustained_threshold(traj, "test_exact", 0.90)
        self.assertEqual(result["status"], "NOT_DEMONSTRATED")

    def test_summary_delay_and_ratio(self) -> None:
        traj = [
            {"step": 100 * s, "train_exact": 1.0 if s >= 2 else 0.5, "test_exact": v,
             "tokens": 6400 * s, "exposures": 12.8 * s}
            for s, v in [(1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0),
                         (6, 0.0), (7, 0.0), (8, 0.0), (9, 0.55), (10, 0.92),
                         (11, 0.96), (12, 0.95)]
        ]
        summary = m.sustained_summary(traj)
        self.assertEqual(summary["M99"]["step"], 200)
        self.assertEqual(summary["G90"]["step"], 1100)
        self.assertEqual(summary["post_mem_delay_90_steps"], 900)
        self.assertIsNotNone(summary["exposure_ratio_90"])
        self.assertGreater(summary["ood_auc_after_M99"], 0.0)

    def test_max_accuracy_is_not_a_claim(self) -> None:
        summary = m.sustained_summary([{"step": 1, "train_exact": 1.0, "test_exact": 1.0,
                                        "tokens": 1, "exposures": 1}])
        self.assertTrue(summary["max_ood_exact_forbidden_as_claim"])
        self.assertEqual(summary["G90"]["status"], "NOT_DEMONSTRATED")


class TaskManifestTest(unittest.TestCase):
    def test_manifest_is_deterministic(self) -> None:
        first = t.build_task_manifest()
        second = t.build_task_manifest()
        self.assertEqual(first["split_sha256"], second["split_sha256"])

    def test_no_commutation_leakage(self) -> None:
        manifest = t.build_task_manifest()
        self.assertEqual(manifest["pair_overlap_train_test"], 0)

    def test_structural_bands_disjoint(self) -> None:
        manifest = t.build_task_manifest()
        for prompt, _ in manifest["test"]:
            tens = int(prompt.split("+")[0]) // 10
            self.assertIn(tens, (6, 7), "test rows must come from the held-out band")

    def test_frozen_manifest_fail_closed(self) -> None:
        import json
        import tempfile
        manifest = t.build_task_manifest()
        manifest["train"][0][1] = "999"
        path = Path(tempfile.mkdtemp()) / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            t.load_or_build_manifest(str(path))

    def test_deterministic_regeneration(self) -> None:
        first = t.t2_rows("train", 100)
        second = t.t2_rows("train", 100)
        self.assertEqual(first, second, "dataset membership must not depend on run seeds")


class ReceiptBindingTest(unittest.TestCase):
    def test_bind_and_verify(self) -> None:
        code = Path(tempfile.mkdtemp()) / "code.py"
        code.write_text("x = 1", encoding="utf-8")
        receipt = m.bind_receipt(
            experiment_id="unit",
            plan_commit_sha256="a" * 40,
            code_paths={"code.py": str(code)},
            config={"seed": 1},
            results={"ok": True},
        )
        self.assertTrue(m.verify_receipt(receipt, {"code.py": str(code)}))
        receipt["results"]["ok"] = False
        self.assertFalse(m.verify_receipt(receipt, {"code.py": str(code)}))

    def test_tampered_code_fails(self) -> None:
        code = Path(tempfile.mkdtemp()) / "code.py"
        code.write_text("x = 1", encoding="utf-8")
        receipt = m.bind_receipt(
            experiment_id="unit",
            plan_commit_sha256="a" * 40,
            code_paths={"code.py": str(code)},
            config={},
            results={},
        )
        code.write_text("x = 2", encoding="utf-8")
        self.assertFalse(m.verify_receipt(receipt, {"code.py": str(code)}))


if __name__ == "__main__":
    unittest.main()
