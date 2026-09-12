"""Demo-command acceptance: real checkpoint verification, refusal on missing or
mismatched evidence, and rejection of unsupported success badges."""
import copy
import importlib.util
import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "COLAB"))
sys.path.insert(0, str(REPO / "experiments" / "ARK-001"))
sys.path.insert(0, str(REPO / "experiments" / "ARK-014"))


def load_module(relpath, name):
    path = REPO / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


binding = load_module("experiments/ARK-014/ark014_binding.py", "test_ark014demo_binding")
runner = load_module("experiments/ARK-014/run_ark014.py", "test_ark014demo_runner")
demo = load_module("experiments/ARK-014/demo.py", "test_ark014demo_demo")

TASK = binding.build_binding_task()
META = runner._row_meta(TASK)


def bind_ark11(device):
    from discovery_v6_common import load_ark11
    ark11 = load_ark11()
    ark11.DEVICE = device
    ark11.RUNNER_HEAD = "test-head"
    ark11.RUNNER_SOURCE_SHA256 = "test"
    return ark11


def build_completed_run(parent: Path, *, max_steps: int = 2) -> tuple[Path, dict]:
    """A real diagnostic-scale run directory with genuine checkpoints."""
    run_dir = parent / "ark014-demo-source-run"
    ark11 = bind_ark11(torch.device("cpu"))
    from discovery_v6_common import RunContext
    ctx = RunContext(torch.device("cpu"), "test-head", time.time(), 30.0, run_dir)
    writer_checkpoints = ctx.output_dir / "checkpoints"
    acquisitions = []
    for regime in ("CANONICAL_TRAIN", "ORDER_AUGMENTED"):
        arm = runner.acquire_arm(ark11, regime=regime, task=TASK, meta=META,
                                 device=ctx.device, ctx=ctx,
                                 writer=type("W", (), {"save": staticmethod(lambda *a, **k: None)})(),
                                 checkpoints_dir=writer_checkpoints, max_steps=max_steps,
                                 eval_every=1, log=lambda *a, **k: None)
        acquisitions.append(arm)
    run = {
        "status": "EXECUTED_OR_PARTIAL_BUDGETED",
        "protocol_scale": "DIAGNOSTIC_SCALE_NOT_PREREGISTERED",
        "acquisitions": acquisitions,
        "retention": [],
        "summary": {"verdict": "ROBUST_BINDING_NOT_ACQUIRED"},
    }
    (run_dir / demo.RESULT_FILE).write_text(json.dumps(run, default=str), encoding="utf-8")
    (run_dir / demo.TASK_FILE).write_text(
        json.dumps(TASK["manifest"], default=str), encoding="utf-8")
    return run_dir, run


class DemoAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.parent = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_happy_path_writes_report_and_examples(self):
        run_dir, _ = build_completed_run(self.parent)
        out = self.parent / "demo-out-1"
        rc = demo.main(["--run-dir", str(run_dir), "--output", str(out)])
        self.assertEqual(rc, 0)
        self.assertTrue((out / "report.html").exists())
        examples = json.loads((out / "examples.json").read_text(encoding="utf-8"))
        self.assertEqual(len(examples["examples"]), 144)  # 12 fact-sets x 4 diagnostics x 3 queries
        self.assertEqual(examples["verdict"], "ROBUST_BINDING_NOT_ACQUIRED")
        for aggregate in examples["aggregate"].values():
            for side in ("baseline", "candidate"):
                self.assertEqual(aggregate[side]["denominator"], 150)

    def test_demo_selection_rule_is_frozen_and_outcome_independent(self):
        first = demo.select_demo_factsets(TASK)
        second = demo.select_demo_factsets(binding.build_binding_task())
        self.assertEqual([binding.fact_signature(f) for f in first],
                         [binding.fact_signature(f) for f in second])
        self.assertEqual(len(first), demo.DEMO_FACTSET_COUNT)
        # Selection cannot depend on model success: it is a pure function of the task.
        ranked = sorted(TASK["sealed_factsets"],
                        key=demo._hash_rank_key)
        self.assertEqual([binding.fact_signature(f) for f in first][:3],
                         [binding.fact_signature(f) for f in ranked[:3]])

    def test_missing_result_file_refuses(self):
        empty = self.parent / "empty-run"
        empty.mkdir()
        out = self.parent / "demo-out-2"
        rc = demo.main(["--run-dir", str(empty), "--output", str(out)])
        self.assertEqual(rc, 2)
        self.assertIn("DEMO REFUSED", (out / "REFUSED.txt").read_text(encoding="utf-8"))
        self.assertFalse((out / "report.html").exists())

    def test_missing_checkpoint_refuses(self):
        run_dir, _ = build_completed_run(self.parent)
        (run_dir / "checkpoints" / "CANONICAL_TRAIN_final.pt").unlink()
        out = self.parent / "demo-out-3"
        rc = demo.main(["--run-dir", str(run_dir), "--output", str(out)])
        self.assertEqual(rc, 2)
        text = (out / "REFUSED.txt").read_text(encoding="utf-8")
        self.assertIn("checkpoint file missing", text)

    def test_mismatched_checkpoint_refuses(self):
        run_dir, _ = build_completed_run(self.parent)
        target = run_dir / "checkpoints" / "ORDER_AUGMENTED_final.pt"
        data = target.read_bytes()
        target.write_bytes(data + b"tampered")
        out = self.parent / "demo-out-4"
        rc = demo.main(["--run-dir", str(run_dir), "--output", str(out)])
        self.assertEqual(rc, 2)
        self.assertIn("checkpoint hash mismatch", (out / "REFUSED.txt").read_text(encoding="utf-8"))

    def test_task_manifest_drift_refuses(self):
        run_dir, _ = build_completed_run(self.parent)
        manifest = json.loads((run_dir / demo.TASK_FILE).read_text(encoding="utf-8"))
        manifest["task_seed"] = 999
        (run_dir / demo.TASK_FILE).write_text(json.dumps(manifest), encoding="utf-8")
        out = self.parent / "demo-out-5"
        rc = demo.main(["--run-dir", str(run_dir), "--output", str(out)])
        self.assertEqual(rc, 2)
        text = (out / "REFUSED.txt").read_text(encoding="utf-8")
        # Body tampering is caught by the manifest's own hash before the
        # frozen-builder comparison would fire.
        self.assertIn("task manifest", text)
        self.assertTrue("tampered" in text or "drift" in text)

    def test_unsupported_success_badge_refuses(self):
        run_dir, run = build_completed_run(self.parent)
        # A receipt claiming QUALIFIED that its own random-weight checkpoints
        # cannot support must be rejected by the demo.
        doctored = copy.deepcopy(run)
        for arm in doctored["acquisitions"]:
            arm["status"] = "QUALIFIED"
            arm["sealed_at_qualification"] = {d: 0.99 for d in binding.DIAGNOSTICS}
        doctored["summary"] = {"verdict": "ORDER_ROBUSTNESS_REPAIRED"}
        (run_dir / demo.RESULT_FILE).write_text(json.dumps(doctored, default=str), encoding="utf-8")
        out = self.parent / "demo-out-6"
        rc = demo.main(["--run-dir", str(run_dir), "--output", str(out)])
        self.assertEqual(rc, 2)
        text = (out / "REFUSED.txt").read_text(encoding="utf-8")
        self.assertIn("DEMO REFUSED", text)

    def test_threshold_check_rejects_badge_when_receipt_values_agree(self):
        # Direct unit of the badge guard: receipt values agree with the
        # recomputed aggregates, but the recomputed BIND_CONTROL diagnostics do
        # not reach the frozen thresholds, so QUALIFIED must be refused.
        aggregates = {f"BIND_CONTROL/{d}": {"exact": 0.5, "denominator": 150}
                      for d in binding.DIAGNOSTICS}
        aggregates.update({f"BIND_SEALED/{d}": {"exact": 0.5, "denominator": 150}
                           for d in binding.DIAGNOSTICS})
        arm = {
            "status": "QUALIFIED",
            "trajectory": [{"step": 5, **{f"control_{d}": 0.5 for d in binding.DIAGNOSTICS}}],
            "checkpoint": {"step": 5},
            "sealed_at_qualification": {d: 0.5 for d in binding.DIAGNOSTICS},
        }
        with self.assertRaises(demo.DemoRefused) as caught:
            demo._verify_against_receipt(arm, aggregates, "fake.pt")
        self.assertIn("unsupported success badge", str(caught.exception))

    def test_failures_are_displayed_not_replaced(self):
        run_dir, _ = build_completed_run(self.parent)
        out = self.parent / "demo-out-7"
        demo.main(["--run-dir", str(run_dir), "--output", str(out)])
        examples = json.loads((out / "examples.json").read_text(encoding="utf-8"))
        # Diagnostic-scale checkpoints from random weights will fail most rows;
        # whatever fails must keep its raw model prediction.
        for record in examples["examples"]:
            if not record["candidate_correct"]:
                self.assertNotEqual(record["candidate_prediction"], record["ground_truth"])
        html = (out / "report.html").read_text(encoding="utf-8")
        self.assertIn("Failures on the demonstrated examples", html)
        self.assertIn("not corrected or hidden", html)

    def test_output_directory_must_be_new(self):
        run_dir, _ = build_completed_run(self.parent)
        out = self.parent / "demo-out-8"
        out.mkdir()
        with self.assertRaises(FileExistsError):
            demo.main(["--run-dir", str(run_dir), "--output", str(out)])


if __name__ == "__main__":
    unittest.main()
