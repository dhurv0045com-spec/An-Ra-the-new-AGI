"""Contract tests for the 100M TPU preflight (no local XLA/model allocation)."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bramastra_lab.research.campaigns import tpu_100m


class TPU100MPreflightTests(unittest.TestCase):
    def test_dedicated_tpu_notebook_cells_are_valid_python(self) -> None:
        path = Path(__file__).resolve().parents[1] / "notebooks" / \
            "bramastra_tpu_100m_preflight.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(notebook["nbformat_minor"], 4)
        code_cells = [cell for cell in notebook["cells"]
                      if cell["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 4)
        for index, cell in enumerate(code_cells):
            with self.subTest(cell=index):
                ast.parse("".join(cell["source"]))
        source = "\n".join("".join(cell["source"])
                            for cell in notebook["cells"])
        code_source = "\n".join("".join(cell["source"])
                                 for cell in code_cells)
        self.assertIn("GIT_REF = os.environ.get('BRAMASTRA_GIT_REF', 'Gandiva')",
                      source)
        self.assertIn("run_tpu_100m_preflight", source)
        self.assertNotIn("bramastra_k8.ipynb", code_source)

    def test_module_import_does_not_import_torch_xla_or_model(self) -> None:
        # Isolate this import check from the order in which a larger suite may
        # already have imported torch_xla or the real model package.
        check = subprocess.run(
            [sys.executable, "-c",
             "import sys; "
             "from bramastra_lab.research.campaigns import tpu_100m; "
             "assert 'torch_xla' not in sys.modules; "
             "assert 'bramastra_lab.research.models' not in sys.modules"],
            check=False, capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stderr)

    def test_sample_plan_uses_eight_distinct_training_rows_deterministically(self) -> None:
        from bramastra_lab.research.campaigns.phases import compiler

        rows = [{"pool": "training", "answer": str(index % 3),
                 "mechanism_id": f"train-{index}"} for index in range(24)]
        fake_compilation = {"target_count": 7,
                            "objective_counts": {"token": 7, "pair": 1}}
        with patch.object(compiler, "load_training_trajectories",
                          return_value=rows), patch.object(
                tpu_100m, "_compile_example", return_value=fake_compilation):
            first = tpu_100m._build_sample_plan(
                "unused", seed=1701, expected_replicas=8)
            second = tpu_100m._build_sample_plan(
                "unused", seed=1701, expected_replicas=8)
        from bramastra_lab.research.config import BuildConfig

        self.assertEqual(first, second)
        self.assertEqual(first["config_identity"],
                         BuildConfig.from_dict(
                             {"model": {"profile": "tpu_100m"}}).identity())
        self.assertEqual(len(first["workers"]), 8)
        self.assertEqual({row["rank"] for row in first["workers"]}, set(range(8)))
        self.assertEqual(len({row["row_index"] for row in first["workers"]}), 8)
        self.assertEqual({row["target_count"] for row in first["workers"]}, {7})

    def test_compiled_pair_count_matches_two_ordered_counterfactual_rows(self) -> None:
        from types import SimpleNamespace
        from bramastra_lab.research.campaigns.phases import compiler

        batch = SimpleNamespace(target_count=7)
        compiled = {
            "weights": {"token": 1.0, "world": 0.5, "action": 0.5,
                        "value": 0.1, "pair": 0.1},
            "enabled": frozenset({"token", "world", "action", "value", "pair"}),
            "world": {"denominator": 1}, "action": {"denominator": 1},
            "value": {"denominator": 1}, "pair": {"denominator": 1},
        }
        rows = ([object(), object()], [object(), object()])
        a = {"pool": "training", "answer": "A", "mechanism_id": "train-a"}
        b = {"pool": "training", "answer": "B", "mechanism_id": "train-b"}
        with patch.object(compiler, "build_batch_for_trajectory", return_value=batch), \
                patch.object(compiler, "compile_channels_for_row",
                             return_value=compiled), \
                patch.object(compiler, "build_pair_rows", return_value=rows):
            prepared = tpu_100m._compile_example(a, b, max_seq=512)

        self.assertEqual(prepared["compiled"]["pair"]["denominator"], 2)
        self.assertEqual(prepared["objective_counts"]["pair"], 2)

    def test_sample_plan_fails_when_no_distinct_answer_pair_exists(self) -> None:
        from bramastra_lab.research.campaigns.phases import compiler

        rows = [{"pool": "training", "answer": "same",
                 "mechanism_id": f"train-{index}"} for index in range(10)]
        with patch.object(compiler, "load_training_trajectories",
                          return_value=rows):
            with self.assertRaisesRegex(ValueError, "no distinct-answer pair"):
                tpu_100m._build_sample_plan(
                    "unused", seed=0, expected_replicas=8)

    def test_sample_plan_rejects_invalid_replica_counts(self) -> None:
        from bramastra_lab.research.campaigns.phases import compiler

        with patch.object(compiler, "load_training_trajectories", return_value=[]):
            for value in (0, -1, True, 1.5):
                with self.subTest(value=value), self.assertRaisesRegex(
                        ValueError, "expected_replicas"):
                    tpu_100m._build_sample_plan(
                        "unused", seed=0, expected_replicas=value)

    def test_sample_plan_fails_closed_if_eight_valid_examples_cannot_be_built(self) -> None:
        from bramastra_lab.research.campaigns.phases import compiler

        rows = [{"pool": "training", "answer": str(index % 2),
                 "mechanism_id": f"train-{index}"} for index in range(9)]
        with patch.object(compiler, "load_training_trajectories",
                          return_value=rows), patch.object(
                tpu_100m, "_compile_example",
                side_effect=ValueError("bad trajectory")):
            with self.assertRaisesRegex(ValueError, "could compile only"):
                tpu_100m._build_sample_plan(
                    "unused", seed=0, expected_replicas=8)

    def test_refuses_before_bundle_or_model_work_without_visible_tpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_dir = str(Path(directory) / "run")
            with patch("bramastra_lab.research.runtime.tpu.inspect_tpu_runtime",
                       return_value={"status": "TPU_RUNTIME_NOT_SELECTED",
                                     "training_started": False}), patch(
                    "bramastra_lab.research.runtime.tpu.launch_tpu_workers") as launch:
                with self.assertRaisesRegex(RuntimeError, "runtime preflight refused"):
                    tpu_100m.run_tpu_100m_preflight(
                        data_dir="must-not-be-read", report_dir=report_dir)
                launch.assert_not_called()
            runtime = json.loads(
                (Path(report_dir) / "runtime.json").read_text(encoding="utf-8"))
            self.assertEqual(runtime["status"], "TPU_RUNTIME_NOT_SELECTED")
            self.assertFalse(runtime["training_started"])

    def test_reports_are_exclusive_and_never_overwrite_previous_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            tpu_100m._write_exclusive_json(path, {"status": "first"})
            with self.assertRaises(FileExistsError):
                tpu_100m._write_exclusive_json(path, {"status": "second"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             {"status": "first"})

    def test_registered_model_size_is_configuration_only_and_exact(self) -> None:
        from bramastra_lab.research.config import BuildConfig

        config = BuildConfig.from_dict({"model": {"profile": "tpu_100m"}})
        self.assertEqual(config.parameter_count(),
                         tpu_100m.EXPECTED_PARAMETER_COUNT)
        self.assertEqual(config.model.max_seq, 512)


if __name__ == "__main__":
    unittest.main()
