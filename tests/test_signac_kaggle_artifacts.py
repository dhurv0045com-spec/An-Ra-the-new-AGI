import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from signac_100m.kaggle_artifacts import (
    create_kaggle_results_bundle,
    format_canary_failure_diagnostics,
)


class KaggleResultsBundleTests(unittest.TestCase):
    def test_failure_diagnostics_show_failed_ranks_and_saved_launcher_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "run-001" / "m102_primary"
            candidate.mkdir(parents=True)
            (candidate / "rank-02.json").write_text(json.dumps({
                "status": "FAIL", "ordinal": 2, "error_type": "RuntimeError",
                "error": "TPU memory allocation failed",
            }))
            (candidate / "rank-00.json").write_text(json.dumps({
                "status": "PASS", "ordinal": 0, "error": "must not appear",
            }))
            (candidate / "resume-rank-03.json").write_text(json.dumps({
                "status": "FAIL", "ordinal": 3, "error_type": "ValueError",
                "error": "resume state mismatch",
            }))
            (candidate / "launch-failure.json").write_text(json.dumps({
                "status": "FAIL", "phase": "restart_worker_group",
                "error_type": "RuntimeError", "error": "XLA child process exited",
                "traceback": "Traceback\n  File 'runner.py', line 12\nRuntimeError",
            }))
            log = root / "m102_primary.launcher.log"
            log.write_text("setup ok\nfinal XLA diagnostic\n", encoding="utf-8")

            message = format_canary_failure_diagnostics(
                candidate="m102_primary",
                return_code=1,
                receipt_root=root / "run-001",
                launcher_log=log,
            )

            self.assertIn("rank-02.json (rank 2): RuntimeError: TPU memory allocation failed", message)
            self.assertIn("resume-rank-03.json (rank 3): ValueError: resume state mismatch", message)
            self.assertIn("launch-failure.json (restart_worker_group): RuntimeError: XLA child process exited", message)
            self.assertIn("final XLA diagnostic", message)
            self.assertNotIn("must not appear", message)

    def test_failure_diagnostics_explain_missing_rank_receipts_and_bound_log_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log = root / "launcher.log"
            log.write_text("x" * 100 + "\nLATEST", encoding="utf-8")

            message = format_canary_failure_diagnostics(
                candidate="m102_primary",
                return_code=1,
                receipt_root=root / "missing",
                launcher_log=log,
                max_log_bytes=20,
            )

            self.assertIn("No rank-scoped FAIL receipt was written", message)
            self.assertIn("LATEST", message)
            self.assertNotIn("x" * 21, message)

    def test_failure_diagnostics_reject_path_traversal_candidate(self):
        with self.assertRaises(ValueError):
            format_canary_failure_diagnostics(
                candidate="..",
                return_code=1,
                receipt_root=".",
            )

    def test_bundle_contains_hashed_results_receipts_checkpoint_and_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "working"
            repo = root / "repo"
            receipt_root = output / "signac_100m_distributed" / "run-001"
            checkpoint = receipt_root / "candidate" / "restart.pt"
            output.mkdir()
            (repo / "signac_100m").mkdir(parents=True)
            (repo / "v5_training").mkdir()
            (repo / "v5_model").mkdir()
            (repo / "notebooks").mkdir()
            (repo / "tools").mkdir()
            (repo / "docs" / "signac_100m").mkdir(parents=True)
            (repo / "signac_100m" / "spec.py").write_text("PARAMETERS = 102\n")
            (repo / "v5_training" / "trainer.py").write_text("checkpoint_every = 200\n")
            (repo / "v5_model" / "core.py").write_text("class Model: pass\n")
            (repo / "notebooks" / "SIGNAC_100M_KAGGLE_TPU_PREFLIGHT.ipynb").write_text("{}\n")
            (repo / "tools" / "signac_100m_preflight.py").write_text("pass\n")
            (repo / "docs" / "signac_100m" / "KAGGLE_TPU_RUNBOOK.md").write_text("runbook\n")
            (output / "signac_100m_static_preflight.json").write_text('{"verdict":"BLOCKED"}\n')
            (output / "signac_100m_all_core_canaries.json").write_text('{"run_id":"old"}\n')
            (output / "signac_100m_post_canary_preflight.json").write_text('{"run_id":"old"}\n')
            checkpoint.parent.mkdir(parents=True)
            checkpoint_bytes = b"checkpoint-payload" * 100
            checkpoint.write_bytes(checkpoint_bytes)
            (receipt_root / "candidate" / "rank-00.json").write_text('{"rank":0}\n')
            (receipt_root / "candidate" / "candidate.launcher.log").write_text(
                "TPU launcher output\n", encoding="utf-8",
            )

            bundle_result = create_kaggle_results_bundle(
                output_root=output,
                repo_root=repo,
                run_id="run-001",
                status="failed",
                receipt_root=receipt_root,
                source_identity={"source_tree_sha256": "ab" * 32},
                error="synthetic canary failure",
            )

            archive_path = Path(bundle_result["path"])
            self.assertTrue(archive_path.is_file())
            self.assertEqual(bundle_result["sha256"], hashlib.sha256(archive_path.read_bytes()).hexdigest())
            self.assertTrue(Path(bundle_result["checksum_path"]).is_file())
            with zipfile.ZipFile(archive_path) as archive:
                self.assertIsNone(archive.testzip())
                names = set(archive.namelist())
                self.assertIn("results/signac_100m_static_preflight.json", names)
                self.assertNotIn("results/signac_100m_all_core_canaries.json", names)
                self.assertNotIn("results/signac_100m_post_canary_preflight.json", names)
                self.assertIn("receipts/run-001/candidate/restart.pt", names)
                self.assertIn("receipts/run-001/candidate/candidate.launcher.log", names)
                self.assertIn("source/signac_100m/spec.py", names)
                self.assertIn("source/v5_model/core.py", names)
                manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
                records = {entry["path"]: entry for entry in manifest["files"]}
                self.assertEqual(manifest["status"], "failed")
                self.assertEqual(manifest["checkpoint_policy"]["every_optimizer_updates"], 200)
                self.assertEqual(
                    records["receipts/run-001/candidate/restart.pt"]["sha256"],
                    hashlib.sha256(checkpoint_bytes).hexdigest(),
                )
                self.assertNotIn("BUNDLE_MANIFEST.json", records)

    def test_rejects_path_traversal_run_ids_and_unknown_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "working"
            output.mkdir()
            repo = root / "repo"
            repo.mkdir()
            for run_id, status in (("../escape", "complete"), ("valid", "pending")):
                with self.subTest(run_id=run_id, status=status):
                    with self.assertRaises(ValueError):
                        create_kaggle_results_bundle(
                            output_root=output,
                            repo_root=repo,
                            run_id=run_id,
                            status=status,
                        )


if __name__ == "__main__":
    unittest.main()
