"""B11 integration tests: the real command path end to end.

Non-learned parts (inspect, prepare-data determinism, package refusals) run
always. The full learned path — train -> checkpoint -> fresh-process resume
-> infer -> evaluate -> package — performs optimizer updates and is therefore
gated behind ``BRAMASTRA_LEARNED_CHECKS=1`` under the cumulative smoke budget.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEARNED_CHECKS = os.environ.get("BRAMASTRA_LEARNED_CHECKS") == "1"
LEARNED_REASON = ("learned checks deferred: owner has not authorized optimizer updates "
                  "(set BRAMASTRA_LEARNED_CHECKS=1 to run)")
FIXTURES = os.path.join(REPO_ROOT, "engineering", "reports", "B2", "fixtures")

SMOKE_CONFIG = {
    "model": {"profile": "tiny"},
    "training": {"seed": 1234, "learning_rate": 0.02, "max_updates": 8,
                 "pair_loss_weight": 0.0},
}


def run_cli(*args: str, learned: bool = False) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": REPO_ROOT}
    if learned:
        env["BRAMASTRA_LEARNED_CHECKS"] = "1"
    return subprocess.run([sys.executable, "-m", "bramastra_lab.research.cli", *args],
                          capture_output=True, text=True, cwd=REPO_ROOT, env=env)


def write_config(directory: str, overrides: dict | None = None) -> str:
    config = json.loads(json.dumps(SMOKE_CONFIG))
    if overrides:
        for section, values in overrides.items():
            config.setdefault(section, {}).update(values)
    path = os.path.join(directory, "smoke-config.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)
    return path


class PrepareDataTests(unittest.TestCase):
    def test_prepare_data_is_deterministic_and_inventoried(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_config(tmp)
            out_a = os.path.join(tmp, "prepared-a")
            out_b = os.path.join(tmp, "prepared-b")
            first = run_cli("prepare-data", "--manifest", os.path.join(FIXTURES, "manifest.json"),
                            "--out", out_a, "--config", config_path)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = run_cli("prepare-data", "--manifest", os.path.join(FIXTURES, "manifest.json"),
                             "--out", out_b, "--config", config_path)
            self.assertEqual(second.returncode, 0, second.stderr)
            report_a = json.loads(first.stdout)
            report_b = json.loads(second.stdout)
            self.assertEqual(report_a["identity"], report_b["identity"])
            self.assertEqual(report_a["split_inventory"]["training"]["examples"], 8)
            self.assertEqual(report_a["split_inventory"]["development"]["examples"], 4)
            self.assertGreater(report_a["split_inventory"]["training"]["supervised_targets"], 0)
            self.assertTrue(os.path.exists(os.path.join(out_a, "rows-training.jsonl")))

    def test_prepare_data_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_config(tmp)
            out = os.path.join(tmp, "prepared")
            run_cli("prepare-data", "--manifest", os.path.join(FIXTURES, "manifest.json"),
                    "--out", out, "--config", config_path)
            again = run_cli("prepare-data", "--manifest", os.path.join(FIXTURES, "manifest.json"),
                            "--out", out, "--config", config_path)
            self.assertEqual(again.returncode, 2)
            self.assertIn("already exists", again.stderr)

    def test_inspect_reports_data_not_ready_without_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_config(tmp)
            completed = run_cli("inspect", "--config", config_path, "--json")
            report = json.loads(completed.stdout)
            self.assertEqual(report["config_status"], "CONFIG_VALID")
            self.assertEqual(report["data_status"], "DATA_NOT_READY")

    def test_train_refuses_unknown_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_config(tmp)
            missing = run_cli("train", "--config", config_path,
                              "--data", os.path.join(tmp, "nowhere"),
                              "--run-dir", os.path.join(tmp, "run"), "--max-updates", "2")
            self.assertEqual(missing.returncode, 2)
            self.assertIn("prepared.json", missing.stderr)


@unittest.skipUnless(LEARNED_CHECKS, LEARNED_REASON)
class IntegratedLoopWiringTests(unittest.TestCase):
    """B2.1 wiring: controller, replay, pair objective — real updates."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _prepare(self, config_path):
        prepared = os.path.join(self.tmp.name, "prepared")
        result = run_cli("prepare-data", "--manifest", os.path.join(FIXTURES, "manifest.json"),
                         "--out", prepared, "--config", config_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        return prepared

    def test_controller_wired_into_training(self) -> None:
        with self.tmp:
            config = json.loads(json.dumps(SMOKE_CONFIG))
            config["controller"] = {
                "mode": "evidence_driven", "controller_pool_id": "controller",
                "formation_threshold": 0.95, "stabilize_window": 2,
                "controller_eval_every": 2, "cooldown_updates": 1,
                "collapse_confirmation_evaluations": 2,
                "recovery_confirmation_evaluations": 2}
            config["training"]["max_updates"] = 6
            config_path = os.path.join(self.tmp.name, "config.json")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)
            prepared = self._prepare(config_path)
            run_dir = os.path.join(self.tmp.name, "run-controller")
            trained = run_cli("train", "--config", config_path, "--data", prepared,
                              "--run-dir", run_dir, "--max-updates", "6", "--smoke",
                              learned=True)
            self.assertEqual(trained.returncode, 0, trained.stderr)
            # Preflight recorded and fully passed.
            preflight = json.loads(open(os.path.join(run_dir, "preflight.json"),
                                        encoding="utf-8").read())
            self.assertTrue(preflight["ready"])
            self.assertIn("controller_split",
                          [g["id"] for g in preflight["gates"]])
            # Controller boundaries were evaluated and recorded.
            events = [json.loads(line) for line in
                      open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8")]
            boundaries = [event for event in events if event["event"] == "controller_boundary"]
            self.assertGreaterEqual(len(boundaries), 2)
            self.assertEqual(boundaries[0]["decision"]["state"], "EXPAND")
            # Controller state persisted into the final checkpoint.
            from bramastra_lab.research.config import BuildConfig
            from bramastra_lab.research.runtime import checkpoint as ckpt

            payload, _ = ckpt.load_checkpoint(run_dir)
            self.assertIsNotNone(payload["controller_state"])
            restored = payload["controller_state"]
            self.assertTrue(restored["acquiring_family"])
            self.assertEqual(restored["state"], "EXPAND")

    def test_replay_wired_into_training(self) -> None:
        with self.tmp:
            config = json.loads(json.dumps(SMOKE_CONFIG))
            config["replay"] = {"enabled": True, "proportion": 0.5,
                                "family_weights": {"switch-world": 1.0,
                                                   "inventory-world": 1.0,
                                                   "program-lab": 1.0}}
            config["training"]["max_updates"] = 4
            config_path = os.path.join(self.tmp.name, "config.json")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)
            prepared = self._prepare(config_path)
            collected = run_cli("collect", "--environments", "switch-world,inventory-world",
                                "--ledger", os.path.join(prepared, "episodes.jsonl"),
                                "--episodes", "2", "--policy", "fixed",
                                "--budget", "6", "--seed", "5")
            self.assertEqual(collected.returncode, 0, collected.stderr)
            collection = json.loads(collected.stdout)
            self.assertEqual(collection["episodes"], 4)
            run_dir = os.path.join(self.tmp.name, "run-replay")
            trained = run_cli("train", "--config", config_path, "--data", prepared,
                              "--run-dir", run_dir, "--max-updates", "4", "--smoke",
                              learned=True)
            self.assertEqual(trained.returncode, 0, trained.stderr)
            report = json.loads(trained.stdout)
            reconciliation = report["replay_reconciliation"]
            self.assertEqual(reconciliation["designated_updates"], 2)
            self.assertGreaterEqual(reconciliation["executed_replay_batches"], 1)
            self.assertTrue(reconciliation["planned"] >= reconciliation["consumed"])

    def test_pair_loss_wired_into_training(self) -> None:
        with self.tmp:
            config = json.loads(json.dumps(SMOKE_CONFIG))
            config["training"]["pair_loss_weight"] = 0.5
            config["training"]["max_updates"] = 4
            config_path = os.path.join(self.tmp.name, "config.json")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)
            prepared = self._prepare(config_path)
            run_dir = os.path.join(self.tmp.name, "run-pair")
            trained = run_cli("train", "--config", config_path, "--data", prepared,
                              "--run-dir", run_dir, "--max-updates", "4", "--smoke",
                              learned=True)
            self.assertEqual(trained.returncode, 0, trained.stderr)
            report = json.loads(trained.stdout)
            self.assertEqual(report["updates"], 4)

    def test_collection_records_failure_categories(self) -> None:
        with self.tmp:
            collected = run_cli("collect", "--environments", "program-lab",
                                "--ledger", os.path.join(self.tmp.name, "episodes.jsonl"),
                                "--episodes", "1", "--policy", "failed-baseline",
                                "--budget", "6", "--seed", "3")
            self.assertEqual(collected.returncode, 0, collected.stderr)
            report = json.loads(collected.stdout)
            self.assertEqual(report["episodes"], 1)
            self.assertEqual(report["successes"], 0)
            self.assertGreater(len(report["failures_by_category"]), 0)
            from bramastra_lab.research.experience.ledger import ExperienceLedger

            ledger = ExperienceLedger(os.path.join(self.tmp.name, "episodes.jsonl"))
            entries = ledger.read_all()
            self.assertEqual(len(entries), 1)
            receipt = entries[0][1]
            self.assertEqual(receipt.success, False)
            self.assertEqual(receipt.notes["failure_category"], "wrong_world_prediction")


@unittest.skipUnless(LEARNED_CHECKS, LEARNED_REASON)
class IntegratedSmokePathTests(unittest.TestCase):
    """The full integrated path with the real trainer (learned smoke)."""

    def test_prepare_train_resume_infer_evaluate_package(self) -> None:
        from bramastra_lab.research.runtime.smoke import SessionLedger

        ledger = SessionLedger(os.path.join(REPO_ROOT, "engineering", "reports", "B2",
                                            "SESSION_LEDGER.json"))
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_config(tmp)
            prepared = os.path.join(tmp, "prepared")
            run_dir = os.path.join(tmp, "run-smoke")

            prepared_result = run_cli("prepare-data",
                                      "--manifest", os.path.join(FIXTURES, "manifest.json"),
                                      "--out", prepared, "--config", config_path)
            self.assertEqual(prepared_result.returncode, 0, prepared_result.stderr)

            trained = run_cli("train", "--config", config_path, "--data", prepared,
                              "--run-dir", run_dir, "--max-updates", "8", "--smoke",
                              learned=True)
            self.assertEqual(trained.returncode, 0, trained.stderr)
            train_report = json.loads(trained.stdout)
            self.assertEqual(train_report["updates"], 8)
            first_checkpoint = train_report["checkpoint_id"]

            resumed = run_cli("resume", "--run-dir", run_dir, "--max-updates", "4",
                              "--expect-parent", first_checkpoint, learned=True)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            resume_report = json.loads(resumed.stdout)
            self.assertEqual(resume_report["updates_before"], 8)
            self.assertEqual(resume_report["updates_after"], 12)

            request_path = os.path.join(tmp, "request.json")
            with open(request_path, "w", encoding="utf-8") as handle:
                json.dump({"prompt_events": [["goal", {"question": "2+2?"}]]}, handle)
            inference_result = run_cli("infer", "--config", config_path,
                                       "--checkpoint", run_dir,
                                       "--input", request_path, learned=True)
            self.assertEqual(inference_result.returncode, 0, inference_result.stderr)
            inference_report = json.loads(inference_result.stdout)
            self.assertIn("answer", inference_report["generation"])
            self.assertIn("complete", inference_report["generation"])

            eval_path = os.path.join(tmp, "evaluation")
            os.makedirs(eval_path)
            eval_report_path = os.path.join(eval_path, "report.json")
            evaluated = run_cli("evaluate", "--checkpoint", run_dir, "--config", config_path,
                                "--data", prepared, "--split", "development",
                                "--out", eval_report_path, learned=True)
            self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            evaluation = json.loads(evaluated.stdout)
            self.assertEqual(evaluation["metrics"]["count"], 4)
            self.assertTrue(evaluation["metrics"]["recomputed"])

            package_path = os.path.join(tmp, "package.json")
            packaged = run_cli("package", "--run-dir", run_dir, "--out", package_path)
            self.assertEqual(packaged.returncode, 0, packaged.stderr)
            package_manifest = json.loads(open(package_path, encoding="utf-8").read())
            self.assertNotEqual(package_manifest["latest_checkpoint"]["checkpoint_id"],
                                first_checkpoint)
            self.assertFalse(package_manifest["readiness"]["scientific_qualification"])
            self.assertIn("DATA_NOT_READY", package_manifest["readiness"]["real_data_supply"])
            self.assertGreater(package_manifest["smoke_ledger"]["cpu_optimizer_updates"], 0)

    def test_train_without_updates_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_config(tmp)
            trained = run_cli("train", "--config", config_path,
                              "--data", os.path.join(tmp, "x"),
                              "--run-dir", os.path.join(tmp, "run"), learned=True)
            self.assertEqual(trained.returncode, 2)
            # Fail-closed preflight reports every failing gate at once.
            self.assertIn("preflight failed", trained.stderr)
            self.assertIn("prepared_data", trained.stderr)


if __name__ == "__main__":
    unittest.main()
