"""B01 focused tests: the CLI parses without side effects and inspect reports
separate readiness dimensions."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TINY_CONFIG = {
    "model": {"profile": "tiny"},
    "training": {"max_updates": 5},
}

REQUIRED_SUBCOMMANDS = ["inspect", "prepare-data", "train", "resume", "infer",
                        "evaluate", "package"]


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "bramastra_lab.research.cli", *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
    )


class CliHelpTests(unittest.TestCase):
    def test_help_is_side_effect_free(self) -> None:
        completed = run_cli("--help")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for command in REQUIRED_SUBCOMMANDS:
            self.assertIn(command, completed.stdout)

    def test_each_subcommand_help_works(self) -> None:
        for command in REQUIRED_SUBCOMMANDS:
            completed = run_cli(command, "--help")
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cli_module_import_does_not_import_torch(self) -> None:
        probe = subprocess.run(
            [sys.executable, "-c",
             "import sys; import bramastra_lab.research.cli; "
             "assert 'torch' not in sys.modules, 'CLI import pulled in torch'; print('ok')"],
            capture_output=True, text=True, cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": REPO_ROOT},
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertIn("ok", probe.stdout)


class InspectTests(unittest.TestCase):
    def write_config(self, directory: str, raw: dict) -> str:
        path = os.path.join(directory, "config.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(raw, handle)
        return path

    def test_inspect_valid_config_without_data_is_data_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self.write_config(tmp, TINY_CONFIG)
            completed = run_cli("inspect", "--config", config_path, "--json")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(report["config_status"], "CONFIG_VALID")
            self.assertEqual(report["data_status"], "DATA_NOT_READY")
            self.assertEqual(report["device_status"], "DEVICE_UNVERIFIED")
            self.assertEqual(report["parameter_count"], 117_312 + 2 * 64)
            self.assertNotEqual(report["overall"], "TRAINING_READY_FOR_DECLARED_DEVICE")

    def test_inspect_invalid_config_reports_config_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self.write_config(tmp, {"model": {"profile": "tiny"}, "junk": {}})
            completed = run_cli("inspect", "--config", config_path, "--json")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(report["config_status"], "CONFIG_INVALID")
            self.assertIn("junk", report["detail"])

    def test_inspect_reports_readiness_honestly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self.write_config(tmp, TINY_CONFIG)
            completed = run_cli("inspect", "--config", config_path, "--json")
            report = json.loads(completed.stdout)
            # A valid config with no corpus and no device check is never "ready".
            self.assertEqual(report["overall"], "NOT_READY_FOR_DECLARED_DEVICE")


if __name__ == "__main__":
    unittest.main()
