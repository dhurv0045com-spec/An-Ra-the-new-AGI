"""Receipt integrity checks need no tensor library or accelerator."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unittest


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class BramastraReceiptTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / "artifacts" / "bramastra"

    def test_manifests_bind_exact_sources_datasets_and_matched_arms(self):
        runs = sorted(self.root.glob("*/result.json"))
        self.assertEqual(len(runs), 3)
        for result_path in runs:
            with self.subTest(run=result_path.parent.name):
                folder = result_path.parent
                manifest = json.loads((folder / "manifest.json").read_text())
                result = json.loads(result_path.read_text())
                datasets = json.loads((folder / "datasets.json").read_text())
                self.assertEqual(result["manifest_sha256"], canonical_hash(manifest))
                for name, expected in manifest["source_files_sha256"].items():
                    self.assertEqual(hashlib.sha256((folder / "source_snapshot" / name).read_bytes()).hexdigest(), expected)
                identities = set()
                for name, rows in datasets.items():
                    self.assertEqual(canonical_hash(rows), manifest["dataset_sha256"][name])
                    current = {row["world_id"] for row in rows}
                    self.assertFalse(current.intersection(identities))
                    identities.update(current)
                arms = [json.loads((folder / f"{name}.json").read_text()) for name in ("without_terminal", "with_terminal")]
                self.assertEqual(arms[0]["initial_parameter_sha256"], arms[1]["initial_parameter_sha256"])
                for arm in arms:
                    self.assertEqual(arm["completed_updates"], manifest["steps_per_arm"])
                    self.assertFalse(arm["budget_stopped"])
                    self.assertNotEqual(arm["initial_parameter_sha256"], arm["final_parameter_sha256"])
                    self.assertEqual(arm["counts"]["executed_token_positions"], manifest["steps_per_arm"] * manifest["batch_size"] * manifest["model"]["max_seq"])
                    for key in ("parameters_exact", "optimizer_exact", "sampler_equal"):
                        self.assertTrue(arm["continuation"][key])

    def test_reported_generation_metrics_recompute_from_raw_predictions(self):
        for arm_path in self.root.glob("*/*terminal.json"):
            arm = json.loads(arm_path.read_text())
            for name, report in arm["evaluation"].items():
                with self.subTest(arm=arm_path, split=name):
                    records = report["records"]
                    groups = defaultdict(list)
                    for row in records:
                        correct = row["stop_reason"] == "EOS" and row["prediction"] == row["answer"]
                        self.assertEqual(correct, row["correct"])
                        groups[row["world_id"]].append(correct)
                    self.assertEqual(len(records), report["n_queries"])
                    self.assertEqual(len(groups), report["n_worlds"])
                    self.assertEqual(sum(row["correct"] for row in records) / len(records), report["exact_accuracy"])
                    self.assertEqual(sum(all(values) for values in groups.values()) / len(groups), report["all_queries_correct_rate"])
                    self.assertEqual(dict(Counter(row["stop_reason"] for row in records)), report["stop_histogram"])

    def test_analysis_binds_its_raw_inputs_and_source(self):
        analysis = json.loads((self.root / "analysis.json").read_text())
        self.assertEqual(hashlib.sha256((self.root / "analysis_source.py").read_bytes()).hexdigest(), analysis["analyzer_source_sha256"])
        for run_name, run in analysis["runs"].items():
            folder = self.root / run_name
            self.assertEqual(hashlib.sha256((folder / "result.json").read_bytes()).hexdigest(), run["result_sha256"])
            for arm_name, arm in run["arms"].items():
                self.assertEqual(hashlib.sha256((folder / f"{arm_name}.json").read_bytes()).hexdigest(), arm["raw_receipt_sha256"])


if __name__ == "__main__":
    unittest.main()
