"""Unit tests for senora.triquetra_bridge."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from e0_cognition.evaluation_generators import Split, build_evaluation_suite
from senora.evaluator import CasePrediction
from senora.triquetra_bridge import export_triquetra_records, generate_causal_records


class TestSenoraTriquetraBridge(unittest.TestCase):
    def test_export_and_format_causal_records(self) -> None:
        suite = build_evaluation_suite(Split.DEVELOPMENT, seed=101, groups_per_family=1)
        preds = [
            CasePrediction(case_id=c.case_id, raw_output=c.answer, constrained_output=c.answer)
            for c in suite.cases
        ]
        records = generate_causal_records(
            predictions=preds,
            cases=suite.cases,
            checkpoint_sha256="c" * 64,
            treatment_arm="cognition-mixture-15-ce",
            seed=42,
        )

        self.assertEqual(len(records), len(suite.cases))
        self.assertTrue(all(r.evaluator_truth.is_correct for r in records))
        self.assertEqual(records[0].schema, "senora-triquetra-causal-record/v1")

        with tempfile.TemporaryDirectory() as td:
            out_file = Path(td) / "records.jsonl"
            export_triquetra_records(records, out_file)
            self.assertTrue(out_file.exists())
            lines = out_file.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), len(suite.cases))
            parsed = json.loads(lines[0])
            self.assertIn("policy_observation", parsed)
            self.assertIn("evaluator_truth", parsed)


if __name__ == "__main__":
    unittest.main()