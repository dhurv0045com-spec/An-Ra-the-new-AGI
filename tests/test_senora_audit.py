"""Unit tests for senora.audit."""

from __future__ import annotations

import unittest

from senora.audit import run_audit


class TestSenoraAudit(unittest.TestCase):
    def test_run_audit_structure(self) -> None:
        report = run_audit()
        self.assertEqual(report.schema, "senora-audit-report/v1")
        self.assertEqual(report.branch, "senora")
        self.assertEqual(report.branch_origin, "esoes@85f44b7")

        required_categories = {
            "data_pipeline",
            "trainer",
            "checkpoint_resume",
            "evaluator",
            "cognition_benchmark",
            "experiment_identity",
            "statistical_protocol",
            "remote_launch_readiness",
        }
        self.assertEqual(set(report.readiness_scores.keys()), required_categories)
        for cat, score in report.readiness_scores.items():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)

        self.assertAlmostEqual(
            report.mean_readiness_score,
            sum(report.readiness_scores.values()) / len(report.readiness_scores),
        )

        # Check blocker categories
        blockers = report.blockers
        self.assertIn("software", blockers)
        self.assertIn("data", blockers)
        self.assertIn("measurement", blockers)
        self.assertIn("compute", blockers)
        self.assertIn("external_custody", blockers)

        # Software should have no blockers (fully implemented), but compute/data should
        self.assertEqual(len(blockers["software"]), 0)
        self.assertGreater(len(blockers["compute"]), 0)
        self.assertGreater(len(blockers["data"]), 0)
        self.assertFalse(report.ready_for_remote_launch)


if __name__ == "__main__":
    unittest.main()