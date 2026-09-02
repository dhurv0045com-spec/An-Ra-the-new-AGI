"""Unit tests for senora.audit."""

from __future__ import annotations

import unittest

from senora.audit import run_audit


class TestSenoraAudit(unittest.TestCase):
    def test_audit_report_structure_and_binary_gates(self) -> None:
        report = run_audit()
        self.assertEqual(report.schema, "senora-audit-report/v2")
        self.assertEqual(report.branch, "senora")
        self.assertEqual(report.branch_origin, "esoes@85f44b7")

        # 11 software and measurement gates must PASS
        self.assertEqual(report.binary_gates["MODEL_CONSTRUCTOR"], "PASS")
        self.assertEqual(report.binary_gates["REAL_TRAIN_STEP"], "PASS")
        self.assertEqual(report.binary_gates["REAL_CE"], "PASS")
        self.assertEqual(report.binary_gates["REAL_QSWAP"], "PASS")
        self.assertEqual(report.binary_gates["REAL_DATA_READER"], "PASS")
        self.assertEqual(report.binary_gates["CHECKPOINT_RESTORE"], "PASS")
        self.assertEqual(report.binary_gates["REMOTE_RUNNER"], "PASS")
        self.assertEqual(report.binary_gates["REMOTE_CANARY_SPEC"], "PASS")
        self.assertEqual(report.binary_gates["DEVELOPMENT_EVALUATOR"], "PASS")
        self.assertEqual(report.binary_gates["FRESH_FIREWALL"], "PASS")
        self.assertEqual(report.binary_gates["STATISTICAL_PROMOTION"], "PASS")

        # External gates are BLOCKED
        self.assertEqual(report.binary_gates["DATA_MANIFEST"], "BLOCKED")
        self.assertEqual(report.binary_gates["SEALED_CUSTODY"], "BLOCKED")

        self.assertFalse(report.ready_for_remote_launch)


if __name__ == "__main__":
    unittest.main()