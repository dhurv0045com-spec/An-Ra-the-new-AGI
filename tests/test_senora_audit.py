"""Unit tests for senora.audit."""

from __future__ import annotations

import unittest

from senora.audit import run_audit


class TestSenoraAudit(unittest.TestCase):
    def test_audit_report_structure_and_execution_map(self) -> None:
        report = run_audit()
        self.assertEqual(report.schema, "senora-audit-report/v4")
        self.assertEqual(report.branch, "senora")
        self.assertEqual(report.branch_origin, "esoes@85f44b7")

        # 11 software and measurement gates must PASS
        self.assertEqual(report.execution_map["MODEL"], "PASS")
        self.assertEqual(report.execution_map["CE"], "PASS")
        self.assertEqual(report.execution_map["QSWAP"], "PASS")
        self.assertEqual(report.execution_map["OPTIMIZER"], "PASS")
        self.assertEqual(report.execution_map["DATA_READER"], "PASS")
        self.assertEqual(report.execution_map["CHECKPOINT"], "PASS")
        self.assertEqual(report.execution_map["REMOTE_CANARY"], "READY_BUT_UNEXECUTED")
        self.assertEqual(report.execution_map["REMOTE_RUNNER"], "READY_BUT_UNEXECUTED")
        self.assertEqual(report.execution_map["DEV_EVALUATION"], "PASS")
        self.assertEqual(report.execution_map["STRUCTURAL_OOD_DEV"], "PASS")
        self.assertEqual(report.execution_map["FRESH_FIREWALL"], "PASS")
        self.assertEqual(report.execution_map["STATISTICS"], "PASS")
        self.assertEqual(report.execution_map["RESULT_CLASSIFIER"], "PASS")
        self.assertEqual(report.execution_map["M102_GATE"], "BLOCKED")

        self.assertFalse(report.ready_for_remote_launch)


if __name__ == "__main__":
    unittest.main()