"""CPU-only tests for the reviewed K8 implementation gate."""
from __future__ import annotations

import unittest

from bramastra_lab.research.campaigns.readiness import (
    IMPLEMENTATION_READINESS_SCHEMA,
    ImplementationReadinessError,
    implementation_readiness,
    require_implementation_ready,
)


class K8ImplementationReadinessTests(unittest.TestCase):
    def test_gate_is_fail_closed_with_structured_phase_inventory(self) -> None:
        report = implementation_readiness()

        self.assertEqual(report["schema"], IMPLEMENTATION_READINESS_SCHEMA)
        self.assertFalse(report["ready"])
        self.assertEqual(report["blocked_phases"], ["E1", "E2", "E3", "E4", "E5", "E6"])
        self.assertEqual(
            {item["phase"] for item in report["dispositions"]},
            set(report["blocked_phases"]),
        )
        for item in report["dispositions"]:
            self.assertEqual(item["status"], "blocked")
            self.assertTrue(item["blockers"])
            self.assertTrue(item["required_evidence"])

    def test_require_raises_typed_error_and_preserves_report(self) -> None:
        with self.assertRaises(ImplementationReadinessError) as raised:
            require_implementation_ready()

        self.assertFalse(raised.exception.report["ready"])
        self.assertIn("E5", str(raised.exception))

    def test_imports_and_cuda_context_cannot_clear_gate(self) -> None:
        baseline = implementation_readiness()
        contextual = implementation_readiness(source_identity="unrelated-source")

        self.assertFalse(baseline["ready"])
        self.assertFalse(contextual["ready"])
        self.assertEqual(baseline["blocked_phases"], contextual["blocked_phases"])
        self.assertEqual(
            baseline["dispositions"], contextual["dispositions"])
        self.assertEqual(contextual["source_identity"], "unrelated-source")


if __name__ == "__main__":
    unittest.main()
