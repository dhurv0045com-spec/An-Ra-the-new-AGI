"""Verified status: auditors, freshness model, derived readiness (M21-M23)."""

from __future__ import annotations

import unittest
from pathlib import Path

from anra_v5.status import (
    READINESS_DEPS,
    build_status,
    classify_freshness,
)

ROOT = Path(__file__).resolve().parents[1]


class FreshnessTests(unittest.TestCase):
    def test_head_receipt_is_exact(self) -> None:
        import subprocess

        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        verdict = classify_freshness(ROOT, head)
        self.assertEqual(verdict["state"], "EXACT_HEAD_TESTED")

    def test_missing_and_bogus_receipts(self) -> None:
        self.assertEqual(
            classify_freshness(ROOT, None)["state"], "NO_TEST_EVIDENCE"
        )
        self.assertEqual(
            classify_freshness(ROOT, "0" * 40)["state"], "STALE_IMPLEMENTATION_TESTS"
        )


class StatusVerifierTests(unittest.TestCase):
    def test_status_derives_from_verifiers(self) -> None:
        status = build_status(repo_root=ROOT)
        self.assertNotIn("test file contains", json_dumps(status))
        verified = status["verified"]
        for key in ("UNIT_TEST_BASELINE", "EVALUATION_CAUSALITY", "EXACT_RESUME"):
            self.assertIn("via", verified[key])
            self.assertIn(verified[key]["status"], {
                "VERIFIED", "STALE", "FAILED", "NOT_DEMONSTRATED", "NO_TEST_EVIDENCE",
            })
        # Red-team #19: no PASS may rest on code existence.
        for key, entry in verified.items():
            if entry["status"] == "VERIFIED":
                self.assertIn("via", entry)
                self.assertTrue(entry["via"])

    def test_readiness_graph_is_derived(self) -> None:
        status = build_status(repo_root=ROOT)
        nodes = status["readiness"]
        for name, spec in READINESS_DEPS.items():
            self.assertIn(name, nodes)
            self.assertEqual(nodes[name].get("requires"), spec["requires"])
        p35a = nodes["P35A_EXECUTION_READY"]
        self.assertIn("DATASET_QUALIFIED", p35a["via"])
        self.assertEqual(p35a["status"], "BLOCKED")
        self.assertIn(nodes["DATASET_QUALIFIED"]["status"], {"BLOCKED"})


def json_dumps(value: object) -> str:
    import json

    return json.dumps(value, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
