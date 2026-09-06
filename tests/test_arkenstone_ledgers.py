"""Ledger self-verification tests (GAP 2/4 pattern, Arkenstone-side)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "docs" / "arkenstone"))

import verify_ledgers as v  # noqa: E402
from branch_relations import BRANCH_RELATIONS, KNOWN_CEILING  # noqa: E402


class LedgerVerificationTest(unittest.TestCase):
    def test_verification_passes(self) -> None:
        result = v.verify()
        self.assertEqual(result["status"], "PASS", result["problems"])

    def test_experiment_rows_present(self) -> None:
        result = v.verify()
        self.assertGreaterEqual(result["experiment_rows"], 4, "ARK-001..004 rows expected")

    def test_tampered_log_fails(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs/arkenstone").mkdir(parents=True)
            (root / "experiments").mkdir()
            log = "| ARK-999 | fake | — | EXECUTED | experiments/ARK-999/NOPE.json |\n"
            (root / "docs/arkenstone/EXPERIMENT_LOG.md").write_text(log, encoding="utf-8")
            (root / "docs/arkenstone/README.md").write_text("x", encoding="utf-8")
            result = v.verify(root)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(any("ARK-999" in p for p in result["problems"]))


class BranchRelationsTest(unittest.TestCase):
    def test_every_branch_has_all_fields(self) -> None:
        for name, entry in BRANCH_RELATIONS.items():
            for field in ("proves", "done_looks_like", "connects_to"):
                self.assertIn(field, entry, f"{name} missing {field}")
                self.assertTrue(entry[field].strip(), f"{name}.{field} empty")

    def test_ceiling_recorded(self) -> None:
        self.assertIn("PROPOSES", KNOWN_CEILING)


if __name__ == "__main__":
    unittest.main()
