"""Ledger verification v2 tests: mutation detection, drift, legacy receipts."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "docs" / "arkenstone"))

import verify_ledgers as v  # noqa: E402
from branch_relations import BRANCH_RELATIONS, KNOWN_CEILING  # noqa: E402


def _make_fake_repo(tmp: Path) -> Path:
    """A minimal governed tree with one compact-scheme and one legacy receipt."""
    root = tmp / "repo"
    (root / "docs/arkenstone").mkdir(parents=True)
    (root / "experiments/ARK-X").mkdir(parents=True)
    (root / "docs/arkenstone/EXPERIMENT_LOG.md").write_text(
        "| ARK-X | t | — | EXECUTED | experiments/ARK-X/RESULT.json |\n", encoding="utf-8")
    for name in ("MECHANISM_TOURNAMENT.md", "NEGATIVE_RESULTS.md",
                 "AGI_FEATURE_LEDGER.md", "NOVELTY_REGISTER.md", "README.md"):
        (root / f"docs/arkenstone/{name}").write_text("x", encoding="utf-8")
    compact_payload = {"experiment_id": "X", "results": {"ok": True}}
    compact = dict(compact_payload, receipt_sha256=hashlib.sha256(
        json.dumps(compact_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    (root / "experiments/ARK-X/RESULT.json").write_text(
        json.dumps(compact, indent=2) + "\n", encoding="utf-8")
    legacy_payload = {"experiment_id": "LEGACY", "results": {"ok": True}}
    legacy = dict(legacy_payload, receipt_sha256=hashlib.sha256(
        json.dumps(legacy_payload, sort_keys=True).encode()).hexdigest())
    (root / "experiments/ARK-X/RESULT_legacy.json").write_text(
        json.dumps(legacy, indent=2) + "\n", encoding="utf-8")
    return root


class LedgerVerificationV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.root = _make_fake_repo(self.tmp)

    def test_clean_repo_passes_and_accepts_legacy_receipts(self) -> None:
        result = v.verify(self.root)
        self.assertEqual(result["status"], "PASS", result["problems"])
        schemes = {Path(c["file"]).name: c["scheme"] for c in result["receipt_checks"]}
        self.assertEqual(schemes["RESULT.json"], "compact")
        self.assertEqual(schemes["RESULT_legacy.json"], "default-separators")

    def test_receipt_mutation_detected(self) -> None:
        v.verify(self.root)  # baseline
        receipt = self.root / "experiments/ARK-X/RESULT.json"
        data = json.loads(receipt.read_text(encoding="utf-8"))
        data["results"]["ok"] = False
        receipt.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        result = v.verify(self.root)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("HASH_MISMATCH" in p for p in result["problems"]))

    def test_ledger_drift_detected(self) -> None:
        v.verify(self.root)  # baseline stamps governed hashes
        ledger = self.root / "docs/arkenstone/MECHANISM_TOURNAMENT.md"
        ledger.write_text("tampered", encoding="utf-8")
        result = v.verify(self.root)
        self.assertIn("docs/arkenstone/MECHANISM_TOURNAMENT.md",
                      result["drifted_since_last_check"])

    def test_missing_referenced_artifact_fails(self) -> None:
        v.verify(self.root)
        # log references experiments/ARK-X/RESULT.json which exists; remove it
        (self.root / "experiments/ARK-X/RESULT.json").unlink()
        result = v.verify(self.root)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("missing" in p for p in result["problems"]))

    def test_legacy_receipt_never_rewritten(self) -> None:
        before = (self.root / "experiments/ARK-X/RESULT_legacy.json").read_text(encoding="utf-8")
        v.verify(self.root)
        v.verify(self.root)
        after = (self.root / "experiments/ARK-X/RESULT_legacy.json").read_text(encoding="utf-8")
        self.assertEqual(before, after, "verifier must not mutate history")


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
