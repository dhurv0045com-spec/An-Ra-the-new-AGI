from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from e2_architecture.scoring_policy_fixture import _neutral_anchors


class E2ScoringPolicyFixtureTests(unittest.TestCase):
    def test_committed_fixture_receipt_is_source_bound_and_passes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        receipt = json.loads(
            (root / "artifacts/e2/scoring_policy_fixture.json").read_text(encoding="utf-8")
        )
        normalized = (root / "e2_architecture/scoring_policy_fixture.py").read_text(encoding="utf-8").replace("\r\n", "\n")
        self.assertEqual(
            receipt["implementation_sha256"],
            hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(receipt["status"], "PASS_FIXTURE_COMPILATION")
        self.assertTrue(all(receipt["checks"].values()))
        self.assertNotEqual(
            receipt["development"]["fixture_sha256"],
            receipt["fresh"]["fixture_sha256"],
        )
        for split in ("development", "fresh"):
            self.assertEqual(receipt[split]["groups"], 256)
            self.assertTrue(all(receipt[split]["checks"].values()))
            self.assertLessEqual(
                max(receipt[split]["surface_family_counts"].values())
                - min(receipt[split]["surface_family_counts"].values()),
                1,
            )

    def test_neutral_anchor_panels_are_deterministic_and_disjoint(self) -> None:
        anchors = _neutral_anchors("a" * 64, 16_384)
        self.assertEqual(anchors, _neutral_anchors("a" * 64, 16_384))
        self.assertEqual((len(anchors), len(anchors[0]), len(anchors[1])), (2, 4, 4))
        self.assertTrue(set(anchors[0]).isdisjoint(anchors[1]))


if __name__ == "__main__":
    unittest.main()
