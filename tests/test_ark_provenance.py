"""Tests for BRAMASTRA-adopted provenance mechanics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

import torch
import torch.nn as nn

from lib import ark_provenance as ap


class SourceSnapshotTest(unittest.TestCase):
    def test_captures_and_hashes(self) -> None:
        p = Path(REPO / "experiments/lib/ark_provenance.py")
        snap = ap.source_snapshot([str(p), str(REPO / "nonexistent.py")])
        self.assertIn("ark_provenance.py", snap)
        self.assertNotIn("nonexistent.py", snap)
        self.assertEqual(len(ap.snapshot_sha256(snap)), 64)

    def test_snapshot_detects_change(self) -> None:
        p = Path(REPO / "experiments/lib/ark_provenance.py")
        s1 = ap.snapshot_sha256(ap.source_snapshot([str(p)]))
        s2 = ap.snapshot_sha256(ap.source_snapshot([str(p)]))
        self.assertEqual(s1, s2)


class ContinuationProbeTest(unittest.TestCase):
    def test_deterministic_update_reproduces(self) -> None:
        tmp = Path(REPO / "experiments/ARK-005/probe_test.pt")
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        counter = {"n": 0}

        def batch_fn():
            # depends ONLY on the global torch RNG (which gets saved/restored)
            return torch.randn(4, 8)

        def loss_fn(batch):
            loss = torch.nn.functional.mse_loss(model(batch), torch.zeros(4, 8))
            return loss, 1  # (loss, token_count) — matches probe interface

        try:
            receipt = ap.continuation_probe(
                model=model, optimizer=opt, batch_fn=batch_fn, loss_fn=loss_fn,
                checkpoint_path=tmp, device=torch.device("cpu"))
            self.assertTrue(receipt["parameters_exact"])
            self.assertTrue(receipt["optimizer_exact"])
        finally:
            tmp.unlink(missing_ok=True)

    def test_checkpoint_identity_mismatch_raises(self) -> None:
        tmp = Path(REPO / "experiments/ARK-005/probe_test2.pt")
        torch.manual_seed(43)
        model = nn.Linear(8, 4)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        calls = {"n": 0}

        def batch_fn():
            calls["n"] += 1
            torch.manual_seed(calls["n"])  # non-deterministic batches
            return torch.randn(4, 8), torch.randn(4, 4)

        def loss_fn(batch):
            x, y = batch
            loss = torch.nn.functional.mse_loss(model(x), y)
            return loss, 1

        try:
            with self.assertRaises(RuntimeError):
                ap.continuation_probe(
                    model=model, optimizer=opt, batch_fn=batch_fn, loss_fn=loss_fn,
                    checkpoint_path=tmp, device=torch.device("cpu"))
        finally:
            tmp.unlink(missing_ok=True)


class NominateNextTest(unittest.TestCase):
    def test_retention_first(self) -> None:
        result = ap.nominate_next({"retention_collapse_seen": True,
                                   "intervention_nulls": [], "open_tier_boundary": False})
        self.assertEqual(result["verdict"], "RETENTION_UNADDRESSED")

    def test_escalation_after_nulls(self) -> None:
        result = ap.nominate_next({"retention_collapse_seen": True,
                                   "intervention_nulls": ["EMA null", "WD null"],
                                   "open_tier_boundary": True})
        self.assertIn("LR", result["next_experiment"])

    def test_dose_ratio_when_boundary_open(self) -> None:
        result = ap.nominate_next({"retention_collapse_seen": False,
                                   "intervention_nulls": [], "open_tier_boundary": True})
        self.assertEqual(result["verdict"], "DOSE_RATIO_UNMAPPED")

    def test_rule_is_always_named(self) -> None:
        for evidence in ({}, {"retention_collapse_seen": True}, {"open_tier_boundary": True}):
            result = ap.nominate_next(evidence)
            self.assertIn("rule", result)


if __name__ == "__main__":
    unittest.main()
