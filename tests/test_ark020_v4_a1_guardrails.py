"""Post-red-team edge tests for ARK-020 V4 durability A1.1 guardrails."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
V4_DIR = HERE.parent / "experiments" / "ARK-020-V4"
sys.path.insert(0, str(V4_DIR))
sys.path.insert(0, str(HERE.parent / "experiments" / "ARK-019"))
sys.path.insert(0, str(HERE.parent / "experiments" / "ARK-018"))

import ark020_v4_core as C  # noqa: E402
import ark020_v4_durability as A1  # noqa: E402
import ark020_v4_a1_guardrails as G  # noqa: E402
import run_ark020_v4 as R  # noqa: E402

A1.install(R)
G.install(R)


class TestFinalizeCheckpointScan(unittest.TestCase):
    def _checkpoint(self, root: Path) -> Path:
        p = root / "matched_sets" / "p31801_b429001" / "GUARDIAN_HYBRID"
        p.mkdir(parents=True, exist_ok=True)
        cp = p / "RESUME.pt"
        torch.save({
            "schema": "arkenstone-ark020-v4-arm-ckpt/v1",
            "parent_seed": 31801,
            "b_order_seed": 429001,
            "c_order_seed": 429003,
            "d_order_seed": 429005,
            "arm": "GUARDIAN_HYBRID",
            "dose_b": 8,
            "parent_sha": "parent",
            "task_hash": "tasks",
            "cap16x": 1.0,
            "phase_idx": len(C.PHASES),
            "phase": "FINALIZE",
            "phase_step": 0,
            "registry": {"capabilities": {}},
            "controller": C.initial_controller("GUARDIAN_HYBRID"),
            "counters": {},
            "phase_confirm": {"B": 100, "C": 100, "D": 100},
            "global_confirm": {"B": 100, "C": 2100, "D": 3600},
            "b_streaks": {"B": 0, "C": 0, "D": 0},
        }, cp)
        return cp

    def test_finalize_sentinel_is_resumable_without_frozen_global_step_crash(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(R, "OUT", Path(td)), \
             mock.patch.object(A1, "_validate_checkpoint_receipts", return_value=[]):
            self._checkpoint(Path(td))
            info = R.resume_scan(drive_ok=True)
        self.assertEqual(info["SAFE_ACTION"], "RESUME")
        self.assertEqual(info["CHECKPOINT_IDENTITY"], "PASS")
        self.assertEqual(info["ACTIVE_PHASE"], "FINALIZE")
        self.assertEqual(info["GLOBAL_STEP"], C.CONTINUATION_HORIZON)
        self.assertEqual(info["RESUME_MEANING"], "FINALIZE_ONLY_NO_MORE_TRAINING_UPDATES")

    def test_finalize_identity_failure_stops(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(R, "OUT", Path(td)), \
             mock.patch.object(A1, "_validate_checkpoint_receipts", return_value=["bad receipt"]):
            self._checkpoint(Path(td))
            info = R.resume_scan(drive_ok=True)
        self.assertEqual(info["SAFE_ACTION"], "STOP — CHECKPOINT IDENTITY FAILURE")
        self.assertEqual(info["HARDENED_GATE"], "FAIL")

    def test_multiple_checkpoints_fail_closed_before_scanner(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(R, "OUT", Path(td)):
            root = Path(td)
            self._checkpoint(root)
            p2 = root / "matched_sets" / "p31902_b429002" / "PLASTIC_HIGH"
            p2.mkdir(parents=True)
            torch.save({"schema": "x"}, p2 / "RESUME.pt")
            info = R.resume_scan(drive_ok=True)
        self.assertEqual(info["SAFE_ACTION"], "STOP — CHECKPOINT IDENTITY FAILURE")
        self.assertTrue(any("multiple active checkpoints" in e for e in info["HARDENED_ERRORS"]))


class TestOperatorIdentityAndLock(unittest.TestCase):
    def test_executable_identity_binds_guardrail_source(self):
        ident = A1.executable_identity(R)
        self.assertIn("durability_guardrails", ident["files"])
        self.assertEqual(ident["implementation_revision"], "A1.1")

    def test_lock_is_v4_without_transient_v3_writer(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(R, "OUT", Path(td)), \
             mock.patch.object(R, "_pinned_commit", return_value="deadbeef"):
            body = R.write_lock()
            disk = json.loads((Path(td) / "CAMPAIGN_LOCK.json").read_text())
        self.assertEqual(body["experiment"], "ARK-020-V4")
        self.assertEqual(disk["experiment"], "ARK-020-V4")
        self.assertEqual(disk["implementation_revision"], "A1.1")


class TestCompletedArmImmutability(unittest.TestCase):
    def test_completed_result_bypasses_boundary_migration(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(R, "OUT", Path(td)):
            ps, bs, arm = 31801, 429001, "PLASTIC_HIGH"
            ad, rp, cp, _pp = R.arm_paths(ps, bs, arm)
            ad.mkdir(parents=True)
            parent = {"model": {"w": torch.tensor([1.0])}}
            parent_sha = R.V3.state_hash(parent["model"])
            R.savej(rp, {
                "schema": "arkenstone-ark020-v4-arm/v1",
                "status": "COMPLETE",
                "parent_seed": ps,
                "b_order_seed": bs,
                "c_order_seed": 429003,
                "d_order_seed": 429005,
                "arm": arm,
                "dose_b": 8,
                "cap16x": 1.0,
                "parent_sha": parent_sha,
            })
            torch.save({"phase_idx": 0, "phase": "B", "phase_step": 2000}, cp)
            with mock.patch.object(A1, "migrate_boundary_checkpoint",
                                   side_effect=AssertionError("must not migrate completed arm")):
                got = R.run_set_arm(ps, bs, arm, 8, parent, 1.0, {}, {}, {},
                                    torch.device("cpu"), 0.0)
        self.assertEqual(got["status"], "COMPLETE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
