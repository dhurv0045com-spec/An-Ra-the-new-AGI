"""Regression tests for ARK-020 V4 durability amendment A1."""
from __future__ import annotations

import json
import subprocess
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
import run_ark020_v4 as R  # noqa: E402

A1.install(R)
HARDENED = V4_DIR / "run_ark020_v4_hardened.py"


def registry(*caps):
    reg = C.init_registry()
    for i, cap in enumerate(caps):
        C.register_capability(reg, cap, i * 25)
    return reg


def boundary_payload(phase: str, step: int, arm: str = "GUARDIAN_HYBRID"):
    idx = C.PHASES.index(phase)
    return {
        "schema": "arkenstone-ark020-v4-arm-ckpt/v1",
        "phase_idx": idx,
        "phase": phase,
        "phase_step": step,
        "arm": arm,
        "registry": registry("A", *(["B"] if phase in ("C", "D") else []),
                             *(["C"] if phase == "D" else [])),
        "controller": C.initial_controller(arm),
    }


class TestHardenedCLI(unittest.TestCase):
    """The exact operator entry point must execute, not only internal functions."""

    def test_scan_drive_unavailable_exact_cli(self):
        proc = subprocess.run(
            [sys.executable, str(HARDENED), "--mode", "scan", "--drive-ok", "False"],
            capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-1000:])
        marker = "@@SCAN_JSON@@"
        line = next((x for x in proc.stdout.splitlines() if x.startswith(marker)), None)
        self.assertIsNotNone(line)
        info = json.loads(line[len(marker):])
        self.assertEqual(info["SAFE_ACTION"], "STOP — DRIVE UNAVAILABLE")

    def test_unknown_arg_rejected(self):
        proc = subprocess.run(
            [sys.executable, str(HARDENED), "--mode", "scan", "--not-real", "1"],
            capture_output=True, text=True, timeout=300,
        )
        self.assertNotEqual(proc.returncode, 0)


class TestPhaseBoundaryResume(unittest.TestCase):
    """Exact requested edge cases: 1999/2000 and 1499/1500."""

    def test_preboundary_steps_do_not_advance(self):
        for phase, step in (("B", 1999), ("C", 1499), ("D", 1499)):
            with self.subTest(phase=phase, step=step):
                p = boundary_payload(phase, step)
                out, changed = A1.advance_boundary_payload(R, p)
                self.assertFalse(changed)
                self.assertEqual(out["phase"], phase)
                self.assertEqual(out["phase_step"], step)

    def test_b_2000_advances_once_to_c(self):
        p = boundary_payload("B", 2000)
        out, changed = A1.advance_boundary_payload(R, p)
        self.assertTrue(changed)
        self.assertEqual((out["phase_idx"], out["phase"], out["phase_step"]), (1, "C", 0))
        self.assertEqual(
            out["controller"]["post_phase_floor_until"]["A"],
            R.global_step_of(0, C.PHASE_UPDATES["B"]) + C.POST_PHASE_SPARSE_FLOOR,
        )
        again, changed2 = A1.advance_boundary_payload(R, out)
        self.assertFalse(changed2)
        self.assertEqual(again["phase"], "C")

    def test_c_1500_advances_once_to_d(self):
        p = boundary_payload("C", 1500)
        out, changed = A1.advance_boundary_payload(R, p)
        self.assertTrue(changed)
        self.assertEqual((out["phase_idx"], out["phase"], out["phase_step"]), (2, "D", 0))
        again, changed2 = A1.advance_boundary_payload(R, out)
        self.assertFalse(changed2)

    def test_d_1500_advances_to_finalize(self):
        p = boundary_payload("D", 1500)
        out, changed = A1.advance_boundary_payload(R, p)
        self.assertTrue(changed)
        self.assertEqual(out["phase_idx"], len(C.PHASES))
        self.assertEqual(out["phase"], "FINALIZE")
        self.assertEqual(out["phase_step"], 0)

    def test_plastic_boundary_does_not_invent_guardian_floor(self):
        p = boundary_payload("B", 2000, arm="PLASTIC_HIGH")
        out, changed = A1.advance_boundary_payload(R, p)
        self.assertTrue(changed)
        self.assertEqual(out["controller"]["post_phase_floor_until"], {})


class TestFailClosedResumeScan(unittest.TestCase):
    def _write_checkpoint(self, root: Path, *, missing: str | None = None):
        ms = root / "matched_sets" / "p31801_b429001" / "PLASTIC_HIGH"
        ms.mkdir(parents=True, exist_ok=True)
        task_hashes = {k: f"hash-{k}" for k in ("A", "B", "C", "D")}
        payload = {
            "schema": "arkenstone-ark020-v4-arm-ckpt/v1",
            "parent_seed": 31801,
            "b_order_seed": 429001,
            "c_order_seed": 429003,
            "d_order_seed": 429005,
            "arm": "PLASTIC_HIGH",
            "dose_b": 8,
            "parent_sha": "acquired-parent",
            "task_hash": R.hjson(task_hashes),
            "cap16x": 1.0,
            "phase_idx": 0,
            "phase": "B",
            "phase_step": 100,
            "registry": registry("A"),
            "controller": C.initial_controller("PLASTIC_HIGH"),
            "counters": {},
            "phase_confirm": {"B": None, "C": None, "D": None},
            "global_confirm": {"B": None, "C": None, "D": None},
            "b_streaks": {"B": 0, "C": 0, "D": 0},
        }
        if missing:
            payload.pop(missing)
        torch.save(payload, ms / "RESUME.pt")
        (root / "ENTRY_RECEIPT.json").write_text(json.dumps({"task_hashes": task_hashes}))
        (root / "PARENT_IDENTITIES.json").write_text(json.dumps({
            "parents": {"31801": {"acquired_parent_model_sha256": "acquired-parent"}}
        }))
        (root / "DOSE_SELECTION_IMPORTED.json").write_text(json.dumps({
            "v4_local_selected_slots": 8, "verification_status": "verified"
        }))
        return ms / "RESUME.pt"

    def test_partial_identity_never_becomes_resume(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(R, "OUT", Path(td)):
            self._write_checkpoint(Path(td))
            # Deliberately no executable identity receipt.
            info = R.resume_scan(drive_ok=True)
        self.assertEqual(info["SAFE_ACTION"], "STOP — CHECKPOINT IDENTITY FAILURE")
        self.assertEqual(info["HARDENED_GATE"], "FAIL")

    def test_missing_required_checkpoint_field_stops(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(R, "OUT", Path(td)):
            self._write_checkpoint(Path(td), missing="cap16x")
            info = R.resume_scan(drive_ok=True)
        self.assertEqual(info["SAFE_ACTION"], "STOP — CHECKPOINT IDENTITY FAILURE")
        self.assertTrue(any("cap16x" in e for e in info["HARDENED_ERRORS"]))

    def test_orphan_partial_without_checkpoint_stops(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(R, "OUT", Path(td)):
            p = Path(td) / "matched_sets" / "p1_b2" / "PLASTIC_HIGH"
            p.mkdir(parents=True)
            (p / "PARTIAL.json").write_text(json.dumps({"status": "PARTIAL_SESSION"}))
            info = R.resume_scan(drive_ok=True)
        self.assertEqual(info["SAFE_ACTION"], "STOP — CHECKPOINT IDENTITY FAILURE")
        self.assertTrue(any("orphan" in e for e in info["HARDENED_ERRORS"]))

    def test_fully_verified_checkpoint_can_resume(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(R, "OUT", Path(td)), \
             mock.patch.object(A1, "ensure_executable_receipt", return_value={"status": "PASS"}):
            self._write_checkpoint(Path(td))
            info = R.resume_scan(drive_ok=True)
        self.assertEqual(info["SAFE_ACTION"], "RESUME")
        self.assertEqual(info["CHECKPOINT_IDENTITY"], "PASS")
        self.assertEqual(info["HARDENED_GATE"], "PASS")


class TestExecutableIdentity(unittest.TestCase):
    def test_existing_identity_mismatch_fails_closed(self):
        class FakeR:
            OUT = None
        with tempfile.TemporaryDirectory() as td:
            FakeR.OUT = Path(td)
            p = FakeR.OUT / A1.EXECUTABLE_RECEIPT
            p.write_text(json.dumps({
                "schema": "x", "experiment": "ARK-020-V4",
                "engineering_amendment": A1.AMENDMENT_ID,
                "git_commit": "old", "files": {},
            }))
            current = {
                "schema": "arkenstone-ark020-v4-executable-identity-a1/v1",
                "experiment": "ARK-020-V4",
                "engineering_amendment": A1.AMENDMENT_ID,
                "git_commit": "new",
                "files": {"runner": "abc"},
            }
            with mock.patch.object(A1, "executable_identity", return_value=current):
                with self.assertRaisesRegex(RuntimeError, "executable identity mismatch"):
                    A1.ensure_executable_receipt(FakeR, allow_create=False)


class TestStrongResumeGate(unittest.TestCase):
    def test_composite_gate_requires_both_production_and_controller_smokes(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(R, "OUT", Path(td)), \
             mock.patch.dict(A1._ORIGINALS, {
                 "exact_resume_smoke": mock.Mock(return_value={"status": "PASS"})
             }), \
             mock.patch.object(A1, "controller_registry_resume_smoke",
                               return_value={"status": "FAIL", "checks": {"registry": False}}):
            with self.assertRaisesRegex(RuntimeError, "durability smoke failed"):
                A1.hardened_exact_resume_smoke(R, {}, 8, {}, {}, {}, torch.device("cpu"))

    def test_install_patches_production_preexecution_hook(self):
        self.assertTrue(R._ARK020_V4_A1_INSTALLED)
        self.assertIsNot(R.exact_resume_smoke, A1._ORIGINALS["exact_resume_smoke"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
