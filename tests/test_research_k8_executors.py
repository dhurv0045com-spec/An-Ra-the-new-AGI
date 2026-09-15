"""Executor-level tests for E1, E3 and E6 (F10/F11/F20 evidence).

The data-layer and compiler pieces are covered elsewhere; these tests drive
the actual phase executors end-to-end on one run directory with the
production ledger, real published parent payloads and recording doubles —
no optimizer steps, no GPU.
"""
import json
import os
import tempfile
import unittest

import torch


def _bundle_fixture():
    from bramastra_lab.research.data.k8_bundle import build_k8_bundle

    data_dir = tempfile.mkdtemp()
    build_k8_bundle(data_dir, training_mechanisms=8, controller_mechanisms=2,
                    development_mechanisms=2, confirmation_mechanisms=2,
                    tool_mechanisms=4, tool_heldout=2, meta_train=2,
                    meta_validate=1, meta_confirm=1)
    return data_dir


def _job(phase, *, arm, seed, parent, data_dir, run_dir):
    from bramastra_lab.research.campaigns.phases.types import JobInput

    return JobInput(phase=phase, slot=0, arm=arm, seed=seed, parent=parent,
                    physical_device="cpu", local_device="cpu",
                    data_dir=data_dir, run_dir=run_dir, precision="fp32",
                    deadline=9e9)


def _wire_parent(run_dir: str, job_id: str, seed: int, index: int) -> str:
    """Publish a real tiny parent payload + production ledger receipt."""
    from bramastra_lab.research.campaigns.supervisor import CampaignLedger
    from bramastra_lab.research.runtime import checkpoint as ckpt

    arm = job_id.split("-")[1]
    manifest = ckpt.publish_checkpoint(
        run_dir=run_dir, run_id=job_id, update_index=index,
        payload={"model": {"w": torch.arange(4, dtype=torch.float32) + index},
                 "counters": {"optimizer_updates": index}},
        config_identity="cfg-executor", tokenizer_identity="tok-executor",
        data_identity="data-executor", parent_checkpoint_id=None,
        code_identity="code-executor", phase="E1", arm=arm, seed=seed)
    ledger = CampaignLedger(run_dir)
    try:
        if ledger.deadline() is None:
            ledger.record_allocation("alloc-executors", "src", "data", 480.0)
        reservation = ledger.reserve(
            job_id, worker="executors", device="cpu", phase="E1", arm=arm,
            seed=seed, reserved_seconds=1.0)
        ledger.close_reservation(
            reservation.reservation_id, status="completed",
            committed_updates=index, attempted_updates=index,
            supervised_exposure=index, device_seconds=0.0,
            checkpoint_identity=manifest.checkpoint_id)
    finally:
        ledger.close()
    return manifest.checkpoint_id


class E1ExecutorTests(unittest.TestCase):
    def test_both_arms_complete_with_honest_fixture_evidence(self) -> None:
        from bramastra_lab.research.campaigns.phases import e1
        from bramastra_lab.research.campaigns.phases.ops import (
            RecordingDoubleOps)

        data_dir = _bundle_fixture()
        run_dir = tempfile.mkdtemp()
        ops = RecordingDoubleOps()
        identities = {}
        for arm in ("A", "B"):
            res = e1.execute(_job("E1", arm=arm, seed=1701, parent=None,
                                  data_dir=data_dir, run_dir=run_dir),
                             ops=ops, update_target=2)
            self.assertEqual(res.status, "completed",
                             f"E1 arm {arm} failed: {res.error}")
            identities[arm] = res.checkpoint_identity
            self.assertTrue(res.checkpoint_identity)
        # Paired arms carry distinct checkpoint identities (never shared).
        self.assertNotEqual(identities["A"], identities["B"])
        # Doubles never produce learned evidence (no optimizer authority).
        self.assertEqual(res.evidence_kind, "fixture")


class E3ExecutorTests(unittest.TestCase):
    def test_t0_t1_forks_from_same_parent_independently(self) -> None:
        from bramastra_lab.research.campaigns.phases import e3
        from bramastra_lab.research.campaigns.phases.ops import (
            RecordingDoubleOps)

        data_dir = _bundle_fixture()
        run_dir = tempfile.mkdtemp()
        _wire_parent(run_dir, "E1-B-1701", 1701, 1)
        ops = RecordingDoubleOps()
        results = {}
        for arm, target in (("T0", 2), ("T1", 4)):
            # T1 requires a multiple of 4 for the exact 75/25 mixture.
            res = e3.execute(_job("E3", arm=arm, seed=1701,
                                  parent="E1-B-1701", data_dir=data_dir,
                                  run_dir=run_dir),
                             ops=ops, update_target=target)
            self.assertEqual(res.status, "completed",
                             f"E3 arm {arm} failed: {res.error}")
            results[arm] = res
        # Independent child lineages (never one shared checkpoint).
        self.assertNotEqual(results["T0"].checkpoint_identity,
                            results["T1"].checkpoint_identity)


    def test_t0_refuses_rows_carrying_heldout_identities(self) -> None:
        """Negative control: the protected-exclusion proof must fire when a
        training row carries a held-out mechanism identity."""
        from bramastra_lab.research.campaigns.phases import e3
        from bramastra_lab.research.campaigns.phases.ops import (
            RecordingDoubleOps)

        data_dir = _bundle_fixture()
        run_dir = tempfile.mkdtemp()
        _wire_parent(run_dir, "E1-B-1701", 1701, 1)
        # Corrupt one tool-training row into a held-out identity.
        tool_path = os.path.join(data_dir, "tools", "tool_tasks.jsonl")
        rows = [json.loads(line) for line in open(tool_path, encoding="utf-8")
                if line.strip()]
        heldout_id = next(r["mechanism_id"] for r in rows
                          if r.get("split") == "tool-heldout")
        corrupted = 0
        for row in rows:
            if row.get("split") == "tool-training" and corrupted < 1:
                row["mechanism_id"] = heldout_id
                corrupted += 1
        with open(tool_path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        res = e3.execute(_job("E3", arm="T0", seed=1701, parent="E1-B-1701",
                              data_dir=data_dir, run_dir=run_dir),
                         ops=RecordingDoubleOps(), update_target=2)
        self.assertEqual(res.status, "failed")
        self.assertIn("heldout mechanism", (res.error or ""))


class E6ExecutorTests(unittest.TestCase):
    def test_export_executor_verifies_real_payload_bundle(self) -> None:
        from bramastra_lab.research.campaigns.phases import e6
        from bramastra_lab.research.campaigns.phases.e6 import REQUIRED_FILES

        data_dir = _bundle_fixture()
        run_dir = tempfile.mkdtemp()
        # Both registered seeds' E1-B parents, real restorable payloads.
        _wire_parent(run_dir, "E1-B-1701", 1701, 1)
        _wire_parent(run_dir, "E1-B-1702", 1702, 1)
        res = e6.execute(_job("E6", arm=None, seed=1701, parent=None,
                              data_dir=data_dir, run_dir=run_dir))
        self.assertEqual(res.status, "completed",
                         f"E6 failed: {res.error}")
        out_dir = os.path.join(run_dir, "K8-results")
        for name in REQUIRED_FILES:
            self.assertTrue(os.path.exists(os.path.join(out_dir, name)),
                            f"missing export file {name}")
        self.assertTrue(res.checkpoint_identity)

    def test_export_executor_refuses_without_required_parents(self) -> None:
        from bramastra_lab.research.campaigns.phases import e6

        data_dir = _bundle_fixture()
        run_dir = tempfile.mkdtemp()
        # Ledger exists but only ONE required parent is wired.
        _wire_parent(run_dir, "E1-B-1701", 1701, 1)
        res = e6.execute(_job("E6", arm=None, seed=1701, parent=None,
                              data_dir=data_dir, run_dir=run_dir))
        self.assertEqual(res.status, "failed")
        self.assertIn("E1-B-1702", (res.error or ""))


if __name__ == "__main__":
    unittest.main()
