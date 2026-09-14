"""K8 real-execution regression tests (contracts S1-S7).

No optimizer steps, no GPU, no paid compute. Doubles and tiny payloads only;
the large k8-campaign model is never instantiated here (see
engineering/reports/K8_REAL_EXECUTION_20260914/verify_local.py for the
campaign-config integration proof).
"""
import json
import os
import tempfile
import unittest

import torch


def _tiny_payload(update_index=0):
    return {"model": {"w": torch.zeros(4)},
            "counters": {"optimizer_updates": update_index}}


def _save_tiny(run_dir, run_id, update_index, *, suffix=None):
    from bramastra_lab.research.runtime import checkpoint as ckpt

    return ckpt.save_checkpoint(
        run_dir, _tiny_payload(update_index), run_id=run_id,
        update_index=update_index, config_identity="cfg-test",
        tokenizer_identity="tok-test", data_identity="data-test",
        parent_checkpoint_id=None, code_identity="code-test",
        dir_suffix=suffix)


class ParentExactMatchTests(unittest.TestCase):
    def test_substring_does_not_match(self) -> None:
        from bramastra_lab.research.campaigns.phases.types import ParentRef
        from bramastra_lab.research.campaigns.supervisor import CampaignLedger

        run = tempfile.mkdtemp()
        manifest = _save_tiny(run, "r1", 0)
        ledger = CampaignLedger(run)
        ledger.record_allocation("a", "src", "data", 480.0)
        r = ledger.reserve("E1-B-17010", worker="w", device="cuda:0",
                           phase="E1", arm="B", seed=17010,
                           reserved_seconds=10.0)
        ledger.close_reservation(
            r.reservation_id, status="completed", committed_updates=1,
            attempted_updates=1, supervised_exposure=1, device_seconds=1.0,
            checkpoint_identity=manifest.checkpoint_id)
        ledger.close()
        # E1-B-1701 must NOT match E1-B-17010 (exact lineage only).
        with self.assertRaises(ValueError):
            ParentRef(lookup_key="E1-B-1701").resolve(run)
        # Exact hit resolves.
        rec = ParentRef(lookup_key="E1-B-17010").resolve(run)
        self.assertEqual(rec["checkpoint_id"], manifest.checkpoint_id)
        self.assertEqual(rec["run_dir"], run)

    def test_run_dir_carried(self) -> None:
        from bramastra_lab.research.campaigns.phases.types import ParentRef

        run = tempfile.mkdtemp()
        manifest = _save_tiny(run, "r1", 0)
        from bramastra_lab.research.campaigns.supervisor import CampaignLedger
        ledger = CampaignLedger(run)
        ledger.record_allocation("a", "src", "data", 480.0)
        r = ledger.reserve("E1-B-1701", worker="w", device="cuda:0",
                           phase="E1", arm="B", seed=1701,
                           reserved_seconds=10.0)
        ledger.close_reservation(
            r.reservation_id, status="completed", committed_updates=1,
            attempted_updates=1, supervised_exposure=1, device_seconds=1.0,
            checkpoint_identity=manifest.checkpoint_id)
        ledger.close()
        rec = ParentRef(lookup_key="E1-B-1701").resolve(run)
        self.assertEqual(rec["run_dir"], run)


class CheckpointNamespaceTests(unittest.TestCase):
    def test_same_index_distinct_lineages(self) -> None:
        run = tempfile.mkdtemp()
        m1 = _save_tiny(run, "E1-A-1701-0", 0, suffix="E1-A-1701")
        m2 = _save_tiny(run, "E1-B-1701-0", 0, suffix="E1-B-1701")
        self.assertNotEqual(m1.checkpoint_id, m2.checkpoint_id)
        dirs = sorted(os.listdir(os.path.join(run, "checkpoints")))
        self.assertTrue(any(d.startswith("update-000000000000-E1-A-1701") for d in dirs))
        self.assertTrue(any(d.startswith("update-000000000000-E1-B-1701") for d in dirs))
        from bramastra_lab.research.runtime.checkpoint import load_checkpoint
        _, l1 = load_checkpoint(run, checkpoint_id=m1.checkpoint_id)
        _, l2 = load_checkpoint(run, checkpoint_id=m2.checkpoint_id)
        self.assertEqual(l1.checkpoint_id, m1.checkpoint_id)
        self.assertEqual(l2.checkpoint_id, m2.checkpoint_id)

    def test_legacy_names_unchanged(self) -> None:
        run = tempfile.mkdtemp()
        _save_tiny(run, "r1", 1)
        self.assertIn("update-000000000001",
                      os.listdir(os.path.join(run, "checkpoints")))


class WorkerPropagationTests(unittest.TestCase):
    def test_missing_training_target_fails(self) -> None:
        from bramastra_lab.research.campaigns import worker

        tmp = tempfile.mkdtemp()
        open(os.path.join(tmp, "manifest.json"), "w").write("{}")
        run = tempfile.mkdtemp()
        out = worker.run_worker_phase(
            phase="E1", device="cpu", arm="A", seed=1701, data_dir=tmp,
            run_dir=run, precision="fp32", deadline=9e9,
            physical_device="cpu", slot=0, parent=None,
            job_id="E1-A-1701", update_target=None)
        self.assertEqual(out["status"], "failed")
        self.assertIn("explicit update_target", out.get("error", ""))

    def test_spec_propagates_parent_slot_job(self) -> None:
        from bramastra_lab.research.campaigns import process_supervision as ps

        seen: dict = {}

        def echo_worker(**kwargs):
            seen.update(kwargs)
            return {"status": "failed", "error": "echo",
                    "committed_updates": 0, "attempted_updates": 0,
                    "supervised_exposure": 0, "device_seconds": 0.0,
                    "checkpoint_identity": None}

        import sys
        sys.modules["echo_mod"] = type(sys)("echo_mod")
        sys.modules["echo_mod"].fn = echo_worker
        spec = {"job_id": "E3-T1-1701", "phase": "E3", "physical_device": "cpu",
                "device": "cpu", "arm": "T1", "seed": 1701, "slot": 2,
                "parent": "E1-B-1701", "update_target": 80,
                "data_dir": tempfile.mkdtemp(), "run_dir": tempfile.mkdtemp(),
                "precision": "fp32", "deadline": 9e9}
        ps._child_execute(dict(spec), worker_fn_path="echo_mod:fn")
        self.assertEqual(seen.get("job_id"), "E3-T1-1701")
        self.assertEqual(seen.get("slot"), 2)
        self.assertEqual(seen.get("parent"), "E1-B-1701")
        self.assertEqual(seen.get("update_target"), 80)


class EvidenceKindTests(unittest.TestCase):
    def test_fixture_never_qualifies(self) -> None:
        from bramastra_lab.research.campaigns.runner import (
            _phase_success_for_output)
        self.assertFalse(_phase_success_for_output(
            "E1", {"status": "completed", "committed_updates": 2,
                   "evidence_kind": "fixture"}))
        self.assertTrue(_phase_success_for_output(
            "E1", {"status": "completed", "committed_updates": 2,
                   "evidence_kind": "learned-campaign"}))

    def test_learned_requires_phase_and_work(self) -> None:
        from bramastra_lab.research.campaigns.phases.types import (
            PhaseResult, EVIDENCE_LEARNED_CAMPAIGN)
        # Missing phase with zero work is refused (no marker spoofing).
        with self.assertRaises(ValueError):
            PhaseResult(status="completed", committed_updates=0,
                        evidence_kind=EVIDENCE_LEARNED_CAMPAIGN,
                        extra={}).validate()
        # E2/E6 legitimately carry zero committed with explicit phase.
        PhaseResult(status="completed", committed_updates=0,
                    evidence_kind=EVIDENCE_LEARNED_CAMPAIGN,
                    extra={"phase": "E2", "evaluated_cases": 2}).validate()
        PhaseResult(status="completed", committed_updates=0,
                    evidence_kind=EVIDENCE_LEARNED_CAMPAIGN,
                    extra={"phase": "E6", "export_dir": "/tmp/x"}).validate()


class E5AnchorTests(unittest.TestCase):
    def test_missing_anchor_fails_without_synthesis(self) -> None:
        from bramastra_lab.research.campaigns.phases.types import JobInput
        from bramastra_lab.research.campaigns.phases.ops import RecordingDoubleOps
        from bramastra_lab.research.campaigns.phases import e5

        tmp = tempfile.mkdtemp()
        open(os.path.join(tmp, "manifest.json"), "w").write("{}")
        os.makedirs(os.path.join(tmp, "meta"), exist_ok=True)
        open(os.path.join(tmp, "meta", "meta_tasks.jsonl"), "w").write("")
        run = tempfile.mkdtemp()
        job = JobInput(phase="E5", slot=0, arm=None, seed=1701, parent=None,
                       physical_device="cpu", local_device="cpu",
                       data_dir=tmp, run_dir=run, precision="fp32",
                       deadline=9e9)
        res = e5.execute(job, ops=RecordingDoubleOps(), tasks_per_block=1)
        self.assertEqual(res.status, "failed")
        self.assertIn("explicit adaptation anchor", (res.error or ""))


if __name__ == "__main__":
    unittest.main()
