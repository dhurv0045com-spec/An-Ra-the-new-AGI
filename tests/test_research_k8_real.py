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
        open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8").write("{}")
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
        open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8").write("{}")
        os.makedirs(os.path.join(tmp, "meta"), exist_ok=True)
        open(os.path.join(tmp, "meta", "meta_tasks.jsonl"), "w", encoding="utf-8").write("")
        run = tempfile.mkdtemp()
        job = JobInput(phase="E5", slot=0, arm=None, seed=1701, parent=None,
                       physical_device="cpu", local_device="cpu",
                       data_dir=tmp, run_dir=run, precision="fp32",
                       deadline=9e9)
        res = e5.execute(job, ops=RecordingDoubleOps(), tasks_per_block=1)
        self.assertEqual(res.status, "failed")
        self.assertIn("explicit adaptation anchor", (res.error or ""))


class E5OrderingTests(unittest.TestCase):
    """U08 acceptance: train P0 BEFORE capture; no pretraining choice may
    survive the training boundary; successors decode their own choices."""

    def _e5_fixture(self):
        from bramastra_lab.research.campaigns.supervisor import CampaignLedger

        data_dir = tempfile.mkdtemp()
        open(os.path.join(data_dir, "manifest.json"), "w", encoding="utf-8").write("{}")
        os.makedirs(os.path.join(data_dir, "meta"), exist_ok=True)
        with open(os.path.join(data_dir, "meta", "meta_tasks.jsonl"), "w",
                  encoding="utf-8") as handle:
            # Two meta-training tasks: Block A (archive) and the distinct
            # fresh Block B slice the successors train on.
            handle.write(json.dumps({
                "pool": "meta-training", "meta_task_id": "mt-0",
                "family": "rule-inquiry"}) + "\n")
            handle.write(json.dumps({
                "pool": "meta-training", "meta_task_id": "mt-1",
                "family": "rule-inquiry"}) + "\n")
            handle.write(json.dumps({
                "pool": "meta-confirmation", "meta_task_id": "mc-0",
                "family": "rule-inquiry"}) + "\n")
        run = tempfile.mkdtemp()
        manifest = _save_tiny(run, "E1-B-1701", 3)
        ledger = CampaignLedger(run)
        ledger.record_allocation("alloc-e5", "src", "data", 480.0)
        reservation = ledger.reserve(
            "E1-B-1701", worker="w", device="cpu", phase="E1", arm="B",
            seed=1701, reserved_seconds=60.0)
        ledger.close_reservation(
            reservation.reservation_id, status="completed",
            committed_updates=1, attempted_updates=1, supervised_exposure=1,
            device_seconds=1.0, checkpoint_identity=manifest.checkpoint_id)
        ledger.close()
        return data_dir, run

    def test_p0_choice_captured_after_training_boundary(self) -> None:
        from bramastra_lab.research.campaigns.phases import e5
        from bramastra_lab.research.campaigns.phases.ops import (
            RecordingDoubleOps)
        from bramastra_lab.research.campaigns.phases.types import JobInput

        data_dir, run = self._e5_fixture()
        job = JobInput(phase="E5", slot=0, arm=None, seed=1701,
                       parent="E1-B-1701", physical_device="cpu",
                       local_device="cpu", data_dir=data_dir, run_dir=run,
                       precision="fp32", deadline=9e9)
        order: list[str] = []
        real_capture = e5._capture_proposer_choice
        real_train = e5._train_method_selection

        def spy_capture(*args, **kwargs):
            order.append("capture")
            return real_capture(*args, **kwargs)

        def spy_train(*args, **kwargs):
            order.append("train")
            return real_train(*args, **kwargs)

        e5._capture_proposer_choice = spy_capture
        e5._train_method_selection = spy_train
        try:
            res = e5.execute(job, ops=RecordingDoubleOps(),
                             tasks_per_block=1)
        finally:
            e5._capture_proposer_choice = real_capture
            e5._train_method_selection = real_train
        self.assertEqual(res.status, "completed",
                         f"E5 double run failed: {res.error}")
        # The P0 choice was captured strictly AFTER P0 training ran.
        self.assertIn("train", order)
        self.assertIn("capture", order)
        self.assertLess(order.index("train"), order.index("capture"))
        artifact = json.load(open(
            os.path.join(run, "phase_outputs", "E5", "E5-1701.json"),
            encoding="utf-8"))
        self.assertTrue(artifact["p0_capture"]["captured_after_training"])
        # P1 applies the captured (post-training) recipe; P_fixed stays M0.
        self.assertIn(artifact["successors"]["P1"]["method"],
                      ("M0", "M1", "M2"))
        self.assertEqual(artifact["successors"]["P1"]["method"],
                         artifact["p0_choice"])
        self.assertEqual(artifact["successors"]["P_fixed"]["method"], "M0")
        # Independent per-policy confirmation choices exist (no copying).
        self.assertIn("P1", artifact["confirmation_choices"])
        self.assertIn("P_fixed", artifact["confirmation_choices"])
        self.assertIn("P0", artifact["confirmation_choices"])
        self.assertEqual(artifact["confirmation_choices"]["fixed_M0"], "M0")
        self.assertIn("random", artifact["confirmation_choices"])
        # Successor captures record their own decoding provenance.
        self.assertIn("P1", artifact["successor_choice_captures"])
        self.assertIn("P_fixed", artifact["successor_choice_captures"])
        # Two DISTINCT measured archives: P0 trained on Block A, the
        # successors on the fresh Block B (never the same archive twice).
        self.assertIn("archive_b_identity", artifact)
        self.assertNotEqual(artifact["archive_b_identity"],
                            artifact["archive_identity"])
        self.assertEqual(artifact["archive_blocks"]["A"]["tasks"], 1)
        self.assertEqual(artifact["archive_blocks"]["B"]["tasks"], 1)
        for row in artifact["archive_b_rows"]:
            self.assertEqual(row["block"], "B")


class ReceiptSpoofTests(unittest.TestCase):
    def test_extra_cannot_override_reserved_keys(self) -> None:
        from bramastra_lab.research.campaigns.phases.types import PhaseResult
        with self.assertRaises(ValueError):
            PhaseResult(status="completed", committed_updates=2,
                        evidence_kind="fixture",
                        extra={"phase": "E1", "status": "failed"}).validate()
        res = PhaseResult(status="completed", committed_updates=2,
                          evidence_kind="fixture",
                          extra={"phase": "E1", "stream_id": "s"})
        d = res.to_dict()
        self.assertEqual(d["status"], "completed")
        self.assertEqual(d["committed_updates"], 2)
        self.assertEqual(d["phase"], "E1")

    def test_direct_id_requires_ledger_receipt(self) -> None:
        from bramastra_lab.research.campaigns.phases.types import ParentRef

        run = tempfile.mkdtemp()
        manifest = _save_tiny(run, "r1", 0)
        # No ledger at all: direct ID must fail (never arbitrary payload).
        with self.assertRaises(ValueError):
            ParentRef(checkpoint_id=manifest.checkpoint_id).resolve(run)


class E2ExplicitTests(unittest.TestCase):
    def test_eval_cases_required(self) -> None:
        from bramastra_lab.research.campaigns.phases.types import JobInput
        from bramastra_lab.research.campaigns.phases.ops import RecordingDoubleOps
        from bramastra_lab.research.campaigns.phases import e2

        tmp = tempfile.mkdtemp()
        open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8").write("{}")
        run = tempfile.mkdtemp()
        job = JobInput(phase="E2", slot=0, arm=None, seed=1701,
                       parent="E1-B-1701", physical_device="cpu",
                       local_device="cpu", data_dir=tmp, run_dir=run,
                       precision="fp32", deadline=9e9)
        res = e2.execute(job, ops=RecordingDoubleOps(), eval_cases=None)
        self.assertEqual(res.status, "failed")
        self.assertIn("explicit eval_cases", (res.error or ""))


class E3ExactMixtureTests(unittest.TestCase):
    def test_non_multiple_of_four_refused(self) -> None:
        from bramastra_lab.research.campaigns.phases.e3 import (
            _build_training_stream)
        import tempfile as tf
        from bramastra_lab.research.data.k8_bundle import build_k8_bundle

        tmp = tf.mkdtemp()
        build_k8_bundle(tmp, training_mechanisms=8, controller_mechanisms=1,
                        development_mechanisms=1, confirmation_mechanisms=1,
                        tool_mechanisms=3, tool_heldout=1,
                        meta_train=1, meta_validate=1, meta_confirm=1)
        with self.assertRaises(ValueError):
            _build_training_stream(tmp, 1701, "T1", 2)


if __name__ == "__main__":
    unittest.main()
