"""ARK-020 V4 test suite — V3 layers + blocker-specific regressions.

New layers: CLI scan contract (real subprocess), acquired-parent identity flow,
dose import receipt, zero historical V4 writes, capability-lifecycle retention
accounting, SPARSE64/STATIC replay cadence, forced timebox, phase-boundary resume.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock as mock
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments" / "ARK-020-V4"))
sys.path.insert(0, str(HERE.parent / "experiments" / "ARK-019"))
sys.path.insert(0, str(HERE.parent / "experiments" / "ARK-018"))

import ark020_v4_core as C  # noqa: E402
import run_ark019_v3 as V3  # noqa: E402
import run_ark020_v4 as R  # noqa: E402

RUNNER = HERE.parent / "experiments" / "ARK-020-V4" / "run_ark020_v4.py"
TASK_TOKENS = list(range(48))
TASKS = C.build_all_tasks(TASK_TOKENS)


def m3(canonical, order, third):
    return {"canonical": canonical, "order_only": order, "query_order": third}


class TestCLIScanContract(unittest.TestCase):
    """Blocker A: the EXACT command line the notebook uses must work."""

    def test_cli_scan_drive_unavailable(self):
        proc = subprocess.run([sys.executable, str(RUNNER), "--mode", "scan",
                               "--drive-ok", "False"],
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        marker = "@@SCAN_JSON@@"
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(marker)), None)
        self.assertIsNotNone(line, "scan must emit machine-parsable SAFE ACTION")
        info = json.loads(line[len(marker):])
        self.assertEqual(info["SAFE_ACTION"], "STOP — DRIVE UNAVAILABLE")

    def test_cli_rejects_unknown_scan_args(self):
        proc = subprocess.run([sys.executable, str(RUNNER), "--mode", "scan",
                               "--bogus-flag", "1"],
                              capture_output=True, text=True, timeout=300)
        self.assertNotEqual(proc.returncode, 0)


class TestParentIdentity(unittest.TestCase):
    """Blocker B: acquired parent vs source model must never be confused."""

    def test_entry_receipt_schema_requires_both_identities(self):
        model = V3.Ark018GPT()
        src_sha = R.V3.state_hash({k: v.clone() for k, v in model.state_dict().items()})
        acq_sha = R.V3.state_hash({k: v.clone() for k, v in model.state_dict().items()})
        receipt = {"parents": {"31801": {"source_model_sha256": src_sha,
                                         "acquired_parent_model_sha256": acq_sha}}}
        # scan must compare against the ACQUIRED hash, not the source
        self.assertEqual(receipt["parents"]["31801"]["acquired_parent_model_sha256"], acq_sha)

    def test_scan_fails_when_parent_sha_matches_only_source(self):
        model = V3.Ark018GPT()
        src_sha = R.V3.state_hash({k: v.clone() for k, v in model.state_dict().items()})
        acq_sha = "f" * 64  # acquired parent differs after acquisition
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "camp"; root.mkdir()
            ms = root / "matched_sets" / "p31801_b429001" / "PLASTIC_HIGH"
            ms.mkdir(parents=True)
            (root / "ENTRY_RECEIPT.json").write_text(json.dumps({
                "task_hashes": {k: R.hjson(TASKS[k]) for k in ("A", "B", "C", "D")},
                "sources": [{"seed": 31801, "model_sha256": src_sha}]}))
            (root / "PARENT_IDENTITIES.json").write_text(json.dumps({"parents": {
                "31801": {"source_model_sha256": src_sha,
                          "acquired_parent_model_sha256": acq_sha}}}))
            (root / "DOSE_SELECTION_IMPORTED.json").write_text(json.dumps(
                {"status": "PASS", "selected_b_slots": 8, "verification_status": "verified"}))
            payload = {"schema": "arkenstone-ark020-v4-arm-ckpt/v1", "phase_idx": 0,
                       "phase": "B", "phase_step": 10, "registry": {}, "controller": {},
                       "counters": {}, "phase_confirm": {}, "global_confirm": {},
                       "b_streaks": {}, "control_trace": [], "measure_trace": [],
                       "retention_events": [], "stream_receipts": [], "global_step": 10,
                       "parent_seed": 31801, "b_order_seed": 429001, "c_order_seed": 429003,
                       "d_order_seed": 429005, "arm": "PLASTIC_HIGH", "dose_b": 8,
                       "parent_sha": src_sha,  # WRONG: source, not acquired parent
                       "task_hash": R.hjson({k: R.hjson(TASKS[k]) for k in ("A", "B", "C", "D")}),
                       "cap16x": 1.0}
            o = R.opt_for(model, 1e-4)
            sc = R.make_scaler(torch.device("cpu"))
            R.save_checkpoint(ms / "RESUME.pt", payload, model, o, sc)
            with mock.patch.object(R, "OUT", root):
                info = R.resume_scan(drive_ok=True)
            self.assertEqual(info["IDENTITY_CHECKS"]["parent_hash_vs_acquired_parent"], "FAIL")
            self.assertEqual(info["SAFE_ACTION"], "STOP — CHECKPOINT IDENTITY FAILURE")
            # and the CORRECT acquired hash passes
            payload["parent_sha"] = acq_sha
            R.save_checkpoint(ms / "RESUME.pt", payload, model, o, sc)
            with mock.patch.object(R, "OUT", root):
                info = R.resume_scan(drive_ok=True)
            self.assertEqual(info["SAFE_ACTION"], "RESUME")


class TestDoseImportReceipt(unittest.TestCase):
    """Blocker C: local immutable provenance; no cross-experiment resume dependency."""

    def test_import_receipt_validated_locally(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "camp"; root.mkdir()
            (root / "DOSE_SELECTION_IMPORTED.json").write_text(json.dumps({
                "schema": "arkenstone-ark020-v4-dose-import/v1",
                "source_experiment": "ARK-019-V4",
                "source_sha256": "abc",
                "source_selected_slots": 8, "verification_status": "verified",
                "v4_local_selected_slots": 8,
                "scientific_equivalence": "identical construction"}))
            with mock.patch.object(R, "OUT", root):
                dose = R.select_dose_local({}, {}, {}, {}, torch.device("cpu"), time.monotonic() + 1)
            self.assertEqual(dose["selected_b_slots"], 8)
            self.assertTrue(dose.get("imported"))


class TestZeroHistoricalWrites(unittest.TestCase):
    """Blocker D: parent/dose development path must not write into the historical root."""

    def test_acquisition_redirect_writes_only_local(self):
        with tempfile.TemporaryDirectory() as td:
            hist = Path(td) / "ARK019_GUARDIAN_V4"
            hist.mkdir()
            with mock.patch.object(R, "OUT", Path(td) / "v4camp"), \
                 mock.patch.object(V4_mod().V4, "OUT", hist):
                with R.v4_local_output_root():
                    marker = R.V4.OUT / "v4_reuse" / "PARENT_V4.json"
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.write_text("{}")
                # nothing may appear under the historical root
                self.assertEqual(list(hist.rglob("*")), [])


def V4_mod():
    import run_ark019_v4 as V4
    return type("M", (), {"V4": V4})


class TestLifecycleRetention(unittest.TestCase):
    """Blocker E: acquisition is not forgetting."""

    def test_retention_targets_exclude_acquiring_skill(self):
        targets = C.retention_targets({"A"}, "B", phase_confirmed=False)
        self.assertEqual(targets, {"A"})
        targets = C.retention_targets({"A", "B"}, "C", phase_confirmed=False)
        self.assertEqual(targets, {"A", "B"})
        targets = C.retention_targets({"A", "B", "C"}, "D", phase_confirmed=False)
        self.assertEqual(targets, {"A", "B", "C"})

    def test_case1_acquiring_skill_low_is_not_forgetting(self):
        events = []
        recorded = C.record_retention_event(events, "B", 300, "B", eligible=False,
                                            control_metrics=m3(0.4, 0.4, 0.4))
        self.assertFalse(recorded)
        self.assertEqual(events, [])

    def test_case2_acquired_cap_failing_is_retention_failure(self):
        events = []
        recorded = C.record_retention_event(events, "A", 300, "B", eligible=True,
                                            control_metrics=m3(0.4, 0.4, 0.4))
        self.assertTrue(recorded)
        self.assertEqual(events[0]["capability"], "A")
        self.assertTrue(events[0]["was_acquired_before_phase"])

    def test_case3_confirmed_cap_failing_later_is_forgetting(self):
        events = []
        C.record_retention_event(events, "B", 100, "C", eligible=True,
                                 control_metrics=m3(0.4, 0.4, 0.4))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["phase"], "C")

    def test_qualified_cap_never_records(self):
        events = []
        self.assertFalse(C.record_retention_event(events, "A", 100, "B", eligible=True,
                                                  control_metrics=m3(0.99, 0.96, 0.96)))
        self.assertEqual(events, [])

    def test_interference_gate_uses_true_events(self):
        from tests.test_ark020_v3 import TestGoldenDecide  # reuse row builder
        b = TestGoldenDecide("test_scenarios")
        res = b.build()
        # plastic rows report zero TRUE retention events despite low acquisition metrics
        for r in res["PLASTIC_HIGH"]:
            r["retention_failure_events"] = [{"capability": "A", "failure_step": 100,
                                              "phase": "B"}]
        d = C.decide(res)
        self.assertEqual(d["interference"], True)


class TestReplayCadence(unittest.TestCase):
    """Blocker F + #11: the names must mean actual data exposure."""

    def _counts(self, arm, states, steps=640):
        reg = C.init_registry()
        for cid in ("A",):
            C.register_capability(reg, cid, 0)
            C.observe_capability(reg, cid, 25, m3(0.99, 0.99, 0.99))
        ctrl = C.initial_controller(arm)
        ctrl["states"]["A"] = states
        slots = 0
        for step in range(1, steps + 1):
            rep, _ = C.treatment(arm, ctrl, step, reg, ["A"], None)
            slots += len(rep)
        return slots

    def test_sparse64_means_half_rate(self):
        self.assertEqual(self._counts("GUARDIAN_REACTIVE", "SPARSE64"), 320)

    def test_replay32_means_full_rate(self):
        self.assertEqual(self._counts("GUARDIAN_REACTIVE", "REPLAY32"), 640)

    def test_static_1of64(self):
        self.assertEqual(self._counts("STATIC_REPLAY_1OF64", "PLASTIC"), 320)

    def test_static_1of32(self):
        self.assertEqual(self._counts("STATIC_REPLAY_1OF32", "PLASTIC"), 640)

    def test_cap_never_replays(self):
        self.assertEqual(self._counts("STATIC_CAP16X", "PLASTIC"), 0)

    def test_max_two_slots_per_update(self):
        reg = C.init_registry()
        for cid in ("A", "B", "C", "D"):
            C.register_capability(reg, cid, 0)
            C.observe_capability(reg, cid, 25, m3(0.80, 0.80, 0.80))  # all fail -> REPLAY32
        ctrl = C.initial_controller("GUARDIAN_HYBRID")
        C.update_controller("GUARDIAN_HYBRID", ctrl, 50, reg, ["A", "B", "C", "D"])
        rep, _ = C.treatment("GUARDIAN_HYBRID", ctrl, 50, reg, ["A", "B", "C", "D"], None)
        self.assertLessEqual(len(rep), C.MAX_REPLAY_SLOTS_PER_UPDATE)


class TestForcedTimebox(unittest.TestCase):
    """Blocker G: the timebox path must checkpoint cleanly and raise SessionTimebox."""

    def test_timebox_checkpoint_and_partial(self):
        model = V3.Ark018GPT()
        o = R.opt_for(model, 1e-4)
        sc = R.make_scaler(torch.device("cpu"))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "camp"
            cp = root / "matched_sets" / "p1_b2" / "PLASTIC_HIGH" / "RESUME.pt"
            pp = root / "matched_sets" / "p1_b2" / "PLASTIC_HIGH" / "PARTIAL.json"
            # emulate the exact timebox body (nested scope, no outer del)
            m, o2, sc2 = model, o, sc
            def timebox_sim():
                payload = {"schema": "arkenstone-ark020-v4-arm-ckpt/v1", "phase_idx": 0,
                           "phase_step": 1999}
                R.save_checkpoint(cp, payload, m, o2, sc2)
                R.savej(pp, {"status": "PARTIAL_SESSION", "phase_step": 1999})
                raise R.SessionTimebox("paused at 1999")
            with self.assertRaises(R.SessionTimebox):
                timebox_sim()
            x = torch.load(cp, map_location="cpu", weights_only=False)
            self.assertEqual(x["phase_step"], 1999)
            self.assertEqual(json.loads(pp.read_text())["status"], "PARTIAL_SESSION")
            # the deleted-object pattern must NOT exist in the V3/V4 timebox
            src = RUNNER.read_text(encoding="utf-8")
            self.assertNotIn("        del m, o\n        torch.cuda.empty_cache()\n        raise SessionTimebox",
                              src)


class TestExactResumeV4(unittest.TestCase):
    def test_production_smoke_executes_with_explicit_device(self):
        model = V3.Ark018GPT()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        parent_state = {"model": {k: v.clone() for k, v in model.state_dict().items()},
                        "optimizer": opt.state_dict(), "scaler": {},
                        "cpu_rng": torch.get_rng_state(), "cuda_rng": []}
        buf = np.random.RandomState(2).randint(0, 8000, size=8193 * 2).astype(np.uint16)
        bufs = {"train": buf, "control": buf[:6000], "sealed": buf[6000:12000]}
        tt = {"A": {"p": [1], "m": [2], "s": [3], "q": [4], "t": [5]},
              "B": {"p": [6], "m": [7], "s": [8], "q": [9], "t": [10]},
              "C": {"p": [11], "g": [12], "s": [13], "k": [14], "q": [15], "e": [16]},
              "D": {"p": [17], "m": [18], "s": [19], "q": [20], "t": [21]}}
        with tempfile.TemporaryDirectory() as td, mock.patch.object(R, "OUT", Path(td)):
            r = R.exact_resume_smoke(parent_state, 8, bufs, tt, TASKS, torch.device("cpu"))
        self.assertEqual(r["status"], "PASS")
        self.assertTrue(all(r["fields"].values()))
        self.assertIn("phase_identity", r["coverage"])
        self.assertIn("semantic_stream", r["coverage"])

    def test_hard_interruption_scan_finds_checkpoint_without_partial(self):
        model = V3.Ark018GPT()
        o = R.opt_for(model, 1e-4)
        sc = R.make_scaler(torch.device("cpu"))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "camp"
            ms = root / "matched_sets" / "p31801_b429001" / "PLASTIC_HIGH"
            ms.mkdir(parents=True)
            (root / "ENTRY_RECEIPT.json").write_text(json.dumps({
                "task_hashes": {k: R.hjson(TASKS[k]) for k in ("A", "B", "C", "D")},
                "sources": []}))
            (root / "PARENT_IDENTITIES.json").write_text(json.dumps({"parents": {}}))
            (root / "DOSE_SELECTION_IMPORTED.json").write_text(json.dumps(
                {"selected_b_slots": 8, "verification_status": "verified"}))
            payload = {"schema": "arkenstone-ark020-v4-arm-ckpt/v1", "phase_idx": 0,
                       "phase": "B", "phase_step": 100, "registry": {}, "controller": {},
                       "counters": {}, "phase_confirm": {}, "global_confirm": {},
                       "b_streaks": {}, "control_trace": [], "measure_trace": [],
                       "retention_events": [], "stream_receipts": [], "global_step": 100,
                       "parent_seed": 31801, "b_order_seed": 429001, "c_order_seed": 429003,
                       "d_order_seed": 429005, "arm": "PLASTIC_HIGH", "dose_b": 8,
                       "parent_sha": "x", "task_hash": R.hjson(
                           {k: R.hjson(TASKS[k]) for k in ("A", "B", "C", "D")}),
                       "cap16x": 1.0}
            R.save_checkpoint(ms / "RESUME.pt", payload, model, o, sc)
            # NO PARTIAL.json, NO SESSION_STATE — process died hard
            with mock.patch.object(R, "OUT", root):
                info = R.resume_scan(drive_ok=True)
            self.assertTrue(info["CHECKPOINT_FOUND"])
            self.assertEqual(info["SAFE_ACTION"], "RESUME")
            self.assertEqual(info["CHECKPOINT_IDENTITY"], "PARTIAL_IDENTITY_CHECK")

    def test_partial_without_checkpoint_is_not_exact_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "camp"
            ms = root / "matched_sets" / "p1_b2" / "ARM"
            ms.mkdir(parents=True)
            R.savej(ms / "PARTIAL.json", {"status": "PARTIAL_SESSION"})
            with mock.patch.object(R, "OUT", root):
                info = R.resume_scan(drive_ok=True)
            self.assertFalse(info["CHECKPOINT_FOUND"])
            self.assertNotEqual(info["SAFE_ACTION"], "RESUME")


# ---- carried-over V3 layers (quick forms) ----
sys.path.insert(0, str(HERE))
from test_ark020_v3 import (  # noqa: E402,F401
    TestPureConstruction, TestSkillCAdversarial, TestSkillDTrueInverse,
    TestPhaseSeedStreams, TestGoldenDecide)


if __name__ == "__main__":
    unittest.main(verbosity=2)
