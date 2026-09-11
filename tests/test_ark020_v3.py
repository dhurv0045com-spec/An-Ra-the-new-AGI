"""ARK-020 V3 test suite — pure, task validity (adversarial), stream, contracts,
checkpoint/fail-closed, golden decisions, operator scan, and the PRODUCTION
exact-resume smoke executed directly (the test V2 never wrote)."""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
import time
import unittest
import unittest.mock as mock
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments" / "ARK-020-V3"))
sys.path.insert(0, str(HERE.parent / "experiments" / "ARK-019"))
sys.path.insert(0, str(HERE.parent / "experiments" / "ARK-018"))

import ark020_v3_core as C  # noqa: E402
import run_ark019_v3 as V3  # noqa: E402
import run_ark020_v3 as R  # noqa: E402


def m3(canonical, order, third):
    return {"canonical": canonical, "order_only": order, "query_order": third}


TASK_TOKENS = list(range(48))
TASKS = C.build_all_tasks(TASK_TOKENS)
C_T = {"p": [901], "g": [902], "s": [903], "k": [904], "q": [905], "e": [906]}


class TestPureConstruction(unittest.TestCase):
    def test_48_token_gate(self):
        with self.assertRaises(RuntimeError):
            C.build_all_tasks(list(range(47)))

    def test_splits_disjoint_and_sized(self):
        for sk in ("A", "B", "C", "D"):
            sp = TASKS[sk]
            self.assertEqual([len(sp[k]) for k in ("train", "parent_control", "main_control",
                                                   "validation", "sealed")], [400, 50, 50, 50, 50])

    def test_determinism(self):
        t2 = C.build_all_tasks(TASK_TOKENS)
        self.assertEqual(TASKS["C"]["train"], t2["C"]["train"])
        self.assertEqual(TASKS["D"]["train"], t2["D"]["train"])


class TestSkillCAdversarial(unittest.TestCase):
    def test_derivable_via_intermediate(self):
        for split in ("train", "main_control", "validation", "sealed"):
            for chains, xq, ans in TASKS["sem"]["C"][split][:100]:
                chain = next(c for c in chains["chains"] if c[0] == xq)
                self.assertEqual(ans, chain[2])
                self.assertTrue(any(c[1] == chain[1] and c[2] == chain[2]
                                    for c in chains["chains"]))

    def test_m_matcher_oracle_perfect_on_rendered(self):
        acc = C.m_matcher_oracle_accuracy(TASKS["sem"]["C"]["sealed"], 300, 77)
        self.assertEqual(acc, 1.0)

    def test_positional_shortcut_near_chance(self):
        score = C.positional_shortcut_score(TASKS["sem"]["C"]["sealed"], 900, 123)
        self.assertLess(score, 0.45, f"positional shortcut too effective: {score}")
        self.assertGreater(score, 0.20, f"score implausibly low: {score}")

    def test_mutation_tied_renderer_is_caught(self):
        # If the V2 tied-order renderer is reintroduced, the shortcut must read ~1.0.
        import random as _r
        rng = _r.Random(123)
        sem = TASKS["sem"]["C"]["sealed"]
        hits = 0
        n = 300
        for j in range(n):
            idx = rng.randrange(len(sem))
            chains, xq, ans = sem[idx]
            ids = C.render_compose_tied(C_T, chains, xq, C.augmented_perm3(7, j, idx))
            info = C.parse_rendered_compose(ids, C_T, xq)
            if info["y_at_same_ordinal"] == info["y_true"]:
                hits += 1
        tied_score = hits / n
        self.assertGreater(tied_score, 0.95,
                           "mutation detector broken: tied renderer should score ~1.0")
        free_score = C.positional_shortcut_score(sem, n, 123)
        self.assertLess(free_score, tied_score - 0.5)

    def test_no_mode_structurally_aligns_blocks(self):
        for mode in ("canonical", "reversed", "query_order", "augmented"):
            aligned = 0
            n = 240
            for j in range(n):
                idx = j % len(TASKS["sem"]["C"]["sealed"])
                chains, xq, ans = TASKS["sem"]["C"]["sealed"][idx]
                ids = C.render_compose(C_T, chains, xq, mode, 31, j, idx)
                info = C.parse_rendered_compose(ids, C_T, xq)
                if info["y_at_same_ordinal"] == info["y_true"]:
                    aligned += 1
            self.assertLess(aligned / n, 0.60, f"mode {mode} aligns blocks too often")


class TestSkillDTrueInverse(unittest.TestCase):
    def d_orientation_ok(self, sem_rows):
        """Every fact renders owner->object; query is an object; answer its owner."""
        t = {"p": [911], "m": [912], "s": [913], "q": [914], "t": [915]}
        for f, q_obj, a_owner in sem_rows[:80]:
            # fact orientation: owner token precedes its object in the rendered facts
            for (owner, obj) in f:
                ids, _ = C.task_rows("D", t, [(f, obj, owner)], [0], "canonical", 1, 1)[0]
                self.assertLess(ids.index(owner), ids.index(obj))
            self.assertIn(q_obj, [b for (_o, b) in f])       # query from object group
            self.assertIn(a_owner, [o for (o, _b) in f])     # answer from owner group
            # the ordered (object -> owner) pair is never presented as a fact
            self.assertNotIn((q_obj, a_owner), [tuple(p) for p in f])

    def test_v3_semantics_are_genuinely_inverse(self):
        self.d_orientation_ok(TASKS["sem"]["D"]["main_control"])
        self.d_orientation_ok(TASKS["sem"]["D"]["sealed"])

    def test_mutation_pre_reversed_is_caught(self):
        # If D is built the V2 way, the queried (object -> owner) pair appears
        # DIRECTLY as a presented fact — the inverse operation becomes unnecessary.
        old_style = [([(b, o) for (o, b) in f], q, a)
                     for (f, q, a) in TASKS["sem"]["D"]["sealed"][:20]]
        exposed = 0
        for f, q_obj, a_owner in old_style:
            if any(p1 == q_obj and p2 == a_owner for (p1, p2) in f):
                exposed += 1
        self.assertGreater(exposed, 0,
                           "mutation detector broken: pre-reversed D should expose (q, a)")
        # the genuine V3 construction never exposes the ordered pair
        for f, q_obj, a_owner in TASKS["sem"]["D"]["sealed"][:20]:
            self.assertFalse(any(p1 == q_obj and p2 == a_owner for (p1, p2) in f))


class TestPhaseSeedStreams(unittest.TestCase):
    def test_phase_seeds_distinct_and_bound(self):
        seeds = C.PHASE_ORDER_SEEDS
        self.assertEqual(len({*seeds["B"], *seeds["C"], *seeds["D"]}), 6)

    def test_changing_c_seed_changes_c_stream_only(self):
        # task-sample streams: same B/D seeds give identical streams under C-seed change
        def stream(seed, tag):
            return [C.deterministic_indices(seed, st, 8, 1000, f"main-{tag}-task")
                    for st in range(1, 6)]
        b0 = stream(C.PHASE_ORDER_SEEDS["B"][0], "B")
        c0 = stream(C.PHASE_ORDER_SEEDS["C"][0], "C")
        c1 = stream(C.PHASE_ORDER_SEEDS["C"][1], "C")
        d0 = stream(C.PHASE_ORDER_SEEDS["D"][0], "D")
        self.assertNotEqual(c0, c1)          # C seed change changes C stream
        b0_again = stream(C.PHASE_ORDER_SEEDS["B"][0], "B")
        self.assertEqual(b0, b0_again)       # B untouched
        self.assertEqual(d0, stream(C.PHASE_ORDER_SEEDS["D"][0], "D"))  # D untouched
        # real-text stream: per-phase seed changes real batch starts
        import run_ark020_v3 as R
        buf = np.random.RandomState(0).randint(0, 8000, size=9000).astype(np.uint16)
        starts_b = R.real_batch(buf, 2, C.PHASE_ORDER_SEEDS["B"][0], 1, torch.device("cpu"))[2]
        starts_b2 = R.real_batch(buf, 2, C.PHASE_ORDER_SEEDS["B"][0], 1, torch.device("cpu"))[2]
        starts_c = R.real_batch(buf, 2, C.PHASE_ORDER_SEEDS["C"][0], 1, torch.device("cpu"))[2]
        self.assertEqual(starts_b, starts_b2)
        self.assertNotEqual(starts_b, starts_c)


class TestGoldenDecide(unittest.TestCase):
    KEYS = [(31801, 429001), (31801, 429002), (31902, 429001), (31902, 429002)]

    def row(self, arm, ps, bs, *, conf=(300, 250, 250), fails=(), all_q=True,
            area=0.97, science=4.30, replay=300.0, duty=0.2):
        final = {c: m3(0.50, 0.50, 0.50) if (c in fails and not all_q) else m3(0.98, 0.96, 0.96)
                 for c in C.SKILLS}
        return {"arm": arm, "parent_seed": ps, "b_order_seed": bs,
                "phase_confirmation": {"B": conf[0], "C": conf[1], "D": conf[2]},
                "final_sealed": final,
                "sealed_area": {c: (0.55 if (c in fails and not all_q) else area) for c in C.SKILLS},
                "control_failure_steps": {c: [100] for c in fails},
                "recovery": {c: True for c in fails},
                "final_science_sealed_nll": science,
                "counters": {"replay_slots": replay,
                             "protection_updates": duty * C.CONTINUATION_HORIZON},
                "counters_by_phase": {
                    "B": {"replay_slots": replay / 3, "protection_updates": duty * 2000, "updates": 2000},
                    "C": {"replay_slots": replay / 3, "protection_updates": duty * 1500, "updates": 1500},
                    "D": {"replay_slots": replay / 3, "protection_updates": duty * 1500, "updates": 1500}}}

    def build(self, *, plastic_fails=("A",), guardian_fails=(), guardian_replay=300.0,
              guardian_all_q=True, conf=(300, 250, 250), plastic_conf=None,
              per_arm_overrides=None):
        out = {}
        for arm in C.ARMS:
            rows = []
            for (ps, bs) in self.KEYS:
                if arm == "PLASTIC_HIGH":
                    destroyed = bool(plastic_fails)
                    rows.append(self.row(arm, ps, bs, conf=plastic_conf or conf,
                                         fails=plastic_fails, all_q=False,
                                         area=0.60 if destroyed else 0.97,
                                         science=4.30, replay=0.0, duty=0.0))
                elif arm.startswith("STATIC"):
                    rows.append(self.row(arm, ps, bs, conf=conf, fails=(), all_q=True,
                                         area=0.97, science=4.32,
                                         replay=2500.0 if arm == "STATIC_REPLAY_1OF64" else 5000.0,
                                         duty=1.0))
                else:
                    rows.append(self.row(arm, ps, bs, conf=conf, fails=guardian_fails,
                                         all_q=guardian_all_q, area=0.96, science=4.35,
                                         replay=guardian_replay, duty=0.3))
            out[arm] = rows
        if per_arm_overrides:
            for arm, idx, patch in per_arm_overrides:
                out[arm][idx].update(patch)
        return out

    def test_scenarios(self):
        self.assertEqual(C.decide(self.build())["verdict"], "GUARDIAN_MULTI_SKILL_CANDIDATE")
        self.assertEqual(C.decide(self.build(guardian_replay=6000.0))["verdict"],
                         "GUARDIAN_SUPPORTED_NOT_EFFICIENT")
        res = self.build(guardian_fails=("A",), guardian_all_q=False)
        self.assertEqual(C.decide(res)["verdict"], "GUARDIAN_NOT_SUPPORTED_MULTI_SKILL")
        self.assertEqual(C.decide(self.build(plastic_fails=()))["verdict"],
                         "INCONCLUSIVE_LOW_INTERFERENCE")
        for skill, conf in (("B", (None, 250, 250)), ("C", (300, None, 250)),
                            ("D", (300, 250, None))):
            self.assertEqual(C.decide(self.build(plastic_conf=conf))["verdict"],
                             f"INCONCLUSIVE_FORMATION_INSTABILITY_PHASE_{skill}")
        res = self.build()
        del res["STATIC_CAP16X"][0]
        self.assertEqual(C.decide(res)["verdict"], "INCONCLUSIVE_INCOMPLETE_MATCHED_SETS")
        res = self.build()
        row = dict(res["GUARDIAN_REACTIVE"][0]); row["b_order_seed"] = 999999
        res["GUARDIAN_REACTIVE"][0] = row
        self.assertEqual(C.decide(res)["verdict"], "INCONCLUSIVE_MATCHING_ERROR")

    def test_acquisition_too_slow_is_quality_only(self):
        res = self.build(plastic_conf=(300, 250, 250), conf=(600, 500, 500))
        self.assertEqual(C.decide(res)["verdict"], "GUARDIAN_SUPPORTED_NOT_EFFICIENT")

    def test_median_rule_not_per_run(self):
        res = self.build(per_arm_overrides=[("GUARDIAN_REACTIVE", 0, {
            "phase_confirmation": {"B": 750, "C": 625, "D": 625}})])
        d = C.decide(res)
        self.assertEqual(d["verdict"], "GUARDIAN_MULTI_SKILL_CANDIDATE")
        self.assertTrue(d["guardian_details"]["GUARDIAN_REACTIVE"]
                        ["phase_relative_confirmation_within_1_5x_plastic_median"])

    def test_science_regression_blocks_success(self):
        overrides = [(arm, i, {"final_science_sealed_nll": 4.30 * 1.10})
                     for arm in ("GUARDIAN_REACTIVE", "GUARDIAN_PREDICTIVE", "GUARDIAN_HYBRID")
                     for i in range(4)]
        res = self.build(per_arm_overrides=overrides)
        d = C.decide(res)
        self.assertEqual(d["verdict"], "GUARDIAN_SUPPORTED_NOT_EFFICIENT")
        # a single regressing arm is disqualified without blocking the others
        res2 = self.build(per_arm_overrides=[
            ("GUARDIAN_REACTIVE", i, {"final_science_sealed_nll": 4.30 * 1.10})
            for i in range(4)])
        d2 = C.decide(res2)
        self.assertFalse(d2["guardian_details"]["GUARDIAN_REACTIVE"]["qualifies"])
        self.assertTrue(d2["guardian_details"]["GUARDIAN_PREDICTIVE"]["qualifies"])

    def test_predictive_adds_value_flag(self):
        res = self.build()
        for i in (0, 1):
            res["GUARDIAN_REACTIVE"][i]["final_sealed"]["A"] = m3(0.50, 0.50, 0.50)
            res["GUARDIAN_REACTIVE"][i]["sealed_area"]["A"] = 0.55
            res["GUARDIAN_REACTIVE"][i]["control_failure_steps"] = {"A": [100]}
        d = C.decide(res)
        self.assertIn("PREDICTIVE_ADDS_VALUE", d["flags"])


class TestContractsWithRealV4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = torch.device("cpu")
        torch.manual_seed(0)
        cls.model = V3.Ark018GPT()
        cls.tiny_state = {k: v.clone() for k, v in cls.model.state_dict().items()}

    def test_v4_signature_contract(self):
        self.assertEqual([p.name for p in inspect.signature(V4_acq_parent()).parameters.values()],
                         ["seed", "prep", "bufs", "tA", "A", "d", "deadline"])

    def test_v4_bmetrics_and_mixed_update_with_v3_objects(self):
        import run_ark019_v4 as V4
        tA = {"p": [1], "m": [2], "s": [3], "q": [4], "t": [5]}
        m = V4.bmetrics(self.model, tA, TASKS["sem"]["A"]["main_control"][:8], self.d)
        self.assertIn("qualified", m)
        buf = np.random.RandomState(0).randint(0, 512, size=8193 * 2).astype(np.uint16)
        wanted = {"tok.weight", "blocks.0.attn.qkv.weight", "blocks.4.mlp.2.weight",
                  "blocks.9.attn.qkv.weight", "ln_f.weight"}
        names = [n for n, _ in self.model.named_parameters() if n in wanted]
        o = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        sc = V3.make_scaler(self.d)
        rec = V4.mixed_update(self.model, o, sc, buf, tA, TASKS["sem"]["A"]["train"][:16],
                              555001, 1, 8, tA, TASKS["sem"]["A"]["train"][:16], 0, None,
                              self.d, names, tag="contract")
        self.assertIn("loss", rec)

    def test_v4_dose_pilot_boundary(self):
        import run_ark019_v4 as V4
        tB = {"p": [6], "m": [7], "s": [8], "q": [9], "t": [10]}
        buf = np.random.RandomState(1).randint(0, 512, size=8193 * 2).astype(np.uint16)
        bufs = {"train": buf, "control": buf[:6000], "sealed": buf[6000:12000]}
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        parent = {"model": self.tiny_state, "optimizer": opt.state_dict(), "scaler": {},
                  "cpu_rng": torch.get_rng_state(), "cuda_rng": []}
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(V4, "OUT", Path(td)), \
                 mock.patch.object(V4.C, "PILOT_MAX_UPDATES", 2), \
                 mock.patch.object(V4.C, "PILOT_EVAL_EVERY", 1), \
                 mock.patch.object(V4.C, "PILOT_STREAK", 1):
                r = V4.run_dose_pilot(31801, 8, 419001, parent, bufs, tB,
                                      TASKS["sem"]["B"], self.d, time.monotonic() + 600)
            self.assertIn("status", r)
            self.assertTrue((Path(td) / "dose_pilot" / "p31801_slots8" / "RESULT.json").exists())


def V4_acq_parent():
    import run_ark019_v4 as V4
    return V4.acquire_parent


class TestExactResumeProduction(unittest.TestCase):
    """Directly executes the production exact_resume_smoke (V2 never did)."""

    def test_production_smoke_executes_and_passes(self):
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
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(R, "OUT", Path(td)):
            r = R.exact_resume_smoke(parent_state, 8, bufs, tt, TASKS, torch.device("cpu"))
        self.assertEqual(r["status"], "PASS")
        self.assertTrue(all(r["fields"].values()))
        for field in ("model", "optimizer", "scaler", "cpu_rng", "registry", "controller",
                      "counters", "phase_identity", "confirmations", "telemetry",
                      "semantic_stream"):
            self.assertIn(field, r["coverage"])
        self.assertNotIn("cuda_rng", [f for f, v in r["fields"].items() if v == "cpu-device"])

    def test_resume_fails_closed_on_identity_mutation(self):
        model = V3.Ark018GPT()
        o = R.opt_for(model, 1e-4)
        sc = R.make_scaler(torch.device("cpu"))
        payload = {"schema": "arkenstone-ark020-v3-arm-ckpt/v1", "phase_idx": 0, "phase": "B",
                   "phase_step": 5, "registry": {}, "controller": {}, "counters": {},
                   "phase_confirm": {}, "global_confirm": {}, "b_streaks": {},
                   "control_trace": [], "measure_trace": [], "stream_receipts": [],
                   "global_step": 5}
        expected = {"schema": "arkenstone-ark020-v3-arm-ckpt/v1", "arm": "GUARDIAN_HYBRID",
                    "b_order_seed": 429001, "c_order_seed": 429003, "d_order_seed": 429005}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ck.pt"
            R.save_checkpoint(p, {**expected, **payload}, model, o, sc)
            for bad in ({"b_order_seed": 429002}, {"c_order_seed": 429004},
                        {"d_order_seed": 429006}, {"arm": "STATIC_CAP16X"},
                        {"schema": "arkenstone-ark020-v2-arm-ckpt/v1"}):
                with self.assertRaises(RuntimeError):
                    R.load_checkpoint(p, {**expected, **bad}, torch.device("cpu"))


class TestOperatorScan(unittest.TestCase):
    def _prep(self, td):
        root = Path(td) / "camp"
        root.mkdir(parents=True)
        return root

    def test_fresh_campaign(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._prep(td)
            with mock.patch.object(R, "OUT", root):
                info = R.resume_scan(drive_ok=True)
            self.assertEqual(info["SAFE_ACTION"], "START NEW CAMPAIGN")

    def test_drive_unavailable(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(R, "OUT", Path(td) / "camp"):
                info = R.resume_scan(drive_ok=False)
            self.assertEqual(info["SAFE_ACTION"], "STOP — DRIVE UNAVAILABLE")

    def test_active_and_stale_and_malformed_lock(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._prep(td)
            with mock.patch.object(R, "OUT", root):
                R.write_lock()
                info = R.resume_scan(drive_ok=True)
                self.assertEqual(info["SAFE_ACTION"], "WAIT — ACTIVE WRITER LOCK EXISTS")
                # stale
                import time as _t
                lock = root / "CAMPAIGN_LOCK.json"
                body = json.loads(lock.read_text())
                body["timestamp"] = _t.time() - 7 * 3600
                lock.write_text(json.dumps(body))
                info = R.resume_scan(drive_ok=True)
                self.assertEqual(info["WRITER_LOCK"], "STALE")
                # malformed
                lock.write_text("{not json")
                info = R.resume_scan(drive_ok=True)
                self.assertEqual(info["SAFE_ACTION"], "STOP — CHECKPOINT IDENTITY FAILURE")

    def test_identity_verification(self):
        model = V3.Ark018GPT()
        o = R.opt_for(model, 1e-4)
        sc = R.make_scaler(torch.device("cpu"))
        with tempfile.TemporaryDirectory() as td:
            root = self._prep(td)
            ms = root / "matched_sets" / "p31801_b429001" / "GUARDIAN_HYBRID"
            ms.mkdir(parents=True)
            (root / "ENTRY_RECEIPT.json").write_text(json.dumps({
                "task_hashes": {k: R.hjson(TASKS[k]) for k in ("A", "B", "C", "D")},
                "sources": [{"seed": 31801, "model_sha256": R.V3.state_hash(
                    {k: v.clone() for k, v in model.state_dict().items()})}]}))
            payload = {"schema": "arkenstone-ark020-v3-arm-ckpt/v1", "phase_idx": 0,
                       "phase": "B", "phase_step": 100, "registry": {}, "controller": {},
                       "counters": {}, "phase_confirm": {}, "global_confirm": {},
                       "b_streaks": {}, "control_trace": [], "measure_trace": [],
                       "stream_receipts": [], "global_step": 100,
                       "parent_seed": 31801, "b_order_seed": 429001, "c_order_seed": 429003,
                       "d_order_seed": 429005, "arm": "GUARDIAN_HYBRID", "dose_b": 8,
                       "parent_sha": R.V3.state_hash(
                           {k: v.clone() for k, v in model.state_dict().items()}),
                       "task_hash": R.hjson({k: R.hjson(TASKS[k]) for k in ("A", "B", "C", "D")}),
                       "cap16x": 1.0}
            (root / "DOSE_SELECTION.json").write_text(json.dumps(
                {"status": "PASS", "selected_b_slots": 8}))
            R.save_checkpoint(ms / "RESUME.pt", payload, model, o, sc)
            with mock.patch.object(R, "OUT", root):
                info = R.resume_scan(drive_ok=True)
            self.assertEqual(info["CHECKPOINT_IDENTITY"], "PASS")
            self.assertEqual(info["SAFE_ACTION"], "RESUME")
            # without the dose record the scan must NOT claim a full PASS
            (root / "DOSE_SELECTION.json").unlink()
            with mock.patch.object(R, "OUT", root):
                info = R.resume_scan(drive_ok=True)
            self.assertEqual(info["CHECKPOINT_IDENTITY"], "PARTIAL_IDENTITY_CHECK")
            self.assertEqual(info["SAFE_ACTION"], "RESUME")
            (root / "DOSE_SELECTION.json").write_text(json.dumps(
                {"status": "PASS", "selected_b_slots": 8}))
            # corrupt the task hash -> identity failure
            payload["task_hash"] = "deadbeef"
            R.save_checkpoint(ms / "RESUME.pt", payload, model, o, sc)
            with mock.patch.object(R, "OUT", root):
                info = R.resume_scan(drive_ok=True)
            self.assertEqual(info["SAFE_ACTION"], "STOP — CHECKPOINT IDENTITY FAILURE")
            # wrong campaign version -> version mismatch stop
            payload["task_hash"] = R.hjson({k: R.hjson(TASKS[k]) for k in ("A", "B", "C", "D")})
            payload["schema"] = "arkenstone-ark020-v2-arm-ckpt/v1"
            R.save_checkpoint(ms / "RESUME.pt", payload, model, o, sc)
            with mock.patch.object(R, "OUT", root):
                info = R.resume_scan(drive_ok=True)
            self.assertEqual(info["SAFE_ACTION"], "STOP — CAMPAIGN VERSION MISMATCH")


class TestNoVacuousTests(unittest.TestCase):
    def test_no_always_true_constructs_in_v3(self):
        for path in (HERE.parent / "experiments" / "ARK-020-V3" / "run_ark020_v3.py",
                     HERE.parent / "experiments" / "ARK-020-V3" / "ark020_v3_core.py"):
            src = path.read_text(encoding="utf-8")
            self.assertNotIn("or True", src, f"vacuous construct in {path.name}")
            self.assertNotIn("== 7 or", src, f"vacuous construct in {path.name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
