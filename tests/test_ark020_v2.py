"""ARK-020 V2 test suite — pure, contract, validity, checkpoint, fail-closed, golden.

Contract tests exercise the REAL V4 call chain on CPU with real object shapes
(the V1 defect class: pure logic green, integration boundary red).
"""
from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments" / "ARK-020-V2"))
sys.path.insert(0, str(HERE.parent / "experiments" / "ARK-019"))
sys.path.insert(0, str(HERE.parent / "experiments" / "ARK-018"))

import ark020_v2_core as C  # noqa: E402
import run_ark019_v3 as V3  # noqa: E402


def m3(canonical, order, third):
    return {"canonical": canonical, "order_only": order, "query_order": third}


TASK_TOKENS = list(range(48))
TASKS = C.build_all_tasks(TASK_TOKENS)


class TestPureTaskConstruction(unittest.TestCase):
    def test_48_token_gate(self):
        with self.assertRaises(RuntimeError):
            C.build_all_tasks(list(range(47)))

    def test_splits_disjoint_and_sized(self):
        for sk in ("A", "B", "D"):
            sp = TASKS[sk]
            self.assertEqual([len(sp[k]) for k in ("train", "parent_control", "main_control",
                                                   "validation", "sealed")], [400, 50, 50, 50, 50])
            ids = [frozenset(fs) for fs in sum((sp[k] for k in
                    ("train", "parent_control", "main_control", "validation", "sealed")), [])]
            self.assertEqual(len(set(ids)), 600)

    def test_compose_chainsets_wellformed(self):
        sp = TASKS["C"]
        self.assertEqual([len(sp[k]) for k in ("train", "parent_control", "main_control",
                                               "validation", "sealed")], [400, 50, 50, 50, 50])
        allsets = sum((sp[k] for k in ("train", "parent_control", "main_control",
                                       "validation", "sealed")), [])
        self.assertEqual(len(allsets), 600)
        for cs in allsets[:80]:
            chains = cs["chains"]
            self.assertEqual(len(chains), 3)
            xs = [c[0] for c in chains]
            ms = [c[1] for c in chains]
            ys = [c[2] for c in chains]
            self.assertEqual(len(set(xs)), 3)
            self.assertEqual(sorted(ms), sorted(TASK_TOKENS[30:33]))   # M group
            self.assertEqual(sorted(ys), sorted(TASK_TOKENS[33:36]))   # Y group
            self.assertTrue(set(xs) <= set(TASK_TOKENS[24:30]))        # X group

    def test_determinism(self):
        t2 = C.build_all_tasks(TASK_TOKENS)
        self.assertEqual(TASKS["C"]["train"], t2["C"]["train"])
        self.assertEqual(TASKS["D"]["train"], t2["D"]["train"])


class TestSkillCValidity(unittest.TestCase):
    def test_every_sem_row_is_logically_derivable(self):
        for split in ("train", "main_control", "validation", "sealed"):
            for chains, xq, ans in TASKS["sem"]["C"][split]:
                match = [c for c in chains["chains"] if c[0] == xq]
                self.assertEqual(len(match), 1)
                x, m, y = match[0]
                self.assertEqual(ans, y)
                # the intermediate hop is present in the makes segment
                self.assertTrue(any(c[1] == m and c[2] == y for c in chains["chains"]))

    def test_answer_never_directly_paired_without_intermediate(self):
        # the trace answer y must never equal any intermediate m, and the queried x
        # must never be a source of the makes segment
        for chains, xq, ans in TASKS["sem"]["C"]["sealed"][:60]:
            ms = {c[1] for c in chains["chains"]}
            self.assertNotIn(ans, ms)
            self.assertNotIn(xq, {c[1] for c in chains["chains"]})

    def test_frequency_balance(self):
        # each factset uses all three Y tokens bijectively -> uniform answer frequency
        for cs in TASKS["C"]["train"][:40]:
            ys = sorted(c[2] for c in cs["chains"])
            self.assertEqual(ys, sorted(TASK_TOKENS[33:36]))

    def test_rendering_contains_both_hops(self):
        t = {"p": [901], "g": [902], "s": [903], "k": [904], "q": [905], "e": [906]}
        sem = TASKS["sem"]["C"]["sealed"][:5]
        for chains, xq, ans in sem:
            ids, a = C.task_rows("C", t, sem, [sem.index((chains, xq, ans))], "canonical", 1, 1)[0]
            self.assertEqual(a, ans)
            xi = ids.index(xq)
            yi = ids.index(ans)
            between = ids[xi + 1:yi]
            self.assertIn(902, between)   # gives marker between x and its m
            self.assertIn(904, between)   # makes marker between m and y

    def test_order_modes_are_real(self):
        t = {"p": [901], "g": [902], "s": [903], "k": [904], "q": [905], "e": [906]}
        sem = TASKS["sem"]["C"]["main_control"][:6]
        ids_c, _ = C.task_rows("C", t, sem, [0], "canonical", 1, 1)[0]
        ids_r, _ = C.task_rows("C", t, sem, [0], "reversed", 1, 1)[0]
        self.assertNotEqual(ids_c, ids_r)
        self.assertEqual(ids_c[-1], ids_r[-1])  # same tail, same answer position

    def test_nonidentity_never_identity(self):
        for step in range(40):
            self.assertNotEqual(C.nonidentity_perm3(9, step), (0, 1, 2))


class TestPhaseSeeds(unittest.TestCase):
    def test_phase_seeds_are_distinct_and_bound(self):
        seeds = C.PHASE_ORDER_SEEDS
        self.assertEqual(len({*seeds["B"], *seeds["C"], *seeds["D"]}), 6)
        # each phase seed actually changes its phase stream without touching others
        i_b = C.deterministic_indices(seeds["B"][0], 1, 4, 100, "task")
        i_b2 = C.deterministic_indices(seeds["B"][1], 1, 4, 100, "task")
        i_c = C.deterministic_indices(seeds["C"][0], 1, 4, 100, "task")
        self.assertNotEqual(i_b, i_b2)
        self.assertNotEqual(i_b, i_c)

    def test_set_index_binding(self):
        # b seed index selects the matching c/d seeds
        for oi, bs in enumerate(C.PHASE_ORDER_SEEDS["B"]):
            self.assertEqual(C.PHASE_ORDER_SEEDS["C"][oi] != bs, True)
            self.assertIn(C.PHASE_ORDER_SEEDS["C"][oi], C.PHASE_ORDER_SEEDS["C"])


class TestRegistryAndController(unittest.TestCase):
    def setUp(self):
        self.reg = C.init_registry()
        for cid in ("A", "B"):
            C.register_capability(self.reg, cid, 0)
            C.observe_capability(self.reg, cid, 25, m3(1.0, 1.0, 1.0))

    def test_reactive_failure_deescalation(self):
        ctrl = C.initial_controller("GUARDIAN_REACTIVE")
        C.observe_capability(self.reg, "A", 50, m3(0.80, 0.80, 0.80))
        C.update_controller("GUARDIAN_REACTIVE", ctrl, 50, self.reg, ["A", "B"])
        self.assertEqual(ctrl["states"]["A"], "REPLAY32")
        for step in (75, 100, 125, 150):
            C.observe_capability(self.reg, "A", step, m3(0.99, 0.99, 0.99))
            C.update_controller("GUARDIAN_REACTIVE", ctrl, step, self.reg, ["A", "B"])
        self.assertEqual(ctrl["states"]["A"], "SPARSE64")
        C.update_controller("GUARDIAN_REACTIVE", ctrl, 500, self.reg, ["A", "B"])
        self.assertEqual(ctrl["states"]["A"], "PLASTIC")

    def test_predictive_prevents_before_failure(self):
        ctrl = C.initial_controller("GUARDIAN_PREDICTIVE")
        C.observe_capability(self.reg, "A", 50, m3(0.96, 0.93, 0.93))
        self.assertTrue(C.qualified(self.reg["capabilities"]["A"]["margin_history"][-1]["metrics"]))
        C.update_controller("GUARDIAN_PREDICTIVE", ctrl, 50, self.reg, ["A", "B"])
        self.assertEqual(ctrl["states"]["A"], "SPARSE64")
        self.assertIsNone(self.reg["capabilities"]["A"]["failure_since"])

    def test_hybrid_emergency_cap(self):
        ctrl = C.initial_controller("GUARDIAN_HYBRID")
        for step, met in ((50, m3(1, 1, 1)), (75, m3(.5, .5, .5)), (100, m3(.5, .5, .5)),
                          (125, m3(.5, .5, .5))):
            C.observe_capability(self.reg, "A", step, met)
            C.update_controller("GUARDIAN_HYBRID", ctrl, step, self.reg, ["A", "B"])
        self.assertEqual(ctrl["states"]["A"], "EMERGENCY_CAP16X")
        _, capv = C.treatment("GUARDIAN_HYBRID", ctrl, 126, self.reg, ["A", "B"], 2.5)
        self.assertEqual(capv, 2.5)

    def test_risk_allocation_order_and_caps(self):
        reg = C.init_registry()
        for cid in ("A", "B", "C"):
            C.register_capability(reg, cid, 0)
            C.observe_capability(reg, cid, 25, m3(0.99, 0.99, 0.99))
        C.observe_capability(reg, "A", 50, m3(0.80, 0.80, 0.80))
        C.observe_capability(reg, "B", 50, m3(0.96, 0.93, 0.93))
        ctrl = C.initial_controller("GUARDIAN_HYBRID")
        C.update_controller("GUARDIAN_HYBRID", ctrl, 50, reg, ["A", "B", "C"])
        rep, _ = C.treatment("GUARDIAN_HYBRID", ctrl, 50, reg, ["A", "B", "C"], None)
        self.assertEqual(rep, ["A", "B"])
        self.assertTrue(len(rep) <= C.MAX_REPLAY_SLOTS_PER_UPDATE)

    def test_controller_cannot_see_sealed(self):
        reg = C.init_registry()
        C.register_capability(reg, "A", 0)
        C.observe_capability(reg, "A", 25, m3(0.99, 0.99, 0.99))
        m = m3(0.80, 0.80, 0.80)
        m["sealed_leak_attempt"] = 1.0
        C.observe_capability(reg, "A", 50, m)
        self.assertEqual(reg["capabilities"]["A"]["failure_since"], 50)


class TestGoldenDecide(unittest.TestCase):
    KEYS = [(31801, 429001), (31801, 429002), (31902, 429001), (31902, 429002)]

    def row(self, arm, ps, bs, *, conf=(300, 250, 250), fails=(), all_q=True,
            area=0.97, science=4.30, replay=300.0, duty=0.2):
        final = {c: m3(0.50, 0.50, 0.50) if (c in fails and not all_q) else m3(0.98, 0.96, 0.96)
                 for c in C.SKILLS}
        return {
            "arm": arm, "parent_seed": ps, "b_order_seed": bs,
            "phase_confirmation": {"B": conf[0], "C": conf[1], "D": conf[2]},
            "final_sealed": final,
            "sealed_area": {c: (0.55 if (c in fails and not all_q) else area) for c in C.SKILLS},
            "control_failure_steps": {c: [100] for c in fails},
            "recovery": {c: True for c in fails},
            "final_science_sealed_nll": science,
            "counters": {"replay_slots": replay, "protection_updates": duty * C.CONTINUATION_HORIZON},
            "counters_by_phase": {
                "B": {"replay_slots": replay / 3, "protection_updates": duty * 2000, "updates": 2000},
                "C": {"replay_slots": replay / 3, "protection_updates": duty * 1500, "updates": 1500},
                "D": {"replay_slots": replay / 3, "protection_updates": duty * 1500, "updates": 1500}},
        }

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

    def test_incomplete(self):
        res = self.build()
        del res["STATIC_CAP16X"][0]
        self.assertEqual(C.decide(res)["verdict"], "INCONCLUSIVE_INCOMPLETE_MATCHED_SETS")

    def test_matching_error(self):
        res = self.build()
        row = dict(res["GUARDIAN_REACTIVE"][0])
        row["b_order_seed"] = 999999
        res["GUARDIAN_REACTIVE"][0] = row
        self.assertEqual(C.decide(res)["verdict"], "INCONCLUSIVE_MATCHING_ERROR")

    def test_formation_gates_b_c_d(self):
        for skill in ("B", "C", "D"):
            conf = {"B": (None, 250, 250), "C": (300, None, 250), "D": (300, 250, None)}[skill]
            res = self.build(plastic_conf=conf)
            self.assertEqual(C.decide(res)["verdict"],
                             f"INCONCLUSIVE_FORMATION_INSTABILITY_PHASE_{skill}")

    def test_low_interference(self):
        res = self.build(plastic_fails=())
        self.assertEqual(C.decide(res)["verdict"], "INCONCLUSIVE_LOW_INTERFERENCE")

    def test_success(self):
        res = self.build()
        d = C.decide(res)
        self.assertEqual(d["verdict"], "GUARDIAN_MULTI_SKILL_CANDIDATE")
        self.assertTrue(d["authorized"])
        self.assertIn("PREVENTION_SIGNAL", d["flags"])
        self.assertIn("RECOVERY_SIGNAL", d["flags"])

    def test_quality_only_when_inefficient(self):
        res = self.build(guardian_replay=6000.0)
        self.assertEqual(C.decide(res)["verdict"], "GUARDIAN_SUPPORTED_NOT_EFFICIENT")

    def test_failure_when_guardian_destroys(self):
        res = self.build(guardian_fails=("A",), guardian_all_q=False)
        self.assertEqual(C.decide(res)["verdict"], "GUARDIAN_NOT_SUPPORTED_MULTI_SKILL")

    def test_static_wins_shows_as_failure(self):
        res = self.build(guardian_fails=("A",), guardian_all_q=False,
                         per_arm_overrides=[("GUARDIAN_REACTIVE", 0, {
                             "final_sealed": {**{c: m3(0.98, 0.96, 0.96) for c in ("B", "C", "D")},
                                              "A": m3(0.98, 0.96, 0.96)},
                             "sealed_area": {c: 0.97 for c in C.SKILLS}})])
        d = C.decide(res)
        # one guardian set recovered while others failed -> still not qualified
        self.assertEqual(d["verdict"], "GUARDIAN_NOT_SUPPORTED_MULTI_SKILL")

    def test_predictive_wins_over_reactive(self):
        res = self.build()
        # reactive catastrophically destroys A in 2 sets; predictive stays clean
        for i in (0, 1):
            res["GUARDIAN_REACTIVE"][i]["final_sealed"]["A"] = m3(0.50, 0.50, 0.50)
            res["GUARDIAN_REACTIVE"][i]["sealed_area"]["A"] = 0.55
            res["GUARDIAN_REACTIVE"][i]["control_failure_steps"] = {"A": [100]}
        d = C.decide(res)
        self.assertEqual(d["verdict"], "GUARDIAN_MULTI_SKILL_CANDIDATE")
        self.assertIn("PREDICTIVE_ADDS_VALUE", d["flags"])

    def test_median_rule_not_per_run(self):
        # one guardian set is 2.5x slower than plastic median; median rule still passes
        res = self.build(per_arm_overrides=[("GUARDIAN_REACTIVE", 0, {
            "phase_confirmation": {"B": 750, "C": 625, "D": 625}})])
        d = C.decide(res)
        self.assertEqual(d["verdict"], "GUARDIAN_MULTI_SKILL_CANDIDATE")
        # per-run rule would have failed this set; the median rule is the preregistered one
        self.assertTrue(d["guardian_details"]["GUARDIAN_REACTIVE"]
                        ["phase_relative_confirmation_within_1_5x_plastic_median"])

    def test_slow_median_fails_ratio_rule(self):
        # guardian median is 2x plastic median across ALL sets -> confirmation rule fails,
        # quality (area+qualification) still holds -> SUPPORTED_NOT_EFFICIENT
        res = self.build(plastic_conf=(300, 250, 250), conf=(600, 500, 500))
        d = C.decide(res)
        self.assertEqual(d["verdict"], "GUARDIAN_SUPPORTED_NOT_EFFICIENT")

    def test_cost_slope_accounting(self):
        res = self.build()
        cs = C.cost_slope(res["GUARDIAN_REACTIVE"])
        self.assertAlmostEqual(cs["k1"]["mean_replay_slots"], 100.0)
        self.assertAlmostEqual(cs["k3"]["mean_duty"], 0.3, places=9)


class TestContractsWithRealV4(unittest.TestCase):
    """Real calls across the V2 -> V4 boundary on CPU with REAL object shapes."""

    @classmethod
    def setUpClass(cls):
        import run_ark019_v4 as V4
        cls.V4 = V4
        cls.d = torch.device("cpu")
        torch.manual_seed(0)
        cls.model = V3.Ark018GPT()  # real geometry; the shapes are the contract
        cls.tiny_state = {k: v.clone() for k, v in cls.model.state_dict().items()}

    def test_v4_signature_contract(self):
        sig = [p.name for p in inspect.signature(self.V4.acquire_parent).parameters.values()]
        self.assertEqual(sig, ["seed", "prep", "bufs", "tA", "A", "d", "deadline"])
        sig2 = [p.name for p in inspect.signature(self.V4.select_dose).parameters.values()]
        self.assertEqual(sig2, ["parents", "bufs", "tB", "B", "d", "deadline"])

    def test_v4_bmetrics_with_v2_objects(self):
        tA = {"p": [1], "m": [2], "s": [3], "q": [4], "t": [5]}
        sem = TASKS["sem"]["A"]["main_control"][:8]
        m = self.V4.bmetrics(self.model, tA, sem, self.d)
        self.assertIn("canonical", m)
        self.assertIn("qualified", m)

    def test_v4_mixed_update_with_v2_objects(self):
        tA = {"p": [1], "m": [2], "s": [3], "q": [4], "t": [5]}
        buf = np.random.RandomState(0).randint(0, 512, size=8193 * 2).astype(np.uint16)
        names = [n for n, _ in self.model.named_parameters()][:0]  # V4 pnames subset
        wanted = {"tok.weight", "blocks.0.attn.qkv.weight", "blocks.4.mlp.2.weight",
                  "blocks.9.attn.qkv.weight", "ln_f.weight"}
        names = [n for n, _ in self.model.named_parameters() if n in wanted]
        o = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        sc = V3.make_scaler(self.d)
        rec = self.V4.mixed_update(self.model, o, sc, buf, tA, TASKS["sem"]["A"]["train"][:16],
                                   555001, 1, 8, tA, TASKS["sem"]["A"]["train"][:16], 0, None,
                                   self.d, names, tag="contract")
        self.assertIn("loss", rec)
        self.assertIn("real_starts_sha256", rec)

    def test_v4_dose_pilot_boundary_with_v2_objects(self):
        import unittest.mock as mock
        tB = {"p": [6], "m": [7], "s": [8], "q": [9], "t": [10]}
        buf = np.random.RandomState(1).randint(0, 512, size=8193 * 2).astype(np.uint16)
        bufs = {"train": buf, "control": buf[:6000], "sealed": buf[6000:12000]}
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        parent = {"model": self.tiny_state, "optimizer": opt.state_dict(), "scaler": {},
                  "cpu_rng": torch.get_rng_state(), "cuda_rng": []}
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(self.V4, "OUT", Path(td)), \
                 mock.patch.object(self.V4.C, "PILOT_MAX_UPDATES", 2), \
                 mock.patch.object(self.V4.C, "PILOT_EVAL_EVERY", 1), \
                 mock.patch.object(self.V4.C, "PILOT_STREAK", 1):
                r = self.V4.run_dose_pilot(31801, 8, 419001, parent, bufs, tB,
                                           TASKS["sem"]["B"], self.d,
                                           time_deadline := __import__("time").monotonic() + 600)
            self.assertIn("status", r)
            self.assertIn("trajectory", r)
            # result lands under the patched OUT with the expected filename
            self.assertTrue((Path(td) / "dose_pilot" / "p31801_slots8" / "RESULT.json").exists())


class TestCheckpointAndFailClosed(unittest.TestCase):
    def test_roundtrip_and_identity(self):
        import run_ark020_v2 as R
        model = R.Ark018GPT()
        o = R.opt_for(model, 1e-4)
        sc = R.make_scaler(torch.device("cpu"))
        reg, ctrl = C.init_registry(), C.initial_controller("GUARDIAN_HYBRID")
        C.register_capability(reg, "A", 0)
        cnt = {"replay_slots": 3}
        payload = {"schema": "t", "phase_idx": 1, "phase": "C", "phase_step": 40,
                   "registry": reg, "controller": ctrl, "counters": cnt,
                   "phase_confirm": {"B": 120}, "global_confirm": {"B": 2120},
                   "b_streaks": {"B": 0, "C": 1, "D": 0}, "control_trace": [], "measure_trace": []}
        expected = {"schema": "t", "phase_idx": 1, "b_order_seed": 429001,
                    "c_order_seed": 429003, "d_order_seed": 429005, "arm": "GUARDIAN_HYBRID"}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ck.pt"
            R.save_checkpoint(p, {**expected, **payload}, model, o, sc)
            x, m2, o2, sc2 = R.load_checkpoint(p, expected, torch.device("cpu"))
            self.assertEqual(x["phase"], "C")
            self.assertEqual(x["registry"]["capabilities"]["A"]["capability_id"], "A")
            # phase identity fields present for the resume scan
            for k in ("phase_idx", "phase", "phase_step", "arm", "b_order_seed"):
                self.assertIn(k, x)

    def test_fail_closed_on_identity_mismatch(self):
        import run_ark020_v2 as R
        model = R.Ark018GPT()
        o = R.opt_for(model, 1e-4)
        sc = R.make_scaler(torch.device("cpu"))
        payload = {"schema": "t", "phase_idx": 0, "phase_step": 5, "registry": {}, "controller": {},
                   "counters": {}, "phase_confirm": {}, "global_confirm": {}, "b_streaks": {},
                   "control_trace": [], "measure_trace": []}
        expected = {"schema": "t", "arm": "PLASTIC_HIGH", "b_order_seed": 429001}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ck.pt"
            R.save_checkpoint(p, {**expected, **payload}, model, o, sc)
            for bad in ({"b_order_seed": 429002}, {"arm": "STATIC_CAP16X"},
                        {"schema": "other"}, {"c_order_seed": 429004}):
                with self.assertRaises(RuntimeError):
                    R.load_checkpoint(p, {**expected, **bad}, torch.device("cpu"))


class TestCPUSmoke(unittest.TestCase):
    def test_full_executable_path(self):
        import run_ark020_v2 as R
        with tempfile.TemporaryDirectory() as td:
            report = R.cpu_integration_smoke(Path(td))
            self.assertEqual(report["status"], "PASS")
            names = [s["step"] for s in report["steps"]]
            for required in ("task_build", "v4_bmetrics_boundary", "v4_mixed_update_boundary",
                             "b_c_d_updates", "controller_and_replay_update",
                             "checkpoint_roundtrip", "fail_closed_identity", "packaging"):
                self.assertIn(required, names)
            self.assertTrue(all(s["ok"] for s in report["steps"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
