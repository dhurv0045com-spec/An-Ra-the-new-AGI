"""Pure (torch-free) tests for ARK-020 core logic."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments" / "ARK-020"))

import ark020_core as C  # noqa: E402


def m3(canonical, order, third, key="query_order"):
    return {"canonical": canonical, "order_only": order, key: third}


def cap_metrics(canonical, order, third, key="query_order"):
    return m3(canonical, order, third, key)


class TestTaskConstruction(unittest.TestCase):
    def test_48_token_gate(self):
        with self.assertRaises(RuntimeError):
            C.build_all_tasks(list(range(47)))
        with self.assertRaises(RuntimeError):
            C.build_all_tasks([1] * 48)

    def test_splits_disjoint_and_sized(self):
        tasks = C.build_all_tasks(list(range(48)))
        for sk in ("A", "B", "D"):
            sp = tasks[sk]
            self.assertEqual(len(sp["train"]), 400)
            self.assertEqual(len(sp["parent_control"]), 50)
            self.assertEqual(len(sp["main_control"]), 50)
            self.assertEqual(len(sp["validation"]), 50)
            self.assertEqual(len(sp["sealed"]), 50)
            ids = [frozenset(fs) for fs in
                   sp["train"] + sp["parent_control"] + sp["main_control"] + sp["validation"] + sp["sealed"]]
            self.assertEqual(len(ids), 600)
            self.assertEqual(len(set(ids)), 600)  # no factset appears in two splits

    def test_cycle_train_sealed_disjoint(self):
        tasks = C.build_all_tasks(list(range(48)))
        cyc = tasks["C"]
        self.assertEqual(len(cyc["train_keys"]), 9)
        self.assertEqual(len(cyc["sealed_keys"]), 3)
        self.assertEqual(set(cyc["train_keys"]) & set(cyc["sealed_keys"]), set())
        # cycle closes over all 12 tokens
        self.assertEqual(set(cyc["cycle"].keys()), set(range(24, 36)))
        # successor is a permutation cycle: following it 12 steps returns to start
        t = cyc["train_keys"][0]
        seen = [t]
        for _ in range(12):
            t = cyc["cycle"][t]
            seen.append(t)
        self.assertEqual(seen[0], seen[-1])
        self.assertEqual(len(set(seen[:-1])), 12)

    def test_determinism(self):
        t1 = C.build_all_tasks(list(range(48)))
        t2 = C.build_all_tasks(list(range(48)))
        self.assertEqual(t1["C"]["cycle"], t2["C"]["cycle"])
        self.assertEqual(t1["A"]["train"], t2["A"]["train"])
        self.assertEqual(C.deterministic_indices(7, 3, 5, 100, "x"),
                         C.deterministic_indices(7, 3, 5, 100, "x"))
        self.assertNotEqual(C.deterministic_indices(7, 3, 5, 100, "x"),
                            C.deterministic_indices(8, 3, 5, 100, "x"))


class TestMetricsAndRegistry(unittest.TestCase):
    def test_qualified_healthy(self):
        q = m3(0.95, 0.90, 0.90)
        self.assertTrue(C.qualified(q))
        self.assertTrue(C.healthy(q))
        edge = m3(0.90, 0.85, 0.85)
        self.assertTrue(C.qualified(edge))
        self.assertFalse(C.healthy(edge))
        bad = m3(1.0, 1.0, 0.84)
        self.assertFalse(C.qualified(bad))
        d = m3(0.95, 0.95, 0.95, key="distractor")
        self.assertTrue(C.qualified(d))

    def test_observe_failure_and_recovery(self):
        reg = C.init_registry()
        st = C.register_capability(reg, "A", 0)
        for step in (25, 50, 75):
            C.observe_capability(reg, "A", step, m3(0.99, 0.99, 0.99))
        self.assertEqual(st["failure_since"], None)
        self.assertTrue(st["prevented"])
        C.observe_capability(reg, "A", 100, m3(0.80, 0.80, 0.80))
        self.assertEqual(st["failure_since"], 100)
        self.assertFalse(st["prevented"])
        for step in (125, 150, 175, 200):
            C.observe_capability(reg, "A", step, m3(0.99, 0.99, 0.99))
        self.assertEqual(st["failure_since"], None)
        self.assertEqual(st["recovered_events"], 1)

    def test_degradation_rate(self):
        reg = C.init_registry()
        C.register_capability(reg, "B", 0)
        C.observe_capability(reg, "B", 100, m3(1.0, 1.0, 1.0))
        C.observe_capability(reg, "B", 200, m3(0.98, 1.0, 1.0))
        C.observe_capability(reg, "B", 300, m3(0.96, 1.0, 1.0))
        st = reg["capabilities"]["B"]
        self.assertAlmostEqual(st["degradation_rate"], (0.96 - 1.0) / 200 * 100, places=9)
        # rate -0.02/100 does not cross the limit and margin 0.96 is not < 0.95: no warning
        self.assertFalse(C.warning_signal(st))
        C.observe_capability(reg, "B", 400, m3(0.94, 1.0, 1.0))
        self.assertTrue(C.warning_signal(reg["capabilities"]["B"]))  # margin 0.94 < 0.95

    def test_risk_order_deterministic(self):
        reg = C.init_registry()
        for cid in ("A", "B", "C"):
            C.register_capability(reg, cid, 0)
        C.observe_capability(reg, "A", 25, m3(0.96, 0.97, 0.97))
        C.observe_capability(reg, "B", 25, m3(0.93, 0.99, 0.99))
        C.observe_capability(reg, "C", 25, m3(0.93, 0.99, 0.99))
        self.assertEqual(C.risk_order(reg, ["A", "B", "C"]), ["B", "C", "A"])


class TestController(unittest.TestCase):
    def setUp(self):
        self.reg = C.init_registry()
        for cid in ("A", "B"):
            C.register_capability(self.reg, cid, 0)
            C.observe_capability(self.reg, cid, 25, m3(1.0, 1.0, 1.0))

    def test_reactive_formal_failure_and_deescalation(self):
        ctrl = C.initial_controller("GUARDIAN_REACTIVE")
        C.observe_capability(self.reg, "A", 50, m3(0.80, 0.80, 0.80))
        C.update_controller("GUARDIAN_REACTIVE", ctrl, 50, self.reg, ["A", "B"])
        self.assertEqual(ctrl["states"]["A"], "REPLAY32")
        self.assertEqual(ctrl["states"].get("B", "PLASTIC"), "PLASTIC")
        for i, step in enumerate((75, 100, 125, 150), start=1):
            C.observe_capability(self.reg, "A", step, m3(0.99, 0.99, 0.99))
            C.update_controller("GUARDIAN_REACTIVE", ctrl, step, self.reg, ["A", "B"])
        self.assertEqual(ctrl["states"]["A"], "SPARSE64")  # de-escalated after 4-healthy
        # without a floor, the next healthy observation returns to PLASTIC
        C.update_controller("GUARDIAN_REACTIVE", ctrl, 175, self.reg, ["A", "B"])
        self.assertEqual(ctrl["states"]["A"], "PLASTIC")
        # with a post-phase floor, SPARSE64 is held until the floor expires
        C.observe_capability(self.reg, "B", 200, m3(0.80, 0.80, 0.80))
        C.update_controller("GUARDIAN_REACTIVE", ctrl, 200, self.reg, ["A", "B"])
        C.set_post_phase_floor(ctrl, ["B"], 200)
        self.assertEqual(ctrl["states"]["B"], "REPLAY32")
        for step in (225, 250, 275, 300):
            C.observe_capability(self.reg, "B", step, m3(0.99, 0.99, 0.99))
            C.update_controller("GUARDIAN_REACTIVE", ctrl, step, self.reg, ["A", "B"])
        self.assertEqual(ctrl["states"]["B"], "SPARSE64")
        C.update_controller("GUARDIAN_REACTIVE", ctrl, 200 + C.POST_PHASE_SPARSE_FLOOR - 25,
                            self.reg, ["A", "B"])
        self.assertEqual(ctrl["states"]["B"], "SPARSE64")  # floor holds
        C.update_controller("GUARDIAN_REACTIVE", ctrl, 200 + C.POST_PHASE_SPARSE_FLOOR + 25,
                            self.reg, ["A", "B"])
        self.assertEqual(ctrl["states"]["B"], "PLASTIC")

    def test_predictive_fires_before_failure(self):
        ctrl = C.initial_controller("GUARDIAN_PREDICTIVE")
        # margin dips below 0.95 but still qualified (>= 0.90 third, >= 0.90 order)
        C.observe_capability(self.reg, "A", 50, m3(0.96, 0.93, 0.93))
        self.assertTrue(C.qualified(self.reg["capabilities"]["A"]["margin_history"][-1]["metrics"]))
        C.update_controller("GUARDIAN_PREDICTIVE", ctrl, 50, self.reg, ["A", "B"])
        self.assertEqual(ctrl["states"]["A"], "SPARSE64")
        self.assertEqual(self.reg["capabilities"]["A"]["failure_since"], None)  # prevention, not recovery

    def test_reactive_ignores_warning(self):
        ctrl = C.initial_controller("GUARDIAN_REACTIVE")
        C.observe_capability(self.reg, "A", 50, m3(0.96, 0.93, 0.93))
        C.update_controller("GUARDIAN_REACTIVE", ctrl, 50, self.reg, ["A", "B"])
        self.assertEqual(ctrl["states"].get("A", "PLASTIC"), "PLASTIC")

    def test_hybrid_emergency_cap(self):
        ctrl = C.initial_controller("GUARDIAN_HYBRID")
        C.update_controller("GUARDIAN_HYBRID", ctrl, 50, self.reg, ["A", "B"])
        C.observe_capability(self.reg, "A", 75, m3(0.5, 0.5, 0.5))
        C.update_controller("GUARDIAN_HYBRID", ctrl, 75, self.reg, ["A", "B"])
        self.assertEqual(ctrl["states"]["A"], "REPLAY32")
        C.observe_capability(self.reg, "A", 100, m3(0.5, 0.5, 0.5))
        C.update_controller("GUARDIAN_HYBRID", ctrl, 100, self.reg, ["A", "B"])
        C.observe_capability(self.reg, "A", 125, m3(0.5, 0.5, 0.5))
        C.update_controller("GUARDIAN_HYBRID", ctrl, 125, self.reg, ["A", "B"])
        self.assertEqual(ctrl["states"]["A"], "EMERGENCY_CAP16X")
        for step in (150, 175, 200, 225):
            C.observe_capability(self.reg, "A", step, m3(0.99, 0.99, 0.99))
            C.update_controller("GUARDIAN_HYBRID", ctrl, step, self.reg, ["A", "B"])
        self.assertEqual(ctrl["states"]["A"], "SPARSE64")

    def test_treatment_static_and_guardian(self):
        self.assertEqual(C.treatment("PLASTIC_HIGH", {}, 10, {}, ["A"], None), ([], None))
        self.assertEqual(C.treatment("STATIC_CAP16X", {}, 10, {}, ["A"], 2.0), ([], 2.0))
        r64 = [C.treatment("STATIC_REPLAY_1OF64", {}, s, {}, ["A", "B", "C"], None)[0]
               for s in range(1, 9)]
        # steps 2,4,6,8 replay; rotation index = (step//2) % 3 -> B, C, A, B
        self.assertEqual(r64, [[], ["B"], [], ["C"], [], ["A"], [], ["B"]])
        r32 = [C.treatment("STATIC_REPLAY_1OF32", {}, s, {}, ["A", "B"], None)[0]
               for s in range(1, 5)]
        self.assertEqual(r32, [["A"], ["B"], ["A"], ["B"]])
        # guardian: risk-ordered requesting capabilities, max 2 slots
        reg = C.init_registry()
        for cid in ("A", "B", "C"):
            C.register_capability(reg, cid, 0)
            C.observe_capability(reg, cid, 25, m3(0.99, 0.99, 0.99))
        C.observe_capability(reg, "A", 50, m3(0.80, 0.80, 0.80))  # failure -> REPLAY32
        C.observe_capability(reg, "B", 50, m3(0.96, 0.93, 0.93))  # warning (predictive)
        ctrl = C.initial_controller("GUARDIAN_HYBRID")
        C.update_controller("GUARDIAN_HYBRID", ctrl, 50, reg, ["A", "B", "C"])
        self.assertEqual(ctrl["states"]["A"], "REPLAY32")
        self.assertEqual(ctrl["states"]["B"], "SPARSE64")
        rep, cap = C.treatment("GUARDIAN_HYBRID", ctrl, 50, reg, ["A", "B", "C"], 2.5)
        self.assertEqual(rep, ["A", "B"])  # A lowest margin first; C not requesting
        self.assertIsNone(cap)
        # emergency cap activates cap value
        ctrl["states"]["A"] = "EMERGENCY_CAP16X"
        rep, cap = C.treatment("GUARDIAN_HYBRID", ctrl, 51, reg, ["A", "B", "C"], 2.5)
        self.assertEqual(cap, 2.5)
        self.assertIn("A", rep)

    def test_controller_never_sees_sealed(self):
        # structural pin: observe/update signatures take control metrics only; a dict
        # carrying extra sealed keys must not influence decisions (extra keys ignored).
        reg = C.init_registry()
        C.register_capability(reg, "A", 0)
        C.observe_capability(reg, "A", 25, m3(0.99, 0.99, 0.99))
        m = m3(0.80, 0.80, 0.80)
        m["sealed_leak_attempt"] = 1.0
        C.observe_capability(reg, "A", 50, m)
        self.assertEqual(reg["capabilities"]["A"]["failure_since"], 50)


class TestDecide(unittest.TestCase):
    def setUp(self):
        self.keys = [(31801, 429001), (31801, 429002), (31902, 429001), (31902, 429002)]

    def run_row(self, arm, ps, bs, *, conf=(1200, 900, 900), fails=(), all_q=True,
                area=0.97, science=4.30, replay=300.0, duty=0.2, recovery=None):
        fails = list(fails)
        final = {}
        for c in C.SKILLS:
            destroyed = (c in fails) and not all_q
            final[c] = m3(0.50, 0.50, 0.50) if (c in fails and not all_q) else m3(0.98, 0.96, 0.96)
        return {
            "arm": arm, "parent_seed": ps, "order_seed": bs,
            "phase_confirmation": {"B": conf[0], "C": conf[1], "D": conf[2]},
            "final_sealed": final,
            "sealed_area": {c: (0.55 if (c in fails and not all_q) else area) for c in C.SKILLS},
            "control_failure_steps": {c: [100] for c in fails},
            "recovery": recovery if recovery is not None else {c: True for c in fails},
            "final_science_sealed_nll": science,
            "counters": {"replay_slots": replay, "protection_updates": duty * C.CONTINUATION_HORIZON,
                         "real_slots": 32 * C.CONTINUATION_HORIZON, "task_slots": 12 * C.CONTINUATION_HORIZON},
            "counters_by_phase": {
                "B": {"replay_slots": replay / 3, "protection_updates": duty * 2000, "updates": 2000},
                "C": {"replay_slots": replay / 3, "protection_updates": duty * 1500, "updates": 1500},
                "D": {"replay_slots": replay / 3, "protection_updates": duty * 1500, "updates": 1500}},
        }

    def build(self, *, plastic_fails=("A",), guardian_fails=(), guardian_replay=300.0,
              guardian_all_q=True, conf=(1200, 900, 900), plastic_conf=None):
        out = {}
        for arm in C.ARMS:
            rows = []
            for (ps, bs) in self.keys:
                if arm == "PLASTIC_HIGH":
                    destroyed = bool(plastic_fails)
                    rows.append(self.run_row(arm, ps, bs, conf=plastic_conf or conf,
                                             fails=plastic_fails, all_q=False,
                                             area=0.60 if destroyed else 0.97,
                                             science=4.30, replay=0.0, duty=0.0))
                elif arm.startswith("STATIC"):
                    rows.append(self.run_row(arm, ps, bs, conf=conf, fails=(),
                                             all_q=True, area=0.97, science=4.32,
                                             replay=5000.0 if arm != "STATIC_REPLAY_1OF64" else 2500.0,
                                             duty=1.0 if arm == "STATIC_CAP16X" else 1.0))
                else:
                    rows.append(self.run_row(arm, ps, bs, conf=conf, fails=guardian_fails,
                                             all_q=guardian_all_q, area=0.96, science=4.35,
                                             replay=guardian_replay, duty=0.3))
            out[arm] = rows
        return out

    def test_incomplete_sets(self):
        res = self.build()
        del res["STATIC_CAP16X"][0]
        d = C.decide(res)
        self.assertEqual(d["verdict"], "INCONCLUSIVE_INCOMPLETE_MATCHED_SETS")

    def test_matching_error(self):
        res = self.build()
        row = dict(res["GUARDIAN_REACTIVE"][0])
        row["order_seed"] = 999999
        res["GUARDIAN_REACTIVE"][0] = row
        d = C.decide(res)
        self.assertEqual(d["verdict"], "INCONCLUSIVE_MATCHING_ERROR")

    def test_formation_gate(self):
        res = self.build(plastic_conf=(1200, 900, None))
        d = C.decide(res)
        self.assertEqual(d["verdict"], "INCONCLUSIVE_FORMATION_INSTABILITY_PHASE_D")

    def test_low_interference(self):
        res = self.build(plastic_fails=())
        d = C.decide(res)
        self.assertEqual(d["verdict"], "INCONCLUSIVE_LOW_INTERFERENCE")

    def test_guardian_success(self):
        res = self.build()
        d = C.decide(res)
        self.assertEqual(d["verdict"], "GUARDIAN_MULTI_SKILL_CANDIDATE")
        self.assertTrue(d["authorized"])
        self.assertIn("PREVENTION_SIGNAL", d["flags"])

    def test_quality_only_when_inefficient(self):
        res = self.build(guardian_replay=6000.0)
        d = C.decide(res)
        self.assertEqual(d["verdict"], "GUARDIAN_SUPPORTED_NOT_EFFICIENT")

    def test_failure_when_guardian_destroys(self):
        res = self.build(guardian_fails=("A",), guardian_all_q=False)
        d = C.decide(res)
        self.assertEqual(d["verdict"], "GUARDIAN_NOT_SUPPORTED_MULTI_SKILL")


class TestAccounting(unittest.TestCase):
    def test_dedupe(self):
        rows = [{"step": 3, "v": "a"}, {"step": 1, "v": "b"}, {"step": 3, "v": "c"}, {"step": 9, "v": "d"}]
        self.assertEqual([r["step"] for r in C.dedupe_trajectory(rows)], [1, 3, 9])
        self.assertEqual(C.dedupe_trajectory(rows, 5)[-1]["v"], "c")

    def test_cost_slope(self):
        runs = [self.row() for _ in range(4)]
        cs = C.cost_slope(runs)
        self.assertAlmostEqual(cs["k1"]["mean_replay_slots"], 100.0)
        self.assertAlmostEqual(cs["k2"]["mean_duty"], 0.3, places=9)

    def row(self):
        return {"counters_by_phase": {
            "B": {"replay_slots": 100, "protection_updates": 600, "updates": 2000},
            "C": {"replay_slots": 200, "protection_updates": 450, "updates": 1500},
            "D": {"replay_slots": 300, "protection_updates": 300, "updates": 1500}}}


class TestRendering(unittest.TestCase):
    def setUp(self):
        # fake templates: single-int "tokens"
        self.t = {"p": [900], "m": [901], "s": [902], "q": [903], "t": [904]}
        self.facts = ((11, 21), (12, 22), (13, 23))

    def test_binding_canonical_shape_and_answer(self):
        rows = C.task_rows("A", self.t, [(list(self.facts), 12, 22)], [0], "canonical", 1, 1)
        ids, ans = rows[0]
        self.assertEqual(ans, 22)
        self.assertEqual(ids[-1], 904)  # prompt ends with the query tail
        self.assertEqual(ids[-2], 12)   # queried key immediately before tail
        self.assertEqual(ids[0], 900)

    def test_binding_reversed_order(self):
        ids, _ = C.task_rows("A", self.t, [(list(self.facts), 12, 22)], [0], "reversed", 1, 1)[0]
        # first fact rendered should be the last fact (13 means 23)
        self.assertEqual(ids[1:4], [13, 901, 23])

    def test_query_order_moves_query_last(self):
        f = list(self.facts)
        perm = C.query_order_perm(f, 11)  # query is fact 0 -> moved to end
        self.assertEqual(perm[-1], 0)
        self.assertEqual(len(set(perm)), 3)

    def test_nonidentity_is_never_identity(self):
        for step in range(50):
            perm = C.nonidentity_perm3(5, step)
            self.assertNotEqual(perm, (0, 1, 2))

    def test_truncation_keeps_tail(self):
        long_t = {"p": [900] * 300, "m": [901], "s": [902], "q": [903], "t": [904]}
        ids, _ = C.task_rows("A", long_t, [(list(self.facts), 12, 22)], [0], "canonical", 1, 1)[0]
        self.assertEqual(len(ids), C.MAX_PROMPT_IDS)
        self.assertEqual(ids[-1], 904)

    def test_cycle_distractor_counts(self):
        tc = {"p": [910], "m": [911], "s": [912]}
        sem = [(11, 12), (12, 13), (13, 14)]
        wrapped = [(sem, 11, 12)]
        counts = {}
        for mode in ("canonical", "reversed", "query_order"):
            ids, ans = C.task_rows("C", tc, wrapped, [0], mode, 3, 7)[0]
            self.assertEqual(ans, 12)
            counts[mode] = ids.count(912)  # separator count == distractor count
        self.assertEqual(counts, {"canonical": 0, "reversed": 1, "query_order": 3})
        self.assertEqual(C.CYCLE_DISTRACTORS["augmented"], 1)
        self.assertEqual(C.CYCLE_DISTRACTORS["nonidentity"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
