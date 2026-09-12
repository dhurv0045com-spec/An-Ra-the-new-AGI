import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "COLAB"))

def load_runner(number):
    path = REPO / "experiments" / f"ARK-{number:03d}" / f"run_ark{number:03d}.py"
    spec = importlib.util.spec_from_file_location(f"test_ark{number:03d}", path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module); return module

def metrics(area, final=None, g90=None, drop=None):
    return {"AREA": area, "FINAL": area if final is None else final, "G90_CONFIRM": g90, "DROP90_CONFIRM": drop}

def schedule_row(m, areas, drops=None, status="EXECUTED", steps=8000):
    drops = drops or [None] * len(areas)
    return {"acquisition_seed": 909, "continuation_seed": 2702, "status": status, "schedules": {n: {"completed_steps": steps, "sealed_metrics": metrics(a, drop=d)} for (n, _), a, d in zip(m.SCHEDULES, areas, drops)}}

def triplet(seed, order, gain=False, steps=12000):
    def arm(t3, t2, g90=None):
        return {"completed_steps": steps, "t3_control_metrics": metrics(.5, g90=g90), "t3_sealed_metrics": metrics(t3), "t2_control_metrics": metrics(t2), "t2_sealed_metrics": metrics(t2)}
    return {"status": "TRIPLET_EXECUTED", "acquisition_seed": seed, "order_seed": order, "arms": {"FIXED_HIGH": arm(.5,.5), "FIXED_LOW": arm(.2,.8), "ADAPTIVE_HIGH_LOW": arm(.46 if gain else .2, .61 if gain else .5)}}

class RecordingWriter:
    saves = []
    def __init__(self, *a, **k): type(self).saves = []
    def save(self, filename, payload): type(self).saves.append((filename, copy.deepcopy(payload)))

class CampaignTests(unittest.TestCase):
    def test_ark012_complete_and_protection(self):
        m = load_runner(12)
        good = schedule_row(m, [.4,.39,.4,.41,.42,.43], [None,None,None,None,None,8])
        bad = schedule_row(m, [.4,.39,.4,.41,.42,.43], [None,None,None,None,8,8])
        good["schedules"]["SWITCH_90"]["sealed_metrics"]["G90_CONFIRM"] = 600
        good["schedules"]["SWITCH_95"]["sealed_metrics"]["G90_CONFIRM"] = 600
        self.assertEqual(m._summarize([good, dict(good, acquisition_seed=1010)])["verdict"], "STATE_THRESHOLD_SUPPORTED_SCREEN")
        self.assertNotEqual(m._summarize([bad, dict(bad, acquisition_seed=1010)])["verdict"], "STATE_THRESHOLD_SUPPORTED_SCREEN")
        self.assertEqual(m._summarize([schedule_row(m,[.5]*6,steps=1)])["event_sources_executed"], 0)

    def test_ark013_requires_final_area_agreement_and_horizon(self):
        m = load_runner(13)
        rows = [triplet(s,o) for s in m.ACQ_SEEDS for o in m.ORDER_SEEDS]
        for r in rows:
            r["arms"]["ADAPTIVE_HIGH_LOW"]["t3_sealed_metrics"]["AREA"] = .46
            r["arms"]["ADAPTIVE_HIGH_LOW"]["t2_sealed_metrics"]["AREA"] = .61
        self.assertNotEqual(m._summarize(rows)["verdict"], "ADAPTIVE_PARETO_IMPROVEMENT")
        rows = [triplet(s,o,True) for s in m.ACQ_SEEDS for o in m.ORDER_SEEDS]
        for r in rows: r["arms"]["FIXED_HIGH"]["t3_control_metrics"]["G90_CONFIRM"] = 12000
        self.assertEqual(m._summarize(rows)["verdict"], "ADAPTIVE_PARETO_IMPROVEMENT")
        self.assertEqual(m._summarize([triplet(1717,6801,steps=1)])["matched_triplets"], 0)

    def test_campaign_entry_reserves_preserve_skipped_opportunities(self):
        for number, reserve, count in ((12, 35, 3), (13, 50, 4)):
            with self.subTest(experiment=number):
                m = load_runner(number)
                ctx = SimpleNamespace(minutes_left=reserve - .01)
                with patch.object(m, "ReceiptWriter", RecordingWriter), \
                     patch.object(m, "load_ark11", side_effect=AssertionError("training runtime loaded")):
                    result = m.run_campaign(ctx)
                self.assertEqual(result["status"], "BUDGET_BLOCKED")
                self.assertEqual(len(result["results"]), count)
                self.assertTrue(all(r["status"] == "BUDGET_BLOCKED" for r in result["results"]))
                self.assertEqual(RecordingWriter.saves[-1][0], f"ARK-{number:03d}_RESULT.json")

    def test_partial_and_duplicate_evidence_cannot_be_promoted(self):
        m = load_runner(13)
        rows = [triplet(seed, 6801) for seed in m.ACQ_SEEDS]
        self.assertEqual(m._summarize(rows)["verdict"], "INCONCLUSIVE_PARTIAL_TRIPLETS")
        self.assertEqual(m._summarize([rows[0], rows[0]])["matched_triplets"], 0)
        missing = copy.deepcopy(rows[0])
        del missing["arms"]["FIXED_LOW"]
        self.assertEqual(m._summarize([missing])["matched_triplets"], 0)
        m = load_runner(12)
        row = schedule_row(m, [.5] * 6)
        self.assertEqual(m._summarize([row, row])["event_sources_executed"], 0)
        del row["schedules"]["SWITCH_95"]
        self.assertEqual(m._summarize([row])["event_sources_executed"], 0)

    def test_vice_versa_pareto_gain_and_area_only_regression(self):
        m = load_runner(13)
        rows = [triplet(seed, 6801) for seed in m.ACQ_SEEDS]
        for row in rows:
            row["arms"]["FIXED_HIGH"]["t3_control_metrics"]["G90_CONFIRM"] = 600
            adaptive = row["arms"]["ADAPTIVE_HIGH_LOW"]
            adaptive["t3_sealed_metrics"] = metrics(.61)
            adaptive["t2_sealed_metrics"] = metrics(.46)
        self.assertEqual(m._summarize(rows)["verdict"], "ADAPTIVE_PARETO_IMPROVEMENT")
        for row in rows:
            row["arms"]["ADAPTIVE_HIGH_LOW"]["t3_sealed_metrics"]["FINAL"] = .1
        self.assertNotEqual(m._summarize(rows)["verdict"], "ADAPTIVE_PARETO_IMPROVEMENT")

    def test_no_recovery_and_pooled_ordering_are_not_threshold_support(self):
        m = load_runner(12)
        row = schedule_row(m, [.4, .39, .4, .41, .42, .43])
        self.assertNotEqual(m._summarize([row, dict(row, acquisition_seed=1010)])["verdict"],
                            "STATE_THRESHOLD_SUPPORTED_SCREEN")
        first = schedule_row(m, [.4, .2, .7, .6, .8, .9])
        second = schedule_row(m, [.4, .2, .4, .7, .8, .9])
        second["acquisition_seed"] = 1010
        for row in (first, second):
            row["schedules"]["SWITCH_95"]["sealed_metrics"]["G90_CONFIRM"] = 600
        self.assertNotEqual(m._summarize([first, second])["verdict"], "STATE_THRESHOLD_SUPPORTED_SCREEN")

    def test_t3_manifest_deterministic_firewall(self):
        m=load_runner(13); a=m.build_t3carry_manifest(); self.assertEqual(a,m.build_t3carry_manifest())
        train, control, sealed, manifest=a
        self.assertEqual((len(train),len(control)+len(sealed),manifest["commutation_overlap"]),(500,200,0)); self.assertTrue(set(control).isdisjoint(sealed))
        for prompt,_ in train+control+sealed:
            x,y=m._pair_from_prompt(prompt); self.assertGreaterEqual(x%10+y%10,10)

    def test_vectorized_indices(self):
        m=load_runner(11)
        with patch.object(m,"DEVICE",torch.device("cpu")): actual=m.generate_continuation_indices(2702,17,13,500)
        g=torch.Generator().manual_seed(2702); expected=[torch.randint(0,500,(13,),generator=g).tolist() for _ in range(17)]
        self.assertEqual(actual,expected)

    def test_ark012_preserves_completed_schedule(self):
        m=load_runner(12); calls=[]
        fake=SimpleNamespace(load_manifest=lambda:{"train":[["p","a"]],"test":[["p","a"]],"split_sha256":"x"},build_control_sealed_split=lambda r:(r,r,{}),acquire=lambda *a:{"status":"ACQUIRED","snapshot":{},"onset_step":1,"confirmation_step":3},generate_continuation_indices=lambda *a:[[0]]*18000,run_to_threshold=lambda **k:{"status":"TRIGGERED","snapshot":{},"confirmation_absolute_step":1})
        def arm(*a,name,**k):
            calls.append(name)
            if len(calls)==2: raise RuntimeError("later arm failed")
            return {"sealed_metrics":metrics(.5),"completed_steps":8000}
        ctx=SimpleNamespace(device=torch.device("cpu"),head="x",minutes_left=100)
        with patch.multiple(m,ReceiptWriter=RecordingWriter,load_ark11=lambda:fake,bind_ark11_runtime=lambda *a,**k:None,_run_schedule=arm):
            with self.assertRaisesRegex(RuntimeError,"later arm"): m.run_campaign(ctx)
        p=[v for n,v in RecordingWriter.saves if n=="ARK-012_PARTIAL.json"][-1]; self.assertEqual(list(p["results"][0]["schedules"]),["HIGH_CONTINUE"])

    def test_ark013_preserves_completed_arm(self):
        m=load_runner(13); calls=[]
        fake=SimpleNamespace(load_manifest=lambda:{"train":[["p","a"]],"test":[["p","a"]],"split_sha256":"x"},build_control_sealed_split=lambda r:(r,r,{}),acquire=lambda *a:{"status":"ACQUIRED","snapshot":{},"onset_step":1,"confirmation_step":3},generate_continuation_indices=lambda *a:[[0]]*12000)
        def arm(*a,arm,**k):
            calls.append(arm)
            if len(calls)==2: raise RuntimeError("later arm failed")
            return {"t3_control_metrics":metrics(.5),"t3_sealed_metrics":metrics(.5),"t2_control_metrics":metrics(.5),"t2_sealed_metrics":metrics(.5),"completed_steps":12000}
        ctx=SimpleNamespace(device=torch.device("cpu"),head="x",minutes_left=100)
        with patch.multiple(m,ReceiptWriter=RecordingWriter,load_ark11=lambda:fake,bind_ark11_runtime=lambda *a,**k:None,build_t3carry_manifest=lambda:([('p','a')],[('p','a')],[('p','a')],{"manifest_sha256":"m"}),_run_arm=arm):
            with self.assertRaisesRegex(RuntimeError,"later arm failed"): m.run_campaign(ctx)
        p=[v for n,v in RecordingWriter.saves if n=="ARK-013_PARTIAL.json"][-1]; self.assertEqual(list(p["results"][0]["arms"]),["FIXED_HIGH"])

if __name__ == "__main__": unittest.main()
