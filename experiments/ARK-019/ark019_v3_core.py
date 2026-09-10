from __future__ import annotations

import hashlib, itertools, math, random
from typing import Any, Mapping
import numpy as np

EXPERIMENT="ARK-019-V3"
PRETRAIN_SEEDS=(31801,31902)
SKILL_B_SEEDS=(319001,319002)
ARMS=("PLASTIC_HIGH","STATIC_REPLAY_1OF64","GUARDIAN_REPLAY","GUARDIAN_HYBRID")
HIGH_LR=3e-4; LOW_LR=3e-6
HORIZON=1000; EVAL_EVERY=100; CHECKPOINT_EVERY=300
BATCH_SLOTS=32; REAL_SLOTS=28; SKILL_B_SLOTS=4
CAP_SHADOW_STEPS=32
WALL_MINUTES=175; PACKAGING_RESERVE_MINUTES=5; RUNTIME_SAFETY_FACTOR=1.30
R2_BUNDLE_SHA256="c648d3fde569ca66fb34b54e56c09589a020a71bb5f7dcb88e476c224292d32c"
R2_VERDICT="BOTH_LEVERS_SUFFICIENT"


def deterministic_indices(seed:int, step:int, count:int, n:int, tag:str)->list[int]:
    d=hashlib.sha256(f"ark019:{tag}:{seed}:{step}".encode()).digest()
    r=random.Random(int.from_bytes(d[:8],"big")); return [r.randrange(n) for _ in range(count)]


def augmented_perm(seed:int, step:int, idx:int)->tuple[int,int,int]:
    p=list(itertools.permutations(range(3))); d=hashlib.sha256(f"ark019-aug:{seed}:{step}:{idx}".encode()).digest()
    return p[int.from_bytes(d[:8],"big")%6]


def nonidentity_perm(seed:int,*parts:Any)->tuple[int,int,int]:
    p=[x for x in itertools.permutations(range(3)) if x!=(0,1,2)]
    d=hashlib.sha256((str(seed)+":"+":".join(map(str,parts))).encode()).digest(); return p[int.from_bytes(d[:8],"big")%5]


def query_order_perm(facts, query:int)->tuple[int,int,int]:
    q=next(i for i,(k,_v) in enumerate(facts) if int(k)==int(query)); others=[i for i in range(3) if i!=q]
    if q==2: return (2,0,1)
    return tuple(others+[q])  # type: ignore[return-value]


def initial_controller(arm:str)->dict[str,Any]:
    return {"state":"PLASTIC" if arm.startswith("GUARDIAN") else arm,"warning_streak":0,"healthy_streak":0,
            "failure_streak_in_replay32":0,"skill_b_streak":0,"skill_b_qualification_step":None,
            "skill_b_confirmation_step":None,"skill_b_confirm_counters":None,"consolidated":False,"transitions":[]}


def treatment(arm:str,c:Mapping[str,Any],cap16x:float)->tuple[int,float|None]:
    if arm=="PLASTIC_HIGH": return 0,None
    if arm=="STATIC_REPLAY_1OF64": return 64,None
    state=str(c["state"])
    if state=="PLASTIC": return 0,None
    if state in ("SPARSE64","CONSOLIDATE"): return 64,None
    if state=="REPLAY32": return 32,None
    if state=="EMERGENCY_CAP16X": return 32,cap16x
    raise ValueError(state)


def update_controller(arm:str,c:dict[str,Any],step:int,a:Mapping[str,Any],b:Mapping[str,Any])->None:
    if b["qualified"]: c["skill_b_streak"]+=1
    else: c["skill_b_streak"]=0
    if c["skill_b_streak"]>=3 and c["skill_b_confirmation_step"] is None:
        c["skill_b_qualification_step"]=step-2*EVAL_EVERY; c["skill_b_confirmation_step"]=step
        if arm.startswith("GUARDIAN"):
            c["consolidated"]=True
            if c["state"] in ("PLASTIC","SPARSE64"):
                old=c["state"]; c["state"]="CONSOLIDATE"; c["transitions"].append({"step":step,"from":old,"to":"CONSOLIDATE","reason":"SKILL_B_QUALIFIED"})
    if not arm.startswith("GUARDIAN"): return

    qualified=bool(a["qualified"]); healthy=bool(a["healthy_margin"])
    warning=float(a["order_only"])<.90 or float(a["query_order"])<.90
    c["warning_streak"]=c["warning_streak"]+1 if warning else 0
    c["healthy_streak"]=c["healthy_streak"]+1 if healthy else 0
    if c["state"]=="REPLAY32" and not qualified: c["failure_streak_in_replay32"]+=1
    else: c["failure_streak_in_replay32"]=0
    old=c["state"]; new=old; reason=None
    if not qualified:
        if arm=="GUARDIAN_HYBRID" and old=="REPLAY32" and c["failure_streak_in_replay32"]>=2:
            new,reason="EMERGENCY_CAP16X","PERSISTENT_FORMAL_FAILURE"
        elif old not in ("REPLAY32","EMERGENCY_CAP16X"):
            new,reason="REPLAY32","FORMAL_FAILURE"
    elif old=="PLASTIC" and c["warning_streak"]>=2: new,reason="SPARSE64","MARGIN_WARNING"
    elif old=="EMERGENCY_CAP16X" and c["healthy_streak"]>=3: new,reason="SPARSE64","RECOVERED_REMOVE_CAP"
    elif old=="REPLAY32" and c["healthy_streak"]>=3: new,reason="SPARSE64","RECOVERED_DEESCALATE"
    elif old=="SPARSE64" and c["healthy_streak"]>=3 and not c["consolidated"]: new,reason="PLASTIC","HEALTHY_DEESCALATE"
    if c["consolidated"] and new=="PLASTIC": new,reason="CONSOLIDATE","CONSOLIDATION_FLOOR"
    if new!=old:
        c["state"]=new; c["transitions"].append({"step":step,"from":old,"to":new,"reason":reason,"a_control":dict(a),"b_control":dict(b)})
        c["healthy_streak"]=0; c["warning_streak"]=0
        if new!="REPLAY32": c["failure_streak_in_replay32"]=0


def robustness_area(traj:list[Mapping[str,Any]],key:str)->float:
    v=[min(float(r[key]["canonical"]),float(r[key]["order_only"]),float(r[key]["query_order"])) for r in traj if int(r.get("step",0))>0 and key in r]
    return float(np.mean(v)) if v else 0.0


def any_failure(traj:list[Mapping[str,Any]],key:str)->bool:
    return any(int(r.get("step",0))>0 and key in r and not bool(r[key]["qualified"]) for r in traj)


def median_or_inf(xs)->float:
    v=[float(x) for x in xs if x is not None]; return float(np.median(v)) if v else float("inf")


def decide(arms:Mapping[str,Mapping[str,Any]])->dict[str,Any]:
    by={a:[] for a in ARMS}
    for r in arms.values(): by[r["arm"]].append(r)
    if any(len(by[a])<4 for a in ARMS): return {"verdict":"INCONCLUSIVE_INCOMPLETE_MATCHED_SETS","authorized":False}
    key=lambda r:(int(r["parent_seed"]),int(r["skill_b_seed"]))
    maps={a:{key(r):r for r in rows} for a,rows in by.items()}; keys=sorted(maps["PLASTIC_HIGH"])
    if any(set(maps[a])!=set(keys) for a in ARMS): return {"verdict":"INCONCLUSIVE_MATCHING_ERROR","authorized":False}
    ph=[maps["PLASTIC_HIGH"][k] for k in keys]; st=[maps["STATIC_REPLAY_1OF64"][k] for k in keys]
    ph_fail=sum(any_failure(r["trajectory"],"a_sealed") for r in ph)
    ph_area=float(np.mean([robustness_area(r["trajectory"],"a_sealed") for r in ph])); st_area=float(np.mean([robustness_area(r["trajectory"],"a_sealed") for r in st]))
    interference=ph_fail>=2 or (st_area-ph_area)>=.15; ph_b=median_or_inf([r.get("skill_b_confirmation_step") for r in ph])
    details={}; winners=[]
    for arm in ("GUARDIAN_REPLAY","GUARDIAN_HYBRID"):
        rows=[maps[arm][k] for k in keys]; risk=sum(any_failure(r["trajectory"],"a_sealed") for r in rows)/4
        st_risk=sum(any_failure(r["trajectory"],"a_sealed") for r in st)/4; ph_risk=ph_fail/4; bmed=median_or_inf([r.get("skill_b_confirmation_step") for r in rows])
        b_gaps=[]; nll_rel=[]; duties=[]; replay_costs=[]
        for k,r in zip(keys,rows):
            p=maps["PLASTIC_HIGH"][k]
            pr=min(float(p["final_b_sealed"][x]) for x in ("canonical","order_only","query_order")); rr=min(float(r["final_b_sealed"][x]) for x in ("canonical","order_only","query_order")); b_gaps.append(pr-rr)
            pn=float(p["trajectory"][-1]["science"]["sealed"]["nll"]); rn=float(r["trajectory"][-1]["science"]["sealed"]["nll"]); nll_rel.append((rn-pn)/max(pn,1e-12))
            confirm=r.get("skill_b_confirmation_step"); cc=r.get("controller",{}).get("skill_b_confirm_counters")
            if confirm is not None and cc is not None:
                d=max(1,int(confirm)); duties.append(float(cc["protection_updates"])/d); replay_costs.append(float(cc["displaced_real_slots"])/(BATCH_SLOTS*d))
            else:
                duties.append(float(r["counters"]["protection_updates"])/HORIZON); replay_costs.append(float(r["counters"]["displaced_real_slots"])/(BATCH_SLOTS*HORIZON))
        gap=float(np.mean(b_gaps)); duty=float(np.mean(duties)); rc=float(np.mean(replay_costs)); nll_ok=all(x<=.05 for x in nll_rel)
        ok=bool(interference and risk<=st_risk+.10 and (ph_fail==0 or ph_risk-risk>=.30) and math.isfinite(ph_b) and math.isfinite(bmed) and bmed<=1.5*ph_b and gap<=.05 and nll_ok and duty<.60 and rc<.05)
        details[arm]={"old_skill_failure_risk":risk,"median_skill_b_confirmation_step":bmed,"mean_b_final_gap_vs_high":gap,"science_nll_relative_vs_high":nll_rel,"prequal_protection_duty":duty,"prequal_replay_cost":rc,"qualifies":ok}
        if ok:winners.append(arm)
    if not interference: verdict="INCONCLUSIVE_LOW_INTERFERENCE"
    elif winners: verdict="HIERARCHICAL_GUARDIAN_PROXY_CANDIDATE"
    else:
        st_risk=sum(any_failure(r["trajectory"],"a_sealed") for r in st)/4; verdict="STATIC_PROTECTION_ONLY" if st_risk<ph_fail/4 else "CONTROLLER_NOT_SUPPORTED"
    flags=[]
    if "GUARDIAN_REPLAY" in winners: flags.append("SPARSE_REPLAY_GUARDIAN_SUFFICIENT")
    if "GUARDIAN_HYBRID" in winners and "GUARDIAN_REPLAY" not in winners: flags.append("EMERGENCY_CAP_ADDS_VALUE")
    return {"verdict":verdict,"flags":flags,"interference":interference,"plastic_high_failures":ph_fail,"plastic_high_mean_area":ph_area,"static_replay_mean_area":st_area,"plastic_high_median_skill_b_confirmation_step":ph_b,"guardian_details":details,"authorized":verdict=="HIERARCHICAL_GUARDIAN_PROXY_CANDIDATE","claim_ceiling":"REAL_TEXT_PROXY_GUARDIAN_CANDIDATE_ONLY"}
