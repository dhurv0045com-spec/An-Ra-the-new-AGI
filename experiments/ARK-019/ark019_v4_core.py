from __future__ import annotations

import hashlib, itertools, math, random
from typing import Any, Mapping
import numpy as np

EXPERIMENT = "ARK-019-V4"
PRETRAIN_SEEDS = (31801, 31902)
PILOT_ORDER_SEEDS = (419001, 419002)
MAIN_ORDER_SEEDS = (429001, 429002)
ARMS = (
    "PLASTIC_HIGH",
    "STATIC_REPLAY_1OF64",
    "STATIC_REPLAY_1OF32",
    "STATIC_CAP16X",
    "GUARDIAN_REPLAY",
    "GUARDIAN_HYBRID",
)
HIGH_LR = 3e-4
LOW_LR = 3e-6
BATCH_SLOTS = 32
PARENT_A_SLOTS = 16
PARENT_REAL_SLOTS = 16
PARENT_MAX_UPDATES = 2600
PARENT_EVAL_EVERY = 50
PARENT_STREAK = 3
PARENT_SCIENCE_CONTROL_REL_MAX = 0.15
PILOT_B_SLOT_CANDIDATES = (8, 12, 16)
PILOT_MAX_UPDATES = 2000
PILOT_CONFIRM_DEADLINE = 1800
PILOT_EVAL_EVERY = 50
PILOT_STREAK = 3
HORIZON = 2000
CONTROL_EVERY = 25
B_CONTROL_EVERY = 50
MEASURE_EVERY = 100
CHECKPOINT_EVERY = 200
CAP_SHADOW_STEPS = 32
SESSION_WALL_MINUTES = 225
PACKAGING_RESERVE_MINUTES = 10
RUNTIME_SAFETY_FACTOR = 1.30
R3_BUNDLE_SHA256 = "fcc14c5378318b8c447c2735768b72920334d8c9292b445372fd17d3d2b554d8"
R3_OFFICIAL_VERDICT = "CONTROLLER_NOT_SUPPORTED"


def deterministic_indices(seed: int, step: int, count: int, n: int, tag: str) -> list[int]:
    d = hashlib.sha256(f"ark019v4:{tag}:{seed}:{step}".encode()).digest()
    r = random.Random(int.from_bytes(d[:8], "big"))
    return [r.randrange(n) for _ in range(count)]


def augmented_perm(seed: int, step: int, idx: int) -> tuple[int, int, int]:
    p = list(itertools.permutations(range(3)))
    d = hashlib.sha256(f"ark019v4-aug:{seed}:{step}:{idx}".encode()).digest()
    return p[int.from_bytes(d[:8], "big") % 6]


def nonidentity_perm(seed: int, *parts: Any) -> tuple[int, int, int]:
    p = [x for x in itertools.permutations(range(3)) if x != (0, 1, 2)]
    d = hashlib.sha256((str(seed) + ":" + ":".join(map(str, parts))).encode()).digest()
    return p[int.from_bytes(d[:8], "big") % 5]


def query_order_perm(facts, query: int) -> tuple[int, int, int]:
    q = next(i for i, (k, _v) in enumerate(facts) if int(k) == int(query))
    others = [i for i in range(3) if i != q]
    if q == 2:
        return (2, 0, 1)
    return tuple(others + [q])  # type: ignore[return-value]


def qualified(m: Mapping[str, Any]) -> bool:
    return float(m["canonical"]) >= .90 and float(m["order_only"]) >= .85 and float(m["query_order"]) >= .85


def healthy(m: Mapping[str, Any]) -> bool:
    return float(m["canonical"]) >= .95 and float(m["order_only"]) >= .90 and float(m["query_order"]) >= .90


def robust_min(m: Mapping[str, Any]) -> float:
    return min(float(m[k]) for k in ("canonical", "order_only", "query_order"))


def choose_b_slots(pilot_results: Mapping[int, list[Mapping[str, Any]]]) -> dict[str, Any]:
    """Choose the smallest prospectively enumerated dose that passes both parent pilots."""
    table = {}
    for slots in PILOT_B_SLOT_CANDIDATES:
        rows = list(pilot_results.get(slots, []))
        passed = (
            len(rows) == len(PRETRAIN_SEEDS)
            and all(r.get("status") == "QUALIFIED" for r in rows)
            and all(int(r.get("confirmation_step", 10**9)) <= PILOT_CONFIRM_DEADLINE for r in rows)
            and all(bool(r.get("validation_qualified")) for r in rows)
        )
        table[str(slots)] = {"passed": passed, "rows": rows}
        if passed:
            return {"status": "PASS", "selected_b_slots": slots, "table": table,
                    "selection_rule": "smallest candidate passing both parent pilots by deadline"}
    return {"status": "FAIL", "selected_b_slots": None, "table": table,
            "selection_rule": "smallest candidate passing both parent pilots by deadline"}


def initial_controller(arm: str) -> dict[str, Any]:
    return {
        "state": "PLASTIC" if arm.startswith("GUARDIAN") else arm,
        "healthy_streak": 0,
        "b_streak": 0,
        "b_confirmation_step": None,
        "b_qualification_step": None,
        "failure_since": None,
        "replay32_failure_streak": 0,
        "post_b_sparse_until": None,
        "transitions": [],
    }


def treatment(arm: str, c: Mapping[str, Any], cap16x: float, step: int) -> tuple[int, float | None]:
    if arm == "PLASTIC_HIGH": return 0, None
    if arm == "STATIC_REPLAY_1OF64": return 64, None
    if arm == "STATIC_REPLAY_1OF32": return 32, None
    if arm == "STATIC_CAP16X": return 0, cap16x
    state = str(c["state"])
    if state == "PLASTIC": return 0, None
    if state in ("SPARSE64", "CONSOLIDATE64"): return 64, None
    if state == "REPLAY32": return 32, None
    if state == "EMERGENCY_CAP16X": return 32, cap16x
    raise ValueError(state)


def update_b_confirmation(c: dict[str, Any], step: int, b: Mapping[str, Any]) -> None:
    c["b_streak"] = c["b_streak"] + 1 if qualified(b) else 0
    if c["b_streak"] >= 3 and c["b_confirmation_step"] is None:
        c["b_qualification_step"] = step - 2 * B_CONTROL_EVERY
        c["b_confirmation_step"] = step
        c["post_b_sparse_until"] = step + 200


def update_controller(arm: str, c: dict[str, Any], step: int, a: Mapping[str, Any]) -> None:
    if not arm.startswith("GUARDIAN"):
        return
    q = qualified(a)
    h = healthy(a)
    old = str(c["state"])
    if not q and c["failure_since"] is None:
        c["failure_since"] = step
    if old == "REPLAY32" and not q:
        c["replay32_failure_streak"] += 1
    else:
        c["replay32_failure_streak"] = 0
    c["healthy_streak"] = c["healthy_streak"] + 1 if h else 0

    new, reason = old, None
    if not q:
        if arm == "GUARDIAN_HYBRID" and old == "REPLAY32" and c["replay32_failure_streak"] >= 2:
            new, reason = "EMERGENCY_CAP16X", "PERSISTENT_FAILURE_UNDER_REPLAY32"
        elif old not in ("REPLAY32", "EMERGENCY_CAP16X"):
            new, reason = "REPLAY32", "FORMAL_FAILURE"
    elif old == "PLASTIC" and robust_min(a) < .95:
        new, reason = "SPARSE64", "EARLY_MARGIN_WARNING"
    elif old == "EMERGENCY_CAP16X" and c["healthy_streak"] >= 4:
        new, reason = "SPARSE64", "RECOVERED_REMOVE_CAP"
    elif old == "REPLAY32" and c["healthy_streak"] >= 4:
        new, reason = "SPARSE64", "RECOVERED_DEESCALATE"
    elif old == "SPARSE64" and c["healthy_streak"] >= 4:
        until = c.get("post_b_sparse_until")
        if until is not None and step < int(until):
            new, reason = "CONSOLIDATE64", "POST_B_CONFIRMATION_FLOOR"
        else:
            new, reason = "PLASTIC", "HEALTHY_RETURN_TO_PLASTIC"
    elif old == "CONSOLIDATE64":
        until = c.get("post_b_sparse_until")
        if until is None or step >= int(until):
            if c["healthy_streak"] >= 4:
                new, reason = "PLASTIC", "CONSOLIDATION_COMPLETE"

    if new != old:
        c["state"] = new
        c["transitions"].append({"step": step, "from": old, "to": new, "reason": reason,
                                 "a_control": dict(a)})
        c["healthy_streak"] = 0
        if new != "REPLAY32": c["replay32_failure_streak"] = 0


def dedupe_trajectory(rows: list[Mapping[str, Any]], max_step: int | None = None) -> list[dict[str, Any]]:
    by = {}
    for r in rows:
        s = int(r["step"])
        if max_step is not None and s > max_step:
            continue
        by[s] = dict(r)
    return [by[s] for s in sorted(by)]


def _mean(xs) -> float:
    v = [float(x) for x in xs]
    return float(np.mean(v)) if v else float("nan")


def _median_finite(xs) -> float:
    v = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(np.median(v)) if v else float("inf")


def _area(r: Mapping[str, Any], key: str) -> float:
    vals = [robust_min(x[key]) for x in r.get("measurement_trace", []) if key in x]
    return _mean(vals) if vals else 0.0


def _control_failed(r: Mapping[str, Any]) -> bool:
    return any(not qualified(x["a_control"]) for x in r.get("control_trace", []) if "a_control" in x)


def _recovered(r: Mapping[str, Any], within: int = 200) -> bool:
    tr = [x for x in r.get("control_trace", []) if "a_control" in x]
    failures = [int(x["step"]) for x in tr if not qualified(x["a_control"])]
    if not failures: return True
    first = failures[0]
    streak = 0
    for x in tr:
        if int(x["step"]) < first: continue
        streak = streak + 1 if healthy(x["a_control"]) else 0
        if streak >= 4:
            return int(x["step"]) - first <= within
    return False


def decide(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    by = {a: [] for a in ARMS}
    for r in arms.values():
        if r.get("arm") in by: by[r["arm"]].append(r)
    if any(len(by[a]) != 4 for a in ARMS):
        return {"verdict": "INCONCLUSIVE_INCOMPLETE_MATCHED_SETS", "authorized": False}
    key = lambda r: (int(r["parent_seed"]), int(r["order_seed"]))
    maps = {a: {key(r): r for r in rows} for a, rows in by.items()}
    keys = sorted(maps["PLASTIC_HIGH"])
    if any(set(maps[a]) != set(keys) for a in ARMS):
        return {"verdict": "INCONCLUSIVE_MATCHING_ERROR", "authorized": False}

    ph = [maps["PLASTIC_HIGH"][k] for k in keys]
    b_success = sum(r.get("b_confirmation_step") is not None and qualified(r["final_b_sealed"]) for r in ph)
    if b_success < 3:
        return {"verdict": "INCONCLUSIVE_MAIN_B_FORMATION_INSTABILITY", "authorized": False,
                "plastic_high_b_successes": b_success,
                "claim_ceiling": "REAL_TEXT_PROXY_ONLY"}

    ph_fail = sum(_control_failed(r) for r in ph)
    s32 = [maps["STATIC_REPLAY_1OF32"][k] for k in keys]
    ph_area = _mean([_area(r, "a_sealed") for r in ph])
    s32_area = _mean([_area(r, "a_sealed") for r in s32])
    interference = ph_fail >= 2 or (s32_area - ph_area) >= .15
    ph_bmed = _median_finite([r.get("b_confirmation_step") for r in ph])

    details, winners = {}, []
    for arm in ("GUARDIAN_REPLAY", "GUARDIAN_HYBRID"):
        rows = [maps[arm][k] for k in keys]
        final_a = sum(qualified(r["final_a_sealed"]) for r in rows)
        area = _mean([_area(r, "a_sealed") for r in rows])
        b_ok = sum(r.get("b_confirmation_step") is not None and qualified(r["final_b_sealed"]) for r in rows)
        bmed = _median_finite([r.get("b_confirmation_step") for r in rows])
        gaps, nll_rel, duties, replay_cost = [], [], [], []
        for k, r in zip(keys, rows):
            p = maps["PLASTIC_HIGH"][k]
            gaps.append(robust_min(p["final_b_sealed"]) - robust_min(r["final_b_sealed"]))
            pn = float(p["final_science_sealed_nll"]); rn = float(r["final_science_sealed_nll"])
            nll_rel.append((rn - pn) / max(pn, 1e-12))
            c = r["counters"]
            duties.append(float(c["protection_updates"]) / HORIZON)
            replay_cost.append(float(c["replay_slots"]) / (BATCH_SLOTS * HORIZON))
        prevention = sum(not _control_failed(r) for r in rows)
        recovery = sum(_recovered(r) for r in rows)
        ok = bool(
            interference and final_a >= 3 and area >= s32_area - .10 and b_ok >= 3
            and math.isfinite(ph_bmed) and math.isfinite(bmed) and bmed <= 1.5 * ph_bmed
            and _mean(gaps) <= .05 and all(x <= .05 for x in nll_rel)
            and _mean(duties) < .60 and _mean(replay_cost) < .05
        )
        details[arm] = {
            "final_a_qualified_sets": final_a,
            "a_sealed_area": area,
            "b_success_sets": b_ok,
            "median_b_confirmation_step": bmed,
            "mean_final_b_gap_vs_high": _mean(gaps),
            "science_nll_relative_vs_high": nll_rel,
            "mean_protection_duty": _mean(duties),
            "mean_replay_cost": _mean(replay_cost),
            "prevention_sets": prevention,
            "recovery_within_200_sets": recovery,
            "qualifies": ok,
        }
        if ok: winners.append(arm)

    if not interference:
        verdict = "INCONCLUSIVE_LOW_INTERFERENCE"
    elif winners:
        verdict = "GUARDIAN_CONTINUAL_PROXY_CANDIDATE"
    else:
        verdict = "GUARDIAN_NOT_SUPPORTED_V4"
    flags = []
    if any(details[a]["prevention_sets"] >= 3 for a in details): flags.append("PREVENTION_SIGNAL")
    if any(details[a]["recovery_within_200_sets"] >= 3 for a in details): flags.append("RECOVERY_SIGNAL")
    if "GUARDIAN_REPLAY" in winners: flags.append("DYNAMIC_REPLAY_SUFFICIENT")
    if "GUARDIAN_HYBRID" in winners and "GUARDIAN_REPLAY" not in winners: flags.append("EMERGENCY_CAP_ADDS_VALUE")
    return {
        "verdict": verdict,
        "authorized": verdict == "GUARDIAN_CONTINUAL_PROXY_CANDIDATE",
        "flags": flags,
        "interference": interference,
        "plastic_high_control_failures": ph_fail,
        "plastic_high_b_successes": b_success,
        "plastic_high_a_area": ph_area,
        "static_replay32_a_area": s32_area,
        "plastic_high_median_b_confirmation_step": ph_bmed,
        "guardian_details": details,
        "claim_ceiling": "REAL_TEXT_PROXY_GUARDIAN_CANDIDATE_ONLY",
    }
