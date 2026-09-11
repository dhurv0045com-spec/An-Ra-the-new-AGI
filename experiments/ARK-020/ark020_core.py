"""ARK-020 core logic — torch-free, deterministic, testable.

Contains: constants, task-split construction (binding, inverse retrieval, successor
cycle), capability registry, generalized Guardian controller (reactive / predictive /
hybrid), risk-based replay allocation, static rotation, cost accounting, and the
frozen verdict logic. Controller functions accept CONTROL metrics only; SEALED data
never reaches them (structurally enforced by signatures).
"""
from __future__ import annotations

import hashlib
import itertools
import math
import random
from typing import Any, Iterable, Mapping

EXPERIMENT = "ARK-020"

PARENT_SEEDS = (31801, 31902)
PHASE_ORDER_SEEDS = {"B": (429001, 429002), "C": (429003, 429004), "D": (429005, 429006)}
PILOT_ORDER_SEEDS = (419001, 419002)

ARMS = (
    "PLASTIC_HIGH",
    "STATIC_REPLAY_1OF64",
    "STATIC_REPLAY_1OF32",
    "STATIC_CAP16X",
    "GUARDIAN_REACTIVE",
    "GUARDIAN_PREDICTIVE",
    "GUARDIAN_HYBRID",
)
GUARDIAN_ARMS = ("GUARDIAN_REACTIVE", "GUARDIAN_PREDICTIVE", "GUARDIAN_HYBRID")
STATIC_ARMS = ("STATIC_REPLAY_1OF64", "STATIC_REPLAY_1OF32", "STATIC_CAP16X")

SKILLS = ("A", "B", "C", "D")
SKILL_FAMILY = {"A": "relational_binding", "B": "relational_binding",
                "C": "rule_induction", "D": "inverse_retrieval"}
PHASE_OF_SKILL = {"B": 0, "C": 1, "D": 2}
PHASE_UPDATES = {"B": 2000, "C": 1500, "D": 1500}
PHASE_TASK_SLOTS = {"B": None, "C": 12, "D": 12}  # B filled from dose selection
CONTINUATION_HORIZON = sum(PHASE_UPDATES.values())

BATCH_SLOTS = 32
MAX_REPLAY_SLOTS_PER_UPDATE = 2
PILOT_B_SLOT_CANDIDATES = (8, 12, 16)
PILOT_MAX_UPDATES = 2000
PILOT_CONFIRM_DEADLINE = 1800

HIGH_LR = 3e-4
LOW_LR = 3e-6
CAP_SHADOW_STEPS = 32
CAP_MULTIPLIER = 16.0

PARENT_A_SLOTS = 16
PARENT_REAL_SLOTS = 16
PARENT_MAX_UPDATES = 2600
PARENT_EVAL_EVERY = 50
PARENT_STREAK = 3
PARENT_SCIENCE_CONTROL_REL_MAX = 0.15

CONTROL_EVERY = 25
MEASURE_EVERY = 100
CHECKPOINT_EVERY = 200

QUALIFIED_THRESHOLD = {"canonical": 0.90, "order_only": 0.85, "third": 0.85}
HEALTHY_THRESHOLD = {"canonical": 0.95, "order_only": 0.90, "third": 0.90}
WARNING_MARGIN_BELOW = 0.95
DEGRADATION_RATE_LIMIT = -0.03  # per 100 updates, slope over last 3 CONTROL observations
CONFIRM_STREAK = 3
HEALTHY_DEESCALATE_STREAK = 4
POST_PHASE_SPARSE_FLOOR = 200
EMERGENCY_REPLAY32_FAILURE_STREAK = 2
RECOVERY_WINDOW_UPDATES = 400

SESSION_WALL_MINUTES = 225
PACKAGING_RESERVE_MINUTES = 10
RUNTIME_SAFETY_FACTOR = 1.30

GUARDIAN_SUCCESS_RULES = {
    "final_sealed_all_capabilities_qualified_sets": 3,
    "mean_sealed_area_within_of_best_static": 0.10,
    "median_phase_confirmation_vs_plastic_max_ratio": 1.5,
    "science_sealed_nll_relative_max_per_set": 0.05,
    "mean_protection_duty_max": 0.60,
    "mean_replay_fraction_max": 0.05,
}

VERDICT_SUCCESS = "GUARDIAN_MULTI_SKILL_CANDIDATE"
VERDICT_QUALITY_ONLY = "GUARDIAN_SUPPORTED_NOT_EFFICIENT"
VERDICT_FAILURE = "GUARDIAN_NOT_SUPPORTED_MULTI_SKILL"


# --------------------------------------------------------------------- metrics

def third_key(metrics: Mapping[str, Any]) -> str:
    """Capabilities carry canonical + order_only + one capability-specific third mode."""
    for k in ("query_order", "distractor"):
        if k in metrics:
            return k
    raise KeyError("metrics missing third robustness mode")


def robust_min(metrics: Mapping[str, Any]) -> float:
    return min(float(metrics["canonical"]), float(metrics["order_only"]),
               float(metrics[third_key(metrics)]))


def qualified(metrics: Mapping[str, Any]) -> bool:
    t = third_key(metrics)
    return (float(metrics["canonical"]) >= QUALIFIED_THRESHOLD["canonical"]
            and float(metrics["order_only"]) >= QUALIFIED_THRESHOLD["order_only"]
            and float(metrics[t]) >= QUALIFIED_THRESHOLD["third"])


def healthy(metrics: Mapping[str, Any]) -> bool:
    t = third_key(metrics)
    return (float(metrics["canonical"]) >= HEALTHY_THRESHOLD["canonical"]
            and float(metrics["order_only"]) >= HEALTHY_THRESHOLD["order_only"]
            and float(metrics[t]) >= HEALTHY_THRESHOLD["third"])


# ------------------------------------------------------- task construction (pure)

def split5(keys: Iterable[int], vals: Iterable[int], seed: int) -> dict[str, list]:
    """V4-compatible five-way factset split."""
    keys, vals = list(keys), list(vals)
    x = [tuple(zip(kt, vt)) for kt in itertools.combinations(keys, 3)
         for vt in itertools.permutations(vals, 3)]
    random.Random(seed).shuffle(x)
    if len(x) < 600:
        raise RuntimeError(f"insufficient factsets: {len(x)}")
    return {"train": x[:400], "parent_control": x[400:450], "main_control": x[450:500],
            "validation": x[500:550], "sealed": x[550:600]}


def semantics(fs: Iterable[Iterable[tuple[int, int]]]) -> list[tuple[list, int, int]]:
    return [(list(f), q, a) for f in fs for q, a in f]


def inverse_split5(keys: Iterable[int], vals: Iterable[int], seed: int) -> dict[str, list]:
    """Inverse retrieval: facts are (value, key) pairs; value queried, key answered."""
    keys, vals = list(keys), list(vals)
    x = [tuple(zip(kt, vt)) for kt in itertools.combinations(vals, 3)
         for vt in itertools.permutations(keys, 3)]
    random.Random(seed).shuffle(x)
    if len(x) < 600:
        raise RuntimeError(f"insufficient inverse factsets: {len(x)}")
    return {"train": x[:400], "parent_control": x[400:450], "main_control": x[450:500],
            "validation": x[500:550], "sealed": x[550:600]}


def successor_cycle(tokens: Iterable[int], seed: int) -> dict[int, int]:
    """Fixed permutation cycle: successor(t) for each token; deterministic from seed."""
    ts = list(tokens)
    perm = ts[:]
    random.Random(seed).shuffle(perm)
    return {perm[i]: perm[(i + 1) % len(perm)] for i in range(len(perm))}


def cycle_task(tokens: Iterable[int], seed: int) -> dict[str, Any]:
    """Rule-induction skill: successor cycle on 12 tokens.

    9 train keys (rule trained on all 9), subdivided deterministically into
    3 control / 3 validation / 3 extra-train for evaluation; 3 sealed keys are
    NEVER trained (induction measurement only).
    """
    ts = list(tokens)
    if len(ts) != 12 or len(set(ts)) != 12:
        raise RuntimeError("cycle task requires 12 distinct tokens")
    cyc = successor_cycle(ts, seed)
    d = hashlib.sha256(f"ark020-cycle-split:{seed}".encode()).digest()
    shuffled = ts[:]
    random.Random(int.from_bytes(d[:8], "big")).shuffle(shuffled)
    train_keys = shuffled[:9]
    control_keys, validation_keys = train_keys[:3], train_keys[3:6]
    sealed_keys = shuffled[9:]
    train_pairs = [(k, cyc[k]) for k in train_keys]
    return {"cycle": cyc, "train_keys": train_keys, "control_keys": control_keys,
            "validation_keys": validation_keys, "sealed_keys": sealed_keys,
            "train_pairs": train_pairs,
            "train_sem": [(train_pairs, k, cyc[k]) for k in train_keys],
            "control_sem": [(train_pairs, k, cyc[k]) for k in control_keys],
            "validation_sem": [(train_pairs, k, cyc[k]) for k in validation_keys],
            "sealed_sem": [(train_pairs, k, cyc[k]) for k in sealed_keys],
            "seed": seed}


def build_all_tasks(token_ids: list[int]) -> dict[str, Any]:
    if len(token_ids) != 48 or len(set(token_ids)) != 48:
        raise RuntimeError(f"need 48 distinct tokens, got {len(token_ids)}")
    g = {"A": token_ids[0:12], "B": token_ids[12:24], "C": token_ids[24:36], "D": token_ids[36:48]}
    A = split5(g["A"][:6], g["A"][6:12], 524218)   # identical to V4 skill A
    B = split5(g["B"][:6], g["B"][6:12], 524219)   # identical to V4 skill B
    D = inverse_split5(g["D"][:6], g["D"][6:12], 524220)
    C = cycle_task(g["C"], 524221)
    return {"groups": g, "A": A, "B": B, "C": C, "D": D,
            "sem": {"A": {k: semantics(v) for k, v in A.items()},
                    "B": {k: semantics(v) for k, v in B.items()},
                    "D": {k: semantics(v) for k, v in D.items()},
                    "C": {"train": C["train_sem"], "control": C["control_sem"],
                          "validation": C["validation_sem"], "sealed": C["sealed_sem"]}}}


def deterministic_indices(seed: int, step: int, count: int, n: int, tag: str) -> list[int]:
    d = hashlib.sha256(f"ark020:{tag}:{seed}:{step}".encode()).digest()
    r = random.Random(int.from_bytes(d[:8], "big"))
    return [r.randrange(n) for _ in range(count)]


def nonidentity_perm3(seed: int, *parts: Any) -> tuple[int, int, int]:
    p = [x for x in itertools.permutations(range(3)) if x != (0, 1, 2)]
    d = hashlib.sha256((str(seed) + ":" + ":".join(map(str, parts))).encode()).digest()
    return p[int.from_bytes(d[:8], "big") % 5]


def augmented_perm3(seed: int, step: int, idx: int) -> tuple[int, int, int]:
    p = list(itertools.permutations(range(3)))
    d = hashlib.sha256(f"ark020-aug:{seed}:{step}:{idx}".encode()).digest()
    return p[int.from_bytes(d[:8], "big") % 6]


def query_order_perm(facts, query) -> tuple[int, int, int]:
    q = next(i for i, (k, _v) in enumerate(facts) if int(k) == int(query))
    others = [i for i in range(3) if i != q]
    if q == 2:
        return (2, 0, 1)
    return tuple(others + [q])  # type: ignore[return-value]


# ------------------------------------------------------------------- rendering

MAX_PROMPT_IDS = 255  # V3.CONTEXT - 1


def render_binding(t: Mapping[str, Any], f, q, order) -> list[int]:
    """'Facts: k means v; … Query: k means' -> answer token (same shape for B/D)."""
    ids = list(t["p"])
    for i in order:
        k, v = f[i]
        ids += [int(k)] + list(t["m"]) + [int(v)] + list(t["s"])
    ids += list(t["q"]) + [int(q)] + list(t["t"])
    return ids[-MAX_PROMPT_IDS:]


def render_cycle(t: Mapping[str, Any], query, n_distractors: int, train_pairs,
                 seed: int, step: int) -> list[int]:
    """C prompt: 'Chain: [k next; ]* k next' -> successor token. 0/1/3 distractors."""
    ids = list(t["p"])
    if n_distractors:
        picks = deterministic_indices(seed, step, n_distractors, len(train_pairs), "cyc-distract")
        for i in picks:
            k, _v = train_pairs[i]
            ids += [int(k)] + list(t["m"]) + list(t["s"])
    ids += [int(query)] + list(t["m"])
    return ids[-MAX_PROMPT_IDS:]


CYCLE_DISTRACTORS = {"canonical": 0, "reversed": 1, "query_order": 3,
                     "augmented": 1, "nonidentity": 2}


def task_rows(cap: str, t, sem, idx, mode, seed, step):
    out = []
    if cap in ("A", "B", "D"):
        for i in idx:
            f, q, a = sem[int(i)]
            if mode == "canonical":
                order = (0, 1, 2)
            elif mode == "reversed":
                order = (2, 1, 0)
            elif mode == "query_order":
                order = query_order_perm(f, q)
            elif mode == "augmented":
                order = augmented_perm3(seed, step, int(i))
            elif mode == "nonidentity":
                order = nonidentity_perm3(seed, step, int(i))
            else:
                raise ValueError(mode)
            out.append((render_binding(t, f, q, order), int(a)))
        return out
    n_dis = CYCLE_DISTRACTORS[mode]
    train_pairs = sem[0][0] if sem else []
    for i in idx:
        _pairs, q, a = sem[int(i)]
        out.append((render_cycle(t, q, n_dis, train_pairs, seed, step), int(a)))
    return out


# --------------------------------------------------------------------- registry

def init_registry() -> dict[str, Any]:
    return {"capabilities": {}}


def register_capability(reg: dict, cap_id: str, step: int) -> dict:
    st = {"capability_id": cap_id, "family": SKILL_FAMILY[cap_id],
          "qualified_step": step, "healthy_streak": 0, "margin_history": [],
          "degradation_rate": 0.0, "protection_level": "PLASTIC",
          "replay_slots_used": 0, "last_healthy_step": step,
          "failure_since": None, "replay32_failure_streak": 0,
          "recovered_events": 0, "prevented": True}
    reg["capabilities"][cap_id] = st
    return st


def degradation_rate(history: list[Mapping[str, Any]]) -> float:
    """Slope of robust-min over the last 3 CONTROL observations, per 100 updates."""
    pts = [(int(h["step"]), robust_min(h["metrics"])) for h in history][-3:]
    if len(pts) < 2:
        return 0.0
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    span = x1 - x0
    if span <= 0:
        return 0.0
    return (y1 - y0) / span * 100.0


def observe_capability(reg: dict, cap_id: str, step: int,
                       control_metrics: Mapping[str, Any]) -> dict:
    """CONTROL-only observation; returns the updated state. Never sees SEALED data."""
    st = reg["capabilities"][cap_id]
    st["margin_history"].append({"step": int(step), "metrics": dict(control_metrics)})
    st["margin_history"] = st["margin_history"][-12:]
    st["degradation_rate"] = degradation_rate(st["margin_history"])
    q, h = qualified(control_metrics), healthy(control_metrics)
    st["healthy_streak"] = st["healthy_streak"] + 1 if h else 0
    if h:
        st["last_healthy_step"] = int(step)
    if not q and st["failure_since"] is None:
        st["failure_since"] = int(step)
        st["prevented"] = False
    if q and st["failure_since"] is not None and st["healthy_streak"] >= HEALTHY_DEESCALATE_STREAK:
        st["failure_since"] = None
        st["recovered_events"] += 1
    return st


def risk_order(reg: dict, cap_ids: Iterable[str]) -> list[str]:
    """Lowest current robust-min first; deterministic tiebreak by capability id."""

    def key(cid: str):
        hist = reg["capabilities"][cid]["margin_history"]
        m = robust_min(hist[-1]["metrics"]) if hist else 0.0
        return (m, cid)

    return sorted(cap_ids, key=key)


def warning_signal(st: Mapping[str, Any]) -> bool:
    hist = st["margin_history"]
    if not hist:
        return False
    if robust_min(hist[-1]["metrics"]) < WARNING_MARGIN_BELOW:
        return True
    return bool(st["degradation_rate"] <= DEGRADATION_RATE_LIMIT)


# ------------------------------------------------------------------- controller

def initial_controller(arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(arm)
    return {"arm": arm, "states": {}, "transitions": [],
            "replay32_failure_streaks": {}, "post_phase_floor_until": {}}


def _set_level(ctrl: dict, step: int, cap_id: str, level: str, reason: str) -> None:
    old = ctrl["states"].get(cap_id, "PLASTIC")
    if old != level:
        ctrl["states"][cap_id] = level
        ctrl["transitions"].append({"step": int(step), "capability": cap_id,
                                    "from": old, "to": level, "reason": reason})


def update_controller(arm: str, ctrl: dict, step: int, reg: dict,
                      qualified_caps: list[str]) -> None:
    """Per-capability Guardian state machine. CONTROL metrics only (via registry)."""
    if arm not in GUARDIAN_ARMS:
        return
    predictive = arm in ("GUARDIAN_PREDICTIVE", "GUARDIAN_HYBRID")
    hybrid = arm == "GUARDIAN_HYBRID"
    for cap_id in qualified_caps:
        st = reg["capabilities"][cap_id]
        old = ctrl["states"].get(cap_id, "PLASTIC")
        hist = st["margin_history"]
        latest_ok = bool(hist) and qualified(hist[-1]["metrics"])
        if old == "REPLAY32" and not latest_ok:
            ctrl["replay32_failure_streaks"][cap_id] = ctrl["replay32_failure_streaks"].get(cap_id, 0) + 1
        else:
            ctrl["replay32_failure_streaks"][cap_id] = 0
        if not latest_ok:
            if hybrid and old == "REPLAY32" and \
               ctrl["replay32_failure_streaks"].get(cap_id, 0) >= EMERGENCY_REPLAY32_FAILURE_STREAK:
                _set_level(ctrl, step, cap_id, "EMERGENCY_CAP16X", "PERSISTENT_FAILURE_UNDER_REPLAY32")
            elif old not in ("REPLAY32", "EMERGENCY_CAP16X"):
                _set_level(ctrl, step, cap_id, "REPLAY32", "FORMAL_FAILURE")
        elif old == "PLASTIC" and predictive and warning_signal(st):
            _set_level(ctrl, step, cap_id, "SPARSE64", "PREDICTIVE_WARNING")
        elif old == "EMERGENCY_CAP16X" and st["healthy_streak"] >= HEALTHY_DEESCALATE_STREAK:
            _set_level(ctrl, step, cap_id, "SPARSE64", "RECOVERED_REMOVE_CAP")
        elif old == "REPLAY32" and st["healthy_streak"] >= HEALTHY_DEESCALATE_STREAK:
            _set_level(ctrl, step, cap_id, "SPARSE64", "RECOVERED_DEESCALATE")
        elif old == "SPARSE64" and st["healthy_streak"] >= HEALTHY_DEESCALATE_STREAK:
            until = ctrl["post_phase_floor_until"].get(cap_id)
            if until is not None and step < int(until):
                _set_level(ctrl, step, cap_id, "SPARSE64", "POST_PHASE_FLOOR")
            else:
                _set_level(ctrl, step, cap_id, "PLASTIC", "HEALTHY_RETURN_TO_PLASTIC")


def set_post_phase_floor(ctrl: dict, cap_ids: Iterable[str], step: int,
                         floor: int = POST_PHASE_SPARSE_FLOOR) -> None:
    for c in cap_ids:
        ctrl["post_phase_floor_until"][c] = int(step) + int(floor)


def treatment(arm: str, ctrl: Mapping[str, Any], step: int, reg: dict,
              old_caps: list[str], cap16x: float | None) -> tuple[list[str], float | None]:
    """Returns (replay_caps ordered by risk, cap_value). Max 2 replay slots per update."""
    if arm == "PLASTIC_HIGH":
        return [], None
    if arm == "STATIC_REPLAY_1OF64":
        if step % 2 == 0 and old_caps:
            return [old_caps[(step // 2) % len(old_caps)]], None
        return [], None
    if arm == "STATIC_REPLAY_1OF32":
        if old_caps:
            return [old_caps[(step - 1) % len(old_caps)]], None
        return [], None
    if arm == "STATIC_CAP16X":
        return [], cap16x
    states = ctrl["states"]
    requesting = [c for c in old_caps
                  if states.get(c, "PLASTIC") in ("SPARSE64", "REPLAY32", "EMERGENCY_CAP16X")]
    ordered = risk_order(reg, requesting)
    replay = ordered[:MAX_REPLAY_SLOTS_PER_UPDATE]
    cap = cap16x if any(states.get(c) == "EMERGENCY_CAP16X" for c in old_caps) else None
    return replay, cap


# ------------------------------------------------------------- result packaging

def dedupe_trajectory(rows: list[Mapping[str, Any]], max_step: int | None = None) -> list[dict]:
    by: dict[int, dict] = {}
    for r in rows:
        s = int(r["step"])
        if max_step is not None and s > max_step:
            continue
        by[s] = dict(r)
    return [by[s] for s in sorted(by)]


def mean(xs: list[float]) -> float:
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs: list[float]) -> float:
    v = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not v:
        return float("inf")
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def _failed_caps(run: Mapping[str, Any]) -> list[str]:
    return [c for c, steps in run.get("control_failure_steps", {}).items() if steps]


def _recovered_all(run: Mapping[str, Any]) -> bool:
    fails = _failed_caps(run)
    if not fails:
        return True
    rec = run.get("recovery", {})
    return all(bool(rec.get(c)) for c in fails)


def cost_slope(arm_runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Protection cost as retained capabilities accumulate k=1,2,3 (phase ends B, C, D)."""
    out = {}
    for phase, k in (("B", "k1"), ("C", "k2"), ("D", "k3")):
        slots, duties = [], []
        for r in arm_runs:
            c = r.get("counters_by_phase", {}).get(phase, {})
            slots.append(float(c.get("replay_slots", 0)))
            duties.append(float(c.get("protection_updates", 0)) / max(1, int(c.get("updates", 1))))
        out[k] = {"mean_replay_slots": mean(slots), "mean_duty": mean(duties)}
    return out


def decide(arm_results: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    """Frozen verdict logic. Inputs are CONTROL-derived summaries + SEALED measurements."""
    by: dict[str, list] = {a: [] for a in ARMS}
    for rows in arm_results.values():
        for r in rows:
            if r.get("arm") in by:
                by[r["arm"]].append(r)
    if any(len(by[a]) != 4 for a in ARMS):
        return {"verdict": "INCONCLUSIVE_INCOMPLETE_MATCHED_SETS", "authorized": False}
    key = lambda r: (int(r["parent_seed"]), int(r["order_seed"]))  # noqa: E731
    maps = {a: {key(r): r for r in rows} for a, rows in by.items()}
    keys = sorted(maps["PLASTIC_HIGH"])
    if any(set(maps[a]) != set(keys) for a in ARMS):
        return {"verdict": "INCONCLUSIVE_MATCHING_ERROR", "authorized": False}

    # Formation gates: the unprotected plastic reference must qualify each new skill.
    for phase_skill in ("B", "C", "D"):
        ok = sum(1 for r in maps["PLASTIC_HIGH"].values()
                 if r["phase_confirmation"].get(phase_skill) is not None)
        if ok < 3:
            return {"verdict": f"INCONCLUSIVE_FORMATION_INSTABILITY_PHASE_{phase_skill}",
                    "authorized": False, "plastic_qualifications": ok, "phase": phase_skill}

    ph = [maps["PLASTIC_HIGH"][k] for k in keys]
    ph_fail_sets = sum(1 for r in ph if _failed_caps(r))
    ph_area = mean([mean(list(r["sealed_area"].values())) for r in ph])
    s32 = [maps["STATIC_REPLAY_1OF32"][k] for k in keys]
    s32_area = mean([mean(list(r["sealed_area"].values())) for r in s32])
    s32_total_replay = mean([float(r["counters"]["replay_slots"]) for r in s32])
    interference = ph_fail_sets >= 2 or (s32_area - ph_area) >= 0.15
    if not interference:
        return {"verdict": "INCONCLUSIVE_LOW_INTERFERENCE", "authorized": False,
                "plastic_failure_sets": ph_fail_sets, "plastic_area": ph_area,
                "static_1of32_area": s32_area}

    ph_med = {p: median([r["phase_confirmation"][p] for r in ph
                         if r["phase_confirmation"].get(p) is not None])
              for p in ("B", "C", "D")}

    details: dict[str, Any] = {}
    winners: list[str] = []
    for arm in GUARDIAN_ARMS:
        rows = [maps[arm][k] for k in keys]
        final_all_q = sum(1 for r in rows if all(qualified(r["final_sealed"][c]) for c in SKILLS))
        area = mean([mean(list(r["sealed_area"].values())) for r in rows])
        conf_ok, gaps, nll_rel, duties, replay_fracs = [], [], [], [], []
        for k, r in zip(keys, rows):
            p = maps["PLASTIC_HIGH"][k]
            conf_ok.append(all(
                r["phase_confirmation"].get(ph_) is not None and
                r["phase_confirmation"][ph_] <=
                GUARDIAN_SUCCESS_RULES["median_phase_confirmation_vs_plastic_max_ratio"] * ph_med[ph_]
                for ph_ in ("B", "C", "D")))
            gaps.append(mean([robust_min(p["final_sealed"][c]) - robust_min(r["final_sealed"][c])
                              for c in SKILLS]))
            pn, rn = float(p["final_science_sealed_nll"]), float(r["final_science_sealed_nll"])
            nll_rel.append((rn - pn) / max(pn, 1e-12))
            duties.append(float(r["counters"]["protection_updates"]) / CONTINUATION_HORIZON)
            replay_fracs.append(float(r["counters"]["replay_slots"]) / (BATCH_SLOTS * CONTINUATION_HORIZON))
        prevention = sum(1 for r in rows if not _failed_caps(r))
        recovery = sum(1 for r in rows if _recovered_all(r))
        total_replay = mean([float(r["counters"]["replay_slots"]) for r in rows])
        ok = bool(
            final_all_q >= GUARDIAN_SUCCESS_RULES["final_sealed_all_capabilities_qualified_sets"]
            and area >= s32_area - GUARDIAN_SUCCESS_RULES["mean_sealed_area_within_of_best_static"]
            and all(conf_ok)
            and mean(gaps) <= 0.05 and all(x <= 0.05 for x in nll_rel)
            and mean(duties) < GUARDIAN_SUCCESS_RULES["mean_protection_duty_max"]
            and mean(replay_fracs) < GUARDIAN_SUCCESS_RULES["mean_replay_fraction_max"]
            and total_replay <= s32_total_replay
        )
        details[arm] = {
            "final_all_qualified_sets": final_all_q,
            "all_capability_sealed_area": area,
            "phase_confirmation_within_plastic_sets": sum(conf_ok),
            "mean_final_retention_gap_vs_plastic": mean(gaps),
            "science_nll_relative_vs_plastic": nll_rel,
            "mean_protection_duty": mean(duties),
            "mean_replay_fraction": mean(replay_fracs),
            "mean_total_replay_slots": total_replay,
            "static_1of32_total_replay_slots": s32_total_replay,
            "prevention_sets": prevention,
            "recovery_within_400_sets": recovery,
            "cost_slope": cost_slope(rows),
            "qualifies": ok,
        }
        if ok:
            winners.append(arm)

    quality_pass = [a for a, d in details.items()
                    if d["final_all_qualified_sets"] >= 3
                    and d["all_capability_sealed_area"] >= s32_area - 0.10]
    if winners:
        verdict = VERDICT_SUCCESS
    elif quality_pass:
        verdict = VERDICT_QUALITY_ONLY
    else:
        verdict = VERDICT_FAILURE
    flags = []
    if any(details[a]["prevention_sets"] >= 3 for a in details):
        flags.append("PREVENTION_SIGNAL")
    if any(details[a]["recovery_within_400_sets"] >= 3 for a in details):
        flags.append("RECOVERY_SIGNAL")
    pred_ok = any(details[a]["qualifies"] for a in ("GUARDIAN_PREDICTIVE", "GUARDIAN_HYBRID"))
    reactive_ok = details["GUARDIAN_REACTIVE"]["qualifies"] if "GUARDIAN_REACTIVE" in details else False
    if pred_ok and not reactive_ok:
        flags.append("PREDICTIVE_ADDS_VALUE")
    return {"verdict": verdict, "authorized": verdict == VERDICT_SUCCESS, "flags": flags,
            "interference": interference, "plastic_failure_sets": ph_fail_sets,
            "plastic_area": ph_area, "static_1of32_area": s32_area,
            "guardian_details": details,
            "claim_ceiling": "REAL_TEXT_PROXY_MULTI_SKILL_GUARDIAN_CANDIDATE_ONLY"}
