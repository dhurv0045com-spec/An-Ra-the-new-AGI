"""ARK-020 V3 campaign runner — repaired integration, phase seeds, phase-relative
metrics, truthful exact resume, CPU smoke, and a read-only resume scan.

Reuses V3/V4 machinery with EXACT object shapes (flat templates + semantic splits —
the V1 defect #1 repair) and adds integration-contract-callable helpers.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ARK-019"))
sys.path.insert(0, str(HERE.parent / "ARK-018"))

import ark020_v4_core as C  # noqa: E402
import run_ark019_v4 as V4  # noqa: E402
import run_ark019_v3 as V3  # noqa: E402
from ark018_v3_common import Ark018GPT, eval_buffer, autocast_ctx, make_scaler, model_state_hash  # noqa: E402

OUT = Path("/content/drive/MyDrive/genisis-arkenstone/ARK020_V4_CONTINUAL")
TOKEN_THRESHOLDS = (256, 128, 64, 32, 16, 8)
EXPECTED_TOKENS = 48
CAP_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


class SessionTimebox(RuntimeError):
    pass


def savej(p: Path, x: Mapping[str, Any]) -> None:
    V3.savej(p, x)


def hjson(x: Any) -> str:
    return V3.hjson(x)


def hfile(p: Path) -> str:
    return V3.hfile(p)


def setup() -> None:
    V3.setup()


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def opt_for(m, lr):
    return torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


def snapshot(m, o, sc):
    return {"model": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()},
            "optimizer": copy.deepcopy(o.state_dict()), "scaler": copy.deepcopy(sc.state_dict()),
            "cpu_rng": torch.get_rng_state().cpu(),
            "cuda_rng": [x.cpu() for x in torch.cuda.get_rng_state_all()]}


def restore(s, d, lr=C.HIGH_LR):
    m = Ark018GPT().to(d)
    m.load_state_dict(s["model"])
    o = opt_for(m, lr)
    o.load_state_dict(s["optimizer"])
    V3.opt_to(o, d)
    for g in o.param_groups:
        g["lr"] = lr
    sc = make_scaler(d)
    sc.load_state_dict(s.get("scaler", {}))
    torch.set_rng_state(s["cpu_rng"])
    try:
        torch.cuda.set_rng_state_all([x for x in s["cuda_rng"]])
    except Exception:
        pass
    return m, o, sc


def optimizer_hash(o) -> str:
    return V4.optimizer_hash(o)


def save_checkpoint(path: Path, payload: Mapping[str, Any], m, o, sc) -> None:
    body = dict(payload)
    body.update(snapshot(m, o, sc))
    path.parent.mkdir(parents=True, exist_ok=True)
    q = path.with_suffix(path.suffix + ".tmp")
    torch.save(body, q)
    q.replace(path)


def load_checkpoint(path: Path, expected: Mapping[str, Any], d, lr=C.HIGH_LR):
    x = torch.load(path, map_location="cpu", weights_only=False)
    for k, v in expected.items():
        if x.get(k) != v:
            raise RuntimeError(f"checkpoint identity mismatch {k}: {x.get(k)!r} != {v!r}")
    m, o, sc = restore(x, d, lr)
    return x, m, o, sc


# ------------------------------------------------------------------ token gate

def select_tokens48(tok, counts) -> list[int]:
    for threshold in TOKEN_THRESHOLDS:
        candidates = []
        for tid in range(min(len(counts), tok.get_vocab_size())):
            if counts[tid] < threshold:
                continue
            text = tok.decode([tid])
            stripped = text.strip()
            if not (4 <= len(stripped) <= 10 and stripped.isascii() and stripped.isalpha() and stripped.islower()):
                continue
            if tok.encode(text).ids != [tid]:
                continue
            if int(hashlib.sha256(stripped.encode()).hexdigest(), 16) % 4 != 0:
                continue
            candidates.append((-int(counts[tid]), tid, stripped))
        candidates.sort()
        if len(candidates) >= EXPECTED_TOKENS:
            return [x[1] for x in candidates[:EXPECTED_TOKENS]]
    raise RuntimeError(
        f"ARK-020 V3 token gate failed: fewer than {EXPECTED_TOKENS} eligible single-token words")


# --------------------------------------------------------------- task rendering

def templates(tok) -> dict[str, dict[str, list[int]]]:
    return {
        "A": {"p": tok.encode("Facts: ").ids, "m": tok.encode(" means ").ids,
              "s": tok.encode("; ").ids, "q": tok.encode("Query: ").ids, "t": tok.encode(" means").ids},
        "B": {"p": tok.encode("Map: ").ids, "m": tok.encode(" -> ").ids,
              "s": tok.encode(" | ").ids, "q": tok.encode("Requested ").ids, "t": tok.encode(" =>").ids},
        "D": {"p": tok.encode("Owners: ").ids, "m": tok.encode(" holds ").ids,
              "s": tok.encode("; ").ids, "q": tok.encode("Who holds ").ids, "t": tok.encode(" ?").ids},
        "C": {"p": tok.encode("Links: ").ids, "g": tok.encode(" gives ").ids,
              "s": tok.encode("; ").ids, "k": tok.encode(" makes ").ids,
              "q": tok.encode("Trace: ").ids, "e": tok.encode(" to").ids},
    }


def logits_answers(m, rs, d):
    w = max(len(p) for p, _ in rs)
    x = torch.zeros((len(rs), w), dtype=torch.long, device=d)
    a = torch.tensor([a for _, a in rs], dtype=torch.long, device=d)
    for i, (p, _) in enumerate(rs):
        x[i, -len(p):] = torch.tensor(p, dtype=torch.long, device=d)
    return m(x)[:, -1, :], a


def binding_loss(m, rs, d):
    z, a = logits_answers(m, rs, d)
    return F.cross_entropy(z.float(), a, reduction="none")


@torch.no_grad()
def mode_acc(m, cap, t, sem, d, mode):
    hit = n = 0
    for st in range(0, len(sem), 64):
        idx = list(range(st, min(st + 64, len(sem))))
        z, a = logits_answers(m, C.task_rows(cap, t, sem, idx, mode, 0, 0), d)
        hit += int((z.argmax(-1) == a).sum())
        n += len(a)
    return hit / max(1, n)


def _finish(rec: dict) -> dict:
    rec["robust_min"] = C.robust_min(rec)
    rec["qualified"] = C.qualified(rec)
    rec["healthy"] = C.healthy(rec)
    return rec


@torch.no_grad()
def cap_control_metrics(m, cap, t, tasks, d) -> dict:
    sem = tasks["sem"][cap]["main_control"]
    rec = {"canonical": mode_acc(m, cap, t, sem, d, "canonical"),
           "order_only": mode_acc(m, cap, t, sem, d, "reversed"),
           "query_order": mode_acc(m, cap, t, sem, d, "query_order")}
    return _finish(rec)


@torch.no_grad()
def cap_validation_metrics(m, cap, t, tasks, d) -> dict:
    sem = tasks["sem"][cap]["validation"]
    rec = {"canonical": mode_acc(m, cap, t, sem, d, "canonical"),
           "order_only": mode_acc(m, cap, t, sem, d, "reversed"),
           "query_order": mode_acc(m, cap, t, sem, d, "query_order")}
    return _finish(rec)


@torch.no_grad()
def cap_sealed_metrics(m, cap, t, tasks, d) -> dict:
    sem = tasks["sem"][cap]["sealed"]
    rec = {"canonical": mode_acc(m, cap, t, sem, d, "canonical"),
           "order_only": mode_acc(m, cap, t, sem, d, "reversed"),
           "query_order": mode_acc(m, cap, t, sem, d, "query_order")}
    rec["robust_min"] = C.robust_min(rec)
    rec["qualified"] = C.qualified(rec)
    return rec


def science(m, bufs, d, sealed=True):
    out = {"control": eval_buffer(m, bufs["control"], d, 73001, sequences=24)}
    if sealed:
        out["sealed"] = eval_buffer(m, bufs["sealed"], d, 73002, sequences=24)
    return out


def real_batch(buf, n, seed, step, d, tag="main"):
    xs, ys, starts = [], [], []
    need = V3.CONTEXT + 1
    for slot in range(n):
        z = int.from_bytes(hashlib.sha256(
            f"ark020v4-real:{tag}:{seed}:{step}:{slot}".encode()).digest()[:8], "big") % (len(buf) - need + 1)
        r = np.asarray(buf[z:z + need], dtype=np.int64)
        xs.append(r[:-1])
        ys.append(r[1:])
        starts.append(z)
    return (torch.tensor(np.stack(xs), dtype=torch.long, device=d),
            torch.tensor(np.stack(ys), dtype=torch.long, device=d), starts)


def pnames(m):
    wanted = {"tok.weight", "blocks.0.attn.qkv.weight", "blocks.4.mlp.2.weight",
              "blocks.9.attn.qkv.weight", "ln_f.weight"}
    return [n for n, _ in m.named_parameters() if n in wanted]


def psnap(m, names):
    p = dict(m.named_parameters())
    return {n: p[n].detach().float().clone() for n in names}


def pdelta(x, m):
    p = dict(m.named_parameters())
    return math.sqrt(sum(float(((p[n].detach().float() - v) ** 2).sum()) for n, v in x.items()))


def backup(m):
    return [p.detach().clone() for p in m.parameters()]


def fulldelta(b, m):
    return math.sqrt(sum(float(((p.detach().float() - x.float()) ** 2).sum()) for x, p in zip(b, m.parameters())))


def cap_project(b, m, raw, cap):
    if raw <= cap or raw <= 0:
        return raw
    scale = cap / raw
    with torch.no_grad():
        for x, p in zip(b, m.parameters()):
            p.copy_(x + (p - x) * scale)
    return cap


def displacement(m, parent):
    return math.sqrt(sum(float(((v.detach().float().cpu() - parent[k].float()) ** 2).sum())
                         for k, v in m.state_dict().items()))


def projected_grad(m, names, d, rows_rs=None, x=None, y=None) -> torch.Tensor:
    m.zero_grad(set_to_none=True)
    with autocast_ctx(d):
        if rows_rs is not None:
            loss = binding_loss(m, rows_rs, d).sum()
        else:
            z = m(x)
            loss = F.cross_entropy(z.float().reshape(-1, z.size(-1)), y.reshape(-1))
    loss.backward()
    params = dict(m.named_parameters())
    chunks = [params[n].grad.detach().float().flatten().cpu() for n in names if params[n].grad is not None]
    m.zero_grad(set_to_none=True)
    return torch.cat(chunks) if chunks else torch.zeros(1)


def cosine(a, b) -> float:
    denom = float(a.norm().item() * b.norm().item())
    return float(torch.dot(a, b).item() / denom) if denom > 0 else 0.0


def mixed_update(m, o, sc, train_buf, phase_cap, phase_t, phase_sem, stream_seed, step,
                 task_slots, replay, cap, d, names, tag="main", full_delta=False):
    """replay: list of (cap_id, template, semantics) — risk-ordered old capabilities.

    stream_seed is the PHASE's own order seed (defect #3 repair).
    """
    replay = list(replay)[:C.MAX_REPLAY_SLOTS_PER_UPDATE]
    real_slots = C.BATCH_SLOTS - int(task_slots) - len(replay)
    if real_slots <= 0:
        raise RuntimeError("non-positive real-text slots")
    x, y, starts = real_batch(train_buf, real_slots, stream_seed, step, d, tag=tag)
    ti = C.deterministic_indices(stream_seed, step, task_slots, len(phase_sem), f"{tag}-task")
    tr = C.task_rows(phase_cap, phase_t, phase_sem, ti, "augmented", stream_seed, step)
    o.zero_grad(set_to_none=True)
    with autocast_ctx(d):
        z = m(x)
        tl = F.cross_entropy(z.float().reshape(-1, z.size(-1)), y.reshape(-1),
                             reduction="none").view(real_slots, -1).mean(1)
        pieces = [tl, binding_loss(m, tr, d)]
        rmeta = []
        for cap_id, t_r, sem_r in replay:
            ri = C.deterministic_indices(stream_seed, step, 1, len(sem_r),
                                         f"{tag}-replay-{CAP_INDEX[cap_id]}")[0]
            rows_r = C.task_rows(cap_id, t_r, sem_r, [ri], "nonidentity", stream_seed + 17, step)
            pieces.append(binding_loss(m, rows_r, d))
            rmeta.append({"capability": cap_id, "index": ri})
        loss = torch.cat(pieces).sum() / C.BATCH_SLOTS
    sc.scale(loss).backward()
    sc.unscale_(o)
    grad = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
    pb = psnap(m, names)
    fb = backup(m) if cap is not None or full_delta else None
    sc.step(o)
    sc.update()
    pd = pdelta(pb, m)
    raw = applied = None
    capped = False
    if fb is not None:
        raw = fulldelta(fb, m)
        applied = raw
        if cap is not None and raw > cap:
            applied = cap_project(fb, m, raw, cap)
            capped = True
    return {"loss": float(loss.detach()), "grad": grad, "projected_delta": pd,
            "raw_full_delta": raw, "applied_full_delta": applied, "capped": capped,
            "replay": rmeta, "replay_slots": len(rmeta), "task_slots": int(task_slots),
            "real_slots": real_slots, "real_starts_sha256": hjson(starts)}


# ------------------------------------------------------------- phase machinery

def _dev() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def global_step_of(phase_idx: int, step: int) -> int:
    offs = [0, C.PHASE_UPDATES["B"], C.PHASE_UPDATES["B"] + C.PHASE_UPDATES["C"]]
    return offs[phase_idx] + step


def arm_paths(ps, bs, arm):
    p = OUT / "matched_sets" / f"p{ps}_b{bs}" / arm
    return p, p / "RESULT.json", p / "RESUME.pt", p / "PARTIAL.json"


def run_set_arm(ps, bs, arm, dose_b, parent_state, cap16, bufs, tt, tasks, d, deadline,
                session_log=print):
    ad, rp, cp, pp = arm_paths(ps, bs, arm)
    ad.mkdir(parents=True, exist_ok=True)
    # per-phase order seeds, bound into identity (defect #3 repair)
    oi = C.PHASE_ORDER_SEEDS["B"].index(bs)
    c_seed = C.PHASE_ORDER_SEEDS["C"][oi]
    d_seed = C.PHASE_ORDER_SEEDS["D"][oi]
    phase_stream = {"B": bs, "C": c_seed, "D": d_seed}
    parent_sha = V3.state_hash(parent_state["model"])
    task_hash = hjson({k: hjson(tasks[k]) for k in ("A", "B", "C", "D")})
    expected = {"schema": "arkenstone-ark020-v4-arm-ckpt/v1", "parent_seed": ps,
                "b_order_seed": bs, "c_order_seed": c_seed, "d_order_seed": d_seed,
                "arm": arm, "dose_b": int(dose_b), "parent_sha": parent_sha,
                "task_hash": task_hash, "cap16x": float(cap16)}
    if rp.exists():
        r = json.loads(rp.read_text())
        if r.get("status") == "COMPLETE" and r.get("parent_sha") == parent_sha:
            return r
        raise RuntimeError(f"incompatible ARK-020 V3 arm result {rp}")
    if cp.exists():
        x, m, o, sc = load_checkpoint(cp, expected, d)
        start_phase = int(x["phase_idx"])
        start_step = int(x["phase_step"])
        reg, ctrl = x["registry"], x["controller"]
        cnt = x["counters"]
        b_streaks = x["b_streaks"]
        control_trace = C.dedupe_trajectory(x.get("control_trace", []))
        measure_trace = C.dedupe_trajectory(x.get("measure_trace", []))
        retention_events = list(x.get("retention_events", []))
        phase_confirm = x["phase_confirm"]
        global_confirm = x["global_confirm"]
    else:
        m, o, sc = restore(parent_state, d, C.HIGH_LR)
        start_phase, start_step = 0, 0
        reg, ctrl = C.init_registry(), C.initial_controller(arm)
        C.register_capability(reg, "A", 0)
        cnt = {"replay_slots": 0, "real_slots": 0, "task_slots": 0, "capped_steps": 0,
               "protection_updates": 0, "projected_path": 0.0, "replay_slots_by_cap": {},
               "counters_by_phase": {p: {"replay_slots": 0, "protection_updates": 0,
                                         "updates": C.PHASE_UPDATES[p]} for p in C.PHASES}}
        b_streaks = {"B": 0, "C": 0, "D": 0}
        control_trace, measure_trace = [], []
        retention_events = []
        phase_confirm = {"B": None, "C": None, "D": None}
        global_confirm = {"B": None, "C": None, "D": None}
        control_trace.append({"step": 0, "phase": "PARENT",
                              "metrics": {"A": cap_control_metrics(m, "A", tt, tasks, d)}})
        measure_trace.append({"step": 0, "phase": "PARENT",
                              "sealed": {"A": cap_sealed_metrics(m, "A", tt, tasks, d)},
                              "science": science(m, bufs, d, sealed=True),
                              "full_displacement": 0.0, "diagnostics": {}})
    names = pnames(m)
    parent_model = parent_state["model"]
    phase_slots = {"B": dose_b, "C": C.PHASE_TASK_SLOTS["C"], "D": C.PHASE_TASK_SLOTS["D"]}

    def timebox(phase_idx, phase, step):
        payload = {**expected, "phase_idx": phase_idx, "phase": phase, "phase_step": step - 1,
                   "registry": reg, "controller": ctrl, "counters": cnt,
                   "phase_confirm": phase_confirm, "global_confirm": global_confirm,
                   "retention_events": retention_events,
                   "control_trace": control_trace, "measure_trace": measure_trace,
                   "b_streaks": b_streaks}
        save_checkpoint(cp, payload, m, o, sc)
        savej(pp, {"schema": "arkenstone-ark020-v4-partial/v1", "status": "PARTIAL_SESSION",
                   "parent_seed": ps, "b_order_seed": bs, "arm": arm, "phase": phase,
                   "phase_step": step - 1, "global_step": global_step_of(phase_idx, step - 1),
                   "counters": cnt})
        raise SessionTimebox(f"arm {ps}/{bs}/{arm} paused at {phase} step {step-1}")

    for phase_idx in range(start_phase, len(C.PHASES)):
        phase = C.PHASES[phase_idx]
        slots = phase_slots[phase]
        stream = phase_stream[phase]
        phase_sem = tasks["sem"][phase]["train"]
        phase_t = tt[phase]
        start = start_step + 1 if phase_idx == start_phase else 1
        for step in range(start, C.PHASE_UPDATES[phase] + 1):
            if time.monotonic() >= deadline:
                timebox(phase_idx, phase, step)
            gstep = global_step_of(phase_idx, step)
            qualified_old = [c for c in ("A", "B", "C", "D")
                             if c in reg["capabilities"] and c != phase]
            if arm == "PLASTIC_HIGH":
                replay, capv = [], None
            else:
                replay, capv = C.treatment(arm, ctrl, gstep, reg, qualified_old, cap16)
            cnt["protection_updates"] += int(bool(replay or capv is not None))
            cnt["counters_by_phase"][phase]["protection_updates"] += int(bool(replay or capv is not None))
            rec = mixed_update(m, o, sc, bufs["train"], phase, phase_t, phase_sem, stream, step,
                               slots, [(cid, tt[cid], tasks["sem"][cid]["train"]) for cid in replay],
                               capv, d, names, tag=f"{arm}-{phase}")
            cnt["replay_slots"] += rec["replay_slots"]
            cnt["counters_by_phase"][phase]["replay_slots"] += rec["replay_slots"]
            cnt["real_slots"] += rec["real_slots"]
            cnt["task_slots"] += int(slots)
            cnt["capped_steps"] += int(rec["capped"])
            cnt["projected_path"] += float(rec["projected_delta"])
            for cid in replay:
                cnt["replay_slots_by_cap"][cid] = cnt["replay_slots_by_cap"].get(cid, 0) + 1
            if step % C.CONTROL_EVERY == 0:
                caps_now = sorted(set(qualified_old) | {phase})
                metrics = {c: cap_control_metrics(m, c, tt, tasks, d) for c in caps_now}
                targets = C.retention_targets(set(reg["capabilities"].keys()), phase,
                                              phase_confirm.get(phase) is not None)
                for cid in qualified_old:
                    C.observe_capability(reg, cid, gstep, metrics[cid])
                    if cid in targets:
                        C.record_retention_event(retention_events, cid, gstep, phase, True,
                                                 metrics[cid])
                C.update_controller(arm, ctrl, gstep, reg, qualified_old)
                b_streaks[phase] = b_streaks[phase] + 1 if C.qualified(metrics[phase]) else 0
                if b_streaks[phase] >= C.CONFIRM_STREAK and phase_confirm[phase] is None:
                    val = cap_validation_metrics(m, phase, tt, tasks, d)
                    if C.qualified(val):
                        phase_confirm[phase] = step          # PHASE-RELATIVE (defect #4 fix)
                        global_confirm[phase] = gstep        # recorded, never compared
                        C.register_capability(reg, phase, gstep)
                        C.set_post_phase_floor(ctrl, [phase], gstep)
                        b_streaks[phase] = 0
                control_trace.append({"step": gstep, "phase": phase, "metrics": metrics,
                                      "controller": copy.deepcopy(ctrl)})
                session_log(f"ARK020V4 [{ps}/{bs}/{arm}/{phase}] {step}/{C.PHASE_UPDATES[phase]} "
                            f"A={metrics['A']['robust_min']:.3f} "
                            f"{phase}={metrics[phase]['robust_min']:.3f} state={ctrl['states']}")
            if step % C.MEASURE_EVERY == 0:
                sealed = {c: cap_sealed_metrics(m, c, tt, tasks, d) for c in caps_now}
                sci = science(m, bufs, d, sealed=True)
                disp = displacement(m, parent_model)
                diag = {}
                if qualified_old:
                    tri = C.deterministic_indices(stream, step, 8, len(phase_sem), "diag-task")
                    gt = projected_grad(m, names, d,
                                        rows_rs=C.task_rows(phase, phase_t, phase_sem, tri,
                                                            "augmented", stream, step))
                    for cid in qualified_old:
                        csem = tasks["sem"][cid]["train"]
                        cri = C.deterministic_indices(stream, step, 4, len(csem),
                                                      f"diag-{CAP_INDEX[cid]}")
                        gc = projected_grad(m, names, d,
                                            rows_rs=C.task_rows(cid, tt[cid], csem, cri,
                                                                "nonidentity", stream + 31, step))
                        diag[cid] = {"task_vs_old_grad_cosine": cosine(gt, gc)}
                measure_trace.append({"step": gstep, "phase": phase, "sealed": sealed,
                                      "science": sci, "full_displacement": disp,
                                      "last_projected_delta": rec["projected_delta"],
                                      "diagnostics": diag})
                savej(pp, {"schema": "arkenstone-ark020-v4-partial/v1", "status": "RUNNING",
                           "parent_seed": ps, "b_order_seed": bs, "arm": arm, "phase": phase,
                           "phase_step": step, "global_step": gstep, "counters": cnt,
                           "measure_trace": measure_trace[-3:]})
            if step % C.CHECKPOINT_EVERY == 0:
                payload = {**expected, "phase_idx": phase_idx, "phase": phase, "phase_step": step,
                           "registry": reg, "controller": ctrl, "counters": cnt,
                           "phase_confirm": phase_confirm, "global_confirm": global_confirm,
                           "retention_events": retention_events,
                           "control_trace": control_trace, "measure_trace": measure_trace,
                           "b_streaks": b_streaks}
                save_checkpoint(cp, payload, m, o, sc)
        if arm in C.GUARDIAN_ARMS:
            C.set_post_phase_floor(ctrl, list(reg["capabilities"].keys()), gstep)
    control_trace = C.dedupe_trajectory(control_trace)
    measure_trace = C.dedupe_trajectory(measure_trace)
    f = measure_trace[-1]
    failure_steps: dict[str, list[int]] = {}
    recovery: dict[str, bool] = {}
    for ev in retention_events:
        failure_steps.setdefault(ev["capability"], []).append(int(ev["failure_step"]))
    for cid, steps in failure_steps.items():
        first = steps[0]
        ok = False
        for row in control_trace:
            s = int(row["step"])
            if s < first or cid not in row["metrics"]:
                continue
            if s - first > C.RECOVERY_WINDOW_UPDATES:
                break
            if C.healthy(row["metrics"][cid]):
                ok = True
                break
        recovery[cid] = bool(ok)
    sealed_area = {}
    for cid in C.SKILLS:
        vals = [row["sealed"][cid]["robust_min"] for row in measure_trace if cid in row.get("sealed", {})]
        sealed_area[cid] = C.mean(vals) if vals else 0.0
    r = {"schema": "arkenstone-ark020-v4-arm/v1", "status": "COMPLETE", "parent_seed": ps,
         "b_order_seed": bs, "c_order_seed": c_seed, "d_order_seed": d_seed,
         "arm": arm, "dose_b": int(dose_b), "cap16x": float(cap16), "parent_sha": parent_sha,
         "phase_confirmation": phase_confirm, "global_confirmation": global_confirm,
         "final_sealed": f["sealed"], "sealed_area": sealed_area,
         "control_failure_steps": failure_steps, "recovery": recovery,
         "final_science_sealed_nll": float(f["science"]["sealed"]["nll"]),
         "retention_failure_events": retention_events,
         "counters": cnt, "counters_by_phase": cnt["counters_by_phase"],
         "controller": ctrl,
         "registry": {"capabilities": {
             k: {kk: vv for kk, vv in v.items() if kk != "margin_history"}
             for k, v in reg["capabilities"].items()}},
         "control_trace": control_trace, "measure_trace": measure_trace,
         "final_model_sha256": model_state_hash(m), "final_optimizer_sha256": optimizer_hash(o)}
    savej(rp, r)
    cp.unlink(missing_ok=True)
    del m, o
    torch.cuda.empty_cache()
    return r


def collect_arm_results(root: Path | None = None):
    root = root or OUT
    out = {}
    for p in root.glob("matched_sets/p*_b*/*/RESULT.json"):
        r = json.loads(p.read_text())
        out[f"{r['parent_seed']}:{r['b_order_seed']}:{r['arm']}"] = r
    return out


def group_for_decide(collected):
    grouped = {a: [] for a in C.ARMS}
    for r in collected.values():
        if r.get("arm") in grouped:
            grouped[r["arm"]].append(r)
    return grouped


# ------------------------------------------------------- exact resume (defect #5)

def state_hashes(st: dict) -> dict:
    """Canonical semantic hashes for EVERY state class that matters."""
    return {
        "model": model_state_hash(st["m"]),
        "optimizer": optimizer_hash(st["o"]),
        "scaler": hjson(st["sc"].state_dict()),
        "cpu_rng": hjson(torch.get_rng_state().tolist()),
        "cuda_rng": hjson([x.tolist() for x in torch.cuda.get_rng_state_all()])
        if torch.cuda.is_available() else "cpu-device",
        "registry": hjson(st["reg"]),
        "controller": hjson(st["ctrl"]),
        "counters": hjson(st["cnt"]),
        "phase_identity": hjson({"phase_idx": st["phase_idx"], "phase": st["phase"],
                                 "phase_step": st["phase_step"], "global_step": st["global_step"],
                                 "b_order_seed": st["b_seed"], "c_order_seed": st["c_seed"],
                                 "d_order_seed": st["d_seed"]}),
        "confirmations": hjson({"phase_confirm": st["phase_confirm"],
                                "global_confirm": st["global_confirm"],
                                "b_streaks": st["b_streaks"]}),
        "telemetry": hjson({"control_trace_tail": st["control_trace"][-3:],
                            "measure_trace_tail": st["measure_trace"][-3:]}),
        "semantic_stream": hjson(st["stream_receipts"]),
    }


def _fresh_state(parent_state, seeds: dict, device: torch.device) -> dict:
    m, o, sc = restore(parent_state, device)
    reg = C.init_registry()
    C.register_capability(reg, "A", 0)
    return {"m": m, "o": o, "sc": sc, "reg": reg,
            "ctrl": C.initial_controller("GUARDIAN_HYBRID"),
            "cnt": {"replay_slots": 0, "protection_updates": 0, "capped_steps": 0,
                    "replay_slots_by_cap": {}},
            "phase_idx": 0, "phase": "B", "phase_step": 0,
            "global_step": 0, "b_seed": seeds["B"], "c_seed": seeds["C"], "d_seed": seeds["D"],
            "phase_confirm": {"B": None, "C": None, "D": None},
            "global_confirm": {"B": None, "C": None, "D": None},
            "b_streaks": {"B": 0, "C": 0, "D": 0},
            "control_trace": [], "measure_trace": [], "stream_receipts": []}


def _advance(st, tasks, tt, bufs, dose_b, start: int, stop: int) -> list:
    rows = []
    names = pnames(st["m"])
    for step in range(start, stop + 1):
        rep = ([("A", tt["A"], tasks["sem"]["A"]["train"])]
               if step % 2 == 0 else [])
        r = mixed_update(st["m"], st["o"], st["sc"], bufs["train"], "B", tt["B"],
                         tasks["sem"]["B"]["train"], st["b_seed"], step, dose_b, rep, None,
                         _dev(), names, tag="resume-smoke")
        st["phase_step"] = step
        st["global_step"] += 1
        st["cnt"]["replay_slots"] += r["replay_slots"]
        st["stream_receipts"].append({"step": step, "real_starts_sha256": r["real_starts_sha256"]})
        rows.append((step, r["replay_slots"], r["real_starts_sha256"], round(r["loss"], 10)))
    return rows


def exact_resume_smoke(parent_state, dose_b, bufs, tt, tasks, d) -> dict:
    """Production preexecution gate: 10 uninterrupted == 5 + SAVE + destroy + LOAD + 5.

    Every state class is hashed independently (mission #4); the checkpoint round trip
    uses the REAL save_checkpoint/load_checkpoint with the REAL identity binding, and
    reconstruction rebuilds the full working state from the loaded payload.
    """
    # CUDA determinism: SDPA backward is nondeterministic by default on GPU, which
    # breaks exact-resume equality. Enforce deterministic kernels for the smoke
    # (the campaign runs under setup(), but the smoke must be self-sufficient).
    from ark018_v3_common import setup_reproducibility
    setup_reproducibility()
    p = OUT / "EXACT_RESUME_SMOKE_V4.json"
    if p.exists():
        return json.loads(p.read_text())
    seeds = {"B": C.PHASE_ORDER_SEEDS["B"][0], "C": C.PHASE_ORDER_SEEDS["C"][0],
             "D": C.PHASE_ORDER_SEEDS["D"][0]}
    a = _fresh_state(parent_state, seeds, d)
    t1 = _advance(a, tasks, tt, bufs, dose_b, 1, 10)
    h1 = state_hashes(a)
    del a
    b = _fresh_state(parent_state, seeds, d)
    t2a = _advance(b, tasks, tt, bufs, dose_b, 1, 5)
    parent_sha = V3.state_hash(parent_state["model"])
    task_hash = hjson({k: hjson(tasks[k]) for k in ("A", "B", "C", "D")})
    expected = {"schema": "arkenstone-ark020-v4-arm-ckpt/v1", "parent_seed": 31801,
                "b_order_seed": seeds["B"], "c_order_seed": seeds["C"],
                "d_order_seed": seeds["D"], "arm": "GUARDIAN_HYBRID", "dose_b": int(dose_b),
                "parent_sha": parent_sha, "task_hash": task_hash, "cap16x": 1.0}
    payload = {**expected, "phase_idx": b["phase_idx"], "phase": b["phase"],
               "phase_step": b["phase_step"], "registry": b["reg"], "controller": b["ctrl"],
               "counters": b["cnt"], "phase_confirm": b["phase_confirm"],
               "global_confirm": b["global_confirm"], "b_streaks": b["b_streaks"],
               "control_trace": b["control_trace"], "measure_trace": b["measure_trace"],
               "stream_receipts": b["stream_receipts"], "global_step": b["global_step"]}
    q = OUT / "_smoke_ckpt.pt"
    save_checkpoint(q, payload, b["m"], b["o"], b["sc"])
    del b
    x, m2, o2, sc2 = load_checkpoint(q, expected, d)
    b2 = {"m": m2, "o": o2, "sc": sc2, "reg": x["registry"], "ctrl": x["controller"],
          "cnt": x["counters"], "phase_idx": x["phase_idx"], "phase": x["phase"],
          "phase_step": x["phase_step"], "global_step": x["global_step"],
          "b_seed": seeds["B"], "c_seed": seeds["C"], "d_seed": seeds["D"],
          "phase_confirm": x["phase_confirm"], "global_confirm": x["global_confirm"],
          "b_streaks": x["b_streaks"], "control_trace": list(x["control_trace"]),
          "measure_trace": list(x["measure_trace"]),
          "stream_receipts": list(x["stream_receipts"])}
    t2b = _advance(b2, tasks, tt, bufs, dose_b, 6, 10)
    h2 = state_hashes(b2)
    q.unlink(missing_ok=True)
    fields = {k: h1[k] == h2[k] for k in h1}
    ok = t1 == t2a + t2b and all(fields.values())
    r = {"schema": "arkenstone-ark020-v4-exact-resume/v1", "status": "PASS" if ok else "FAIL",
         "telemetry_identical": t1 == t2a + t2b, "fields": fields,
         "coverage": sorted(h1.keys()), "note": "CUDA RNG hashed only on CUDA devices; "
         "on CPU the field is marked cpu-device (split evidence per mission #5)"}
    savej(p, r)
    if not ok:
        raise RuntimeError(f"ARK-020 V3 exact-resume smoke failed: {fields}")
    del m2, o2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return r


# ------------------------------------------------------------------ packaging

LOCK_STALE_HOURS = 6.0


def _session_uuid() -> str:
    import uuid
    return uuid.uuid4().hex[:16]


def write_lock() -> dict:
    import socket
    import uuid
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "CAMPAIGN_LOCK.json"
    body = {"experiment": "ARK-020-V3", "executable_commit": _pinned_commit(),
            "session_uuid": uuid.uuid4().hex, "timestamp": time.time(),
            "host": socket.gethostname()}
    lock.write_text(json.dumps(body, indent=2))
    return body


def read_lock() -> tuple[str, dict]:
    """Returns (classification, info): ACTIVE | STALE | MALFORMED | NONE."""
    lock = OUT / "CAMPAIGN_LOCK.json"
    if not lock.exists():
        return "NONE", {}
    try:
        body = json.loads(lock.read_text())
        required = {"experiment", "executable_commit", "session_uuid", "timestamp"}
        if not required.issubset(body.keys()):
            return "MALFORMED", {"path": str(lock)}
    except Exception:
        return "MALFORMED", {"path": str(lock)}
    age_h = (time.time() - float(body["timestamp"])) / 3600.0
    return ("ACTIVE" if age_h < LOCK_STALE_HOURS else "STALE"), {**body, "age_hours": round(age_h, 2)}


def advisory_lock():
    cls, info = read_lock()
    if cls == "ACTIVE":
        raise RuntimeError(
            f"ACTIVE writer lock on {OUT} (session {info.get('session_uuid')}, "
            f"age {info.get('age_hours')}h). Never run two concurrent sessions.")
    if cls == "MALFORMED":
        raise RuntimeError(f"Malformed campaign lock on {OUT}; inspect manually before rerun.")
    return write_lock()


def _pinned_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def package(root: Path, final=False):
    manifests = {}
    for p in sorted(root.rglob("*.json")):
        if p.name == "ZIP_MANIFEST.json":
            continue
        manifests[str(p.relative_to(root))] = hfile(p)
    savej(root / "ZIP_MANIFEST.json", {"schema": "arkenstone-ark020-v4-manifest/v1", "members": manifests})
    name = "ARKENSTONE_ARK020_V4_CONTINUAL_RESULTS.zip" if final else "ARKENSTONE_ARK020_V4_CONTINUAL_PARTIAL.zip"
    z = root / name
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for p in sorted(root.rglob("*.json")):
            f.write(p, str(p.relative_to(root)))
    (root / (name + ".sha256")).write_text(hfile(z) + "  " + name + "\n")
    return {"path": str(z), "sha256": hfile(z)}


# ------------------------------------------------------- resume scan (cell 0)

def resume_scan(drive_ok: bool = True) -> dict:
    """READ-ONLY operator scan. Verifies as much identity as possible without
    mutating state; never overclaims (mission #15/#16)."""
    info: dict[str, Any] = {
        "CAMPAIGN_ROOT": str(OUT),
        "EXPERIMENT_VERSION": "ARK-020-V4",
        "SAFE_ACTION": None,
    }
    if not drive_ok:
        info["SAFE_ACTION"] = "STOP — DRIVE UNAVAILABLE"
        return info
    info["PINNED_COMMIT"] = _pinned_commit()
    cls, lock_info = read_lock()
    info["WRITER_LOCK"] = cls
    if cls == "ACTIVE":
        info["SAFE_ACTION"] = "WAIT — ACTIVE WRITER LOCK EXISTS"
        info["LOCK"] = lock_info
        return info
    if cls == "MALFORMED":
        info["SAFE_ACTION"] = "STOP — CHECKPOINT IDENTITY FAILURE"
        info["LOCK"] = "MALFORMED (inspect manually)"
        return info
    collected = collect_arm_results()
    info["COMPLETED_ARMS"] = len(collected)
    info["TOTAL_ARMS"] = len(C.ARMS) * 4
    partials = sorted(OUT.glob("matched_sets/p*_b*/*/RESUME.pt"))
    receipt = OUT / "ENTRY_RECEIPT.json"
    receipt_data = {}
    if receipt.exists():
        try:
            receipt_data = json.loads(receipt.read_text())
        except Exception:
            receipt_data = {}
    session = OUT / "SESSION_STATE.json"
    if session.exists():
        try:
            info["LAST_SESSION_STATUS"] = json.loads(session.read_text()).get("status")
        except Exception:
            info["LAST_SESSION_STATUS"] = "UNREADABLE"
    if not partials:
        info["CHECKPOINT_FOUND"] = False
        info["CHECKPOINT_IDENTITY"] = "N/A"
        info["SAFE_ACTION"] = ("RESUME" if info["COMPLETED_ARMS"] > 0
                               else "START NEW CAMPAIGN")
        return info
    info["CHECKPOINT_FOUND"] = True
    latest = max(partials, key=lambda p: p.stat().st_mtime)
    try:
        x = torch.load(latest, map_location="cpu", weights_only=False)
    except Exception as exc:
        info["CHECKPOINT_IDENTITY"] = f"FAIL ({type(exc).__name__})"
        info["SAFE_ACTION"] = "STOP — CHECKPOINT IDENTITY FAILURE"
        return info
    checks: dict[str, str] = {}
    if x.get("schema") != "arkenstone-ark020-v4-arm-ckpt/v1":
        checks["schema"] = "FAIL"
    checks["experiment_version"] = "PASS" if str(x.get("schema", "")).startswith(
        "arkenstone-ark020-v4") else "FAIL"
    for f in ("parent_seed", "b_order_seed", "c_order_seed", "d_order_seed", "arm",
              "dose_b", "parent_sha", "task_hash", "cap16x"):
        checks[f] = "PRESENT" if f in x else "MISSING"
    if receipt_data:
        th = receipt_data.get("task_hashes", {})
        if th and x.get("task_hash") != hjson({k: v for k, v in th.items()}):
            checks["task_hash_vs_receipt"] = "FAIL"
        else:
            checks["task_hash_vs_receipt"] = "PASS"
        pid = OUT / "PARENT_IDENTITIES.json"
        if pid.exists():
            try:
                pdata = json.loads(pid.read_text())["parents"]
                acq = pdata[str(x.get("parent_seed"))]["acquired_parent_model_sha256"]
                checks["parent_hash_vs_acquired_parent"] = ("PASS" if x.get("parent_sha") == acq
                                                            else "FAIL")
            except Exception:
                checks["parent_hash_vs_acquired_parent"] = "PARTIAL_IDENTITY_CHECK"
        else:
            checks["parent_hash_vs_acquired_parent"] = "PARTIAL_IDENTITY_CHECK (receipt absent)"
    else:
        checks["entry_receipt"] = "PARTIAL_IDENTITY_CHECK (receipt absent)"
    dose_file = OUT / "DOSE_SELECTION_IMPORTED.json"
    if x.get("dose_b") is not None and dose_file.exists():
        try:
            dd = json.loads(dose_file.read_text())
            slots = int(dd.get("v4_local_selected_slots", dd.get("selected_b_slots", -1)))
            checks["dose_vs_record"] = ("PASS" if slots == int(x["dose_b"])
                                        and dd.get("verification_status") == "verified"
                                        else "FAIL")
        except Exception:
            checks["dose_vs_record"] = "PARTIAL_IDENTITY_CHECK"
    else:
        checks["dose_vs_record"] = "PARTIAL_IDENTITY_CHECK (no imported dose receipt)"
    info["ACTIVE_ARM"] = x.get("arm")
    info["ACTIVE_PHASE"] = x.get("phase")
    info["SAVED_PHASE_STEP"] = x.get("phase_step")
    info["GLOBAL_STEP"] = global_step_of(int(x.get("phase_idx", 0)), int(x.get("phase_step", 0)))
    info["IDENTITY_CHECKS"] = checks
    if x.get("schema") != "arkenstone-ark020-v4-arm-ckpt/v1":
        info["CHECKPOINT_IDENTITY"] = "FAIL"
        info["SAFE_ACTION"] = "STOP — CAMPAIGN VERSION MISMATCH"
    elif any(v == "FAIL" for v in checks.values()):
        info["CHECKPOINT_IDENTITY"] = "FAIL"
        info["SAFE_ACTION"] = "STOP — CHECKPOINT IDENTITY FAILURE"
    elif any(v.startswith("PARTIAL") for v in checks.values()):
        info["CHECKPOINT_IDENTITY"] = "PARTIAL_IDENTITY_CHECK"
        info["SAFE_ACTION"] = "RESUME"
    else:
        info["CHECKPOINT_IDENTITY"] = "PASS"
        info["SAFE_ACTION"] = "RESUME"
    return info


def calibrate(parent_state, dose_b, bufs, tt, tasks, d):
    p = OUT / "RUNTIME_CALIBRATION.json"
    if p.exists():
        return json.loads(p.read_text())
    rec = {}
    rep2 = [("A", tt["A"], tasks["sem"]["A"]["train"]),
            ("B", tt["B"], tasks["sem"]["B"]["train"])]
    cases = [("PLASTIC", [], None), ("REPLAY2", rep2, None), ("CAP", [], 1e9)]
    for label, rep, capv in cases:
        m, o, sc = restore(parent_state, d)
        names = pnames(m)
        for step in (1, 2):
            mixed_update(m, o, sc, bufs["train"], "B", tt["B"], tasks["sem"]["B"]["train"],
                         449991, step, dose_b, rep, capv, d, names, tag="runtime",
                         full_delta=label == "CAP")
        if d.type == "cuda":
            torch.cuda.synchronize()
        st = time.monotonic()
        for step in range(3, 8):
            mixed_update(m, o, sc, bufs["train"], "B", tt["B"], tasks["sem"]["B"]["train"],
                         449991, step, dose_b, rep, capv, d, names, tag="runtime",
                         full_delta=label == "CAP")
        if d.type == "cuda":
            torch.cuda.synchronize()
        rec[label] = {"update_seconds": (time.monotonic() - st) / 5}
        del m, o
        if d.type == "cuda":
            torch.cuda.empty_cache()
    m, o, sc = restore(parent_state, d)
    st = time.monotonic()
    cap_control_metrics(m, "A", tt, tasks, d)
    cap_control_metrics(m, "B", tt, tasks, d)
    control_sec = time.monotonic() - st
    st = time.monotonic()
    for cap in ("A", "B", "C", "D"):
        cap_sealed_metrics(m, cap, tt, tasks, d)
    science(m, bufs, d, sealed=True)
    measure_sec = time.monotonic() - st
    del m, o
    if d.type == "cuda":
        torch.cuda.empty_cache()
    worst = max(v["update_seconds"] for v in rec.values())
    n_sets = len(C.PARENT_SEEDS) * 2
    main_updates = n_sets * len(C.ARMS) * C.CONTINUATION_HORIZON
    control_events = n_sets * len(C.ARMS) * (C.CONTINUATION_HORIZON // C.CONTROL_EVERY)
    measure_events = n_sets * len(C.ARMS) * (C.CONTINUATION_HORIZON // C.MEASURE_EVERY)
    est = (main_updates * worst + control_events * control_sec + measure_events * measure_sec) * C.RUNTIME_SAFETY_FACTOR
    sessions = max(1, math.ceil(est / ((C.SESSION_WALL_MINUTES - C.PACKAGING_RESERVE_MINUTES) * 60)))
    r = {"schema": "arkenstone-ark020-v4-runtime/v1", "status": "PASS", "per_update_seconds": rec,
         "control_battery_seconds": control_sec, "measure_battery_seconds": measure_sec,
         "estimated_main_seconds": est, "estimated_main_sessions": sessions,
         "projected_gpu_hours": est / 3600.0,
         "protocol_changes_from_runtime": False}
    savej(p, r)
    return r


def cap_calibration(ps, bs, parent, bufs, tt, tasks, d, deadline):
    p = OUT / "matched_sets" / f"p{ps}_b{bs}" / "CAP_CALIBRATION.json"
    if p.exists():
        return json.loads(p.read_text())
    if time.monotonic() >= deadline:
        raise SessionTimebox("before cap calibration")
    m, o, sc = restore(parent, d, C.LOW_LR)
    names = pnames(m)
    ds = []
    for step in range(1, C.CAP_SHADOW_STEPS + 1):
        rec = mixed_update(m, o, sc, bufs["train"], "B", tt["B"], tasks["sem"]["B"]["train"],
                           bs, step, 12, [], None, d, names, tag="cap-shadow", full_delta=True)
        ds.append(float(rec["applied_full_delta"]))
    med = float(np.median(ds))
    r = {"schema": "arkenstone-ark020-v4-capcal/v1", "parent_seed": ps, "b_order_seed": bs,
         "shadow_lr": C.LOW_LR, "steps": C.CAP_SHADOW_STEPS, "median_low_delta": med,
         "cap16x": C.CAP_MULTIPLIER * med, "parent_model_sha256": V3.state_hash(parent["model"])}
    savej(p, r)
    del m, o
    if d.type == "cuda":
        torch.cuda.empty_cache()
    return r


# ------------------------------------------------------- CPU integration smoke

def cpu_integration_smoke(root: Path) -> dict:
    """Full executable path on CPU with tiny geometry. NOT scientific evidence.

    Proves: task build, V4 boundary shapes, one B/C/D update each, controller
    observation, replay update, checkpoint save/load, packaging, result schema.
    """
    root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schema": "arkenstone-ark020-v4-cpu-smoke/v1", "steps": []}

    def step(name, ok, detail=""):
        report["steps"].append({"step": name, "ok": bool(ok), "detail": str(detail)[:300]})
        if not ok:
            raise RuntimeError(f"smoke step failed: {name}: {detail}")

    d = torch.device("cpu")
    # 1. task build with tiny token ids
    ids = list(range(100, 148))
    tasks = C.build_all_tasks(ids)
    step("task_build", len(tasks["sem"]["C"]["train"]) == 1200)
    # 2. tiny model + buffers shaped like the real substrate
    model = Ark018GPT()  # production geometry: restore() rebuilds this exact shape
    rngbuf = np.random.RandomState(0).randint(0, 8000, size=8193 * 2).astype(np.uint16)
    # 3. V4 boundary shapes: bmetrics with flat template + semantics (V1 defect #1 class)
    tok_like = {k: [900 + i for i in range(1)] for k in ("p", "m", "s", "q", "t")}
    tA_flat = {"p": [1], "m": [2], "s": [3], "q": [4], "t": [5]}
    try:
        V4.bmetrics(model, tA_flat, tasks["sem"]["A"]["main_control"][:8], d)
        step("v4_bmetrics_boundary", True)
    except Exception as exc:
        step("v4_bmetrics_boundary", False, repr(exc))
    # 4. V4 mixed_update shape contract
    names = pnames(model)
    o = opt_for(model, 1e-4)
    sc = make_scaler(d)
    try:
        rec = V4.mixed_update(model, o, sc, rngbuf, tA_flat, tasks["sem"]["A"]["train"][:16],
                              555001, 1, 8, tA_flat, tasks["sem"]["A"]["train"][:16], 0, None,
                              d, names, tag="smoke")
        step("v4_mixed_update_boundary", "loss" in rec)
    except Exception as exc:
        step("v4_mixed_update_boundary", False, repr(exc))
    # 5. one B/C/D update each through the V2 path (per-phase seeds differ)
    o2 = opt_for(model, 1e-4)
    sc2 = make_scaler(d)
    outs = {}
    for phase, seed in (("B", 429001), ("C", 429003), ("D", 429005)):
        outs[phase] = mixed_update(model, o2, sc2, rngbuf, phase, templates_like(phase),
                                   tasks["sem"][phase]["train"][:16], seed, 1, 8, [], None,
                                   d, names, tag=f"smoke-{phase}")
    step("b_c_d_updates", all("loss" in v for v in outs.values()))
    # 6. controller observation + replay update
    reg, ctrl = C.init_registry(), C.initial_controller("GUARDIAN_HYBRID")
    C.register_capability(reg, "A", 0)
    mm = {"canonical": 0.96, "order_only": 0.93, "query_order": 0.93}
    C.observe_capability(reg, "A", 1, mm)
    C.update_controller("GUARDIAN_HYBRID", ctrl, 1, reg, ["A"])
    rep, capv = C.treatment("GUARDIAN_HYBRID", ctrl, 2, reg, ["A"], None)
    rec = mixed_update(model, o2, sc2, rngbuf, "B", templates_like("B"),
                       tasks["sem"]["B"]["train"][:16], 429001, 2, 8,
                       [(cid, templates_like(cid), tasks["sem"][cid]["train"][:16]) for cid in rep],
                       capv, d, names, tag="smoke-replay")
    step("controller_and_replay_update", rec["replay_slots"] == len(rep))
    # 7. checkpoint save/load with identity
    payload = {"schema": "smoke", "phase_idx": 0, "phase_step": 2, "registry": reg,
               "controller": ctrl, "counters": {}, "phase_confirm": {}, "global_confirm": {},
               "b_streaks": {}, "control_trace": [], "measure_trace": []}
    q = root / "_smoke_ckpt.pt"
    save_checkpoint(q, payload, model, o2, sc2)
    x, m2, o3, sc3 = load_checkpoint(q, payload, d)
    step("checkpoint_roundtrip", x["schema"] == "smoke")
    # 8. fail-closed identity
    bad = dict(payload)
    bad["phase_step"] = 3
    try:
        load_checkpoint(q, bad, d)
        step("fail_closed_identity", False, "corrupted identity accepted")
    except RuntimeError:
        step("fail_closed_identity", True)
    q.unlink(missing_ok=True)
    # 9. packaging
    (root / "dummy.json").write_text("{}")
    bundle = package(root, final=False)
    step("packaging", Path(bundle["path"]).exists())
    report["status"] = "PASS"
    return report


_TEMPLATES_TINY_CACHE: dict[str, dict] = {}


def templates_like(cap: str) -> dict:
    """Tiny single-token templates for CPU smoke (ids chosen to be distinct)."""
    if cap not in _TEMPLATES_TINY_CACHE:
        base = {"A": (1, 2, 3, 4, 5), "B": (6, 7, 8, 9, 10),
                "D": (11, 12, 13, 14, 15),
                "C": (16, 17, 18, 19, 20, 21)}
        if cap == "C":
            p, g, s, k, q, e = base["C"]
            _TEMPLATES_TINY_CACHE[cap] = {"p": [p], "g": [g], "s": [s], "k": [k], "q": [q], "e": [e]}
        else:
            p, m, s, q, t = base[cap]
            _TEMPLATES_TINY_CACHE[cap] = {"p": [p], "m": [m], "s": [s], "q": [q], "t": [t]}
    return _TEMPLATES_TINY_CACHE[cap]


# ------------------------------------------------------------------ run modes

import contextlib


@contextlib.contextmanager
def v4_local_output_root():
    """Redirect the V4 machinery's output root into THIS campaign (mission option C).

    Guarantees zero writes beneath the historical ARK019_GUARDIAN_V4 root; that
    directory is only ever READ (verified artifacts)."""
    old = V4.OUT
    local = OUT / "v4_reuse"
    local.mkdir(parents=True, exist_ok=True)
    V4.OUT = local
    try:
        yield local
    finally:
        V4.OUT = old


def _load_parent_model_state(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def historical_parent_readonly(seed: int):
    """READ-ONLY reuse of a verified historical V4 parent. Never writes."""
    rp = V4.OUT / "parents" / f"seed_{seed}" / "PARENT_V4.json"
    sp = V4.OUT / "parents" / f"seed_{seed}" / "PARENT_V4.pt"
    if not (rp.exists() and sp.exists()):
        return None
    try:
        r = json.loads(rp.read_text())
    except Exception:
        return None
    if r.get("status") != "QUALIFIED":
        return None
    if r.get("parent_model_sha256") != V3.state_hash(_load_parent_model_state(sp)):
        return None
    r["parent_model_sha256"] = r.get("parent_model_sha256") or V3.state_hash(
        _load_parent_model_state(sp))
    return r


def acquire_parent_local(seed, prep, bufs, tt, tasks, d, deadline):
    """Acquire A-parent without mutating the historical ARK019_GUARDIAN_V4 root.

    Returns (receipt, source_model_sha256)."""
    src_path, _src = V4.source_checkpoint(seed, prep)
    src_sha = hfile(src_path)
    hist = historical_parent_readonly(seed)
    if hist is not None:
        hist["reused_readonly"] = True
        return hist, src_sha
    with v4_local_output_root():
        r = V4.acquire_parent(seed, prep, bufs, tt["A"], tasks["sem"]["A"], d, deadline)
    if r.get("status") != "QUALIFIED":
        return r, src_sha
    lp = OUT / "v4_reuse" / "parents" / f"seed_{seed}" / "PARENT_V4.pt"
    r["parent_model_sha256"] = V3.state_hash(_load_parent_model_state(lp))
    (OUT / "parents" / f"seed_{seed}").mkdir(parents=True, exist_ok=True)
    savej(OUT / "parents" / f"seed_{seed}" / "PARENT.json", r)
    return r, src_sha


def load_parent_local(seed):
    lp = OUT / "v4_reuse" / "parents" / f"seed_{seed}" / "PARENT_V4.pt"
    if lp.exists():
        return torch.load(lp, map_location="cpu", weights_only=False)
    return V4.load_parent(seed)  # read-only load of a verified historical parent


def select_dose_local(parents, bufs, tt, tasks, d, deadline):
    """Dose selection with immutable local provenance (blocker C repair)."""
    local = OUT / "DOSE_SELECTION_IMPORTED.json"
    if local.exists():
        imported = json.loads(local.read_text())
        if imported.get("verification_status") == "verified":
            return {"status": "PASS", "selected_b_slots": imported["v4_local_selected_slots"],
                    "imported": True}
        raise RuntimeError("imported dose receipt exists but is not verified")
    hist = V4.OUT / "DOSE_SELECTION.json"
    if hist.exists():
        src = json.loads(hist.read_text())
        if src.get("status") != "PASS":
            raise RuntimeError("historical V4 dose selection exists but did not pass")
        slots = int(src["selected_b_slots"])
        receipt = {"schema": "arkenstone-ark020-v4-dose-import/v1",
                   "source_experiment": "ARK-019-V4", "source_file": str(hist),
                   "source_sha256": hfile(hist), "source_selected_slots": slots,
                   "verification_status": "verified", "v4_local_selected_slots": slots,
                   "import_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "scientific_equivalence": "identical B task construction and identical "
                                             "slot candidates; the dose is a task-slot "
                                             "constant shared across all arms, so "
                                             "campaign-local reuse is scientifically "
                                             "equivalent to re-selection"}
        savej(local, receipt)
        return {"status": "PASS", "selected_b_slots": slots, "imported": True}
    with v4_local_output_root():
        dose = V4.select_dose(parents, bufs, tt["B"], tasks["sem"]["B"], d, deadline)
    if dose.get("status") == "PASS":
        slots = int(dose["selected_b_slots"])
        receipt = {"schema": "arkenstone-ark020-v4-dose-import/v1",
                   "source_experiment": "ARK-020-V4-local-pilot",
                   "source_file": str(OUT / "v4_reuse" / "DOSE_SELECTION.json"),
                   "source_sha256": hfile(OUT / "v4_reuse" / "DOSE_SELECTION.json"),
                   "source_selected_slots": slots, "verification_status": "verified",
                   "v4_local_selected_slots": slots,
                   "import_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "scientific_equivalence": "locally selected with the V4 pilot rule"}
        savej(local, receipt)
    return dose


def run_all():
    if not torch.cuda.is_available():
        raise RuntimeError("ARK-020 V3 campaign requires Colab CUDA/T4; "
                           "use --mode smoke for the CPU integration smoke")
    setup()
    d = device()
    OUT.mkdir(parents=True, exist_ok=True)
    advisory_lock()
    started = time.monotonic()
    deadline = started + (C.SESSION_WALL_MINUTES - C.PACKAGING_RESERVE_MINUTES) * 60
    try:
        prep, tok, bufs, counts = V3.load_substrate()
        ids = select_tokens48(tok, counts)
        tasks = C.build_all_tasks(ids)
        tt = templates(tok)
        sources = []
        for seed in C.PARENT_SEEDS:
            p, x = V4.source_checkpoint(seed, prep)
            sources.append({"seed": seed, "sha256": hfile(p), "model_sha256": V3.state_hash(x["model"])})
        savej(OUT / "ENTRY_RECEIPT.json", {
            "schema": "arkenstone-ark020-v4-entry/v1",
            "science_sha256": V3.EXPECTED_SCIENCE_SHA,
            "tokenizer_sha256": prep["tokenizer_sha256"],
            "sources": sources, "selected_token_ids": ids,
            "skill_groups": tasks["groups"],
            "task_hashes": {k: hjson(tasks[k]) for k in ("A", "B", "C", "D")},
            "phase_order_seeds": C.PHASE_ORDER_SEEDS,
            "session_uuid": _session_uuid()})
        parents, parent_identities = {}, {}
        for seed in C.PARENT_SEEDS:
            r, src_sha = acquire_parent_local(seed, prep, bufs, tt, tasks, d, deadline)
            if r.get("status") != "QUALIFIED":
                result = {"schema": "arkenstone-ark020-v4-result/v1", "status": "BLOCKED_BEFORE_MAIN",
                          "verdict": "INCONCLUSIVE_PARENT_GATE_FAILED", "seed": seed, "authorized": False}
                savej(OUT / "ARK-020_V4_RESULT.json", result)
                result["bundle"] = package(OUT, final=True)
                return result
            parents[seed] = load_parent_local(seed)
            parent_identities[str(seed)] = {
                "source_model_sha256": src_sha,
                "acquired_parent_model_sha256": r["parent_model_sha256"],
                "parent_result_sha256": hfile(OUT / "parents" / f"seed_{seed}" / "PARENT.json")
                if (OUT / "parents" / f"seed_{seed}" / "PARENT.json").exists() else "reused-historical",
                "parent_science_gate": {"status": r.get("status"),
                                        "science_control_relative": r.get("science_control_relative")}}
        savej(OUT / "PARENT_IDENTITIES.json",
              {"schema": "arkenstone-ark020-v4-parent-identities/v1", "parents": parent_identities})
        dose = select_dose_local(parents, bufs, tt, tasks, d, deadline)
        if dose.get("status") != "PASS":
            result = {"schema": "arkenstone-ark020-v4-result/v1", "status": "BLOCKED_BEFORE_MAIN",
                      "verdict": "INCONCLUSIVE_NO_VIABLE_SKILL_B_DOSE",
                      "dose_selection": dose, "authorized": False}
            savej(OUT / "ARK-020_V4_RESULT.json", result)
            result["bundle"] = package(OUT, final=True)
            return result
        dose_b = int(dose["selected_b_slots"])
        smoke = exact_resume_smoke(parents[C.PARENT_SEEDS[0]], dose_b, bufs, tt, tasks, d)
        runtime = calibrate(parents[C.PARENT_SEEDS[0]], dose_b, bufs, tt, tasks, d)
        savej(OUT / "PREEXECUTION_GATE.json", {
            "schema": "arkenstone-ark020-v4-preexec/v1", "status": "PASS",
            "dose_selection": {"selected_b_slots": dose_b},
            "exact_resume": smoke, "runtime": runtime})
        cap_by_set = {}
        for ps in C.PARENT_SEEDS:
            for bs in C.PHASE_ORDER_SEEDS["B"]:
                cap_by_set[f"{ps}_{bs}"] = cap_calibration(ps, bs, parents[ps], bufs, tt, tasks, d, deadline)
        for ps in C.PARENT_SEEDS:
            for bs in C.PHASE_ORDER_SEEDS["B"]:
                cap16 = float(cap_by_set[f"{ps}_{bs}"]["cap16x"])
                for arm in C.ARMS:
                    run_set_arm(ps, bs, arm, dose_b, parents[ps], cap16, bufs, tt, tasks, d, deadline)
        dec = C.decide(group_for_decide(collect_arm_results()))
        result = {"schema": "arkenstone-ark020-v4-result/v1", "status": "COMPLETE",
                  "decision": dec, "dose_b": dose_b,
                  "wall_seconds_last_session": time.monotonic() - started,
                  "claim_boundary": "real-text proxy multi-skill continual controller only"}
        savej(OUT / "ARK-020_V4_RESULT.json", result)
        result["bundle"] = package(OUT, final=True)
        return result
    except SessionTimebox as e:
        state = {"schema": "arkenstone-ark020-v4-session/v1", "status": "PARTIAL_SESSION",
                 "message": str(e), "wall_seconds": time.monotonic() - started,
                 "instruction": "rerun the same frozen notebook; exact Drive checkpoints resume work"}
        savej(OUT / "SESSION_STATE.json", state)
        state["bundle"] = package(OUT, final=False)
        return state
    finally:
        try:
            (OUT / "CAMPAIGN_LOCK.json").unlink(missing_ok=True)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "smoke", "scan"], default="all")
    ap.add_argument("--smoke-root", default="/content/ark020_v4_smoke")
    ap.add_argument("--drive-ok", default="True")
    args = ap.parse_args()
    if args.mode == "scan":
        info = resume_scan(drive_ok=str(args.drive_ok).lower() == "true")
        print("@@SCAN_JSON@@" + json.dumps(info))
        print(json.dumps(info, indent=2))
        return 0
    if args.mode == "smoke":
        r = cpu_integration_smoke(Path(args.smoke_root))
        print(json.dumps(r, indent=2))
        return 0 if r["status"] == "PASS" else 1
    try:
        r = run_all()
        print("ARK-020 V3 STATUS", r.get("status"))
        print("VERDICT", r.get("decision", {}).get("verdict", r.get("verdict")))
        print("BUNDLE", r.get("bundle"))
        return 0
    except Exception as e:
        OUT.mkdir(parents=True, exist_ok=True)
        savej(OUT / "ARK-020_V4_FAILURE.json", {"schema": "arkenstone-ark020-v4-failure/v1",
                                                "status": "FAILED", "exception": type(e).__name__,
                                                "message": str(e), "traceback": traceback.format_exc()})
        traceback.print_exc()
        try:
            package(OUT, final=False)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
