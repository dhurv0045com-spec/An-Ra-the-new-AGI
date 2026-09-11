"""ARK-020 campaign runner — multi-skill continual cognition generalization.

Reuses the audited ARK-018/019 substrate and V4 machinery (parents, dose stage,
mixed updates, exact-resume checkpoints, runtime calibration, packaging), extended
to four skills, a capability registry, risk-allocated replay, and reactive vs
predictive Guardian controllers. All controller decisions flow through
ark020_core on CONTROL metrics only. SEALED data never influences any decision.
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

import ark020_core as C  # noqa: E402
import run_ark019_v4 as V4  # noqa: E402
import run_ark019_v3 as V3  # noqa: E402
from ark018_v3_common import Ark018GPT, eval_buffer, autocast_ctx, make_scaler, model_state_hash  # noqa: E402

OUT = Path("/content/drive/MyDrive/genisis-arkenstone/ARK020_CONTINUAL_V1")
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
    if not torch.cuda.is_available():
        raise RuntimeError("ARK-020 requires Colab CUDA/T4")
    return torch.device("cuda")


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
    torch.cuda.set_rng_state_all(s["cuda_rng"])
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
        f"ARK-020 token gate failed: fewer than {EXPECTED_TOKENS} eligible single-token words "
        f"at thresholds {TOKEN_THRESHOLDS}; campaign refuses to run with overlapping skill vocabularies")


# --------------------------------------------------------------- task rendering

def templates(tok) -> dict[str, dict[str, list[int]]]:
    return {
        "A": {"p": tok.encode("Facts: ").ids, "m": tok.encode(" means ").ids,
              "s": tok.encode("; ").ids, "q": tok.encode("Query: ").ids, "t": tok.encode(" means").ids},
        "B": {"p": tok.encode("Map: ").ids, "m": tok.encode(" -> ").ids,
              "s": tok.encode(" | ").ids, "q": tok.encode("Requested ").ids, "t": tok.encode(" =>").ids},
        "D": {"p": tok.encode("Owners: ").ids, "m": tok.encode(" holds ").ids,
              "s": tok.encode("; ").ids, "q": tok.encode("Who holds ").ids, "t": tok.encode(" ?").ids},
        "C": {"p": tok.encode("Chain: ").ids, "m": tok.encode(" next").ids,
              "s": tok.encode("; ").ids},
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


def _finish_metrics(rec: dict) -> dict:
    rec["robust_min"] = C.robust_min(rec)
    rec["qualified"] = C.qualified(rec)
    rec["healthy"] = C.healthy(rec)
    return rec


@torch.no_grad()
def cap_control_metrics(m, cap, t, tasks, d) -> dict:
    """CONTROL metrics per capability (never SEALED).

    For C, 'order_only' means one distractor pair in context and 'distractor'
    means three distractor pairs (documented mapping; thresholds unchanged).
    """
    if cap in ("A", "B", "D"):
        sem = tasks["sem"][cap]["main_control"]
        rec = {"canonical": mode_acc(m, cap, t, sem, d, "canonical"),
               "order_only": mode_acc(m, cap, t, sem, d, "reversed"),
               "query_order": mode_acc(m, cap, t, sem, d, "query_order")}
    else:
        sem = tasks["sem"]["C"]["control"]
        rec = {"canonical": mode_acc(m, cap, t, sem, d, "canonical"),
               "order_only": mode_acc(m, cap, t, sem, d, "reversed"),
               "distractor": mode_acc(m, cap, t, sem, d, "query_order")}
    return _finish_metrics(rec)


@torch.no_grad()
def cap_validation_metrics(m, cap, t, tasks, d) -> dict:
    """VALIDATION metrics — used for confirmation gates (never SEALED)."""
    if cap in ("A", "B", "D"):
        sem = tasks["sem"][cap]["validation"]
        rec = {"canonical": mode_acc(m, cap, t, sem, d, "canonical"),
               "order_only": mode_acc(m, cap, t, sem, d, "reversed"),
               "query_order": mode_acc(m, cap, t, sem, d, "query_order")}
    else:
        sem = tasks["sem"]["C"]["validation"]
        rec = {"canonical": mode_acc(m, cap, t, sem, d, "canonical"),
               "order_only": mode_acc(m, cap, t, sem, d, "reversed"),
               "distractor": mode_acc(m, cap, t, sem, d, "query_order")}
    return _finish_metrics(rec)


@torch.no_grad()
def cap_sealed_metrics(m, cap, t, tasks, d) -> dict:
    """SEALED metrics — measurement only, never fed to any controller."""
    if cap in ("A", "B", "D"):
        sem = tasks["sem"][cap]["sealed"]
        rec = {"canonical": mode_acc(m, cap, t, sem, d, "canonical"),
               "order_only": mode_acc(m, cap, t, sem, d, "reversed"),
               "query_order": mode_acc(m, cap, t, sem, d, "query_order")}
    else:
        sem = tasks["sem"]["C"]["sealed"]
        rec = {"canonical": mode_acc(m, cap, t, sem, d, "canonical"),
               "order_only": mode_acc(m, cap, t, sem, d, "reversed"),
               "distractor": mode_acc(m, cap, t, sem, d, "query_order")}
    rec["robust_min"] = C.robust_min(rec)
    rec["qualified"] = C.qualified(rec)
    return rec


def science(m, bufs, d, sealed=True):
    out = {"control": eval_buffer(m, bufs["control"], d, 72001, sequences=24)}
    if sealed:
        out["sealed"] = eval_buffer(m, bufs["sealed"], d, 72002, sequences=24)
    return out


def real_batch(buf, n, seed, step, d, tag="main"):
    xs, ys, starts = [], [], []
    need = V3.CONTEXT + 1
    for slot in range(n):
        z = int.from_bytes(hashlib.sha256(
            f"ark020-real:{tag}:{seed}:{step}:{slot}".encode()).digest()[:8], "big") % (len(buf) - need + 1)
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
    """replay: list of (cap_id, template, semantics) — risk-ordered old capabilities."""
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


# ------------------------------------------------------------------ campaign

PHASES = ("B", "C", "D")


def global_step(phase_idx: int, step: int) -> int:
    offs = [0, C.PHASE_UPDATES["B"], C.PHASE_UPDATES["B"] + C.PHASE_UPDATES["C"]]
    return offs[phase_idx] + step


def arm_paths(ps, bs, arm):
    p = OUT / "matched_sets" / f"p{ps}_b{bs}" / arm
    return p, p / "RESULT.json", p / "RESUME.pt", p / "PARTIAL.json"


def run_set_arm(ps, bs, arm, dose_b, parent_state, cap16, bufs, tt, tasks, d, deadline):
    ad, rp, cp, pp = arm_paths(ps, bs, arm)
    ad.mkdir(parents=True, exist_ok=True)
    parent_sha = V3.state_hash(parent_state["model"])
    task_hash = hjson({k: hjson(tasks[k]) for k in ("A", "B", "C", "D")})
    expected = {"schema": "arkenstone-ark020-arm-ckpt/v1", "parent_seed": ps, "order_seed": bs,
                "arm": arm, "dose_b": int(dose_b), "parent_sha": parent_sha,
                "task_hash": task_hash, "cap16x": float(cap16)}
    if rp.exists():
        r = json.loads(rp.read_text())
        if r.get("status") == "COMPLETE" and r.get("parent_sha") == parent_sha:
            return r
        raise RuntimeError(f"incompatible ARK-020 arm result {rp}")
    if cp.exists():
        x, m, o, sc = load_checkpoint(cp, expected, d)
        start_phase = int(x["phase_idx"])
        start_step = int(x["phase_step"])
        reg, ctrl = x["registry"], x["controller"]
        cnt = x["counters"]
        b_streaks = x["b_streaks"]
        control_trace = C.dedupe_trajectory(x.get("control_trace", []))
        measure_trace = C.dedupe_trajectory(x.get("measure_trace", []))
        phase_confirm = x["phase_confirm"]
    else:
        m, o, sc = restore(parent_state, d, C.HIGH_LR)
        start_phase, start_step = 0, 0
        reg, ctrl = C.init_registry(), C.initial_controller(arm)
        C.register_capability(reg, "A", 0)
        cnt = {"replay_slots": 0, "real_slots": 0, "task_slots": 0, "capped_steps": 0,
               "protection_updates": 0, "projected_path": 0.0, "replay_slots_by_cap": {},
               "counters_by_phase": {p: {"replay_slots": 0, "protection_updates": 0,
                                         "updates": C.PHASE_UPDATES[p]} for p in PHASES}}
        b_streaks = {"B": 0, "C": 0, "D": 0}
        control_trace, measure_trace = [], []
        phase_confirm = {"B": None, "C": None, "D": None}
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
                   "registry": reg, "controller": ctrl, "counters": cnt, "phase_confirm": phase_confirm,
                   "control_trace": control_trace, "measure_trace": measure_trace,
                   "b_streaks": b_streaks}
        save_checkpoint(cp, payload, m, o, sc)
        savej(pp, {"schema": "arkenstone-ark020-partial/v1", "status": "PARTIAL_SESSION",
                   "parent_seed": ps, "order_seed": bs, "arm": arm, "phase": phase,
                   "step": step - 1, "counters": cnt})
        del m, o
        torch.cuda.empty_cache()
        raise SessionTimebox(f"arm {ps}/{bs}/{arm} paused at {phase} step {step-1}")

    for phase_idx in range(start_phase, len(PHASES)):
        phase = PHASES[phase_idx]
        slots = phase_slots[phase]
        phase_sem = tasks["sem"][phase]["train"]
        phase_t = tt[phase]
        start = start_step + 1 if phase_idx == start_phase else 1
        for step in range(start, C.PHASE_UPDATES[phase] + 1):
            if time.monotonic() >= deadline:
                timebox(phase_idx, phase, step)
            gstep = global_step(phase_idx, step)
            qualified_old = [c for c in ("A", "B", "C", "D")
                             if c in reg["capabilities"] and c != phase]
            if arm == "PLASTIC_HIGH":
                replay, capv = [], None
            else:
                replay, capv = C.treatment(arm, ctrl, gstep, reg, qualified_old, cap16)
            cnt["protection_updates"] += int(bool(replay or capv is not None))
            cnt["counters_by_phase"][phase]["protection_updates"] += int(bool(replay or capv is not None))
            rec = mixed_update(m, o, sc, bufs["train"], phase, phase_t, phase_sem, bs, step,
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
                for cid in qualified_old:
                    C.observe_capability(reg, cid, gstep, metrics[cid])
                C.update_controller(arm, ctrl, gstep, reg, qualified_old)
                if b_streaks[phase] is None:
                    b_streaks[phase] = 0
                b_streaks[phase] = b_streaks[phase] + 1 if C.qualified(metrics[phase]) else 0
                if b_streaks[phase] >= C.CONFIRM_STREAK and phase_confirm[phase] is None:
                    val = cap_validation_metrics(m, phase, tt, tasks, d)
                    if C.qualified(val):
                        phase_confirm[phase] = gstep
                        C.register_capability(reg, phase, gstep)
                        C.set_post_phase_floor(ctrl, [phase], gstep)
                        b_streaks[phase] = 0
                control_trace.append({"step": gstep, "phase": phase, "metrics": metrics,
                                      "controller": copy.deepcopy(ctrl)})
                print(f"ARK020 [{ps}/{bs}/{arm}/{phase}] {step}/{C.PHASE_UPDATES[phase]} "
                      f"A={metrics['A']['robust_min']:.3f} "
                      f"{phase}={metrics[phase]['robust_min']:.3f} state={ctrl['states']}", flush=True)
            if step % C.MEASURE_EVERY == 0:
                sealed = {c: cap_sealed_metrics(m, c, tt, tasks, d) for c in caps_now}
                sci = science(m, bufs, d, sealed=True)
                disp = displacement(m, parent_model)
                diag = {}
                if qualified_old:
                    tri = C.deterministic_indices(bs, step, 8, len(phase_sem), "diag-task")
                    gt = projected_grad(m, names, d,
                                        rows_rs=C.task_rows(phase, phase_t, phase_sem, tri,
                                                          "augmented", bs, step))
                    for cid in qualified_old:
                        csem = tasks["sem"][cid]["train"]
                        cri = C.deterministic_indices(bs, step, 4, len(csem), f"diag-{CAP_INDEX[cid]}")
                        gc = projected_grad(m, names, d,
                                            rows_rs=C.task_rows(cid, tt[cid], csem, cri,
                                                              "nonidentity", bs + 31, step))
                        diag[cid] = {"task_vs_old_grad_cosine": cosine(gt, gc)}
                measure_trace.append({"step": gstep, "phase": phase, "sealed": sealed,
                                      "science": sci, "full_displacement": disp,
                                      "last_projected_delta": rec["projected_delta"],
                                      "diagnostics": diag})
                savej(pp, {"schema": "arkenstone-ark020-partial/v1", "status": "RUNNING",
                           "parent_seed": ps, "order_seed": bs, "arm": arm, "phase": phase,
                           "step": step, "counters": cnt, "measure_trace": measure_trace[-3:]})
            if step % C.CHECKPOINT_EVERY == 0:
                payload = {**expected, "phase_idx": phase_idx, "phase": phase, "phase_step": step,
                           "registry": reg, "controller": ctrl, "counters": cnt,
                           "phase_confirm": phase_confirm, "control_trace": control_trace,
                           "measure_trace": measure_trace, "b_streaks": b_streaks}
                save_checkpoint(cp, payload, m, o, sc)
        if arm in C.GUARDIAN_ARMS:
            C.set_post_phase_floor(ctrl, list(reg["capabilities"].keys()), gstep)
    # finalize
    control_trace = C.dedupe_trajectory(control_trace)
    measure_trace = C.dedupe_trajectory(measure_trace)
    f = measure_trace[-1]
    failure_steps: dict[str, list[int]] = {}
    recovery: dict[str, bool] = {}
    for row in control_trace:
        for cid, mm in row["metrics"].items():
            if not C.qualified(mm):
                failure_steps.setdefault(cid, []).append(int(row["step"]))
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
    r = {"schema": "arkenstone-ark020-arm/v1", "status": "COMPLETE", "parent_seed": ps,
         "order_seed": bs, "arm": arm, "dose_b": int(dose_b), "cap16x": float(cap16),
         "parent_sha": parent_sha, "phase_confirmation": phase_confirm,
         "final_sealed": f["sealed"], "sealed_area": sealed_area,
         "control_failure_steps": failure_steps, "recovery": recovery,
         "final_science_sealed_nll": float(f["science"]["sealed"]["nll"]),
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


def collect_arm_results():
    out = {}
    for p in OUT.glob("matched_sets/p*_b*/*/RESULT.json"):
        r = json.loads(p.read_text())
        out[f"{r['parent_seed']}:{r['order_seed']}:{r['arm']}"] = r
    return out


def group_for_decide(collected):
    grouped = {a: [] for a in C.ARMS}
    for r in collected.values():
        if r.get("arm") in grouped:
            grouped[r["arm"]].append(r)
    return grouped


def exact_resume_smoke(parent_state, dose_b, bufs, tt, tasks, d):
    p = OUT / "EXACT_RESUME_SMOKE_V1.json"
    if p.exists():
        return json.loads(p.read_text())

    def advance(m, o, sc, start, stop):
        tr = []
        names = pnames(m)
        for step in range(start, stop + 1):
            rep = [("A", tt["A"], tasks["sem"]["A"]["train"])] if step % 2 == 0 else []
            r = mixed_update(m, o, sc, bufs["train"], "B", tt["B"], tasks["sem"]["B"]["train"],
                             439991, step, dose_b, rep, None, d, names, tag="resume-smoke")
            tr.append((step, r["replay_slots"], r["real_starts_sha256"], round(r["loss"], 10)))
        return tr

    a, ao, asc = restore(parent_state, d)
    t1 = advance(a, ao, asc, 1, 10)
    h1 = (model_state_hash(a), optimizer_hash(ao))
    b, bo, bsc = restore(parent_state, d)
    t2a = advance(b, bo, bsc, 1, 5)
    ss = snapshot(b, bo, bsc)
    del b, bo
    torch.cuda.empty_cache()
    c2, co, csc = restore(ss, d)
    t2b = advance(c2, co, csc, 6, 10)
    h2 = (model_state_hash(c2), optimizer_hash(co))
    ok = h1 == h2 and t1 == t2a + t2b
    r = {"schema": "arkenstone-ark020-exact-resume/v1", "status": "PASS" if ok else "FAIL",
         "model_optimizer_identical": h1 == h2, "telemetry_identical": t1 == t2a + t2b}
    savej(p, r)
    if not ok:
        raise RuntimeError("ARK-020 exact-resume smoke failed")
    del a, ao, c2, co
    torch.cuda.empty_cache()
    return r


def calibrate(parent_state, dose_b, bufs, tt, tasks, d):
    p = OUT / "RUNTIME_CALIBRATION_V1.json"
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
        torch.cuda.synchronize()
        st = time.monotonic()
        n = 5
        for step in range(3, 3 + n):
            mixed_update(m, o, sc, bufs["train"], "B", tt["B"], tasks["sem"]["B"]["train"],
                         449991, step, dose_b, rep, capv, d, names, tag="runtime",
                         full_delta=label == "CAP")
        torch.cuda.synchronize()
        rec[label] = {"update_seconds": (time.monotonic() - st) / n}
        del m, o
        torch.cuda.empty_cache()
    m, o, sc = restore(parent_state, d)
    torch.cuda.synchronize()
    st = time.monotonic()
    cap_control_metrics(m, "A", tt, tasks, d)
    cap_control_metrics(m, "B", tt, tasks, d)
    control_sec = time.monotonic() - st
    torch.cuda.synchronize()
    st = time.monotonic()
    for cap in ("A", "B", "C", "D"):
        cap_sealed_metrics(m, cap, tt, tasks, d)
    science(m, bufs, d, sealed=True)
    measure_sec = time.monotonic() - st
    del m, o
    torch.cuda.empty_cache()
    worst = max(v["update_seconds"] for v in rec.values())
    n_sets = len(C.PARENT_SEEDS) * 2
    main_updates = n_sets * len(C.ARMS) * C.CONTINUATION_HORIZON
    control_events = n_sets * len(C.ARMS) * (C.CONTINUATION_HORIZON // C.CONTROL_EVERY)
    measure_events = n_sets * len(C.ARMS) * (C.CONTINUATION_HORIZON // C.MEASURE_EVERY)
    est = (main_updates * worst + control_events * control_sec + measure_events * measure_sec) * C.RUNTIME_SAFETY_FACTOR
    sessions = max(1, math.ceil(est / ((C.SESSION_WALL_MINUTES - C.PACKAGING_RESERVE_MINUTES) * 60)))
    r = {"schema": "arkenstone-ark020-runtime/v1", "status": "PASS", "per_update_seconds": rec,
         "control_battery_seconds": control_sec, "measure_battery_seconds": measure_sec,
         "estimated_main_seconds": est, "estimated_main_sessions": sessions,
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
    r = {"schema": "arkenstone-ark020-capcal/v1", "parent_seed": ps, "order_seed": bs,
         "shadow_lr": C.LOW_LR, "steps": C.CAP_SHADOW_STEPS, "median_low_delta": med,
         "cap16x": C.CAP_MULTIPLIER * med, "parent_model_sha256": V3.state_hash(parent["model"])}
    savej(p, r)
    del m, o
    torch.cuda.empty_cache()
    return r


def advisory_lock():
    lock = OUT / "CAMPAIGN_LOCK.txt"
    OUT.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        age_h = (time.time() - lock.stat().st_mtime) / 3600.0
        if age_h < 6:
            raise RuntimeError(
                f"another session appears active on {OUT} (lock age {age_h:.1f}h). "
                "Only one concurrent Colab session may write the same campaign root.")
    lock.write_text(f"locked {time.time()}\n")
    return lock


def package(final=False):
    manifests = {}
    for p in sorted(OUT.rglob("*.json")):
        if p.name == "ZIP_MANIFEST_V1.json":
            continue
        manifests[str(p.relative_to(OUT))] = hfile(p)
    savej(OUT / "ZIP_MANIFEST_V1.json", {"schema": "arkenstone-ark020-manifest/v1", "members": manifests})
    name = "ARKENSTONE_ARK020_CONTINUAL_RESULTS.zip" if final else "ARKENSTONE_ARK020_CONTINUAL_PARTIAL.zip"
    z = OUT / name
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for p in sorted(OUT.rglob("*.json")):
            f.write(p, str(p.relative_to(OUT)))
    (OUT / (name + ".sha256")).write_text(hfile(z) + "  " + name + "\n")
    return {"path": str(z), "sha256": hfile(z)}


def run_all():
    setup()
    d = device()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = advisory_lock()
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
        savej(OUT / "ENTRY_RECEIPT_V1.json", {
            "schema": "arkenstone-ark020-entry/v1",
            "science_sha256": V3.EXPECTED_SCIENCE_SHA,
            "tokenizer_sha256": prep["tokenizer_sha256"],
            "sources": sources, "selected_token_ids": ids,
            "skill_groups": tasks["groups"],
            "task_hashes": {k: hjson(tasks[k]) for k in ("A", "B", "C", "D")},
            "v4_status_note": "ARK-019 V4 scientific result was NOT present in the repository "
                              "at ARK-020 design time; ARK-020 stands on R2 + V3.1 evidence only"})
        parents = {}
        for seed in C.PARENT_SEEDS:
            r = V4.acquire_parent(seed, prep, bufs, {"A": tt["A"]}, tasks["A"], d, deadline)
            if r.get("status") != "QUALIFIED":
                result = {"schema": "arkenstone-ark020-result/v1", "status": "BLOCKED_BEFORE_MAIN",
                          "verdict": "INCONCLUSIVE_PARENT_GATE_FAILED", "seed": seed, "authorized": False}
                savej(OUT / "ARK-020_RESULT.json", result)
                result["bundle"] = package(final=True)
                return result
            parents[seed] = V4.load_parent(seed)
        dose_path = V4.OUT / "DOSE_SELECTION.json"
        if dose_path.exists():
            dose = json.loads(dose_path.read_text())
            if dose.get("status") != "PASS":
                raise RuntimeError("V4 dose selection exists but did not pass")
        else:
            dose = V4.select_dose(parents, bufs, {"B": tt["B"]}, tasks["B"], d, deadline)
        if dose.get("status") != "PASS":
            result = {"schema": "arkenstone-ark020-result/v1", "status": "BLOCKED_BEFORE_MAIN",
                      "verdict": "INCONCLUSIVE_NO_VIABLE_SKILL_B_DOSE",
                      "dose_selection": dose, "authorized": False}
            savej(OUT / "ARK-020_RESULT.json", result)
            result["bundle"] = package(final=True)
            return result
        dose_b = int(dose["selected_b_slots"])
        smoke = exact_resume_smoke(parents[C.PARENT_SEEDS[0]], dose_b, bufs, tt, tasks, d)
        runtime = calibrate(parents[C.PARENT_SEEDS[0]], dose_b, bufs, tt, tasks, d)
        savej(OUT / "PREEXECUTION_GATE_V1.json", {
            "schema": "arkenstone-ark020-preexec/v1", "status": "PASS",
            "dose_selection": {"selected_b_slots": dose_b},
            "exact_resume": smoke, "runtime": runtime})
        cap_by_set = {}
        for ps in C.PARENT_SEEDS:
            for oi in (0, 1):
                bs = C.PHASE_ORDER_SEEDS["B"][oi]
                cap_by_set[f"{ps}_{bs}"] = cap_calibration(ps, bs, parents[ps], bufs, tt, tasks, d, deadline)
        for ps in C.PARENT_SEEDS:
            for oi in (0, 1):
                bs = C.PHASE_ORDER_SEEDS["B"][oi]
                cap16 = float(cap_by_set[f"{ps}_{bs}"]["cap16x"])
                for arm in C.ARMS:
                    run_set_arm(ps, bs, arm, dose_b, parents[ps], cap16, bufs, tt, tasks, d, deadline)
        dec = C.decide(group_for_decide(collect_arm_results()))
        result = {"schema": "arkenstone-ark020-result/v1", "status": "COMPLETE",
                  "decision": dec, "dose_b": dose_b,
                  "wall_seconds_last_session": time.monotonic() - started,
                  "claim_boundary": "real-text proxy multi-skill continual controller only"}
        savej(OUT / "ARK-020_RESULT.json", result)
        result["bundle"] = package(final=True)
        return result
    except SessionTimebox as e:
        state = {"schema": "arkenstone-ark020-session/v1", "status": "PARTIAL_SESSION",
                 "message": str(e), "wall_seconds": time.monotonic() - started,
                 "instruction": "rerun the same frozen notebook; exact Drive checkpoints resume work"}
        savej(OUT / "SESSION_STATE_V1.json", state)
        state["bundle"] = package(final=False)
        return state
    finally:
        try:
            lock.unlink(missing_ok=True)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all"], default="all")
    _ = ap.parse_args()
    try:
        r = run_all()
        print("ARK-020 STATUS", r.get("status"))
        print("VERDICT", r.get("decision", {}).get("verdict", r.get("verdict")))
        print("BUNDLE", r.get("bundle"))
        return 0
    except Exception as e:
        OUT.mkdir(parents=True, exist_ok=True)
        savej(OUT / "ARK-020_FAILURE.json", {"schema": "arkenstone-ark020-failure/v1",
                                             "status": "FAILED", "exception": type(e).__name__,
                                             "message": str(e), "traceback": traceback.format_exc()})
        traceback.print_exc()
        try:
            package(final=False)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
