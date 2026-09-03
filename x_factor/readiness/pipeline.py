"""Canonical readiness v2 executable pipeline (Mission 1).

ONE clear v2 result object computed from REAL measured inputs:

  canaries (P0-P4) -> canary_rule()
  ladder raw+oracle per rung -> classify_capability() + assess_identifiability()
  frontier ranks -> check_frontier()
  legal answer-blind arms (E5dup/sham, E7sel) on candidate rung -> legal_headroom()
  per-task signatures -> response_diversity()
  discordants -> power_gate()
  replication artifact binding -> replication_ok (never assumed)
  LP rank vs chance -> qv_lite_vs_chance
  -> decide_readiness() + x0_permitted() + x1_permitted()

No schema renames over v1 logic. No synthetic placeholders. No hardcoded PASS.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

import sys as _sys

_XF = Path(__file__).resolve().parents[1]
if str(_XF) not in _sys.path:
    _sys.path.insert(0, str(_XF))

from observed import make_visible  # noqa: E402
from provenance import sha256_file  # noqa: E402

from readiness.canaries import CANARIES, canary_rule, gen_canary  # noqa: E402
from readiness.frontier import check_frontier  # noqa: E402
from readiness.gate import _greedy, _lp, _strict, oracle_prompt  # noqa: E402 (neural helpers only)
from readiness.identifiability import required_n_mcnemar  # noqa: E402
from readiness.ladder import RUNGS, gen_tasks  # noqa: E402
from readiness.readiness_v2 import (  # noqa: E402
    decide_readiness,
    legal_headroom,
    power_gate,
    response_diversity,
    x0_permitted,
    x1_permitted,
)
from readiness.status import (  # noqa: E402
    assess_identifiability,
    chance_report,
    classify_capability,
    wilson,
)

_ENTITY_PATS = (re.compile(r"ref of\s+([A-Za-z]+)", re.IGNORECASE),
                re.compile(r"belongs to the\s+([A-Za-z]+)", re.IGNORECASE),
                re.compile(r"held by the\s+([A-Za-z]+)", re.IGNORECASE))


def _parsed_entity(query: str) -> str:
    for pat in _ENTITY_PATS:
        m = pat.search(query)
        if m:
            return m.group(1).lower()
    return ""


def _matched_line(block: str, ent: str) -> str:
    for line in block.splitlines():
        if ent and ent in re.sub(r"[^a-z0-9]", "", line.lower()):
            return line
    return ""


def e5_dup(vt) -> str:  # VisibleTask only (answer-blind)
    line = _matched_line(vt.context, _parsed_entity(vt.query))
    return f"{vt.context}\n{line}\n{vt.query}\nAnswer:"


def e5_sham(vt) -> str:  # VisibleTask only (matched control)
    import hashlib
    import random

    ent = _parsed_entity(vt.query)
    mine = _matched_line(vt.context, ent)
    others = [l for l in vt.context.splitlines() if l.strip() and l != mine]
    seed = int(hashlib.sha256(f"{vt.task_id}|sham".encode()).hexdigest()[:12], 16)
    pick = random.Random(seed).choice(others) if others else mine
    return f"{vt.context}\n{pick}\n{vt.query}\nAnswer:"


def e7_sel(vt) -> str:  # VisibleTask only (answer-blind)
    line = _matched_line(vt.context, _parsed_entity(vt.query))
    return f"{line}\n{vt.query}\nAnswer:"


def run_canaries(model, tok, device, seed: int, n: int = 12) -> dict:
    per = {}
    for cid in CANARIES:
        k = 0
        for i in range(n):
            t = gen_canary(cid, seed, i)
            out = _greedy(model, tok, t["prompt"], device)
            k += _strict(out, t["gold"])
        per[cid] = {"k": k, "n": n, "rate": round(k / n, 4),
                    "wilson95": [round(v, 4) for v in wilson(k, n)]}
    rule = canary_rule({cid: {"k": v["k"], "n": v["n"]} for cid, v in per.items()})
    ok = None if rule["verdict"] == "MISSING_METRICS" else (rule["verdict"] == "CANARIES_OK")
    return {"per_canary": per, "rule": rule, "canary_ok": ok}


def run_legal_subset(model, tok, device, tasks: list[dict]) -> dict:
    """Real answer-blind arms on a frozen task subset. Gold = scoring only."""
    sigs, arms = [], {"e5dup": [], "e5sham": [], "e7sel": [], "raw": []}
    for t in tasks:
        vt = make_visible(t["id"], t["block"], t["query"], list(t["codes"]))
        r0 = _strict(_greedy(model, tok, t["prompt"], device), t["gold"])
        d = _strict(_greedy(model, tok, e5_dup(vt), device), t["gold"])
        s = _strict(_greedy(model, tok, e5_sham(vt), device), t["gold"])
        e = _strict(_greedy(model, tok, e7_sel(vt), device), t["gold"])
        arms["raw"].append(r0)
        arms["e5dup"].append(d)
        arms["e5sham"].append(s)
        arms["e7sel"].append(e)
        sigs.append((d, s, e))
    rates = {k: sum(v) / len(v) for k, v in arms.items()}
    disc = {}
    for a in ("e5dup", "e5sham", "e7sel"):
        disc[a] = sum(1 for x, y in zip(arms[a], arms["raw"]) if x == 1 and y == 0)
    # LP rank vs chance on same subset (single-query rank, no cross-query norm)
    rank_ok = []
    for t in tasks:
        row = [_lp(model, tok, t["prompt"], c, device) for c in t["codes"]]
        rank_ok.append(1 if int(np.argmax(row)) == t["codes"].index(t["gold"]) else 0)
    chance = 1.0 / len(tasks[0]["codes"])
    return {"n": len(tasks), "rates": {k: round(v, 4) for k, v in rates.items()},
            "discord_vs_raw": disc, "signatures": [list(s) for s in sigs],
            "rank1": round(sum(rank_ok) / len(rank_ok), 4),
            "rank1_vs_chance": round(sum(rank_ok) / len(rank_ok) - chance, 4),
            "chance": chance}


def check_replication(ref: dict | None, param_sha: str) -> dict:
    """Replication is real only with a bound artifact on the same checkpoint."""
    if ref is None:
        return {"replication_ok": None, "note": "no replication evidence supplied"}
    import json

    p = Path(ref.get("artifact", ""))
    if not p.exists():
        return {"replication_ok": False, "note": f"artifact missing: {p}"}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        return {"replication_ok": False, "note": f"artifact unreadable: {e}"}
    prov = doc.get("provenance", {})
    same_ckpt = prov.get("parameter_sha256") == param_sha
    return {"replication_ok": bool(same_ckpt),
            "artifact": str(p), "artifact_sha256": sha256_file(str(p)),
            "same_checkpoint": same_ckpt,
            "note": "bound artifact, same checkpoint" if same_ckpt else
                    "artifact checkpoint mismatch: not a replication"}


def run_readiness_v2(model, tok, payload, *, checkpoint: str, param_sha: str,
                     ckpt_sha: str, tok_sha: str, exp_sha: str, commit: str,
                     seed: int, rungs: tuple, n_per_rung: int, device: str,
                     stage: str, protocol_sha: str | None = None,
                     replication_ref: dict | None = None,
                     budget_n: int | None = None,
                     subset_n: int = 8) -> dict:
    t0 = time.strftime("%Y-%m-%d %H:%M:%S")
    can = run_canaries(model, tok, device, seed)
    rung_rows, frontier_in = {}, []
    for rung in rungs:
        tasks = gen_tasks(rung, seed, n_per_rung)
        raw_ok = [_strict(_greedy(model, tok, t["prompt"], device), t["gold"]) for t in tasks]
        orb_ok = [_strict(_greedy(model, tok, oracle_prompt(t), device), t["gold"]) for t in tasks]
        fails = sum(1 for r in raw_ok if r == 0)
        rep = sum(1 for r, o in zip(raw_ok, orb_ok) if r == 0 and o == 1)
        disc = rep  # oracle-only repairs among failures (paired by task)
        k = len(tasks[0]["codes"])
        orb_rate = sum(orb_ok) / len(tasks)
        ident = assess_identifiability(len(tasks), fails, disc, orb_rate, chance=1.0 / k)
        cap = classify_capability(len(tasks), sum(raw_ok),
                                  orb_rate, None, 1.0 / k,
                                  can["canary_ok"])
        rung_rows[rung] = {"n": len(tasks), "k": k,
                           "raw": chance_report(sum(raw_ok), len(tasks), 1.0 / k),
                           "oracle_rate": round(sum(orb_ok) / len(tasks), 4),
                           "n_failures": fails, "n_oracle_repairs": rep,
                           "capability": cap, "identifiability": ident}
        frontier_in.append({"rung": rung, "raw_k": sum(raw_ok), "n": len(tasks)})
    frontier = check_frontier(frontier_in)
    # candidate rung: first IDENTIFIABLE/MARGINAL rung with oracle headroom.
    # Suspended entirely when primitive canaries fail: a canary-failed
    # regime must not be citable as a "candidate pocket".
    cand = None
    if can["canary_ok"] is not False:
        for rung in rungs:
            v = rung_rows[rung]
            if (v["identifiability"]["identifiability"] in ("IDENTIFIABLE", "MARGINAL")
                    and v["oracle_rate"] - v["raw"]["acc"] >= 0.10 and v["n_failures"] >= 5):
                cand = rung
                break
    cand_note = (None if cand or can["canary_ok"] is not False else
                 "candidacy suspended: PRIMITIVE_CANARY_FAILED")
    legal, diversity, power, qv_vc = None, None, None, None
    if cand is not None:
        tasks = gen_tasks(cand, seed, n_per_rung)[:subset_n]
        legal = run_legal_subset(model, tok, device, tasks)
        diversity = response_diversity([tuple(s) for s in legal["signatures"]],
                                       {a: legal["discord_vs_raw"][a]
                                        for a in ("e5dup", "e5sham", "e7sel")})
        # primary comparison: best legal arm vs raw (paired discordants)
        p01 = max(legal["discord_vs_raw"].values()) / legal["n"]
        p10 = 0.0
        power = power_gate(required_n_mcnemar(p01, p10), budget_n)
        qv_vc = legal["rank1_vs_chance"]
    lh = (legal_headroom(legal["rates"]["raw"],
                         max(legal["rates"]["e5dup"], legal["rates"]["e7sel"]),
                         rung_rows[cand]["oracle_rate"]) if legal else None)
    repl = check_replication(replication_ref, param_sha)
    cap_top = (rung_rows[cand]["capability"] if cand else
               {"capability": "INSUFFICIENT", "notes": [cand_note or "no candidate rung"]})
    ident_top = (rung_rows[cand]["identifiability"] if cand else
                 {"identifiability": "NOT_IDENTIFIABLE",
                  "reason": cand_note or "no candidate rung"})
    decision = decide_readiness(
        stage, cap_top, ident_top, can["canary_ok"],
        frontier["verdict"],
        (lh["legal_gap"] if lh else None),
        (diversity["status"] if diversity else None),
        (power["status"] if power else None),
        protocol_sha, repl["replication_ok"],
        rung_rows[cand]["n"] if cand else 0, qv_lite_vs_chance=qv_vc)
    legal_lo = 0.0
    if legal and cand:
        fails_n = rung_rows[cand]["n_failures"]
        legal_lo = wilson(max(legal["discord_vs_raw"].values()), max(fails_n, 1))[0]
    x0 = x0_permitted(cap_top["capability"], can["canary_ok"],
                      round(legal_lo, 4),
                      diversity["status"] if diversity else None,
                      legal["n"] if legal else 0, stage == "qualify")
    x1 = x1_permitted(None, None)  # no X0 evidence exists yet
    return {
        "schema": "anra-cognition-readiness/v2",
        "phase": "CALIBRATION" if stage == "calibrate" else "QUALIFICATION",
        "timestamp": t0,
        "checkpoint_identity": {"path": checkpoint, "checkpoint_sha256": ckpt_sha,
                                "parameter_sha256": param_sha},
        "tokenizer_sha256": tok_sha, "experiment_source_sha256": exp_sha,
        "runtime_commit": commit, "protocol_sha": protocol_sha,
        "design": {"seed": seed, "n_per_rung": n_per_rung, "rungs": list(rungs),
                   "subset_n": subset_n, "budget_n": budget_n,
                   "primary_comparison": "best_legal_vs_raw"},
        "capability_family": "binding",
        "primitive_canaries": can,
        "frontier": frontier,
        "rung_results": rung_rows,
        "candidate_rung": cand,
        "candidate_note": cand_note,
        "legal_intervention_results": legal,
        "legal_headroom": lh,
        "response_diversity": diversity,
        "power": power,
        "replication": repl,
        "substrate_capability": cap_top.get("capability"),
        "experiment_identifiability": ident_top.get("identifiability"),
        "research_readiness": decision["readiness"],
        "readiness_reason": decision["reason"],
        "x0_permission": x0,
        "x1_permission": x1,
        "blockers": ([] if decision["readiness"] in ("READY_SCOPED", "READY")
                     else [decision["reason"]]),
    }
