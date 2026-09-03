"""Cognition Readiness Gate runner v1 — FROZEN HISTORICAL SEMANTICS.

Superseded by readiness v2 (status.py, readiness_v2.py, frontier.py,
canaries.py). v1 receipts (schema anra-cognition-readiness/v1) remain valid
history but v1 MUST NOT be used for new qualification claims: v1 conflates
capability/identifiability/readiness, ignores chance and uncertainty, has no
calibration/qualification split, and emitted READY from N=12 pilots.
New code paths use --mode calibrate|qualify in qualify_checkpoint.py.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import torch

import sys as _sys

_XF = Path(__file__).resolve().parents[1]
if str(_XF) not in _sys.path:
    _sys.path.insert(0, str(_XF))
_RT = _XF / "_runtime"
if str(_RT) not in _sys.path:
    _sys.path.insert(0, str(_RT))

from checkpoint_identity import load_core, resolve_checkpoint  # noqa: E402
from provenance import git_head, param_sha256_from_state_dict, sha256_file, sha256_json  # noqa: E402

from readiness.ladder import RUNGS, gen_tasks, oracle_prompt  # noqa: E402
from readiness.identifiability import check_identifiability  # noqa: E402

CODE_RE = re.compile(r"\b[A-Z]{2,4}-\d{3,4}\b")


@torch.no_grad()
def _greedy(model, tok, prompt, device, max_new=12) -> str:
    ids = [tok.bos_token_id, *tok.encode(prompt)]
    cur, out = list(ids), []
    for _ in range(max_new):
        logits = model(torch.tensor([cur], dtype=torch.long, device=device))[:, -1, :]
        nxt = int(logits.argmax(dim=-1))
        if nxt == tok.eos_token_id:
            break
        out.append(nxt)
        cur.append(nxt)
    return tok.decode(out)


def _strict(out: str, gold: str) -> int:
    c = CODE_RE.findall(out)
    return int(len(c) == 1 and c[0] == gold)


@torch.no_grad()
def _lp(model, tok, prompt, cand, device) -> float:
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(f" {cand}.")
    ids = torch.tensor([[tok.bos_token_id, *p_ids, *c_ids]], dtype=torch.long, device=device)
    lp = torch.log_softmax(model(ids)[0].float(), -1)
    return sum(float(lp[pos - 1, ids[0, pos]]) for pos in range(1 + len(p_ids), ids.shape[1]))


def run_gate(checkpoint: str, seed: int, n_per_rung: int, device: str,
             rungs: tuple = RUNGS, qv_lite_n: int = 8) -> dict:
    torch.manual_seed(seed)
    ckpt = str(resolve_checkpoint(checkpoint))
    model, tok, payload = load_core(ckpt, device)
    from anra_core.config import CANONICAL_CONFIG
    param_sha = param_sha256_from_state_dict(model.state_dict())
    ckpt_sha = sha256_file(ckpt)
    try:
        tok_ident = tok.identity()
    except Exception:
        tok_ident = {"vocab": "canonical-v4-32k"}
    exp_sha = sha256_file(str(Path(__file__).resolve()))

    rung_rows = {}
    for rung in rungs:
        tasks = gen_tasks(rung, seed, n_per_rung)
        raw_ok, orb_ok = [], []
        for t in tasks:
            raw_ok.append(_strict(_greedy(model, tok, t["prompt"], device), t["gold"]))
            orb_ok.append(_strict(_greedy(model, tok, oracle_prompt(t), device), t["gold"]))
        fails = [t for t, r in zip(tasks, raw_ok) if r == 0]
        rep = sum(1 for t, o in zip(tasks, orb_ok)
                  if t in fails and o == 1)
        # discordant approx: oracle-only repairs among failures
        disc = sum(1 for t in fails
                   if _strict(_greedy(model, tok, oracle_prompt(t), device), t["gold"]) == 1)
        ident = check_identifiability(len(tasks), sum(raw_ok), len(fails), rep, disc,
                                      chance=1.0 / max(len(tasks[0]["codes"]), 1))
        rung_rows[rung] = {"n": len(tasks), "raw_rate": round(sum(raw_ok) / len(tasks), 4),
                           "oracle_rate": round(sum(orb_ok) / len(tasks), 4),
                           "n_failures": len(fails), "n_oracle_repairs": rep,
                           "identifiability": ident}
        print(f"[{rung}] raw={rung_rows[rung]['raw_rate']} oracle={rung_rows[rung]['oracle_rate']} "
              f"-> {ident['substrate_adequacy']}/{ident['decision']}", flush=True)

    partial = [r for r, v in rung_rows.items()
               if v["identifiability"]["substrate_adequacy"] == "ADEQUATE"]
    # QV-lite in first partial rung
    qv_lite = None
    if partial:
        r0 = partial[0]
        tasks = [t for t in gen_tasks(r0, seed, n_per_rung)][:qv_lite_n]
        ranks = []
        for t in tasks:
            row = [_lp(model, tok, t["prompt"], c, device) for c in t["codes"]]
            gold_i = t["codes"].index(t["gold"])
            ranks.append(1 if int(np.argmax(row)) == gold_i else 0)
        qv_lite = {"rung": r0, "n": len(tasks),
                   "raw_rank1": round(sum(ranks) / len(ranks), 4),
                   "note": "single-query lite: no cross-query normalization; full QV matrix runs only if gate READY"}

    if partial:
        # require oracle headroom on the partial rung
        best = max(partial, key=lambda r: rung_rows[r]["oracle_rate"] - rung_rows[r]["raw_rate"])
        gap = rung_rows[best]["oracle_rate"] - rung_rows[best]["raw_rate"]
        if rung_rows[best]["oracle_rate"] >= 0.40 and gap >= 0.20:
            classification = "READY_FOR_BINDING_CAUSAL_RESEARCH"
        else:
            classification = "MARGINAL"
        permitted = ["query-value matrix", "answer-blind intervention basis", "X0/X1 pilots"]
        blockers: list[str] = []
    else:
        all_floor = all(v["identifiability"]["substrate_adequacy"] in ("FLOOR_LIMITED", "ORACLE_LIMITED")
                        for v in rung_rows.values())
        classification = "NOT_READY_FLOOR" if all_floor else "MARGINAL"
        permitted = ["software validation", "negative-control use", "historical comparison"]
        blockers = [f"{r}:{v['identifiability']['substrate_adequacy']}" for r, v in rung_rows.items()]
    receipt = {
        "schema": "anra-cognition-readiness/v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": {"checkpoint": ckpt, "checkpoint_sha256": ckpt_sha,
                       "parameter_sha256": param_sha,
                       "config_sha256": sha256_json(payload["model_config"]),
                       "tokenizer_sha256": sha256_json(tok_ident),
                       "runtime_commit": git_head(Path(__file__).resolve().parents[2]),
                       "experiment_source_sha256": exp_sha},
        "design": {"seed": seed, "n_per_rung": n_per_rung, "rungs": list(rungs)},
        "capability_family": "binding",
        "frontier": rung_rows,
        "partial_rungs": partial,
        "qv_lite": qv_lite,
        "classification": classification,
        "blockers": blockers,
        "permitted_next_experiments": permitted,
    }
    return receipt
