"""Causal selection experiment: five runtime arms + counterfactual query
normalization (OBSERVED-ONLY, no training).

Arms per target:
  FREE          free greedy decode (status quo)
  RAW           full-sequence candidate argmax under the ACTUAL query
  CONSTRAINED   greedy restricted to candidate-code continuations
  NORMALIZED    argmax of counterfactual-query-normalized scores
                ADJ_i = logP(v_i | actual_query) - mean_j!=i logP(v_i | q_j)
  NORM_EXACT    NORMALIZED selection + emit that selected candidate exactly

Observed-only contract: candidates come from the context; gold never enters
arm construction; the verifier alone decides pass/fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


def _git_state() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             text=True).strip())
        return {"source_commit": commit, "dirty": dirty}
    except Exception:
        return {"source_commit": None, "dirty": None}


@torch.no_grad()
def free_greedy(model, tok, prompt, max_new_tokens=10):
    device = next(model.parameters()).device
    ids = [tok.bos_token_id, *tok.encode(prompt)]
    out = []
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([ids], dtype=torch.long, device=device))[:, -1, :]
        nxt = int(logits.argmax(dim=-1).item())
        if nxt == tok.eos_token_id:
            break
        out.append(nxt)
        ids.append(nxt)
    return tok.decode(out)


@torch.no_grad()
def constrained_greedy(model, tok, prompt, codes, max_new_tokens=10):
    device = next(model.parameters()).device
    full = [f" {c}." for c in codes]
    ids = [tok.bos_token_id, *tok.encode(prompt)]
    text = ""
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([ids], dtype=torch.long, device=device))[0, -1, :]
        order = torch.argsort(logits, descending=True)
        picked = None
        for tid in order.tolist():
            cand = text + tok.decode([tid])
            if any(s.startswith(cand) for s in full):
                picked = (tid, cand)
                break
        if picked is None:
            break
        tid, text = picked
        ids.append(tid)
        if text in full:
            return text, True
    return text, False


@torch.no_grad()
def completion_logprob(model, tok, prompt, completion):
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(completion)
    ids = torch.tensor([[tok.bos_token_id, *p_ids, *c_ids]], dtype=torch.long,
                       device=next(model.parameters()).device)
    logits = model(ids)[0]
    lp = torch.log_softmax(logits.float(), dim=-1)
    return sum(float(lp[pos - 1, ids[0, pos]].item())
               for pos in range(1 + len(p_ids), ids.shape[1]))


def strict_single_code(out: str, code_re) -> bool:
    c = code_re.findall(out)
    return len(c) == 1


def run_arms(module, model, tok, code_re, checkpoint_sha: str,
             source_commit, dirty, fixture_sha: str,
             label: str, out_path: str) -> dict:
    """Run all five arms on every target of the fixture module's groups."""
    groups = module.build_groups()
    rows = []
    t0 = time.time()

    for gi, g in enumerate(groups):
        recs = g["displayed_facts"]
        codes = [r["code"] for r in recs]
        prompts = {qi: module.build_query_prompt(g, qi)
                   for qi in range(len(recs))}

        # RAW scores under actual queries + counterfactual baselines
        # raw[q][i] = logP(v_i | q_q); cf[i][j] = logP(v_i | q_j), j != i
        raw = [[completion_logprob(model, tok, prompts[q],
                                   f" {recs[i]['code']}.")
                for i in range(len(recs))] for q in range(len(recs))]
        for qi in range(len(recs)):
            prompt = prompts[qi]
            gold = recs[qi]["code"]

            adj_scores = []
            for i in range(len(recs)):
                others = [raw[j][i] for j in range(len(recs)) if j != qi]
                baseline = sum(others) / len(others)
                adj_scores.append(raw[qi][i] - baseline)

            arm_raw_pick = max(range(len(recs)), key=lambda i: raw[qi][i])
            arm_norm_pick = max(range(len(recs)), key=lambda i: adj_scores[i])

            out_free = free_greedy(model, tok, prompt)
            out_constr, completed = constrained_greedy(model, tok, prompt, codes)

            # NORM_EXACT: emit exactly the normalized-selected candidate
            norm_sel_code = recs[arm_norm_pick]["code"]
            out_exact = f" {norm_sel_code}."

            def ok_free():
                c = code_re.findall(out_free)
                return len(c) == 1 and c[0] == gold

            row = {
                "gi": gi, "qi": qi, "gold": gold,
                "raw_rank_of_gold": 1 + sum(1 for j in range(len(recs))
                                            if raw[qi][j] > raw[qi][qi]),
                "adj_rank_of_gold": 1 + sum(
                    1 for j in range(len(recs))
                    if j != arm_norm_pick and adj_scores[j] > adj_scores[arm_norm_pick]),
                "FREE_ok": ok_free(), "FREE_out": out_free.strip()[:24],
                "RAW_ok": recs[arm_raw_pick]["code"] == gold,
                "CONSTRAINED_ok": _constr_ok(out_constr, gold),
                "NORMALIZED_ok": recs[arm_norm_pick]["code"] == gold,
                "NORM_EXACT_ok": recs[arm_norm_pick]["code"] == gold,
                "NORM_EXACT_out": out_exact.strip(),
                "adj_top2_margin": (
                    sorted(adj_scores)[-1] - sorted(adj_scores)[-2])
                    if len(adj_scores) >= 2 else 0.0,
                "raw_top2_margin": (
                    sorted(raw[qi])[-1] - sorted(raw[qi])[-2]),
            }
            rows.append(row)
        print(f"group {gi}: done", flush=True)

    def acc(arm):
        n = sum(1 for r in rows if r[f"{arm}_ok"])
        return f"{n}/{len(rows)}", round(n / len(rows), 4)

    arms = ["FREE", "RAW", "CONSTRAINED", "NORMALIZED", "NORM_EXACT"]
    summary = {a: dict(zip(("n", "rate"), acc(a))) for a in arms}

    report = {
        "schema": "anra-causal-selection-arms/v1",
        "label": label,
        "checkpoint_sha256": checkpoint_sha,
        "fixture_sha256": fixture_sha,
        "provenance": {
            **_git_state(),
            "tokenizer": tok.identity(),
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
            "decode_config": {"max_new_tokens": 10, "greedy": True},
            "weights_modified": False,
        },
        "arms_summary": summary,
        "per_item_rows": rows,
        "wall_seconds": round(time.time() - t0, 1),
    }
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return report


def _constr_ok(out_constr: str, gold: str) -> bool:
    return out_constr.strip() == f" {gold}."
