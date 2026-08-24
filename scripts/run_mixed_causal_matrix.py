"""MIXED-CAUSAL-v1 FULL intervention-outcome matrix runner (v2, corrected).

For every task, runs EVERY applicable action and records
(ObservedState, action, output, verifier result, cost).

This produces the causal learning data: P(success | observed state, action).
The old policy's chosen action is irrelevant here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

CODE_RE = __import__("re").compile(r"\b[A-Z]{3}-\d{3}\b")


@torch.no_grad()
def free_greedy(model, tok, prompt, max_new_tokens=12):
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
def constrained_greedy(model, tok, prompt, codes, max_new_tokens=12):
    device = next(model.parameters()).device
    full = [f" {c}." for c in codes]
    ids = [tok.bos_token_id, *tok.encode(prompt)]
    text = ""
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([ids], dtype=torch.long, device=device))[0, -1, :]
        for tid in torch.argsort(logits, descending=True).tolist():
            cand = text + tok.decode([tid])
            if any(s.startswith(cand) for s in full):
                text = cand
                ids.append(tid)
                break
        else:
            break
        if text in full:
            return text.strip().rstrip("."), True
    return text.strip(), False


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


def main() -> None:
    import connector.experiments.mixed_causal_v1 as mc
    from connector.experiments.counterfactual_normalization import (
        normalize_scores, build_counterfactual_queries,
        verify_byte_identical_context, argmax)
    from anra_core.checkpoint import load_core_checkpoint
    from anra_core.tokenizer import V4Tokenizer

    FIX = mc.fixture_hash()
    model, _, ident = load_core_checkpoint(
        "checkpoints/anra-v4-20k-sft6-queryswap-replication.pt",
        legacy_unverified=True)
    tok = V4Tokenizer.load_canonical()
    tasks = mc.build_tasks()

    rows = []
    t0 = time.time()

    for ti, task in enumerate(tasks):
        applicable = mc.applicable_actions(task)
        base_prompt = mc.build_prompt(task)
        observed = {
            "task_id": task["task_id"],
            "n_candidates": len(task["candidates"]),
            "output_arity": task["output_arity"],
            "format_name": task["fmt"],
            "query_target_index": task["query_target_index"],
            "n_alt_query_targets": len(task["alt_query_targets"]),
            "applicable_actions": list(applicable),
        }

        # ---- shared scoring pass for candidate tasks with cf structure --
        raw_scores = norm_scores = None
        raw_pick_code = norm_pick_code = None
        if len(task["candidates"]) >= 2 and task["alt_query_targets"]:
            candidates = task["candidates"]
            actual_idx = task["query_target_index"]
            raw_scores = [completion_logprob(
                model, tok, base_prompt, f" {c}.") for c in candidates]
            cf_prompts = mc.counterfactual_queries(task)
            verify_byte_identical_context(cf_prompts)
            cf_indices = sorted(cf_prompts.keys())
            cf_scores_by_q = {
                j: [completion_logprob(model, tok, cf_prompts[j],
                                       f" {c}.") for c in candidates]
                for j in cf_indices}
            norm_scores = normalize_scores(actual_idx, raw_scores,
                                           cf_scores_by_q)
            raw_pick_code = candidates[argmax(raw_scores)]
            norm_pick_code = candidates[argmax(norm_scores)]

        free_out = free_greedy(model, tok, base_prompt)
        fc = CODE_RE.findall(free_out)
        free_code = fc[0] if len(fc) == 1 else None

        # ---- execute EVERY applicable action ----
        results = {}
        for action in applicable:
            if action == "NO_CHANGE":
                emitted = free_out
            elif action == "ABSTAIN":
                emitted = None
            elif action == "CONSTRAINED":
                emitted, _ = constrained_greedy(model, tok, base_prompt,
                                                task["candidates"])
            elif action == "NORMALIZED":
                emitted = f" {norm_pick_code}."
            elif action == "NORM_EXACT":
                emitted = f" {norm_pick_code}."
            ok = False if emitted is None else mc.verify(task, emitted)
            from connector.experiments.mixed_causal_v1 import COSTS
            results[action] = {"output": (emitted or "").strip()[:48],
                               "verifier_pass": bool(ok),
                               "cost": mc.COSTS[action]}

        rows.append({
            "observed": observed,
            "raw_pick_code": raw_pick_code,
            "norm_pick_code": norm_pick_code,
            "free_out_code": free_code,
            "raw_scores": raw_scores,
            "norm_scores": norm_scores,
            "actions": results,
            # evaluator-only:
            "gold_code": task["gold"],
            "family_analysis": task["family"],
        })
        if ti % 10 == 9:
            print(f"{ti+1}/{len(tasks)}", flush=True)

    report = {
        "schema": "anra-mixed-causal-matrix/v2",
        "role": "DEVELOPMENT — causal learning data; family labels are "
                "analysis-only and must never enter policy inputs",
        "fixture_sha256": FIX,
        "checkpoint_sha256": ident.parameter_sha256,
        "n_tasks": len(rows),
        "per_task_rows": rows,
        "wall_seconds": round(time.time() - t0, 1),
    }
    out = ROOT / "output/mixed_causal_matrix_v2.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"matrix written: {out} ({len(rows)} tasks)")


if __name__ == "__main__":
    main()
