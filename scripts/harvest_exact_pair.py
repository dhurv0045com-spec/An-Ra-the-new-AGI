"""Collect EXACT_PAIR (multi-emission) outcomes on the consumed MC-v9
fixture for policy v8 development training.

EXACT_PAIR: emit top-2 candidates by normalized score, normalized-rank
order. Applicable only when output_arity == 2 and n_candidates >= 2.
Observable-only: uses no gold. Consumed-fixture reuse is sanctioned as
development data for the next generation (tested on fresh MC-v10).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")


@torch.no_grad()
def free_greedy(model, tok, prompt, max_new_tokens=16):
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
    import connector.experiments.mixed_causal_v9 as mc
    from connector.experiments.counterfactual_normalization import (
        normalize_scores, verify_byte_identical_context, argmax)
    from anra_core.checkpoint import load_core_checkpoint
    from anra_core.tokenizer import V4Tokenizer
    sys.path.insert(0, str(ROOT))
    from scripts.train_self_model_v3 import observed_features as ofeat

    model, _, ident = load_core_checkpoint(
        "checkpoints/anra-v4-20k-sft6-queryswap-replication.pt",
        legacy_unverified=True)
    tok = V4Tokenizer.load_canonical()
    tasks = mc.build_tasks()

    rows = []
    for task in tasks:
        if task["family"] != "composition":
            continue
        candidates = task["candidates"]
        base_prompt = mc.build_prompt(task)

        # Composition tasks name both targets in one query; there are no
        # alternative query targets, so normalization is undefined here.
        # Observable ranking = raw scores over visible codes.
        raw_scores = [completion_logprob(model, tok, base_prompt,
                                         f" {c}.") for c in candidates]
        order = sorted(range(len(candidates)),
                       key=lambda i: raw_scores[i], reverse=True)
        pair = f" {candidates[order[0]]} {candidates[order[1]]}."
        norm_scores = raw_scores  # observable proxy; no CF queries exist
        pair_pass = mc.verify(task, pair)

        # NO_CHANGE baseline for the same composition task
        free_out = free_greedy(model, tok, base_prompt)
        nc_pass = mc.verify(task, free_out)

        feats = ofeat({
            "observed": {"n_candidates": len(candidates),
                         "output_arity": task["output_arity"],
                         "format_name": task["fmt"]},
            "raw_pick_code": candidates[argmax(raw_scores)],
            "norm_pick_code": candidates[order[0]],
            "free_out_code": None,
            "raw_scores": raw_scores,
            "norm_scores": norm_scores,
        })

        rows.append({
            "features": feats,
            "actions_pass": {
                "NO_CHANGE": bool(nc_pass),
                "EXACT_PAIR": bool(pair_pass),
            },
            "retained_pair_output": pair.strip(),
            "family_analysis_only": task["family"],
        })

    out = ROOT / "output/exact_pair_harvest_v8.json"
    out.write_text(json.dumps({
        "schema": "anra-exact-pair-harvest/v1",
        "checkpoint_sha256": ident.parameter_sha256,
        "source_fixture": "mixed_causal_v9 (consumed; dev use sanctioned)",
        "n_rows": len(rows),
        "exact_pair_successes": sum(1 for r in rows
                                    if r["actions_pass"]["EXACT_PAIR"]),
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"WROTE {out}: {len(rows)} composition rows, "
          f"{sum(1 for r in rows if r['actions_pass']['EXACT_PAIR'])} "
          f"EXACT_PAIR successes")


if __name__ == "__main__":
    main()
