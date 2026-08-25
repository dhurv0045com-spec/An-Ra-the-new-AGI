"""Harvest FULL feature vectors + all-arm outcomes from consumed MC fixtures.

MC-v4/v5/v6 promotion runners only stored the adaptive action's outcome.
This re-runs every applicable arm on those fixtures to build complete
training rows (observed state, action, pass) for policy v7.

Consumed-fixture reuse for DEVELOPMENT training is sanctioned: v7 will be
tested on a fresh MC-v7 fixture frozen after v7's freeze.
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
    import connector.experiments.mixed_causal_v4 as mc4
    import connector.experiments.mixed_causal_v5 as mc5
    import connector.experiments.mixed_causal_v6 as mc6
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

    harvested = []
    for mod, tag in ((mc4, "mc4"), (mc5, "mc5"), (mc6, "mc6")):
        tasks = mod.build_tasks()
        print(f"harvesting {tag}: {len(tasks)} tasks", flush=True)
        for task in tasks:
            candidates = task["candidates"]
            base_prompt = mod.build_prompt(task)
            raw_scores = norm_scores = None
            raw_pick_code = norm_pick_code = None
            if len(candidates) >= 2 and task["alt_query_targets"]:
                actual_idx = task["query_target_index"]
                raw_scores = [completion_logprob(model, tok, base_prompt,
                                                 f" {c}.") for c in candidates]
                cfs = mod.counterfactual_queries(task)
                verify_byte_identical_context(cfs)
                cf_scores = {j: [completion_logprob(model, tok, cfs[j],
                                                    f" {c}.") for c in candidates]
                             for j in sorted(cfs)}
                norm_scores = normalize_scores(actual_idx, raw_scores, cf_scores)
                raw_pick_code = candidates[argmax(raw_scores)]
                norm_pick_code = candidates[argmax(norm_scores)]

            free_out = free_greedy(model, tok, base_prompt)
            fc = CODE_RE.findall(free_out)
            free_code = fc[0] if len(fc) == 1 else None

            feats = ofeat({
                "observed": {"n_candidates": len(candidates),
                             "output_arity": task["output_arity"],
                             "format_name": task["fmt"]},
                "raw_pick_code": raw_pick_code,
                "norm_pick_code": norm_pick_code,
                "free_out_code": free_code,
                "raw_scores": raw_scores,
                "norm_scores": norm_scores,
            })

            actions = {}
            for action in ("NO_CHANGE", "CONSTRAINED", "NORMALIZED"):
                if action == "NO_CHANGE":
                    emitted = free_out
                    if not mc6.verify(task, emitted) and len(candidates) < 2:
                        continue  # non_candidate: NO_CHANGE is the only arm
                elif action == "CONSTRAINED":
                    if not candidates or task["output_arity"] > 1:
                        continue
                    emitted, _ = constrained_greedy(model, tok, base_prompt,
                                                    candidates)
                else:
                    if norm_scores is None:
                        continue
                    emitted = f" {norm_pick_code}."
                actions[action] = bool(mod.verify(task, emitted))
            # keep only tasks where at least NO_CHANGE ran
            if "NO_CHANGE" not in actions:
                continue
            harvested.append({
                "source_fixture": tag,
                "features": feats,
                "actions_pass": actions,
            })
        print(f"  {tag} done: total rows {len(harvested)}", flush=True)

    out = ROOT / "output/harvest_v7_pool.json"
    out.write_text(json.dumps({
        "schema": "anra-mc-harvest/v1",
        "checkpoint_sha256": ident.parameter_sha256,
        "sources": ["mixed_causal_v4", "mixed_causal_v5", "mixed_causal_v6"],
        "n_rows": len(harvested),
        "rows": harvested,
    }, indent=2), encoding="utf-8")
    print(f"WROTE {out} with {len(harvested)} rows")


if __name__ == "__main__":
    main()
