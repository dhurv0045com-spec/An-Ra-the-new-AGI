"""Harvest SLOT_PAIR outcomes on consumed MC-v10 composition tasks.

Compares three observable pair-emission strategies on the same tasks:
  WHOLE_SEQ: rank by whole-sequence logprob (the v8 approach, 22%)
  SLOT_PAIR: per-slot scoring (this technology)

Records both so v9's training data contains the contrast. Consumed-
fixture reuse sanctioned as development data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import connector.experiments.mixed_causal_v10 as mc
    from connector.experiments.slot_pair import (
        slot_pair_scores, choose_pair, pair_emission)
    from anra_core.checkpoint import load_core_checkpoint
    from anra_core.tokenizer import V4Tokenizer
    sys.path.insert(0, str(ROOT))
    from scripts.train_self_model_v3 import observed_features as ofeat

    model, _, ident = load_core_checkpoint(
        "checkpoints/anra-v4-20k-sft6-queryswap-replication.pt",
        legacy_unverified=True)
    tok = V4Tokenizer.load_canonical()

    @torch.no_grad()
    def completion_logprob(prompt, completion):
        p_ids = tok.encode(prompt)
        c_ids = tok.encode(completion)
        ids = torch.tensor([[tok.bos_token_id, *p_ids, *c_ids]],
                           dtype=torch.long,
                           device=next(model.parameters()).device)
        logits = model(ids)[0]
        lp = torch.log_softmax(logits.float(), dim=-1)
        return sum(float(lp[pos - 1, ids[0, pos]].item())
                   for pos in range(1 + len(p_ids), ids.shape[1]))

    rows = []
    n_whole = n_slot = 0
    for task in mc.build_tasks():
        if task["family"] != "composition":
            continue
        candidates = task["candidates"]
        base_prompt = mc.build_prompt(task)
        raw_scores = [completion_logprob(base_prompt, f" {c}.")
                      for c in candidates]

        # WHOLE_SEQ strategy (v8)
        order = sorted(range(len(candidates)),
                       key=lambda i: raw_scores[i], reverse=True)
        whole_pair = pair_emission(candidates[order[0]], candidates[order[1]])
        whole_ok = bool(mc.verify(task, whole_pair))
        n_whole += whole_ok

        # SLOT_PAIR strategy (new technology)
        s0, s1 = slot_pair_scores(completion_logprob, model, tok,
                                  base_prompt, candidates)
        f_code, s_code = choose_pair(candidates, s0, s1)
        slot_pair = pair_emission(f_code, s_code)
        slot_ok = bool(mc.verify(task, slot_pair))
        n_slot += slot_ok

        feats = ofeat({
            "observed": {"n_candidates": len(candidates),
                         "output_arity": task["output_arity"],
                         "format_name": task["fmt"]},
            "raw_pick_code": None,
            "norm_pick_code": None,
            "free_out_code": None,
            "raw_scores": raw_scores,
            "norm_scores": None,
        })
        rows.append({
            "features": feats,
            "actions_pass": {
                "NO_CHANGE": False,
                "EXACT_PAIR": whole_ok,
                "SLOT_PAIR": slot_ok,
            },
            "retained_pair_output": slot_pair.strip(),
        })

    out = ROOT / "output/slot_pair_harvest_v9.json"
    out.write_text(json.dumps({
        "schema": "anra-slot-pair-harvest/v1",
        "checkpoint_sha256": ident.parameter_sha256,
        "source_fixture": "mixed_causal_v10 (consumed; dev use sanctioned)",
        "n_rows": len(rows),
        "whole_seq_successes": n_whole,
        "slot_pair_successes": n_slot,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"composition rows: {len(rows)}")
    print(f"WHOLE_SEQ: {n_whole}   SLOT_PAIR: {n_slot}")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
