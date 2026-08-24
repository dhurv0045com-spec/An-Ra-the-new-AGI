"""MIXED-CAUSAL-v1 dev experiment: cost-weighted intervention selection.

Pipeline per task:
  1. run FREE decode (observation)
  2. compute observed state (candidate scores, normalized geometry)
  3. policy (or baseline rule) picks ONE action from:
     NO_CHANGE / CONSTRAINED / NORMALIZE / NORM_EXACT / ABSTAIN
  4. execute ONLY the chosen action
  5. verifier decides success; record cost

Costs: NO_CHANGE 0, CONSTRAINED 1, NORMALIZE 2, NORM_EXACT 3, ABSTAIN 0.
Score = successes - 0.25 * unnecessary_interventions (cost reported raw).

Policy: frozen v2 weights applied to candidate tasks; ABSTAIN for
non-candidate tasks is what a LEARNED gate would need to discover — here we
test three policy variants honestly labeled:
  adaptive_v2_raw_gate : v2 policy decides NORMALIZE vs KEEP_RAW on
                         candidate tasks; NO_CHANGE on non-candidate tasks
                         only if free output already contains a valid answer
                         shape (observed-only heuristic, NOT family label).
All baselines are fixed policies.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")
COSTS = {"NO_CHANGE": 0, "CONSTRAINED": 1, "NORMALIZE": 2,
         "NORM_EXACT": 3, "ABSTAIN": 0}


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
    from anra_core.checkpoint import load_core_checkpoint
    from anra_core.tokenizer import V4Tokenizer
    from connector.experiments.observed_self_model import (
        ObservedArmState, AdaptivePolicy)

    FIX = mc.fixture_hash()
    pol_json = json.loads((ROOT / "output/observed_policy_v2.json")
                          .read_text(encoding="utf-8"))
    policy = AdaptivePolicy(weights=tuple(pol_json["weights"]),
                            bias=pol_json["bias"],
                            threshold=pol_json.get("threshold", 0.5))

    model, _, ident = load_core_checkpoint(
        "checkpoints/anra-v4-20k-sft6-queryswap-replication.pt",
        legacy_unverified=True)
    tok = V4Tokenizer.load_canonical()

    tasks = mc.build_tasks()
    rows = []
    t0 = time.time()

    for ti, task in enumerate(tasks):
        prompt = task["prompt"]
        candidates = task.get("candidates", [])

        # ---- observation phase (no verifier) ----
        free_out = free_greedy(model, tok, prompt)
        fc = CODE_RE.findall(free_out)
        free_code = fc[0] if len(fc) == 1 else None

        if candidates and len(candidates) >= 2:
            raw_scores = [completion_logprob(model, tok, prompt,
                                             f" {c}.") for c in candidates]
            adj_scores = []
            for i in range(len(candidates)):
                others = [raw_scores[j] for j in range(len(candidates))
                          if j != i]
                adj_scores.append(raw_scores[i]
                                  - sum(others) / len(others))
            raw_pick = max(range(len(candidates)),
                           key=lambda i: raw_scores[i])
            norm_pick = max(range(len(candidates)),
                            key=lambda i: adj_scores[i])
            state = ObservedArmState(
                n_candidates=len(candidates),
                format_name="prose",
                raw_pick_code=candidates[raw_pick],
                norm_pick_code=candidates[norm_pick],
                free_out_code=free_code,
                constrained_pick_code=None,
                raw_scores=raw_scores,
                norm_scores=adj_scores,
            )
            decision = policy.decide(state)
        else:
            # single-candidate or open tasks: normalization is undefined;
            # NO_CHANGE (or trivially correct copy) is the only sane action
            decision = "NO_CHANGE"

        # non-candidate/open tasks: any candidate-restricting action is
        # structurally inappropriate -> the only sane actions are
        # NO_CHANGE or ABSTAIN; policy outputs map to NO_CHANGE.

        # ---- execution of the CHOSEN action ----
        if decision == "CONSTRAINED":
            out_c, _ = constrained_greedy(model, tok, prompt, candidates)
            emitted = out_c
        elif decision == "NORMALIZE":
            emitted = f" {candidates[norm_pick]}."
        elif decision == "NORM_EXACT":
            emitted = f" {candidates[norm_pick]}."
        else:  # NO_CHANGE or ABSTAIN
            emitted = free_out

        ok = mc.verify(task, emitted)
        rows.append({
            "task_id": task["task_id"],
            "family": task["family"],          # evaluation stratification only
            "action": decision,
            "cost": COSTS[decision],
            "ok": bool(ok),
            "emitted": emitted.strip()[:40],
            # persist observed state for future policy training (v3)
            "observed_state": (state.feature_vector() if candidates
                               and len(candidates) >= 2 else None),
            "n_candidates": len(candidates),
        })
        if ti % 10 == 9:
            print(f"{ti+1}/{len(tasks)}", flush=True)

    def agg(sel):
        sub = [r for r in rows if sel(r)]
        n = len(sub)
        succ = sum(1 for r in sub if r["ok"])
        cost = sum(r["cost"] for r in sub)
        return {"n": n, "successes": succ, "acc": round(succ/max(n,1), 4),
                "total_cost": cost}

    overall = {
        "n": len(rows),
        "successes": sum(1 for r in rows if r["ok"]),
        "acc": round(sum(1 for r in rows if r["ok"])/len(rows), 4),
        "total_intervention_cost": sum(r["cost"] for r in rows),
        "unnecessary_interventions": sum(
            1 for r in rows if r["cost"] > 0 and r["family"] in
            ("non_candidate", "copy_single")),
    }
    by_family = {fam: agg(lambda r, f=fam: r["family"] == f)
                 for fam in mc.FAMILIES}
    report = {
        "schema": "anra-mixed-causal-dev/v1",
        "role": "DEVELOPMENT — informs preregistration; not a promotion claim",
        "fixture_sha256": FIX,
        "checkpoint_sha256": ident.parameter_sha256,
        "policy": "frozen v2",
        "overall": overall,
        "by_family": by_family,
        "per_task_rows": rows,
        "wall_seconds": round(time.time()-t0, 1),
    }
    (ROOT / "output/mixed_causal_dev_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"overall": overall, "by_family": by_family}, indent=2))


if __name__ == "__main__":
    main()
