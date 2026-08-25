"""MIXED-CAUSAL-v4 FINAL REPLICATION: policy v4 vs all baselines.

Frozen policy v4 (SHA 937ded8c..., standardized features) decides one
action per task from observed state; verifier scores afterward.
"""
from __future__ import annotations

import json
import math
import random
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


def top2_margin(scores):
    s = sorted(scores or [])
    return s[-1] - s[-2] if len(s) >= 2 else 0.0


def main() -> None:
    import connector.experiments.mixed_causal_v4 as mc
    from connector.experiments.counterfactual_normalization import (
        normalize_scores, verify_byte_identical_context, argmax)
    from anra_core.checkpoint import load_core_checkpoint
    from anra_core.tokenizer import V4Tokenizer
    sys.path.insert(0, str(ROOT))
    from scripts.train_self_model_v3 import observed_features as ofeat

    FIX = mc.fixture_hash()
    pol = json.loads((ROOT / "output/self_model_v4.json").read_text(encoding="utf-8"))
    lam = pol["lambda"]
    means = pol["standardization"]["means"]
    stds = pol["standardization"]["stds"]
    models = pol["models"]

    model, _, ident = load_core_checkpoint(
        "checkpoints/anra-v4-20k-sft6-queryswap-replication.pt",
        legacy_unverified=True)
    tok = V4Tokenizer.load_canonical()
    tasks = mc.build_tasks()
    rows = []

    for ti, task in enumerate(tasks):
        applicable = [a for a in mc.applicable_actions(task) if a != "ABSTAIN"]
        base_prompt = mc.build_prompt(task)
        candidates = task["candidates"]

        raw_scores = norm_scores = None
        raw_pick_code = norm_pick_code = None
        if len(candidates) >= 2 and task["alt_query_targets"]:
            actual_idx = task["query_target_index"]
            raw_scores = [completion_logprob(model, tok, base_prompt,
                                             f" {c}.") for c in candidates]
            cfs = mc.counterfactual_queries(task)
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

        feats_raw = ofeat({
            "observed": {"n_candidates": len(candidates),
                         "output_arity": task["output_arity"],
                         "format_name": task["fmt"]},
            "raw_pick_code": raw_pick_code,
            "norm_pick_code": norm_pick_code,
            "free_out_code": free_code,
            "raw_scores": raw_scores,
            "norm_scores": norm_scores,
        })
        feats = [(x - m) / s for x, m, s in zip(feats_raw, means, stds)]

        def predict(action):
            if action not in models:
                return 0.0
            m = models[action]
            z = sum(w * x for w, x in zip(m["weights"], feats)) + m["bias"]
            return 1 / (1 + math.exp(-z))

        utilities = {a: predict(a) - lam * mc.COSTS[a] for a in applicable}
        adaptive = max(utilities, key=utilities.get)

        results = {}
        for action in applicable:
            if action == "NO_CHANGE":
                emitted = free_out
            elif action == "CONSTRAINED":
                emitted, _ = constrained_greedy(model, tok, base_prompt,
                                                candidates)
            else:
                emitted = f" {norm_pick_code}."
            results[action] = {
                "output": (emitted or "").strip()[:48],
                "pass": bool(mc.verify(task, emitted)),
                "cost": mc.COSTS[action],
            }

        rows.append({
            "adaptive_action": adaptive,
            "actions": results,
            "raw_top2": top2_margin(raw_scores),
            "norm_top2": top2_margin(norm_scores),
            "raw_pick_code": raw_pick_code,
            "norm_pick_code": norm_pick_code,
            "free_out_code": free_code,
            "family_analysis": task["family"],
        })
        if ti % 30 == 29:
            print(f"{ti+1}/{len(tasks)}", flush=True)

    n = len(rows)

    def run_policy(choose):
        succ = reg = cost = 0
        for r in rows:
            a = choose(r)
            e = r["actions"][a]
            cost += e["cost"]
            succ += e["pass"]
            if a != "NO_CHANGE" and not e["pass"] and \
                    r["actions"]["NO_CHANGE"]["pass"]:
                reg += 1
        return {"succ": succ, "reg": reg, "cost": cost}

    policies = {}
    policies["ALWAYS_NO_CHANGE"] = run_policy(lambda r: "NO_CHANGE")
    policies["ALWAYS_CONSTRAINED"] = run_policy(
        lambda r: "CONSTRAINED" if "CONSTRAINED" in r["actions"] else "NO_CHANGE")
    policies["ALWAYS_NORMALIZED"] = run_policy(
        lambda r: "NORMALIZED" if "NORMALIZED" in r["actions"] else "NO_CHANGE")
    policies["ADAPTIVE_v4"] = run_policy(lambda r: r["adaptive_action"])

    def rule_A(r):
        if "NORMALIZED" in r["actions"] and r["norm_top2"] > r["raw_top2"]:
            return "NORMALIZED"
        return "NO_CHANGE"

    def rule_B(r):
        if ("NORMALIZED" in r["actions"] and r["norm_top2"] >= 0.5
                and r["norm_pick_code"] != r["raw_pick_code"]):
            return "NORMALIZED"
        return "NO_CHANGE"

    def rule_C(r):
        if "CONSTRAINED" in r["actions"] and \
                r["free_out_code"] != r["raw_pick_code"]:
            return "CONSTRAINED"
        return "NO_CHANGE"

    def rule_D(r):
        if "NORMALIZED" in r["actions"] and r["raw_top2"] < 0.75:
            return "NORMALIZED"
        if "CONSTRAINED" in r["actions"] and r["raw_top2"] < 0.25:
            return "CONSTRAINED"
        return "NO_CHANGE"

    policies["SIMPLE_A_norm_margin_gt_raw"] = run_policy(rule_A)
    policies["SIMPLE_B_disagree_highconf"] = run_policy(rule_B)
    policies["SIMPLE_C_free_vs_raw_disagree"] = run_policy(rule_C)
    policies["SIMPLE_D_weak_raw_margin"] = run_policy(rule_D)

    oracle = sum(1 for r in rows
                 if any(e["pass"] for e in r["actions"].values()))

    rng = random.Random(20261103)
    pairs = {}
    adaptive_passes = [(r["actions"][r["adaptive_action"]]["pass"]) for r in rows]
    for name, p in policies.items():
        if name == "ADAPTIVE_v4":
            continue
        chooser = {
            "ALWAYS_NO_CHANGE": lambda r: "NO_CHANGE",
            "ALWAYS_CONSTRAINED": lambda r: "CONSTRAINED" if "CONSTRAINED" in r["actions"] else "NO_CHANGE",
            "ALWAYS_NORMALIZED": lambda r: "NORMALIZED" if "NORMALIZED" in r["actions"] else "NO_CHANGE",
            "SIMPLE_A_norm_margin_gt_raw": rule_A,
            "SIMPLE_B_disagree_highconf": rule_B,
            "SIMPLE_C_free_vs_raw_disagree": rule_C,
            "SIMPLE_D_weak_raw_margin": rule_D,
        }[name]
        diffs = [ap - (1 if r["actions"][chooser(r)]["pass"] else 0)
                 for ap, r in zip(adaptive_passes, rows)]
        mean_d = sum(diffs) / n
        boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                       for _ in range(10000))
        ci95 = [boots[int(0.025 * 10000)], boots[min(9999, int(0.975 * 10000))]]
        sd = (sum((d - mean_d) ** 2 for d in diffs) / max(n - 1, 1)) ** 0.5
        se = sd / math.sqrt(n)
        z = mean_d / se if se > 0 else 0.0
        pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        pairs[name] = {"mean_diff": round(mean_d, 4),
                       "ci95": [round(ci95[0], 4), round(ci95[1], 4)],
                       "p_two_sided_normal_approx": round(pval, 5)}

    # per-family
    from collections import defaultdict
    byfam = defaultdict(lambda: {"adaptive": 0, "best_const": 0, "n": 0})
    for r in rows:
        fam = r["family_analysis"]
        byfam[fam]["n"] += 1
        byfam[fam]["adaptive"] += r["actions"][r["adaptive_action"]]["pass"]
        const_best = max(r["actions"][a]["pass"] for a in
                         ("NO_CHANGE", "CONSTRAINED", "NORMALIZED")
                         if a in r["actions"])
        byfam[fam]["best_const"] += const_best

    report = {
        "schema": "anra-mixed-causal-v4-replication/v1",
        "fixture_sha256": FIX,
        "checkpoint_sha256": ident.parameter_sha256,
        "policy_sha256": pol["parameter_sha256"],
        "policy_frozen_commit": "1861e0f",
        "n_tasks": n,
        "policies": policies,
        "oracle_successes": oracle,
        "paired_adaptive_vs_others": pairs,
        "per_family": {k: dict(v) for k, v in byfam.items()},
        "per_task_rows": [{"adaptive_action": r["adaptive_action"],
                           "family_analysis": r["family_analysis"]}
                          for r in rows],
    }
    (ROOT / "output/mixed_causal_v4_replication.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"policies": policies, "oracle": oracle,
                      "paired": pairs, "per_family":
                      report["per_family"]}, indent=2))


if __name__ == "__main__":
    main()
