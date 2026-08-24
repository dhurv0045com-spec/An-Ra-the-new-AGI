"""MIXED-CAUSAL-v2 FINAL PROMOTION TEST (preregistered).

Frozen policy v3 decides one action per task from observed state only;
verifier scores afterward. Baselines computed from the same full-matrix
data. PRIMARY: adaptive beats every fixed policy on verifier utility.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")


def main() -> None:
    import connector.experiments.mixed_causal_v2 as mc
    from connector.experiments.counterfactual_normalization import (
        normalize_scores, verify_byte_identical_context, argmax)
    from anra_core.checkpoint import load_core_checkpoint
    from anra_core.tokenizer import V4Tokenizer
    sys.path.insert(0, str(ROOT))
    from scripts.run_mixed_causal_matrix import (
        free_greedy, constrained_greedy, completion_logprob)

    FIX = mc.fixture_hash()
    pol = json.loads((ROOT / "output/self_model_v3.json").read_text(encoding="utf-8"))
    lam = pol["utility_rule"]["lambda"]
    models = pol["models"]

    model, _, ident = load_core_checkpoint(
        "checkpoints/anra-v4-20k-sft6-queryswap-replication.pt",
        legacy_unverified=True)
    tok = V4Tokenizer.load_canonical()
    tasks = mc.build_tasks()
    rows = []

    for ti, task in enumerate(tasks):
        applicable = mc.applicable_actions(task)
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

        # ---- observed features (same 12 as v3 training) ----
        def top2(s):
            s = sorted(s); return s[-1] - s[-2] if len(s) >= 2 else 0.0

        def std(s):
            if not s:
                return 0.0
            m = sum(s) / len(s)
            return (sum((x - m) ** 2 for x in s) / len(s)) ** 0.5

        feats = [
            float(len(candidates)), float(task["output_arity"]),
            {"prose": 0, "table": 1}.get(task["fmt"], 2),
            top2(raw_scores or []), top2(norm_scores or []),
            std(raw_scores or []), std(norm_scores or []),
            1.0 if (raw_scores and raw_pick_code == norm_pick_code) else 0.0,
            float(free_code is not None and free_code == raw_pick_code),
            float(free_code is not None and free_code == norm_pick_code),
            float(bool(raw_scores)),
            (1 - sum(1 for x in norm_scores[1:] if x > norm_scores[0])
             / max(len(norm_scores) - 1, 1)) if norm_scores else 0.0,
        ]

        # ---- execute every applicable action ----
        results = {}
        for action in applicable:
            if action == "NO_CHANGE":
                emitted = free_out
            elif action == "ABSTAIN":
                emitted = None
            elif action == "CONSTRAINED":
                emitted, _ = constrained_greedy(model, tok, base_prompt,
                                                candidates)
            else:  # NORMALIZED / NORM_EXACT
                emitted = f" {norm_pick_code}."
            results[action] = {
                "output": (emitted or "").strip()[:48],
                "pass": bool(mc.verify(task, emitted)) if emitted is not None else False,
                "cost": mc.COSTS[action],
            }

        # ---- adaptive v3 decision (BEFORE verifier; features already built) --
        def predict(action):
            if action not in models:
                return 0.0
            m = models[action]
            z = sum(w * x for w, x in zip(m["weights"], feats)) + m["bias"]
            return 1 / (1 + 2.718281828459045 ** (-z))

        utilities = {a: predict(a) - lam * mc.COSTS[a] for a in applicable}
        adaptive = max(utilities, key=utilities.get)

        rows.append({
            "observed": {"n_candidates": len(candidates),
                         "output_arity": task["output_arity"],
                         "format_name": task["fmt"],
                         "applicable_actions": list(applicable)},
            "features": feats,
            "adaptive_action": adaptive,
            "adaptive_utilities": {a: round(u, 4) for a, u in utilities.items()},
            "actions": results,
            "gold_code": task["gold"],          # evaluator-only
            "family_analysis": task["family"],  # evaluator-only
        })
        if ti % 10 == 9:
            print(f"{ti+1}/{len(tasks)}", flush=True)

    # ---- baselines over the same matrix ----
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
        return succ, reg, cost

    always_nc = run_policy(lambda r: "NO_CHANGE")
    cons_rows = [r for r in rows if "CONSTRAINED" in r["actions"]]
    always_c = run_policy(
        lambda r: "CONSTRAINED" if "CONSTRAINED" in r["actions"] else "NO_CHANGE")
    always_n = run_policy(
        lambda r: ("NORMALIZED" if "NORMALIZED" in r["actions"] else "NO_CHANGE"))
    hand = run_policy(lambda r: (
        "NORMALIZED" if "NORMALIZED" in r["actions"] and r.get("norm_scores")
        and (sorted(r["norm_scores"])[-1] - sorted(r["norm_scores"])[-2]) >=
        (sorted(r["raw_scores"])[-1] - sorted(r["raw_scores"])[-2])
        else "NO_CHANGE"))
    random.seed(7)
    rnd = run_policy(lambda r: random.choice(
        [a for a in r["observed"]["applicable_actions"]]))
    adaptive = run_policy(lambda r: r["adaptive_action"])
    oracle = sum(1 for r in rows
                 if any(e["pass"] for e in r["actions"].values()))

    report = {
        "schema": "anra-mixed-causal-v2-promotion/v1",
        "fixture_sha256": FIX,
        "checkpoint_sha256": ident.parameter_sha256,
        "policy_frozen_at_commit": "3123625",
        "n_tasks": n,
        "baselines": {
            "ALWAYS_NO_CHANGE": f"{always_nc[0]}/{n} (reg {always_nc[1]}, cost {always_nc[2]})",
            "ALWAYS_CONSTRAINED_WHEN_APPLICABLE": f"{always_c[0]}/{n} (reg {always_c[1]}, cost {always_c[2]})",
            "ALWAYS_NORMALIZED_WHEN_APPLICABLE": f"{always_n[0]}/{n} (reg {always_n[1]}, cost {always_n[2]})",
            "HAND_RULE_observed_margins": f"{hand[0]}/{n} (reg {hand[1]}, cost {hand[2]})",
            "RANDOM_APPLICABLE": f"{rnd[0]}/{n} (reg {rnd[1]}, cost {rnd[2]})",
            "ADAPTIVE_v3": f"{adaptive[0]}/{n} (reg {adaptive[1]}, cost {adaptive[2]})",
            "ORACLE_evaluator_only": f"{oracle}/{n}",
        },
        "adaptive_successes": adaptive[0], "adaptive_regressions": adaptive[1],
        "adaptive_cost": adaptive[2],
        "per_task_rows": rows,
    }
    (ROOT / "output/mixed_causal_v2_promotion.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["baselines"], indent=2))


if __name__ == "__main__":
    main()
