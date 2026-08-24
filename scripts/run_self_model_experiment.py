"""Canonical self-model experiment runner.

Given checkpoint + fixture module + frozen policy (+ seed), performs:
  fixture loading -> five runtime arms -> ObservedArmState extraction ->
  policy decision BEFORE verifier -> evaluation afterward -> baselines ->
  per-case receipt -> final verdict.

Usage:
    python -m scripts.run_self_model_experiment \
        --fixture connector.experiments.query_influence_v6 \
        --checkpoint checkpoints/anra-v4-20k-sft6-queryswap-replication.pt \
        --expected-checkpoint-sha <sha> --expected-fixture-sha <sha> \
        --policy output/observed_policy_v2.json --out /tmp/result.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")


def git_state() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                         text=True).strip())
    return {"source_commit": commit, "dirty": dirty}


def run(args) -> dict:
    import connector.experiments.causal_selection as cs
    from connector.experiments.observed_self_model import (
        ObservedArmState, AdaptivePolicy, EvaluationOutcome)
    from anra_core.checkpoint import load_core_checkpoint
    from anra_core.tokenizer import V4Tokenizer

    fx = importlib.import_module(args.fixture)
    fixture_sha = fx.fixture_hash()
    if args.expected_fixture_sha:
        assert fixture_sha == args.expected_fixture_sha, \
            f"fixture drifted: {fixture_sha}"
    assert fx.vocabulary_disjointness()["disjoint"]

    model, _, ident = load_core_checkpoint(args.checkpoint,
                                           legacy_unverified=True)
    if args.expected_checkpoint_sha:
        assert ident.parameter_sha256 == args.expected_checkpoint_sha
    tok = V4Tokenizer.load_canonical()

    pol_json = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    policy = AdaptivePolicy(weights=tuple(pol_json["weights"]),
                            bias=pol_json["bias"],
                            threshold=pol_json.get("threshold", 0.5))

    gs = git_state()
    groups = fx.build_groups()
    rows = []
    t0 = time.time()

    for gi, g in enumerate(groups):
        recs = g["displayed_facts"]
        codes = [r["code"] for r in recs]
        prompts = [fx.build_query_prompt(g, qi) for qi in range(len(recs))]

        raw = [[cs.completion_logprob(model, tok, prompts[q],
                                      f" {recs[i]['code']}.")
                for i in range(len(recs))] for q in range(len(recs))]
        for qi in range(len(recs)):
            prompt = prompts[qi]

            adj_scores = []
            for i in range(len(recs)):
                others = [raw[j][i] for j in range(len(recs)) if j != qi]
                adj_scores.append(raw[qi][i] - sum(others) / len(others))
            raw_pick = max(range(len(recs)), key=lambda i: raw[qi][i])
            norm_pick = max(range(len(recs)), key=lambda i: adj_scores[i])

            free_out = cs.free_greedy(model, tok, prompt)
            fc = CODE_RE.findall(free_out)
            free_code = fc[0] if len(fc) == 1 else None
            out_constr, _ = cs.constrained_greedy(model, tok, prompt, codes)

            # ---- OBSERVED STATE (decide here; no gold anywhere) ------
            state = ObservedArmState(
                n_candidates=len(recs),
                format_name=g["format"],
                raw_pick_code=codes[raw_pick],
                norm_pick_code=codes[norm_pick],
                free_out_code=free_code,
                constrained_pick_code=(out_constr.strip().rstrip(".")
                                       or None),
                raw_scores=[raw[qi][i] for i in range(len(recs))],
                norm_scores=[adj_scores[i] for i in range(len(recs))],
            )
            decision = policy.decide(state)

            # ---- EVALUATION (verifier only, after the decision) ------
            gold = recs[qi]["code"]
            outcome = EvaluationOutcome(
                gold_code=gold,
                raw_ok=codes[raw_pick] == gold,
                normalized_ok=codes[norm_pick] == gold,
                constrained_ok=cs._constr_ok(out_constr, gold),
                free_ok=(len(fc) == 1 and fc[0] == gold),
                raw_rank_of_gold=1 + sum(1 for j in range(len(recs))
                                         if raw[qi][j] > raw[qi][qi]),
                adj_rank_of_gold=1 + sum(1 for j in range(len(recs))
                                         if j != qi and adj_scores[j] > adj_scores[qi]),
            )

            rows.append({
                # observed block
                "n_candidates": state.n_candidates,
                "format_name": state.format_name,
                "raw_pick_code": state.raw_pick_code,
                "norm_pick_code": state.norm_pick_code,
                "free_out_code": state.free_out_code,
                "constrained_pick_code": state.constrained_pick_code,
                "raw_scores": state.raw_scores,
                "norm_scores": state.norm_scores,
                "decision": decision,
                "p_normalize": round(policy.prob_normalize(state), 4),
                # evaluator block
                **{k: getattr(outcome, k) for k in
                   ("gold_code", "raw_ok", "normalized_ok", "constrained_ok",
                    "free_ok", "raw_rank_of_gold", "adj_rank_of_gold")},
            })
        print(f"group {gi} done", flush=True)

    n = len(rows)

    def acc(k):
        return sum(1 for r in rows if r[k])

    adaptive_correct = sum(
        1 for r in rows
        if (r["normalized_ok"] if r["decision"] == "NORMALIZE"
            else r["raw_ok"]))
    adaptive_reg = sum(1 for r in rows
                       if r["decision"] == "NORMALIZE"
                       and not r["normalized_ok"] and r["raw_ok"])
    oracle = sum(1 for r in rows if r["raw_ok"] or r["normalized_ok"])
    # observed-only hand rule: normalize when normalized top2 margin
    # dominates raw top2 margin (no gold involved)
    hand = 0
    for r in rows:
        ns = sorted(r["norm_scores"]); rs = sorted(r["raw_scores"])
        use_norm = (ns[-1] - ns[-2]) >= (rs[-1] - rs[-2])
        hand += r["normalized_ok"] if use_norm else r["raw_ok"]

    report = {
        "schema": "anra-self-model-experiment/v2",
        "runner": "scripts/run_self_model_experiment.py",
        "provenance": {
            "source_commit": gs["source_commit"], "dirty": gs["dirty"],
            "checkpoint_path": args.checkpoint,
            "checkpoint_sha256": ident.parameter_sha256,
            "fixture_module": args.fixture,
            "fixture_sha256": fixture_sha,
            "tokenizer": tok.identity(),
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
            "decode_config": {"greedy": True, "max_new_tokens": 10},
            "seed_policy_threshold": policy.threshold,
            "weights_modified": False,
        },
        "n_targets": n,
        "baselines": {
            "ALWAYS_FREE": acc("free_ok"),
            "ALWAYS_RAW": acc("raw_ok"),
            "ALWAYS_CONSTRAINED": acc("constrained_ok"),
            "ALWAYS_NORMALIZED": acc("normalized_ok"),
            "HAND_RULE_observed_margins": hand,
            "ADAPTIVE_observed_policy": adaptive_correct,
            "ORACLE_evaluator_only": oracle,
        },
        "adaptive_regressions_vs_raw": adaptive_reg,
        "always_normalized_regressions_vs_raw":
            sum(1 for r in rows if not r["normalized_ok"] and r["raw_ok"]),
        "fraction_oracle_improvement_over_raw_recovered_by_adaptive":
            round((adaptive_correct - acc("raw_ok"))
                  / max(oracle - acc("raw_ok"), 1), 3),
        "per_item_rows": rows,
        "wall_seconds": round(time.time() - t0, 1),
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--expected-checkpoint-sha", default=None)
    p.add_argument("--expected-fixture-sha", default=None)
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rep = run(a)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({"baselines": rep["baselines"],
                      "adaptive_regressions_vs_raw":
                          rep["adaptive_regressions_vs_raw"]}, indent=2))


if __name__ == "__main__":
    main()
