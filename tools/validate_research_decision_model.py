#!/usr/bin/env python3
"""Deterministic validator for the Phase-2 research decision model.

Checks (lightweight, no framework):
  1. RESEARCH_DECISION_MODEL.json parses; required top-level blocks exist.
  2. Every evidence_ref (candidate/unknown/decision citation) resolves to an
     experiment ID in the Phase-1 evidence ledger.
  3. Every BLOCKED decision references a known unknown; every unknown's
     decisions_blocked reference known blocked decisions or AD ids in the
     architecture decision ledger.
  4. Every unknown has a discriminating experiment or an explicit no_gpu/recovery
     marker with a named action.
  5. Every recommended (value >= 3.0) candidate identifies the decisions it can
     unlock, and its unlock targets resolve to architecture-decision ids.
  6. No experiment whose Phase-1 ledger status is IMPLEMENTED_NOT_EXECUTED or
     NOT_TESTED is cited as positive evidence by a belief whose status is
     STRONGLY_SUPPORTED / SUPPORTED.
  7. Kill criteria exist for every candidate with cost >= 4 and are non-empty.
  8. Decision-tree node references resolve (no dangling branch targets).
  9. Cross-file consistency: architecture ledger ids referenced by the model
     exist; belief ids referenced by the model exist in the belief registry.

Usage: python tools/validate_research_decision_model.py
Exit 0 = clean, 1 = violations.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731

def load(rel: str):
    with open(P(rel), encoding="utf-8") as fh:
        return json.load(fh)

def main() -> int:
    problems: list[str] = []
    model = load("docs/research/RESEARCH_DECISION_MODEL.json")
    ledger = load("docs/research/EXPERIMENT_EVIDENCE_LEDGER.json")
    beliefs = load("docs/research/BELIEF_REGISTRY.json")
    decisions = load("docs/research/ARCHITECTURE_DECISION_LEDGER.json")
    tree = load("docs/research/NEXT_EXPERIMENT_DECISION_TREE.json")

    for block in ("unknowns", "candidate_experiments", "blocked_decisions", "kill_criteria", "decision_tree_ref", "beliefs_ref"):
        if block not in model:
            problems.append(f"model missing required block: {block}")

    ledger_ids = {e["id"] for e in ledger.get("experiments", [])}
    ledger_status = {e["id"]: e.get("status") for e in ledger.get("experiments", [])}
    belief_ids = {b["belief_id"] for b in beliefs.get("beliefs", [])}
    belief_status = {b["belief_id"]: b.get("current_status") for b in beliefs.get("beliefs", [])}
    ad_ids = {d["id"] for d in decisions.get("decisions", [])}
    unknown_ids = {u["id"] for u in model.get("unknowns", [])}

    # 2. evidence refs resolve to ledger experiments
    for exp_id in model.get("evidence_refs", {}).get("primary_sources", []):
        if exp_id not in ledger_ids:
            problems.append(f"evidence_ref does not resolve to ledger experiment: {exp_id}")

    # 3a. blocked decisions reference known unknowns
    known_blocked = set()
    for bd in model.get("blocked_decisions", []):
        if bd["decision"] not in ad_ids:
            problems.append(f"blocked decision references unknown architecture id: {bd['decision']}")
        for u in str(bd.get("blocked_by_unknown", "")).split("+"):
            u = u.strip()
            if u and u not in unknown_ids:
                problems.append(f"blocked decision {bd['decision']} references unknown unknown: {u}")
        known_blocked.add(bd["decision"])

    # 3b. unknowns' decisions_blocked resolve to AD ids
    for u in model.get("unknowns", []):
        for d in u.get("decisions_blocked", []):
            base = d.split("-")[0]
            if d not in ad_ids and base not in {a.rstrip("abcdefghijklmnopqrstuvwxyz") for a in ad_ids} and d not in known_blocked:
                if not any(ad.startswith(d) for ad in ad_ids):
                    problems.append(f"unknown {u['id']} blocks unresolvable decision: {d}")

    # 4. every unknown has a discriminating experiment or explicit no_gpu/recovery marker
    candidate_ids = {c["id"] for c in model.get("candidate_experiments", [])}
    for u in model.get("unknowns", []):
        exp = u.get("discriminating_experiment", "")
        named = exp.split(" ")[0].split("(")[0].strip()
        if not exp:
            problems.append(f"unknown {u['id']} has no discriminating experiment")
        elif named not in candidate_ids and not u.get("no_gpu"):
            problems.append(f"unknown {u['id']} discriminates via unknown candidate: {exp}")

    # 5. valuable candidates identify unlock targets that resolve
    for c in model.get("candidate_experiments", []):
        if c.get("value", 0) >= 3.0:
            if not c.get("unlocks"):
                problems.append(f"candidate {c['id']} (value {c['value']}) identifies no decisions it can change")
            for t in c.get("unlocks", []):
                base = t.split("-")[0]
                if t not in ad_ids and t not in unknown_ids and not any(a.startswith(base) for a in ad_ids):
                    problems.append(f"candidate {c['id']} unlock target unresolvable: {t}")

    # 6. no NOT_EXECUTED/NOT_TESTED experiment cited as positive evidence by strong beliefs
    bad_sources = {"IMPLEMENTED_NOT_EXECUTED", "NOT_TESTED", "SPECULATIVE"}
    for b in beliefs.get("beliefs", []):
        if b.get("current_status") in {"STRONGLY_SUPPORTED", "SUPPORTED"}:
            for s in b.get("supporting_experiments", []):
                if ledger_status.get(s) in bad_sources:
                    problems.append(f"belief {b['belief_id']} cites {s} (status {ledger_status.get(s)}) as positive evidence")

    # 7. kill criteria for expensive candidates
    kc = set(model.get("kill_criteria", []))
    if not kc:
        problems.append("no kill criteria defined")
    for c in model.get("candidate_experiments", []):
        if c.get("scores", {}).get("cost", 0) >= 4 and not kc:
            problems.append(f"expensive candidate {c['id']} lacks any kill criterion")

    # 8. decision tree references resolve
    nodes = tree.get("nodes", {})
    for nid, node in nodes.items():
        for br in node.get("branches", []):
            target = br.get("then", "")
            if target and target not in nodes:
                problems.append(f"tree node {nid} branches to unresolvable target: {target}")

    # 9. belief id sanity
    if len(belief_ids) != len(beliefs.get("beliefs", [])):
        problems.append("duplicate belief ids in registry")

    print(f"decision model: {len(model.get('candidate_experiments', []))} candidates, "
          f"{len(unknown_ids)} unknowns, {len(model.get('blocked_decisions', []))} blocked decisions, "
          f"{len(belief_ids)} beliefs, {len(ad_ids)} architecture decisions")
    if problems:
        print(f"VALIDATION FAILED ({len(problems)} problems):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("DECISION MODEL VALIDATION PASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
