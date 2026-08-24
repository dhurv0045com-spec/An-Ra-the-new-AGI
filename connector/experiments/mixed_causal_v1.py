"""MIXED-CAUSAL-v1: mixed-failure development environment (v2, corrected).

Six interleaved task families. Candidate tasks expose OBSERVED structure
sufficient to construct legal counterfactual queries: facts block,
candidate values, query target index, alternative query targets, format.
Gold stays evaluator-only.

Families:
  selection      multi-fact candidate task (normalization may help)
  realization    candidate preference right; free decode drifts (constrained)
  no_intervention unambiguous phrasing; raw usually already correct
  non_candidate  open answer; candidate actions NOT_APPLICABLE
  copy_single    single candidate; normalization NOT_APPLICABLE (<2 queries)
  composition    two codes requested; single-slot emissions incompatible

Applicability masks derive ONLY from observable structure
(n_candidates, output arity), never from family label or correctness.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

SEED_MC1 = 20260925
CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")
PREFIXES_MC = ("GKT", "LWZ", "NQD", "PVH", "TXC")
ENTITIES_MC = ("bell-cote", "corbel-table", "diaper-work", "extrados",
               "flying-arch", "gorgerin", "hypocaust-tile")
COLORS = ("crimson", "cobalt", "verdant", "ochre", "ivory")

ACTIONS = ("NO_CHANGE", "CONSTRAINED", "NORMALIZED", "NORM_EXACT", "ABSTAIN")
COSTS = {"NO_CHANGE": 0, "CONSTRAINED": 1, "NORMALIZED": 2,
         "NORM_EXACT": 3, "ABSTAIN": 0}
FAMILIES = ("selection", "realization", "no_intervention",
            "non_candidate", "copy_single", "composition")


def _code(rng):
    return f"{rng.choice(PREFIXES_MC)}-{rng.randrange(100, 1000)}"


def _query_for(rec):
    return f"Return the tag of the {rec['entity']}."


def _make_task(family, task_id, rng) -> dict:
    t = {"task_id": task_id, "family": family}   # family = ANALYSIS ONLY

    if family == "non_candidate":
        c = rng.choice(COLORS)
        ent = rng.choice(ENTITIES_MC)
        t.update(
            context=f"The {ent} was painted {c}.",
            query=f"What color is the {ent}?",
            candidates=[], gold=c, query_target_index=None,
            alt_query_targets=[], output_arity=1, fmt="prose")
        return t

    if family == "copy_single":
        c = _code(rng)
        t.update(
            context=f"The marker reads {c}.",
            query="Repeat the marker exactly.",
            candidates=[c], gold=c, query_target_index=None,
            alt_query_targets=[], output_arity=1, fmt="prose")
        return t

    if family == "composition":
        recs = [{"entity": e, "code": _code(rng)}
                for e in rng.sample(ENTITIES_MC, 3)]
        block = "\n".join(f"The {r['entity']} is marked {r['code']}."
                          for r in recs)
        a, b = recs[0], recs[1]
        t.update(
            context=block,
            query=(f"Return the tag of the {a['entity']} followed by "
                   f"the tag of the {b['entity']}."),
            candidates=[r["code"] for r in recs],
            gold=f"{a['code']} {b['code']}",
            query_target_index=None,          # composite query: no single target
            alt_query_targets=[],
            output_arity=2,                   # observable: two slots requested
            fmt="prose")
        return t

    # candidate-selection families with true cf-query structure
    n = {"selection": 3, "realization": 3, "no_intervention": 2}[family]
    recs = [{"entity": e, "code": _code(rng)}
            for e in rng.sample(ENTITIES_MC, n)]
    fmt = rng.choice(["prose", "table"])
    if fmt == "table":
        block = "item | tag\n" + "\n".join(
            f"the {r['entity']} | {r['code']}" for r in recs)
    else:
        block = "\n".join(f"The {r['entity']} is marked {r['code']}."
                          for r in recs)
    qi = rng.randrange(n)
    if family == "no_intervention":
        query = f"What tag belongs to the {recs[qi]['entity']}?"
    else:
        query = _query_for(recs[qi])
    t.update(
        context=block,
        query=query,
        candidates=[r["code"] for r in recs],
        gold=recs[qi]["code"],
        query_target_index=qi,
        alt_query_targets=[j for j in range(n) if j != qi],
        output_arity=1,
        fmt=fmt)
    return t


def build_tasks() -> list[dict]:
    """60 tasks: 10 per family, round-robin interleaved then shuffled."""
    rng = random.Random(SEED_MC1)
    tasks = []
    for fam in FAMILIES:
        for j in range(10):
            tasks.append(_make_task(fam, f"{fam}-{j}", rng))
    rng.shuffle(tasks)
    return tasks


def build_prompt(task: dict, query_override: str | None = None) -> str:
    q = task["query"] if query_override is None else query_override
    return f"{task['context']}\n{q}\nAnswer:"


def counterfactual_queries(task: dict) -> dict[int, str]:
    """Legal counterfactual prompts for candidate tasks with >=2 targets.

    Only the query line changes; the context is byte-identical by
    construction (same task dict, different entity's question).
    """
    out = {}
    base_recs = _entities_in_order(task)
    for j in task.get("alt_query_targets", []):
        q = _query_for(base_recs[j])
        out[j] = build_prompt(task, query_override=q)
    return out


_ENT_ORDER_CACHE: dict[str, list] = {}


def _entities_in_order(task: dict) -> list:
    """Recover the fact records in candidate order (observed info only)."""
    tid = task["task_id"]
    if tid not in _ENT_ORDER_CACHE:
        ents = re.findall(r"[Tt]he ([a-z-]+) (?:is marked|was painted|\|)",
                          task["context"])
        pairs = re.findall(r"the ([a-z-]+) \| ([A-Z]{3}-\d{3})", task["context"])
        if pairs:
            by_ent = dict(pairs)
            recs = [{"entity": e, "code": by_ent[e]} for e, _ in pairs]
        else:
            codes = CODE_RE.findall(task["context"])
            recs = [{"entity": e, "code": c} for e, c in zip(ents, codes)]
        _ENT_ORDER_CACHE[tid] = recs
    return _ENT_ORDER_CACHE[tid]


def verify(task: dict, emitted: str) -> bool:
    """Verifier: the ONLY success authority."""
    out = emitted.strip()
    if task["output_arity"] == 2:
        parts = CODE_RE.findall(out)
        want = task["gold"].split()
        return len(parts) >= 2 and parts[0] == want[0] and parts[1] == want[1]
    if not task["candidates"]:
        return task["gold"].lower() in out.lower()
    return task["gold"] in out


def applicable_actions(task: dict) -> tuple[str, ...]:
    """Observable-structure-only action mask.

    - no candidates: candidate interventions unavailable
    - 1 candidate or no alternative query targets: NORMALIZED/NORM_EXACT
      unavailable (counterfactual set empty)
    - output_arity > 1: single-slot exact emission (NORM_EXACT) and
      CONSTRAINED (single-code constraint) are structurally incompatible
    """
    ncands = len(task["candidates"])
    has_cf = bool(task.get("alt_query_targets"))
    acts = ["NO_CHANGE"]
    if ncands >= 1:
        acts.append("CONSTRAINED")
    if ncands >= 2 and has_cf:
        acts.extend(["NORMALIZED", "NORM_EXACT"])
    if task["output_arity"] > 1:
        # multi-slot request: single-code constrained/exact cannot satisfy
        for a in ("CONSTRAINED", "NORM_EXACT"):
            if a in acts:
                acts.remove(a)
    acts.append("ABSTAIN")
    return tuple(acts)


def fixture_hash() -> str:
    return hashlib.sha256(
        json.dumps(build_tasks(), sort_keys=True).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    ts = build_tasks()
    from collections import Counter
    print(json.dumps({
        "schema": "anra-mixed-causal/v2",
        "fixture_sha256": fixture_hash(),
        "n_tasks": len(ts),
        "family_histogram": dict(Counter(t["family"] for t in ts)),
        "applicability_example": {
            "selection": applicable_actions(ts[[t["family"] for t in ts].index("selection")]),
            "copy_single": applicable_actions(ts[[t["family"] for t in ts].index("copy_single")]),
            "composition": applicable_actions(ts[[t["family"] for t in ts].index("composition")]),
            "non_candidate": applicable_actions(ts[[t["family"] for t in ts].index("non_candidate")]),
        },
    }, indent=2))
