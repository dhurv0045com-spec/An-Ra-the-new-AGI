"""MIXED-CAUSAL-v6: fresh promotion fixture for self-model v6.

Generated ONLY after v6 freeze (commit f3eef84). Fresh seed, entities,
prefixes. 180 tasks (30 per family × 6). Same causal contract.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

SEED_MC6 = 20261201
PREFIXES_MC6 = ("HQL", "NZR", "SBK", "VXP", "YDM")
ENTITIES_MC6 = ("quatrefoil-panel", "spandrel-arch", "traceried-light",
                "tympanum-slab", "volute-scroll")
COLORS_MC6 = ("russet", "cerulean", "gamboge", "bone-white")

CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")
COSTS = {"NO_CHANGE": 0, "CONSTRAINED": 1, "NORMALIZED": 2}
FAMILIES = ("selection", "realization", "no_intervention",
            "non_candidate", "copy_single", "composition")


def _code(rng):
    return f"{rng.choice(PREFIXES_MC6)}-{rng.randrange(100, 1000)}"


def _query_for(rec):
    return f"Return the tag of the {rec['entity']}."


def _make_task(family, task_id, rng):
    t = {"task_id": task_id, "family": family}

    if family == "non_candidate":
        c = rng.choice(COLORS_MC6)
        ent = rng.choice(ENTITIES_MC6)
        t.update(context=f"The {ent} was painted {c}.",
                 query=f"What color is the {ent}?",
                 candidates=[], gold=c, query_target_index=None,
                 alt_query_targets=[], output_arity=1, fmt="prose")
        return t

    if family == "copy_single":
        c = _code(rng)
        t.update(context=f"The marker reads {c}.",
                 query="Repeat the marker exactly.",
                 candidates=[c], gold=c, query_target_index=None,
                 alt_query_targets=[], output_arity=1, fmt="prose")
        return t

    if family == "composition":
        recs = [{"entity": e, "code": _code(rng)}
                for e in rng.sample(ENTITIES_MC6, 3)]
        block = "\n".join(f"The {r['entity']} is marked {r['code']}."
                          for r in recs)
        a, b = recs[0], recs[1]
        t.update(context=block,
                 query=(f"Return the tag of the {a['entity']} followed by "
                        f"the tag of the {b['entity']}."),
                 candidates=[r["code"] for r in recs],
                 gold=f"{a['code']} {b['code']}",
                 query_target_index=None, alt_query_targets=[],
                 output_arity=2, fmt="prose")
        return t

    n = {"selection": 3, "realization": 3, "no_intervention": 2}[family]
    recs = [{"entity": e, "code": _code(rng)}
            for e in rng.sample(ENTITIES_MC6, n)]
    fmt = rng.choice(["prose", "table"])
    if fmt == "table":
        block = "item | tag\n" + "\n".join(
            f"the {r['entity']} | {r['code']}" for r in recs)
    else:
        block = "\n".join(f"The {r['entity']} is marked {r['code']}."
                          for r in recs)
    qi = rng.randrange(n)
    q = (f"What tag belongs to the {recs[qi]['entity']}?"
         if family == "no_intervention" else _query_for(recs[qi]))
    t.update(context=block, query=q,
             candidates=[r["code"] for r in recs],
             gold=recs[qi]["code"], query_target_index=qi,
             alt_query_targets=[j for j in range(n) if j != qi],
             output_arity=1, fmt=fmt)
    return t


def build_tasks() -> list[dict]:
    rng = random.Random(SEED_MC6)
    tasks = []
    for fam in FAMILIES:
        for j in range(30):
            tasks.append(_make_task(fam, f"{fam}-{j}", rng))
    rng.shuffle(tasks)
    return tasks


def build_prompt(task, query_override=None):
    q = task["query"] if query_override is None else query_override
    return f"{task['context']}\n{q}\nAnswer:"


def counterfactual_queries(task: dict) -> dict[int, str]:
    ents = re.findall(r"[Tt]he ([a-z-]+) is marked", task["context"])
    pairs = re.findall(r"the ([a-z-]+) \| ([A-Z]{3}-\d{3})", task["context"])
    if pairs:
        by_ent = dict(pairs)
        recs = [{"entity": e, "code": by_ent[e]} for e, _ in pairs]
    else:
        codes = CODE_RE.findall(task["context"])
        recs = [{"entity": e, "code": c} for e, c in zip(ents, codes)]
    return {j: build_prompt(task, query_override=_query_for(recs[j]))
            for j in task.get("alt_query_targets", [])}


def verify(task, emitted) -> bool:
    out = emitted.strip()
    if task["output_arity"] == 2:
        parts = CODE_RE.findall(out)
        want = task["gold"].split()
        return len(parts) >= 2 and parts[0] == want[0] and parts[1] == want[1]
    if not task["candidates"]:
        return task["gold"].lower() in out.lower()
    return task["gold"] in out


def applicable_actions(task):
    ncands = len(task["candidates"])
    has_cf = bool(task.get("alt_query_targets"))
    acts = ["NO_CHANGE"]
    if ncands >= 1:
        acts.append("CONSTRAINED")
    if ncands >= 2 and has_cf:
        acts.append("NORMALIZED")
    if task["output_arity"] > 1 and "CONSTRAINED" in acts:
        acts.remove("CONSTRAINED")
    acts.append("ABSTAIN")
    return tuple(acts)


def fixture_hash() -> str:
    return hashlib.sha256(
        json.dumps(build_tasks(), sort_keys=True).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    ts = build_tasks()
    from collections import Counter
    print(json.dumps({
        "schema": "anra-mixed-causal/v6-promotion",
        "fixture_sha256": fixture_hash(),
        "n_tasks": len(ts),
        "family_histogram": dict(Counter(t["family"] for t in ts)),
    }, indent=2))
