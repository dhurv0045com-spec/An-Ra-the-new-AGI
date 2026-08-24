"""MIXED-CAUSAL-v1: mixed-failure development environment.

Six interleaved task families where NO single intervention dominates:

  SELECTION      multi-fact candidate tasks (normalization can help)
  REALIZATION    candidate preference exists; free decode fails (constrained helps)
  NO_INTERVENTION tasks normal decoding already solves (interventions regress)
  NON_CANDIDATE  open answer; candidate restriction is inappropriate
  COPY_SINGLE    trivial copy/single-fact; extra intervention = pure cost
  COMPOSITION    answer derives from MULTIPLE values; single-pick arms fail

Verifiers per family:
  SELECTION/REALIZATION/NO_INTERVENTION/COPY_SINGLE: emitted code == gold
  NON_CANDIDATE: emitted text contains the gold open value (e.g. a color word)
  COMPOSITION: emitted text contains BOTH required codes in order
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

SEED_MC1 = 20260925
CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")
FAMILIES = ("selection", "realization", "no_intervention",
            "non_candidate", "copy_single", "composition")

# vocabulary shared across families (fresh vs all consumed corpora)
PREFIXES_MC = ("GKT", "LWZ", "NQD", "PVH", "TXC")
ENTITIES_MC = ("bell-cote", "corbel-table", "diaper-work", "extrados",
               "flying-arch", "gorgerin", "hypocaust-tile")


def _code(rng):
    return f"{rng.choice(PREFIXES_MC)}-{rng.randrange(100, 1000)}"


def build_tasks() -> list[dict]:
    """60 tasks, 10 per family, interleaved by round-robin."""
    rng = random.Random(SEED_MC1)
    tasks = []
    for fam in FAMILIES:
        for j in range(10):
            tasks.append(_make_task(fam, f"{fam}-{j}", rng))
    rng.shuffle(tasks)
    return tasks


def _facts_block(rng, n, fmt="prose"):
    ents = rng.sample(ENTITIES_MC, n)
    recs = [{"entity": e, "code": _code(rng)} for e in ents]
    if fmt == "table":
        block = "item | tag\n" + "\n".join(
            f"the {r['entity']} | {r['code']}" for r in recs)
    else:
        block = "\n".join(f"The {r['entity']} is marked {r['code']}."
                          for r in recs)
    return recs, block


def _make_task(family: str, task_id: str, rng) -> dict:
    t = {"task_id": task_id, "family": family}

    if family == "selection":
        recs, block = _facts_block(rng, 3)
        target = rng.choice(recs)
        t.update(prompt=f"{block}\nReturn the tag of the {target['entity']}.\nAnswer:",
                 candidates=[r["code"] for r in recs], gold=target["code"])

    elif family == "realization":
        recs, block = _facts_block(rng, 2)
        # gold is the FIRST fact's code with a distinctive high prior setup:
        # the query names it directly; free decode tends to drift to the
        # more frequent-format second entity.
        target = recs[0]
        t.update(prompt=f"{block}\nReturn the tag of the {target['entity']}.\nAnswer:",
                 candidates=[r["code"] for r in recs], gold=target["code"])

    elif family == "no_intervention":
        recs, block = _facts_block(rng, 2)
        target = recs[0]
        # unambiguous phrasing; raw greedy typically already correct
        t.update(prompt=f"{block}\nWhat tag belongs to the {target['entity']}?\nAnswer:",
                 candidates=[r["code"] for r in recs], gold=target["code"])

    elif family == "non_candidate":
        colors = ["crimson", "cobalt", "verdant", "ochre", "ivory"]
        c = rng.choice(colors)
        ent = rng.choice(ENTITIES_MC)
        t.update(prompt=f"The {ent} was painted {c}.\nWhat color is the {ent}?\nAnswer:",
                 candidates=[], gold=c)

    elif family == "copy_single":
        c = _code(rng)
        t.update(prompt=f"The marker reads {c}.\nRepeat the marker exactly.\nAnswer:",
                 candidates=[c], gold=c)

    elif family == "composition":
        recs, block = _facts_block(rng, 3)
        a, b = recs[0], recs[1]
        t.update(prompt=(f"{block}\nReturn the tag of the {a['entity']} "
                         f"followed by the tag of the {b['entity']}.\nAnswer:"),
                 candidates=[r["code"] for r in recs],
                 gold=f"{a['code']} {b['code']}")
    return t


def verify(task: dict, emitted: str) -> bool:
    """Verifier: the ONLY success authority."""
    out = emitted.strip()
    if task["family"] == "composition":
        parts = CODE_RE.findall(out)
        want = task["gold"].split()
        return len(parts) >= 2 and parts[0] == want[0] and parts[1] == want[1]
    if task["family"] == "non_candidate":
        return task["gold"].lower() in out.lower()
    return task["gold"] in out


def fixture_hash() -> str:
    return hashlib.sha256(
        json.dumps(build_tasks(), sort_keys=True).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    ts = build_tasks()
    from collections import Counter
    print(json.dumps({
        "schema": "anra-mixed-causal/v1",
        "fixture_sha256": fixture_hash(),
        "n_tasks": len(ts),
        "family_histogram": dict(Counter(t["family"] for t in ts)),
    }, indent=2))
