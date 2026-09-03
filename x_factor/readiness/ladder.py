"""Binding difficulty ladder B0-B7 (Mission 4/5).

Independently varies facts/format/familiarity to locate the capability
frontier: ROBUST -> PARTIAL -> FLOOR. Cognition work belongs in PARTIAL.

B0 1 fact (realization floor check)
B1 2 facts, familiar grammar
B2 3 facts, familiar grammar
B3 4 facts, familiar grammar (DEV regime)
B4 4 facts, permuted display order
B5 4 facts, paraphrased grammar, same lexicon
B6 4 facts, new lexicon
B7 4 facts, new value format (new prefixes + 4-digit codes)

Deterministic (SHA256 seeds). DEV-compatible surfaces for B0-B4.
"""

from __future__ import annotations

import hashlib
import random

RUNGS = ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7")

DEV_OBJECTS = ("aviary", "barbican", "cloister", "dolmen", "gaol", "impound",
               "lancet", "nave", "oratory", "portcullis")
DEV_PREFIXES = ("AVR", "BQW", "CTY", "DZN", "EKH", "FMP", "GQS", "HUB")
OOD_OBJECTS = ("keep", "jamb", "transept", "apse", "scriptorium",
               "refectory", "belfry", "chancel", "sacristy", "undercroft")
OOD_PREFIXES = ("JKL", "MNP", "QRS", "TVW", "XYZ", "KQR")


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def gen_tasks(rung: str, seed: int, n: int, k_override: int | None = None):
    if rung not in RUNGS:
        raise ValueError(f"unknown rung {rung}")
    new_lex = rung in ("B6", "B7")
    new_val = rung == "B7"
    objs_pool = OOD_OBJECTS if new_lex else DEV_OBJECTS
    prefs = OOD_PREFIXES if new_val else DEV_PREFIXES
    ndig = 4 if new_val else 3
    kmap = {"B0": 1, "B1": 2, "B2": 3, "B3": 4, "B4": 4, "B5": 4, "B6": 4, "B7": 4}
    k = k_override or kmap[rung]
    rng = random.Random(_seed(seed, rung, n))
    out = []
    for i in range(n):
        objs = rng.sample(objs_pool, k)
        codes = [f"{rng.choice(prefs)}-{rng.randrange(10**(ndig-1), 10**ndig)}" for _ in objs]
        tgt = rng.randrange(k)
        if rung == "B5":
            block = "\n".join(f"Ref {c} is kept by the {o}." for o, c in zip(objs, codes))
            q = f"Which ref belongs to the {objs[tgt].capitalize()}? Respond with only the ref."
        else:
            order = list(range(k))
            if rung == "B4":
                random.Random(_seed(seed, rung, i, "perm")).shuffle(order)
            disp = [(objs[j], codes[j]) for j in order]
            block = "\n".join(f"{o.capitalize()} keeps ref {c}." for o, c in disp)
            q = f"Return ONLY the ref of {objs[tgt].capitalize()}."
        out.append({"id": f"{rung.lower()}-{i:03d}", "block": block, "query": q,
                    "prompt": f"{block}\n{q}\nAnswer:", "gold": codes[tgt],
                    "codes": codes, "rung": rung})
    return out


def oracle_prompt(task) -> str:
    return f"{task['block']}\nRecall: {task['gold']}.\n{task['query']}\nAnswer:"
