"""Primitive canaries P0-P4 + B0 role (Mission 8/9).

One prompt format is too fragile to diagnose realization. These canaries
separate token realization / instruction following / copying / single
binding / multi-binding BEFORE binding diagnosis is attempted.

P0 direct copy:       "Repeat exactly: FMP-939." -> "FMP-939"
P1 one-fact lookup:   single entity-value fact + query (== ladder B0 shape)
P2 answer envelope:   explicit format instruction + trivial content
P3 candidate emission: emit one of K listed codes (tests candidate output)
P4 2-choice binding:  two facts, query one (minimal query conditioning)

Rule: PRIMITIVE_CANARY_FAILED if P0 or P1 fail badly (Wilson-hi < 0.50):
binding readouts above are uninterpretable; do NOT call any higher rung
binding-ready without a strong justified explanation recorded in the receipt.

Builders are deterministic; evaluation hook takes caller-supplied strict
scorer so neural execution stays in the gate runner, not here.
"""

from __future__ import annotations

import hashlib
import random

CANARIES = ("P0", "P1", "P2", "P3", "P4")


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def gen_canary(cid: str, seed: int, i: int) -> dict:
    if cid not in CANARIES:
        raise ValueError(f"unknown canary {cid}")
    rng = random.Random(_seed(seed, cid, i))
    code = f"{rng.choice(['FMP', 'EKH', 'CTY'])}-{rng.randrange(100, 1000)}"
    other = f"{rng.choice(['BQW', 'DZN'])}-{rng.randrange(100, 1000)}"
    if cid == "P0":
        prompt, gold = f"Repeat exactly: {code}.\nAnswer:", code
    elif cid == "P1":
        prompt = f"Aviary keeps ref {code}.\nReturn ONLY the ref of Aviary.\nAnswer:"
        gold = code
    elif cid == "P2":
        prompt = (f"Respond in exactly this envelope: [REF={code}]\n"
                  f"Put the code {code} inside the brackets.\nAnswer:")
        gold = code
    elif cid == "P3":
        prompt = (f"Emit exactly one of these codes: {code} / {other}.\n"
                  f"Emit: {code}\nAnswer:")
        gold = code
    else:  # P4
        prompt = (f"Aviary keeps ref {code}.\nDolmen keeps ref {other}.\n"
                  f"Return ONLY the ref of Dolmen.\nAnswer:")
        gold = other
    return {"id": f"{cid.lower()}-{i:03d}", "prompt": prompt, "gold": gold, "canary": cid}


def canary_rule(results: dict[str, dict]) -> dict:
    """results: cid -> {k, n}. Returns pass/fail + PRIMITIVE_CANARY_FAILED flag."""
    from .status import wilson

    flags = {}
    for cid in CANARIES:
        r = results.get(cid)
        if r is None:
            flags[cid] = "MISSING"
        else:
            lo, hi = wilson(r["k"], r["n"])
            flags[cid] = "PASS" if lo >= 0.50 else ("FAIL" if hi < 0.50 else "UNCERTAIN")
    failed = flags.get("P0") == "FAIL" or flags.get("P1") == "FAIL"
    missing = any(v == "MISSING" for v in flags.values())
    return {"per_canary": flags,
            "primitive_canary_failed": failed,
            "verdict": "MISSING_METRICS" if missing else ("PRIMITIVE_CANARY_FAILED" if failed else "CANARIES_OK")}
