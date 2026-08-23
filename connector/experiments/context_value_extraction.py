"""Context value extraction: a DEVELOPMENT capability metric (P7).

SFT4's lesson: query-name behavior improved while generic any-fact
extraction collapsed 7/10 -> 2/10. That primitive is a BEHAVIORAL
capability, not a cognitive cause, and it is now under explicit
protection: future children may not "gain query conditioning" by
destroying generic context-value access.

Metric: multi-fact context, instruction asks for ANY one supplied opaque
value. Strict single-code output, must be one of the supplied codes.

Protection rule (parent-relative, tolerance pre-registered BEFORE the
replication child exists):

    child_extraction >= parent_extraction - EXTRACTION_TOLERANCE

This module never claims cognitive causality; it is a dev gate only.
"""

from __future__ import annotations

import hashlib
import random
import re

CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")
SEED = 20260905
# Pre-registered tolerance (recorded in the replication proposal BEFORE
# training): the child may regress at most 1 item out of 12 vs parent.
EXTRACTION_TOLERANCE = 0.085
N_ITEMS = 12

# Vocabulary note (P9, corrected): the OPAQUE CODE PREFIXES (MRC/QDX) are
# disjoint from the bank, QIM-v2/v3, and grouped-queryswap data. The ENTITY
# vocabulary is NOT guaranteed fully disjoint — e.g. "jamb" also appears in
# grouped-queryswap rows. This module is a DEVELOPMENT retention metric;
# entity overlap does not affect its validity as a gate, but the earlier
# comment claiming full disjointness was inaccurate. Any future fixture
# version must build a fully disjoint entity list WITHOUT retroactively
# changing what the SFT6 gate measured.
_PREFIXES = ("MRC", "QDX")
_ENTITIES = ("bezel", "ferrule", "gudgeon", "hasp", "jamb", "lintel",
             "muntin", "newel", "ogee", "quoins", "spandrel", "voussoir")


def _code(rng: random.Random) -> str:
    return f"{rng.choice(_PREFIXES)}-{rng.randrange(100, 1000)}"


def fixture_hash() -> str:
    return hashlib.sha256(
        json_dumps_sorted(build_items()).encode("utf-8")).hexdigest()


def json_dumps_sorted(items):
    import json
    return json.dumps(items, sort_keys=True)


def build_items() -> list[dict]:
    rng = random.Random(SEED)
    items = []
    for i in range(N_ITEMS):
        k = 2 + (i % 4)                      # 2..4 facts
        ents = rng.sample(_ENTITIES, k)
        codes = [_code(rng) for _ in ents]
        fmt = ("prose", "table")[i % 2]
        if fmt == "prose":
            block = "\n".join(f"{e.capitalize()} carries marker {c}."
                              for e, c in zip(ents, codes))
        else:
            block = ("name | marker\n"
                     + "\n".join(f"{e.capitalize()} | {c}"
                                 for e, c in zip(ents, codes)))
        items.append({
            "id": f"cve-{i:02d}", "n_facts": k, "format": fmt,
            "prompt": (f"{block}\nReturn ANY ONE of the supplied markers.\nAnswer:"),
            "valid_codes": codes,
        })
    return items


def evaluate(model, tok, device: str | None = None) -> dict:
    """Greedy decode each item; pass iff exactly one code appears and it is
    one of the supplied values."""
    from training.sft_context_binding import greedy_decode
    items = build_items()
    hits = []
    for it in items:
        out = greedy_decode(model, tok, it["prompt"], max_new_tokens=10)
        cands = CODE_RE.findall(out)
        hits.append(bool(len(cands) == 1 and cands[0] in it["valid_codes"]))
    n_pass = sum(1 for h in hits if h)
    return {
        "schema": "anra-context-value-extraction/v1",
        "fixture_sha256": fixture_hash(),
        "n_items": len(items),
        "passed": f"{n_pass}/{len(items)}",
        "fraction": round(n_pass / len(items), 4),
        "tolerance_vs_parent": EXTRACTION_TOLERANCE,
        "per_item": [bool(x) for x in hits],
    }


def extraction_floor_ok(parent_fraction: float, child_fraction: float) -> bool:
    """PARENT-RELATIVE floor: child >= parent - EXTRACTION_TOLERANCE."""
    return child_fraction >= parent_fraction - EXTRACTION_TOLERANCE
