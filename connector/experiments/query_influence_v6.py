"""Query Influence Matrix v6: self-model transfer test fixture.

DEVELOPMENT_REPLICATION_ONLY for the observed-only adaptive-policy
evaluation (connector/experiments/observed_self_model.py). Frozen BEFORE
the policy's final evaluation; never used for training the policy.

Fresh vocabulary (prefixes RTN/VXW/YZK/BQP/DMH verified absent from every
consumed corpus and sealed OOD suite; entities all-new, checked
programmatically). 60 independent groups, k = 2..5 facts, three formats.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

SEED_V6 = 20260920
PREFIXES_V6 = ("RTN", "VXW", "YZK", "BQP", "DMH")
ENTITIES_V6 = ("claxon-post", "fenestella", "girder-rib", "hexafoil",
               "impost", "keystone-palmette", "lucet-knot", "mouchette",
               "nailhead-band", "ogee-arch")
N_GROUPS_V6 = 60
DIAGNOSTIC_VERSION = "anra-query-influence/v6-self-model-transfer"


def build_groups() -> list[dict]:
    rng = random.Random(SEED_V6)
    groups = []
    for gi in range(N_GROUPS_V6):
        k = 2 + (gi % 4)          # k in {2,3,4,5}
        ents = rng.sample(ENTITIES_V6, min(k, len(ENTITIES_V6)))
        codes = [f"{rng.choice(PREFIXES_V6)}-{rng.randrange(100, 1000)}"
                 for _ in ents]
        records = [{"entity": e, "code": c,
                    "line": f"The {e} is marked {c}."}
                   for e, c in zip(ents, codes)]
        rng.shuffle(records)
        fmt = ("prose", "table", "list")[gi % 3]
        groups.append({"displayed_facts": records, "format": fmt})
    return groups


def fixture_hash() -> str:
    return hashlib.sha256(
        json.dumps(build_groups(), sort_keys=True).encode("utf-8")).hexdigest()


def vocabulary_disjointness() -> dict:
    consumed_prefixes = set()
    corpus = ""
    for p in ("data/grouped_queryswap/train.jsonl",
              "data/grouped_queryswap/heldout.jsonl",
              "data/capability_bank/train.jsonl",
              "data/capability_bank/dev.jsonl",
              "connector/experiments/ood_battery/items.json",
              "connector/experiments/ood2_battery/items.json",
              "connector/experiments/ood3_battery/items.json",
              "connector/experiments/ood4_battery/items.json"):
        f = Path(p)
        if f.exists():
            text = f.read_text(encoding="utf-8")
            corpus += text
            consumed_prefixes |= set(re.findall(r"\b([A-Z]{3})-\d{3}\b", text))
    pref_hits = sorted(p for p in PREFIXES_V6
                       if p in consumed_prefixes
                       or re.search(rf"\b{p}-\d{{3}}\b", corpus))
    ent_hits = sorted(e for e in ENTITIES_V6
                      if e in corpus or e.replace("-", " ") in corpus)
    return {"prefix_hits": pref_hits, "entity_hits": ent_hits,
            "disjoint": not (pref_hits or ent_hits)}


def _query(rec: dict) -> str:
    return f"Return the tag of the {rec['entity']}."


def build_query_prompt(group: dict, target_index: int) -> str:
    recs = group["displayed_facts"]
    fmt = group.get("format")
    if fmt == "table":
        block = ("item | tag\n"
                 + "\n".join(f"the {r['entity']} | {r['code']}" for r in recs))
    elif fmt == "list":
        block = "\n".join(f"- {r['code']}: {r['entity']}" for r in recs)
    else:
        block = "\n".join(r["line"] for r in recs)
    return f"{block}\n{_query(recs[target_index])}\nAnswer:"


if __name__ == "__main__":
    print(json.dumps({
        "schema": DIAGNOSTIC_VERSION,
        "fixture_sha256": fixture_hash(),
        "vocab_disjointness": vocabulary_disjointness(),
        "n_groups": N_GROUPS_V6,
    }, indent=2))
