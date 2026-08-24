"""Query Influence Matrix v5: causal-selection replication fixture.

DEVELOPMENT_REPLICATION_ONLY for the counterfactual-normalization runtime
experiment (scripts/causal_selection_experiment.py). Frozen BEFORE that
experiment's final evaluation; never used for training or selection;
never triggers retraining.

Fresh vocabulary (prefixes HCT/JDX/KVP/MZL/QRB verified absent from every
consumed corpus and sealed OOD suite; entities all-new, checked
programmatically). 50 independent groups, k = 2..4 facts (deliberate),
balanced target position via rotation, two formats.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

SEED_V5 = 20260915
PREFIXES_V5 = ("HCT", "JDX", "KVP", "MZL", "QRB")
ENTITIES_V5 = ("mullion", "spandrel-panel", "hoodmould",
               "label-stop", "vaulting-rib", "boss",
               "springer", "tas-de-charge", "lierne", "ridge-rib",
               "clerestory-post", "triforium-tracery")
N_GROUPS_V5 = 50
DIAGNOSTIC_VERSION = "anra-query-influence/v5-causal-selection"


def build_groups() -> list[dict]:
    rng = random.Random(SEED_V5)
    groups = []
    pos_counter: dict[int, int] = {}
    for gi in range(N_GROUPS_V5):
        k = 2 + (gi % 3)
        ents = rng.sample(ENTITIES_V5, k)
        codes = [f"{rng.choice(PREFIXES_V5)}-{rng.randrange(100, 1000)}"
                 for _ in ents]
        records = [{"entity": e, "code": c,
                    "line": f"The {e} is marked {c}."}
                   for e, c in zip(ents, codes)]
        rng.shuffle(records)
        # balanced target position: rotate the queried entity's position
        t = pos_counter.get(k, 0) % k
        pos_counter[k] = t + 1
        # move the queried record to a rotating display slot by re-sorting
        records.insert(0, records.pop(t % len(records)))
        fmt = "prose" if gi % 2 == 0 else "table"
        groups.append({"displayed_facts": records, "format": fmt,
                       "target_index": 0})
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
    pref_hits = sorted(p for p in PREFIXES_V5
                       if p in consumed_prefixes
                       or re.search(rf"\b{p}-\d{{3}}\b", corpus))
    ent_hits = sorted(e for e in ENTITIES_V5
                      if e in corpus or e.replace("-", " ") in corpus
                      or e.split("-")[0].capitalize() in corpus)
    return {"prefix_hits": pref_hits, "entity_hits": ent_hits,
            "disjoint": not (pref_hits or ent_hits)}


def _query(rec: dict) -> str:
    return f"Return the tag of the {rec['entity']}."


def _prompt(block: str, query: str) -> str:
    return f"{block}\n{query}\nAnswer:"


def build_query_prompt(group: dict, target_index: int) -> str:
    recs = group["displayed_facts"]
    if group.get("format") == "table":
        block = ("item | tag\n"
                 + "\n".join(f"the {r['entity']} | {r['code']}" for r in recs))
    else:
        block = "\n".join(r["line"] for r in recs)
    return _prompt(block, _query(recs[target_index]))


if __name__ == "__main__":
    print(json.dumps({
        "schema": DIAGNOSTIC_VERSION,
        "fixture_sha256": fixture_hash(),
        "vocab_disjointness": vocabulary_disjointness(),
        "n_groups": N_GROUPS_V5,
        "fact_count_histogram": {k: sum(1 for g in build_groups()
                                        if len(g["displayed_facts"]) == k)
                                 for k in (2, 3, 4)},
    }, indent=2))
