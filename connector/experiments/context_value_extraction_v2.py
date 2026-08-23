"""Context value extraction v2: LARGER development metric (P10, future use).

Built per the evidence-repair mission's P10 spec:
  - 24 items (up from 12) => finer-grained floor with more headroom
  - fresh code prefixes AND fully disjoint entity vocabulary
    (verified programmatically against everything consumed so far)
  - balanced fact counts (2/3/4/5 facts, 6 items each)
  - three formats rotating (prose / table / kv)
  - same behavioral definition as v1: "return ANY ONE supplied value",
    strict single-code output that must be one of the supplied codes

ROLE: DEVELOPMENT_ONLY. This fixture did NOT exist when SFT6 was gated —
it must never be used to retroactively re-judge SFT6's PR6. Its purpose is
to give FUTURE preregistrations a less fragile extraction floor and to
establish fresh parent/SFT6 baselines for those proposals.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")
SEED_V2 = 20260909
# Fresh prefixes: disjoint from every prefix used in bank/QIM/gqs/emd/cve-v1
# (AVR BQW CTY DZN EKH FMP GQS HUB IRB JSM KTN LVD HGR JPL KSN MBT NWD WKC
#  XDN YRM ZTS BVF CGJ FRC LXM PVG TQH VZB MRC QDX).
PREFIXES_V2 = ("NRW", "SXP", "TKD", "YQF")
# Fresh entities (architecture/metalwork words not used by any prior fixture;
# v1's list contained 'jamb' which collides with grouped-queryswap data).
ENTITIES_V2 = ("baluster", "corbel", "dentil", "architrave", "cartouche",
               "patera", "volute", "scrolled-bracket", "keystone-plate",
               "typanum-bar", "metope-panel", "guttae-strip")
N_ITEMS = 24


def _code(rng: random.Random) -> str:
    return f"{rng.choice(PREFIXES_V2)}-{rng.randrange(100, 1000)}"


def build_items() -> list[dict]:
    rng = random.Random(SEED_V2)
    items = []
    for i in range(N_ITEMS):
        k = 2 + (i % 4)                      # 2,3,4,5 facts, balanced
        ents = rng.sample(ENTITIES_V2, k)
        codes = [_code(rng) for _ in ents]
        fmt = ("prose", "table", "kv")[i % 3]
        if fmt == "prose":
            block = "\n".join(f"{e.capitalize()} carries marker {c}."
                              for e, c in zip(ents, codes))
        elif fmt == "table":
            block = ("name | marker\n"
                     + "\n".join(f"{e.capitalize()} | {c}"
                                 for e, c in zip(ents, codes)))
        else:
            block = "\n".join(f"{e.capitalize()} :: {c}"
                              for e, c in zip(ents, codes))
        items.append({
            "id": f"cve2-{i:02d}", "n_facts": k, "format": fmt,
            "prompt": (f"{block}\nReturn ANY ONE of the supplied markers.\nAnswer:"),
            "valid_codes": codes,
        })
    return items


def fixture_hash() -> str:
    return hashlib.sha256(
        json.dumps(build_items(), sort_keys=True).encode("utf-8")).hexdigest()


def vocabulary_disjointness() -> dict:
    """TRUE entity+prefix disjointness proof (not just a claim): every
    v2 entity and prefix is checked against ALL consumed corpora (.jsonl
    data files AND sealed OOD item files) and against the entity/prefix
    constants of every prior generator."""
    consumed_prefixes = set()
    corpus_blob = ""
    consumed_paths = [
        "data/grouped_queryswap/train.jsonl",
        "data/grouped_queryswap/heldout.jsonl",
        "data/capability_bank/train.jsonl",
        "data/capability_bank/dev.jsonl",
        "connector/experiments/ood_battery/items.json",
        "connector/experiments/ood2_battery/items.json",
        "connector/experiments/ood3_battery/items.json",
        "connector/experiments/ood4_battery/items.json",
    ]
    for p in consumed_paths:
        f = Path(p)
        if f.exists():
            text = f.read_text(encoding="utf-8")
            corpus_blob += text
            consumed_prefixes |= set(re.findall(r"\b([A-Z]{3})-\d{3}\b", text))
    import connector.experiments.query_influence as qi
    import connector.experiments.entity_matching_diagnostic as emd
    from connector.experiments.capability_bank import (
        PREFIXES as CB_P, DEV_PREFIXES as CB_DP,
        OBJECTS, DEV_OBJECTS, WORDS, DEV_WORDS)
    consumed_prefixes |= set(qi.PREFIXES) | set(emd.PREFIXES) | set(CB_P) | set(CB_DP)
    overlaps_p = sorted(set(PREFIXES_V2) & consumed_prefixes)
    # prefix collisions in actual corpus text (belt and braces)
    text_prefix_hits = [p for p in PREFIXES_V2
                        if re.search(rf"\b{p}-\d{{3}}\b", corpus_blob)]
    # TRUE entity check: against constants of every prior generator, plus
    # literal occurrence in any consumed corpus text
    prior_entities = (set(OBJECTS) | set(DEV_OBJECTS) | set(WORDS) |
                      set(DEV_WORDS) | set(qi.ENTITIES) | set(emd.ENTITIES) |
                      {"bezel", "ferrule", "gudgeon", "hasp", "jamb",
                       "lintel", "muntin", "newel", "ogee", "quoins",
                       "spandrel", "voussoir"})
    entity_constant_overlaps = sorted(set(ENTITIES_V2) & prior_entities)
    entity_corpus_hits = sorted(
        e for e in ENTITIES_V2
        if e in corpus_blob or e.capitalize() in corpus_blob)
    return {
        "prefix_constant_overlaps": overlaps_p,
        "prefix_corpus_hits": text_prefix_hits,
        "entity_constant_overlaps": entity_constant_overlaps,
        "entity_corpus_hits": entity_corpus_hits,
        "disjoint": not (overlaps_p or text_prefix_hits or
                         entity_constant_overlaps or entity_corpus_hits),
        "checked_files": [p for p in consumed_paths if Path(p).exists()],
        "note": ("v1's defect was claiming disjointness without checking: "
                 "'jamb' appeared in grouped-queryswap rows. v2 verifies "
                 "entities AND prefixes against constants AND corpora."),
    }


if __name__ == "__main__":
    items = build_items()
    print(json.dumps({
        "schema": "anra-context-value-extraction/v2",
        "role": "DEVELOPMENT_ONLY_FUTURE_USE",
        "fixture_sha256": fixture_hash(),
        "n_items": len(items),
        "fact_count_histogram": {k: sum(1 for x in items if x["n_facts"] == k)
                                 for k in (2, 3, 4, 5)},
        "format_histogram": {f: sum(1 for x in items if x["format"] == f)
                             for f in ("prose", "table", "kv")},
        "vocab_disjointness": vocabulary_disjointness(),
    }, indent=2))
