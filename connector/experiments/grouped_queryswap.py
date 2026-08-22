"""Grouped query-swap training data (tp-grouped-queryswap-001).

The training unit is the GROUP: one fact block, three queries, three
different opaque codes. Everything but the query->value mapping is constant
within a group, so gradient pressure can only come from conditioning on the
query. Output type: opaque codes ONLY (no entity-name objective — SFT4's
mistake). Vocab: bank TRAIN pools only. Mix 50% groups / 50% protected
rehearsal from the committed bank train split.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from connector.experiments.capability_bank import OBJECTS, PREFIXES

N_GROUPS = 70


def _code(rng):
    return f"{rng.choice(PREFIXES)}-{rng.randrange(100, 1000)}"


def build(seed: int = 808):
    rng = random.Random(seed)
    items = []
    for gi in range(N_GROUPS):
        k = 2 + (gi % 3)
        ents = rng.sample(OBJECTS, k)
        recs = [{"e": e, "c": _code(rng)} for e in ents]
        rng.shuffle(recs)
        fmt = "prose" if gi % 2 == 0 else "table"
        if fmt == "prose":
            block = "\n".join(f"{r['e'].capitalize()} holds ref {r['c']}." for r in recs)
        else:
            block = "item | ref\n" + "\n".join(
                f"{r['e'].capitalize()} | {r['c']}" for r in recs)
        for r in recs:
            q = f"Return ONLY the ref of {r['e'].capitalize()}."
            items.append({"family": "queryswap_group", "group_id": f"gqs-{gi:03d}",
                          "prompt": f"{block}\n{q}\nAnswer:",
                          "completion": f" {r['c']}.", "gold": r["c"],
                          "answer": r["c"], "protocol": fmt})
    bank = [json.loads(l) for l in
            Path("data/capability_bank/train.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rehearsal = []
    for fam, n in (("single_fact", 40), ("tool_result", 40), ("copy", 40),
                   ("protocol_transfer", 35), ("symbolic_ops", 35)):
        pool = [b for b in bank if b["family"] == fam]
        rehearsal += rng.sample(pool, min(n, len(pool)))
    for it in rehearsal:
        it.setdefault("gold", it.get("answer", ""))
        it.setdefault("protocol", it.get("format", "nl"))
    items = items + rehearsal
    rng.shuffle(items)
    held = items[:40]
    return items[len(held):], held


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/grouped_queryswap")
    a = p.parse_args()
    train, held = build()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("\n".join(json.dumps(x) for x in train), encoding="utf-8")
    (out / "heldout.jsonl").write_text("\n".join(json.dumps(x) for x in held), encoding="utf-8")
    fams = Counter(x["family"] for x in train)
    print(json.dumps({"total": len(train), "per_family": dict(fams),
                      "percentages": {k: f"{v/len(train):.1%}" for k, v in fams.items()}}, indent=2))


if __name__ == "__main__":
    main()
