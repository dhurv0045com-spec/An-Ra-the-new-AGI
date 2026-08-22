"""Query-focused corrective curriculum: attack recency-biased conditioning.

Diagnosis (entity_matching_diagnostic, 2026-08-23, both checkpoints):
  P0 query-entity recognition 0/10; query-swap output change 0-2/18;
  any-fact extraction 5-7/10; WRONG_FACT_VALUE dominates; entity-match
  HINT rescues 0-1/10 while value-supply rescues 6-7/10.
=> The model emits salient context values but is barely conditioned on the
   query; attention is recency-biased (bindings form near the answer marker).

Minimal intervention: train the query->binding association directly, with
the QUERY and TARGET FACT placed at systematically varied distances from the
answer marker, plus explicit query-recognition items. Vocabularies come from
the capability-bank TRAIN pools only (diagnostic vocab stays unseen).

Mix (exact, emitted in composition.json):
  40% targeted   multi-fact, query position rotated (front/mid/end), k=2-4
  20% query_rec  output the queried entity NAME itself (P0)
  15% single_fact rehearsal (protected)
  25% tool/copy/protocol rehearsal (protected, sampled from bank train.jsonl)

Pre-registered predictions (evaluated on the DEV diagnostic, never sealed):
  PR1 P0 query-recognition 0/10 -> >=5/10
  PR2 query-swap output-changed 0-2/18 -> >=8/18
  PR3 exact-condition paired ~2/12 -> >=6/12
  PR4 any-fact extraction no regression (>=5/10)
  PR5 protected retention within parent-0.10 (dev bank families)
Falsification: PR1+PR2 fail => query-conditioning not trainable this way.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from connector.experiments.capability_bank import OBJECTS, PREFIXES, WORDS, _render

N_TARGETED = 140
N_QUERY_REC = 60


def _code(rng: random.Random) -> str:
    return f"{rng.choice(PREFIXES)}-{rng.randrange(100, 1000)}"


def build(seed: int = 555):
    rng = random.Random(seed)
    items = []

    for i in range(N_TARGETED):
        k = 2 + (i % 3)
        objs = rng.sample(OBJECTS, k)
        codes = [_code(rng) for _ in objs]
        target = i % k
        fmt = ("prose", "table", "kv", "dialogue")[i % 4]
        facts = [f"{o.capitalize()} keeps ref {c}." for o, c in zip(objs, codes)]
        q = f"the ref of {objs[target].capitalize()}"
        # Rotate QUERY POSITION relative to the facts: query-after (recency-
        # friendly), query-before, query-split (question sandwiched mid-facts).
        slot = i % 3
        if slot == 0:
            prompt = _render(fmt, facts, f"Return ONLY {q}.\nAnswer:")
        elif slot == 1:
            rendered = _render(fmt, facts, f"Return ONLY {q}.")
            prompt = rendered.replace("Answer:", "")
            prompt = f"Question: Return ONLY {q}.\n\nContext follows.\n{prompt}"
            # answer marker at the very end keeps format learnable
            prompt = prompt.rstrip() + "\nAnswer:"
        else:
            head, *tail = facts
            prompt = (f"{head}\nQuestion: Return ONLY {q}.\n" +
                      "\n".join(tail) + "\nAnswer:")
        items.append({"family": "query_targeted", "prompt": prompt,
                      "completion": f" {codes[target]}.",
                      "answer": codes[target], "gold": codes[target],
                      "protocol": fmt, "query_slot": slot})

    for i in range(N_QUERY_REC):
        k = 2 + (i % 3)
        objs = rng.sample(OBJECTS, k)
        codes = [_code(rng) for _ in objs]
        target = i % k
        facts = [f"{o.capitalize()} keeps ref {c}." for o, c in zip(objs, codes)]
        slot = i % 3
        q_line = f"Which entity is asked for if the task is: return the ref of {objs[target].capitalize()}?"
        if slot == 0:
            prompt = "\n".join(facts) + f"\n{q_line}\nAnswer with the entity name only.\nAnswer:"
        elif slot == 1:
            prompt = f"{q_line}\n\n" + "\n".join(facts) + "\nAnswer with the entity name only.\nAnswer:"
        else:
            head, *tail = facts
            prompt = (f"{head}\n{q_line}\n" + "\n".join(tail) +
                      "\nAnswer with the entity name only.\nAnswer:")
        items.append({"family": "query_recognition", "prompt": prompt,
                      "completion": f" {objs[target].capitalize()}.",
                      "answer": objs[target], "gold": objs[target],
                      "protocol": "nl", "query_slot": slot})

    # rehearsal: reuse committed bank train items (protected capabilities)
    bank = [json.loads(l) for l in
            Path("data/capability_bank/train.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rehearsal = rng.sample([b for b in bank if b["family"] == "single_fact"], 20)
    rehearsal += rng.sample([b for b in bank if b["family"] in
                             ("tool_result", "copy", "protocol_transfer")], 35)
    items = items + rehearsal
    # Normalize to the trainer's expected schema regardless of origin.
    for it in items:
        it.setdefault("gold", it.get("answer", ""))
        it.setdefault("protocol", it.get("format", "nl"))
    rng.shuffle(items)

    heldout = [dict(x, prompt=x["prompt"], completion=x["completion"])
               for x in items[:30]]
    return items[len(heldout):], heldout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/query_focus_sft")
    parser.add_argument("--seed", type=int, default=555)
    args = parser.parse_args()
    train, held = build(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(x) for x in train), encoding="utf-8")
    (out / "heldout.jsonl").write_text(
        "\n".join(json.dumps(x) for x in held), encoding="utf-8")
    fams = Counter(x["family"] for x in train)
    comp = {"total": len(train), "per_family": dict(fams),
            "percentages": {k: f"{v / len(train):.1%}" for k, v in fams.items()},
            "query_slot_histogram": dict(Counter(
                x.get("query_slot") for x in train if "query_slot" in x))}
    (out / "composition.json").write_text(json.dumps(comp, indent=2))
    print(json.dumps(comp, indent=2))


if __name__ == "__main__":
    main()
