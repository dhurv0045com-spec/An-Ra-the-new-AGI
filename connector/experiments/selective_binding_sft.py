"""Selective-binding SFT data: content-addressed lookup among multiple facts.

Diagnosis this attacks (measured, frozen OOD suite ce1c99c5):
  - child extracts a single code from context (E: 13/15 across untrained
    protocols) but binds POSITIONALLY under multiple facts — asked for
    Compass's code among three facts it returns the FIRST fact's code;
  - last-digit fidelity is fragile (LMP-185 -> LMP-187): digit-diverse codes.

Structural split (no random slicing): train and held-out use DISJOINT object
and code-prefix vocabularies, and an automated audit rejects any exact
prompt/prompt+completion overlap while reporting lexical overlap.

Mix (EXACT, audited in output/EVIDENCE_MANIFEST.json — counts win over any
comment): 622 selection items (85.7%) + 104 rehearsal items (14.3%). This is
NOT the ~30% rehearsal the original design intended; the 14.3% actual mix is
a leading hypothesis for the grandchild's protocol-transfer regression.
Selection items: 2-4 facts, varied target position, paraphrases, irrelevant
lines, corrections, counterfactual twins that teach dependence.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from connector.experiments.context_binding_sft import make_items as rehearsal_items

TRAIN_OBJECTS = ("anchor", "bellow", "cinder", "dovetail", "escarp", "flange",
                 "girder", "hatchway", "inlet", "joist", "kiln", "lintel",
                 "manifold", "newel", "oriel", "plinth", "quoin", "rafter",
                 "sill", "truss", "undercarriage", "vault", "wainscot",
                 "yoke", "zocle")
HELDOUT_OBJECTS = ("algae", "brix", "cwm", "diabase", "eolian", "feldspar",
                   "gypsum2", "hardscape", "igneous", "jasper")
TRAIN_PREFIXES = ("FRB", "GWM", "HNT", "JVK", "KDL", "MQR", "NSC", "PZF")
HELDOUT_PREFIXES = ("QXS", "RDB", "TFN", "VGL", "WJH", "YMK")
IRRELEVANT = ("The floor was mopped on Tuesday.",
              "Lunch is scheduled for noon.",
              "The north gate stays open.",
              "Rain is expected by Friday.")

QUERIES = ("Return ONLY the code assigned to {t}.",
           "Which code is assigned to {t}? Reply with the code only.",
           "State the code for {t}.",
           "{t}'s code, and nothing else:",)


def _code(rng: random.Random, prefixes) -> str:
    return f"{rng.choice(prefixes)}-{rng.randrange(100, 1000)}"


def _selection_items(rng: random.Random, n: int, objects, prefixes) -> list[dict]:
    items = []
    for _ in range(n):
        k = rng.choice((2, 3, 3, 4))
        objs = rng.sample(objects, k)
        codes = [_code(rng, prefixes) for _ in objs]
        target = rng.randrange(k)
        lines = [f"Object {o.capitalize()} has code {c}." for o, c in zip(objs, codes)]
        if rng.random() < 0.15:
            lines.insert(rng.randrange(len(lines) + 1), rng.choice(IRRELEVANT))
        rng.shuffle(lines)
        query = rng.choice(QUERIES).format(t=objs[target].capitalize())
        prompt = "\n".join(lines) + f"\n{query}\nAnswer:"
        items.append({"family": "selection", "protocol": "prose",
                      "prompt": prompt, "completion": f" {codes[target]}.",
                      "gold": codes[target]})

        if rng.random() < 0.35:  # counterfactual twin: teach dependence
            cf_codes = list(codes)
            cf_codes[target] = _code(rng, prefixes)
            cf_lines = [f"Object {o.capitalize()} has code {c}."
                        for o, c in zip(objs, cf_codes)]
            rng.shuffle(cf_lines)
            items.append({"family": "selection_cf", "protocol": "prose",
                          "prompt": "\n".join(cf_lines) + f"\n{query}\nAnswer:",
                          "completion": f" {cf_codes[target]}.",
                          "gold": cf_codes[target]})

        if rng.random() < 0.12:  # correction with recency rule
            new_code = _code(rng, prefixes)
            o = objs[target]
            prompt = (f"Object {o.capitalize()} has code {codes[target]}.\n"
                      f"Correction: the registry now lists {o.capitalize()} "
                      f"with code {new_code}.\nReturn ONLY the current code "
                      f"for {o.capitalize()}.\nAnswer:")
            items.append({"family": "selection_correction", "protocol": "prose",
                          "prompt": prompt, "completion": f" {new_code}.",
                          "gold": new_code})
    return items


def _audit(train: list[dict], held: list[dict]) -> dict:
    train_prompts = {x["prompt"] for x in train}
    held_prompts = {x["prompt"] for x in held}
    exact = train_prompts & held_prompts
    pairs = {(x["prompt"], x["completion"]) for x in train} & \
            {(x["prompt"], x["completion"]) for x in held}
    held_vocab = {w for x in held for w in x["prompt"].split()}
    train_vocab = {w for x in train for w in x["prompt"].split()}
    lexical = sorted((held_vocab & train_vocab) -
                     {"Object", "has", "code.", "Return", "ONLY", "the",
                      "assigned", "to.", "Answer:", "Which", "is", "Reply",
                      "with", "only.", "State", "for", "and", "nothing",
                      "else:", "'s", "code,", "Correction:", "registry",
                      "now", "lists", "current"})
    if exact or pairs:
        raise SystemExit(f"OVERLAP AUDIT FAILED: exact={len(exact)} pairs={len(pairs)}")
    return {"exact_prompt_overlap": 0, "prompt_completion_overlap": 0,
            "lexical_overlap_tokens": lexical}


def build(seed: int = 991):
    rng = random.Random(seed)
    train = _selection_items(rng, 420, TRAIN_OBJECTS, TRAIN_PREFIXES)
    held = _selection_items(random.Random(seed + 7), 40, HELDOUT_OBJECTS, HELDOUT_PREFIXES)
    rehearsal = rehearsal_items(random.Random(seed + 21), 26)[: len(train) // 3]
    train = train + rehearsal
    rng.shuffle(train)
    return train, held


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/sft_selective_binding")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, held = build()
    audit = _audit(train, held)
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(x) for x in train), encoding="utf-8")
    (out / "heldout.jsonl").write_text(
        "\n".join(json.dumps(x) for x in held), encoding="utf-8")
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    fams: dict[str, int] = {}
    for x in train:
        fams[x["family"]] = fams.get(x["family"], 0) + 1
    print(json.dumps({"train": len(train), "heldout": len(held),
                      "families": fams, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
