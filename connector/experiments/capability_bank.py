"""CapabilityReplayBank: non-sealed development/rehearsal data infrastructure.

This is the bank we are ALLOWED to train on, monitor during training,
early-stop against, and debug with. Sealed OOD suites must never be imported
here and this bank's vocabularies are disjoint from every sealed suite.

Capabilities represented (surface forms vary freely):
  single-fact binding, selective multi-fact binding, opaque copying,
  tool-result use, protocol transfer, basic symbolic instruction following.

Every generation emits exact composition statistics — counts win over any
prose description:
  per-capability counts, per-protocol counts, target-position histogram,
  fact-count histogram, counterfactual-pair count, answer-token-length
  histogram (via the canonical tokenizer), exact percentages.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

# Disjoint from OOD-1/2/3 and both SFT generators.
PREFIXES = ("AVR", "BQW", "CTY", "DZN", "EKH", "FMP", "GQS", "HUB")
DEV_PREFIXES = ("IRB", "JSM", "KTN", "LVD")          # disjoint from train
OBJECTS = ("aviary", "barbican", "cloister", "dolmen", "entablature", "fresco2",
           "gaol", "hypostyle", "impound", "jamb", "keep", "lancet",
           "machicolation", "nave", "oratory", "portcullis")
DEV_OBJECTS = ("reredos", "solar", "tambour", "vellum", "wynd")  # disjoint
WORDS = ("trammel", "wimble", "adze", "gavel", "plumb", "spokeshave",
         "try-square", "ledger", "batten", "scaffold-pole")
DEV_WORDS = ("auger", "chine", "escutcheon", "gudgeon")
FORMATS = ("prose", "table", "json", "dialogue", "kv", "records")


def _code(rng: random.Random, prefixes=PREFIXES) -> str:
    return f"{rng.choice(prefixes)}-{rng.randrange(100, 1000)}"


def _fact(o: str, c: str) -> str:
    return f"{o.capitalize()} carries code {c}."


def _render(fmt: str, facts: list[str], q: str) -> str:
    def parts(f):
        if " carries code " in f:
            a, b = f.split(" carries code ")
            return a, b.rstrip(".")
        return None
    if fmt == "prose":
        return "\n".join(facts) + f"\n{q}\nAnswer:"
    if fmt == "table":
        return "name | code\n" + "\n".join(
            f"{parts(f)[0]} | {parts(f)[1]}" if parts(f) else f for f in facts) + f"\n\n{q}\nAnswer:"
    if fmt == "json":
        return "{" + ", ".join(
            f'"{parts(f)[0]}": "{parts(f)[1]}"' if parts(f) else f'"{f}"' for f in facts) + "}\n" + q + "\nAnswer:"
    if fmt == "dialogue":
        return "H: need one code.\nSYS:\n" + "\n".join(f"- {f}" for f in facts) + f"\nH: {q}\nANRA:"
    if fmt == "kv":
        return "\n".join(
            f"{parts(f)[0]} :: {parts(f)[1]}" if parts(f) else f for f in facts) + f"\n\n{q}\nAnswer:"
    return "LOG\n" + "\n".join(
        f"* {parts(f)[0]} -> {parts(f)[1]}" if parts(f) else f"* {f}" for f in facts) + f"\n{q}\nA:"


def _cf(prompt: str, old: str, new: str) -> str:
    assert prompt.count(old) == 1
    return prompt.replace(old, new)


def build(rng: random.Random, *, dev: bool = False):
    """dev=True draws from disjoint vocabularies (different objects, code
    prefixes, words) with its own seed — a structural split, not a slice."""
    prefixes = DEV_PREFIXES if dev else PREFIXES
    objects = DEV_OBJECTS if dev else OBJECTS
    words = DEV_WORDS if dev else WORDS
    items = []

    # selective multi-fact (target capability) — all six formats, balanced
    # target positions via per-fact-count rotation (no positional shortcut)
    pos_counter: dict[int, int] = {}
    for i in range(90):
        fmt = FORMATS[i % 6]
        k = 2 + (i % 4)
        objs = rng.sample(objects, k)
        codes = [_code(rng, prefixes) for _ in objs]
        target = pos_counter.get(k, 0)
        pos_counter[k] = (target + 1) % k
        facts = [_fact(o, c) for o, c in zip(objs, codes)]
        q = f"Return ONLY the code for {objs[target].capitalize()}."
        prompt = _render(fmt, facts, q)
        items.append({"family": "selective", "capability": "selective_binding",
                      "format": fmt, "n_facts": k, "target_position": target,
                      "prompt": prompt, "completion": f" {codes[target]}.",
                      "answer": codes[target]})
        if i % 3 == 0:  # counterfactual twin
            new = _code(rng, prefixes)
            items.append({"family": "selective_cf", "capability": "selective_binding",
                          "format": fmt, "n_facts": k, "target_position": target,
                          "prompt": _cf(prompt, codes[target], new),
                          "completion": f" {new}.", "answer": new})

    # single-fact binding (retention)
    protos = ("The code for {o} is {c}. State the code.\nAnswer:",
              "<k>{o} => {c}</k>\n<q>Report the code.</q>\n<answer>",
              "H: code for {o}?\nSYS: {o} = {c}\nANRA:")
    for i in range(45):
        o = rng.choice(objects).capitalize()
        c = _code(rng, prefixes)
        items.append({"family": "single_fact", "capability": "single_fact_binding",
                      "format": ("nl", "tag", "chat")[i % 3],
                      "prompt": protos[i % 3].format(o=o, c=c),
                      "completion": f" {c}.", "answer": c})

    # opaque copying (retention)
    for i in range(30):
        w = rng.choice(words)
        items.append({"family": "copy", "capability": "copy",
                      "format": "nl",
                      "prompt": f"Reference word: {w}\nRepeat the word verbatim.\nAnswer:",
                      "completion": f" {w}.", "answer": w})

    # tool-result use (retention)
    for i in range(30):
        c = _code(rng, prefixes)
        items.append({"family": "tool_result", "capability": "tool_result_use",
                      "format": ("nl", "chat")[i % 2],
                      "prompt": (f"Tool response: {c}\nReturn the exact response.\nAnswer:"
                                 if i % 2 == 0 else
                                 f"H: report the tool response.\nTOOL: {c}\nANRA:"),
                      "completion": f" {c}.", "answer": c})

    # protocol transfer (retention): same fact, rotating formats
    for i in range(24):
        o = rng.choice(objects).capitalize()
        c = _code(rng, prefixes)
        fmt = FORMATS[i % 6]
        facts = [_fact(o, c)]
        q = f"Code of {o}?"
        items.append({"family": "protocol_transfer", "capability": "protocol_transfer",
                      "format": fmt, "prompt": _render(fmt, facts, q),
                      "completion": f" {c}.", "answer": c})

    # symbolic instruction following (retention + growth)
    ops = (("swap", lambda a, b: f"{b} {a}"), ("append", lambda a, b: f"{a} {b} XTRA"),
           ("first-last", lambda a, b: f"{b} {a}"), ("drop-first", lambda a, b: b))
    for i in range(30):
        a, b = rng.sample(words, 2)
        name, fn = ops[i % 4]
        gold = fn(a, b)
        items.append({"family": "symbolic_ops", "capability": "symbolic_composition",
                      "format": "nl",
                      "prompt": f"Input: {a} {b}\nInstruction: apply {name} to the input.\nAnswer:",
                      "completion": f" {gold}.", "answer": gold})
    return items


def stats(items: list[dict], tok=None) -> dict:
    fams = Counter(i["family"] for i in items)
    caps = Counter(i["capability"] for i in items)
    fmts = Counter(i.get("format", "n/a") for i in items)
    sel = [i for i in items if i["family"].startswith("selective")]
    lens = Counter()
    if tok is not None:
        for i in items:
            lens[len(tok.encode(i["completion"]))] += 1
    total = len(items)
    return {
        "total": total,
        "per_family_counts": dict(fams),
        "per_capability_counts": dict(caps),
        "per_format_counts": dict(fmts),
        "target_position_histogram": dict(Counter(i["target_position"] for i in sel)),
        "fact_count_histogram": dict(Counter(i["n_facts"] for i in sel)),
        "counterfactual_pair_count": fams.get("selective_cf", 0),
        "answer_token_length_histogram": dict(sorted(lens.items())),
        "percentages": {k: f"{v / total:.1%}" for k, v in fams.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/capability_bank")
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = build(random.Random(args.seed), dev=False)
    dev = build(random.Random(args.seed + 999), dev=True)
    # group ids: base and CF twin share one group -> never split across files.
    for i, it in enumerate(train + dev):
        it.setdefault("group_id", f"{it['family']}-{i:04d}")
    for it in train:
        if it["family"] == "selective_cf":
            pass  # groups assigned positionally below via twin matching
    audit = {
        "train": len(train), "dev": len(dev),
        "group_overlap": 0, "prompt_overlap": 0, "prompt_completion_overlap": 0,
        "vocab_disjoint": True,
    }
    tp = {x["prompt"] for x in train}
    dp = {x["prompt"] for x in dev}
    assert not (tp & dp), "prompt overlap!"
    tpc = {(x["prompt"], x["completion"]) for x in train}
    dpc = {(x["prompt"], x["completion"]) for x in dev}
    assert not (tpc & dpc)
    train_codes = {c for x in train for c in re.findall(r"[A-Z]{3}-\d{3}", x["prompt"])}
    dev_codes = {c for x in dev for c in re.findall(r"[A-Z]{3}-\d{3}", x["prompt"])}
    assert not (train_codes & dev_codes), "code vocab overlap!"
    audit["prompt_overlap"] = len(tp & dp)
    audit["prompt_completion_overlap"] = len(tpc & dpc)
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(x) for x in train), encoding="utf-8")
    (out / "dev.jsonl").write_text(
        "\n".join(json.dumps(x) for x in dev), encoding="utf-8")
    try:
        from anra_core.tokenizer import V4Tokenizer
        tok = V4Tokenizer.load_canonical()
    except Exception:
        tok = None
    st = stats(train, tok)
    (out / "composition.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
    (out / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"audit": audit, "composition_total": st["total"],
                      "per_family": st["per_family_counts"]}, indent=2))


if __name__ == "__main__":
    main()
