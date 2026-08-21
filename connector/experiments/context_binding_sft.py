"""Context-binding SFT dataset generator: the corrective layer the probes demand.

Measured deficits (P1-P6, three checkpoints): the model cannot lift a nonce
fact from context into an answer (0/5 everywhere), cannot follow a word-op
plan, and answers only from the question's final phrase. This generator
produces exactly that corrective experience — nonce items the model cannot
have memorized, in both protocols (NL 60% / tag 40%), with a disjoint held-out
split for training-time eval. The true held-out test remains the P1-P6 probe
battery: its nonce alphabet never appears here.

Item families (mirroring the probes):
  P1 nonce knowledge   context nonce fact -> answer the nonce
  P2 word-plan ops     swap two words, then append one (no arithmetic)
  P3 verbatim echo     copy the supplied word
  P4 tool result       report the supplied exact value

Output JSONL rows: {"family", "protocol", "prompt", "completion", "gold"}.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

NL = "Fact: {k}\nInstructions: {plan}\nQuestion: {q}\nAnswer:"
TAG = "<k>{k}</k>\n<plan>{plan}</plan>\n<q>{q}</q>\n<answer>"

# Disjoint from the probe battery's nonce alphabet (probe: copper/tin/cobalt/
# nickel/zinc, MAV/ZOR/KEL/BUN/TAV, ember/quartz/linen/marble/cedar).
MATERIALS = ("basalt", "saffron", "indigo", "mahogany", "alabaster", "teak",
             "cinnabar", "lapis", "obsidian", "amber", "jute", "slate",
             "porcelain", "brass", "cobble", "flint", "gypsum", "ivory",
             "marzipan", "ochre", "pumice", "quince", "sable", "tungsten",
             "verdigris", "willow", "yarrow", "zephyr", "basil", "cork")
CODE_PREFIXES = ("QRV", "LMN", "XDP", "JKW", "TFZ", "BCYG", "NRH", "VQS",
                 "PLM", "ZWXT")
ECHO_WORDS = ("harbor", "lantern", "meadow", "orchard", "pebble", "ripple",
              "saddle", "thistle", "velvet", "walnut", "anchor", "bramble",
              "cascade", "driftwood", "emberless", "fjord", "glimmer",
              "hollow", "juniper", "kestrel")
WORD_PAIRS = (("IRON", "SILK"), ("MAPLE", "OAK"), ("EAST", "WEST"),
              ("SUN", "MOON"), ("RIVER", "HILL"), ("GOLD", "SILVER"),
              ("WOLF", "CROW"), ("SUMMER", "WINTER"), ("STONE", "CLOUD"),
              ("PAPER", "INK"))
APPEND_WORDS = ("AMBER", "NORTH", "VELVET", "QUARTZ", "PINE", "COPPER",
                "MARBLE", "SILVER", "WINTER", "ORCHID")
SUM_BOUNDS = (3, 99)


def _nonce_code(rng: random.Random) -> str:
    return f"{rng.choice(CODE_PREFIXES)}-{rng.randrange(100, 999)}"


def _render(protocol: str, k: str, plan: str, q: str) -> str:
    return (NL if protocol == "nl" else TAG).format(k=k, plan=plan, q=q)


def make_items(rng: random.Random, n_per_family: int):
    items = []
    for _ in range(n_per_family):
        protocol = "nl" if rng.random() < 0.6 else "tag"

        # P1 nonce knowledge.
        material, code = rng.choice(MATERIALS), _nonce_code(rng)
        items.append({
            "family": "P1_knowledge", "protocol": protocol,
            "prompt": _render(protocol,
                              f"The private identifier for {material} is {code}.",
                              f"State the private identifier for {material}.",
                              f"What is the private identifier for {material}?"),
            "completion": f" {code}.", "gold": code,
        })

        # P2 word-plan operation (swap, then append).
        (left, right), extra = rng.choice(WORD_PAIRS), rng.choice(APPEND_WORDS)
        items.append({
            "family": "P2_plan", "protocol": protocol,
            "prompt": _render(protocol, f"Input: {left} {right}",
                              f"1. Swap the words. 2. Add {extra}.",
                              "Apply the instructions to the input."),
            "completion": f" {right} {left} {extra}.",
            "gold": f"{right} {left} {extra}",
        })

        # P3 verbatim echo.
        word = rng.choice(ECHO_WORDS)
        items.append({
            "family": "P3_echo", "protocol": protocol,
            "prompt": _render(protocol, f"Reference word: {word}",
                              "Repeat the requested word verbatim.",
                              f"Echo exactly this word: {word}"),
            "completion": f" {word}.", "gold": word,
        })

        # P4 tool result.
        a = rng.randrange(*SUM_BOUNDS)
        b = rng.randrange(*SUM_BOUNDS)
        items.append({
            "family": "P4_tool", "protocol": protocol,
            "prompt": _render(protocol, f"Calculator output for {a} + {b}: {a + b}",
                              "Read the calculator output and report it.",
                              f"Use the calculator to add {a} and {b}."),
            "completion": f" {a + b}.", "gold": str(a + b),
        })
    return items


def build(n_per_family: int = 250, seed: int = 77, heldout: int = 60):
    rng = random.Random(seed)
    items = make_items(rng, n_per_family)
    rng.shuffle(items)
    return items[heldout:], items[:heldout]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/sft_context_binding")
    parser.add_argument("--n-per-family", type=int, default=250)
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, held = build(args.n_per_family, args.seed)
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(x) for x in train), encoding="utf-8")
    (out / "heldout.jsonl").write_text(
        "\n".join(json.dumps(x) for x in held), encoding="utf-8")

    counts: dict[str, int] = {}
    prot: dict[str, int] = {}
    for x in train:
        counts[x["family"]] = counts.get(x["family"], 0) + 1
        prot[x["protocol"]] = prot.get(x["protocol"], 0) + 1
    print(json.dumps({"train": len(train), "heldout": len(held),
                      "families": counts, "protocols": prot}, indent=2))


if __name__ == "__main__":
    main()
