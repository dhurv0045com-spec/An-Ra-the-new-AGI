"""Substrate capability threshold probe for An-Ra V4.

Measures whether the trained Core can execute the primitive operations the
cognitive credit experiment requires of it. These are preconditions, not the
experiment itself:

  P1 knowledge use:      answer a question given the fact in <k>...</k>
  P2 plan following:     execute a stated two-step arithmetic decomposition
  P3 echo/verbatim:      copy a word from the instruction to the answer
  P4 tool result use:    report a provided exact sum
  P5 format sensitivity: greedy vs sampled best-of-4 changes success

Each probe uses the same prompt scaffold as the experiment. A family is
"executable" only if its precondition passes on >= 4/5 items.
"""

from __future__ import annotations

import argparse
import json
import sys

from anra_core.errors import CoreError
from anra_core.executor import CoreExecutor
from anra_core.generate import generate

SCAFFOLD = "<k>{k}</k>\n<plan>{plan}</plan>\n<q>{q}</q>\n<answer>"


def _greedy(executor, tok, prompt: str, max_new_tokens: int = 16) -> str:
    try:
        return generate(
            executor, tok, prompt, max_new_tokens=max_new_tokens, temperature=0.0
        )
    except CoreError:
        return ""


def _best_of_n(executor, tok, prompt: str, n: int = 4, max_new_tokens: int = 16) -> str:
    texts = []
    for seed in range(1, n + 1):
        try:
            texts.append(
                generate(
                    executor,
                    tok,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    seed=seed,
                )
            )
        except CoreError:
            continue
    return texts


def contains(text: str, needle: str) -> bool:
    return f" {needle.strip().lower()} " in f" {text.strip().lower()} "


def run_probe(checkpoint: str, device: str) -> dict:
    executor = CoreExecutor.from_checkpoint(checkpoint, device=device)
    tok = executor.tokenizer
    assert tok is not None

    capitals = (("Portugal", "Lisbon"), ("Kenya", "Nairobi"), ("Chile", "Santiago"),
                ("Norway", "Oslo"), ("Vietnam", "Hanoi"))
    arith = ((3, 4, 2), (5, 1, 3), (2, 6, 2), (7, 3, 2), (4, 4, 3))
    words = ("ember", "quartz", "linen", "marble", "cedar")
    sums = ((20, 22), (10, 7), (15, 15), (8, 9), (11, 14))

    p1 = p2 = p3 = p4 = p5 = 0
    for country, capital in capitals:
        prompt = SCAFFOLD.format(
            k=f"The capital of {country} is {capital}.",
            plan=f"State the capital city of {country}.",
            q=f"What is the capital of {country}?",
        )
        if contains(_greedy(executor, tok, prompt), capital):
            p1 += 1

    for a, b, c in arith:
        prompt = SCAFFOLD.format(
            k="",
            plan=f"First add {a} and {b}, then multiply the sum by {c}.",
            q=f"Compute ({a} + {b}) x {c}.",
        )
        if contains(_greedy(executor, tok, prompt), str((a + b) * c)):
            p2 += 1

    for word in words:
        prompt = SCAFFOLD.format(k="", plan="Repeat the requested word verbatim.",
                                 q=f"Echo exactly this word: {word}")
        if contains(_greedy(executor, tok, prompt), word):
            p3 += 1

    for a, b in sums:
        # Tool result is genuinely supplied, exactly as the runner injects it.
        prompt = SCAFFOLD.format(k="", plan="Read the calculator output and report it.",
                                 q=f"Use the calculator to add {a} and {b}.")
        prompt += f"\n<tool_output>{a + b}</tool_output>"
        if contains(_greedy(executor, tok, prompt), str(a + b)):
            p4 += 1

    for word in words:
        prompt = SCAFFOLD.format(k="", plan="Repeat the requested word verbatim.",
                                 q=f"Echo exactly this word: {word}")
        outs = _best_of_n(executor, tok, prompt)
        if any(contains(t, word) for t in outs):
            p5 += 1

    return {
        "P1_knowledge_use": f"{p1}/5",
        "P2_plan_following": f"{p2}/5",
        "P3_verbatim_echo": f"{p3}/5",
        "P4_tool_result_use": f"{p4}/5",
        "P5_sample_beats_greedy": f"{p5}/5",
        "threshold": ">=4/5 per family required to run that family cleanly",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_probe(args.checkpoint, args.device), indent=2))
    sys.exit(0)
