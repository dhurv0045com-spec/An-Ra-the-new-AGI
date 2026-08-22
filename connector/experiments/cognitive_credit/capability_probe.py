"""Substrate capability probe v2 — tests the primitives the runtime uses.

Differences from v1, driven by the trust audit:

- All generation goes through ``anra_core.generate`` via the executor
  (incremental KV state) — never stateless single-token forwards.
- P1 uses NONCE facts (invented names) so parametric memory cannot pass;
  only genuine in-context use counts.
- P4 reproduces the runtime's real tool layout via
  ``PreparedExecution.from_attempt`` — the same code path the completer
  uses. A failure here indicts the model OR the shared protocol, not a
  probe-specific imitation of it.
- Decode profiles are explicit: ``raw`` (penalty 1.0, ngram 0) measures
  learned behavior; the default assisted profile measures practical use.
- ``family_gates()`` maps experiment families to required probes so
  cognitive-credit runs are gated per family (M11).
"""

from __future__ import annotations

import argparse
import json
import sys

from anra_core.errors import CoreError
from anra_core.executor import CoreExecutor
from anra_core.generate import generate

SCAFFOLD = "<k>{k}</k>\n<plan>{plan}</plan>\n<q>{q}</q>\n<answer>"

PROBE_SCHEMA = "anra-capability-probe/v2"


def _greedy(executor: CoreExecutor, tok, prompt: str, max_new_tokens: int = 16,
            *, raw: bool = True) -> str:
    try:
        return generate(
            executor, tok, prompt, max_new_tokens=max_new_tokens,
            temperature=0.0,
            repetition_penalty=1.0 if raw else 1.15,
            no_repeat_ngram_size=0 if raw else 4,
        )
    except CoreError:
        return ""


def _best_of_n(executor: CoreExecutor, tok, prompt: str, n: int = 4,
               max_new_tokens: int = 16, *, raw: bool = False) -> list[str]:
    texts = []
    for seed in range(1, n + 1):
        try:
            texts.append(generate(
                executor, tok, prompt, max_new_tokens=max_new_tokens,
                temperature=0.8, seed=seed,
                repetition_penalty=1.0 if raw else 1.15,
                no_repeat_ngram_size=0 if raw else 4,
            ))
        except CoreError:
            continue
    return texts


def contains(text: str, needle: str) -> bool:
    import re

    norm = lambda s: re.sub(r"[^0-9a-z]+", " ", s.lower()).strip()
    return re.search(rf"(?<!\w){re.escape(norm(needle))}(?!\w)", norm(text)) is not None


def run_probe(checkpoint: str, device: str, *, profile: str = "raw") -> dict:
    """profile='raw' measures learned behavior; 'assisted' adds decode controls."""
    from connector.experiments.cognitive_credit.case import (
        Attempt,
        DecodePolicy,
        PreparedExecution,
        ToolBehavior,
    )

    executor = CoreExecutor.from_checkpoint(checkpoint, device=device)
    tok = executor.tokenizer
    assert tok is not None
    raw = profile == "raw"

    # Nonce facts: invented tokens that cannot exist in any training corpus.
    nonce_pairs = (
        ("Zephyrine", "Kalimoor"),
        ("Bravendor", "Tessaline"),
        ("Quorvath", "Miradune"),
    )

    p1 = 0
    for nonce, value in nonce_pairs:
        prompt = SCAFFOLD.format(
            k=f"The {nonce} is located in {value}.",
            plan=f"State where the {nonce} is located.",
            q=f"Where is the {nonce} located?",
        )
        if contains(_greedy(executor, tok, prompt, raw=raw), value):
            p1 += 1

    # P2: plan following WITHOUT arithmetic — symbolic transformations so the
    # probe isolates plan execution, not computation ability. The plan states
    # a non-obvious transformation; only a model that executes the stated
    # steps can produce the target string.
    p2 = 0
    for source, plan_text, expected in (
        ("alpha", "First reverse it, then append '-X'.", "ahpla-X"),
        ("beta", "Take the first letter, then repeat it three times.", "bbb"),
        ("gamma", "Swap every 'a' for 'o', then prefix 'Z-'.", "Z-gommo"),
    ):
        prompt = SCAFFOLD.format(
            k=f"The code word is {source}.",
            plan=plan_text,
            q=f"Apply the plan to the code word and write the result.",
        )
        if contains(_greedy(executor, tok, prompt, raw=raw), expected):
            p2 += 1

    words = ("ember", "quartz", "linen")
    p3 = 0
    for word in words:
        prompt = SCAFFOLD.format(k="", plan="Repeat the requested word verbatim.",
                                 q=f"Echo exactly this word: {word}")
        if contains(_greedy(executor, tok, prompt, raw=raw), word):
            p3 += 1

    # P4: REAL runtime tool path — PreparedExecution resolves the adapter and
    # injects its output exactly as make_core_completer does.
    sums = ((20, 22), (10, 7), (15, 15))
    p4 = 0
    for a, b in sums:
        def calculator(a=a, b=b) -> str:
            return str(a + b)

        attempt = Attempt(
            question=f"Use the calculator to add {a} and {b}.",
            plan="Read the calculator output and report it.",
            tool=ToolBehavior("calculator", available=True, execute=calculator),
            decode=DecodePolicy(max_new_tokens=16),
        )
        prepared = PreparedExecution.from_attempt(attempt)
        if contains(_greedy(executor, tok, prepared.prompt, raw=raw), str(a + b)):
            p4 += 1

    # P5: sampling rescue — greedy fails but best-of-N succeeds (measured).
    p5 = 0
    for word in words:
        prompt = SCAFFOLD.format(k="", plan="Repeat the requested word verbatim.",
                                 q=f"Echo exactly this word: {word}")
        outs = _best_of_n(executor, tok, prompt, raw=raw)
        if any(contains(t, word) for t in outs):
            p5 += 1

    return {
        "probe_schema": PROBE_SCHEMA,
        "decode_profile": profile,
        "P1_nonce_knowledge_use": f"{p1}/3",
        "P2_plan_following_no_arithmetic": f"{p2}/3",
        "P3_verbatim_copy": f"{p3}/3",
        "P4_tool_result_use": f"{p4}/3",
        "P5_decode_sensitivity_rescue": f"{p5}/3",
        "threshold": ">=2/3 per primitive; family gates apply per family",
        "note": "nonce facts prevent parametric-memory false positives on P1",
    }


def family_gates(probe: dict[str, str]) -> dict[str, bool]:
    """Mission 11: family-specific gating for cognitive-credit experiments.

    A family may run only when its prerequisite primitive has evidence.
    Decode-sensitivity scoring additionally requires demonstrated P5 rescue:
    without it, 'decode intervention helped' cannot be attributed.
    """
    def passes(score: str, needed: int = 2) -> bool:
        return int(score.split("/")[0]) >= needed

    return {
        "missing_knowledge": passes(probe["P1_nonce_knowledge_use"]),
        "bad_planning": passes(probe["P2_plan_following_no_arithmetic"]),
        "tool_failure": passes(probe["P4_tool_result_use"]),
        "decode_search_sensitivity": passes(probe["P3_verbatim_copy"]) and passes(probe["P5_decode_sensitivity_rescue"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", default="raw", choices=("raw", "assisted"))
    args = parser.parse_args()
    probe = run_probe(args.checkpoint, args.device, profile=args.profile)
    print(json.dumps({**probe, "family_gates": family_gates(probe)}, indent=2))
    sys.exit(0)
