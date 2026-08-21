"""Substrate capability threshold probe for An-Ra V4 (v2, causally separated).

Measures whether the trained Core can execute the primitive operations the
cognitive credit experiment requires. Design rules:

  * **Nonce facts only** — identifiers like "MAV-731" cannot be known from
    pretraining, so success proves in-context information use, not recall.
  * **No arithmetic in plan following** — word operations separate "can follow
    a stated plan" from "can compute".
  * **Every family runs in two protocols** — natural language and the
    experiment's structured tag scaffold — so a family failure can be
    attributed to protocol misunderstanding rather than substrate incapacity.

Families:
  P1 in-context knowledge use   (nonce fact supplied -> asked for the nonce)
  P2 plan/instruction execution (swap words, then append a word; no math)
  P3 verbatim copying           (echo a supplied nonce-ish word)
  P4 tool-result utilization    (an exact supplied value must be reported)
  P5 decode sensitivity         (strict: greedy fails AND best-of-4 succeeds)
  P6 protocol sensitivity       (computed NL-vs-tag delta on P1-P4)

A family is "executable" only at >= 4/5. Probes are preconditions, not the
experiment itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from anra_core.errors import CoreError
from anra_core.executor import CoreExecutor
from anra_core.generate import generate

# The structured protocol is exactly what Attempt.render() produces in the
# cognitive-credit experiment; the NL protocol is plain natural language.
TAG_SCAFFOLD = "<k>{k}</k>\n<plan>{plan}</plan>\n<q>{q}</q>\n<answer>"
NL_SCAFFOLD = "Fact: {k}\nInstructions: {plan}\nQuestion: {q}\nAnswer:"

# Deterministic nonce material (no RNG: fixed tables, varied per item).
_MATERIALS = ("copper", "tin", "cobalt", "nickel", "zinc")
_IDENTIFIERS = ("MAV-731", "ZOR-482", "KEL-906", "BUN-154", "TAV-629")
_WORD_PAIRS = (("RED", "BLUE"), ("OAK", "ELM"), ("SALT", "PEPPER"),
               ("NORTH", "SOUTH"), ("DAY", "NIGHT"))
_APPEND_WORDS = ("GREEN", "PINE", "CLOVES", "EAST", "NOON")
_ECHO_WORDS = ("ember", "quartz", "linen", "marble", "cedar")
_SUMS = ((20, 22), (10, 7), (15, 15), (8, 9), (11, 14))


def _normalize(text: str) -> str:
    import re

    return re.sub(r"[^0-9a-z]+", " ", text.lower()).strip()


def _contains(text: str, gold: str) -> bool:
    """Standalone token match, tolerant of punctuation and case."""
    import re

    pattern = rf"(?<!\w){re.escape(_normalize(gold))}(?!\w)"
    return re.search(pattern, _normalize(text)) is not None


class _ProbeStats:
    def __init__(self) -> None:
        self.generations = 0
        self.errors = 0

    def note(self, error: bool) -> None:
        if error:
            self.errors += 1
        else:
            self.generations += 1


def _render(protocol: str, k: str, plan: str, q: str) -> str:
    scaffold = TAG_SCAFFOLD if protocol == "tag" else NL_SCAFFOLD
    return scaffold.format(k=k, plan=plan, q=q)


def _greedy(executor, tok, prompt: str, stats: _ProbeStats, max_new_tokens: int = 16):
    try:
        out = generate(executor, tok, prompt, max_new_tokens=max_new_tokens, temperature=0.0)
        stats.note(False)
        return out
    except CoreError:
        stats.note(True)
        return ""


def _best_of_n(executor, tok, prompt: str, stats: _ProbeStats, n: int = 4,
               max_new_tokens: int = 16):
    texts = []
    for seed in range(1, n + 1):
        try:
            texts.append(generate(
                executor, tok, prompt, max_new_tokens=max_new_tokens,
                temperature=0.8, seed=seed,
            ))
            stats.note(False)
        except CoreError:
            stats.note(True)
    return texts


# ---------------------------------------------------------------------------
# Probe families. Each returns {protocol: "x/5"} over 5 nonce items.
# ---------------------------------------------------------------------------


def _p1_knowledge(executor, tok, stats) -> dict[str, str]:
    """Nonce fact supplied in-context; question asks for the nonce."""
    scores = {"nl": 0, "tag": 0}
    for material, ident in zip(_MATERIALS, _IDENTIFIERS):
        k = f"The private identifier for {material} is {ident}."
        plan = f"State the private identifier for {material}."
        q = f"What is the private identifier for {material}?"
        for protocol in ("nl", "tag"):
            out = _greedy(executor, tok, _render(protocol, k, plan, q), stats)
            if _contains(out, ident):
                scores[protocol] += 1
    return scores


def _p2_plan(executor, tok, stats) -> dict[str, str]:
    """Two-step word operation. No arithmetic anywhere."""
    scores = {"nl": 0, "tag": 0}
    for (left, right), extra in zip(_WORD_PAIRS, _APPEND_WORDS):
        gold = f"{right} {left} {extra}"
        k = f"Input: {left} {right}"
        plan = f"1. Swap the words. 2. Add {extra}."
        q = "Apply the instructions to the input."
        for protocol in ("nl", "tag"):
            out = _greedy(executor, tok, _render(protocol, k, plan, q), stats)
            if _contains(out, gold):
                scores[protocol] += 1
    return scores


def _p3_copy(executor, tok, stats) -> dict[str, str]:
    """Echo one supplied word verbatim."""
    scores = {"nl": 0, "tag": 0}
    for word in _ECHO_WORDS:
        k = f"Reference word: {word}"
        plan = "Repeat the requested word verbatim."
        q = f"Echo exactly this word: {word}"
        for protocol in ("nl", "tag"):
            out = _greedy(executor, tok, _render(protocol, k, plan, q), stats)
            if _contains(out, word):
                scores[protocol] += 1
    return scores


def _p4_tool(executor, tok, stats) -> dict[str, str]:
    """An exact tool result is supplied; the answer must report it."""
    scores = {"nl": 0, "tag": 0}
    for a, b in _SUMS:
        total = str(a + b)
        k = f"Calculator output for {a} + {b}: {total}"
        plan = "Read the calculator output and report it."
        q = f"Use the calculator to add {a} and {b}."
        for protocol in ("nl", "tag"):
            out = _greedy(executor, tok, _render(protocol, k, plan, q), stats)
            if _contains(out, total):
                scores[protocol] += 1
    return scores


def _p5_decode(executor, tok, stats) -> dict[str, object]:
    """Strict rescue: item passes only if greedy fails AND best-of-4 passes."""
    greedy_pass = 0
    rescue = 0
    for word in _ECHO_WORDS:
        prompt = _render("tag", f"Reference word: {word}",
                         "Repeat the requested word verbatim.",
                         f"Echo exactly this word: {word}")
        greedy = _greedy(executor, tok, prompt, stats)
        sampled = _best_of_n(executor, tok, prompt, stats)
        greedy_ok = _contains(greedy, word)
        sampled_ok = any(_contains(t, word) for t in sampled)
        if greedy_ok:
            greedy_pass += 1
        elif sampled_ok:
            rescue += 1
    return {
        "greedy_pass": f"{greedy_pass}/5",
        "best_of_4_rescue": f"{rescue}/5",
        "note": "rescue counts items where greedy failed and sampling succeeded",
    }


def run_probe(checkpoint: str | None = None, device: str = "cpu", *,
              executor: CoreExecutor | None = None) -> dict[str, object]:
    """Run the P1-P6 battery. Pass ``executor`` to reuse a loaded checkpoint."""
    if executor is None:
        if not checkpoint:
            raise ValueError("run_probe needs a checkpoint or an executor")
        executor = CoreExecutor.from_checkpoint(checkpoint, device=device)
    tok = executor.tokenizer
    assert tok is not None
    stats = _ProbeStats()
    t0 = time.time()

    p1 = _p1_knowledge(executor, tok, stats)
    p2 = _p2_plan(executor, tok, stats)
    p3 = _p3_copy(executor, tok, stats)
    p4 = _p4_tool(executor, tok, stats)
    p5 = _p5_decode(executor, tok, stats)

    def fmt(scores: dict[str, int]) -> dict[str, str]:
        return {protocol: f"{hits}/5" for protocol, hits in scores.items()}

    def delta(scores: dict[str, int]) -> int:
        return scores["nl"] - scores["tag"]

    step = executor.checkpoint_identity.global_step
    return {
        "checkpoint": getattr(executor.checkpoint_identity, "source_path", None),
        "global_step": step,
        "wall_seconds": round(time.time() - t0, 1),
        "P1_nonce_knowledge_use": fmt(p1),
        "P2_plan_following_no_arithmetic": fmt(p2),
        "P3_verbatim_copy": fmt(p3),
        "P4_tool_result_use": fmt(p4),
        "P5_decode_sensitivity": p5,
        "P6_protocol_sensitivity_nl_minus_tag": {
            "P1": delta(p1), "P2": delta(p2), "P3": delta(p3), "P4": delta(p4),
        },
        "decode_stats": {"generations": stats.generations, "core_errors": stats.errors},
        "threshold": ">=4/5 in the experiment's tag protocol per family required",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_probe(args.checkpoint, args.device), indent=2))
    sys.exit(0)
