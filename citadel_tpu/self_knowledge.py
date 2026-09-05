"""Self-knowledge curriculum rows + text-answer scoring (T1D Arm F).

The operator's hypothesis: a learner that knows what it is — identity, body,
infrastructure, purpose, motivation, abilities, limits — can use that
self-knowledge from its training stream. This module renders that knowledge
in the FROZEN calculator row grammar (`prompt = answer`, lowercase,
alphabet-safe, answer <= 8 tokens, row <= 64 chars) so production
split_prompt_target / answer_spans / generation work unchanged.

Deterministic like tiered_data: `self_row(i)` is a pure function of the
index. Training and probe streams use DIFFERENT question forms (form-level
disjointness asserted by the CONSTRUCTION table). Text answers are scored by
`text_exact` — NEVER by the arithmetic integer normalizer, which maps every
non-numeric string to a shared None and would fake correctness.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

GENERATOR_VERSION = "self-knowledge/1.0"
SELF_TRAIN_N = 12_000
SELF_PROBE_N = 96
SELF_ROW_FRACTION = 7  # every 7th drawn row in arm F (frozen)

# ---------------------------------------------------------------- structure
# (domain, prompt_form, answer) — prompt_form is a printf template with %s
# slots drawn deterministically from fillers. TRAINING forms and PROBE forms
# are disjoint by construction (asserted at import time).
_TRAIN_FACTS: list[tuple[str, str, str]] = [
    ("identity", "i am %s", "an ra"),
    ("identity", "my name is %s marker", "an ra"),
    ("body", "my body is %s", "a small transformer"),
    ("body", "my mind is %s wide", "192"),
    ("infrastructure", "i run on %s", "a colab tpu"),
    ("infrastructure", "my home is %s", "a colab tpu"),
    ("purpose", "my purpose is %s", "to learn like a child"),
    ("purpose", "i exist to %s", "learn"),
    ("motivation", "when i fail i %s", "try again"),
    ("motivation", "i learn from my %s", "mistakes"),
    ("abilities", "i can %s numbers", "add"),
    ("abilities", "i am learning %s", "arithmetic"),
    ("limits", "i am %s", "small"),
    ("limits", "i make %s", "mistakes"),
    ("mission", "my goal is to learn fast with %s data", "little"),
    ("mission", "i was built by %s", "the operator"),
]

_PROBE_FACTS: list[tuple[str, str, str]] = [
    ("identity", "who am i marker %s", "an ra"),
    ("identity", "state your name %s", "an ra"),
    ("body", "what kind of thing are you %s", "a small transformer"),
    ("body", "how wide is your mind %s", "192"),
    ("infrastructure", "what hardware do i use %s", "a colab tpu"),
    ("infrastructure", "where do you live %s", "a colab tpu"),
    ("purpose", "why do you exist %s", "to learn like a child"),
    ("purpose", "what are you here for %s", "to learn like a child"),
    ("motivation", "what do you do after mistakes %s", "try again"),
    ("motivation", "how do you improve %s", "learn from my mistakes"),
    ("abilities", "what can you do with numbers %s", "add"),
    ("abilities", "name your subject %s", "arithmetic"),
    ("limits", "are you large %s", "no"),
    ("limits", "do you ever err %s", "yes"),
    ("mission", "how much data do you need %s", "little"),
    ("mission", "who is your builder %s", "the operator"),
]

_FILLERS = ("now", "today", "honestly", "truly", "friend", "clearly")
_FORMS = {form for _, form, _ in _TRAIN_FACTS}
_PROBE_FORMS = {form for _, form, _ in _PROBE_FACTS}
if _FORMS & _PROBE_FORMS:
    raise AssertionError("self-knowledge train/probe form collision")


def _row(text: str) -> str:
    """Render a fact as a frozen-grammar row (prompt = answer)."""
    prompt, answer = text.rsplit("|ANSWER|", 1)
    row = f"{prompt}= {answer.strip()}"
    if len(row) > 64:
        raise AssertionError(f"self row too long: {row!r}")
    if any(c not in "0123456789+-*/= \nabcdefghijklmnopqrstuvwxyz>" for c in row):
        raise AssertionError(f"self row leaves the alphabet: {row!r}")
    return row


def _render(fact: tuple[str, str, str], fill: str) -> str:
    _, form, answer = fact
    if form.count("%s") != 1:
        raise AssertionError(f"self form must have exactly one slot: {form!r}")
    template = form.replace("%s", fill)
    return _row(f"{template}|ANSWER|{answer}")


def self_row(i: int, *, split: str = "train") -> tuple[str, dict[str, Any]]:
    """Deterministic self-knowledge row (pure function of i and split)."""
    if split not in ("train", "probe"):
        raise ValueError(f"unknown self split {split!r}")
    facts = _TRAIN_FACTS if split == "train" else _PROBE_FACTS
    n = len(facts)
    digest = hashlib.sha256(f"self-knowledge/{split}/{i}".encode()).digest()
    fill = _FILLERS[digest[0] % len(_FILLERS)]
    fact = facts[(i + (digest[1] % n)) % n]
    if split == "train" and i >= SELF_TRAIN_N:
        raise ValueError(f"self train index {i} beyond declared pool {SELF_TRAIN_N}")
    if split == "probe" and i >= SELF_PROBE_N:
        raise ValueError(f"self probe index {i} beyond declared pool {SELF_PROBE_N}")
    row = _render(fact, fill)
    return row, {"domain": fact[0], "answer": fact[2], "form": fact[1],
                 "template": fact[1], "self_generator": GENERATOR_VERSION,
                 "split": split, "index": i}


def self_probe_rows() -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Frozen held-out probe set: (rows, targets, meta). Different question
    forms from every training form — leakage fails the arm loudly."""
    rows, targets, meta = [], [], []
    for i in range(SELF_PROBE_N):
        row, m = self_row(i, split="probe")
        rows.append(row)
        _, target = row.rsplit("=", 1)
        targets.append(target.strip())
        meta.append(m)
    return rows, targets, meta


def text_exact(prediction: str, target: str) -> bool:
    """Text exact-match: casefold, collapse whitespace, strip answer
    newlines/commentary (first line wins, mirroring the generation contract).
    Never uses the arithmetic integer normalizer."""
    def norm(s: str) -> str:
        head = (s or "").strip().split("\n", 1)[0]
        return " ".join(head.casefold().split())
    return norm(prediction) == norm(target) and bool(norm(target))


def summarize_text(predictions: list[str], targets: list[str]) -> dict[str, Any]:
    """Text-answer summary with Wilson interval (same shape as cev.summarize)."""
    from citadel_tpu import calculator_eval as cev

    if len(predictions) != len(targets):
        raise ValueError("predictions/targets length mismatch")
    correct = sum(1 for p, t in zip(predictions, targets) if text_exact(p, t))
    total = len(targets)
    lcb, ucb = cev.wilson(correct, total)
    return {"correct": correct, "total": total,
            "accuracy": (correct / total) if total else 0.0,
            "wilson_lcb": lcb, "wilson_ucb": ucb}


def most_common_null(targets: list[str]) -> list[str]:
    """Trivial null: predict the most frequent target for every probe."""
    counts: dict[str, int] = {}
    for t in targets:
        counts[t] = counts.get(t, 0) + 1
    best = max(counts.items(), key=lambda kv: kv[1])[0] if counts else ""
    return [best] * len(targets)


def plan_identity() -> dict[str, Any]:
    """Hashable identity of the self-knowledge data contract (goes into the
    experiment plan identity so resume can never mix plan versions)."""
    return {"generator_version": GENERATOR_VERSION,
            "train_n": SELF_TRAIN_N, "probe_n": SELF_PROBE_N,
            "row_fraction": SELF_ROW_FRACTION,
            "train_forms": sorted(_FORMS), "probe_forms": sorted(_PROBE_FORMS),
            "train_sha256": hashlib.sha256(
                b"\n".join(self_row(i, split="train")[0].encode()
                           for i in range(SELF_TRAIN_N))).hexdigest(),
            "probe_sha256": hashlib.sha256(
                b"\n".join(self_row(i, split="probe")[0].encode()
                           for i in range(SELF_PROBE_N))).hexdigest()}


def data_account() -> dict[str, Any]:
    """§23-style accounting for the self stream (bytes/tokens estimates)."""
    train_rows = [self_row(i, split="train")[0] for i in range(SELF_TRAIN_N)]
    probe_rows = [self_row(i, split="probe")[0] for i in range(SELF_PROBE_N)]
    return {"unique_train_rows": len(set(train_rows)),
            "unique_probe_rows": len(set(probe_rows)),
            "unique_train_bytes": sum(len(r.encode()) for r in set(train_rows)),
            "probe_bytes": sum(len(r.encode()) for r in set(probe_rows)),
            "max_row_chars": max(len(r) for r in train_rows + probe_rows)}


__all__ = ["GENERATOR_VERSION", "SELF_PROBE_N", "SELF_ROW_FRACTION",
           "SELF_TRAIN_N", "data_account", "most_common_null", "plan_identity",
           "self_probe_rows", "self_row", "summarize_text", "text_exact"]
