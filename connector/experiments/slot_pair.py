"""SLOT_PAIR: per-slot candidate scoring for multi-emission composition.

Technology: instead of ranking candidates by whole-sequence logprob of
" CODE.", score each candidate FOR EACH SLOT with slot-specific prompts:

  slot 0 prompt: context + "First tag:"      -> completion_logprob(" CODE.")
  slot 1 prompt: context + "Second tag:"     -> completion_logprob(" CODE.")

The emitted pair is (argmax_slot0, argmax_slot1), excluding repeats when
enough candidates exist. Purely observable — no gold involved.

This module provides the scorer; harvest + runners consume it.
"""
from __future__ import annotations

import torch


def slot_prompts(base_prompt: str) -> tuple[str, str]:
    """Build the two slot-conditioned prompts from a composition base
    prompt. The base ends with 'Answer:'; we replace that tail."""
    head = base_prompt.rsplit("Answer:", 1)[0].rstrip()
    return (f"{head}\nFirst tag:",
            f"{head}\nSecond tag:")


def slot_pair_scores(completion_logprob_fn, model, tok, base_prompt,
                     candidates):
    """Return (slot0_scores, slot1_scores) over candidates, in candidate
    order. completion_logprob_fn is called as fn(prompt, completion)."""
    p0, p1 = slot_prompts(base_prompt)
    s0 = [completion_logprob_fn(p0, f" {c}.") for c in candidates]
    s1 = [completion_logprob_fn(p1, f" {c}.") for c in candidates]
    return s0, s1


def choose_pair(candidates, s0, s1):
    """Pick (first, second) by per-slot argmax, avoiding duplicates when
    there are >= 2 candidates. Returns codes in emission order."""
    order0 = sorted(range(len(candidates)), key=lambda i: s0[i], reverse=True)
    first = order0[0]
    if len(candidates) == 1:
        second = first
    else:
        remaining = [i for i in sorted(range(len(candidates)),
                                       key=lambda i: s1[i], reverse=True)
                     if i != first]
        second = remaining[0]
    return candidates[first], candidates[second]


def pair_emission(first: str, second: str) -> str:
    return f" {first} {second}."
