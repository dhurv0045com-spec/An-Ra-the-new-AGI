"""T1E future-experiment helpers: EOS-supervised rows (unit-tested, NOT run).

T1D postmortem: the model was never supervised to emit EOS, and every one of
15,000 generations ended MAX_TOKENS. These helpers implement the T1E contract
(docs/citadel/experiments/T1E/PLAN.md):

- render rows as  prompt chars + answer chars + EOS_ID  in ONE segment;
- eligible mask = answer chars + EOS (prompt never supervised);
- generation limits split: MAX_CONTENT_TOKENS=8, MAX_GENERATION_STEPS=9.

Pure functions over the existing calculator codec — no training, no model.
"""

from __future__ import annotations

from typing import Any

MAX_CONTENT_TOKENS = 8
MAX_GENERATION_STEPS = MAX_CONTENT_TOKENS + 1  # room for the EOS stop


def row_with_eos(row: str, *, eos_id: int) -> list[int]:
    """Encode `row` and append the real EOS token (same segment).

    The returned sequence is what the trainer packs; the eligible mask from
    `eligibility_with_eos` supervises answer chars + EOS. Production
    `causal_lm_loss` already supports EOS targets when they exist.
    """
    from citadel_tpu import calculator_eval as cev

    ids = cev.encode(row)
    if len(ids) + 1 > 64:
        raise ValueError(f"row + EOS exceeds fixed length 64: {row!r}")
    return ids + [int(eos_id)]


def eligibility_with_eos(row: str, *, eos_id: int) -> list[bool]:
    """Eligible mask over row_with_eos(row): answer characters + EOS only."""
    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import t1c_run as t1c

    ids = row_with_eos(row, eos_id=eos_id)
    plen, alen = t1c.answer_spans([row], 64)[0]
    mask = [False] * len(ids)
    for i in range(plen, plen + alen):
        mask[i] = True
    mask[plen + alen] = True  # the EOS position
    return mask


def termination_classify(stop_reason: str, *, prediction: str,
                         target: str) -> str:
    """T1E classifier: separate STOP correctness from CONTENT correctness.

    Returns one of: EOS_OK (stopped correctly, content judged separately),
    TERMINATION_FAILURE (never stopped properly), PREMATURE_STOP (stopped
    before producing the full target).
    """
    from citadel_tpu import calculator_eval as cev

    if stop_reason == "EOS":
        return "EOS_OK"
    if stop_reason in ("MAX_TOKENS", "NON_ALPHABET", "PAD"):
        return "TERMINATION_FAILURE"
    # NEWLINE or other early stop: premature if the target was longer
    if len(prediction) < len(target) and not prediction.endswith(target[-1]):
        return "PREMATURE_STOP"
    return "EOS_OK" if cev.normalize_answer(prediction) is not None else \
        "TERMINATION_FAILURE"


def content_exact_truncated(prediction: str, target: str) -> bool:
    """POST_HOC diagnostic: content exact at target length, ignoring any
    extra continuation after the target (never a preregistered metric)."""
    return prediction[:len(target)] == target and bool(target)


__all__ = ["MAX_CONTENT_TOKENS", "MAX_GENERATION_STEPS", "content_exact_truncated",
           "eligibility_with_eos", "row_with_eos", "termination_classify"]
