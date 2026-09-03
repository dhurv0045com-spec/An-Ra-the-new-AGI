"""Token-indexed WSD learning-rate schedule for V5-A.

Pure function of pre-update cumulative real non-padding tokens from zero.
Resume, notebook, worker, or pack changes never rewarm it: the same token
count always yields the same learning rate.

- ``[0, 50M)``: linear ``0 -> 3e-4``;
- ``[50M, 4.5B)``: constant ``3e-4``;
- ``[4.5B, 5B]``: linear ``3e-4 -> 3e-5``.
"""

from __future__ import annotations

import hashlib
import json


SCHEDULE_SCHEMA = "anra-v5-schedule/v1"
TOKEN_BUDGET = 5_000_000_000
WARMUP_END_TOKENS = 50_000_000
STABLE_END_TOKENS = 4_500_000_000
PEAK_LEARNING_RATE = 3e-4
FINAL_LEARNING_RATE = 3e-5


def _assert_token_index(cumulative_tokens: int) -> None:
    if isinstance(cumulative_tokens, bool) or not isinstance(cumulative_tokens, int):
        raise ValueError("cumulative tokens must be an integer token index")
    if not 0 <= cumulative_tokens <= TOKEN_BUDGET:
        raise ValueError("cumulative tokens must lie inside [0, 5000000000]")


def lr_at(*, cumulative_tokens: int) -> float:
    """Return the learning rate for a pre-update cumulative token count."""

    _assert_token_index(cumulative_tokens)
    if cumulative_tokens < WARMUP_END_TOKENS:
        return PEAK_LEARNING_RATE * (cumulative_tokens / WARMUP_END_TOKENS)
    if cumulative_tokens < STABLE_END_TOKENS:
        return PEAK_LEARNING_RATE
    progress = (cumulative_tokens - STABLE_END_TOKENS) / (TOKEN_BUDGET - STABLE_END_TOKENS)
    return PEAK_LEARNING_RATE + (FINAL_LEARNING_RATE - PEAK_LEARNING_RATE) * progress


def schedule_receipt() -> dict[str, object]:
    """Return the canonical, hash-bound description of this schedule."""

    receipt: dict[str, object] = {
        "schema": SCHEDULE_SCHEMA,
        "index": "pre-update cumulative real non-padding tokens from zero",
        "token_budget": TOKEN_BUDGET,
        "warmup": {"start": 0, "end": WARMUP_END_TOKENS, "start_lr": 0.0, "end_lr": PEAK_LEARNING_RATE},
        "stable": {"start": WARMUP_END_TOKENS, "end": STABLE_END_TOKENS, "lr": PEAK_LEARNING_RATE},
        "decay": {
            "start": STABLE_END_TOKENS,
            "end": TOKEN_BUDGET,
            "start_lr": PEAK_LEARNING_RATE,
            "end_lr": FINAL_LEARNING_RATE,
            "shape": "linear",
        },
        "rewarm_on_resume_or_pack_change": False,
    }
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    receipt["sha256"] = hashlib.sha256(payload).hexdigest()
    return receipt


__all__ = [
    "FINAL_LEARNING_RATE",
    "PEAK_LEARNING_RATE",
    "SCHEDULE_SCHEMA",
    "STABLE_END_TOKENS",
    "TOKEN_BUDGET",
    "WARMUP_END_TOKENS",
    "lr_at",
    "schedule_receipt",
]
