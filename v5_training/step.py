"""Pure one-update certification for the V5 training step.

Performs no tensor computation and owns no model. Certifies, after one
optimizer update, that the transition satisfies the frozen step contract
before any checkpoint may commit it: finite loss and gradients (a nonfinite
update aborts the run and advances nothing), exact token consumption,
optimizer step and token-indexed schedule advanced exactly once, sampler
cursor advanced within the bound pack, tied embedding/output identity
preserved, and post-clip replica-global gradient norm within 1.0.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping

from .state import TrainingState


STEP_SCHEMA = "anra-v5-step-receipt/v1"
GRAD_CLIP_GLOBAL_L2 = 1.0
# A real backend measures the post-clip norm in fp32; elementwise rounding of
# the clip scale leaves ~1e-7 noise, so the certification bound is 1.0 plus
# measurement noise.  A genuine clip bypass produces norms far beyond this.
_CLIP_TOLERANCE = 1e-6


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def certify_update(
    *,
    before: TrainingState,
    after: TrainingState,
    tokens_by_source: Mapping[str, int],
    loss_finite: bool,
    grad_finite: bool,
    grad_norm_post_clip: float,
    tied_preserved: bool,
) -> dict[str, object]:
    """Certify one completed optimizer update, returning a hash-bound receipt."""

    before.assert_valid()
    after.assert_valid()
    if not loss_finite:
        raise ValueError("abort NONFINITE_LOSS: update may not advance state")
    if not grad_finite:
        raise ValueError("abort NONFINITE_GRADIENT: update may not advance state")
    if not isinstance(grad_norm_post_clip, float) or not math.isfinite(grad_norm_post_clip):
        raise ValueError("abort NONFINITE_GRADIENT: post-clip norm must be a finite float")
    if grad_norm_post_clip < 0.0 or grad_norm_post_clip > GRAD_CLIP_GLOBAL_L2 + _CLIP_TOLERANCE:
        raise ValueError("abort CLIP_BREACH: post-clip replica-global norm exceeds 1.0")
    if not tied_preserved:
        raise ValueError("abort TIED_WEIGHT_BROKEN: embedding/output storage identity lost")
    expected = before.advance(
        tokens_by_source=dict(tokens_by_source),
        cursor=after.cursor,
        rng_state_sha256=after.rng_state_sha256,
        curriculum_phase=after.curriculum_phase,
        parent_checkpoint_sha256=after.parent_checkpoint_sha256,
    )
    if after != expected:
        raise ValueError(
            "abort STATE_MISMATCH: post-update state does not equal the exact one-update advance"
        )
    receipt: dict[str, object] = {
        "schema": STEP_SCHEMA,
        "update": after.global_update,
        "tokens_consumed": after.cumulative_tokens - before.cumulative_tokens,
        "cumulative_tokens": after.cumulative_tokens,
        "schedule_tokens": after.schedule_tokens,
        "before_sha256": before.sha256(),
        "after_sha256": after.sha256(),
        "grad_norm_post_clip": grad_norm_post_clip,
        "tied_preserved": True,
    }
    receipt["sha256"] = hashlib.sha256(_canonical_json(receipt)).hexdigest()
    return receipt


__all__ = [
    "GRAD_CLIP_GLOBAL_L2",
    "STEP_SCHEMA",
    "certify_update",
]
