"""Fail-closed training orchestration over the frozen state machine.

The trainer owns no tensors and performs no mathematics. A caller-supplied
backend executes one optimizer update and reports its metrics; the trainer
advances the token-indexed state exactly once, certifies the step, fences the
run lifecycle, and commits checkpoints through the single-writer store. Any
certification failure aborts the run without advancing committed state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .checkpoint import CheckpointStore
from .runner import RunController
from .state import CursorState, TrainingState
from .step import certify_update


@dataclass(frozen=True, slots=True)
class BackendReport:
    tokens_by_source: Mapping[str, int]
    cursor: CursorState
    rng_state_sha256: str
    loss_finite: bool
    grad_finite: bool
    grad_norm_post_clip: float
    tied_preserved: bool


def train(
    *,
    state: TrainingState,
    controller: RunController,
    store: CheckpointStore,
    payload_builder: Callable[[TrainingState], dict[str, bytes]],
    backend_step: Callable[[TrainingState], BackendReport],
    updates: int,
    checkpoint_every: int,
) -> TrainingState:
    """Run bounded updates; return the advanced training state."""

    if updates <= 0:
        raise ValueError("must run at least one update")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint interval must be positive")
    state.assert_valid()
    parent: str | None = state.parent_checkpoint_sha256
    try:
        for _ in range(updates):
            if state.complete:
                raise ValueError("a completed run cannot advance")
            report = backend_step(state)
            after = state.advance(
                tokens_by_source=dict(report.tokens_by_source),
                cursor=report.cursor,
                rng_state_sha256=report.rng_state_sha256,
                parent_checkpoint_sha256=parent,
            )
            certify_update(
                before=state,
                after=after,
                tokens_by_source=report.tokens_by_source,
                loss_finite=report.loss_finite,
                grad_finite=report.grad_finite,
                grad_norm_post_clip=report.grad_norm_post_clip,
                tied_preserved=report.tied_preserved,
            )
            controller.complete_update()
            state = after
            boundary = state.global_update % checkpoint_every == 0 or state.complete
            if boundary:
                controller.begin_checkpoint()
                parent = store.publish(
                    state=state,
                    payloads=payload_builder(state),
                    expected_parent_sha256=parent,
                )
                controller.commit_checkpoint(checkpoint_sha256=parent)
        if state.complete:
            controller.complete()
    except Exception as exc:
        try:
            controller.fail(code=type(exc).__name__ or "STEP_ABORT")
        except ValueError:
            pass
        raise
    return state


__all__ = ["BackendReport", "train"]
