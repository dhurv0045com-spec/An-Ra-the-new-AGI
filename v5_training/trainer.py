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
    checkpoint_every: int | None,
    should_checkpoint: Callable[[TrainingState], bool] | None = None,
    on_committed: Callable[[TrainingState, str], None] | None = None,
    should_stop: Callable[[TrainingState], bool] | None = None,
    resume_parent_sha256: str | None = None,
) -> TrainingState:
    """Run bounded updates; return the advanced training state.

    ``should_checkpoint`` forces extra checkpoint boundaries (milestones,
    recovery cadence) beyond ``checkpoint_every``. ``on_committed`` observes
    each committed (state, checkpoint_sha256) for milestone receipts.
    ``should_stop`` halts after the current update without completing the
    run, leaving a resumable recovery state for timeboxed sessions.
    ``checkpoint_every=None`` disables periodic boundaries entirely, so
    commits happen only at should_checkpoint boundaries and completion.
    ``resume_parent_sha256`` continues a restored lineage: the next publish
    is fenced against the restored head instead of the state's recorded
    parent (which names the grandparent, the parent at restore time).
    """

    if updates <= 0:
        raise ValueError("must run at least one update")
    if checkpoint_every is not None and checkpoint_every <= 0:
        raise ValueError("checkpoint interval must be positive")
    state.assert_valid()
    if resume_parent_sha256 is not None and (
            len(resume_parent_sha256) != 64
            or any(c not in "0123456789abcdef" for c in resume_parent_sha256)):
        raise ValueError("resume parent must be a lowercase SHA-256")
    parent: str | None = (resume_parent_sha256 if resume_parent_sha256 is not None
                           else state.parent_checkpoint_sha256)
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
            boundary = state.complete
            if checkpoint_every is not None:
                boundary = boundary or state.global_update % checkpoint_every == 0
            if should_checkpoint is not None and should_checkpoint(state):
                boundary = True
            committed_sha: str | None = None
            if boundary:
                controller.begin_checkpoint()
                parent = store.publish(
                    state=state,
                    payloads=payload_builder(state),
                    expected_parent_sha256=parent,
                )
                controller.commit_checkpoint(checkpoint_sha256=parent)
                committed_sha = parent
                if on_committed is not None:
                    on_committed(state, committed_sha)
            if state.complete:
                controller.complete()
                break
            if should_stop is not None and should_stop(state):
                break
    except Exception as exc:
        try:
            controller.fail(code=type(exc).__name__ or "STEP_ABORT")
        except ValueError:
            pass
        raise
    return state


__all__ = ["BackendReport", "train"]
