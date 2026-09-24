"""Fail-closed training orchestration over the frozen state machine.

The trainer owns no tensors and performs no mathematics. A caller-supplied
backend executes one optimizer update and reports its metrics; the trainer
advances the token-indexed state exactly once, certifies the step, fences the
run lifecycle, and commits checkpoints through the single-writer store. Any
certification failure aborts the run without advancing committed state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

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
    local_real_tokens: int | None = None


class CheckpointCoordinator(Protocol):
    """Optional all-rank single-writer boundary for distributed campaigns."""

    @property
    def is_writer(self) -> bool: ...

    def agree_update_result(
        self,
        *,
        state: TrainingState,
        checkpoint_requested: bool,
        stop_requested: bool,
        local_error: str | None = None,
    ) -> None: ...

    def aggregate_update_rng_state(
        self,
        *,
        local_rng_state_sha256: str | None,
        local_real_tokens: int | None,
        expected_global_tokens: int | None,
        expected_prior_global_tokens: int,
        local_error: str | None = None,
    ) -> str: ...

    def publish_checkpoint(
        self,
        *,
        state: TrainingState,
        parent_checkpoint_sha256: str | None,
        store: CheckpointStore,
        payload_builder: Callable[[TrainingState], dict[str, bytes]],
    ) -> str: ...

    def publish_distributed_checkpoint(
        self,
        *,
        state: TrainingState,
        parent_checkpoint_sha256: str | None,
        store: CheckpointStore,
        rank_capture: Any,
        rank_capture_error: str | None,
        payload_builder: Callable[[TrainingState, Sequence[Any]], Mapping[str, bytes]],
    ) -> str: ...

    def run_writer_callback(
        self,
        *,
        callback: Callable[[TrainingState, str], None] | None,
        state: TrainingState,
        checkpoint_sha256: str,
    ) -> None: ...

    def run_observed_callback(
        self,
        *,
        callback: Callable[[TrainingState, str], None] | None,
        state: TrainingState,
        checkpoint_sha256: str,
    ) -> None: ...


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
    on_checkpoint_observed: Callable[[TrainingState, str], None] | None = None,
    should_stop: Callable[[TrainingState], bool] | None = None,
    resume_parent_sha256: str | None = None,
    checkpoint_coordinator: CheckpointCoordinator | None = None,
    rank_capture_builder: Callable[[TrainingState], Any] | None = None,
    distributed_payload_builder: (
        Callable[[TrainingState, Sequence[Any]], Mapping[str, bytes]] | None
    ) = None,
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
    ``checkpoint_coordinator`` makes every rank agree on the publication
    boundary while only rank zero builds and publishes checkpoint bytes.
    ``on_committed`` is writer-only under coordination; the lightweight
    ``on_checkpoint_observed`` callback runs on every rank after the shared
    checkpoint SHA is committed locally.
    ``rank_capture_builder`` plus ``distributed_payload_builder`` selects the
    v2 replicated checkpoint inventory. Every rank captures its compact
    continuation state; rank zero assembles and publishes the shared payloads.
    """

    if updates <= 0:
        raise ValueError("must run at least one update")
    if checkpoint_every is not None and checkpoint_every <= 0:
        raise ValueError("checkpoint interval must be positive")
    if (rank_capture_builder is None) != (distributed_payload_builder is None):
        raise ValueError("distributed checkpoint capture and payload builders are a pair")
    if rank_capture_builder is not None and checkpoint_coordinator is None:
        raise ValueError("distributed checkpoint capture requires a checkpoint coordinator")
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
            before = state
            after = before
            boundary = False
            stop_requested = False
            local_error: str | None = None
            report: BackendReport | None = None
            try:
                report = backend_step(before)
            except Exception as exc:
                if checkpoint_coordinator is None:
                    raise
                local_error = f"{type(exc).__name__}: {str(exc)[:500]}"

            rng_state_sha256: str | None = None
            if report is not None:
                rng_state_sha256 = report.rng_state_sha256
            if checkpoint_coordinator is not None:
                aggregate_rng = getattr(
                    checkpoint_coordinator, "aggregate_update_rng_state", None,
                )
                if callable(aggregate_rng):
                    expected_global_tokens: int | None = None
                    if report is not None:
                        try:
                            expected_global_tokens = sum(report.tokens_by_source.values())
                        except Exception as exc:
                            local_error = f"{type(exc).__name__}: {str(exc)[:500]}"
                    try:
                        rng_state_sha256 = aggregate_rng(
                            local_rng_state_sha256=(
                                report.rng_state_sha256 if report is not None else None
                            ),
                            local_real_tokens=(
                                report.local_real_tokens if report is not None else None
                            ),
                            expected_global_tokens=expected_global_tokens,
                            expected_prior_global_tokens=before.cumulative_tokens,
                            local_error=local_error,
                        )
                    except Exception as exc:
                        local_error = f"{type(exc).__name__}: {str(exc)[:500]}"

            if report is not None and local_error is None:
                try:
                    after = before.advance(
                        tokens_by_source=dict(report.tokens_by_source),
                        cursor=report.cursor,
                        rng_state_sha256=rng_state_sha256,
                        parent_checkpoint_sha256=parent,
                    )
                    certify_update(
                        before=before,
                        after=after,
                        tokens_by_source=report.tokens_by_source,
                        loss_finite=report.loss_finite,
                        grad_finite=report.grad_finite,
                        grad_norm_post_clip=report.grad_norm_post_clip,
                        tied_preserved=report.tied_preserved,
                    )
                    boundary = after.complete
                    if checkpoint_every is not None:
                        boundary = boundary or after.global_update % checkpoint_every == 0
                    if should_checkpoint is not None and should_checkpoint(after):
                        boundary = True
                    if not after.complete and should_stop is not None:
                        stop_requested = should_stop(after)
                        boundary = boundary or stop_requested
                    controller.complete_update()
                    if boundary:
                        controller.begin_checkpoint()
                except Exception as exc:
                    if checkpoint_coordinator is None:
                        raise
                    local_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            if checkpoint_coordinator is not None:
                checkpoint_coordinator.agree_update_result(
                    state=after,
                    checkpoint_requested=boundary,
                    stop_requested=stop_requested,
                    local_error=local_error,
                )
            state = after
            committed_sha: str | None = None
            if boundary:
                if checkpoint_coordinator is None:
                    parent = store.publish(
                        state=state,
                        payloads=payload_builder(state),
                        expected_parent_sha256=parent,
                    )
                elif rank_capture_builder is not None:
                    rank_capture: Any = None
                    rank_capture_error: str | None = None
                    try:
                        rank_capture = rank_capture_builder(state)
                    except Exception as exc:
                        rank_capture_error = f"{type(exc).__name__}: {str(exc)[:500]}"
                    assert distributed_payload_builder is not None
                    parent = checkpoint_coordinator.publish_distributed_checkpoint(
                        state=state,
                        parent_checkpoint_sha256=parent,
                        store=store,
                        rank_capture=rank_capture,
                        rank_capture_error=rank_capture_error,
                        payload_builder=distributed_payload_builder,
                    )
                else:
                    parent = checkpoint_coordinator.publish_checkpoint(
                        state=state,
                        parent_checkpoint_sha256=parent,
                        store=store,
                        payload_builder=payload_builder,
                    )
                committed_sha = parent
                if checkpoint_coordinator is None:
                    controller.commit_checkpoint(checkpoint_sha256=parent)
                    if on_committed is not None:
                        on_committed(state, committed_sha)
                    if on_checkpoint_observed is not None:
                        on_checkpoint_observed(state, committed_sha)
                else:
                    checkpoint_coordinator.run_writer_callback(
                        callback=on_committed,
                        state=state,
                        checkpoint_sha256=committed_sha,
                    )

                    def commit_and_observe(live: TrainingState, checkpoint_sha: str) -> None:
                        controller.commit_checkpoint(checkpoint_sha256=checkpoint_sha)
                        if on_checkpoint_observed is not None:
                            on_checkpoint_observed(live, checkpoint_sha)

                    checkpoint_coordinator.run_observed_callback(
                        callback=commit_and_observe,
                        state=state,
                        checkpoint_sha256=committed_sha,
                    )
            if state.complete:
                controller.complete()
                break
            if stop_requested:
                break
    except Exception as exc:
        try:
            controller.fail(code=type(exc).__name__ or "STEP_ABORT")
        except ValueError:
            pass
        raise
    return state


__all__ = ["BackendReport", "CheckpointCoordinator", "train"]
