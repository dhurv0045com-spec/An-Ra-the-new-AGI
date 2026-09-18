"""Live-verifier appraisal adapter (HORM-004 prerequisite).

Converts a committed, truth-joined evaluation outcome into a hormonal
appraisal WITHOUT importing the evaluation plane: the adapter duck-types
the post-truth shape (``task_id`` + ``correct`` as an exact bool) that only
``ScoredResult`` carries. Pre-truth types are structurally rejected:

- ``VisibleTask`` has no ``correct`` field -> rejected.
- ``CommittedOutput`` has no ``correct`` field -> rejected.
- Truthy non-bools (``1``, ``"yes"``) are rejected: only ``True``/``False``.

Appraisal therefore cannot run on model-visible prompts, unfrozen outputs,
or unverified self-report. It runs only on committed outcomes the evaluator
joined to truth after freezing.
"""

from __future__ import annotations

from typing import Any

from .hormonal_state import HormonalState


def appraise_committed(state: HormonalState, scored: Any) -> str:
    """Appraise one committed outcome; return the applied outcome label."""

    task_id = getattr(scored, "task_id", None)
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("appraisal requires a committed outcome with a task id")
    correct = getattr(scored, "correct", None)
    if correct is True:
        outcome = "success"
    elif correct is False:
        outcome = "failure"
    else:
        raise ValueError(
            "appraisal requires a post-truth boolean `correct`; pre-truth "
            "types (visible tasks, uncommitted outputs) carry no such field"
        )
    state.appraise(outcome)
    return outcome


__all__ = ["appraise_committed"]
