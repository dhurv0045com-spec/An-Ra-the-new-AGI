"""Gold-firewall types: model-visible tasks vs evaluator-only truth.

Evaluation corpora legitimately carry answer truth; the firewall's job is to
make leakage structurally impossible.  ``build_visible_tasks`` projects
truth-carrying records onto ``VisibleTask`` objects that have no truth field
and rejects prompts that embed evaluator-only assignments.  ``EvaluatorTruth``
is a separate type; ``score_committed`` joins it to a committed model output
only after the output is frozen.  Nothing on the model-facing path can read
gold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


VISIBLE_SCHEMA = "anra-v5-visible-task/v1"
TRUTH_SCHEMA = "anra-v5-evaluator-truth/v1"

# Evaluator-only fields dropped at the visible boundary.
TRUTH_FIELDS = frozenset({"gold", "gold_answer", "answer", "truth", "label", "target"})

# Prompt substrings that would leak truth into the model-visible channel.
LEAK_MARKERS = tuple(f"{field}=" for field in TRUTH_FIELDS) + tuple(
    f"{field}:" for field in TRUTH_FIELDS
)


@dataclass(frozen=True, slots=True)
class VisibleTask:
    """Everything the raw model may see -- and structurally nothing else."""

    task_id: str
    cluster_id: str
    family: str
    split: str
    difficulty: str
    prompt: str
    candidates: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_id or not self.prompt:
            raise ValueError("visible tasks need an id and a prompt")
        lowered = self.prompt.lower()
        for marker in LEAK_MARKERS:
            if marker in lowered:
                raise ValueError(
                    f"visible prompt embeds an evaluator-only assignment: {marker!r}"
                )


@dataclass(frozen=True, slots=True)
class EvaluatorTruth:
    """Answer truth, joinable to a VisibleTask by task id only after commit."""

    task_id: str
    gold: str

    def __post_init__(self) -> None:
        if not self.task_id or not self.gold:
            raise ValueError("evaluator truth needs a task id and gold answer")


@dataclass(frozen=True, slots=True)
class CommittedOutput:
    """The model output for one visible task, frozen before truth is joined."""

    task_id: str
    output: str
    candidate_scores: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class ScoredResult:
    """One scored evaluation unit with full task-level provenance."""

    task_id: str
    cluster_id: str
    family: str
    split: str
    difficulty: str
    raw_output: str
    gold: str
    correct: bool
    candidate_scores: tuple[float, ...] | None


def build_visible_tasks(records: Iterable[Mapping[str, object]]) -> list[VisibleTask]:
    """Project truth-carrying evaluation records onto model-visible tasks."""

    visible: list[VisibleTask] = []
    for record in records:
        prompt = str(record["prompt"])
        visible.append(
            VisibleTask(
                task_id=str(record["task_id"]),
                cluster_id=str(record.get("cluster_id", record["task_id"])),
                family=str(record.get("family", "")),
                split=str(record.get("split", "")),
                difficulty=str(record.get("difficulty", "")),
                prompt=prompt,
                candidates=tuple(str(candidate) for candidate in record.get("candidates", ())),
            )
        )
    return visible


def build_evaluator_truth(records: Iterable[Mapping[str, object]]) -> list[EvaluatorTruth]:
    """Extract the evaluator-only truth channel from the same records."""

    truth: list[EvaluatorTruth] = []
    for record in records:
        gold = record.get("gold")
        if gold is None:
            raise ValueError(f"evaluation record {record.get('task_id')!r} lacks gold truth")
        truth.append(EvaluatorTruth(task_id=str(record["task_id"]), gold=str(gold)))
    return truth


def score_committed(
    committed: CommittedOutput,
    visible: VisibleTask,
    truth: EvaluatorTruth,
    *,
    exact_match: bool = True,
) -> ScoredResult:
    """Join committed output with truth; the scorer's only entry point."""

    if committed.task_id != visible.task_id or truth.task_id != visible.task_id:
        raise ValueError("task identity mismatch across visible/committed/truth")
    return ScoredResult(
        task_id=visible.task_id,
        cluster_id=visible.cluster_id,
        family=visible.family,
        split=visible.split,
        difficulty=visible.difficulty,
        raw_output=committed.output,
        gold=truth.gold,
        correct=(committed.output == truth.gold) if exact_match else False,
        candidate_scores=committed.candidate_scores,
    )


__all__ = [
    "LEAK_MARKERS",
    "TRUTH_FIELDS",
    "TRUTH_SCHEMA",
    "VISIBLE_SCHEMA",
    "CommittedOutput",
    "EvaluatorTruth",
    "ScoredResult",
    "VisibleTask",
    "build_evaluator_truth",
    "build_visible_tasks",
    "score_committed",
]
