"""Observed-only firewall: VisibleTask vs EvaluatorTruth boundary.

CORE: native neural computation.
CONNECTOR: diagnosis, candidate scoring, counterfactual tests, intervention
  selection. A Connector intervention that ranks visible candidates is
  Connector-assisted extraction, NOT raw Core capability.
ORACLE: anything consuming evaluator truth (gold, correct pair).

VisibleTask: raw context, query, visible candidate strings, format.
EvaluatorTruth: gold answer, correct entity/value pair, correctness,
  generator latent metadata.

Every answer-blind intervention/policy must accept ONLY VisibleTask.
Tests fail if it consumes EvaluatorTruth.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class VisibleTask:
    task_id: str
    context: str
    query: str
    visible_candidates: tuple[str, ...]
    format: str = "prose"

    def prompt(self) -> str:
        return f"{self.context}\n{self.query}\nAnswer:"


@dataclass(frozen=True, slots=True)
class EvaluatorTruth:
    task_id: str
    gold: str
    correct_entity: str
    correct_value: str
    generator_meta: dict = field(default_factory=dict)


FORBIDDEN_IN_BLIND_SOURCE = frozenset({
    "gold", "target_code", "correct_entity", "correct_value",
    "EvaluatorTruth", "evaluator_truth", "correctness", "is_correct",
})


def assert_answer_blind(fn) -> None:
    """Static guard: answer-blind callables must not reference hidden truth.

    Checks parameter annotations (must not take EvaluatorTruth) and scans
    source for forbidden names. Oracle ceilings intentionally fail this.
    """
    sig = inspect.signature(fn)
    for name, p in sig.parameters.items():
        ann = p.annotation
        ann_s = getattr(ann, "__name__", str(ann))
        if "EvaluatorTruth" in ann_s:
            raise TypeError(f"{fn.__name__} takes EvaluatorTruth: not answer-blind")
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return
    hits = sorted({w for w in FORBIDDEN_IN_BLIND_SOURCE if w in src})
    if hits:
        raise ValueError(f"{fn.__name__} references hidden truth: {hits}")


def make_visible(task_id: str, context: str, query: str,
                 candidates: list[str], fmt: str = "prose") -> VisibleTask:
    return VisibleTask(task_id=task_id, context=context, query=query,
                       visible_candidates=tuple(candidates), format=fmt)


def make_truth(task_id: str, gold: str, entity: str, value: str,
               meta: dict | None = None) -> EvaluatorTruth:
    return EvaluatorTruth(task_id=task_id, gold=gold, correct_entity=entity,
                          correct_value=value, generator_meta=dict(meta or {}))
