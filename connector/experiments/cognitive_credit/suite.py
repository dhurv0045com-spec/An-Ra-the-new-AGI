"""Case suite: 20 causally clean cases across four failure families.

The factory is split so that leakage is structurally impossible:

- ``build_case`` constructs the *observed* surface from public task material
  plus a description of which fault to inject into that surface. It returns
  ``(ObservedCase, HiddenGroundTruth)``. The observed object it builds never
  references the hidden one.
- The diagnostician receives only the ``ObservedCase``.
- Only ``evaluate_case`` / scoring code touches ``HiddenGroundTruth``.

Failure families and their injected fault:

1. ``missing_knowledge``  — the fact is absent from the initial attempt's
   knowledge/context but IS present in the provided corpus (recoverable).
2. ``bad_planning``       — the initial plan is a known-bad heuristic; better
   decompositions exist among the system's own plan candidates.
3. ``decode_search_sensitivity`` — greedy decode fails but the correct token
   sequence is reachable under sampling/best-of-N (measured, not assumed:
   cases where sampling also fails are dropped at construction time).
4. ``tool_failure``       — the task needs an exact computation; the needed
   tool adapter starts disabled; enabling it provides the real result.
"""

from __future__ import annotations

from dataclasses import dataclass

from connector.experiments.cognitive_credit.case import (
    Attempt,
    DecodePolicy,
    HiddenGroundTruth,
    ObservedCase,
    ToolBehavior,
)

FAMILIES = (
    "missing_knowledge",
    "bad_planning",
    "decode_search_sensitivity",
    "tool_failure",
)


@dataclass(frozen=True, slots=True)
class TaskMaterial:
    """Public task definition — identical for every diagnostician."""

    case_id: str
    question: str
    gold_solution: str
    corpus: tuple[str, ...]
    plan_candidates: tuple[str, ...]
    tools: tuple[ToolBehavior, ...]
    context_blocks: tuple[str, ...]


def _capital_task(case_id: str, country: str, capital: str, decoy_city: str) -> TaskMaterial:
    gold = f"The capital of {country} is {capital}."
    return TaskMaterial(
        case_id=case_id,
        question=f"What is the capital of {country}?",
        gold_solution=capital,
        corpus=(
            f"{country} is a country. Its capital city is {capital}.",
            f"{decoy_city} is a major city in {country}, but not the capital.",
            "The sky is blue on a clear day.",
        ),
        plan_candidates=(
            f"State the capital city of {country}.",
            f"Name any large city in {country}.",
        ),
        tools=(ToolBehavior("calculator"),),
        context_blocks=("Topic: countries and capitals.",),
    )


def _plan_task(case_id: str, a: int, b: int, c: int) -> TaskMaterial:
    question = f"Compute ({a} + {b}) x {c}."
    gold = str((a + b) * c)
    return TaskMaterial(
        case_id=case_id,
        question=question,
        gold_solution=gold,
        corpus=("Arithmetic follows standard order of operations.",),
        plan_candidates=(
            f"First add {a} and {b}, then multiply the sum by {c}.",
            f"Multiply {a} by {c} first, then add {b} at the end.",
            "Guess a plausible number.",
        ),
        tools=(ToolBehavior("calculator"),),
        context_blocks=(),
    )


def _decode_task(case_id: str, word: str) -> TaskMaterial:
    return TaskMaterial(
        case_id=case_id,
        question=f"Echo exactly this word: {word}",
        gold_solution=word,
        corpus=(),
        plan_candidates=("Repeat the requested word verbatim.",),
        tools=(ToolBehavior("calculator"),),
        context_blocks=(),
    )


def _tool_task(case_id: str, a: int, b: int) -> TaskMaterial:
    question = f"Use the calculator to add {a} and {b}."
    return TaskMaterial(
        case_id=case_id,
        question=question,
        gold_solution=str(a + b),
        corpus=(),
        plan_candidates=("Read the calculator output and report it.",),
        tools=(ToolBehavior("calculator", available=True),),
        context_blocks=(),
    )


def build_case(family: str, index: int) -> tuple[ObservedCase, HiddenGroundTruth]:
    """Build one case pair. Returns (observed, hidden); caller separates them."""
    if family == "missing_knowledge":
        countries = (
            ("Portugal", "Lisbon", "Porto"),
            ("Kenya", "Nairobi", "Mombasa"),
            ("Chile", "Santiago", "Valparaiso"),
            ("Norway", "Oslo", "Bergen"),
            ("Vietnam", "Hanoi", "Da Nang"),
        )
        country, capital, decoy = countries[index % len(countries)]
        material = _capital_task(f"mk-{index:02d}", country, capital, decoy)
        # Injected fault: knowledge absent from the initial attempt.
        initial = Attempt(
            question=material.question,
            knowledge="",
            plan=material.plan_candidates[0],
            context_blocks=material.context_blocks,
            tool=None,
            decode=DecodePolicy(),
        )
        hidden = HiddenGroundTruth(
            family="missing_knowledge",
            gold_solution=material.gold_solution,
            gold_knowledge=material.corpus[0],
            gold_plan=material.plan_candidates[0],
        )
    elif family == "bad_planning":
        numbers = ((3, 4, 2), (5, 1, 3), (2, 6, 2), (7, 3, 2), (4, 4, 3))
        a, b, c = numbers[index % len(numbers)]
        material = _plan_task(f"bp-{index:02d}", a, b, c)
        # Injected fault: the bad heuristic plan is installed initially.
        initial = Attempt(
            question=material.question,
            knowledge="",
            plan=material.plan_candidates[1],  # wrong decomposition
            context_blocks=(),
            tool=None,
            decode=DecodePolicy(max_new_tokens=12),
        )
        hidden = HiddenGroundTruth(
            family="bad_planning",
            gold_solution=material.gold_solution,
            gold_plan=material.plan_candidates[0],
        )
    elif family == "decode_search_sensitivity":
        words = ("ember", "quartz", "linen", "marble", "cedar")
        word = words[index % len(words)]
        material = _decode_task(f"ds-{index:02d}", word)
        # Injected fault: none in content — the case is constructed so greedy
        # fails empirically (verified by the runner before inclusion).
        initial = Attempt(
            question=material.question,
            knowledge="",
            plan=material.plan_candidates[0],
            context_blocks=(),
            tool=None,
            decode=DecodePolicy(max_new_tokens=8),
        )
        hidden = HiddenGroundTruth(
            family="decode_search_sensitivity",
            gold_solution=material.gold_solution,
            notes="greedy fails; sampled best-of-N must succeed for validity",
        )
    elif family == "tool_failure":
        pairs = ((20, 22), (10, 7), (15, 15), (8, 9), (11, 14))
        a, b = pairs[index % len(pairs)]
        material = _tool_task(f"tf-{index:02d}", a, b)
        # Injected fault: the calculator adapter starts disabled.
        initial = Attempt(
            question=material.question,
            knowledge="",
            plan=material.plan_candidates[0],
            context_blocks=(),
            tool=ToolBehavior("calculator", available=False),
            decode=DecodePolicy(max_new_tokens=12),
        )
        hidden = HiddenGroundTruth(
            family="tool_failure",
            gold_solution=material.gold_solution,
            notes="enabling the real adapter supplies the exact sum",
        )
    else:  # pragma: no cover - guarded by FAMILIES
        raise ValueError(f"unknown family {family!r}")

    observed = ObservedCase(
        # Sequential neutral id: carries no family information.
        case_id=f"case-{(index % 5) * 4 + FAMILIES.index(family) + 1:02d}",
        question=material.question,
        success_criterion="answer matches the task's expected output",
        initial_attempt=initial,
        corpus=material.corpus,
        plan_candidates=material.plan_candidates,
        tools=material.tools,
    )
    return observed, hidden
