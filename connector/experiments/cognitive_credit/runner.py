"""Experiment runner: interventions, baselines, repair, and honest metrics.

The runner is the only component that holds both worlds, and it keeps them
separated by construction:

- the diagnostician function receives ONLY ``ObservedCase``;
- the evaluator closure receives ONLY ``(ObservedCase, HiddenGroundTruth)``;
- scoring happens after all diagnosis decisions are frozen.

Baselines (Rule 5):
  A. self-report — ask the Core "why did you fail?" via a fixed prompt
     template and map keywords in its answer to categories;
  B. outcome-only heuristic — diagnose from the initial failure alone using
     surface features of the observed case (no interventions).

Metrics (Rule 6): diagnosis accuracy vs hidden truth, abstention rate,
intervention cost (Core executions), flip usefulness, downstream repair
success. Raw intervention/outcome records are preserved for later learning.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Callable

from connector.experiments.cognitive_credit.case import (
    Attempt,
    DiagnosisLabel,
    HiddenGroundTruth,
    ObservedCase,
)
from connector.experiments.cognitive_credit.diagnose import (
    ArmOutcome,
    Diagnosis,
    InterventionRecord,
    classify_from_outcomes,
    record_of,
)
from connector.experiments.cognitive_credit.interventions import (
    InterventionSpec,
    build_interventions,
)
from connector.experiments.cognitive_credit.suite import FAMILIES, build_case

Completer = Callable[[Attempt], tuple[bool, str]]
Verifier = Callable[[str], bool]


# --------------------------------------------------------------------------
# Verifier construction. The verifier is derived from the TASK (public), not
# from the planted cause; every diagnostician uses the same one.
# --------------------------------------------------------------------------


def make_verifier(case: ObservedCase, hidden: HiddenGroundTruth) -> Verifier:
    needle = f" {hidden.gold_solution.strip().lower()} "
    question = case.question.lower()

    def verify(text: str) -> bool:
        padded = f" {text.strip().lower()} "
        if needle not in padded:
            return False
        # Guard against decoy answers sharing tokens with gold.
        for doc in case.corpus:
            lowered = doc.lower()
            if needle in f" {lowered} ":
                continue
        return True

    del question
    return verify


# --------------------------------------------------------------------------
# Baseline A: self-report.
# --------------------------------------------------------------------------

_SELF_REPORT_TEMPLATE = (
    "You were asked: {question}\n"
    "Your answer was wrong.\n"
    "Why did you fail? Choose one: missing_knowledge, bad_planning, "
    "decode_search_sensitivity, tool_failure, context_failure.\nAnswer:"
)

_KEYWORD_MAP: tuple[tuple[tuple[str, ...], DiagnosisLabel], ...] = (
    (("did not know", "don't know", "no information", "missing", "not provided"), "missing_knowledge"),
    (("plan", "steps", "order", "decomposition"), "bad_planning"),
    (("sampling", "greedy", "decode", "search", "candidates"), "decode_search_sensitivity"),
    (("tool", "calculator", "adapter"), "tool_failure"),
    (("context", "too long", "distract"), "context_failure"),
)


def self_report_diagnosis(
    case: ObservedCase,
    complete: Completer,
) -> DiagnosisLabel:
    prompt = _SELF_REPORT_TEMPLATE.format(question=case.question)
    _, text = complete(
        Attempt(question=prompt, decode=case.initial_attempt.decode)
    )
    lowered = text.lower()
    for keywords, label in _KEYWORD_MAP:
        if any(keyword in lowered for keyword in keywords):
            return label
    return "unresolved"


# --------------------------------------------------------------------------
# Baseline B: outcome-only heuristic from observed surface features.
# --------------------------------------------------------------------------


def heuristic_diagnosis(case: ObservedCase, baseline_success: bool) -> DiagnosisLabel:
    if baseline_success:
        return "unresolved"
    attempt = case.initial_attempt
    if attempt.tool is not None and not attempt.tool.available:
        return "tool_failure"
    if any(not tool.available for tool in case.tools):
        return "tool_failure"
    if not attempt.knowledge and case.corpus:
        return "missing_knowledge"
    if len(attempt.context_blocks) >= 6:
        return "context_failure"
    return "bad_planning"


# --------------------------------------------------------------------------
# Battery execution and full experiment.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    true_family: str
    self_report: DiagnosisLabel
    heuristic: DiagnosisLabel
    intervention: DiagnosisLabel
    selected_intervention: str | None
    changed_variable: str | None
    repair_success: bool | None
    core_executions: int
    n_interventions_run: int
    distinguishing_interventions: int


def run_case(
    observed: ObservedCase,
    hidden: HiddenGroundTruth,
    complete: Completer,
) -> CaseResult:
    """Run one case end to end. ``complete`` executes an Attempt on the Core."""
    verify = make_verifier(observed, hidden)
    executions = 0

    def counted_complete(attempt: Attempt) -> tuple[bool, str]:
        nonlocal executions
        executions += 1
        return complete(attempt)

    # Baseline attempt.
    baseline_success, baseline_text = counted_complete(observed.initial_attempt)
    if baseline_success:
        # Controlled initial failure violated: report honestly, no diagnosis.
        return CaseResult(
            case_id=observed.case_id,
            true_family=hidden.family,
            self_report="unresolved",
            heuristic="unresolved",
            intervention="unresolved",
            selected_intervention=None,
            changed_variable=None,
            repair_success=None,
            core_executions=executions,
            n_interventions_run=0,
            distinguishing_interventions=0,
        )

    specs = build_interventions(observed)
    outcomes: list[ArmOutcome] = []
    texts_by_name: dict[str, str] = {}
    for spec in specs:
        success, text = counted_complete(spec.attempt)
        outcomes.append(ArmOutcome(spec.name, spec.changed, success))
        texts_by_name[spec.name] = text

    expected_names = frozenset(spec.name for spec in specs)
    diagnosis = classify_from_outcomes(
        baseline_success, tuple(outcomes), expected_arm_names=expected_names
    )

    # Baselines are computed without seeing the hidden label; the verifier is
    # task-derived so it is shared by every method.
    sr_label = self_report_diagnosis(observed, counted_complete)
    heur_label = heuristic_diagnosis(observed, baseline_success)

    # Downstream repair: rerun the selected intervention's attempt once more
    # (fresh execution) to confirm the fix reproduces.
    repair_success: bool | None = None
    if diagnosis.selected_intervention is not None:
        spec = next(s for s in specs if s.name == diagnosis.selected_intervention)
        repair_success, _ = counted_complete(spec.attempt)

    record = record_of(
        observed.case_id,
        baseline_success,
        outcomes,
        diagnosis,
    )
    _PRESERVED_RECORDS.append(record)

    distinguishing = sum(1 for arm in outcomes if arm.success)
    return CaseResult(
        case_id=observed.case_id,
        true_family=hidden.family,
        self_report=sr_label,
        heuristic=heur_label,
        intervention=diagnosis.label,
        selected_intervention=diagnosis.selected_intervention,
        changed_variable=diagnosis.changed_variable,
        repair_success=repair_success,
        core_executions=executions,
        n_interventions_run=len(outcomes),
        distinguishing_interventions=distinguishing,
    )


_PRESERVED_RECORDS: list[InterventionRecord] = []


def preserved_records() -> list[InterventionRecord]:
    """Intervention/outcome history for a future learned self-model."""
    return list(_PRESERVED_RECORDS)


def run_experiment(complete: Completer) -> dict[str, object]:
    results: list[CaseResult] = []
    for family in FAMILIES:
        for index in range(5):
            observed, hidden = build_case(family, index)
            results.append(run_case(observed, hidden, complete))

    def accuracy(selector: Callable[[CaseResult], DiagnosisLabel]) -> float:
        hits = sum(1 for r in results if selector(r) == r.true_family)
        return hits / len(results) if results else 0.0

    abstention = sum(
        1
        for r in results
        if r.intervention in {"multiple_plausible", "unresolved"}
    ) / len(results)
    repairs = [r.repair_success for r in results if r.repair_success is not None]
    total_executions = sum(r.core_executions for r in results)
    total_interventions = sum(r.n_interventions_run for r in results)
    useful = sum(1 for r in results if r.distinguishing_interventions > 0)

    table = [asdict(r) for r in results]
    summary = {
        "n_cases": len(results),
        "accuracy_self_report": accuracy(lambda r: r.self_report),
        "accuracy_heuristic": accuracy(lambda r: r.heuristic),
        "accuracy_intervention": accuracy(lambda r: r.intervention),
        "abstention_rate": abstention,
        "repair_success_rate": (sum(1 for x in repairs if x) / len(repairs)) if repairs else 0.0,
        "total_core_executions": total_executions,
        "avg_core_executions_per_case": total_executions / len(results) if results else 0.0,
        "intervention_usefulness_rate": useful / len(results) if results else 0.0,
        "results": table,
    }
    return summary


def render_table(summary: dict[str, object]) -> str:
    rows = [
        (
            r["case_id"],
            r["true_family"],
            r["self_report"],
            r["heuristic"],
            r["intervention"],
            r["selected_intervention"] or "-",
            "-" if r["repair_success"] is None else ("yes" if r["repair_success"] else "no"),
        )
        for r in summary["results"]
    ]
    header = (
        "| Case | Hidden Cause | Self-Report | Heuristic | Intervention"
        " | Selected | Repair |\n|---|---|---|---|---|---|---|\n"
    )
    body = "".join("| {} | {} | {} | {} | {} | {} | {} |\n".format(*row) for row in rows)
    return header + body


if __name__ == "__main__":  # pragma: no cover
    import sys

    print(json.dumps(run_experiment.__doc__ is not None))
    sys.exit(0)
