"""Experiment runner: interventions, baselines, repair, and honest metrics.

Ownership contract (fixed): completers return raw ``CompletionResult`` outputs
only — they cannot manufacture success. The runner's verifier decides success
for every completion (baseline, arms, repair) identically. The verifier is
derived from the task's expected answer; hidden ground truth is used by the
verifier but by nothing the diagnostician sees.

Baselines:
  A. self-report — ask the Core "why did you fail?" and map keywords;
  B. outcome-only heuristic — surface features of the observed case only.

Metrics: diagnosis accuracy vs hidden truth, abstention rate, intervention
cost (real Core/tool executions), flip usefulness, downstream repair success.
Raw intervention/outcome records are preserved for later learning.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Callable

from connector.experiments.cognitive_credit.case import (
    Attempt,
    CompletionResult,
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
from connector.experiments.cognitive_credit.interventions import build_interventions
from connector.experiments.cognitive_credit.suite import FAMILIES, build_case

Completer = Callable[[Attempt], CompletionResult]
Verifier = Callable[[str], bool]


# --------------------------------------------------------------------------
# Verification. Smallest reliable matcher: punctuation/format tolerant.
# --------------------------------------------------------------------------


def _normalize(text: str) -> str:
    lowered = text.lower()
    # Collapse every non-alphanumeric run to a single space so "42.", "(42)",
    # and "42" all match; word-boundary padding prevents substring hits such
    # as "oslo" inside "oslon" while still allowing "the capital is oslo".
    return re.sub(r"[^0-9a-z]+", " ", lowered).strip()


def contains_answer(text: str, gold: str) -> bool:
    """True iff ``gold`` appears in ``text`` as a standalone normalized token."""
    if not gold.strip():
        return False
    pattern = rf"(?<!\w){re.escape(_normalize(gold))}(?!\w)"
    return re.search(pattern, _normalize(text)) is not None


def make_verifier(case: ObservedCase, hidden: HiddenGroundTruth) -> Verifier:
    del case  # kept for signature symmetry / future task-derived criteria

    def verify(text: str) -> bool:
        return contains_answer(text, hidden.gold_solution)

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
    result = complete(
        Attempt(
            question=_SELF_REPORT_TEMPLATE.format(question=case.question),
            decode=case.initial_attempt.decode,
        )
    )
    lowered = " ".join(result.texts).lower()
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

    def counted_complete(attempt: Attempt) -> CompletionResult:
        nonlocal executions
        result = complete(attempt)
        executions += max(1, result.n_executions)
        return result

    # Baseline attempt.
    baseline = counted_complete(observed.initial_attempt)
    baseline_text = baseline.texts[0] if baseline.texts else ""
    baseline_success = verify(baseline_text)
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
    spec_by_name: dict[str, object] = {}
    for spec in specs:
        arm_result = counted_complete(spec.attempt)
        # Runner-side success over all returned candidates (best-of-N wins).
        arm_success = any(verify(t) for t in arm_result.texts)
        outcomes.append(ArmOutcome(spec.name, spec.changed, arm_success))
        spec_by_name[spec.name] = spec

    expected_names = frozenset(spec.name for spec in specs)
    diagnosis = classify_from_outcomes(
        baseline_success, tuple(outcomes), expected_arm_names=expected_names
    )

    # Baselines are computed without seeing the hidden label; the verifier is
    # task-derived so it is shared by every method.
    sr_label = self_report_diagnosis(observed, counted_complete)
    heur_label = heuristic_diagnosis(observed, baseline_success)

    # Downstream repair: fresh execution of the selected intervention to
    # confirm the fix reproduces under a new seed offset.
    repair_success: bool | None = None
    if diagnosis.selected_intervention is not None and diagnosis.changed_variable == "decode":
        base_spec = spec_by_name[diagnosis.selected_intervention]
        retry_attempt = Attempt(
            question=base_spec.attempt.question,
            knowledge=base_spec.attempt.knowledge,
            plan=base_spec.attempt.plan,
            context_blocks=base_spec.attempt.context_blocks,
            tool=base_spec.attempt.tool,
            decode=dataclasses_replace(base_spec.attempt.decode),
        )
        retry = counted_complete(retry_attempt)
        repair_success = any(verify(t) for t in retry.texts)
    elif diagnosis.selected_intervention is not None:
        # Deterministic interventions: rerun once verbatim to confirm.
        spec = spec_by_name[diagnosis.selected_intervention]
        retry = counted_complete(spec.attempt)
        repair_success = any(verify(t) for t in retry.texts)

    record = record_of(observed.case_id, baseline_success, outcomes, diagnosis)
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


def dataclasses_replace(decode):
    """Fresh seed for a decode-policy retry without importing dataclasses here."""
    import dataclasses

    return dataclasses.replace(decode, seed=decode.seed + 17)


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
        1 for r in results if r.intervention in {"multiple_plausible", "unresolved"}
    ) / len(results)
    repairs = [r.repair_success for r in results if r.repair_success is not None]
    total_executions = sum(r.core_executions for r in results)
    useful = sum(1 for r in results if r.distinguishing_interventions > 0)

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
        "results": [asdict(r) for r in results],
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
