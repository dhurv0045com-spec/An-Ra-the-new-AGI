"""Diagnosis from intervention outcomes, with first-class uncertainty.

Evidence is represented as (intervention -> changed variable -> outcome) so a
later learned self-model can consume the same records.

The classifier reports *what the experiment showed*, never more:

- one flip                    -> ``intervention_helped`` + which variable;
- several flips, same variable -> ``intervention_helped`` (one causal story);
- several variables flipped    -> ``multiple_plausible``;
- nothing helped               -> ``no_intervention_helped`` (a description of
  THIS battery, NOT a claim about the substrate);
- battery incomplete/unreliable-> ``unresolved``.

``substrate_limitation`` is assigned only by explicit evaluator-side policy
(``with_substrate_check``), never inferred from silence.
"""

from __future__ import annotations

from dataclasses import dataclass

from connector.experiments.cognitive_credit.case import DiagnosisLabel

OUTCOME_HELPED: DiagnosisLabel = "intervention_helped"
OUTCOME_NONE: DiagnosisLabel = "no_intervention_helped"


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    intervention_name: str
    changed: str  # ChangedVariable as plain str for serialization friendliness
    success: bool


@dataclass(frozen=True, slots=True)
class InterventionRecord:
    """Preserved raw evidence for later learning (representation-first)."""

    case_id: str
    baseline_success: bool
    outcomes: tuple[ArmOutcome, ...]
    diagnosis: "Diagnosis"


@dataclass(frozen=True, slots=True)
class Diagnosis:
    label: DiagnosisLabel
    # Which single-variable change (if any) fixed the task — the action that
    # matters more than the name of the cause.
    selected_intervention: str | None
    changed_variable: str | None
    flips: tuple[str, ...]
    baseline_success: bool
    all_arms_ran: bool


def classify_from_outcomes(
    baseline_success: bool,
    outcomes: tuple[ArmOutcome, ...],
    *,
    expected_arm_names: frozenset[str] = frozenset(),
) -> Diagnosis:
    """Map a completed battery to an evidence-faithful outcome."""
    flips = tuple(arm for arm in outcomes if arm.success and not baseline_success)
    if baseline_success:
        return Diagnosis("unresolved", None, None, (), True, True)
    if not flips:
        complete = (
            bool(expected_arm_names)
            and {o.intervention_name for o in outcomes} >= set(expected_arm_names)
        )
        return Diagnosis(
            OUTCOME_NONE if complete else "unresolved",
            None,
            None,
            (),
            False,
            complete,
        )
    variables = {arm.changed for arm in flips}
    if len(flips) == 1 or len(variables) == 1:
        winner = flips[0]
        return Diagnosis(
            _label_for(winner.changed),
            winner.intervention_name,
            winner.changed,
            tuple(arm.intervention_name for arm in flips),
            False,
            True,
        )
    return Diagnosis(
        "multiple_plausible",
        None,
        None,
        tuple(arm.intervention_name for arm in flips),
        False,
        True,
    )


def with_substrate_check(diagnosis: Diagnosis, *, capability_floor_passed: bool) -> Diagnosis:
    """Evaluator-side policy: only upgrade to substrate limitation when the
    measured capability floor passed AND no intervention helped. A failed floor
    keeps the weaker, honest label."""
    if diagnosis.label == OUTCOME_NONE and capability_floor_passed:
        return dataclasses_replace(diagnosis, label="model_limitation")
    return diagnosis


def dataclasses_replace(diagnosis: Diagnosis, **changes):
    import dataclasses

    return dataclasses.replace(diagnosis, **changes)


def _label_for(changed: str) -> DiagnosisLabel:
    return {
        "knowledge": "missing_knowledge",
        "plan": "bad_planning",
        "decode": "decode_search_sensitivity",
        "tool": "tool_failure",
        "context": "context_failure",
    }.get(changed, OUTCOME_HELPED)  # type: ignore[return-value]


def record_of(case_id: str, baseline_success: bool, outcomes, diagnosis: Diagnosis) -> InterventionRecord:
    return InterventionRecord(
        case_id=case_id,
        baseline_success=baseline_success,
        outcomes=tuple(outcomes),
        diagnosis=diagnosis,
    )
