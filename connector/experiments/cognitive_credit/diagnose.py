"""Diagnosis from intervention outcomes, with first-class uncertainty.

Evidence is represented as (intervention -> changed variable -> outcome) so a
later learned self-model can consume the same records. The classifier maps
flip patterns to labels but never forces an answer: ties, no-flips, and
contradictions resolve to ``multiple_plausible`` / ``unresolved`` /
``model_limitation`` only with positive evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from connector.experiments.cognitive_credit.case import DiagnosisLabel

# Positive evidence required before blaming the substrate: every actionable
# arm ran and none flipped the verifier while the baseline failed.
MODEL_LIMITATION_LABEL: DiagnosisLabel = "model_limitation"


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
    """Map a completed battery to a diagnosis. Uncertainty-preserving.

    - exactly one flip -> that family;
    - multiple flips of different variables -> ``multiple_plausible``;
    - no flip with a complete battery -> ``model_limitation`` (positive
      evidence: nothing at the Connector layer helped);
    - no flip with an incomplete battery -> ``unresolved`` (the experiment
      itself may have been too weak);
    - baseline already succeeded -> no failure to diagnose.
    """
    flips = tuple(arm for arm in outcomes if arm.success and not baseline_success)
    if baseline_success:
        return Diagnosis(
            label="unresolved",
            selected_intervention=None,
            changed_variable=None,
            flips=(),
            baseline_success=True,
            all_arms_ran=True,
        )
    if not flips:
        complete = bool(expected_arm_names) and {o.intervention_name for o in outcomes} >= set(expected_arm_names)
        label: DiagnosisLabel = MODEL_LIMITATION_LABEL if complete else "unresolved"
        return Diagnosis(
            label=label,
            selected_intervention=None,
            changed_variable=None,
            flips=(),
            baseline_success=False,
            all_arms_ran=complete,
        )
    variables = {arm.changed for arm in flips}
    if len(flips) == 1:
        winner = flips[0]
        return Diagnosis(
            label=_label_for(winner.changed),
            selected_intervention=winner.intervention_name,
            changed_variable=winner.changed,
            flips=(winner.intervention_name,),
            baseline_success=False,
            all_arms_ran=True,
        )
    if len(variables) == 1:
        # Several arms of the same variable flipped: still one causal story.
        winner = flips[0]
        return Diagnosis(
            label=_label_for(winner.changed),
            selected_intervention=winner.intervention_name,
            changed_variable=winner.changed,
            flips=tuple(arm.intervention_name for arm in flips),
            baseline_success=False,
            all_arms_ran=True,
        )
    return Diagnosis(
        label="multiple_plausible",
        selected_intervention=None,
        changed_variable=None,
        flips=tuple(arm.intervention_name for arm in flips),
        baseline_success=False,
        all_arms_ran=True,
    )


def _label_for(changed: str) -> DiagnosisLabel:
    return {
        "knowledge": "missing_knowledge",
        "plan": "bad_planning",
        "decode": "decode_search_sensitivity",
        "tool": "tool_failure",
        "context": "context_failure",
    }.get(changed, "unknown")  # type: ignore[return-value]


def record_of(case_id: str, baseline_success: bool, outcomes, diagnosis: Diagnosis) -> InterventionRecord:
    return InterventionRecord(
        case_id=case_id,
        baseline_success=baseline_success,
        outcomes=tuple(outcomes),
        diagnosis=diagnosis,
    )
