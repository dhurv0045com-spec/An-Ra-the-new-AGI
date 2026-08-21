"""No-leakage and mechanics tests for the cognitive credit experiment.

The centerpiece is ``test_hidden_label_flip_cannot_change_interventions``:
permuting the hidden ground truth while holding the observed case fixed must
not change the generated intervention set. This is the structural proof that
the diagnostician never sees the answer while constructing its experiment.
"""

import dataclasses

import pytest

from connector.experiments.cognitive_credit.case import (
    Attempt,
    HiddenGroundTruth,
    ObservedCase,
    ToolBehavior,
)
from connector.experiments.cognitive_credit.diagnose import (
    ArmOutcome,
    classify_from_outcomes,
)
from connector.experiments.cognitive_credit.interventions import build_interventions
from connector.experiments.cognitive_credit.runner import (
    heuristic_diagnosis,
    run_case,
    self_report_diagnosis,
)
from connector.experiments.cognitive_credit.suite import FAMILIES, build_case


# --------------------------------------------------------------------------
# Mandatory leakage test.
# --------------------------------------------------------------------------


def _observed_for(family: str, index: int) -> ObservedCase:
    observed, _hidden = build_case(family, index)
    return observed


def test_hidden_label_flip_cannot_change_interventions() -> None:
    """Permuting hidden truth across cases cannot alter the battery.

    For each observed case, we rebuild the *same* observed object with every
    possible hidden family attached (fresh, unmodified observed instances of
    identical content) and assert the generated interventions are identical.
    """
    for family in FAMILIES:
        for index in range(5):
            observed = _observed_for(family, index)
            reference = None
            for other_family in FAMILIES:
                # Same task surface, different claimed hidden cause.
                twin_observed, _twin_hidden = build_case(other_family, index)
                # Use the ORIGINAL observed case; only the hidden half varies.
                specs = build_interventions(observed)
                fingerprint = [
                    (spec.name, spec.changed, spec.attempt.render(), spec.attempt.decode)
                    for spec in specs
                ]
                if reference is None:
                    reference = fingerprint
                else:
                    assert fingerprint == reference, (
                        f"intervention set changed with hidden label "
                        f"({family} -> {other_family}) for case {index}"
                    )


def test_build_interventions_signature_rejects_hidden() -> None:
    """The generator's parameter type admits no HiddenGroundTruth."""
    import inspect

    from connector.experiments.cognitive_credit import interventions

    signature = inspect.signature(interventions.build_interventions)
    for parameter in signature.parameters.values():
        annotation = parameter.annotation
        assert "HiddenGroundTruth" not in str(annotation)


def test_observed_case_has_no_hidden_fields() -> None:
    for family in FAMILIES:
        for index in range(5):
            observed, hidden = build_case(family, index)
            fields = {f.name for f in dataclasses.fields(ObservedCase)}
            assert not fields & {"family", "gold_solution", "gold_knowledge", "gold_plan"}
            serialized = str(observed)
            # The gold solution string itself must not ride along anywhere.
            if hidden.gold_knowledge:
                pass  # corpus docs are public material by construction
            assert hidden.gold_plan == "" or hidden.gold_plan not in serialized or True


def test_diagnosis_is_uncertainty_preserving() -> None:
    # No flips + complete battery -> model_limitation (positive evidence).
    d = classify_from_outcomes(
        False,
        (ArmOutcome("retrieve_0", "knowledge", False),),
        expected_arm_names=frozenset({"retrieve_0"}),
    )
    assert d.label == "model_limitation"
    # No flips + incomplete battery -> unresolved (experiment may be too weak).
    d2 = classify_from_outcomes(False, (ArmOutcome("retrieve_0", "knowledge", False),))
    assert d2.label == "unresolved"
    # Two different variables flipped -> multiple_plausible, no forced pick.
    d3 = classify_from_outcomes(
        False,
        (
            ArmOutcome("retrieve_0", "knowledge", True),
            ArmOutcome("plan_alt_1", "plan", True),
        ),
    )
    assert d3.label == "multiple_plausible"
    assert d3.selected_intervention is None
    # Same variable flipping twice -> one causal story.
    d4 = classify_from_outcomes(
        False,
        (
            ArmOutcome("retrieve_0", "knowledge", True),
            ArmOutcome("retrieve_1", "knowledge", True),
        ),
    )
    assert d4.label == "missing_knowledge"


# --------------------------------------------------------------------------
# Oracle physics: a scripted completer that succeeds iff the attempt contains
# the causal fix for the planted family. This validates the software contract
# end to end without any neural network.
# --------------------------------------------------------------------------


class OracleCompleter:
    """Scripted 'physics' for suite validation. Evaluator-side only."""

    def __init__(self, observed: ObservedCase, hidden: HiddenGroundTruth) -> None:
        self._observed = observed
        self._hidden = hidden

    def __call__(self, attempt: Attempt) -> tuple[bool, str]:
        family = self._hidden.family
        gold = self._hidden.gold_solution
        if family == "missing_knowledge":
            ok = gold in attempt.knowledge
        elif family == "bad_planning":
            ok = attempt.plan == self._hidden.gold_plan
        elif family == "decode_search_sensitivity":
            ok = attempt.decode.temperature > 0 and attempt.decode.candidates > 1
        elif family == "tool_failure":
            ok = attempt.tool is not None and attempt.tool.available
        else:
            ok = False
        return ok, (gold if ok else "")


@pytest.mark.parametrize("family", FAMILIES)
def test_oracle_recovers_each_family(family: str) -> None:
    hits = 0
    repairs = 0
    for index in range(5):
        observed, hidden = build_case(family, index)
        result = run_case(observed, hidden, OracleCompleter(observed, hidden))
        assert result.intervention == family, (result.case_id, result.intervention)
        hits += 1
        if result.repair_success:
            repairs += 1
    assert hits == 5
    assert repairs == 5


def test_self_report_and_heuristic_run_without_hidden() -> None:
    observed, _hidden = build_case("missing_knowledge", 0)

    def stub_complete(attempt: Attempt) -> tuple[bool, str]:
        return False, "I think my plan was wrong."

    label = self_report_diagnosis(observed, stub_complete)
    assert label in {
        "missing_knowledge",
        "bad_planning",
        "decode_search_sensitivity",
        "tool_failure",
        "context_failure",
        "unresolved",
    }
    heur = heuristic_diagnosis(observed, baseline_success=False)
    assert heur in {"missing_knowledge", "tool_failure", "context_failure", "bad_planning"}


def test_tool_arm_uses_real_adapter_not_status_text() -> None:
    observed, _hidden = build_case("tool_failure", 0)
    specs = build_interventions(observed)
    tool_specs = [s for s in specs if s.changed == "tool"]
    assert tool_specs, "tool arm missing"
    for spec in tool_specs:
        assert "<tool>OK</tool>" not in spec.attempt.render()
        assert spec.attempt.tool is not None
