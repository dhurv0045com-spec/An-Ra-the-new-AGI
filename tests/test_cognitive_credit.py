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
    CompletionResult,
    DecodePolicy,
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
    contains_answer,
    heuristic_diagnosis,
    make_verifier,
    run_case,
    self_report_diagnosis,
)
from connector.experiments.cognitive_credit.suite import FAMILIES, build_case


# --------------------------------------------------------------------------
# Mandatory leakage test.
# --------------------------------------------------------------------------


def test_hidden_label_flip_cannot_change_interventions() -> None:
    """Permuting hidden truth cannot alter the battery.

    For each observed case we attach every possible hidden family and assert
    the generated interventions depend only on the observed case. The observed
    object passed to ``build_interventions`` is byte-identical each time; if
    any hidden data leaked into generation, some permutation would have to
    change the battery.
    """
    for family in FAMILIES:
        for index in range(5):
            observed, _ = build_case(family, index)
            reference = None
            for other_family in FAMILIES:
                _, hidden_variant = build_case(other_family, index)
                # Rebuild the identical observed surface explicitly so the
                # only thing varying in this loop is the hidden half.
                twin_observed, _ = build_case(family, index)
                assert twin_observed == observed  # surface truly unchanged
                assert hidden_variant.family == other_family  # label did vary
                specs = build_interventions(twin_observed)
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
        assert "HiddenGroundTruth" not in str(parameter.annotation)


def test_observed_case_has_no_hidden_fields() -> None:
    for family in FAMILIES:
        for index in range(5):
            observed, hidden = build_case(family, index)
            fields = {f.name for f in dataclasses.fields(ObservedCase)}
            assert not fields & {"family", "gold_solution", "gold_knowledge", "gold_plan"}
            # A gold plan that is NOT among the public plan candidates must
            # never ride along inside the observed surface.
            if hidden.gold_plan and hidden.gold_plan not in observed.plan_candidates:
                assert hidden.gold_plan not in str(observed)


# --------------------------------------------------------------------------
# Verifier: robust matching, runner-owned success.
# --------------------------------------------------------------------------


def test_verifier_is_format_tolerant() -> None:
    observed, hidden = build_case("missing_knowledge", 0)
    verify = make_verifier(observed, hidden)
    gold = hidden.gold_solution
    assert verify(gold)                       # bare
    assert verify(f"The answer is {gold}.")   # punctuation
    assert verify(f"  {gold.upper()}  ")      # case/whitespace
    assert verify(f"({gold})")                # brackets
    assert not verify("I do not know.")
    assert not verify("")


# --------------------------------------------------------------------------
# Diagnosis: evidence-faithful outcomes, uncertainty preserved.
# --------------------------------------------------------------------------


def test_no_intervention_helped_is_not_model_limitation() -> None:
    d = classify_from_outcomes(
        False,
        (ArmOutcome("retrieve_0", "knowledge", False),),
        expected_arm_names=frozenset({"retrieve_0"}),
    )
    assert d.label == "no_intervention_helped"
    d2 = classify_from_outcomes(False, (ArmOutcome("retrieve_0", "knowledge", False),))
    assert d2.label == "unresolved"
    d3 = classify_from_outcomes(
        False,
        (
            ArmOutcome("retrieve_0", "knowledge", True),
            ArmOutcome("plan_alt_1", "plan", True),
        ),
    )
    assert d3.label == "multiple_plausible"
    assert d3.selected_intervention is None
    d4 = classify_from_outcomes(
        False,
        (
            ArmOutcome("retrieve_0", "knowledge", True),
            ArmOutcome("retrieve_1", "knowledge", True),
        ),
    )
    assert d4.label == "missing_knowledge"


# --------------------------------------------------------------------------
# Oracle physics: scripted completer returns OUTPUTS ONLY; the runner's
# verifier decides success. The oracle cannot manufacture its own labels.
# --------------------------------------------------------------------------


class OracleCompleter:
    """Scripted 'physics': emits the text a perfect executor would produce."""

    def __init__(self, hidden: HiddenGroundTruth) -> None:
        self._hidden = hidden

    def __call__(self, attempt: Attempt) -> CompletionResult:
        family = self._hidden.family
        gold = self._hidden.gold_solution
        fixed = False
        if family == "missing_knowledge":
            fixed = gold in attempt.knowledge
        elif family == "bad_planning":
            fixed = attempt.plan == self._hidden.gold_plan
        elif family == "decode_search_sensitivity":
            fixed = attempt.decode.temperature > 0 and attempt.decode.candidates > 1
        elif family == "tool_failure":
            fixed = attempt.tool is not None and attempt.tool.available
        return CompletionResult(
            texts=(gold,) if fixed else ("no usable output",),
            n_executions=max(1, attempt.decode.candidates),
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_oracle_recovers_each_family(family: str) -> None:
    repairs = 0
    for index in range(5):
        observed, hidden = build_case(family, index)
        result = run_case(observed, hidden, OracleCompleter(hidden))
        assert result.intervention == family, (result.case_id, result.intervention)
        assert result.repair_success, result.case_id
        repairs += 1
    assert repairs == 5


def test_oracle_cannot_pass_when_physics_never_succeeds() -> None:
    """If the scripted world never succeeds, the runner must report exactly
    that — proving completions are graded by the verifier, not by the oracle."""
    observed, hidden = build_case("missing_knowledge", 0)

    class NeverWorks(OracleCompleter):
        def __call__(self, attempt: Attempt) -> CompletionResult:
            return CompletionResult(texts=("still no usable output",), n_executions=1)

    result = run_case(observed, hidden, NeverWorks(hidden))
    assert result.intervention == "no_intervention_helped"
    assert result.repair_success is None


def test_self_report_and_heuristic_run_without_hidden() -> None:
    observed, _hidden = build_case("missing_knowledge", 0)

    def stub_complete(attempt: Attempt) -> CompletionResult:
        return CompletionResult(texts=("I think my plan was wrong.",), n_executions=1)

    label = self_report_diagnosis(observed, stub_complete)
    assert label == "bad_planning"
    heur = heuristic_diagnosis(observed, baseline_success=False)
    assert heur == "missing_knowledge"


# --------------------------------------------------------------------------
# Interventions are real changes.
# --------------------------------------------------------------------------


def test_tool_arm_toggles_real_adapter() -> None:
    observed, _hidden = build_case("tool_failure", 0)
    specs = build_interventions(observed)
    tool_specs = [s for s in specs if s.changed == "tool"]
    assert tool_specs, "tool arm missing"
    spec = tool_specs[0]
    # Baseline adapter is disabled; the arm must enable it and execution must
    # produce the real sum, not a status string.
    assert spec.attempt.tool.available is True
    assert observed.initial_attempt.tool.available is False
    output = spec.attempt.tool.run()
    assert output.isdigit()


def test_decode_arm_requests_multiple_candidates() -> None:
    observed, _hidden = build_case("decode_search_sensitivity", 0)
    specs = build_interventions(observed)
    decode_spec = next(s for s in specs if s.changed == "decode")
    assert decode_spec.attempt.decode.candidates > 1
    assert decode_spec.attempt.decode.temperature > 0


def test_context_arm_is_absent_because_it_would_be_vacuous() -> None:
    """No case here has repositionable baseline knowledge, so a context arm
    would change nothing. Assert the battery stays honest instead."""
    for family in FAMILIES:
        for index in range(5):
            observed, _hidden = build_case(family, index)
            assert observed.initial_attempt.knowledge == "", (family, index)
            specs = build_interventions(observed)
            assert not [s for s in specs if s.changed == "context"]
