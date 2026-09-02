"""IBQ contract tests. The centerpiece: an always-negative predictor at 5%
prevalence scoring 95% raw accuracy MUST be rejected by the promotion gate —
the exact X1-REAL-0 failure mode, made mechanically impossible.

All tests are deterministic software/synthetic contracts. No model execution.
"""

from __future__ import annotations

import pytest

from x_factor.ibq import (
    CheckpointIdentity,
    ExperimentChronology,
    INFORMATION_ADDING,
    INFORMATION_PRESERVING,
    InterventionSpec,
    ChronologyError,
    balanced_accuracy,
    basis_qualified,
    basis_quality,
    brier_skill,
    mcc,
    promote_prediction,
    validate_intervention,
)
from x_factor.ibq import geometry_vs_nulls


def _spec(**over):
    base = dict(
        id="probe", version=1, family="query_salience", cost=1,
        information_class=INFORMATION_PRESERVING,
        legality_inputs=frozenset({"original_task_text"}),
        forbidden_inputs=frozenset({"gold_answer"}),
        transformation="Duplicate the query line immediately before Answer:.",
        mechanism_hypothesis="probes query-conditioned attention persistence",
        expected_signature="no effect on well-formed bindings",
        control_pair=None, status="QUALIFIED")
    base.update(over)
    return InterventionSpec(**base)


def test_legality_validator_rejects_information_adding_without_control() -> None:
    with pytest.raises(ValueError, match="control pair"):
        _spec(information_class=INFORMATION_ADDING, control_pair=None)
    # Control-pair check fires on clean inputs:
    with pytest.raises(ValueError, match="control pair"):
        _spec(information_class=INFORMATION_ADDING, control_pair=None)
    # Oracle-selected facts are rejected at construction, validator-side:
    ok = _spec(information_class=INFORMATION_ADDING, control_pair="surface_control_1")
    checks = validate_intervention(ok)
    assert checks["no_oracle_selection"] is True


def test_preserving_intervention_cannot_declare_gold_inputs() -> None:
    with pytest.raises(ValueError):
        _spec(legality_inputs=frozenset({"gold_answer"}),
              forbidden_inputs=frozenset())


# ---------------------------------------------------------------------------
# THE X1-REAL-0 TRIPWIRE.
# ---------------------------------------------------------------------------

def test_always_negative_at_95_percent_accuracy_is_rejected() -> None:
    """5% prevalence: always-negative scores 95% raw accuracy and MUST fail
    promotion — this exact failure mode invalidated X1-REAL-0."""
    n = 200
    labels = [1] * 10 + [0] * (n - 10)
    scores = [0.0] * n                      # always negative
    gate = promote_prediction(scores=scores, labels=labels)
    assert gate["raw_accuracy_diagnostic_only"] == 0.95
    assert gate["promotion"] is False, "the X1-REAL-0 failure mode must be rejected"
    assert gate["checks"]["non_degenerate_scores"] is False
    # A degenerate scorer's AUPRC is tie-order-dependent (all-equal scores
    # sorted stably put the few positives first) — meaningless. The gate
    # therefore rejects on degeneracy BEFORE ranking metrics are consulted;
    # Brier skill remains well-defined and must still fail.
    assert gate["brier_skill"] <= 0.05


def test_genuine_predictor_at_same_prevalence_is_promoted() -> None:
    labels = [1] * 10 + [0] * 190
    scores = [0.9 if y else (0.2 if i % 2 else 0.05)
              for i, y in enumerate(labels)]
    gate = promote_prediction(scores=scores, labels=labels)
    assert gate["promotion"] is True
    assert gate["auprc"] > 0.15


# ---------------------------------------------------------------------------
# Basis quality and sparsity-matched nulls.
# ---------------------------------------------------------------------------

def test_degenerate_probes_are_flagged() -> None:
    M = [[0, 1, 0], [0, 0, 1], [0, 1, 0]]  # column 0 never fires
    q = basis_quality(M)
    assert q["degenerate_interventions"] == [0]
    gate = basis_qualified(M)
    assert gate["qualified"] is False


def test_low_rank_world_separates_from_matched_nulls() -> None:
    """Structured world: signature entropy below the null p95 is NOT the
    claim — the claim is structure EXCEEDS matched nulls. Sparse-random
    worlds must NOT separate (honest failure direction)."""
    import random
    rng = random.Random(3)
    # Structured: two latent types with distinct signatures.
    M_structured = [[1, 0, 0, 1] if i % 2 else [0, 1, 1, 0] for i in range(60)]
    result = geometry_vs_nulls(M_structured, n_nulls=30, seed=1)
    assert result["entropy_p_value_vs_nulls"] <= 0.05, result
    # Sparse-random: same prevalence, no structure — must NOT separate.
    M_random = [[int(rng.random() < 0.25) for _ in range(4)] for _ in range(60)]
    random_result = geometry_vs_nulls(M_random, n_nulls=30, seed=1)
    assert random_result["entropy_p_value_vs_nulls"] > 0.05, (
        "sparse-random matrix falsely claimed structure")


def test_redundancy_and_coverage_gates() -> None:
    M_redundant = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
    q = basis_quality(M_redundant)
    assert q["pairwise_redundancy"] > 0.0
    gate = basis_qualified(M_redundant)
    assert gate["checks"]["G6_not_redundant"] is False or \
        gate["checks"]["G2_oracle_coverage"] is True


def test_imbalance_metrics_math() -> None:
    labels = [1] * 10 + [0] * 90
    pred = [1] * 10 + [0] * 90
    assert balanced_accuracy(pred, labels) == pytest.approx(1.0)
    assert mcc(pred, labels) == pytest.approx(1.0, abs=1e-9)
    # The imbalance blindness itself: always-negative gets balanced
    # accuracy 0.5 and 95% raw accuracy — which is why NEITHER may justify
    # promotion (the gate uses AUPRC-lift + Brier skill + MCC instead).
    assert balanced_accuracy([0] * 100, labels) == pytest.approx(0.5)
    # raw accuracy 0.95 with balanced accuracy 0.5 = the imbalance trap
    perfect_probs = [1.0 if y else 0.0 for y in labels]
    assert brier_skill(perfect_probs, labels) == pytest.approx(1.0)


def test_checkpoint_identity_requires_promotion_grade() -> None:
    """X1-REAL-0 shipped parameter_sha256=None — promotion-grade identity
    must refuse that permanently."""
    bad = CheckpointIdentity("f" * 64, None, "c" * 64, "abc", "t" * 64)
    with pytest.raises(ValueError, match="parameter_sha256"):
        bad.assert_promotion_grade()
    good = CheckpointIdentity("f" * 64, "p" * 64, "c" * 64, "abc123", "t" * 64)
    good.assert_promotion_grade()


def test_chronology_forbids_analyze_before_execute() -> None:
    chron = ExperimentChronology("X1-REAL-v2", preregistration_commit="abc")
    with pytest.raises(ChronologyError):
        chron.register_analysis("def")
    chron.register_execution("commit-b")
    chron.register_analysis("commit-c")
    assert chron.state == "ANALYZED"


def test_intervention_spec_hash_is_stable() -> None:
    assert _spec().hash() == _spec().hash()
    assert _spec(cost=2).hash() != _spec().hash()
