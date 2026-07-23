from __future__ import annotations

from evaluation.ibs import IBSBenchmark, IBS_DIMENSIONS
from evaluation.promotion import (
    CapabilityPromotionGate,
    DeploymentPromotionGate,
    build_capability_comparison_report,
)
from evaluation.scale_gate import evaluate_scale_up


def test_ibs_has_exact_dimension_distribution() -> None:
    suite = IBSBenchmark()
    assert len(suite.tasks) == 50
    counts = {
        dimension: sum(task.dimension == dimension for task in suite.tasks)
        for dimension in IBS_DIMENSIONS
    }
    assert counts == IBS_DIMENSIONS


def test_capability_and_deployment_promotions_are_separate() -> None:
    baseline = [{"seed": 1301, "overall": 0.6, "dimensions": {"identity": 0.8}}]
    candidate = [{"seed": 1301, "overall": 0.7, "dimensions": {"identity": 0.82}}]
    capability = CapabilityPromotionGate().compare(
        baseline, candidate, owner_baseline=0.8, owner_candidate=0.81
    )
    assert capability.allowed
    deployment = DeploymentPromotionGate().evaluate(
        dict.fromkeys(DeploymentPromotionGate.REQUIRED, True)
    )
    assert deployment.allowed


def test_close_single_run_does_not_promote_without_a_replicate() -> None:
    decision = CapabilityPromotionGate().compare(
        [{"seed": 1301, "overall": 0.60, "dimensions": {"identity": 0.8}}],
        [{"seed": 1301, "overall": 0.605, "dimensions": {"identity": 0.8}}],
        owner_baseline=0.8,
        owner_candidate=0.8,
    )
    assert decision.allowed is False
    assert "clear_or_replicated_improvement" in decision.reasons


def test_scale_gate_uses_capability_not_parameter_count() -> None:
    result = evaluate_scale_up(
        smaller_score=0.6,
        larger_score=0.7,
        smaller_compute=10,
        larger_compute=20,
        identity_similarity=0.95,
        max_compute_budget=25,
    )
    assert result.allowed


def test_capability_comparison_fails_closed_on_missing_evidence() -> None:
    report = build_capability_comparison_report(
        baseline_metrics={},
        candidate_metrics={"novel_problem_solving": 0.9},
        agi_measurements={},
    )
    assert not report["promotion_ready"]
    assert "A-01" in report["insufficient_data"]
