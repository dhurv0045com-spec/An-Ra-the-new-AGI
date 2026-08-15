"""Evidence gate for SFT, RLVR, STaR, DPO, and self-distillation ablations."""

from __future__ import annotations

import math
from collections.abc import Mapping

POST_TRAINING_STAGES = ("sft", "rlvr", "star", "dpo", "self_distillation")


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def evaluate_post_training_ablations(
    reports: Mapping[str, Mapping[str, object]],
    *,
    max_protected_regression: float = 0.02,
) -> dict[str, object]:
    """Require a report and ablation for every post-training method."""
    results: dict[str, dict[str, object]] = {}
    for stage in POST_TRAINING_STAGES:
        report = reports.get(stage, {})
        baseline = _finite_number(report.get("baseline_score"))
        candidate = _finite_number(report.get("candidate_score"))
        protected_regression = _finite_number(report.get("protected_regression"))
        ablation_score = _finite_number(report.get("ablation_score"))
        has_scores = baseline is not None and candidate is not None
        ablation_delta = (
            candidate - ablation_score
            if candidate is not None and ablation_score is not None
            else None
        )
        results[stage] = {
            "report_present": bool(report),
            "scores_valid": has_scores,
            "improved_or_neutral": bool(has_scores and candidate >= baseline),
            "protected_regression_bounded": bool(
                protected_regression is not None
                and protected_regression <= max_protected_regression
            ),
            "ablation_present": ablation_score is not None,
            "ablation_supports_method": bool(
                ablation_delta is not None and ablation_delta > 0.0
            ),
            "ablation_delta": ablation_delta,
        }
    passed = all(
        all(bool(value) for key, value in row.items() if key != "ablation_delta")
        for row in results.values()
    )
    return {
        "schema_version": 1,
        "stages": results,
        "max_protected_regression": max_protected_regression,
        "passed": passed,
    }
