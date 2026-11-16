"""Preregistered 6-task counterbalanced paired productivity & token-cost study protocol and analyzer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

SIX_BENCHMARK_TASKS: tuple[dict[str, Any], ...] = (
    {
        "task_id": "T1_instrument_fault_triage",
        "title": "Isolate and prove an evaluator state-transition fault",
        "backend": "v8",
        "quality_floor_fraction": 0.95,
    },
    {
        "task_id": "T2_confounder_firewall_repair",
        "title": "Diagnose an undeclared confound in an experiment spec and repair it",
        "backend": "v8",
        "quality_floor_fraction": 0.95,
    },
    {
        "task_id": "T3_crash_resume_parity",
        "title": "Recover an interrupted training run cleanly and verify final checkpoint parity",
        "backend": "x1_rd2",
        "quality_floor_fraction": 0.95,
    },
    {
        "task_id": "T4_x1_rd2_architecture_contrast",
        "title": "Author and validate a typed relational slot vs pointer architecture contrast on X1 R/D2",
        "backend": "x1_rd2",
        "quality_floor_fraction": 0.95,
    },
    {
        "task_id": "T5_bundle_replay_and_custody",
        "title": "Verify custody chain and independently replay an exported result bundle",
        "backend": "x1_rd2",
        "quality_floor_fraction": 0.95,
    },
    {
        "task_id": "T6_cohort_leakage_and_paired_stats",
        "title": "Detect transitive cohort leakage and compute Holm-Bonferroni paired seed effects",
        "backend": "v8",
        "quality_floor_fraction": 0.95,
    },
)


@dataclass(frozen=True, slots=True)
class PairedTaskObservation:
    """Single counterbalanced paired observation comparing baseline manual workflow vs Planthir."""

    task_id: str
    participant_id: str
    order: str  # "baseline_first" or "planthir_first"
    baseline_tokens: int
    planthir_tokens: int
    baseline_wall_seconds: float
    planthir_wall_seconds: float
    baseline_quality_score: float
    planthir_quality_score: float
    planthir_passed: bool


def default_protocol_manifest() -> dict[str, Any]:
    return {
        "schema": "anra-planthir-productivity-protocol-v1",
        "design": "counterbalanced_ab_ba_within_participant",
        "status": "STUDY_OBSERVATIONS_PENDING",
        "required_task_ids": [t["task_id"] for t in SIX_BENCHMARK_TASKS],
        "tasks": list(SIX_BENCHMARK_TASKS),
        "primary_endpoints": [
            "paired_token_reduction_fraction",
            "paired_wall_time_reduction_fraction",
            "quality_delta",
        ],
        "quality_noninferiority_margin": -0.05,
        "claimed_efficiency_percent": None,
        "note": "No efficiency percentage may be claimed without logged counterbalanced observations.",
    }


def analyze_productivity_study(
    observations: Sequence[PairedTaskObservation | Mapping[str, Any]],
) -> dict[str, Any]:
    """Analyze real counterbalanced paired observations or emit STUDY_OBSERVATIONS_PENDING when empty."""

    if not observations:
        return {
            **default_protocol_manifest(),
            "observation_count": 0,
            "summary": None,
        }

    normalized: list[PairedTaskObservation] = []
    for item in observations:
        if isinstance(item, PairedTaskObservation):
            normalized.append(item)
        else:
            normalized.append(
                PairedTaskObservation(
                    task_id=str(item["task_id"]),
                    participant_id=str(item["participant_id"]),
                    order=str(item["order"]),
                    baseline_tokens=int(item["baseline_tokens"]),
                    planthir_tokens=int(item["planthir_tokens"]),
                    baseline_wall_seconds=float(item["baseline_wall_seconds"]),
                    planthir_wall_seconds=float(item["planthir_wall_seconds"]),
                    baseline_quality_score=float(item["baseline_quality_score"]),
                    planthir_quality_score=float(item["planthir_quality_score"]),
                    planthir_passed=bool(item["planthir_passed"]),
                )
            )

    defects: list[str] = []
    valid_task_ids = {t["task_id"] for t in SIX_BENCHMARK_TASKS}
    orders_seen = {obs.order for obs in normalized}
    tasks_seen = {obs.task_id for obs in normalized}

    if not orders_seen <= {"baseline_first", "planthir_first"}:
        defects.append("INVALID_ORDER_LABEL")
    if len(orders_seen) < 2:
        defects.append("NOT_COUNTERBALANCED")
    missing_tasks = sorted(valid_task_ids - tasks_seen)
    if missing_tasks:
        defects.append(f"MISSING_BENCHMARK_TASKS:{missing_tasks}")

    token_reductions: list[float] = []
    time_reductions: list[float] = []
    quality_deltas: list[float] = []
    for obs in normalized:
        if obs.baseline_tokens <= 0 or obs.planthir_tokens < 0 or obs.baseline_wall_seconds <= 0 or obs.planthir_wall_seconds < 0:
            defects.append(f"INVALID_MEASUREMENT:{obs.task_id}")
            continue
        if not (
            math.isfinite(obs.baseline_wall_seconds)
            and math.isfinite(obs.planthir_wall_seconds)
            and math.isfinite(obs.baseline_quality_score)
            and math.isfinite(obs.planthir_quality_score)
        ):
            defects.append(f"NONFINITE_OBSERVATION:{obs.task_id}")
            continue
        token_reductions.append((obs.baseline_tokens - obs.planthir_tokens) / float(obs.baseline_tokens))
        time_reductions.append((obs.baseline_wall_seconds - obs.planthir_wall_seconds) / float(obs.baseline_wall_seconds))
        quality_deltas.append(obs.planthir_quality_score - obs.baseline_quality_score)

    mean_tok = sum(token_reductions) / len(token_reductions) if token_reductions else 0.0
    mean_time = sum(time_reductions) / len(time_reductions) if time_reductions else 0.0
    mean_q = sum(quality_deltas) / len(quality_deltas) if quality_deltas else 0.0
    pass_rate = sum(1 for o in normalized if o.planthir_passed) / len(normalized)
    if mean_q < -0.05:
        defects.append("QUALITY_NONINFERIORITY_FAILED")

    return {
        "schema": "anra-planthir-productivity-protocol-v1",
        "status": "STUDY_ANALYZED" if not defects else "STUDY_INVALID_OR_INCOMPLETE",
        "passed": len(defects) == 0,
        "observation_count": len(normalized),
        "tasks_covered": sorted(tasks_seen),
        "orders_covered": sorted(orders_seen),
        "mean_token_reduction_fraction": round(mean_tok, 6),
        "mean_wall_time_reduction_fraction": round(mean_time, 6),
        "mean_quality_delta": round(mean_q, 6),
        "planthir_pass_rate": round(pass_rate, 6),
        "defects": defects,
    }


__all__ = [
    "PairedTaskObservation",
    "SIX_BENCHMARK_TASKS",
    "analyze_productivity_study",
    "default_protocol_manifest",
]
