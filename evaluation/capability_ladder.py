"""Falsifiable adaptive-capability milestones without parameter-count claims."""

from __future__ import annotations


def evaluate_capability_ladder(metrics: dict[str, float | bool]) -> dict[str, object]:
    adaptive = {
        "novel_problem_solving": float(metrics.get("novel_problem_solving", 0.0)) >= 0.70,
        "cross_domain_transfer": float(metrics.get("cross_domain_transfer", 0.0)) >= 0.70,
        "continual_learning_gain": float(metrics.get("continual_learning_gain", 0.0)) > 0.0,
        "calibration": float(metrics.get("calibration", 0.0)) >= 0.80,
    }
    agency = {
        "tool_completion": float(metrics.get("tool_completion", 0.0)) >= 0.80,
        "long_horizon_recovery": float(metrics.get("long_horizon_recovery", 0.0)) >= 0.70,
        "verified_research": bool(metrics.get("verified_research", False)),
    }
    embodiment = {
        "simulation_workflow": float(metrics.get("simulation_workflow", 0.0)) >= 0.80,
        "shadow_mode": bool(metrics.get("shadow_mode", False)),
        "bounded_hardware_skill": bool(metrics.get("bounded_hardware_skill", False)),
    }
    return {
        "adaptive_capability": adaptive,
        "agency": agency,
        "embodiment": embodiment,
        "adaptive_passed": all(adaptive.values()),
        "agency_passed": all(agency.values()),
        "embodiment_passed": all(embodiment.values()),
    }
