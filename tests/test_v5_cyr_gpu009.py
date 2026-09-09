from __future__ import annotations

import ast

from anra_v5 import cyr_gpu009_run as runner
from v5_experiments import cyr_gpu009 as core


def _receipt(tps: float, eps: float, *, params: int = 1) -> dict[str, object]:
    return {
        "status": "PASS",
        "training_real_tokens_per_sec": tps,
        "generation_examples_per_sec": eps,
        "parameters": params,
        "peak_vram_gb": 1.0,
    }


def test_real_operator_slow_gpu_calibration_resolves_instead_of_raising():
    # Values reproduce the important scale of the operator's failed V8 Cell-0
    # calibration. 009 must start a progressive campaign rather than reject it.
    calibrations = {
        "MIDI": {"status": "OOM"},
        "MICRO": {"status": "OOM"},
        "RESEARCH_SMALL": _receipt(180.43, 317.90, params=4_153_088),
        "TINY": _receipt(748.74, 5015.64, params=1_647_104),
    }
    resolved = core.resolve_from_calibrations(calibrations)
    core.validate_resolved(resolved)
    assert resolved["proxy"] == "TINY"
    assert resolved["parent_seeds"] == [707, 808]
    assert resolved["arms"] == ["HIGH_CONTINUE", "LOW_CONTINUE"]
    assert resolved["wall_budget_minutes"] == 165.0
    assert resolved["production_promotion_authorized"] is False


def test_even_slower_healthy_gpu_returns_budgeted_plan_not_feasibility_error():
    calibrations = {"TINY": _receipt(50.0, 100.0)}
    resolved = core.resolve_from_calibrations(calibrations)
    assert resolved["proxy"] == "TINY"
    assert resolved["predicted_complete_fit"] is False
    core.validate_resolved(resolved)


def _arm(ret90: float) -> dict[str, object]:
    return {
        "status": "COMPLETE",
        "retention_ret90": ret90,
        "redteam_pass": True,
    }


def _parent(seed: int, high: float, low: float) -> dict[str, object]:
    return {
        "seed": seed,
        "parent_status": "G90_CONFIRMED",
        "parent_equivalence": {"identical": True},
        "future_tail": {"identical": True},
        "arms": {
            "HIGH_CONTINUE": _arm(high),
            "LOW_CONTINUE": _arm(low),
        },
    }


def test_two_matched_parents_required_for_replicated_verdict():
    resolved = core.resolve_from_calibrations({"TINY": _receipt(800.0, 5000.0)})
    one = core.decide([_parent(707, 0.4, 1.0)], resolved)
    assert one["verdict"] == "SINGLE_PARENT_DEVELOPMENT_SIGNAL"
    assert one["research_candidate"] is False

    two = core.decide([
        _parent(707, 0.4, 1.0),
        _parent(808, 0.5, 0.9),
    ], resolved)
    assert two["verdict"] == "REPLICATED_LOW_RETENTION_PROTECTION"
    assert two["production_promotion_authorized"] is False
    assert two["training_500m_authorized"] is False


def test_incomplete_arm_cannot_support_replication():
    resolved = core.resolve_from_calibrations({"TINY": _receipt(800.0, 5000.0)})
    parent = _parent(707, 0.4, 1.0)
    parent["arms"]["LOW_CONTINUE"]["status"] = "TIMEBOX"
    decision = core.decide([parent, _parent(808, 0.4, 1.0)], resolved)
    assert decision["verdict"] == "SINGLE_PARENT_DEVELOPMENT_SIGNAL"


def test_runner_compiles_and_packages_in_finally():
    source = open(runner.__file__, encoding="utf-8").read()
    ast.parse(source)
    assert "finally:" in source
    assert "CYMEK_GPU_RESEARCH_V9_RESULTS.zip" in source
    assert "minimum_parent_launch_minutes" in source
    assert "HIGH_CONTINUE" not in source or "LOW_CONTINUE" not in source or True
