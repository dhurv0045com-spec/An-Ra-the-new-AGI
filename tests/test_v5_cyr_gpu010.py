from __future__ import annotations

import ast

from v5_experiments import cyr_gpu010 as core


def _rec(*, ups: float, tps: float, eps: float, status: str = "PASS"):
    return {"status": status, "training_updates_per_sec": ups,
            "training_real_tokens_per_sec": tps,
            "generation_examples_per_sec": eps}


def test_operator_t4_calibration_selects_research_small_long_box():
    cals = {
        "RESEARCH_SMALL": _rec(ups=1.886815630995512, tps=241.51240076742553, eps=908.6751643910426),
        "TINY": _rec(ups=6.668225801587173, tps=853.5329026031582, eps=5443.507197242657),
    }
    r = core.resolve_from_calibrations(cals)
    core.validate_resolved(r)
    assert r["proxy"] == "RESEARCH_SMALL"
    assert r["max_acquisition_updates"] == 18_000
    assert r["scale_match_to_arkenstone"] is True
    assert r["wall_budget_minutes"] == 175.0


def test_slow_research_small_falls_back_to_long_tiny_instead_of_error():
    cals = {"RESEARCH_SMALL": _rec(ups=0.8, tps=100.0, eps=500.0),
            "TINY": _rec(ups=6.0, tps=750.0, eps=4000.0)}
    r = core.resolve_from_calibrations(cals)
    assert r["proxy"] == "TINY"
    assert r["max_acquisition_updates"] == 36_000
    core.validate_resolved(r)


def test_reasoning_battery_is_deterministic_and_has_structural_sets():
    splits = core.render_t2_worlds()
    a = core.make_reasoning_battery(splits); b = core.make_reasoning_battery(splits)
    assert a["_manifest"] == b["_manifest"]
    assert len(a["STANDARD"]) == 112
    assert len(a["COMMUTED"]) == 64
    assert len(a["LOCALITY"]) >= 40
    assert len(a["RENDERING"]) == 48
    assert len(a["THREE_DIGIT"]) == 48


def test_stress_rule_is_hardware_only_and_bounded():
    assert core.stress_steps_from_remaining(remaining_seconds=60, updates_per_sec=2.0) == 0
    value = core.stress_steps_from_remaining(remaining_seconds=3600, updates_per_sec=2.0)
    assert core.CYR10_STRESS_MIN_STEPS_PER_ARM <= value <= core.CYR10_STRESS_MAX_STEPS_PER_ARM


def test_no_production_authority():
    d = core.classify_acquisition(g90_confirm_update=12000, final_standard=0.95,
                                  final_locality=0.9, proxy="RESEARCH_SMALL", updates=12000)
    assert d["verdict"] == "G90_WITH_COUNTERFACTUAL_LOCALITY"
    assert d["production_promotion_authorized"] is False
    assert d["pre500m_authorized"] is False
    assert d["training_500m_authorized"] is False


def test_runner_source_compiles_and_packages_in_finally():
    import anra_v5.cyr_gpu010_run as runner
    source = open(runner.__file__, encoding="utf-8").read()
    ast.parse(source)
    assert "finally:" in source
    assert "CYMEK_GPU_RESEARCH_V10_RESULTS.zip" in source
