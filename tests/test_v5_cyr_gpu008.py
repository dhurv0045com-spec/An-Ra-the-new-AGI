from __future__ import annotations

import ast

from anra_v5 import cyr_gpu008_run as runner
from v5_experiments import cyr_gpu006_final as core6
from v5_experiments import cyr_gpu008 as core


def _receipt(tps: float, eps: float) -> dict[str, object]:
    return {
        "status": "PASS",
        "training_real_tokens_per_sec": tps,
        "generation_examples_per_sec": eps,
        "peak_vram_gb": 1.0,
    }


def test_non_tiny_scale_is_preferred_before_tiny_fallback():
    calibrations = {
        "MIDI": {"status": "OOM"},
        "MICRO": {"status": "OOM"},
        "RESEARCH_SMALL": _receipt(1200.0, 10.0),
        "TINY": _receipt(3000.0, 30.0),
    }
    resolved = core.resolve_from_calibrations(calibrations)
    assert resolved["proxy"] == "RESEARCH_SMALL"
    assert resolved["tier"] == "RETENTION_2P_ONLY"
    assert resolved["research_candidate_possible"] is False
    core.validate_resolved(resolved)


def test_fast_non_tiny_gets_full_design():
    calibrations = {name: _receipt(3000.0, 30.0) for name in ("MIDI", "MICRO", "RESEARCH_SMALL", "TINY")}
    resolved = core.resolve_from_calibrations(calibrations)
    assert resolved["proxy"] == "MIDI"
    assert resolved["tier"] == "FULL_3P_TRANSFER3"
    assert resolved["parents"] == 3


def test_compatibility_patch_final_decision_does_not_recurse():
    calibrations = {name: _receipt(1200.0, 10.0) for name in ("MIDI", "MICRO", "RESEARCH_SMALL", "TINY")}
    resolved = core.resolve_from_calibrations(calibrations)
    before = core6.final_decision
    with runner._compatibility_contract(resolved):
        decision = core6.final_decision([], transfer=None)
        assert decision["experiment"] == "CYR-GPU-008"
        assert decision["research_candidate"] is False
        assert decision["production_promotion_authorized"] is False
    assert core6.final_decision is before


def test_runner_compiles_and_restores_globals():
    source = open(runner.__file__, encoding="utf-8").read()
    ast.parse(source)
    assert "finally:" in source
    assert "CYMEK_GPU_RESEARCH_V8_RESULTS.zip" in source
    assert "CYR-GPU-008 refuses non-CUDA" in source
