from __future__ import annotations

import ast
from pathlib import Path

import pytest

from v5_experiments import cyr_gpu007 as core

ROOT = Path(__file__).resolve().parents[1]


def _receipt(tps: float, eps: float) -> dict[str, object]:
    return {
        "status": "PASS",
        "training_real_tokens_per_sec": tps,
        "generation_examples_per_sec": eps,
        "peak_vram_gb": 1.0,
    }


def _all(tps: float, eps: float) -> dict[str, dict[str, object]]:
    return {name: _receipt(tps, eps) for name in core.PROXY_ORDER}


def test_fast_gpu_gets_full_three_parent_transfer_tier():
    resolved = core.resolve_from_calibrations(_all(3000.0, 30.0))
    assert resolved["tier"] == "FULL_3P_TRANSFER3"
    assert resolved["parents"] == 3
    assert resolved["transfer_enabled"] is True
    assert resolved["transfer_target_parents"] == 3
    core.validate_resolved(resolved)


def test_medium_gpu_gets_two_parent_full_causal_tier():
    resolved = core.resolve_from_calibrations(_all(1500.0, 10.0))
    assert resolved["tier"] == "CORE_2P_TRANSFER2"
    assert resolved["parent_seeds"] == [707, 808]
    assert resolved["transfer_min_parents"] == 2
    core.validate_resolved(resolved)


def test_slower_gpu_preserves_replicated_retention_without_fake_transfer():
    resolved = core.resolve_from_calibrations(_all(1200.0, 10.0))
    assert resolved["tier"] == "RETENTION_2P_ONLY"
    assert resolved["parents"] == 2
    assert resolved["transfer_enabled"] is False
    assert resolved["research_candidate_possible"] is False
    core.validate_resolved(resolved)


def test_tiny_is_last_resort_and_cannot_become_research_candidate():
    calibrations = {
        "MIDI": {"status": "OOM"},
        "MICRO": {"status": "OOM"},
        "RESEARCH_SMALL": {"status": "OOM"},
        "TINY": _receipt(1500.0, 10.0),
    }
    resolved = core.resolve_from_calibrations(calibrations)
    assert resolved["proxy"] == "TINY"
    assert resolved["claim_ceiling"] == "TINY_PROXY_DEVELOPMENT_ONLY"
    assert resolved["research_candidate_possible"] is False
    core.validate_resolved(resolved)


def test_impossibly_slow_gpu_still_fails_closed_with_diagnostics():
    with pytest.raises(ValueError, match="cannot afford"):
        core.resolve_from_calibrations(_all(300.0, 1.0))


def test_retention_only_result_cannot_be_upgraded_to_research_candidate():
    resolved = core.resolve_from_calibrations(_all(1200.0, 10.0))
    decision = core.final_decision([], transfer=None, resolved=resolved)
    assert decision["research_candidate"] is False
    assert decision["production_promotion_authorized"] is False


def test_runner_compiles_and_uses_cuda_and_runtime_tier():
    path = ROOT / "anra_v5" / "cyr_gpu007_run.py"
    source = path.read_text("utf-8")
    ast.parse(source)
    assert "torch.cuda.is_available" in source
    assert "CYR-GPU-007 refuses non-CUDA" in source
    assert 'resolved["parent_seeds"]' in source
    assert "CYMEK_GPU_RESEARCH_V7_RESULTS.zip" in source
    assert "calibrate_candidate" in source
