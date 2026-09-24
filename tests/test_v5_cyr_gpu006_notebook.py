"""Runner regressions retained after the superseded V6 notebook was retired."""

import importlib
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "anra_v5" / "cyr_gpu006_run_final.py"


def test_final_runner_imports_without_optional_accelerator_runtime():
    module = importlib.import_module("anra_v5.cyr_gpu006_run_final")
    assert callable(module.run_campaign)
    assert callable(module.calibrate_candidates)
    assert callable(module.transfer_plasticity_replicated)


def test_final_runner_attempts_all_parent_seeds_without_first_g90_break():
    source = RUNNER.read_text("utf-8")
    marker = "for seed in core.CYR6_PARENT_SEEDS:"
    assert marker in source
    block = source[source.index(marker):source.index('campaign["parents"]', source.index(marker))]
    assert "break" not in block


def test_final_runner_full_mode_has_no_cpu_fallback():
    source = RUNNER.read_text("utf-8")
    assert 'device = torch.device("cuda")' in source
    assert "refuses non-CUDA device" in source


def test_final_runner_consumes_resolved_plan_instead_of_recalibrating():
    source = RUNNER.read_text("utf-8")
    run_source = source[source.index("def run_campaign("):]
    assert "resolved = core.validate_resolved(resolved)" in run_source
    assert "calibrate_candidates(" not in run_source
    assert "resolve_from_calibrations(" not in run_source


def test_transfer_is_equal_age_hyst_vs_low_not_parent_vs_postcontinuation():
    source = RUNNER.read_text("utf-8")
    assert 'CYR6_TRANSFER_CANDIDATE' in source
    assert 'CYR6_TRANSFER_COMPARATOR' in source
    assert 'candidate/comparator continuation exposure differs' in source
    assert 'source_continuation_tokens' in source
