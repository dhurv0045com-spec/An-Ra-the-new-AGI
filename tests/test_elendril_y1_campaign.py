"""Campaign coordinator, cumulative budget, analyzer, and Kaggle notebook tests for AN-RA V8-Y1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from anra_v5.elendril.y1_protocol import PROTOCOL_SHA, effective_protocol_digest
from tools.analyze_elendril_y1_results import analyze_y1_archive
from tools.build_elendril_y1_notebook import build_y1_notebook
from tools.elendril_y1_campaign import (
    _load_or_init_budget_ledger,
    _update_budget_ledger,
    run_campaign,
)


def _fake_args(tmp: Path, *, phase: str, source: str, dry_run: bool = False, cpu_canary: bool = False, **kw) -> argparse.Namespace:
    defaults = dict(
        phase=phase,
        source=source,
        output=tmp,
        minutes=None,
        checkpoint_archive=None,
        handoff_archive=None,
        resume_archive=None,
        selection_receipt=None,
        owner_enabled=True,
        enable_d2=False,
        dry_run=dry_run,
        cpu_canary=cpu_canary,
        canary_updates=1,
        skip_source_check=True,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_y1_dry_run_input_guards_and_cumulative_budget(tmp_path: Path) -> None:
    src = "64ed23a433ebb1ce2f620e0db8dcc24f4d8c23be"
    # Reject missing required inputs before CUDA allocation
    with pytest.raises(ValueError, match="checkpoint-archive"):
        run_campaign(_fake_args(tmp_path / "d1", phase="diagnose", source=src))
    with pytest.raises(ValueError, match="followup requires"):
        run_campaign(_fake_args(tmp_path / "f1", phase="followup", source=src))
    with pytest.raises(ValueError, match="not both"):
        run_campaign(_fake_args(tmp_path / "m1", phase="core", source=src, resume_archive=tmp_path / "a.zip", handoff_archive=tmp_path / "b.zip"))

    # Overrides mint non-production protocol digest cleanly
    p_prod, is_p = effective_protocol_digest(None)
    p_ovr, is_o = effective_protocol_digest({"cpu_canary": True})
    assert p_prod == PROTOCOL_SHA and is_p is True
    assert p_ovr != PROTOCOL_SHA and is_o is False

    # Prepare --dry-run runs without GPU and verifies custody
    dry_out = tmp_path / "prep_dry"
    res = run_campaign(_fake_args(dry_out, phase="prepare", source=src, dry_run=True))
    assert res["status"] == "DRY_RUN_READY"
    assert res["custody"]["status"] == "PASS"
    assert (dry_out / "analysis" / "heuristics" / "development.json").is_file()

    # Cumulative budget never resets across resumed sessions
    broot = tmp_path / "budget"
    led = _load_or_init_budget_ledger(broot, phase="core", ceiling_minutes=360)
    led = _update_budget_ledger(broot, led, phase="core", session_elapsed_s=21000.0)
    led2 = _load_or_init_budget_ledger(broot, phase="core", ceiling_minutes=360)
    assert led2["cumulative_seconds_by_phase"]["core"] == pytest.approx(21000.0)
    led2 = _update_budget_ledger(broot, led2, phase="core", session_elapsed_s=600.0)
    with pytest.raises(ValueError, match="cumulative budget exhausted"):
        _load_or_init_budget_ledger(broot, phase="core", ceiling_minutes=360)


def test_y1_canary_e2e_and_offline_analyzer(tmp_path: Path) -> None:
    src = "64ed23a433ebb1ce2f620e0db8dcc24f4d8c23be"
    out = tmp_path / "canary_e2e"
    res = run_campaign(_fake_args(out, phase="canary", source=src, cpu_canary=True, canary_updates=1))
    assert res["delivery_status"] == "PASS"
    assert (out / "exports" / "ELENDRIL_V8_Y1_SUMMARY.zip").is_file()
    assert (out / "exports" / "ELENDRIL_V8_Y1_EVIDENCE.zip").is_file()
    assert (out / "exports" / "ELENDRIL_V8_Y1_FULL.zip").is_file()
    assert (out / "exports" / "FINAL_STATUS.json").is_file()

    # Shortened canary archive cannot pass off as production by default, and replays cleanly when explicitly permitted.
    with pytest.raises(ValueError, match="protocol SHA mismatch"):
        analyze_y1_archive(out / "exports" / "ELENDRIL_V8_Y1_EVIDENCE.zip")
    metrics = analyze_y1_archive(out / "exports" / "ELENDRIL_V8_Y1_EVIDENCE.zip", allow_non_production=True)
    assert metrics["identity"]["phase"] == "canary"
    assert metrics["production_qualified"] is False


def test_y1_notebook_compiles_and_fails_cleanly_when_auto_archive_missing() -> None:
    src = "64ed23a433ebb1ce2f620e0db8dcc24f4d8c23be"
    nb = build_y1_notebook(src)
    assert len(nb["cells"]) == 4
    setup_src = "".join(nb["cells"][1]["source"])
    run_src = "".join(nb["cells"][2]["source"])
    links_src = "".join(nb["cells"][3]["source"])
    compile(setup_src, "<y1-setup>", "exec")
    compile(run_src, "<y1-run>", "exec")
    compile(links_src, "<y1-links>", "exec")

    # Running PHASE='auto' without /kaggle/input/ ELENDRIL_HITCHHIKER_FULL.zip must fail fast (never train silently).
    proc = subprocess.run(
        [sys.executable, "-c", setup_src],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "ELENDRIL_HITCHHIKER_FULL.zip" in proc.stderr


def test_y1_all_phase_single_session_and_checkpoint_pruning(tmp_path: Path) -> None:
    hh_zip = Path("C:/Users/ankit/Downloads/ELENDRIL_HITCHHIKER_FULL.zip")
    if not hh_zip.is_file():
        pytest.skip("ELENDRIL_HITCHHIKER_FULL.zip not present on local machine")
    src = "64ed23a433ebb1ce2f620e0db8dcc24f4d8c23be"
    out = tmp_path / "all_e2e"
    res = run_campaign(
        _fake_args(
            out,
            phase="all",
            source=src,
            checkpoint_archive=hh_zip,
            cpu_canary=True,
            canary_updates=3,
        )
    )
    assert res["status"] == "COMPLETE"
    assert res["delivery_status"] == "PASS"
    assert not (out / ".tmp_hh_restore").exists()
    assert (out / "analysis" / "d0_d1_summary.json").is_file()
    assert (out / "analysis" / "gates" / "t1_gate.json").is_file()
    assert (out / "analysis" / "gates" / "selection_receipt.json").is_file()
    assert (out / "analysis" / "followup_forecast_receipt.json").is_file()
    assert (out / "exports" / "ELENDRIL_V8_Y1_SUMMARY.zip").is_file()
    assert (out / "exports" / "ELENDRIL_V8_Y1_EVIDENCE.zip").is_file()
    assert (out / "exports" / "ELENDRIL_V8_Y1_FULL.zip").is_file()
    assert (out / "exports" / "FINAL_STATUS.json").is_file()

    # Verify checkpoint pruning kept at most 2 generations (LATEST + PREVIOUS) per job despite 3 canary updates
    for job_cp in (out / "checkpoints").iterdir():
        objs = list((job_cp / "objects").iterdir())
        assert 1 <= len(objs) <= 2
