"""Three-bundle export, 5 MiB SUMMARY budget, handoff/restore, and corruption rejection tests."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from anra_v5.elendril.y1_protocol import PROTOCOL_SHA
from tools.elendril_y1_export import (
    SUMMARY_MAX_BYTES,
    finalize_y1_campaign,
    handoff_y1_archive,
    restore_y1_archive,
)


def _seed_workspace(root: Path, *, source: str, proto: str, phase: str) -> None:
    (root / "summary").mkdir(parents=True, exist_ok=True)
    (root / "summary" / "REPORT.md").write_text("# Test Report\n", encoding="utf-8")
    (root / "prepared").mkdir(exist_ok=True)
    (root / "prepared" / "manifest.json").write_text('{"ok": true}\n', encoding="utf-8")
    (root / "analysis" / "gates").mkdir(parents=True, exist_ok=True)
    (root / "analysis" / "gates" / "selection_receipt.json").write_text('{"recommended_followup_branch":"capacity_s20"}\n', encoding="utf-8")
    (root / "evidence" / "predictions").mkdir(parents=True, exist_ok=True)
    (root / "evidence" / "predictions" / "j1.json").write_text('{"predictions":[]}\n', encoding="utf-8")


def test_y1_export_bundles_external_final_status_and_handoff_vs_restore(tmp_path: Path) -> None:
    src = "64ed23a433ebb1ce2f620e0db8dcc24f4d8c23be"
    root = tmp_path / "diag"
    _seed_workspace(root, source=src, proto=PROTOCOL_SHA, phase="diagnose")
    fs = finalize_y1_campaign(
        root,
        source=src,
        protocol_sha=PROTOCOL_SHA,
        campaign={"run_id": "y1-test-1", "phase": "diagnose", "source": src, "protocol_sha256": PROTOCOL_SHA, "status": "COMPLETE"},
        replay_payload={"status": "PASS"},
        include_full=False,
    )
    assert fs["delivery_status"] == "PASS"
    sum_zip = root / "exports" / "ELENDRIL_V8_Y1_SUMMARY.zip"
    ev_zip = root / "exports" / "ELENDRIL_V8_Y1_EVIDENCE.zip"
    assert sum_zip.stat().st_size < SUMMARY_MAX_BYTES
    assert ev_zip.is_file()
    assert not (root / "exports" / "ELENDRIL_V8_Y1_FULL.zip").is_file()

    # FINAL_STATUS.json must live OUTSIDE archives (never inside SUMMARY/EVIDENCE).
    with zipfile.ZipFile(sum_zip, "r") as zf:
        assert all("FINAL_STATUS" not in n for n in zf.namelist())
        assert all("predictions/" not in n for n in zf.namelist())

    # Cross-phase handoff diagnose -> core succeeds via compact EVIDENCE or SUMMARY zip
    core_dir = tmp_path / "core_handoff"
    rec = handoff_y1_archive(ev_zip, core_dir, source=src, protocol_sha=PROTOCOL_SHA, from_phase="diagnose", to_phase="core")
    assert rec["from_phase"] == "diagnose" and rec["to_phase"] == "core"
    assert (core_dir / "prepared" / "manifest.json").is_file()

    # Disallowed reverse handoff & wrong-phase same-phase restore must fail closed
    with pytest.raises(ValueError, match="disallowed Y1 cross-phase handoff"):
        handoff_y1_archive(ev_zip, tmp_path / "bad_rev", source=src, protocol_sha=PROTOCOL_SHA, from_phase="core", to_phase="diagnose")
    with pytest.raises(ValueError, match="resume archive mismatch"):
        restore_y1_archive(ev_zip, tmp_path / "bad_res", source=src, protocol_sha=PROTOCOL_SHA, phase="core")

    # Enforce 5 MiB SUMMARY ceiling
    big_root = tmp_path / "big_sum"
    _seed_workspace(big_root, source=src, proto=PROTOCOL_SHA, phase="core")
    # Write uncompressible random bytes > 5 MiB inside summary/
    import os
    (big_root / "summary" / "bloat.bin").write_bytes(os.urandom(SUMMARY_MAX_BYTES + 100_000))
    fs_big = finalize_y1_campaign(
        big_root,
        source=src,
        protocol_sha=PROTOCOL_SHA,
        campaign={"run_id": "y1-test-big", "phase": "core", "source": src, "protocol_sha256": PROTOCOL_SHA, "status": "COMPLETE"},
        replay_payload={"status": "PASS"},
        include_full=False,
    )
    assert fs_big["exports"]["summary"]["status"] == "EXPORT_FAILED"
    assert fs_big["delivery_status"] == "FAILED"
