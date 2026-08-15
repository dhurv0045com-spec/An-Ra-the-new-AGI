from __future__ import annotations

from pathlib import Path

from anra.anra_paths import ENGINEERING_LOG_FILE, V4_ARCHITECTURE_GATE_FILE


def test_engineering_log_exists():
    assert ENGINEERING_LOG_FILE.exists()
    text = ENGINEERING_LOG_FILE.read_text(encoding="utf-8")
    assert "LOG_STANDARD" in text or "2026-" in text


def test_v4_architecture_gate_exists():
    assert V4_ARCHITECTURE_GATE_FILE.exists()
    text = V4_ARCHITECTURE_GATE_FILE.read_text(encoding="utf-8")
    assert "An-Ra V4 Architecture Gate" in text
    assert "Verification evidence" in text


def test_log_script_dry_run():
    import subprocess
    import sys

    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "log_engineering_change.py"),
            "--component",
            "tests",
            "--type",
            "FIX",
            "--title",
            "dry-run probe",
            "--summary",
            "test only",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    assert proc.returncode == 0
    assert "dry-run" in proc.stdout.lower() or "DATE" in proc.stdout
