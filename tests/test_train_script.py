"""Smoke test for scripts/train.py — verifies it runs without crashing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_train_script_smoke_tiny():
    """Run training for 3 steps with synthetic data. Must not crash."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            "config/tiny.yaml",
            "--max_steps",
            "3",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
    )
    assert result.returncode == 0, (
        f"Training script crashed!\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "step" in result.stdout.lower() or "loss" in result.stdout.lower(), (
        f"Training script produced unexpected output:\n{result.stdout}"
    )


def test_train_script_creates_metrics_file(tmp_path):
    """Training must write a metrics .jsonl file."""
    import json as _json
    import shutil

    del shutil
    del tmp_path

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            "config/tiny.yaml",
            "--max_steps",
            "5",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
    )
    assert result.returncode == 0, f"STDERR:\n{result.stderr}"
    metrics = REPO / "output" / "metrics" / "tiny.jsonl"
    assert metrics.exists(), f"Metrics file not created at {metrics}"
    lines = [line for line in metrics.read_text().splitlines() if line.strip()]
    assert len(lines) > 0
    record = _json.loads(lines[0])
    assert "step" in record and "train_loss" in record
