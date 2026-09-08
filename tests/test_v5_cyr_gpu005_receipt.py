"""CYR-GPU-005 is superseded-before-execution historical evidence.

Its test receipt must remain hash-valid and must continue to describe a passing
005 software test run, but it must NOT be treated as an exact-head readiness
receipt after CYR-GPU-006 changed executable code. Current exact-head closure
belongs to the active Cymek readiness receipt only.
"""
from __future__ import annotations

from pathlib import Path

from v5_training.test_receipt import read_receipt

ROOT = Path(__file__).resolve().parents[1]


def test_cyr_gpu_005_test_receipt_is_intact_historical_evidence():
    receipt = read_receipt(ROOT / "artifacts/v5/cyr_gpu_005_test_receipt.json")
    assert receipt["totals"]["failed"] == 0
    assert receipt["totals"]["passed"] > 0
    assert any(
        "cyr_gpu005" in str(result.get("command", ""))
        or "cyr_gpu005" in str(result.get("files", []))
        for result in receipt["results"]
    ), "historical receipt must still identify the CYR-GPU-005 suites"


def test_cyr_gpu_005_is_explicitly_superseded_before_execution():
    text = (ROOT / "docs/cymek/experiments/CYR-GPU-005/SUPERSEDED.md").read_text("utf-8")
    assert "SUPERSEDED_BEFORE_EXECUTION" in text
    assert "No GPU scientific run exists" in text
