"""Section 49: the CYR-GPU-005 test receipt must be bound to the exact
tree it verifies. If the executable changes, re-run and regenerate."""

from __future__ import annotations

from pathlib import Path

from v5_training.test_receipt import verify_receipt

ROOT = Path(__file__).resolve().parents[1]


def test_cyr_gpu_005_test_receipt_exact_head():
    receipt = verify_receipt(
        ROOT / "artifacts/v5/cyr_gpu_005_test_receipt.json",
        repo_root=ROOT,
        receipt_relpath="artifacts/v5/cyr_gpu_005_test_receipt.json")
    assert receipt["totals"]["failed"] == 0
    assert receipt["totals"]["passed"] > 0
    assert any("cyr_gpu005" in str(result.get("command", ""))
               or "cyr_gpu005" in str(result.get("files", []))
               for result in receipt["results"]), \
        "receipt must record the CYR-GPU-005 suites"
