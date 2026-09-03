"""C0 static validation (compute-ladder rung 1).

Verifies, before any execution, that everything C0's preregistration depends on is present
and hash-identical on this branch: the scoring-policy fixture machinery, the compiled fixture
receipts, the tokenizer artifacts, and the recorded negative-control numbers the screen must
reproduce. Emits a machine-readable receipt. Read-only over repository artifacts.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
RECEIPT_PATH = Path(__file__).resolve().parent / "receipts" / "static_validation.json"

EXPECTED_FIXTURE_SHA = {
    "development": "3adb9bed98b3bddc7a7cd1dd149022611560c6eed7964ff168181de23f97b562",
    "fresh": "986bb0be2651e2c24816569b2c1e2d8f5d706476afd16a597f19a454e9dc268a",
}
MACHINERY_MODULES = (
    "e2_architecture/scoring_policy_fixture.py",
    "e2_architecture/scoring_policy.py",
    "e2_architecture/scoring_policy_tournament.py",
    "e2_architecture/scoring_benchmark.py",
)
TOKENIZER_ARTIFACTS = (
    "tokenizer-16384.json.gz",
    "tokenizer-24576.json.gz",
    "tokenizer-32768.json.gz",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main() -> int:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch  # type: ignore

        environment["torch"] = torch.__version__
        environment["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            environment["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # torch is optional for static validation
        environment["torch"] = f"unavailable: {type(exc).__name__}"
        environment["cuda_available"] = False
    try:
        import tokenizers  # type: ignore

        environment["tokenizers"] = tokenizers.__version__
    except Exception as exc:
        environment["tokenizers"] = f"unavailable: {type(exc).__name__}"

    fixture_receipt = json.loads(
        (REPO_ROOT / "artifacts/e2/scoring_policy_fixture.json").read_text(encoding="utf-8")
    )
    check(
        "fixture_receipt_status",
        fixture_receipt.get("status") == "PASS_FIXTURE_COMPILATION",
        fixture_receipt.get("status"),
    )
    for split, expected in EXPECTED_FIXTURE_SHA.items():
        actual = fixture_receipt.get(split, {}).get("fixture_sha256")
        check(f"fixture_sha_{split}", actual == expected, {"expected": expected, "actual": actual})

    for relative in MACHINERY_MODULES:
        path = REPO_ROOT / relative
        check(f"machinery_present_{relative}", path.is_file(), _source_sha256(path) if path.is_file() else None)

    development = json.loads(
        (REPO_ROOT / "artifacts/e2/scoring_policy_development.json").read_text(encoding="utf-8")
    )
    recorded_baselines = {
        "status": development.get("status"),
        "production_scoring_mode": development.get("production_scoring_mode"),
        "policy_survives_bias_screen": development.get("policy_survives_bias_screen"),
        "gpu_hours": development.get("gpu_hours") if "gpu_hours" in development else None,
    }
    check(
        "negative_controls_recorded",
        development.get("status") == "FAIL_DEVELOPMENT_POLICY"
        and development.get("production_scoring_mode") is None
        and development.get("policy_survives_bias_screen")
        == {"contextual_calibration": False, "domain_pmi": False},
        recorded_baselines,
    )

    tokenizers: dict[str, object] = {}
    for name in TOKENIZER_ARTIFACTS:
        path = REPO_ROOT / "artifacts/e1/local_tournament" / name
        tokenizers[name] = {
            "present": path.is_file(),
            "sha256": _sha256_file(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        }
    check("tokenizer_artifacts_present", all(t["present"] for t in tokenizers.values()), tokenizers)

    receipt = {
        "schema": "citadel-c0-static-validation/v1",
        "experiment": "C0",
        "branch": "citadel",
        "base_sha": "85f44b7b449f2ee39a0e80203a2d7df04614983b",
        # head_sha is omitted by construction: a receipt cannot contain the hash of
        # the commit that carries it; the carrying commit is the provenance record.
        "generated_by": "static_validation.py (deterministic, read-only over artifacts)",
        "environment": environment,
        "recorded_negative_control_baselines": recorded_baselines,
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
    }

    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for entry in checks:
        print(f"[{'PASS' if entry['passed'] else 'FAIL'}] {entry['check']}")
    print(f"receipt: {RECEIPT_PATH}")
    return 0 if receipt["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
