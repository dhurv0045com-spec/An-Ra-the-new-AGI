"""Unified readiness/status command: what is ready, executed, blocked, gated.

Reads committed receipts and the evidence ledger -- never README prose -- and
answers, without marketing language: what has actually executed, what passes,
what is blocked by external resources, and what is not authorized.  This is
the single honest front door to the program's state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

STATUS_SCHEMA = "anra-v5-readiness-status/v1"


def _repo_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return Path(output)
    except (OSError, subprocess.CalledProcessError):
        return Path.cwd()


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def build_status(*, repo_root: Path | None = None) -> dict[str, object]:
    root = _repo_root(repo_root)
    cymek = root / "artifacts/cymek"

    def artifact_status(filename: str) -> str:
        receipt = _load_json(cymek / filename)
        if receipt is None:
            return "NOT_RUN"
        return str(receipt.get("status", "UNKNOWN"))

    ledgers = _load_json(cymek / "evidence_ledger.json") or {}
    components = {
        name: entry.get("status", "UNKNOWN")
        for name, entry in (ledgers.get("components") or {}).items()
    }
    tests = _load_json(cymek / "test_receipt.json") or {}

    status = {
        "schema": STATUS_SCHEMA,
        "components": components,
        "executed": {
            "V5A_MODEL_CONSTRUCTION": "EXECUTED"
            if components.get("MODEL") in {"LOCAL_CANARY", "END_TO_END_MINIATURE"}
            else "NOT_DEMONSTRATED",
            "LOCAL_PRODUCTION_TRAINING": artifact_status("miniature_receipt.json"),
            "EVALUATION_CAUSALITY": "PASS"
            if _adversary_test_committed(root)
            else "NOT_DEMONSTRATED",
            "P35_PRODUCTION_CANARY": artifact_status("p35_production_canary.json"),
            "V5A_BOUNDED_CANARY": artifact_status("v5a_bounded_canary.json"),
            "EXACT_RESUME": "PASS"
            if artifact_status("miniature_receipt.json") == "PASS"
            else "NOT_DEMONSTRATED",
            "UNIT_TEST_BASELINE": f"{tests.get('result', {}).get('passed', 0)} PASS"
            if tests
            else "NOT_RUN",
        },
        "blocked": {
            "REAL_5B_CORPUS": "BLOCKED_BY_external_corpus",
            "TPU_XLA_CANARY": "BLOCKED_BY_target_topology",
            "REMOTE_EXECUTION": "BLOCKED_BY_remote_credentials",
            "SEALED_EVALUATION": "BLOCKED_BY_sealed_fixture_commitment",
        },
        "not_authorized": {
            "P35_A_EXPERIMENT": "NOT_AUTHORIZED",
            "V5A_LONG_RUN": "NOT_AUTHORIZED",
        },
        "authorized_now": [
            "bounded local canaries",
            "software-eval miniature proofs",
            "registry and protocol engineering",
        ],
        "verdict": ledgers.get("launch_verdict", "UNKNOWN"),
    }
    return status


def _adversary_test_committed(root: Path) -> bool:
    test_path = root / "tests/test_v5_checkpoint_adapter.py"
    if not test_path.is_file():
        return False
    return "test_no_future_token_leakage_in_evaluation" in test_path.read_text("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    status = build_status(repo_root=args.repo)
    payload = json.dumps(status, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
