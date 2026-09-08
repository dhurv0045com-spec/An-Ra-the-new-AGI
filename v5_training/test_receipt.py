"""Machine-readable development test receipt bound to an exact commit.

A receipt records what ran (commands, files, pass/fail/skip, device,
environment, timestamp) against ``tested_commit_sha``. Verification is
exact: the working tree at verify time must differ from the tested commit
in NO tracked file except test-receipt artifacts themselves (any
``artifacts/v5/*_test_receipt.json`` — receipts are evidence, not
executable code, and multiple experiments may add receipts between the
tested commit and HEAD). Any executable change after generation makes
the receipt stale by construction — re-run and regenerate, never
hand-edit counts.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping


RECEIPT_SCHEMA = "anra-v5-test-receipt/v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ValueError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def build_receipt(*, tested_commit_sha: str,
                  results: list[dict[str, object]],
                  environment: Mapping[str, Any]) -> dict[str, object]:
    """Assemble a receipt from per-command results (no execution here)."""

    if len(tested_commit_sha) != 40 or any(
            c not in "0123456789abcdef" for c in tested_commit_sha):
        raise ValueError("tested commit must be a full lowercase git SHA-1")
    total_passed = sum(int(item["passed"]) for item in results)
    total_failed = sum(int(item["failed"]) for item in results)
    total_skipped = sum(int(item.get("skipped", 0)) for item in results)
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "tested_commit_sha": tested_commit_sha,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": dict(environment),
        "results": [dict(item) for item in results],
        "totals": {"passed": total_passed, "failed": total_failed,
                   "skipped": total_skipped},
    }
    receipt["sha256"] = hashlib.sha256(_canonical_json(receipt)).hexdigest()
    return receipt


def write_receipt(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(receipt), indent=2, sort_keys=True)
                      + "\n", encoding="utf-8")
    return target


def read_receipt(path: str | Path) -> dict[str, object]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"test receipt unreadable: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("not a test receipt")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if hashlib.sha256(_canonical_json(payload)).hexdigest() != document.get("sha256"):
        raise ValueError("test receipt hash mismatch")
    return document


def _is_receipt_artifact(line: str) -> bool:
    normalized = line.split()[-1].replace("\\", "/") if line.strip() else line
    return normalized.startswith("artifacts/v5/") \
        and normalized.endswith("_test_receipt.json")


def verify_receipt(path: str | Path, *, repo_root: str | Path,
                   receipt_relpath: str) -> dict[str, object]:
    """Prove the receipt describes the current tree exactly.

    Passes iff the receipt hash is intact AND the working tree differs from
    the tested commit only in test-receipt artifacts (its own by name, any
    sibling ``artifacts/v5/*_test_receipt.json`` by kind). Anything else —
    any code, test, or config change — reports STALE.
    """

    receipt = read_receipt(path)
    tested = str(receipt["tested_commit_sha"])
    head = _git(Path(repo_root), "rev-parse", "HEAD")
    if head != tested:
        changed = _git(Path(repo_root), "diff", "--name-only", tested, "HEAD").splitlines()
        changed.extend(_git(Path(repo_root), "status", "--porcelain").splitlines())
        changed = sorted({line.split()[-1].replace("\\", "/") for line in changed if line.strip()})
        nontrivial = [line for line in changed
                      if line != receipt_relpath and not _is_receipt_artifact(line)]
        if nontrivial:
            raise ValueError(
                f"test receipt is STALE: tested {tested[:8]}, HEAD {head[:8]}, "
                f"code changed: {nontrivial[:5]}")
    uncommitted = [line for line in
                   _git(Path(repo_root), "status", "--porcelain").splitlines()
                   if line.strip() and not _is_receipt_artifact(line)]
    if uncommitted:
        raise ValueError(
            f"test receipt is STALE: uncommitted changes: {uncommitted[:5]}")
    return receipt


__all__ = ["RECEIPT_SCHEMA", "build_receipt", "read_receipt",
           "verify_receipt", "write_receipt"]
