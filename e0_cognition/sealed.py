"""Create a public commitment for a sealed fixture held outside the repository."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def build_commitment(*, fixture: Path, repository: Path, custody_id: str) -> dict[str, object]:
    fixture = fixture.resolve()
    repository = repository.resolve()
    if fixture.is_relative_to(repository):
        raise ValueError("sealed fixture must be held outside the repository")
    raw = fixture.read_bytes()
    parsed = json.loads(raw)
    if parsed.get("split") != "sealed":
        raise ValueError("fixture must declare split=sealed")
    if parsed.get("schema") != "esoes-e0-suite/v1":
        raise ValueError("unexpected sealed suite schema")
    if not custody_id.strip():
        raise ValueError("custody id is required")
    return {
        "schema": "esoes-e0-sealed-commitment/v1",
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "fixture_bytes": len(raw),
        "suite_schema": parsed["schema"],
        "generator_version": parsed.get("generator_version", "unknown"),
        "split": "sealed",
        "custody_id": custody_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_or_seed_in_repository": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--custody-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/e0/sealed_commitment.json"))
    args = parser.parse_args()
    commitment = build_commitment(
        fixture=args.fixture, repository=args.repository, custody_id=args.custody_id
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(commitment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMMITTED", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
