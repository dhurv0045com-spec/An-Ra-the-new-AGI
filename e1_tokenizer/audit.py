"""Audit externally produced tokenizer encodings without importing tokenizer libraries."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .probes import PROBES, probe_map


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_receipt(receipt: dict[str, Any], *, artifact_sha256: str) -> dict[str, object]:
    if receipt.get("schema") != "esoes-e1-candidate-encoding/v1":
        raise ValueError("unexpected E1 candidate schema")
    if receipt.get("artifact_sha256") != artifact_sha256:
        raise ValueError("tokenizer artifact hash mismatch")
    vocabulary_size = int(receipt["vocabulary_size"])
    if vocabulary_size <= 256:
        raise ValueError("subword vocabulary must exceed the byte alphabet")
    expected = probe_map()
    rows = receipt.get("encodings", [])
    if {row.get("probe_id") for row in rows} != set(expected):
        raise ValueError("candidate must encode every committed probe exactly once")
    if len(rows) != len(expected):
        raise ValueError("duplicate probe encodings")
    unknown_id = receipt.get("unknown_token_id")
    domain_tokens: dict[str, int] = defaultdict(int)
    domain_bytes: dict[str, int] = defaultdict(int)
    roundtrip_failures: list[str] = []
    unknown_failures: list[str] = []
    range_failures: list[str] = []
    per_probe: dict[str, dict[str, object]] = {}
    for row in rows:
        probe = expected[row["probe_id"]]
        ids = tuple(int(value) for value in row["token_ids"])
        if not ids:
            raise ValueError(f"empty encoding for {probe.probe_id}")
        if row.get("decoded_text") != probe.text:
            roundtrip_failures.append(probe.probe_id)
        if unknown_id is not None and int(unknown_id) in ids:
            unknown_failures.append(probe.probe_id)
        if any(value < 0 or value >= vocabulary_size for value in ids):
            range_failures.append(probe.probe_id)
        byte_count = len(probe.text.encode("utf-8"))
        domain_tokens[probe.domain] += len(ids)
        domain_bytes[probe.domain] += byte_count
        per_probe[probe.probe_id] = {
            "tokens": len(ids),
            "utf8_bytes": byte_count,
            "tokens_per_byte": len(ids) / byte_count,
        }
    total_tokens = sum(domain_tokens.values())
    total_bytes = sum(domain_bytes.values())
    checks = {
        "identity_roundtrip": not roundtrip_failures,
        "zero_unknowns": not unknown_failures,
        "token_ids_in_range": not range_failures,
        "all_probes_present_once": True,
    }
    return {
        "schema": "esoes-e1-static-audit/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "candidate": receipt["tokenizer_name"],
        "vocabulary_size": vocabulary_size,
        "artifact_sha256": artifact_sha256,
        "checks": checks,
        "failures": {
            "roundtrip": roundtrip_failures,
            "unknown": unknown_failures,
            "range": range_failures,
        },
        "metrics": {
            "total_tokens": total_tokens,
            "total_utf8_bytes": total_bytes,
            "tokens_per_byte": total_tokens / total_bytes,
            "tokens_per_byte_by_domain": {
                domain: domain_tokens[domain] / domain_bytes[domain]
                for domain in sorted(domain_tokens)
            },
            "per_probe": per_probe,
        },
        "limitations": [
            "Committed probes are canaries, not the serious external E1 corpus.",
            "Static compression/identity results do not select a tokenizer without matched model training.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    report = audit_receipt(receipt, artifact_sha256=_sha256(args.artifact))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
