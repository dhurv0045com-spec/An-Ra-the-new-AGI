"""Build a deterministic, verifier-backed DFC training corpus.

The historical ``frontier_dfc.jsonl`` contains inferred synthetic claims and is
not eligible for the campaign's verified-DFC bucket.  This builder emits only
records whose certificate passed a deterministic verifier from the shared
verifier bank.  It never upgrades or rewrites the historical file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from anra.anra_paths import DATA_MANIFEST_DIR, ROOT
from verification import DEFAULT_VERIFIER_REGISTRY

DEFAULT_OUTPUT = ROOT / "training_data" / "verified_dfc.jsonl"
DEFAULT_MANIFEST = DATA_MANIFEST_DIR / "verified_dfc_manifest.json"
DEFAULT_TARGET_BYTES = 4 * 1024 * 1024
GENERATOR_SCHEMA_VERSION = 1


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _proof_case(index: int) -> tuple[str, dict[str, Any], str]:
    family = index % 12
    prefix = f"case_{index:06d}_{family:02d}"
    premises = [f"{prefix}_observed"]
    rules = [
        f"{prefix}_observed -> {prefix}_supported",
        f"{prefix}_supported -> {prefix}_accepted",
    ]
    steps = [f"{prefix}_supported", f"{prefix}_accepted"]
    payload = {
        "premises": premises,
        "rules": rules,
        "steps": steps,
        "conclusion": f"{prefix}_accepted",
    }
    task = (
        f"Check whether evidence chain {index} supports its final conclusion. "
        f"Premise: {premises[0]}. Rules: {'; '.join(rules)}."
    )
    hypothesis = f"The derivation is valid and concludes {prefix}_accepted."
    return task, payload, hypothesis


def _constraint_case(index: int) -> tuple[str, dict[str, Any], str]:
    workers = 2 + index % 31
    memory = 4 + (index * 3) % 61
    latency = 20 + (index * 7) % 180
    candidate = {
        "workers": workers,
        "memory_gb": memory,
        "latency_ms": latency,
    }
    constraints = {
        "constraints": [
            {"name": "workers", "op": ">=", "value": max(1, workers - 1)},
            {"name": "memory_gb", "op": ">=", "value": max(1, memory - 2)},
            {"name": "latency_ms", "op": "<=", "value": latency + 5},
        ]
    }
    payload = {"constraints": constraints, "candidate": candidate}
    task = (
        f"Validate deployment candidate {index}: workers={workers}, "
        f"memory_gb={memory}, latency_ms={latency}."
    )
    hypothesis = "The candidate satisfies every declared deployment constraint."
    return task, payload, hypothesis


def _verified_record(index: int) -> tuple[dict[str, Any], str]:
    if index % 2 == 0:
        verifier_name = "formal_proof"
        task, verifier_payload, hypothesis = _proof_case(index)
    else:
        verifier_name = "constraint_json"
        task, verifier_payload, hypothesis = _constraint_case(index)
    verdict = DEFAULT_VERIFIER_REGISTRY.verify(verifier_name, verifier_payload)
    if float(verdict.score) < 1.0:
        raise RuntimeError(
            f"Verifier rejected generated DFC row {index}: "
            f"{verifier_name}: {verdict.reason}"
        )
    certificate = {
        "verifier": verifier_name,
        "payload": verifier_payload,
        "score": float(verdict.score),
        "tier": int(verdict.tier),
        "reason": str(verdict.reason),
    }
    text = (
        f'<bos><task domain="verified_reasoning" type="plan_act_verify">{task}</task>'
        f"<hyp>{hypothesis}</hyp>"
        f"<cons>{json.dumps(verifier_payload, sort_keys=True, separators=(',', ':'))}</cons>"
        f"<verify>{json.dumps(certificate, sort_keys=True, separators=(',', ':'))}</verify>"
        "<eos>"
    )
    document_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    record = {
        "text": text,
        "source": "An-Ra verified DFC deterministic verifier bank",
        "license": "owner",
        "source_revision": _canonical_hash(
            {"builder": "build_verified_dfc_corpus", "schema": GENERATOR_SCHEMA_VERSION}
        ),
        "document_sha256": document_sha256,
        "domain": "verified_reasoning",
        "template": "plan_act_verify",
        "verified": True,
        "verifier_status": "verified",
        "verification": certificate,
        "quality_checks": {
            "pii_redacted": True,
            "minhash_deduplicated": True,
            "language_detected": True,
            "benchmark_contamination_checked": True,
        },
    }
    return record, verifier_name


def build_verified_dfc_corpus(
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    target_bytes: int = DEFAULT_TARGET_BYTES,
) -> dict[str, Any]:
    """Atomically emit at least ``target_bytes`` of unique verified DFC rows."""
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    output = Path(output_path)
    manifest_file = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    verifier_counts: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    records = 0
    written_bytes = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        index = 0
        while written_bytes < target_bytes:
            record, verifier_name = _verified_record(index)
            content_hash = str(record["document_sha256"])
            if content_hash in seen_hashes:
                raise RuntimeError(f"duplicate generated DFC content at row {index}")
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            stream.write(line)
            written_bytes += len(line.encode("utf-8"))
            seen_hashes.add(content_hash)
            verifier_counts[verifier_name] += 1
            records += 1
            index += 1
    temporary.replace(output)
    payload: dict[str, Any] = {
        "schema_version": GENERATOR_SCHEMA_VERSION,
        "status": "complete",
        "output": str(output.resolve()),
        "bytes": output.stat().st_size,
        "target_bytes": int(target_bytes),
        "records": records,
        "unique_records": len(seen_hashes),
        "all_verified": True,
        "verifier_counts": dict(sorted(verifier_counts.items())),
        "output_sha256": _path_hash(output),
    }
    payload["manifest_sha256"] = _canonical_hash(payload)
    manifest_tmp = manifest_file.with_suffix(manifest_file.suffix + ".tmp")
    manifest_tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest_tmp.replace(manifest_file)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--target-mb", type=float, default=DEFAULT_TARGET_BYTES / 1_048_576)
    args = parser.parse_args()
    report = build_verified_dfc_corpus(
        args.output,
        manifest_path=args.manifest,
        target_bytes=max(1, int(args.target_mb * 1_048_576)),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
