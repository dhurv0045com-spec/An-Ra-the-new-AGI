"""Execute the corpus-to-tokenizer Stream-B critical path fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from anra.anra_paths import DATA_MANIFEST_DIR, OUTPUT_V2_DIR, ROOT
from training.v2_config import EXPECTED_SPECIAL_TOKEN_IDS, EXPECTED_TOKENIZER_VOCAB_SIZE
from training.v2_runtime import active_tokenizer_identity, load_or_build_v2_tokenizer

from scripts.build_campaign_slice import (
    CAMPAIGN_SLICE_DIR,
    _default_sources,
    build_campaign_slice,
)
from scripts.download_training_data import publish_fineweb_token_shards

FOUNDATION_CORPUS = ROOT / "training_data" / "foundation_records.jsonl"
FOUNDATION_AUDIT = OUTPUT_V2_DIR / "foundation_records_audit.json"
DOWNLOAD_STATUS = DATA_MANIFEST_DIR / "download_status.json"
STREAM_B_REPORT = OUTPUT_V2_DIR / "stream_b_execution.json"
V4_TOKENIZER = ROOT / "tokenizer" / "tokenizer_v4_32k.json"
EXPECTED_NATIVE_SOURCES = frozenset(
    {
        "FineWeb-Edu",
        "Common Pile Stack v2 open code",
        "FineMath-4+",
        "Common Pile ArXiv science/technical",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Required JSON artifact is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Required JSON artifact is not an object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_canonical_v4() -> dict[str, Any]:
    """Prove the one checked-in tokenizer satisfies the active V4 contract."""
    tokenizer = load_or_build_v2_tokenizer()
    identity = active_tokenizer_identity()
    expected_ids = {token: int(token_id) for token, token_id in EXPECTED_SPECIAL_TOKEN_IDS.items()}
    if identity.get("available") is not True:
        raise RuntimeError("Canonical V4 tokenizer is unavailable")
    if int(identity.get("schema_version", -1)) != 4:
        raise RuntimeError("Canonical tokenizer does not declare schema version 4")
    if int(identity.get("vocab_size", -1)) != EXPECTED_TOKENIZER_VOCAB_SIZE:
        raise RuntimeError("Canonical tokenizer is not the required 32,768-token V4")
    if identity.get("special_token_ids") != expected_ids:
        raise RuntimeError("Canonical V4 special-token IDs changed")
    if int(identity.get("probe_count", 0)) != 500:
        raise RuntimeError("Canonical V4 behavior fingerprint is incomplete")
    if tokenizer.vocab_size != EXPECTED_TOKENIZER_VOCAB_SIZE:
        raise RuntimeError("Loaded tokenizer and V4 identity disagree")
    return {
        "status": "validated",
        **identity,
    }


def validate_native_foundation() -> dict[str, Any]:
    """Require a complete matching audit and all four native source classes."""
    if not FOUNDATION_CORPUS.is_file():
        raise RuntimeError(f"Native foundation corpus is missing: {FOUNDATION_CORPUS}")
    status = _read_json(DOWNLOAD_STATUS)
    audit = _read_json(FOUNDATION_AUDIT)
    corpus_bytes = FOUNDATION_CORPUS.stat().st_size
    source_stats = audit.get("source_stats")
    if status.get("status") != "complete":
        raise RuntimeError("Native foundation downloader has not completed successfully")
    requested_buckets = status.get("requested_buckets")
    bucket_results = status.get("buckets")
    recorded_buckets = {
        str(item.get("bucket"))
        for item in bucket_results
        if isinstance(item, dict) and item.get("bucket")
    } if isinstance(bucket_results, list) else set()
    if not (
        isinstance(requested_buckets, list)
        and "base" in requested_buckets
        and "base" in recorded_buckets
    ):
        raise RuntimeError("Download status does not prove a completed base bucket")
    if audit.get("resume_safe") is not True or audit.get("target_complete") is not True:
        raise RuntimeError("Native foundation audit is not complete and resume-safe")
    if int(audit.get("corpus_size_bytes", -1)) != corpus_bytes:
        raise RuntimeError("Native foundation bytes changed after the published audit")
    if not isinstance(source_stats, dict):
        raise RuntimeError("Native foundation audit has no per-source evidence")
    missing = sorted(EXPECTED_NATIVE_SOURCES - set(source_stats))
    empty = sorted(
        source
        for source in EXPECTED_NATIVE_SOURCES
        if int(dict(source_stats.get(source, {})).get("bytes", 0)) <= 0
    )
    failures = audit.get("failures")
    if missing or empty:
        raise RuntimeError(
            f"Native source coverage failed; missing={missing}, empty={empty}"
        )
    if not isinstance(failures, dict) or any(int(value) for value in failures.values()):
        raise RuntimeError(f"Native foundation audit contains failures: {failures}")
    return {
        "corpus": str(FOUNDATION_CORPUS),
        "corpus_bytes": corpus_bytes,
        "audit_sha256": _sha256(FOUNDATION_AUDIT),
        "sources": sorted(EXPECTED_NATIVE_SOURCES),
    }


def _existing_family_inventory(family: str, profile: str) -> dict[str, Any] | None:
    family_root = DATA_MANIFEST_DIR / f"native_foundation_{family}" / profile
    inventory_path = family_root / "token_inventory.json"
    if not inventory_path.is_file():
        if family_root.exists():
            raise RuntimeError(
                f"Partial immutable {family.upper()} shard publication exists: {family_root}"
            )
        return None
    inventory = _read_json(inventory_path)
    if inventory.get("tokenizer_family") != family:
        raise RuntimeError(f"Shard inventory family mismatch: {inventory_path}")
    for key in ("manifest", "validation_manifest", "test_manifest"):
        path = Path(str(inventory.get(key, "")))
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        if not path.is_file():
            raise RuntimeError(f"Shard inventory references missing {key}: {path}")
    return inventory


def execute_stream_b(*, profile: str = "30gb", publish_shards: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": time.time(),
        "profile": profile,
        "stages": {},
    }
    _atomic_json(STREAM_B_REPORT, report)

    report["stages"]["native_foundation"] = validate_native_foundation()
    _atomic_json(STREAM_B_REPORT, report)

    slice_report = build_campaign_slice(_default_sources(), CAMPAIGN_SLICE_DIR)
    report["stages"]["campaign_slice"] = slice_report
    _atomic_json(STREAM_B_REPORT, report)
    if slice_report.get("ready_for_v4") is not True:
        raise RuntimeError("Seven-source campaign slice did not pass the V4 gate")

    v4_report = validate_canonical_v4()
    report["stages"]["canonical_v4"] = v4_report
    _atomic_json(STREAM_B_REPORT, report)

    if publish_shards:
        for family, tokenizer_path in (("v4", V4_TOKENIZER),):
            inventory = _existing_family_inventory(family, profile)
            if inventory is None:
                inventory = publish_fineweb_token_shards(
                    profile,
                    tokenizer_path=tokenizer_path,
                    tokenizer_family=family,
                )
            report["stages"][f"{family}_token_shards"] = inventory
            _atomic_json(STREAM_B_REPORT, report)

    report["status"] = "complete"
    report["completed_at"] = time.time()
    _atomic_json(STREAM_B_REPORT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="30gb")
    parser.add_argument("--skip-shards", action="store_true")
    args = parser.parse_args()
    try:
        report = execute_stream_b(
            profile=args.profile,
            publish_shards=not args.skip_shards,
        )
    except Exception as exc:
        report = (
            _read_json(STREAM_B_REPORT)
            if STREAM_B_REPORT.is_file()
            else {"schema_version": 1, "stages": {}}
        )
        report["status"] = "failed"
        report["failed_at"] = time.time()
        report["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(STREAM_B_REPORT, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 3
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
