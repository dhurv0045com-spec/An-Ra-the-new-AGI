"""Real corpus acceptance boundary: source contracts and dataset qualification.

A source artifact declares identity, format, counts, class, provenance,
quality, license, and dedup state. ``qualify_dataset`` validates source
hashes, document IDs, dedup, cluster split, contamination posture,
tokenizer compatibility, token availability, mixture sufficiency, and
cognition generator qualification, emitting DATASET_QUALIFIED or an
explicit blocker list. No downloader: qualification judges artifacts.
Unique qualified tokens below budget BLOCKS; silent recycling is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SOURCE_SCHEMA = "anra-v5-source-artifact/v1"
QUALIFICATION_SCHEMA = "anra-v5-dataset-qualification/v1"

CENTER_5B_MIXTURE = {"natural": 3_250_000_000, "code_math_formal": 1_000_000_000, "verified_cognition": 750_000_000}
MIXTURE_TOLERANCE_FRACTION = 0.02


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _assert_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    schema: str
    source_id: str
    artifact_sha256: str
    format: str
    document_count: int
    source_class: str
    provenance: str
    quality_status: str
    license: str
    cluster_dedup_status: str

    def assert_valid(self) -> None:
        if self.schema != SOURCE_SCHEMA:
            raise ValueError("unsupported source-artifact schema")
        for name in ("source_id", "format", "source_class", "provenance",
                     "quality_status", "license", "cluster_dedup_status"):
            if not getattr(self, name):
                raise ValueError(f"source artifact field is required: {name}")
        _assert_sha256("artifact", self.artifact_sha256)
        if self.document_count <= 0:
            raise ValueError("source artifact needs a positive document count")


def qualify_dataset(
    *,
    data_manifest: Mapping[str, Any],
    manifest_audit: Mapping[str, Any] | None,
    tokenizer_receipt: Mapping[str, Any],
    expected_tokenizer_sha256: str,
    family_to_slice: Mapping[str, str],
    mixture_targets: Mapping[str, int],
    required_cognition_families: list[str],
    generator_qualifications: Mapping[str, str],
) -> dict[str, object]:
    """Validate a dataset manifest against acceptance gates."""

    checks: dict[str, bool] = {}
    blockers: list[str] = []
    sources = data_manifest.get("sources") or []
    checks["sources_present"] = bool(sources)
    try:
        raw_hashes = [str(source["raw_sha256"]) for source in sources]
        for digest in raw_hashes:
            _assert_sha256("raw source", digest)
        checks["source_hashes_valid"] = True
    except (ValueError, KeyError, TypeError):
        checks["source_hashes_valid"] = False
    doc_ids = manifest_audit.get("processed_document_sha256", {}) if manifest_audit else {}
    checks["document_ids_tracked"] = bool(doc_ids)
    drops = (manifest_audit or {}).get("exact_duplicate_drops", {})
    checks["dedup_recorded"] = manifest_audit is not None and isinstance(drops, dict)
    splits = {str(source.get("split", "")) for source in sources}
    checks["cluster_split_present"] = "training" in splits
    scan = str(data_manifest.get("contamination_scan_sha256") or "")
    try:
        _assert_sha256("contamination scan", scan)
        checks["contamination_posture_bound"] = True
    except ValueError:
        checks["contamination_posture_bound"] = False
    checks["tokenizer_compatible"] = (
        str(data_manifest.get("tokenizer_sha256") or "") == expected_tokenizer_sha256
        and str((tokenizer_receipt.get("artifact") or {}).get("sha256") or "")
        == expected_tokenizer_sha256
    )
    tokens_by_family = data_manifest.get("tokens_by_family") or {}
    slice_tokens: dict[str, int] = {}
    unmapped = []
    for family, count in tokens_by_family.items():
        target_slice = family_to_slice.get(str(family))
        if target_slice is None:
            unmapped.append(str(family))
        else:
            slice_tokens[target_slice] = slice_tokens.get(target_slice, 0) + int(count)
    checks["families_mapped_to_slices"] = not unmapped
    if unmapped:
        blockers.append(f"unmapped families cannot enter the mixture: {sorted(unmapped)}")
    total_required = sum(mixture_targets.values())
    total_available = sum(slice_tokens.values())
    checks["token_availability"] = total_available >= total_required
    if not checks["token_availability"]:
        blockers.append(
            f"only {total_available} unique qualified tokens against {total_required} budgeted"
        )
    mixture_ok = True
    for target_slice, target in mixture_targets.items():
        available = slice_tokens.get(target_slice, 0)
        lo = target * (1 - MIXTURE_TOLERANCE_FRACTION)
        hi = target * (1 + MIXTURE_TOLERANCE_FRACTION)
        if not lo <= available <= hi:
            mixture_ok = False
            blockers.append(
                f"slice {target_slice}: {available} tokens outside "
                f"[{int(lo)}, {int(hi)}] around {target}"
            )
    checks["mixture_sufficient"] = mixture_ok
    qual_ok = True
    for family in required_cognition_families:
        verdict = generator_qualifications.get(family)
        if verdict != "GENERATOR_QUALIFIED":
            qual_ok = False
            blockers.append(f"cognition family not qualified: {family} ({verdict})")
    checks["cognition_qualified"] = qual_ok
    for name, passed in checks.items():
        if not passed and name not in {
            "families_mapped_to_slices", "token_availability", "mixture_sufficient",
            "cognition_qualified",
        }:
            blockers.append(f"gate failed: {name}")
    status = "DATASET_QUALIFIED" if all(checks.values()) else "BLOCKED_BY_DATASET"
    return {
        "schema": QUALIFICATION_SCHEMA,
        "status": status,
        "checks": checks,
        "blockers": sorted(set(blockers)),
        "slice_tokens": slice_tokens,
        "total_available": total_available,
        "total_required": total_required,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))

    def _load(path_value: str) -> Any:
        return json.loads(Path(path_value).read_text(encoding="utf-8"))

    manifests = request.get("data_manifests") or []
    if not manifests:
        raise ValueError("request needs at least one data manifest path")
    first = _load(manifests[0])
    audit = None
    audit_path = request.get("manifest_audit")
    if audit_path:
        audit = _load(audit_path)
    receipt = qualify_dataset(
        data_manifest=first,
        manifest_audit=audit,
        tokenizer_receipt=_load(request["tokenizer_receipt"]),
        expected_tokenizer_sha256=request["expected_tokenizer_sha256"],
        family_to_slice=request.get("family_to_slice") or {},
        mixture_targets=request.get("mixture_targets") or dict(CENTER_5B_MIXTURE),
        required_cognition_families=request.get("required_cognition_families") or [],
        generator_qualifications=request.get("generator_qualifications") or {},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if receipt["status"] == "DATASET_QUALIFIED" else 1


__all__ = [
    "CENTER_5B_MIXTURE",
    "MIXTURE_TOLERANCE_FRACTION",
    "QUALIFICATION_SCHEMA",
    "SOURCE_SCHEMA",
    "SourceArtifact",
    "qualify_dataset",
]
