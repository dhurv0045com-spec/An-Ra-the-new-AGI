"""Audited, portable V4 supervised-fine-tuning dataset preparation.

The foundation corpus is deliberately not reused here.  This module accepts
licensed conversational JSONL records, canonicalizes them, rejects unsafe or
ambiguous structure, makes group-disjoint splits, and emits immutable manifests
that ``training.posttraining_contract`` can bind to a V4 checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from training.posttraining_contract import REQUIRED_SFT_CATEGORIES

SFT_DATASET_SCHEMA = "anra-sft-dataset/v1"
SFT_SOURCE_RECEIPTS_SCHEMA = "anra-sft-source-receipts/v1"
_ROLES = frozenset({"system", "user", "assistant"})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() == content:
            return
        raise FileExistsError(f"refusing to replace immutable SFT artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalise_text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    # Keep code indentation, Markdown tables, and multi-line tool contracts
    # intact. Only normalise platform line endings and discard outer padding.
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if any(ord(character) < 32 and character not in {"\n", "\t"} for character in normalized):
        raise ValueError(f"{field} contains an unsafe control character")
    if not normalized:
        raise ValueError(f"{field} is empty")
    return normalized


def _normalise_messages(raw: object) -> list[dict[str, str]]:
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("messages must contain a user context and final assistant answer")
    messages: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"messages[{index}] must be an object")
        role = str(item.get("role", "")).strip().lower()
        if role not in _ROLES:
            raise ValueError(f"messages[{index}].role is unsupported: {role!r}")
        messages.append(
            {
                "role": role,
                "content": _normalise_text(
                    item.get("content"),
                    field=f"messages[{index}].content",
                ),
            }
        )
    if messages[-1]["role"] != "assistant":
        raise ValueError("the final SFT message must be an assistant answer")
    if not any(message["role"] == "user" for message in messages[:-1]):
        raise ValueError("SFT conversation has no user message before its answer")
    return messages


def _record_from_raw(
    raw: Mapping[str, object],
    *,
    source_file: Path,
    defaults: Mapping[str, str] | None = None,
) -> dict[str, object]:
    metadata = dict(defaults or {})
    category = str(raw.get("category", metadata.get("category", ""))).strip()
    if category not in REQUIRED_SFT_CATEGORIES:
        raise ValueError(f"unsupported or missing SFT category: {category!r}")
    source_id = _normalise_text(
        raw.get("source_id", metadata.get("source_id", source_file.name)), field="source_id"
    )
    split_group = _normalise_text(raw.get("split_group", source_id), field="split_group")
    license_id = _normalise_text(raw.get("license", metadata.get("license", "")), field="license")
    if license_id.lower() in {"unknown", "unlicensed", "none", "n/a"}:
        raise ValueError(f"source {source_id!r} has no auditable license")
    if "messages" in raw:
        messages = _normalise_messages(raw["messages"])
    else:
        messages = [
            {"role": "user", "content": _normalise_text(raw.get("prompt"), field="prompt")},
            {"role": "assistant", "content": _normalise_text(raw.get("answer"), field="answer")},
        ]
    identity = hashlib.sha256(_canonical_json(messages)).hexdigest()
    return {
        "messages": messages,
        "category": category,
        "source_id": source_id,
        "split_group": split_group,
        "license": license_id,
        "conversation_sha256": identity,
    }


def _read_jsonl(path: Path) -> Iterable[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL in {path}:{line_number}") from error
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL record {path}:{line_number} must be an object")
            yield value


def _load_source_receipts(path: str | Path) -> tuple[Path, dict[Path, dict[str, str]]]:
    receipt_path = Path(path).resolve()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != SFT_SOURCE_RECEIPTS_SCHEMA:
        raise ValueError("unsupported SFT source receipt schema")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("SFT source receipt has no verified sources")
    by_path: dict[Path, dict[str, str]] = {}
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, Mapping):
            raise ValueError(f"SFT source receipt {index} is invalid")
        source_path = Path(str(raw.get("path", ""))).resolve()
        source_id = _normalise_text(raw.get("source_id"), field="receipt source_id")
        license_id = _normalise_text(raw.get("license"), field="receipt license")
        expected_hash = str(raw.get("sha256", "")).strip().lower()
        if len(expected_hash) != 64 or any(
            char not in "0123456789abcdef" for char in expected_hash
        ):
            raise ValueError(f"SFT source receipt {index} has invalid SHA-256")
        if not source_path.is_file() or sha256_file(source_path) != expected_hash:
            raise ValueError(f"SFT source receipt {index} does not match its verified source file")
        if source_path in by_path:
            raise ValueError(f"SFT source receipt repeats path: {source_path}")
        category = str(raw.get("category", "")).strip()
        if category and category not in REQUIRED_SFT_CATEGORIES:
            raise ValueError(f"SFT source receipt {index} has unsupported category {category!r}")
        by_path[source_path] = {
            "source_id": source_id,
            "license": license_id,
            "sha256": expected_hash,
            "category": category,
        }
    return receipt_path, by_path


def _split_for(split_group: str) -> str:
    group_hash = hashlib.sha256(split_group.encode("utf-8")).hexdigest()
    bucket = int(group_hash[:8], 16) % 100
    if bucket < 85:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


@dataclass(frozen=True)
class SFTDatasetBuildResult:
    output_dir: Path
    manifests: dict[str, Path]
    accepted_examples: dict[str, int]
    rejected_examples: int


def build_sft_dataset_v4(
    inputs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    quality_gate_passed: bool,
    licenses_audited: bool,
    source_receipts_path: str | Path | None = None,
    allow_unregistered_inputs: bool = False,
) -> SFTDatasetBuildResult:
    """Build immutable train/validation/test artifacts from licensed JSONL input."""

    if not inputs:
        raise ValueError("at least one SFT JSONL input is required")
    if not quality_gate_passed or not licenses_audited:
        raise PermissionError("SFT build requires explicit quality and license audit approval")
    if source_receipts_path is None and not allow_unregistered_inputs:
        raise PermissionError(
            "canonical SFT build requires verified source receipts; use the source downloader "
            "or explicitly allow an unregistered local pilot"
        )
    receipt_path: Path | None = None
    verified_sources: dict[Path, dict[str, str]] = {}
    if source_receipts_path is not None:
        receipt_path, verified_sources = _load_source_receipts(source_receipts_path)
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    rejected = 0
    seen: set[str] = set()
    for raw_input in inputs:
        source_file = Path(raw_input).resolve()
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        registered_source = verified_sources.get(source_file)
        if verified_sources and registered_source is None:
            raise PermissionError(
                f"SFT input is absent from the verified source receipt: {source_file}"
            )
        for raw in _read_jsonl(source_file):
            try:
                record = _record_from_raw(
                    raw,
                    source_file=source_file,
                    defaults=registered_source,
                )
            except ValueError:
                rejected += 1
                continue
            if registered_source is not None and (
                record["source_id"] != registered_source["source_id"]
                or record["license"] != registered_source["license"]
                or (
                    bool(registered_source["category"])
                    and record["category"] != registered_source["category"]
                )
            ):
                raise ValueError(
                    "SFT record provenance disagrees with its hash-verified source receipt"
                )
            key = str(record["conversation_sha256"])
            if key in seen:
                rejected += 1
                continue
            seen.add(key)
            records.append(record)
    if not records:
        raise ValueError("SFT source inputs contain no accepted records")
    splits: dict[str, list[dict[str, object]]] = {"train": [], "validation": [], "test": []}
    for record in records:
        splits[_split_for(str(record["split_group"]))].append(record)
    all_groups = {str(record["split_group"]) for record in records}
    if len(all_groups) < 3:
        raise ValueError(
            "SFT source inputs need at least three split groups to create train, "
            "validation, and test without source leakage"
        )
    # A small pilot input can land every record in one split. Move an entire
    # declared group, never an individual conversation, to keep source-group
    # isolation true even for tiny owner-curated pilots.
    required_splits = ("validation", "test")
    for position, name in enumerate(required_splits):
        if not splits[name]:
            groups = sorted({str(row["split_group"]) for row in splits["train"]})
            remaining_empty = sum(
                not splits[other] for other in required_splits[position:]
            )
            if len(groups) <= remaining_empty:
                raise ValueError(
                    "SFT source inputs need at least three split groups to create "
                    "train, validation, and test without source leakage"
                )
            chosen_group = groups[0]
            moved = [row for row in splits["train"] if row["split_group"] == chosen_group]
            splits["train"] = [
                row for row in splits["train"] if row["split_group"] != chosen_group
            ]
            splits[name].extend(moved)

    # Validation must exercise every capability category. Hash-bucketing can
    # otherwise leave a category (notably code) entirely in training, making
    # aggregate validation loss overstate coverage. Move complete source
    # groups so conversation and source isolation remain intact.
    validation_categories = {str(row["category"]) for row in splits["validation"]}
    for category in REQUIRED_SFT_CATEGORIES:
        if category in validation_categories:
            continue
        candidates = sorted(
            {
                str(row["split_group"])
                for row in splits["train"]
                if str(row["category"]) == category
            }
        )
        moved_group: str | None = None
        for group in candidates:
            remaining = [
                row
                for row in splits["train"]
                if str(row["category"]) == category
                and str(row["split_group"]) != group
            ]
            if remaining:
                moved_group = group
                break
        if moved_group is None:
            raise ValueError(
                f"SFT validation split is missing category {category!r}; "
                "provide at least two source groups for every required category"
            )
        moved = [
            row
            for row in splits["train"]
            if str(row["split_group"]) == moved_group
        ]
        splits["train"] = [
            row
            for row in splits["train"]
            if str(row["split_group"]) != moved_group
        ]
        splits["validation"].extend(moved)
        validation_categories.add(category)

    train_categories = Counter(str(row["category"]) for row in splits["train"])
    missing = [name for name in REQUIRED_SFT_CATEGORIES if train_categories[name] <= 0]
    if missing:
        raise ValueError(f"accepted train split is missing required SFT categories: {missing}")

    manifests: dict[str, Path] = {}
    accepted: dict[str, int] = {}
    source_licenses = sorted({(str(row["source_id"]), str(row["license"])) for row in records})
    for split, rows in splits.items():
        rows.sort(key=lambda row: str(row["conversation_sha256"]))
        artifact = target / f"sft-v4-{split}.jsonl"
        content = b"".join(_canonical_json(row) + b"\n" for row in rows)
        _write_immutable(artifact, content)
        category_counts = dict(sorted(Counter(str(row["category"]) for row in rows).items()))
        manifest_payload: dict[str, object] = {
            "schema": SFT_DATASET_SCHEMA,
            "split": split,
            "quality_gate_passed": True,
            "licenses_audited": True,
            "accepted_examples": len(rows),
            "rejected_examples": rejected,
            "category_counts": category_counts,
            "source_licenses": [
                {"source_id": source_id, "license": license_id}
                for source_id, license_id in source_licenses
            ],
            "source_receipt_sha256": sha256_file(receipt_path) if receipt_path else None,
            "unregistered_local_pilot": bool(allow_unregistered_inputs and receipt_path is None),
            "split_group_count": len({str(row["split_group"]) for row in rows}),
            "split_group_sha256": hashlib.sha256(
                _canonical_json(sorted({str(row["split_group"]) for row in rows}))
            ).hexdigest(),
            "split_identity_sha256": hashlib.sha256(
                _canonical_json([row["conversation_sha256"] for row in rows])
            ).hexdigest(),
            "artifacts": [
                {
                    "path": artifact.name,
                    "sha256": sha256_file(artifact),
                    "size_bytes": artifact.stat().st_size,
                }
            ],
        }
        manifest = target / f"sft-v4-{split}.manifest.json"
        encoded = (
            json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True).encode(
                "utf-8"
            )
            + b"\n"
        )
        _write_immutable(manifest, encoded)
        manifests[split] = manifest
        accepted[split] = len(rows)
    return SFTDatasetBuildResult(target, manifests, accepted, rejected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an audited V4 SFT dataset")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="licensed source JSONL; repeat",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--approve-quality", action="store_true")
    parser.add_argument("--approve-licenses", action="store_true")
    parser.add_argument("--source-receipts", default=None)
    parser.add_argument("--allow-unregistered-inputs", action="store_true")
    args = parser.parse_args()
    result = build_sft_dataset_v4(
        args.input,
        args.output_dir,
        quality_gate_passed=args.approve_quality,
        licenses_audited=args.approve_licenses,
        source_receipts_path=args.source_receipts,
        allow_unregistered_inputs=args.allow_unregistered_inputs,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "manifests": {name: str(path) for name, path in result.manifests.items()},
                "accepted_examples": result.accepted_examples,
                "rejected_examples": result.rejected_examples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
