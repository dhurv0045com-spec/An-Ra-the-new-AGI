"""Local dataset manifest loader with content hashes and availability state.

A manifest declares local JSONL files, their splits, kinds, trainability and
license/provenance. The loader verifies local availability, hashes file bytes,
derives semantic identities (so renamed duplicates cannot cross splits) and
reports an exact split inventory. Reading metadata never exposes answers to
inference consumers.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
from dataclasses import dataclass
from typing import Any, Mapping

from bramastra_lab.research.contracts.core import content_identity

DATA_NOT_READY = "DATA_NOT_READY"
DATA_AVAILABLE = "DATA_AVAILABLE"

MANIFEST_SCHEMA = "bramastra-dataset-manifest/v1"
SUPPORTED_KINDS = frozenset({"language", "trajectory"})
# 'controller' and 'sealed' pools are declared here and stored separately from
# gradient-training data downstream; they can never silently mix.
SUPPORTED_SPLITS = frozenset({
    "training", "controller", "development", "sealed",
    "training_gain_probe", "strategy_validation", "confirmation",
})


class DatasetError(ValueError):
    """A manifest or its local files violate the data contract."""

    def __init__(self, message: str, *, status: str = DATA_NOT_READY) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Example:
    """One semantic example loaded from a local file."""

    example_id: str
    kind: str
    split: str
    content: Mapping[str, Any]
    semantic_identity: str
    content_identity: str
    group_id: str | None
    source: str
    line_number: int
    trainable: bool
    family: str = "default"
    mechanism_cluster: str | None = None

    def public_input(self) -> Any:
        """The public, learnable input; answers are not part of it."""
        if self.kind == "language":
            return self.content.get("text")
        return self.content.get("prompt_events")

    def answer(self) -> str | None:
        """The teacher answer; training and scoring only, never inference input."""
        return self.content.get("answer")

    def inference_metadata(self) -> dict[str, Any]:
        """Metadata safe to attach to inference/evaluation outputs.

        Deliberately excludes the answer, reward and any outcome field, so
        reading batch or report metadata cannot reveal labels.
        """
        return {
            "example_id": self.example_id,
            "kind": self.kind,
            "split": self.split,
            "semantic_identity": self.semantic_identity,
            "group_id": self.group_id,
            "family": self.family,
            "mechanism_cluster": self.mechanism_cluster,
            "trainable": self.trainable,
            "source": self.source,
        }


@dataclass(frozen=True)
class DatasetHandle:
    """Loaded, verified view over one manifest."""

    name: str
    license: str
    provenance: str
    identity: str
    availability: str
    examples: tuple[Example, ...]
    split_inventory: Mapping[str, Mapping[str, int]]
    duplicate_count: int
    group_count: int

    def examples_for_split(self, split: str) -> tuple[Example, ...]:
        if split not in SUPPORTED_SPLITS:
            raise DatasetError(f"unknown split {split!r}", status=DATA_NOT_READY)
        return tuple(example for example in self.examples if example.split == split)

    def groups_for_split(self, split: str) -> tuple[tuple[Example, ...], ...]:
        """Examples of one split grouped by pair group; ungrouped are singletons."""
        buckets: dict[str, list[Example]] = {}
        ordered: list[Example] = []
        for example in self.examples_for_split(split):
            if example.group_id is None:
                ordered.append(example)
            else:
                buckets.setdefault(example.group_id, []).append(example)
        groups = [tuple(bucket) for _, bucket in sorted(buckets.items())]
        return tuple(groups) + tuple((example,) for example in ordered)


def _validate_relative_path(path: str) -> None:
    if not isinstance(path, str) or not path:
        raise DatasetError("entry.path must be a nonempty relative path")
    if urllib.parse.urlparse(path).scheme:
        raise DatasetError(
            f"entry.path {path!r} looks like a URL; corpus downloads are not authorized")
    if os.path.isabs(path) or ".." in path.replace("\\", "/").split("/"):
        raise DatasetError(
            f"entry.path {path!r} must stay inside the manifest directory")
    if path.startswith("~"):
        raise DatasetError("entry.path must not reference home directories")


def _load_jsonl(
    path: str,
) -> tuple[list[tuple[int, Mapping[str, Any]]], str, int, bool]:
    records: list[tuple[int, Mapping[str, Any]]] = []
    digest = hashlib.sha256()
    byte_count = 0
    has_non_whitespace = False
    try:
        with open(path, "rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                byte_count += len(raw_line)
                has_non_whitespace |= bool(raw_line.strip())
                try:
                    stripped = raw_line.decode("utf-8").strip()
                except UnicodeDecodeError as exc:
                    raise DatasetError(
                        f"{path}:{line_number} is not valid UTF-8: {exc}") from exc
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"{path}:{line_number} is not valid JSON: {exc}")
                if not isinstance(record, Mapping):
                    raise DatasetError(f"{path}:{line_number} must be a JSON object")
                records.append((line_number, record))
    except OSError as exc:
        raise DatasetError(f"cannot read local data file {path}: {exc}")
    return records, digest.hexdigest(), byte_count, has_non_whitespace


def _semantic_content(kind: str, record: Mapping[str, Any]) -> Mapping[str, Any]:
    """The bytes that define the semantic CONTENT digest: input plus teacher answer.

    Surface names, example IDs, provenance and file positions are excluded so
    renamed duplicates are still recognized as the same content. This digest
    is NOT a validated hidden-mechanism equivalence class: two renderings of
    one mechanism that differ in input or answer bytes hash differently, and
    qualified mechanism-cluster identity must be declared separately via the
    manifest's ``mechanism_cluster`` field (required for cluster-transfer
    claims; generalization claims without it are unsupported).
    """
    if kind == "language":
        return {"input": record.get("text")}
    return {"input": record.get("prompt_events"), "answer": record.get("answer")}


def load_dataset(manifest_path: str) -> DatasetHandle:
    """Load and verify one local manifest. Missing files raise DATA_NOT_READY."""
    if not os.path.exists(manifest_path):
        raise DatasetError(f"dataset manifest not found at {manifest_path}")
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"dataset manifest unreadable: {exc}")
    if not isinstance(raw, Mapping):
        raise DatasetError("dataset manifest must be an object")
    if raw.get("schema") != MANIFEST_SCHEMA:
        raise DatasetError(f"dataset manifest schema must be {MANIFEST_SCHEMA!r}")
    unknown = set(raw) - {"schema", "name", "license", "provenance", "entries"}
    if unknown:
        raise DatasetError(f"dataset manifest has unknown fields: {sorted(unknown)}")
    for field in ("name", "license", "provenance"):
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            raise DatasetError(f"dataset manifest {field} must be a nonempty string")
    entries = raw.get("entries")
    if not isinstance(entries, list) or not entries:
        raise DatasetError("dataset manifest declares no entries", status=DATA_NOT_READY)

    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    examples: list[Example] = []
    file_records: list[dict[str, Any]] = []
    seen_semantic: dict[str, str] = {}
    duplicates = 0

    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise DatasetError(f"entry {entry_index} must be an object")
        unknown_entry = set(entry) - {"path", "split", "kind", "trainable", "license",
                                      "provenance"}
        if unknown_entry:
            raise DatasetError(f"entry {entry_index} has unknown fields: {sorted(unknown_entry)}")
        path = entry.get("path")
        _validate_relative_path(path)
        split = entry.get("split")
        if split not in SUPPORTED_SPLITS:
            raise DatasetError(
                f"entry {entry_index} split {split!r} must be one of {sorted(SUPPORTED_SPLITS)}")
        kind = entry.get("kind")
        if kind not in SUPPORTED_KINDS:
            raise DatasetError(f"entry {entry_index} kind must be one of {sorted(SUPPORTED_KINDS)}")
        trainable = entry.get("trainable", True)
        if not isinstance(trainable, bool):
            raise DatasetError(f"entry {entry_index} trainable must be a boolean")
        entry_license = entry.get("license", raw["license"])
        entry_provenance = entry.get("provenance", raw["provenance"])
        for field, value in (("license", entry_license), ("provenance", entry_provenance)):
            if not isinstance(value, str) or not value.strip():
                raise DatasetError(f"entry {entry_index} {field} must be a nonempty string")

        absolute = os.path.join(manifest_dir, path)
        if not os.path.exists(absolute):
            raise DatasetError(
                f"local data file {absolute} is missing; corpus supply is incomplete")
        records, file_sha, file_size, has_non_whitespace = _load_jsonl(absolute)
        if not has_non_whitespace:
            raise DatasetError(f"local data file {absolute} is empty")
        file_records.append({"path": path, "split": split, "kind": kind,
                             "sha256": file_sha, "bytes": file_size,
                             "trainable": trainable, "license": entry_license,
                             "provenance": entry_provenance})

        for line_number, record in records:
            unknown_example = set(record) - {"example_id", "text", "prompt_events", "answer",
                                             "group", "family", "mechanism_cluster",
                                             "trainable"}
            if unknown_example:
                raise DatasetError(
                    f"{path}:{line_number} has unknown fields: {sorted(unknown_example)}")
            example_id = record.get("example_id")
            if not isinstance(example_id, str) or not example_id:
                raise DatasetError(f"{path}:{line_number} example_id must be a nonempty string")
            if kind == "language":
                if not isinstance(record.get("text"), str) or not record["text"]:
                    raise DatasetError(f"{path}:{line_number} language example needs text")
                if record.get("prompt_events") is not None or record.get("answer") is not None:
                    raise DatasetError(f"{path}:{line_number} language example must not declare "
                                       "prompt_events or answer")
            else:
                events = record.get("prompt_events")
                if not isinstance(events, list) or not events:
                    raise DatasetError(
                        f"{path}:{line_number} trajectory example needs prompt_events")
                for event in events:
                    if not isinstance(event, list) or len(event) != 2 \
                            or not isinstance(event[0], str):
                        raise DatasetError(
                            f"{path}:{line_number} prompt_events must be [role, content] pairs")
                if not isinstance(record.get("answer"), str) or not record["answer"]:
                    raise DatasetError(f"{path}:{line_number} trajectory example needs an answer")
            group_id = record.get("group")
            if group_id is not None and (not isinstance(group_id, str) or not group_id):
                raise DatasetError(f"{path}:{line_number} group must be a nonempty string or null")
            family = record.get("family", "default")
            if not isinstance(family, str) or not family:
                raise DatasetError(f"{path}:{line_number} family must be a nonempty string")
            mechanism_cluster = record.get("mechanism_cluster")
            if mechanism_cluster is not None                     and (not isinstance(mechanism_cluster, str) or not mechanism_cluster):
                raise DatasetError(
                    f"{path}:{line_number} mechanism_cluster must be a nonempty string or null")
            # Per-example trainability narrows the file-level rule but can
            # never broaden it: the entry's own eligibility stays immutable
            # across the record loop (B2.2 chief F2).
            example_trainable = record.get("trainable", trainable)
            if not isinstance(example_trainable, bool):
                raise DatasetError(f"{path}:{line_number} trainable must be a boolean")
            effective_trainable = trainable and example_trainable
            semantic = content_identity(_semantic_content(kind, record))
            prior = seen_semantic.get(semantic)
            if prior is not None and prior != split:
                raise DatasetError(
                    f"duplicate semantics across splits: {example_id!r} repeats content already "
                    f"assigned to split {prior!r}; renamed surfaces cannot cross splits")
            if prior is not None:
                duplicates += 1
            seen_semantic[semantic] = split
            examples.append(Example(
                example_id=example_id, kind=kind, split=split, content=dict(record),
                semantic_identity=semantic, content_identity=content_identity(record),
                group_id=group_id,
                source=f"{path}:{line_number}", line_number=line_number,
                trainable=effective_trainable,
                family=family, mechanism_cluster=mechanism_cluster))

    if not examples:
        raise DatasetError("manifest references contain no examples", status=DATA_NOT_READY)

    grouped = [example for example in examples if example.group_id is not None]
    group_buckets: dict[str, list[Example]] = {}
    for example in grouped:
        group_buckets.setdefault(example.group_id, []).append(example)
    for group_id, members in group_buckets.items():
        if len({member.split for member in members}) != 1:
            raise DatasetError(
                f"pair group {group_id!r} spans multiple splits; counterfactual pairs must "
                "stay in one split")

    inventory: dict[str, dict[str, int]] = {}
    for example in examples:
        bucket = inventory.setdefault(example.split, {"examples": 0, "trainable": 0})
        bucket["examples"] += 1
        if example.trainable:
            bucket["trainable"] += 1
    identity_payload = {
        "schema": MANIFEST_SCHEMA, "name": raw["name"], "license": raw["license"],
        "provenance": raw["provenance"], "files": file_records,
        "duplicate_semantics": duplicates,
    }
    return DatasetHandle(
        name=raw["name"], license=raw["license"], provenance=raw["provenance"],
        identity=content_identity(identity_payload),
        availability=DATA_AVAILABLE, examples=tuple(examples),
        split_inventory={split: dict(counts) for split, counts in sorted(inventory.items())},
        duplicate_count=duplicates, group_count=len(group_buckets),
    )
