"""Immutable data-source registry: no anonymous folders (M3).

Every external dataset lives here with identity, version, class, artifact
hashes, format, counts, provenance, license metadata, acquisition method,
quality tier, language/domain data, processing policy, and lifecycle
status. Status transitions only forward toward QUALIFIED (or sideways to
REJECTED/UNAVAILABLE); history is preserved, never rewritten.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


REGISTRY_SCHEMA = "anra-v5-source-registry/v1"

SOURCE_CLASSES = frozenset({"natural", "code", "math", "formal", "dialogue", "cognition"})
STATUSES = (
    "DISCOVERED",
    "ACQUIRED",
    "IDENTITY_VERIFIED",
    "PROCESSING",
    "QUALIFIED",
    "REJECTED",
    "UNAVAILABLE",
)
_FORWARD = {
    "DISCOVERED": {"ACQUIRED", "REJECTED", "UNAVAILABLE"},
    "ACQUIRED": {"IDENTITY_VERIFIED", "REJECTED", "UNAVAILABLE"},
    "IDENTITY_VERIFIED": {"PROCESSING", "REJECTED"},
    "PROCESSING": {"QUALIFIED", "REJECTED"},
    "QUALIFIED": set(),
    "REJECTED": set(),
    "UNAVAILABLE": {"ACQUIRED"},
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _assert_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class DataSource:
    source_id: str
    version: str
    source_class: str
    artifact_sha256: str
    format: str
    document_count: int
    byte_count: int
    provenance: str
    license: str
    acquisition_method: str
    quality_tier: str
    language_domain: str
    processing_policy: str
    status: str
    status_history: tuple[str, ...]

    def assert_valid(self) -> None:
        if not self.source_id or not self.version:
            raise ValueError("source identity and version are required")
        if self.source_class not in SOURCE_CLASSES:
            raise ValueError(f"unknown source class: {self.source_class}")
        _assert_sha256("artifact", self.artifact_sha256)
        if not self.format or self.document_count < 0 or self.byte_count < 0:
            raise ValueError("format and nonnegative counts are required")
        for name in ("provenance", "license", "acquisition_method", "quality_tier",
                     "language_domain", "processing_policy"):
            if not getattr(self, name):
                raise ValueError(f"source metadata is required: {name}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown status: {self.status}")
        if not self.status_history or self.status_history[-1] != self.status:
            raise ValueError("status history must end at the current status")

    def transition(self, status: str) -> "DataSource":
        """Move lifecycle status forward; history is preserved."""

        import dataclasses

        self.assert_valid()
        if status not in STATUSES:
            raise ValueError(f"unknown status: {status}")
        if status not in _FORWARD[self.status]:
            raise ValueError(f"illegal status transition: {self.status} -> {status}")
        updated = dataclasses.replace(self, status=status, status_history=(*self.status_history, status))
        updated.assert_valid()
        return updated


@dataclass(frozen=True, slots=True)
class DataSourceRegistry:
    schema: str
    sources: tuple[DataSource, ...]

    def assert_valid(self) -> None:
        if self.schema != REGISTRY_SCHEMA:
            raise ValueError("unsupported source-registry schema")
        ids = [source.source_id for source in self.sources]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate source ids")
        for source in self.sources:
            source.assert_valid()

    def sha256(self) -> str:
        self.assert_valid()
        return hashlib.sha256(
            _canonical_json(
                {
                    "schema": self.schema,
                    "sources": [
                        {
                            "source_id": source.source_id,
                            "version": source.version,
                            "source_class": source.source_class,
                            "artifact_sha256": source.artifact_sha256,
                            "format": source.format,
                            "document_count": source.document_count,
                            "byte_count": source.byte_count,
                            "provenance": source.provenance,
                            "license": source.license,
                            "acquisition_method": source.acquisition_method,
                            "quality_tier": source.quality_tier,
                            "language_domain": source.language_domain,
                            "processing_policy": source.processing_policy,
                            "status": source.status,
                            "status_history": list(source.status_history),
                        }
                        for source in self.sources
                    ],
                }
            )
        ).hexdigest()

    def with_source(self, source: DataSource) -> "DataSourceRegistry":
        import dataclasses

        source.assert_valid()
        if any(existing.source_id == source.source_id for existing in self.sources):
            raise ValueError(f"source already registered: {source.source_id}")
        updated = dataclasses.replace(self, sources=(*self.sources, source))
        updated.assert_valid()
        return updated

    def with_transition(self, source_id: str, status: str) -> "DataSourceRegistry":
        import dataclasses

        updated_sources = tuple(
            source.transition(status) if source.source_id == source_id else source
            for source in self.sources
        )
        if all(source.source_id != source_id for source in self.sources):
            raise ValueError(f"unknown source: {source_id}")
        updated = dataclasses.replace(self, sources=updated_sources)
        updated.assert_valid()
        return updated

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DataSourceRegistry":
        if value.get("schema") != REGISTRY_SCHEMA or not isinstance(value.get("sources"), list):
            raise ValueError("source-registry fields do not match schema")
        sources = []
        for item in value["sources"]:
            sources.append(DataSource(
                source_id=str(item["source_id"]), version=str(item["version"]),
                source_class=str(item["source_class"]), artifact_sha256=str(item["artifact_sha256"]),
                format=str(item["format"]), document_count=int(item["document_count"]),
                byte_count=int(item["byte_count"]), provenance=str(item["provenance"]),
                license=str(item["license"]), acquisition_method=str(item["acquisition_method"]),
                quality_tier=str(item["quality_tier"]), language_domain=str(item["language_domain"]),
                processing_policy=str(item["processing_policy"]), status=str(item["status"]),
                status_history=tuple(str(entry) for entry in item["status_history"]),
            ))
        registry = cls(schema=REGISTRY_SCHEMA, sources=tuple(sources))
        registry.assert_valid()
        return registry


__all__ = ["REGISTRY_SCHEMA", "SOURCE_CLASSES", "STATUSES", "DataSource", "DataSourceRegistry"]
