"""Fail-closed source, tokenizer, dataset, and pack manifest contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .lineage import ArtifactIdentity, _assert_sha256


ALLOWED_LICENSE_CATEGORIES = {
    "public-domain",
    "permissive",
    "licensed",
    "first-party-authorized",
}
ALLOWED_SPLITS = {"training", "development", "sealed", "fresh"}


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    authorization_category: str
    acquired_date: str
    raw_sha256: str
    split: str
    domain: str

    def assert_valid(self) -> None:
        if not self.source_id or not self.domain or not self.acquired_date:
            raise ValueError("source identity, date, and domain are required")
        if self.authorization_category not in ALLOWED_LICENSE_CATEGORIES:
            raise ValueError("source authorization category is not allowed")
        if self.split not in ALLOWED_SPLITS:
            raise ValueError("unknown data split")
        _assert_sha256("raw source", self.raw_sha256)


@dataclass(frozen=True, slots=True)
class TokenizerReceipt:
    schema: str
    artifact: ArtifactIdentity
    vocabulary_size: int
    special_token_ids: Mapping[str, int]
    trainer_config_sha256: str
    corpus_manifest_sha256: str
    identity_roundtrip_passed: bool
    unknown_rate: float

    def assert_valid(self) -> None:
        self.artifact.assert_valid()
        _assert_sha256("trainer config", self.trainer_config_sha256)
        _assert_sha256("tokenizer corpus", self.corpus_manifest_sha256)
        if self.vocabulary_size <= 256:
            raise ValueError("vocabulary is too small for the subword contract")
        if len(set(self.special_token_ids.values())) != len(self.special_token_ids):
            raise ValueError("special token ids must be distinct")
        if any(not 0 <= value < self.vocabulary_size for value in self.special_token_ids.values()):
            raise ValueError("special token id outside vocabulary")
        if not self.identity_roundtrip_passed or self.unknown_rate != 0.0:
            raise ValueError("tokenizer must round-trip with zero unknowns")


@dataclass(frozen=True, slots=True)
class DataManifest:
    schema: str
    manifest_id: str
    tokenizer_sha256: str
    filter_version: str
    dedup_version: str
    contamination_scan_sha256: str
    sources: tuple[SourceRecord, ...]
    tokens_by_family: Mapping[str, int]
    total_tokens: int

    def assert_valid(self) -> None:
        _assert_sha256("tokenizer", self.tokenizer_sha256)
        _assert_sha256("contamination scan", self.contamination_scan_sha256)
        if not self.manifest_id or not self.filter_version or not self.dedup_version:
            raise ValueError("manifest/filter/dedup identities are required")
        if not self.sources:
            raise ValueError("data manifest requires sources")
        for source in self.sources:
            source.assert_valid()
        hashes = [source.raw_sha256 for source in self.sources]
        if len(hashes) != len(set(hashes)):
            raise ValueError("duplicate raw source hashes")
        if any(value < 0 for value in self.tokens_by_family.values()):
            raise ValueError("token totals cannot be negative")
        if sum(self.tokens_by_family.values()) != self.total_tokens:
            raise ValueError("tokens_by_family must equal total_tokens")


@dataclass(frozen=True, slots=True)
class PackShard:
    shard_id: str
    sha256: str
    byte_size: int
    sequence_count: int
    token_count: int

    def assert_valid(self) -> None:
        _assert_sha256("pack shard", self.sha256)
        if not self.shard_id or min(self.byte_size, self.sequence_count, self.token_count) <= 0:
            raise ValueError("pack shard counts must be positive")


@dataclass(frozen=True, slots=True)
class PackManifest:
    schema: str
    tokenizer_sha256: str
    data_manifest_sha256: str
    packer_version: str
    cursor_schema: str
    shards: tuple[PackShard, ...]
    total_tokens: int

    def assert_valid(self) -> None:
        _assert_sha256("tokenizer", self.tokenizer_sha256)
        _assert_sha256("data manifest", self.data_manifest_sha256)
        if not self.packer_version or not self.cursor_schema or not self.shards:
            raise ValueError("packer, cursor, and shards are required")
        for shard in self.shards:
            shard.assert_valid()
        hashes = [shard.sha256 for shard in self.shards]
        if len(hashes) != len(set(hashes)):
            raise ValueError("duplicate pack shard hash")
        if sum(shard.token_count for shard in self.shards) != self.total_tokens:
            raise ValueError("shard tokens must equal pack total")


def assert_source_disjoint(*manifests: DataManifest) -> None:
    """Reject raw-source reuse across independently claimed manifests/splits."""

    seen: dict[str, str] = {}
    for manifest in manifests:
        manifest.assert_valid()
        for source in manifest.sources:
            previous = seen.get(source.raw_sha256)
            if previous is not None:
                raise ValueError(
                    f"source hash {source.raw_sha256} appears in both {previous} and {manifest.manifest_id}"
                )
            seen[source.raw_sha256] = manifest.manifest_id
