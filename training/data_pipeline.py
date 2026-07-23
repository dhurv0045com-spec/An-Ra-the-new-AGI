"""Deterministic, license-aware preprocessing for the canonical V4 corpus."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from retrieval import CorpusDedupIndex

from training.data_ledger import DataEntropyLedger, DataQuality
from training.v2_config import (
    CANONICAL_V4_VOCAB_SIZE,
    TOKENIZER_SCHEMA_VERSION,
)

CANONICAL_TOKENIZER_VERSION = f"v4-{CANONICAL_V4_VOCAB_SIZE}"


def _require_v4_tokenizer(
    tokenizer_version: str,
    *,
    tokenizer_schema_version: int = TOKENIZER_SCHEMA_VERSION,
) -> None:
    """Reject legacy or ambiguous tokenizer identities at publication time."""

    if str(tokenizer_version).strip().lower() != CANONICAL_TOKENIZER_VERSION:
        raise ValueError(
            "New data publications require tokenizer_version="
            f"{CANONICAL_TOKENIZER_VERSION!r}; V3 and 16k lineages are retired"
        )
    if int(tokenizer_schema_version) != TOKENIZER_SCHEMA_VERSION:
        raise ValueError(
            "New data publications require tokenizer schema "
            f"{TOKENIZER_SCHEMA_VERSION}, got {tokenizer_schema_version}"
        )

DFC_TAGS = (
    "[GOAL]",
    "[CONSTRAINT]",
    "[HYPOTHESIS]",
    "[ACTION]",
    "[RESULT]",
    "[VERIFY]",
    "[UPDATE]",
)


@dataclass(frozen=True)
class SourceRecord:
    text: str
    source: str
    license: str
    bucket: str
    quality: DataQuality
    verifier_status: str = "not_applicable"
    source_revision: str = "unknown"
    civ_score: float = 1.0
    source_class: str = ""


def validate_dfc(text: str) -> bool:
    from verification import DEFAULT_VERIFIER_REGISTRY

    tags = DFC_TAGS
    if "<task" in text:
        tags = (
            "<task",
            "</task>",
            "<hyp>",
            "</hyp>",
            "<cons>",
            "</cons>",
            "<verify>",
            "</verify>",
        )
    result = DEFAULT_VERIFIER_REGISTRY.verify(
        "dfc_format",
        {"text": text, "tags": tags},
    )
    return float(result.score) == 1.0


class ShardedDataPipeline:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        tokenizer_version: str,
        shard_size: int = 10_000,
        ledger: DataEntropyLedger | None = None,
        style_filter: Callable[[str], bool] | None = None,
        civ_floor: float = 0.85,
        dedup_index: CorpusDedupIndex | None = None,
        near_duplicate_check: bool = True,
    ) -> None:
        _require_v4_tokenizer(tokenizer_version)
        self.output_dir = Path(output_dir)
        self.tokenizer_version = CANONICAL_TOKENIZER_VERSION
        self.shard_size = int(shard_size)
        self.ledger = ledger or DataEntropyLedger()
        self.style_filter = style_filter or (lambda text: bool(text.strip()))
        self.civ_floor = float(civ_floor)
        self.dedup_index = dedup_index or CorpusDedupIndex()
        self.near_duplicate_check = bool(near_duplicate_check)

    @staticmethod
    def _license_allowed(record: SourceRecord) -> bool:
        raw_parts = re.split(
            r"\s+(?:AND|OR)\s+|[,;]",
            record.license.strip(),
            flags=re.IGNORECASE,
        )
        licenses = {
            part.strip().lower().replace("_", "-")
            for part in raw_parts
            if part.strip()
        }
        allowed = {
            "odc-by",
            "odc-by-1.0",
            "commoncrawl-terms",
            "cc-by",
            "cc-by-4.0",
            "cc-by-sa",
            "cc0",
            "mit",
            "apache-2.0",
            "bsd",
            "bsd-2-clause",
            "bsd-3-clause",
            "isc",
            "mpl-2.0",
            "public-domain",
            "unlicense",
            "owner",
        }
        if "fineweb" in record.source.lower():
            return licenses in ({"odc-by"}, {"odc-by-1.0"})
        return bool(licenses) and licenses <= allowed

    def _reject_reason(
        self,
        record: SourceRecord,
        *,
        seen_hashes: set[str],
    ) -> tuple[str | None, float]:
        if not record.source.strip() or not record.text.strip():
            return "source_validation", 0.0
        if not self._license_allowed(record):
            return "license_or_provenance", 0.0
        content_hash = hashlib.sha256(record.text.strip().encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            return "duplicate", 0.0
        if self.near_duplicate_check:
            decision = self.dedup_index.check_and_add(
                record.text,
                record_id=content_hash,
                metadata={"source": record.source, "bucket": record.bucket},
            )
            if decision.duplicate:
                return "duplicate", 0.0
        seen_hashes.add(content_hash)
        keep, score = self.ledger.evaluate(record.quality)
        if not keep:
            return "del_below_0.65", score
        if not self.style_filter(record.text):
            return "style_filter", score
        if record.civ_score < self.civ_floor:
            return "civ_gate", score
        if record.bucket == "dfc" and not validate_dfc(record.text):
            return "invalid_dfc", score
        return None, score

    def preprocess(self, records: Iterable[SourceRecord]) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists():
            raise FileExistsError(
                "Published shard set is immutable; create a new revision "
                f"directory: {self.output_dir}"
            )
        accepted: list[tuple[SourceRecord, float]] = []
        rejected: list[dict[str, object]] = []
        seen_hashes: set[str] = set()
        for record in records:
            reason, score = self._reject_reason(record, seen_hashes=seen_hashes)
            if reason is None:
                accepted.append((record, score))
            else:
                rejected.append({"source": record.source, "score": score, "reason": reason})

        manifests: list[dict[str, object]] = []
        for shard_index, start in enumerate(range(0, len(accepted), self.shard_size)):
            rows = accepted[start : start + self.shard_size]
            shard_path = self.output_dir / f"shard-{shard_index:05d}.jsonl"
            content = "".join(
                json.dumps(
                    {
                        "text": record.text,
                        "source": record.source,
                        "license": record.license,
                        "bucket": record.bucket,
                        "quality": asdict(record.quality),
                        "quality_score": score,
                        "verifier_status": record.verifier_status,
                        "source_revision": record.source_revision,
                        "civ_score": record.civ_score,
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                )
                + "\n"
                for record, score in rows
            )
            shard_path.write_text(content, encoding="utf-8")
            manifests.append(
                {
                    "path": shard_path.name,
                    "records": len(rows),
                    "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }
            )

        manifest = {
            "schema_version": 1,
            "tokenizer_version": self.tokenizer_version,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "shards": manifests,
            "ledger": self.ledger.report(),
            "rejections": rejected,
            "pipeline_order": [
                "source_validation",
                "license_provenance_validation",
                "deduplication",
                "DEL",
                "style_filter",
                "CIV_gate",
                "tokenizer_validation",
                "local_shard_creation",
                "bucket_registration",
            ],
        }
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(manifest_path)
        return manifest


class TokenShardPublisher:
    """Publish immutable local uint16 token shards with reproducible manifests."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        tokenizer_version: str,
        tokenizer_schema_version: int = TOKENIZER_SCHEMA_VERSION,
        tokens_per_shard: int = 10_000_000,
        tokenizer_sha256: str = "unknown",
    ) -> None:
        _require_v4_tokenizer(
            tokenizer_version,
            tokenizer_schema_version=tokenizer_schema_version,
        )
        self.output_dir = Path(output_dir)
        self.tokenizer_version = CANONICAL_TOKENIZER_VERSION
        self.tokenizer_schema_version = TOKENIZER_SCHEMA_VERSION
        self.tokens_per_shard = int(tokens_per_shard)
        self.tokenizer_sha256 = str(tokenizer_sha256)
        if self.tokens_per_shard <= 0:
            raise ValueError("tokens_per_shard must be positive")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _token_ids(tokenizer: object, text: str) -> list[int]:
        values = tokenizer.encode(text)
        if hasattr(values, "ids"):
            values = values.ids
        ids = [int(value) for value in values]
        if any(value < 0 or value > np.iinfo(np.uint16).max for value in ids):
            raise ValueError("Tokenizer emitted an ID outside the uint16 range")
        return ids

    def publish(
        self,
        records: Iterable[SourceRecord],
        tokenizer: object,
        *,
        allow_partial_final: bool = False,
        minimum_replay_tokens: dict[str, int] | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, object]:
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists():
            raise FileExistsError(f"Token shard publication is immutable: {manifest_path}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        buffer: list[int] = []
        buffer_segments: list[dict[str, object]] = []
        shards: list[dict[str, object]] = []
        source_revisions: set[str] = set()
        licenses: set[str] = set()
        validator = ShardedDataPipeline(
            self.output_dir / "_validation",
            tokenizer_version=self.tokenizer_version,
            # Native corpus publication consumes a hash/MinHash-audited source.
            # Rebuilding a retrieval index for millions of unchanged records is
            # redundant and unbounded; exact duplicate rejection remains live.
            near_duplicate_check=False,
        )
        seen_hashes: set[str] = set()
        rejection_counts: dict[str, int] = {}
        accepted_records = 0
        source_counts: dict[str, int] = {}
        source_token_counts: dict[str, int] = {}
        source_class_token_counts: dict[str, int] = {}
        source_class_replayed_tokens: dict[str, int] = {}
        verifier_counts: dict[str, int] = {}
        active_source_class = ""
        replay_targets = {
            str(name): max(0, int(tokens))
            for name, tokens in (minimum_replay_tokens or {}).items()
        }
        oversized_replay = {
            name: tokens
            for name, tokens in replay_targets.items()
            if tokens > self.tokens_per_shard
        }
        if oversized_replay:
            raise ValueError(
                "minimum replay targets must fit inside one source-pure shard: "
                f"{oversized_replay}"
            )

        def consume_segment_metadata(token_count: int) -> dict[str, object]:
            remaining = int(token_count)
            records: set[str] = set()
            sources: dict[str, int] = {}
            source_classes: dict[str, int] = {}
            revisions: set[str] = set()
            shard_licenses: set[str] = set()
            verifier_statuses: dict[str, int] = {}
            quality_sums: dict[str, float] = {}
            while remaining > 0 and buffer_segments:
                segment = buffer_segments[0]
                take = min(remaining, int(segment["tokens"]))
                source = str(segment["source"])
                sources[source] = sources.get(source, 0) + take
                source_class = str(segment["source_class"])
                source_classes[source_class] = source_classes.get(source_class, 0) + take
                records.add(str(segment["record_hash"]))
                revisions.add(str(segment["revision"]))
                shard_licenses.add(str(segment["license"]))
                verifier = str(segment["verifier_status"])
                verifier_statuses[verifier] = verifier_statuses.get(verifier, 0) + take
                quality = segment["quality"]
                if isinstance(quality, dict):
                    for name, value in quality.items():
                        quality_sums[name] = quality_sums.get(name, 0.0) + float(value) * take
                segment["tokens"] = int(segment["tokens"]) - take
                remaining -= take
                if int(segment["tokens"]) == 0:
                    buffer_segments.pop(0)
            return {
                "record_count": len(records),
                "source_token_mix": sources,
                "source_class_token_mix": source_classes,
                "source_revisions": sorted(revisions),
                "licenses": sorted(shard_licenses),
                "verifier_token_distribution": verifier_statuses,
                "quality_distribution": {
                    name: round(value / max(1, token_count), 6)
                    for name, value in quality_sums.items()
                },
            }

        def write_shard(values: list[int], index: int, *, partial: bool) -> None:
            array = np.asarray(values, dtype=np.uint16)
            target = self.output_dir / f"tokens-{index:05d}.npy"
            temporary = target.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
            temporary.replace(target)
            metadata = consume_segment_metadata(int(array.size))
            source_classes = dict(metadata["source_class_token_mix"])
            if len(source_classes) != 1:
                raise RuntimeError("Token shard publication mixed source classes")
            shards.append(
                {
                    "path": target.name,
                    "tokens": int(array.size),
                    "dtype": "uint16",
                    "partial": partial,
                    "sha256": self._sha256(target),
                    "tokenizer_sha256": self.tokenizer_sha256,
                    "source_class": next(iter(source_classes)),
                    **metadata,
                }
            )
            if progress_callback is not None:
                progress_callback(dict(shards[-1]))

        def materialize_minimum_replay(source_class: str) -> None:
            target = replay_targets.get(source_class, 0)
            existing = source_class_token_counts.get(source_class, 0)
            needed = max(0, target - existing)
            if needed == 0:
                return
            if not buffer or not buffer_segments:
                raise RuntimeError(
                    f"Cannot replay empty source class {source_class!r} to {target} tokens"
                )
            templates: list[tuple[list[int], dict[str, object]]] = []
            cursor = 0
            for segment in buffer_segments:
                count = int(segment["tokens"])
                if count <= 0:
                    continue
                templates.append((list(buffer[cursor : cursor + count]), dict(segment)))
                cursor += count
            if cursor != len(buffer) or not templates:
                raise RuntimeError("Replay metadata does not cover the pending token buffer")
            while needed > 0:
                for token_template, segment_template in templates:
                    take = min(needed, len(token_template))
                    if take <= 0:
                        continue
                    buffer.extend(token_template[:take])
                    replay_segment = dict(segment_template)
                    replay_segment["tokens"] = take
                    buffer_segments.append(replay_segment)
                    source = str(replay_segment["source"])
                    source_token_counts[source] = source_token_counts.get(source, 0) + take
                    source_class_token_counts[source_class] = (
                        source_class_token_counts.get(source_class, 0) + take
                    )
                    source_class_replayed_tokens[source_class] = (
                        source_class_replayed_tokens.get(source_class, 0) + take
                    )
                    needed -= take
                    if needed == 0:
                        break

        for record in records:
            reason, _ = validator._reject_reason(
                record,
                seen_hashes=seen_hashes,
            )
            if reason is not None:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            accepted_records += 1
            source_revisions.add(record.source_revision)
            licenses.add(record.license)
            source_counts[record.source] = source_counts.get(record.source, 0) + 1
            verifier_counts[record.verifier_status] = (
                verifier_counts.get(record.verifier_status, 0) + 1
            )
            token_ids = self._token_ids(tokenizer, record.text)
            eos_id = int(getattr(tokenizer, "eos_token_id", -1))
            if eos_id >= 0 and (not token_ids or token_ids[-1] != eos_id):
                token_ids.append(eos_id)
            source_token_counts[record.source] = source_token_counts.get(record.source, 0) + len(
                token_ids
            )
            source_class = record.source_class.strip() or record.bucket.strip() or "unclassified"
            if active_source_class and source_class != active_source_class and buffer:
                materialize_minimum_replay(active_source_class)
                write_shard(buffer, len(shards), partial=True)
                buffer = []
            active_source_class = source_class
            source_class_token_counts[source_class] = (
                source_class_token_counts.get(source_class, 0) + len(token_ids)
            )
            buffer.extend(token_ids)
            buffer_segments.append(
                {
                    "tokens": len(token_ids),
                    "source": record.source,
                    "source_class": source_class,
                    "revision": record.source_revision,
                    "license": record.license,
                    "verifier_status": record.verifier_status,
                    "quality": asdict(record.quality),
                    "record_hash": hashlib.sha256(record.text.strip().encode("utf-8")).hexdigest(),
                }
            )
            while len(buffer) >= self.tokens_per_shard:
                write_shard(
                    buffer[: self.tokens_per_shard],
                    len(shards),
                    partial=False,
                )
                del buffer[: self.tokens_per_shard]
        if buffer:
            materialize_minimum_replay(active_source_class)
        if buffer and allow_partial_final:
            write_shard(buffer, len(shards), partial=True)
            buffer = []

        manifest = {
            "schema_version": 4,
            "tokenizer_schema_version": self.tokenizer_schema_version,
            "tokenizer_version": self.tokenizer_version,
            "tokenizer_sha256": self.tokenizer_sha256,
            "tokens_per_shard": self.tokens_per_shard,
            "total_tokens": sum(int(item["tokens"]) for item in shards),
            "pending_tokens": len(buffer),
            "accepted_records": accepted_records,
            "source_record_mix": source_counts,
            "source_token_mix": source_token_counts,
            "source_class_token_mix": source_class_token_counts,
            "source_class_replayed_tokens": source_class_replayed_tokens,
            "verifier_record_distribution": verifier_counts,
            "rejection_counts": rejection_counts,
            "quality": validator.ledger.report(),
            "source_revisions": sorted(source_revisions),
            "licenses": sorted(licenses),
            "commoncrawl_terms_validated": any(
                "commoncrawl" in value.lower() or "odc-by" in value.lower() for value in licenses
            ),
            "shards": shards,
            "pipeline_order": [
                "source_validation",
                "license_provenance_validation",
                "deduplication",
                "DEL",
                "style_filter",
                "CIV_gate",
                "tokenizer_validation",
                "local_shard_creation",
                "bucket_registration",
            ],
        }
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
        return manifest
