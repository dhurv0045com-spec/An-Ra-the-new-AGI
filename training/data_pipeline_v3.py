"""Deterministic local-shard preprocessing for AN-RA V3."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from retrieval import CorpusDedupIndex

from training.data_ledger import DataEntropyLedger, DataQuality

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


def validate_dfc(text: str) -> bool:
    from verification import DEFAULT_VERIFIER_REGISTRY

    result = DEFAULT_VERIFIER_REGISTRY.verify(
        "dfc_format",
        {"text": text, "tags": DFC_TAGS},
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
    ) -> None:
        self.output_dir = Path(output_dir)
        self.tokenizer_version = tokenizer_version
        self.shard_size = int(shard_size)
        self.ledger = ledger or DataEntropyLedger()
        self.style_filter = style_filter or (lambda text: bool(text.strip()))
        self.civ_floor = float(civ_floor)
        self.dedup_index = dedup_index or CorpusDedupIndex()

    @staticmethod
    def _license_allowed(record: SourceRecord) -> bool:
        license_name = record.license.strip().lower().replace("_", "-")
        if "fineweb" in record.source.lower():
            return license_name in {"odc-by", "odc-by-1.0"}
        return license_name in {
            "odc-by",
            "odc-by-1.0",
            "commoncrawl-terms",
            "cc-by",
            "cc-by-4.0",
            "mit",
            "apache-2.0",
            "bsd",
            "bsd-2-clause",
            "bsd-3-clause",
            "isc",
            "mpl-2.0",
            "public-domain",
            "owner",
        }

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
        decision = self.dedup_index.check_and_add(
            record.text,
            record_id=content_hash,
            metadata={"source": record.source, "bucket": record.bucket},
        )
        if content_hash in seen_hashes or decision.duplicate:
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
        tokenizer_schema_version: int = 3,
        tokens_per_shard: int = 10_000_000,
        tokenizer_sha256: str = "unknown",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.tokenizer_version = str(tokenizer_version)
        self.tokenizer_schema_version = int(tokenizer_schema_version)
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
        )
        seen_hashes: set[str] = set()
        rejection_counts: dict[str, int] = {}
        accepted_records = 0
        source_counts: dict[str, int] = {}
        source_token_counts: dict[str, int] = {}
        verifier_counts: dict[str, int] = {}

        def consume_segment_metadata(token_count: int) -> dict[str, object]:
            remaining = int(token_count)
            records: set[str] = set()
            sources: dict[str, int] = {}
            revisions: set[str] = set()
            shard_licenses: set[str] = set()
            verifier_statuses: dict[str, int] = {}
            quality_sums: dict[str, float] = {}
            while remaining > 0 and buffer_segments:
                segment = buffer_segments[0]
                take = min(remaining, int(segment["tokens"]))
                source = str(segment["source"])
                sources[source] = sources.get(source, 0) + take
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
            shards.append(
                {
                    "path": target.name,
                    "tokens": int(array.size),
                    "dtype": "uint16",
                    "partial": partial,
                    "sha256": self._sha256(target),
                    "tokenizer_sha256": self.tokenizer_sha256,
                    **metadata,
                }
            )

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
            buffer.extend(token_ids)
            buffer_segments.append(
                {
                    "tokens": len(token_ids),
                    "source": record.source,
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
        if buffer and allow_partial_final:
            write_shard(buffer, len(shards), partial=True)
            buffer = []

        manifest = {
            "schema_version": 3,
            "tokenizer_schema_version": self.tokenizer_schema_version,
            "tokenizer_version": self.tokenizer_version,
            "tokenizer_sha256": self.tokenizer_sha256,
            "tokens_per_shard": self.tokens_per_shard,
            "total_tokens": sum(int(item["tokens"]) for item in shards),
            "pending_tokens": len(buffer),
            "accepted_records": accepted_records,
            "source_record_mix": source_counts,
            "source_token_mix": source_token_counts,
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
