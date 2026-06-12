"""Deterministic local-shard preprocessing for AN-RA V3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

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


def validate_dfc(text: str) -> bool:
    positions = [text.find(tag) for tag in DFC_TAGS]
    return all(position >= 0 for position in positions) and positions == sorted(positions)


class ShardedDataPipeline:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        tokenizer_version: str,
        shard_size: int = 10_000,
        ledger: DataEntropyLedger | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.tokenizer_version = tokenizer_version
        self.shard_size = int(shard_size)
        self.ledger = ledger or DataEntropyLedger()

    def preprocess(self, records: Iterable[SourceRecord]) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        accepted: list[tuple[SourceRecord, float]] = []
        rejected: list[dict[str, object]] = []
        for record in records:
            keep, score = self.ledger.evaluate(record.quality)
            if record.bucket == "dfc" and not validate_dfc(record.text):
                keep = False
                score = 0.0
            if keep:
                accepted.append((record, score))
            else:
                rejected.append({"source": record.source, "score": score})

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
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return manifest
