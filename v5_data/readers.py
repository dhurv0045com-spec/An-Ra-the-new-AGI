"""Parquet source readers: committed file bytes to attributed raw documents.

Each reader yields ``RawRecord`` tuples with stable source-local IDs derived
from dataset-native keys (never bare row numbers where a native key exists).
Readers do not clean, filter, or tokenize; every downstream transformation
is versioned separately so raw bytes stay traceable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class RawRecord:
    source_id: str
    local_id: str
    text: str
    metadata: tuple[tuple[str, str], ...]


def read_fineweb_edu(path: Path, *, source_id: str) -> Iterator[RawRecord]:
    """Read a FineWeb-Edu parquet file (text, id, url, language columns)."""

    table = _parquet_table(path, ["text", "id", "url"])
    texts = table.column("text").to_pylist()
    ids = table.column("id").to_pylist()
    urls = table.column("url").to_pylist()
    for text, row_id, url in zip(texts, ids, urls):
        if not isinstance(text, str) or not text.strip():
            continue
        yield RawRecord(
            source_id=source_id,
            local_id=f"fineweb:{row_id}",
            text=text,
            metadata=(("url", str(url or "")),),
        )


def read_finemath(path: Path, *, source_id: str) -> Iterator[RawRecord]:
    """Read a FineMath parquet file, keeping mathematical text verbatim."""

    table = _parquet_table(path, ["text", "url"])
    texts = table.column("text").to_pylist()
    urls = table.column("url").to_pylist()
    for index, (text, url) in enumerate(zip(texts, urls)):
        if not isinstance(text, str) or not text.strip():
            continue
        yield RawRecord(
            source_id=source_id,
            local_id=f"finemath:{index}:{abs(hash(text)) % 10**12}",
            text=text,
            metadata=(("url", str(url or "")),),
        )


def read_smoltalk(path: Path, *, source_id: str) -> Iterator[RawRecord]:
    """Render SmolTalk dialogues as role-tagged transcripts, structure kept."""

    table = _parquet_table(path, ["messages", "source"])
    for index, (messages, origin) in enumerate(
        zip(table.column("messages").to_pylist(), table.column("source").to_pylist())
    ):
        turns = []
        for message in messages or []:
            role = str(message.get("role", "unknown")).upper()
            content = str(message.get("content", "") or "")
            if content.strip():
                turns.append(f"{role}: {content}")
        if not turns:
            continue
        yield RawRecord(
            source_id=source_id,
            local_id=f"smoltalk:{index}",
            text="\n".join(turns),
            metadata=(("origin", str(origin or "")), ("turns", str(len(turns)))),
        )


def _parquet_table(path: Path, columns: list[str]) -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("reading parquet sources requires pyarrow") from exc
    if not path.is_file():
        raise ValueError(f"source file absent: {path}")
    return parquet.read_table(path, columns=columns)


__all__ = ["RawRecord", "read_finemath", "read_fineweb_edu", "read_smoltalk"]
