"""License-bound, resumable acquisition of V4 SFT JSONL source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.posttraining_contract import REQUIRED_SFT_CATEGORIES

SFT_SOURCE_REGISTRY_SCHEMA = "anra-sft-source-registry/v1"
_COPY_BLOCK_SIZE = 8 * 1024 * 1024


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(_COPY_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _validate_digest(value: object, *, field: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return digest


def _safe_filename(value: object) -> str:
    filename = str(value).strip()
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError(f"unsafe SFT source filename: {filename!r}")
    return filename


@dataclass(frozen=True)
class SFTSource:
    source_id: str
    url: str
    filename: str
    sha256: str
    license: str
    category: str | None


def load_sft_source_registry(
    path: str | Path, *, allow_local_sources: bool = False
) -> tuple[SFTSource, ...]:
    registry_path = Path(path).resolve()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SFT_SOURCE_REGISTRY_SCHEMA:
        raise ValueError("unsupported SFT source registry schema")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("SFT source registry requires at least one source")
    sources: list[SFTSource] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, dict):
            raise ValueError(f"SFT source {index} must be an object")
        source_id = str(raw.get("source_id", "")).strip()
        license_id = str(raw.get("license", "")).strip()
        category = str(raw.get("category", "")).strip() or None
        url = str(raw.get("url", "")).strip()
        parsed = urllib.parse.urlparse(url)
        allowed_schemes = {"https"}
        if allow_local_sources:
            allowed_schemes.add("file")
        if parsed.scheme not in allowed_schemes:
            raise ValueError(
                f"SFT source {source_id or index!r} must use an approved URL scheme "
                f"{sorted(allowed_schemes)}"
            )
        if (
            not source_id
            or source_id in seen
            or not license_id
            or license_id.lower()
            in {
                "unknown",
                "unlicensed",
                "none",
            }
        ):
            raise ValueError(f"SFT source {index} has missing/duplicate provenance or license")
        seen.add(source_id)
        if category is not None and category not in REQUIRED_SFT_CATEGORIES:
            raise ValueError(f"SFT source {source_id} has unsupported category {category!r}")
        sources.append(
            SFTSource(
                source_id=source_id,
                url=url,
                filename=_safe_filename(raw.get("filename", "")),
                sha256=_validate_digest(raw.get("sha256"), field=f"source {source_id} sha256"),
                license=license_id,
                category=category,
            )
        )
    return tuple(sources)


def _download(source: SFTSource, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(source.url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        response = urllib.request.urlopen(request, timeout=60)  # noqa: S310 - digest verifies bytes
    except Exception as error:
        raise RuntimeError(f"could not download SFT source {source.source_id}: {error}") from error
    status = getattr(response, "status", None)
    append = offset > 0 and status == 206
    if offset and not append:
        partial.unlink(missing_ok=True)
    with response, partial.open("ab" if append else "wb") as handle:
        shutil.copyfileobj(response, handle, _COPY_BLOCK_SIZE)
    actual = sha256_file(partial)
    if actual != source.sha256:
        raise ValueError(
            f"downloaded SFT source digest mismatch for {source.source_id}: "
            f"expected={source.sha256} actual={actual}"
        )
    os.replace(partial, destination)


def download_sft_sources(
    registry_path: str | Path,
    output_dir: str | Path,
    *,
    allow_local_sources: bool = False,
) -> dict[str, Any]:
    """Fetch every registered source and produce a receipt for dataset building."""

    registry = Path(registry_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    sources = load_sft_source_registry(registry, allow_local_sources=allow_local_sources)
    receipts: list[dict[str, object]] = []
    for source in sources:
        target = destination / source.filename
        if target.is_file() and sha256_file(target) == source.sha256:
            status = "already_verified"
        else:
            target.unlink(missing_ok=True)
            _download(source, target)
            status = "downloaded_verified"
        receipts.append(
            {
                "source_id": source.source_id,
                "url": source.url,
                "license": source.license,
                "category": source.category,
                "path": str(target),
                "sha256": source.sha256,
                "size_bytes": target.stat().st_size,
                "status": status,
            }
        )
    report = {
        "schema": "anra-sft-source-receipts/v1",
        "registry_sha256": sha256_file(registry),
        "sources": receipts,
    }
    receipt = destination / "sft-v4-source-receipts.json"
    temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, receipt)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Download hash-verified V4 SFT source JSONL files")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-local-sources", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            download_sft_sources(
                args.registry,
                args.output_dir,
                allow_local_sources=args.allow_local_sources,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
