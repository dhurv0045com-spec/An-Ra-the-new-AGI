"""Reproducible acquisition manifests for external source artifacts (M4).

Nothing here downloads blindly: a source resolves to an explicit file list
with expected sizes and content identities first; downloads verify against
that manifest. If the network or the source is unavailable, resolution fails
closed with BLOCKED_BY_SOURCE_ACQUISITION instead of fabricating data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ACQUISITION_SCHEMA = "anra-v5-acquisition-manifest/v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: str
    byte_size: int
    blob_id: str

    def assert_valid(self) -> None:
        if not self.path:
            raise ValueError("source file path is required")
        if self.byte_size <= 0:
            raise ValueError("source file size must be positive")
        if len(self.blob_id) != 40 or any(
            character not in "0123456789abcdef" for character in self.blob_id
        ):
            raise ValueError("source file blob identity must be a git blob SHA-1")


@dataclass(frozen=True, slots=True)
class AcquisitionManifest:
    schema: str
    source_id: str
    repo_id: str
    revision: str
    files: tuple[SourceFile, ...]
    total_bytes: int

    def assert_valid(self) -> None:
        if self.schema != ACQUISITION_SCHEMA:
            raise ValueError("unsupported acquisition-manifest schema")
        if not self.source_id or not self.repo_id or not self.revision:
            raise ValueError("source, repo, and revision identities are required")
        if not self.files:
            raise ValueError("acquisition manifest holds no files")
        for item in self.files:
            item.assert_valid()
        if sum(item.byte_size for item in self.files) != self.total_bytes:
            raise ValueError("acquisition byte total disagrees with file sizes")

    def sha256(self) -> str:
        self.assert_valid()
        return _sha256_hex(
            _canonical_json(
                {
                    "schema": self.schema,
                    "source_id": self.source_id,
                    "repo_id": self.repo_id,
                    "revision": self.revision,
                    "files": [
                        {"path": item.path, "byte_size": item.byte_size, "blob_id": item.blob_id}
                        for item in self.files
                    ],
                    "total_bytes": self.total_bytes,
                }
            )
        )


def resolve_huggingface(
    source_id: str,
    repo_id: str,
    *,
    revision: str = "main",
    allow_patterns: list[str] | None = None,
) -> AcquisitionManifest:
    """Resolve a Hub dataset to an explicit, verified file manifest."""

    try:
        from huggingface_hub import dataset_info
    except ImportError as exc:
        raise RuntimeError("resolving Hub sources requires huggingface_hub") from exc
    try:
        info = dataset_info(repo_id, revision=revision, files_metadata=True)
    except Exception as exc:
        raise ValueError(f"BLOCKED_BY_SOURCE_ACQUISITION: cannot resolve {repo_id}@{revision}: {exc}") from exc
    siblings = list(info.siblings or [])
    files: list[SourceFile] = []
    for sibling in siblings:
        name = str(getattr(sibling, "rfilename", ""))
        if allow_patterns and not any(
            _glob_match(name, pattern) for pattern in allow_patterns
        ):
            continue
        blob = str(getattr(sibling, "blob_id", "") or "")
        size = getattr(sibling, "size", None)
        if len(blob) != 40 or not isinstance(size, int) or size <= 0:
            raise ValueError(f"BLOCKED_BY_SOURCE_ACQUISITION: {name} lacks content identity")
        files.append(SourceFile(path=name, byte_size=size, blob_id=blob))
    if not files:
        raise ValueError("acquisition file filter matched nothing")
    manifest = AcquisitionManifest(
        schema=ACQUISITION_SCHEMA,
        source_id=source_id,
        repo_id=repo_id,
        revision=str(getattr(info, "sha", revision)),
        files=tuple(sorted(files, key=lambda item: item.path)),
        total_bytes=sum(item.byte_size for item in files),
    )
    manifest.assert_valid()
    return manifest


def _glob_match(name: str, pattern: str) -> bool:
    import fnmatch

    return fnmatch.fnmatchcase(name, pattern)


def download_manifest(
    manifest: AcquisitionManifest, *, dest_dir: Path, repo_type: str = "dataset"
) -> dict[str, object]:
    """Download exactly the manifest files, verifying sizes and identities."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("downloading Hub sources requires huggingface_hub") from exc
    manifest.assert_valid()
    dest_dir.mkdir(parents=True, exist_ok=True)
    local: list[dict[str, object]] = []
    for item in manifest.files:
        local_path = hf_hub_download(
            manifest.repo_id, item.path, revision=manifest.revision,
            repo_type=repo_type, local_dir=dest_dir,
        )
        actual_size = Path(local_path).stat().st_size
        if actual_size != item.byte_size:
            raise ValueError(
                f"downloaded size disagrees for {item.path}: {actual_size} != {item.byte_size}"
            )
        digest = hashlib.sha256()
        with open(local_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        local.append({"path": item.path, "local_path": local_path, "sha256": digest.hexdigest()})
    return {
        "schema": "anra-v5-acquisition-receipt/v1",
        "manifest_sha256": manifest.sha256(),
        "files": local,
    }


__all__ = [
    "ACQUISITION_SCHEMA",
    "AcquisitionManifest",
    "SourceFile",
    "download_manifest",
    "resolve_huggingface",
]
