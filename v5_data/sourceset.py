"""Git-frozen scientific source sets: exact commit bytes, never worktree discovery.

Canary and experiment corpora must come from an exact commit tree via
``git ls-tree`` / ``git show``. Untracked files (``temp.txt``) cannot enter
because they are not in the tree; dirty tracked files cannot leak because
only committed blob bytes are consumed. Raw bytes and processed content
stay distinct forever: SourceSetManifest binds raw identities, while
ProcessedDocumentManifest binds the transformation separately.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


SOURCESCHEMA = "anra-v5-sourceset-manifest/v1"
PROCESSED_SCHEMA = "anra-v5-processed-documents/v1"


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, check=True, text=True
    )
    return completed.stdout


def _git_bytes(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True)
    return completed.stdout


def _ls_tree(repo: Path, commit: str) -> list[tuple[str, str, str]]:
    """List (mode, blob_sha, path) for every blob in one plumbing call."""

    raw = subprocess.run(
        ["git", "ls-tree", "-r", "-z", commit], cwd=repo,
        capture_output=True, check=True,
    ).stdout.split(b"\0")
    entries: list[tuple[str, str, str]] = []
    for record in raw:
        if not record:
            continue
        meta, _, path = record.partition(b"\t")
        mode, kind, blob = meta.decode("ascii").split(" ")
        if kind != "blob":
            continue
        entries.append((mode, blob, path.decode("utf-8", errors="surrogateescape")))
    return entries


def _batch_contents(repo: Path, blobs: list[str]) -> dict[str, bytes]:
    """Fetch many blob bytes through one ``git cat-file --batch`` process."""

    if not blobs:
        return {}
    completed = subprocess.run(
        ["git", "cat-file", "--batch"],
        input=("\n".join(blobs) + "\n").encode("ascii"),
        cwd=repo, capture_output=True, check=True,
    )
    stream = completed.stdout
    out: dict[str, bytes] = {}
    position = 0
    for blob in blobs:
        header_end = stream.index(b"\n", position)
        header = stream[position:header_end].decode("ascii").split(" ")
        if header[0] != blob:
            raise ValueError(f"batch fetch mismatch for {blob}")
        if header[1] == "missing":
            raise ValueError(f"blob missing from object store: {blob}")
        if header[1] != "blob":
            raise ValueError(f"not a blob object: {blob}")
        size = int(header[2])
        start = header_end + 1
        out[blob] = stream[start:start + size]
        position = start + size + 1
    return out


def resolve_commit(repo: Path, revision: str = "HEAD") -> str:
    """Resolve a revision to a full commit SHA, failing closed."""

    commit = _git(repo, "rev-parse", revision).strip()
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("cannot pin an exact source commit")
    return commit


@dataclass(frozen=True, slots=True)
class SourceEntry:
    path: str
    blob_sha256: str
    raw_sha256: str
    byte_size: int
    category: str


@dataclass(frozen=True, slots=True)
class SourceSetManifest:
    schema: str
    source_commit: str
    inclusion_rule: str
    inclusion_rule_sha256: str
    worktree_clean_at_freeze: bool
    entries: tuple[SourceEntry, ...]

    def assert_valid(self) -> None:
        if self.schema != SOURCESCHEMA:
            raise ValueError("unsupported sourceset schema")
        if len(self.source_commit) != 40:
            raise ValueError("source commit must be a full SHA-1")
        if not self.inclusion_rule or not self.entries:
            raise ValueError("inclusion rule and entries are required")
        paths = [entry.path for entry in self.entries]
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate source paths")
        for entry in self.entries:
            if len(entry.blob_sha256) != 40 or len(entry.raw_sha256) != 64:
                raise ValueError("entry identities must be git blob SHA-1 + SHA-256")
            if entry.byte_size < 0 or not entry.category:
                raise ValueError("entry size/category invalid")

    def sha256(self) -> str:
        self.assert_valid()
        return _sha256_hex(
            _canonical_json(
                {
                    "schema": self.schema,
                    "source_commit": self.source_commit,
                    "inclusion_rule": self.inclusion_rule,
                    "inclusion_rule_sha256": self.inclusion_rule_sha256,
                    "worktree_clean_at_freeze": self.worktree_clean_at_freeze,
                    "entries": [asdict(entry) for entry in self.entries],
                }
            )
        )


def freeze_sourceset(
    repo: Path,
    *,
    revision: str,
    suffixes: tuple[str, ...],
    exclude_parts: tuple[str, ...],
    categories: Mapping[str, str],
    max_bytes: int,
) -> SourceSetManifest:
    """Freeze tracked text files from an exact commit tree."""

    commit = resolve_commit(repo, revision)
    rule = (
        f"suffixes={sorted(suffixes)};exclude={sorted(exclude_parts)};"
        f"max_bytes={max_bytes};tree={commit}"
    )
    candidates: list[tuple[str, str]] = []
    for _mode, blob, path in _ls_tree(repo, commit):
        suffix = "".join(Path(path).suffixes[-1:])
        if suffix.lower() not in suffixes:
            continue
        if any(part in exclude_parts for part in Path(path).parts[:-1]):
            continue
        candidates.append((path, blob))
    candidates.sort()
    contents = _batch_contents(repo, [blob for _, blob in candidates])
    entries: list[SourceEntry] = []
    total = 0
    for path, blob in candidates:
        raw = contents[blob]
        if not raw.strip():
            continue
        total += len(raw)
        if total > max_bytes:
            break
        suffix = "".join(Path(path).suffixes[-1:]).lower()
        category = "other"
        for suffix_key, label in categories.items():
            if suffix == suffix_key:
                category = label
        entries.append(
            SourceEntry(
                path=path,
                blob_sha256=blob,
                raw_sha256=_sha256_hex(raw),
                byte_size=len(raw),
                category=category,
            )
        )
    if not entries:
        raise ValueError("frozen source set is empty")
    clean = not _git(repo, "status", "--porcelain").strip()
    manifest = SourceSetManifest(
        schema=SOURCESCHEMA,
        source_commit=commit,
        inclusion_rule=rule,
        inclusion_rule_sha256=_sha256_hex(rule.encode("utf-8")),
        worktree_clean_at_freeze=clean,
        entries=tuple(entries),
    )
    manifest.assert_valid()
    return manifest


def verify_sourceset(repo: Path, manifest: SourceSetManifest) -> None:
    """Re-hash every entry from the pinned commit; refuse drift or substitution."""

    manifest.assert_valid()
    if resolve_commit(repo, manifest.source_commit) != manifest.source_commit:
        raise ValueError("pinned source commit is not reachable")
    tree = {
        path: blob for _mode, blob, path in _ls_tree(repo, manifest.source_commit)
    }
    contents = _batch_contents(
        repo, [entry.blob_sha256 for entry in manifest.entries]
    )
    for entry in manifest.entries:
        if tree.get(entry.path) != entry.blob_sha256:
            raise ValueError(f"source entry changed or vanished: {entry.path}")
        raw = contents[entry.blob_sha256]
        if _sha256_hex(raw) != entry.raw_sha256 or len(raw) != entry.byte_size:
            raise ValueError(f"source bytes disagree with manifest: {entry.path}")


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    doc_id: str
    raw_sha256: str
    processed_sha256: str
    text: str


@dataclass(frozen=True, slots=True)
class ProcessedDocumentManifest:
    schema: str
    sourceset_sha256: str
    transform_spec: str
    transform_spec_sha256: str
    documents: tuple[ProcessedDocument, ...]

    def assert_valid(self) -> None:
        if self.schema != PROCESSED_SCHEMA:
            raise ValueError("unsupported processed-documents schema")
        if len(self.sourceset_sha256) != 64 or len(self.transform_spec_sha256) != 64:
            raise ValueError("manifest requires pinned sourceset and transform identities")
        if not self.transform_spec or not self.documents:
            raise ValueError("transform spec and documents are required")

    def sha256(self) -> str:
        self.assert_valid()
        return _sha256_hex(
            _canonical_json(
                {
                    "schema": self.schema,
                    "sourceset_sha256": self.sourceset_sha256,
                    "transform_spec": self.transform_spec,
                    "transform_spec_sha256": self.transform_spec_sha256,
                    "documents": [asdict(doc) for doc in self.documents],
                }
            )
        )


__all__ = [
    "PROCESSED_SCHEMA",
    "SOURCESCHEMA",
    "ProcessedDocument",
    "ProcessedDocumentManifest",
    "SourceEntry",
    "SourceSetManifest",
    "freeze_sourceset",
    "resolve_commit",
    "verify_sourceset",
]
