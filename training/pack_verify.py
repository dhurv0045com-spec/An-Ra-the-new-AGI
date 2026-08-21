"""Fail-closed verification for V4 continuation token packs.

A pack is a directory of ``train/*.npy`` int16/int32 token shards plus a
``manifest.json`` declaring shard names, dtypes, token counts, and SHA-256
hashes. The TPU trainer refuses to start unless every declared hash matches.
This closes the gap that let a wrong dataset silently reach the trainer.

Manifest schema (v1):
{
  "schema": "anra-token-pack/v1",
  "block_size": 2048,
  "total_tokens": int,            # tokens across train shards
  "shards": [
    {"file": "train/shard_00000.npy", "tokens": int, "sha256": "..."},
    ...
  ]
}
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class PackVerificationError(RuntimeError):
    """Raised when a token pack fails manifest verification."""


@dataclass(frozen=True)
class VerifiedPack:
    root: Path
    block_size: int
    total_tokens: int
    shard_paths: tuple[Path, ...]
    total_windows: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pack(root: Path) -> VerifiedPack:
    """Verify a token pack against its manifest. Raises on any mismatch."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise PackVerificationError(
            f"token pack has no manifest.json: {root}. "
            "Refusing to guess the dataset (fail closed)."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackVerificationError(f"unreadable manifest: {exc}") from exc

    if manifest.get("schema") != "anra-token-pack/v1":
        raise PackVerificationError(
            f"unsupported pack schema {manifest.get('schema')!r}; "
            "expected 'anra-token-pack/v1'"
        )
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise PackVerificationError("manifest declares no shards")

    verified: list[Path] = []
    total_tokens = 0
    for entry in shards:
        rel = str(entry.get("file", ""))
        expected_hash = str(entry.get("sha256", ""))
        declared_tokens = entry.get("tokens")
        if not rel or len(expected_hash) != 64:
            raise PackVerificationError(f"malformed shard entry: {entry}")
        shard_path = root / rel
        if not shard_path.is_file():
            raise PackVerificationError(f"declared shard missing: {rel}")
        actual_hash = _sha256_file(shard_path)
        if actual_hash != expected_hash:
            raise PackVerificationError(
                f"shard hash mismatch for {rel}: "
                f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
            )
        if isinstance(declared_tokens, int):
            array = np.load(shard_path, mmap_mode="r")
            actual_tokens = int(array.shape[-1]) if array.ndim else 0
            del array
            if actual_tokens != declared_tokens:
                raise PackVerificationError(
                    f"token count mismatch for {rel}: manifest says "
                    f"{declared_tokens}, file holds {actual_tokens}"
                )
            total_tokens += declared_tokens
        verified.append(shard_path)

    block_size = int(manifest.get("block_size", 2048))
    if block_size <= 0 or total_tokens <= block_size:
        raise PackVerificationError(
            f"pack too small: {total_tokens} tokens at block {block_size}"
        )
    return VerifiedPack(
        root=root,
        block_size=block_size,
        total_tokens=total_tokens,
        shard_paths=tuple(verified),
        total_windows=total_tokens // (block_size + 1),
    )


def build_manifest(
    root: Path, *, block_size: int, shard_glob: str = "train/*.npy"
) -> dict[str, object]:
    """Build a v1 manifest from existing .npy shards (used by pack authors)."""
    root = Path(root)
    entries = []
    total = 0
    for shard_path in sorted(root.glob(shard_glob)):
        array = np.load(shard_path, mmap_mode="r")
        tokens = int(array.shape[-1]) if array.ndim else 0
        del array
        entries.append(
            {
                "file": shard_path.relative_to(root).as_posix(),
                "tokens": tokens,
                "sha256": _sha256_file(shard_path),
            }
        )
        total += tokens
    return {
        "schema": "anra-token-pack/v1",
        "block_size": block_size,
        "total_tokens": total,
        "shards": entries,
    }
