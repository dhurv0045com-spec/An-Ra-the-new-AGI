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
    """Raised when a token pack fails manifest or semantic verification."""


@dataclass(frozen=True)
class VerifiedPack:
    root: Path
    block_size: int
    total_tokens: int
    shard_paths: tuple[Path, ...]
    total_windows: int
    manifest_sha256: str  # resume identity: binds pack_step to THIS data


def _semantic_validate(array: np.ndarray, rel: str, *, vocab_size: int) -> int:
    """A correct hash does not imply valid model input. Enforce semantics."""
    if array.ndim != 1:
        raise PackVerificationError(f"shard {rel} must be a 1-D token array, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise PackVerificationError(f"shard {rel} must have integer dtype, got {array.dtype}")
    if array.size == 0:
        raise PackVerificationError(f"shard {rel} is empty")
    minimum = int(array.min())
    maximum = int(array.max())
    if minimum < 0:
        raise PackVerificationError(f"shard {rel} contains negative token IDs (min={minimum})")
    if maximum >= vocab_size:
        raise PackVerificationError(
            f"shard {rel} token ID {maximum} exceeds vocab_size {vocab_size}"
        )
    return int(array.shape[0])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pack(
    root: Path, *, vocab_size: int = 32_768, expected_block_size: int | None = None
) -> VerifiedPack:
    """Verify a token pack: hashes, semantics, and manifest consistency.

    ``expected_block_size`` (when given) must equal the manifest block size -
    the trainer passes the model's context so a mismatched pack cannot train.
    """
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

    seen_paths: set[str] = set()
    verified: list[Path] = []
    total_tokens = 0
    for entry in shards:
        rel = str(entry.get("file", ""))
        expected_hash = str(entry.get("sha256", ""))
        declared_tokens = entry.get("tokens")
        if not rel or len(expected_hash) != 64:
            raise PackVerificationError(f"malformed shard entry: {entry}")
        if rel in seen_paths:
            raise PackVerificationError(f"duplicate shard path in manifest: {rel}")
        seen_paths.add(rel)
        shard_path = root / rel
        if not shard_path.is_file():
            raise PackVerificationError(f"declared shard missing: {rel}")
        actual_hash = _sha256_file(shard_path)
        if actual_hash != expected_hash:
            raise PackVerificationError(
                f"shard hash mismatch for {rel}: "
                f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
            )
        array = np.load(shard_path, mmap_mode="r")
        # Semantic validation runs even when the manifest omits counts:
        # a correctly-hashed malformed array must still be rejected.
        actual_tokens = _semantic_validate(array, rel, vocab_size=vocab_size)
        del array
        if isinstance(declared_tokens, int) and actual_tokens != declared_tokens:
            raise PackVerificationError(
                f"token count mismatch for {rel}: manifest says "
                f"{declared_tokens}, file holds {actual_tokens}"
            )
        total_tokens += actual_tokens
        verified.append(shard_path)

    declared_total = manifest.get("total_tokens")
    if isinstance(declared_total, int) and declared_total != total_tokens:
        raise PackVerificationError(
            f"manifest total_tokens {declared_total} != shard sum {total_tokens}"
        )

    block_size = int(manifest.get("block_size", 2048))
    if expected_block_size is not None and block_size != expected_block_size:
        raise PackVerificationError(
            f"pack block_size {block_size} != required {expected_block_size}"
        )
    if block_size <= 0 or total_tokens <= block_size:
        raise PackVerificationError(
            f"pack too small: {total_tokens} tokens at block {block_size}"
        )
    manifest_sha = _sha256_file(manifest_path)
    return VerifiedPack(
        root=root,
        block_size=block_size,
        total_tokens=total_tokens,
        shard_paths=tuple(verified),
        total_windows=total_tokens // (block_size + 1),
        manifest_sha256=manifest_sha,
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
