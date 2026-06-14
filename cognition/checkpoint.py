"""Separate cognitive-extension checkpoint and release manifest lifecycle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_cognitive_extension(
    extension: nn.Module,
    checkpoint_path: str | Path,
    *,
    base_checkpoint_hash: str,
    tokenizer_hash: str,
    source_commit: str,
    release: str,
    training_state: dict[str, object] | None = None,
) -> dict[str, object]:
    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(
        {
            "extension_schema_version": 1,
            "base_checkpoint_hash": base_checkpoint_hash,
            "tokenizer_hash": tokenizer_hash,
            "source_commit": source_commit,
            "release": release,
            "state_dict": extension.state_dict(),
            "extension_manifest": extension.manifest(),
            "training_state": training_state or {},
        },
        temporary,
    )
    temporary.replace(target)
    manifest = {
        "schema_version": 1,
        "release": release,
        "checkpoint": str(target),
        "checkpoint_sha256": _sha256(target),
        "base_checkpoint_hash": base_checkpoint_hash,
        "tokenizer_hash": tokenizer_hash,
        "source_commit": source_commit,
        "created_at": time.time(),
    }
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_cognitive_extension(
    extension: nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_base_hash: str,
    expected_tokenizer_hash: str,
) -> dict[str, object]:
    target = Path(checkpoint_path)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if payload.get("base_checkpoint_hash") != expected_base_hash:
        raise ValueError("Cognitive extension base-checkpoint contract mismatch.")
    if payload.get("tokenizer_hash") != expected_tokenizer_hash:
        raise ValueError("Cognitive extension tokenizer contract mismatch.")
    extension.load_state_dict(payload["state_dict"], strict=True)
    return payload
