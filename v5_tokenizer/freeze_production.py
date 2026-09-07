"""B3 — freeze the production tokenizer artifact + identity receipt.

Freezes the E1 tournament 24,576-entry artifact (identity verified through
`load_verified_tokenizer`: artifact SHA-256, vocabulary size, special tokens
PAD 0 / UNK 1 / BOS 2 / EOS 3, no normalizer) into an immutable production
identity receipt bound to the trainer-config and corpus-manifest hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

IDENTITY_RECEIPT_SCHEMA = "anra-v5-production-tokenizer-identity/v1"
ARTIFACT_RELPATH = "artifacts/e1/local_tournament/tokenizer-24576.json.gz"
RESULT_RELPATH = "artifacts/e1/local_tournament/result.json"


def freeze_production_identity(*, repo_root: str | Path,
                               out_path: str | Path | None = None
                               ) -> dict[str, Any]:
    """Verify the tournament artifact and emit the frozen production
    identity receipt. Fails closed on any identity mismatch."""
    from v5_tokenizer.artifact import load_verified_tokenizer

    root = Path(repo_root)
    artifact = root / ARTIFACT_RELPATH
    result = json.loads((root / RESULT_RELPATH).read_text(encoding="utf-8"))
    row = next(r for r in result["candidate_rows"]
               if r["vocabulary_size"] == 24_576)
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if artifact_sha != row["artifact_sha256"]:
        raise ValueError("artifact bytes contradict the tournament receipt")
    trainer_payload = json.dumps(result["trainer"], sort_keys=True).encode(
        "utf-8")
    trainer_sha = hashlib.sha256(trainer_payload).hexdigest()
    # full verification: bytes, vocab, specials, normalization
    _backend, identity = load_verified_tokenizer(
        artifact, expected_sha256=artifact_sha, vocabulary_size=24_576,
        trainer_config_sha256=trainer_sha,
        corpus_manifest_sha256=result["corpus_manifest_sha256"])
    receipt = {
        "schema": IDENTITY_RECEIPT_SCHEMA,
        "artifact_path": ARTIFACT_RELPATH,
        "artifact_sha256": artifact_sha,
        "vocabulary_size": identity.vocabulary_size,
        "special_token_ids": dict(identity.special_token_ids),
        "trainer_config_sha256": identity.trainer_config_sha256,
        "corpus_manifest_sha256": identity.corpus_manifest_sha256,
        "status": "FROZEN",
    }
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    return receipt


__all__ = ["freeze_production_identity"]
