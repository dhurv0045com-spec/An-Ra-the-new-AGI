# ruff: noqa: E402
"""Stream A baseline freeze (MASTER_PLAN Stage 1.1, step 1).

Emits one frozen-hash artifact covering the four baseline identities:

- checkpoint: full-file SHA-256 of the real 500M checkpoint (streamed);
  reported as blocked when the artifact is not on disk (it never enters git).
- tokenizer: file/vocabulary hashes plus the 500-probe encode/decode
  fingerprint, executed live and cross-checked against the frozen manifest.
- config: SHA-256 over the canonical frontier model contract and over the
  `training/v2_config.py` source bytes.
- corpus manifests: SHA-256 of every manifest under output/v2/data_manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anra.anra_paths import (
    DATA_MANIFEST_DIR,
    FRONTIER_CHECKPOINT,
    OUTPUT_V2_DIR,
    ROOT,
    TOKENIZER_MANIFEST,
)
from training.v2_config import (
    CANONICAL_SPECIAL_TOKEN_IDS,
    CANONICAL_VOCAB_SIZE,
    CHECKPOINT_SCHEMA_VERSION,
    TOKENIZER_SCHEMA_VERSION,
    V2_FRONTIER,
    V2_FRONTIER_PARAMETER_COUNT,
    V2_FRONTIER_TRANSFORMER_PARAMETER_COUNT,
)
from training.v2_runtime import _active_tokenizer_identity

BASELINE_FREEZE = OUTPUT_V2_DIR / "baseline_freeze.json"


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(explicit: str | None = None) -> Path:
    """Resolution order: explicit arg, ANRA_CHECKPOINT_PATH, canonical path."""
    candidate = explicit or os.environ.get("ANRA_CHECKPOINT_PATH", "").strip()
    if candidate:
        path = Path(candidate).expanduser()
        return path if path.is_absolute() else (ROOT / path).resolve()
    return FRONTIER_CHECKPOINT


def freeze_checkpoint(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "available": False,
            "status": "blocked_on_artifact",
            "searched": str(path),
            "note": "The real 500M checkpoint lives outside git; set "
            "ANRA_CHECKPOINT_PATH or restore it to the canonical path.",
        }
    return {
        "available": True,
        "status": "frozen",
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def freeze_tokenizer() -> dict[str, object]:
    identity = _active_tokenizer_identity()
    if not identity.get("available"):
        return {"available": False, "status": "missing"}
    recorded = {}
    if TOKENIZER_MANIFEST.exists():
        recorded = json.loads(TOKENIZER_MANIFEST.read_text(encoding="utf-8"))
    probe_match = (
        str(recorded.get("probe_sha256", "")) == str(identity["probe_sha256"])
        if recorded
        else None
    )
    return {
        **identity,
        "status": "frozen",
        "manifest_path": str(TOKENIZER_MANIFEST),
        "manifest_probe_sha256": recorded.get("probe_sha256"),
        "probe_match_vs_manifest": probe_match,
    }


def freeze_config() -> dict[str, object]:
    contract = {
        "model_profile": "frontier",
        "n_embd": V2_FRONTIER.n_embd,
        "n_layer": V2_FRONTIER.n_layer,
        "n_head": V2_FRONTIER.n_head,
        "n_kv_head": V2_FRONTIER.n_kv_head,
        "block_size": V2_FRONTIER.block_size,
        "parameter_count": V2_FRONTIER_PARAMETER_COUNT,
        "transformer_parameter_count": V2_FRONTIER_TRANSFORMER_PARAMETER_COUNT,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "tokenizer_schema_version": TOKENIZER_SCHEMA_VERSION,
        "canonical_vocab_size": CANONICAL_VOCAB_SIZE,
        "canonical_special_token_ids": CANONICAL_SPECIAL_TOKEN_IDS,
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    source = ROOT / "training" / "v2_config.py"
    return {
        "status": "frozen",
        "contract": contract,
        "contract_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_path": str(source),
        "source_sha256": _sha256_file(source),
    }


def freeze_corpus_manifests() -> dict[str, object]:
    manifests: dict[str, dict[str, object]] = {}
    if DATA_MANIFEST_DIR.exists():
        for path in sorted(DATA_MANIFEST_DIR.glob("*.json")):
            manifests[path.name] = {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return {
        "status": "frozen" if manifests else "empty",
        "directory": str(DATA_MANIFEST_DIR),
        "manifests": manifests,
        "count": len(manifests),
    }


def build_freeze_report(checkpoint: Path) -> dict[str, object]:
    tokenizer = freeze_tokenizer()
    report = {
        "schema_version": 1,
        "generated_at": time.time(),
        "checkpoint": freeze_checkpoint(checkpoint),
        "tokenizer": tokenizer,
        "config": freeze_config(),
        "corpus_manifests": freeze_corpus_manifests(),
    }
    report["frozen"] = bool(
        tokenizer.get("status") == "frozen"
        and tokenizer.get("probe_match_vs_manifest") is not False
        and report["config"]["status"] == "frozen"
        and report["corpus_manifests"]["count"] >= 1
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Stream A baseline hashes.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--json-out", default=str(BASELINE_FREEZE))
    parser.add_argument(
        "--allow-missing-checkpoint",
        action="store_true",
        help="Freeze tokenizer/config/corpus identities even while the "
        "checkpoint artifact is still blocked.",
    )
    args = parser.parse_args()

    report = build_freeze_report(resolve_checkpoint(args.checkpoint))
    output = Path(args.json_out)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["frozen"]:
        return 2
    if not report["checkpoint"]["available"] and not args.allow_missing_checkpoint:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
