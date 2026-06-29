from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from anra.anra_paths import FRONTIER_CHECKPOINT, OUTPUT_V2_DIR, ROOT
from runtime.safe_load import safe_torch_load
from training.v2_config import (
    CHECKPOINT_SCHEMA_VERSION,
    EXPECTED_TOKENIZER_VOCAB_SIZE,
    TOKENIZER_V4_VOCAB_SIZE,
    V2_FRONTIER,
    frontier_parameter_count,
)


def _sha256_prefix(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _state_dict(blob: Any) -> dict[str, torch.Tensor]:
    if not isinstance(blob, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    state = blob.get("model", blob.get("model_state_dict", blob.get("model_state")))
    if not isinstance(state, dict):
        raise KeyError("checkpoint has no model/model_state_dict/model_state dictionary")
    return state


def _tensor_shape(state: dict[str, torch.Tensor], *suffixes: str) -> list[int] | None:
    for key, value in state.items():
        if any(key.endswith(suffix) for suffix in suffixes) and isinstance(value, torch.Tensor):
            return list(value.shape)
    return None


def _checkpoint_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    blob = safe_torch_load(path, map_location="cpu")
    if not isinstance(blob, dict):
        raise TypeError(f"unsupported checkpoint type: {type(blob).__name__}")
    state = _state_dict(blob)
    model_config = blob.get("model_config", {})
    tokenizer_contract = blob.get("tokenizer_contract", {})
    token_shape = _tensor_shape(state, "token_embedding_table.weight")
    lm_shape = _tensor_shape(state, "lm_head.weight")
    first_block = _tensor_shape(state, "blocks.0.attn.q_proj.weight")
    checkpoint_vocab = int(
        (model_config.get("vocab_size") if isinstance(model_config, dict) else 0)
        or (token_shape[0] if token_shape else 0)
    )
    expected_tokenizer_schema = 4 if checkpoint_vocab == TOKENIZER_V4_VOCAB_SIZE else 3

    report = {
        "ok": True,
        "path": str(path),
        "size_mb": round(path.stat().st_size / 1_048_576, 2),
        "sha256_prefix": _sha256_prefix(path),
        "checkpoint_schema_version": blob.get("checkpoint_schema_version"),
        "tokenizer_schema_version": blob.get("tokenizer_schema_version"),
        "global_step": int(blob.get("global_step", blob.get("step", 0)) or 0),
        "best_loss": float(blob.get("best_loss", float("inf"))),
        "sessions_completed": int(blob.get("sessions_completed", 0) or 0),
        "source_commit": blob.get("source_commit", "unknown"),
        "data_profile": blob.get("data_profile", "unknown"),
        "training_data_layout": blob.get("training_data_layout", "unknown"),
        "model_config": model_config,
        "tokenizer_contract": tokenizer_contract,
        "tensor_shapes": {
            "token_embedding_table.weight": token_shape,
            "lm_head.weight": lm_shape,
            "blocks.0.attn.q_proj.weight": first_block,
        },
        "expected_frontier_params": (
            frontier_parameter_count(checkpoint_vocab)
            if checkpoint_vocab >= EXPECTED_TOKENIZER_VOCAB_SIZE
            else None
        ),
    }

    errors: list[str] = []
    warnings: list[str] = []
    checkpoint_schema = int(blob.get("checkpoint_schema_version", 0) or 0)
    if checkpoint_schema <= 0 or checkpoint_schema > CHECKPOINT_SCHEMA_VERSION:
        errors.append(
            f"unsupported checkpoint_schema_version={checkpoint_schema}; "
            f"runtime maximum={CHECKPOINT_SCHEMA_VERSION}"
        )
    elif checkpoint_schema < CHECKPOINT_SCHEMA_VERSION:
        warnings.append(
            f"checkpoint schema {checkpoint_schema} requires named migration to "
            f"schema {CHECKPOINT_SCHEMA_VERSION} on load"
        )
    if blob.get("tokenizer_schema_version") != expected_tokenizer_schema:
        errors.append(
            f"tokenizer_schema_version={blob.get('tokenizer_schema_version')} "
            f"expected={expected_tokenizer_schema}"
        )
    contract_vocab = tokenizer_contract.get("vocab_size")
    if contract_vocab is None and token_shape and token_shape[0] == EXPECTED_TOKENIZER_VOCAB_SIZE:
        warnings.append(
            "legacy checkpoint has no tokenizer_contract.vocab_size; tensor shape "
            "is compatible and runtime tokenizer probes are required"
        )
    elif int(contract_vocab or 0) != checkpoint_vocab:
        errors.append(f"tokenizer_contract.vocab_size={contract_vocab} expected={checkpoint_vocab}")
    expected_config = {
        "vocab_size": checkpoint_vocab,
        "n_embd": V2_FRONTIER.n_embd,
        "n_layer": V2_FRONTIER.n_layer,
        "n_head": V2_FRONTIER.n_head,
        "n_kv_head": V2_FRONTIER.n_kv_head,
        "block_size": V2_FRONTIER.block_size,
    }
    if isinstance(model_config, dict):
        for key, expected in expected_config.items():
            actual = model_config.get(key)
            if int(actual or 0) != int(expected):
                errors.append(f"model_config.{key}={actual} expected={expected}")
    else:
        errors.append("model_config is missing or not a dict")
    if checkpoint_vocab not in {EXPECTED_TOKENIZER_VOCAB_SIZE, TOKENIZER_V4_VOCAB_SIZE}:
        errors.append(f"unsupported append-only vocabulary size: {checkpoint_vocab}")
    if token_shape is None or token_shape[0] != checkpoint_vocab:
        errors.append(f"token embedding shape mismatch: {token_shape}")
    if lm_shape is None or lm_shape[0] != checkpoint_vocab:
        errors.append(f"lm_head shape mismatch: {lm_shape}")
    if int(blob.get("global_step", blob.get("step", 0)) or 0) <= 0:
        errors.append("global_step is not positive")
    if not tokenizer_contract.get("vocabulary_sha256"):
        warnings.append("legacy checkpoint has no vocabulary hash; runtime probe gate is required")
    if not tokenizer_contract.get("probe_sha256"):
        warnings.append("legacy checkpoint has no 500-probe tokenizer fingerprint")

    report["ok"] = not errors
    report["errors"] = errors
    report["warnings"] = warnings
    report["migration_required"] = checkpoint_schema < CHECKPOINT_SCHEMA_VERSION
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an iterate500 frontier checkpoint.")
    parser.add_argument("--checkpoint", default=str(FRONTIER_CHECKPOINT))
    parser.add_argument(
        "--json-out",
        default=str(OUTPUT_V2_DIR / "frontier_checkpoint_proof.json"),
        help="Where to write the proof report.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    report = _checkpoint_report(checkpoint)
    output = Path(args.json_out)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 2)


if __name__ == "__main__":
    main()
