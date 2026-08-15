# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from anra_brain import ANRA_V4_ARCHITECTURE_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__" and not __package__:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.check_frontier_checkpoint", *sys.argv[1:]],
        cwd=REPO_ROOT,
        check=False,
    )
    raise SystemExit(completed.returncode)

from anra.anra_paths import ANRA_V4_CHECKPOINT, OUTPUT_V2_DIR, ROOT
from runtime.safe_load import safe_torch_load
from training.v2_config import (
    ANRA_V4_MODEL,
    CHECKPOINT_SCHEMA_VERSION,
    EXPECTED_TOKENIZER_VOCAB_SIZE,
    model_parameter_count,
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
    expected_tokenizer_schema = 4

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
        "expected_model_params": model_parameter_count(ANRA_V4_MODEL, checkpoint_vocab),
    }

    errors: list[str] = []
    warnings: list[str] = []
    checkpoint_schema = int(blob.get("checkpoint_schema_version", 0) or 0)
    if checkpoint_schema != CHECKPOINT_SCHEMA_VERSION:
        errors.append(
            f"canonical V4 requires checkpoint_schema_version={CHECKPOINT_SCHEMA_VERSION}; "
            f"got {checkpoint_schema}"
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
        "architecture_version": ANRA_V4_ARCHITECTURE_VERSION,
        "vocab_size": checkpoint_vocab,
        "pad_token_id": ANRA_V4_MODEL.pad_token_id,
        "n_embd": ANRA_V4_MODEL.n_embd,
        "n_layer": ANRA_V4_MODEL.n_layer,
        "n_head": ANRA_V4_MODEL.n_head,
        "n_kv_head": ANRA_V4_MODEL.n_kv_head,
        "d_ff": ANRA_V4_MODEL.d_ff,
        "block_size": ANRA_V4_MODEL.block_size,
        "rope_base": ANRA_V4_MODEL.rope_base,
        "rms_norm_eps": ANRA_V4_MODEL.rms_norm_eps,
        "dropout": ANRA_V4_MODEL.dropout,
        "mod_layers": ANRA_V4_MODEL.mod_layers,
        "base_seq_len": ANRA_V4_MODEL.base_seq_len,
        "target_seq_len": ANRA_V4_MODEL.target_seq_len,
        "use_qk_norm": ANRA_V4_MODEL.use_qk_norm,
        "sliding_window": ANRA_V4_MODEL.sliding_window,
        "full_attention_every": ANRA_V4_MODEL.full_attention_every,
        "use_mtp": False,
        "use_moe": False,
        "initialization_scheme": "depth_scaled_residual_v1",
    }
    if isinstance(model_config, dict):
        for key, expected in expected_config.items():
            actual = model_config.get(key)
            if actual != expected:
                errors.append(f"model_config.{key}={actual} expected={expected}")
    else:
        errors.append("model_config is missing or not a dict")
    if checkpoint_vocab != EXPECTED_TOKENIZER_VOCAB_SIZE:
        errors.append(f"canonical V4 vocabulary must be 32768, got {checkpoint_vocab}")
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
    manifest_hashes = blob.get("data_manifests", blob.get("dataset_manifest_hashes", {}))
    manifest_payloads = blob.get("data_manifest_payloads", {})
    embedded_manifests_valid = bool(manifest_hashes)
    if isinstance(manifest_hashes, dict) and isinstance(manifest_payloads, dict):
        for name, expected_digest in manifest_hashes.items():
            payload = manifest_payloads.get(name)
            if not isinstance(payload, (bytes, bytearray)) or hashlib.sha256(
                bytes(payload)
            ).hexdigest() != str(expected_digest):
                embedded_manifests_valid = False
                break
    else:
        embedded_manifests_valid = False
    report["embedded_data_manifests"] = {
        "count": len(manifest_hashes) if isinstance(manifest_hashes, dict) else 0,
        "complete": embedded_manifests_valid,
    }
    if checkpoint_schema >= 6 and not embedded_manifests_valid:
        errors.append("schema v6 checkpoint is missing complete hash-verified data manifests")
    elif not embedded_manifests_valid:
        warnings.append("legacy checkpoint is not self-contained for corpus manifest restoration")

    report["ok"] = not errors
    report["errors"] = errors
    report["warnings"] = warnings
    report["migration_required"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the canonical An-Ra V4 checkpoint.")
    parser.add_argument("--checkpoint", default=str(ANRA_V4_CHECKPOINT))
    parser.add_argument(
        "--json-out",
        default=str(OUTPUT_V2_DIR / "v4_checkpoint_proof.json"),
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
