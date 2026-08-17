from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import torch

from .config import CANONICAL_CONFIG, CoreConfig
from .contracts import CheckpointIdentity
from .errors import CheckpointIncompatibleError
from .model import AnRaCore

_IGNORED_PREFIXES = (
    "esv_module.",
    "rim_modules.",
    "mod_routers.",
    "residual_depth_logits",
    "dstp_temperature_log",
    "layer_temperature_bias_log",
)


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _unwrap(payload: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not isinstance(payload, dict):
        raise CheckpointIncompatibleError(
            "checkpoint payload is not a mapping",
            details={"type": type(payload).__name__},
        )
    for key in ("model_state_dict", "model"):
        if isinstance(payload.get(key), dict):
            return dict(payload[key]), payload
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return dict(payload), {}
    raise CheckpointIncompatibleError(
        "checkpoint has no model_state_dict, model, or raw tensor state",
        details={"keys": list(payload.keys()) if isinstance(payload, dict) else []},
    )


def _normalize_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    if "token_embedding_table.weight" not in state and "token_embedding.weight" in state:
        state["token_embedding_table.weight"] = state["token_embedding.weight"]
    return state


def _validate_serialized_aliases(state: dict[str, torch.Tensor]) -> set[str]:
    """Prove historical module aliases carry the same dense tensors they name."""
    aliases: dict[str, str] = {
        "token_embedding.weight": "token_embedding_table.weight",
        "lm_head.weight": "token_embedding_table.weight",
    }
    pattern = re.compile(r"^blocks\.(\d+)\._normed_mlp\.(0|1)(\..+)$")
    for key in state:
        match = pattern.fullmatch(key)
        if not match:
            continue
        layer, member, suffix = match.groups()
        target = (
            f"blocks.{layer}.norm_2{suffix}"
            if member == "0"
            else f"blocks.{layer}.mlp{suffix}"
        )
        aliases[key] = target
    accepted: set[str] = set()
    for alias, target in aliases.items():
        if alias not in state:
            continue
        if target not in state or not torch.equal(state[alias], state[target]):
            raise CheckpointIncompatibleError(
                f"serialized weight alias drift: {alias} != {target}",
                details={"alias": alias, "target": target},
            )
        accepted.add(alias)
    return accepted


def _validate_config(payload: dict[str, Any], config: CoreConfig) -> None:
    saved = payload.get("model_config") or payload.get("config")
    if not isinstance(saved, dict):
        return
    aliases = {
        "architecture_version": ("architecture_version",),
        "vocab_size": ("vocab_size",),
        "pad_token_id": ("pad_token_id",),
        "d_model": ("d_model", "n_embd", "width"),
        "n_layers": ("n_layers", "n_layer", "layers"),
        "n_heads": ("n_heads", "n_head", "query_heads"),
        "n_kv_heads": ("n_kv_heads", "n_kv_head", "kv_heads"),
        "head_dim": ("head_dim",),
        "d_ff": ("d_ff", "ffn_width"),
        "block_size": ("block_size", "context_length"),
        "rms_norm_eps": ("rms_norm_eps",),
        "dropout": ("dropout",),
        "rope_base": ("rope_base",),
        "base_seq_len": ("base_seq_len",),
        "target_seq_len": ("target_seq_len",),
        "qk_norm": ("use_qk_norm", "qk_norm"),
        "sliding_window": ("sliding_window",),
        "full_attention_every": ("full_attention_every",),
        "use_mtp": ("use_mtp",),
        "use_moe": ("use_moe",),
        "initialization_scheme": ("initialization_scheme",),
    }
    for field, keys in aliases.items():
        present = next((saved[key] for key in keys if key in saved), None)
        if present is not None and present != getattr(config, field):
            raise CheckpointIncompatibleError(
                f"checkpoint architecture drift: {field}={present!r}",
                details={"field": field, "expected": getattr(config, field), "got": present},
            )


def load_core_checkpoint(
    checkpoint: str | Path,
    *,
    config: CoreConfig = CANONICAL_CONFIG,
) -> tuple[AnRaCore, dict[str, Any], CheckpointIdentity]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    file_hash = _file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state, metadata = _unwrap(payload)
    state = _normalize_keys(state)
    accepted_aliases = _validate_serialized_aliases(state)
    _validate_config(metadata, config)

    model = AnRaCore(config)
    expected = model.state_dict()
    missing = sorted(key for key in expected if key not in state and key != "lm_head.weight")
    if missing:
        raise CheckpointIncompatibleError(
            f"checkpoint is missing {len(missing)} dense tensors; first={missing[0]}",
            details={"missing_count": len(missing), "first_missing": missing[0], "missing": missing},
        )
    selected: dict[str, torch.Tensor] = {}
    for key, expected_tensor in expected.items():
        source_key = "token_embedding_table.weight" if key == "lm_head.weight" else key
        tensor = state[source_key]
        if tuple(tensor.shape) != tuple(expected_tensor.shape):
            raise CheckpointIncompatibleError(
                f"tensor shape mismatch for {source_key}: {tuple(tensor.shape)} != {tuple(expected_tensor.shape)}",
                details={"tensor": source_key, "expected_shape": tuple(expected_tensor.shape), "got_shape": tuple(tensor.shape)},
            )
        selected[key] = tensor
    unknown = sorted(
        key for key in state
        if key not in expected
        and key not in accepted_aliases
        and not key.startswith(_IGNORED_PREFIXES)
    )
    if unknown:
        raise CheckpointIncompatibleError(
            f"unknown checkpoint tensor outside dense core: {unknown[0]}",
            details={"unknown_count": len(unknown), "unknown_tensors": unknown},
        )
    model.load_state_dict(selected, strict=True)
    model.lm_head.weight = model.token_embedding_table.weight
    model.eval()

    step = metadata.get("global_step", metadata.get("step"))
    stage = metadata.get("training_stage", metadata.get("stage"))
    commit = metadata.get("source_commit", metadata.get("commit"))
    has_valid_contract = bool(metadata.get("tokenizer_contract"))

    identity = CheckpointIdentity(
        checkpoint_sha256=file_hash,
        source_path=str(path),
        global_step=int(step) if step is not None else None,
        training_stage=str(stage) if stage is not None else None,
        source_commit=str(commit) if commit is not None else None,
        tokenizer_contract_valid=has_valid_contract,
    )

    return model, metadata, identity
