from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import torch

from .config import CANONICAL_CONFIG, CoreConfig
from .contracts import CheckpointIdentity
from .errors import CheckpointIncompatibleError, RepresentationIncompatibleError
from .model import AnRaCore
from .tokenizer import V4Tokenizer

_LEGACY_MOD_LAYERS = frozenset({4, 6, 8, 10, 12, 14, 16})
_HASH_CHUNK_BYTES = 4 * 1024 * 1024


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _parameter_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash the normalized dense tensor contract independently of its container file."""
    hasher = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        header = f"{name}\0{tuple(tensor.shape)}\0{tensor.dtype}\0".encode()
        hasher.update(header)
        raw = tensor.view(torch.uint8).reshape(-1)
        for start in range(0, raw.numel(), _HASH_CHUNK_BYTES):
            chunk = raw[start : start + _HASH_CHUNK_BYTES]
            hasher.update(chunk.numpy().tobytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _historical_dormant_shape(key: str, config: CoreConfig) -> tuple[int, ...] | None:
    scalar_controls = {
        "residual_depth_logits": (config.n_layers,),
        "dstp_temperature_log": (config.n_layers,),
        "layer_temperature_bias_log": (config.n_layers,),
        "esv_module.predictor.0.weight": (3, min(64, config.d_model)),
        "esv_module.predictor.0.bias": (3,),
        "esv_module.state": (3,),
    }
    if key in scalar_controls:
        return scalar_controls[key]

    mod_match = re.fullmatch(r"mod_routers\.(\d+)\.(gate\.weight|capacity_control|context_weights)", key)
    if mod_match:
        layer, member = int(mod_match.group(1)), mod_match.group(2)
        if layer not in _LEGACY_MOD_LAYERS or layer >= config.n_layers:
            return None
        return {
            "gate.weight": (1, config.d_model),
            "capacity_control": (),
            "context_weights": (3,),
        }[member]

    rim_match = re.fullmatch(
        r"rim_modules\.(\d+)\."
        r"(raw_alpha|projection\.parametrizations\.weight\.original|"
        r"projection\.parametrizations\.weight\.0\._u|"
        r"projection\.parametrizations\.weight\.0\._v)",
        key,
    )
    if rim_match:
        layer, member = int(rim_match.group(1)), rim_match.group(2)
        if layer >= config.n_layers:
            return None
        esv_width = min(64, config.d_model)
        return {
            "raw_alpha": (),
            "projection.parametrizations.weight.original": (config.d_model, esv_width),
            "projection.parametrizations.weight.0._u": (config.d_model,),
            "projection.parametrizations.weight.0._v": (esv_width,),
        }[member]
    return None


def _validate_historical_dormant_tensors(
    state: dict[str, torch.Tensor], config: CoreConfig
) -> set[str]:
    """Accept only the exact dormant V4 pilot tensors known to the historical ABI."""
    accepted: set[str] = set()
    for key, tensor in state.items():
        expected_shape = _historical_dormant_shape(key, config)
        if expected_shape is None:
            continue
        if tuple(tensor.shape) != expected_shape:
            raise CheckpointIncompatibleError(
                f"historical dormant tensor shape mismatch for {key}",
                details={
                    "tensor": key,
                    "expected_shape": expected_shape,
                    "got_shape": tuple(tensor.shape),
                },
            )
        accepted.add(key)
    return accepted


def _verify_tokenizer_contract(
    metadata: dict[str, Any], *, legacy_unverified: bool
) -> tuple[bool, bool]:
    contract = metadata.get("tokenizer_contract")
    present = contract is not None
    if not present:
        if legacy_unverified:
            return False, False
        raise RepresentationIncompatibleError(
            "checkpoint is missing its tokenizer contract; use legacy_unverified=True only for explicit forensic loading"
        )
    try:
        V4Tokenizer.load_canonical().assert_checkpoint_contract(contract)
    except RepresentationIncompatibleError:
        # Old contract formats (e.g. no "available" flag) can fail format
        # checks while the vocabulary is in fact identical. Legacy forensic
        # mode proceeds unverified; strict mode still fails closed.
        if legacy_unverified:
            return True, False
        raise
    return True, True


def _verify_artifact_contract(
    metadata: dict[str, Any], *, legacy_unverified: bool
) -> tuple[str | None, int | None]:
    """Require explicit artifact type/version outside forensic compatibility mode."""
    artifact_class = metadata.get("checkpoint_artifact_class")
    schema_version = metadata.get("checkpoint_schema_version")
    if artifact_class in {"full_resume", "model_only"} and isinstance(schema_version, int):
        if schema_version > 0:
            return str(artifact_class), schema_version
    if legacy_unverified:
        return None, None
    raise CheckpointIncompatibleError(
        "checkpoint is missing a supported artifact class/schema version",
        details={"artifact_class": artifact_class, "schema_version": schema_version},
    )


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
    legacy_unverified: bool = False,
) -> tuple[AnRaCore, dict[str, Any], CheckpointIdentity]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    file_hash = _file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state, metadata = _unwrap(payload)
    state = _normalize_keys(state)
    accepted_aliases = _validate_serialized_aliases(state)
    dormant_tensors = _validate_historical_dormant_tensors(state, config)
    _validate_config(metadata, config)
    artifact_class, artifact_schema_version = _verify_artifact_contract(
        metadata, legacy_unverified=legacy_unverified
    )

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
        and key not in dormant_tensors
    )
    if unknown:
        raise CheckpointIncompatibleError(
            f"unknown checkpoint tensor outside dense core: {unknown[0]}",
            details={"unknown_count": len(unknown), "unknown_tensors": unknown},
        )
    contract_present, contract_verified = _verify_tokenizer_contract(
        metadata, legacy_unverified=legacy_unverified
    )
    parameter_hash = _parameter_sha256(selected)

    model.load_state_dict(selected, strict=True)
    model.lm_head.weight = model.token_embedding_table.weight
    model.eval()

    step = metadata.get("global_step", metadata.get("step"))
    stage = metadata.get("training_stage", metadata.get("stage"))
    commit = metadata.get("source_commit", metadata.get("commit"))
    identity = CheckpointIdentity(
        checkpoint_sha256=file_hash,
        source_path=str(path),
        global_step=int(step) if step is not None else None,
        training_stage=str(stage) if stage is not None else None,
        source_commit=str(commit) if commit is not None else None,
        tokenizer_contract_valid=contract_verified,
        parameter_sha256=parameter_hash,
        tokenizer_contract_present=contract_present,
        tokenizer_contract_verified=contract_verified,
        ignored_tensor_names=tuple(sorted(dormant_tensors)),
        legacy_unverified=legacy_unverified and not contract_verified,
        artifact_class=artifact_class,
        artifact_schema_version=artifact_schema_version,
    )

    return model, metadata, identity


# Canonical parameter identity: one implementation for trainer, loader,
# evaluator, lineage, and promotion. Tied weights hash once per stored name.
parameter_sha256 = _parameter_sha256
