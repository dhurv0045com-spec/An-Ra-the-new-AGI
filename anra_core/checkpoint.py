from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import torch

from .config import CANONICAL_CONFIG, CoreConfig
from .model import AnRaCore

_IGNORED_PREFIXES = (
    "esv_module.", "rim_modules.", "mod_routers.", "residual_depth_logits",
    "dstp_temperature_log", "layer_temperature_bias_log",
)


def _unwrap(payload: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a mapping")
    for key in ("model_state_dict", "model"):
        if isinstance(payload.get(key), dict):
            return dict(payload[key]), payload
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return dict(payload), {}
    raise ValueError("checkpoint has no model_state_dict, model, or raw tensor state")


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
            raise ValueError(f"serialized weight alias drift: {alias} != {target}")
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
            raise ValueError(f"checkpoint architecture drift: {field}={present!r}")


def load_core_checkpoint(
    checkpoint: str | Path,
    *,
    config: CoreConfig = CANONICAL_CONFIG,
) -> tuple[AnRaCore, dict[str, Any]]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state, metadata = _unwrap(payload)
    state = _normalize_keys(state)
    accepted_aliases = _validate_serialized_aliases(state)
    _validate_config(metadata, config)

    model = AnRaCore(config)
    expected = model.state_dict()
    missing = sorted(key for key in expected if key not in state and key != "lm_head.weight")
    if missing:
        raise ValueError(f"checkpoint is missing {len(missing)} dense tensors; first={missing[0]}")
    selected: dict[str, torch.Tensor] = {}
    for key, expected_tensor in expected.items():
        source_key = "token_embedding_table.weight" if key == "lm_head.weight" else key
        tensor = state[source_key]
        if tuple(tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"tensor shape mismatch for {source_key}: {tuple(tensor.shape)} != "
                f"{tuple(expected_tensor.shape)}"
            )
        selected[key] = tensor
    unknown = sorted(
        key for key in state
        if key not in expected
        and key not in accepted_aliases
        and not key.startswith(_IGNORED_PREFIXES)
    )
    if unknown:
        raise ValueError(f"unknown checkpoint tensor outside dense core: {unknown[0]}")
    model.load_state_dict(selected, strict=True)
    model.lm_head.weight = model.token_embedding_table.weight
    model.eval()
    return model, metadata
