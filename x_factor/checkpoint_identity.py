"""Strict checkpoint identity contract (Mission 11/12).

NO SILENT FALLBACK: if the requested checkpoint path does not exist, every
serious entry point must FAIL with REQUESTED CHECKPOINT NOT FOUND instead of
substituting another file. A silent fallback can invalidate an experiment.

Every serious run binds: path, file SHA256, canonical parameter SHA256,
config SHA, tokenizer SHA, step/stage/lineage metadata, source commit,
experiment SHA, protocol SHA. Any identity change -> new experiment identity.
"""

from __future__ import annotations

from pathlib import Path

import sys as _sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
_RT = _HERE / "_runtime"
if str(_RT) not in _sys.path:
    _sys.path.insert(0, str(_RT))


class CheckpointNotFound(FileNotFoundError):
    pass


class UnsupportedArchitecture(ValueError):
    """Config parses but matches no known An-Ra architecture profile.

    Distinct from BAD_CHECKPOINT (corrupt/unreadable file): the file may be
    perfectly valid, just not a Core this Triquetra instrument knows how to
    load. Do NOT silently coerce it into a V4 loader.
    """


# Minimum clean extension point (Mission 22). New architectures add one entry
# plus a loader branch in qualify_checkpoint; nothing else changes.
ARCH_PROFILES = {
    "anra_v4_rope_interleaved_v1": {
        "label": "V4_180M",
        "expect": {"vocab_size": 32768, "n_layers": 18, "d_model": 896},
        "loader": "anra_core.model:AnRaCore",
        "tokenizer": "anra_core.tokenizer:V4Tokenizer.load_canonical",
    },
}


def match_architecture_profile(model_config: dict) -> dict:
    """Return profile + mismatches, or raise UnsupportedArchitecture."""
    ver = model_config.get("architecture_version")
    prof = ARCH_PROFILES.get(ver or "")
    if prof is None:
        raise UnsupportedArchitecture(
            f"UNSUPPORTED_ARCHITECTURE: version={ver!r} matches no known profile "
            f"{sorted(ARCH_PROFILES)}. This is not a BAD_CHECKPOINT verdict.")
    mism = {k: (model_config.get(k), v) for k, v in prof["expect"].items()
            if model_config.get(k) != v}
    if mism:
        raise UnsupportedArchitecture(
            f"UNSUPPORTED_ARCHITECTURE: profile {prof['label']} mismatch: {mism}.")
    return {"profile": prof["label"], "version": ver}


def resolve_checkpoint(path: str | Path) -> Path:
    """Return the path or raise. Never substitute another checkpoint."""
    p = Path(path)
    if not p.exists():
        raise CheckpointNotFound(
            f"REQUESTED CHECKPOINT NOT FOUND: {path}. "
            "Refusing to fall back to another file."
        )
    return p


def identify_checkpoint(path: str | Path, _cache: dict | None = None) -> dict:
    """Bind full identity. Param SHA is computed from tensor bytes, never metadata."""
    from provenance import param_sha256_from_state_dict, sha256_file, sha256_json

    p = resolve_checkpoint(path)
    import torch

    payload = torch.load(str(p), map_location="cpu", weights_only=False)
    from anra_core.config import CoreConfig, CANONICAL_CONFIG

    cfg = CoreConfig(**{k: payload["model_config"][k]
                        for k in CANONICAL_CONFIG.__dataclass_fields__})
    from anra_core.model import AnRaCore

    model = AnRaCore(cfg)
    model.load_state_dict({k: v for k, v in payload["model_state_dict"].items()
                           if k != "lm_head.weight"}, strict=False)
    model.lm_head.weight = model.token_embedding_table.weight
    ident = {
        "path": str(p),
        "checkpoint_sha256": sha256_file(str(p)),
        "parameter_sha256": param_sha256_from_state_dict(model.state_dict()),
        "config_sha256": sha256_json(payload["model_config"]),
        "global_step": payload.get("global_step"),
        "training_stage": payload.get("training_stage"),
        "source_commit": payload.get("source_commit"),
        "source_checkpoint": payload.get("source_checkpoint"),
    }
    del model
    import gc
    gc.collect()
    return ident


def load_core(path: str | Path, device: str):
    """Strict load: resolve (no fallback) -> build -> eval model + tokenizer."""
    import torch

    p = resolve_checkpoint(path)
    from anra_core.config import CoreConfig, CANONICAL_CONFIG
    from anra_core.model import AnRaCore
    from anra_core.tokenizer import V4Tokenizer

    payload = torch.load(str(p), map_location="cpu", weights_only=False)
    cfg = CoreConfig(**{k: payload["model_config"][k]
                        for k in CANONICAL_CONFIG.__dataclass_fields__})
    model = AnRaCore(cfg)
    model.load_state_dict({k: v for k, v in payload["model_state_dict"].items()
                           if k != "lm_head.weight"}, strict=False)
    model.lm_head.weight = model.token_embedding_table.weight
    return model.to(device).eval(), V4Tokenizer.load_canonical(), payload
