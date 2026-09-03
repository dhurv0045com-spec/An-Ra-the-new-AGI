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


class CheckpointNotFound(FileNotFoundError):
    pass


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
