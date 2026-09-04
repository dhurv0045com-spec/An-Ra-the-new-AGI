"""Content-addressed checkpoint save/load for the TPU path (host-side I/O).

Format reuses Cymek's component contract (model/optimizer/scheduler/rng/
cursor/ledger/training-state) where present, but this module stands alone so
the Kaggle notebook has one obvious load command. Hashes are computed on CPU
from detached tensors (audit A12: never per-step inside the compiled region).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def state_dict_sha256(state: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode())
        t = state[name]
        try:
            h.update(str(tuple(t.shape)).encode("ascii"))
            h.update(bytes(t.detach().to("cpu").float().contiguous().numpy().tobytes()))
        except Exception:
            h.update(repr(t).encode())
    return h.hexdigest()


def save(model, path: str, meta: dict[str, Any] | None = None) -> str:
    """Save model state + meta to path. Returns file SHA-256."""
    import torch

    payload = {
        "model_state": {k: v.detach().to("cpu") for k, v in model.state_dict().items()},
        "meta": dict(meta or {}),
    }
    payload["meta"]["param_sha256"] = state_dict_sha256(payload["model_state"])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(p))
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_into(model, path: str) -> dict[str, Any]:
    """Load checkpoint file into model (strict). Returns stored meta."""
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    return dict(payload.get("meta", {}))


def load_command(checkpoint: str) -> str:
    """One obvious reuse command printed into every training receipt."""
    return (
        "from v5_model.core import initialize\n"
        "from anra_v5.miniature_run import MINI_SPEC\n"
        "from citadel_tpu import checkpoint as ckpt\n"
        f"model = initialize(MINI_SPEC, <seed>)\n"
        f"meta = ckpt.load_into(model, {checkpoint!r})\n"
    )


__all__ = ["load_command", "load_into", "save", "state_dict_sha256"]
