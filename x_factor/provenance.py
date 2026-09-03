"""Provenance utilities: permanent fix for parameter_sha256 = null regression.

Every Triquetra receipt must bind:
  - checkpoint file SHA256 (streaming, works for multi-GB files)
  - canonical ordered-parameter SHA256 (computed from tensors, not metadata)
  - model config SHA256 (canonical JSON of model_config)
  - tokenizer identity (canonical V4 identity dict)
  - runtime source SHA / commit
  - generator / experiment / analysis source SHAs
  - condition-registry SHA

No receipt may contain parameter_sha256 = null.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def param_sha256_from_state_dict(state_dict) -> str:
    """Canonical ordered-parameter SHA, same construction as causal_decomposition.

    Format per tensor: name + NUL + shape tuple + NUL + dtype str + NUL, then raw bytes.
    Tensors are visited in sorted(name) order, moved to CPU, made contiguous.
    """
    import torch

    h = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        t = state_dict[name].detach().cpu().contiguous()
        h.update(f"{name}\0{tuple(t.shape)}\0{t.dtype}\0".encode())
        # view as bytes without copy where possible
        try:
            h.update(t.view(torch.uint8).reshape(-1).numpy().tobytes())
        except Exception:
            import numpy as np

            h.update(np.ascontiguousarray(t.numpy()).tobytes())
    return h.hexdigest()


def file_sha(path: str | Path) -> str:
    return sha256_file(path)


def git_head(repo: str | Path = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=15,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def git_status_short(repo: str | Path = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=15,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"
