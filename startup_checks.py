"""Startup validation for CUDA Flash SDP backend requirements."""
from __future__ import annotations

import torch

_REMEDIATION = (
    "Remediation: run on a CUDA-capable GPU build of PyTorch, install matching NVIDIA drivers/CUDA toolkit, "
    "and ensure PyTorch was built with FlashAttention SDP support."
)


def assert_flash_sdp_ready(entrypoint: str) -> None:
    """Abort startup unless Flash SDP is enabled and CUDA backend is available."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"[{entrypoint}] CUDA is unavailable. { _REMEDIATION }"
        )

    if not torch.backends.cuda.is_flash_attention_available():
        raise RuntimeError(
            f"[{entrypoint}] torch.backends.cuda.is_flash_attention_available() is False. {_REMEDIATION}"
        )

    if not torch.backends.cuda.flash_sdp_enabled():
        raise RuntimeError(
            f"[{entrypoint}] torch.backends.cuda.flash_sdp_enabled() is False. "
            "Set this at startup via torch.backends.cuda.enable_flash_sdp(True). "
            f"{_REMEDIATION}"
        )
