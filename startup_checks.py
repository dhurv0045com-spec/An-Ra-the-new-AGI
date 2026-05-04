"""Startup validation for runtime and training-contract invariants."""
from __future__ import annotations

import warnings

import torch

from anra_paths import V3_TOKENIZER_FILE
from tokenizer.subword_tokenizer import SubwordTokenizer
from training.v2_config import EXPECTED_PAD_TOKEN_ID, EXPECTED_SPECIAL_TOKENS, EXPECTED_TOKENIZER_VOCAB_SIZE

_REMEDIATION = (
    "Remediation: run on a CUDA-capable GPU build of PyTorch, install matching NVIDIA drivers/CUDA toolkit, "
    "and ensure PyTorch was built with FlashAttention SDP support."
)


def assert_flash_sdp_ready(entrypoint: str) -> None:
    """Warn when Flash SDP acceleration is unavailable."""
    if not torch.cuda.is_available():
        warnings.warn(
            f"[{entrypoint}] Flash SDP unavailable: CUDA is unavailable. Training will continue without "
            f"FlashAttention acceleration. {_REMEDIATION}",
            UserWarning,
            stacklevel=2,
        )
        return

    if not torch.backends.cuda.is_flash_attention_available():
        warnings.warn(
            f"[{entrypoint}] Flash SDP unavailable: torch.backends.cuda.is_flash_attention_available() is False. "
            f"Training will continue without FlashAttention acceleration. {_REMEDIATION}",
            UserWarning,
            stacklevel=2,
        )
        return

    if not torch.backends.cuda.flash_sdp_enabled():
        warnings.warn(
            f"[{entrypoint}] Flash SDP unavailable: torch.backends.cuda.flash_sdp_enabled() is False. "
            "Training will continue without FlashAttention acceleration. "
            "Set this at startup via torch.backends.cuda.enable_flash_sdp(True). "
            f"{_REMEDIATION}",
            UserWarning,
            stacklevel=2,
        )


def assert_v2_training_contract(entrypoint: str, tokenizer_path=V3_TOKENIZER_FILE) -> None:
    """Abort startup when tokenizer IDs cannot safely train or load V2 checkpoints."""
    path = tokenizer_path
    if not path.exists():
        raise RuntimeError(f"[{entrypoint}] required tokenizer missing: {path}")
    tokenizer = SubwordTokenizer.load(path)
    if tokenizer.vocab_size != EXPECTED_TOKENIZER_VOCAB_SIZE:
        raise RuntimeError(
            f"[{entrypoint}] tokenizer vocab_size={tokenizer.vocab_size}, "
            f"expected={EXPECTED_TOKENIZER_VOCAB_SIZE}"
        )
    if tokenizer.pad_token_id != EXPECTED_PAD_TOKEN_ID:
        raise RuntimeError(
            f"[{entrypoint}] tokenizer pad_token_id={tokenizer.pad_token_id}, "
            f"expected={EXPECTED_PAD_TOKEN_ID}"
        )
    if tokenizer.special_tokens[: len(EXPECTED_SPECIAL_TOKENS)] != EXPECTED_SPECIAL_TOKENS:
        raise RuntimeError(
            f"[{entrypoint}] tokenizer special tokens={tokenizer.special_tokens}, "
            f"expected prefix={EXPECTED_SPECIAL_TOKENS}"
        )
