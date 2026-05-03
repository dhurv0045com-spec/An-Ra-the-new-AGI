"""Startup validation for runtime and training-contract invariants."""
from __future__ import annotations

import torch

from anra_paths import V3_TOKENIZER_FILE
from tokenizer.subword_tokenizer import SubwordTokenizer
from training.v2_config import EXPECTED_PAD_TOKEN_ID, EXPECTED_SPECIAL_TOKENS, EXPECTED_TOKENIZER_VOCAB_SIZE

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
