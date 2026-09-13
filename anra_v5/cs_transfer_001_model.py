"""Matched physical-vocabulary model construction for CS-TRANSFER-001.

A naïve same-seed initialization of differently sized embedding tables would
shift the RNG stream and therefore change every later block tensor.  That is a
fatal confound for a physical-class-space experiment.  This module constructs
one V=24,576 reference, then derives the V=4,096 arm by copying every shared
parameter byte-for-byte, including embedding rows 0..4095.  The only trainable
state that differs at update zero is the existence of rows 4096..24575 in the
large tied embedding/output matrix.
"""
from __future__ import annotations

import hashlib
from typing import Any

from v5_contracts.model_spec import ModelSpec
from v5_model.core import initialize as build_core

COMMON_VOCAB = 4096
FULL_VOCAB = 24576


def spec_for(vocabulary_size: int) -> ModelSpec:
    if vocabulary_size not in (COMMON_VOCAB, FULL_VOCAB):
        raise ValueError("CS-TRANSFER-001 permits only physical V4096 or V24576")
    return ModelSpec(
        schema="anra-v5-model-spec/v1",
        family="dense-decoder-transformer",
        vocabulary_size=vocabulary_size,
        width=256,
        layers=8,
        query_heads=4,
        kv_heads=2,
        head_dimension=64,
        ffn_width=1024,
        context_length=1024,
        rope_base=10_000.0,
        norm_epsilon=1e-5,
        tied_embeddings=True,
        qk_norm=True,
        qk_norm_affine=True,
        linear_bias=False,
        dropout=0.0,
    )


def _tensor_sha(tensor, torch) -> str:
    payload = tensor.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def build_matched_pair(*, seed: int, torch_module: Any = None):
    """Return `(small, full, receipt)` with exact shared initialization."""
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    full = build_core(spec_for(FULL_VOCAB), seed=seed, torch_module=torch)
    small = build_core(spec_for(COMMON_VOCAB), seed=seed, torch_module=torch)

    full_params = dict(full.named_parameters())
    small_params = dict(small.named_parameters())
    if set(full_params) != set(small_params):
        raise ValueError("parameter names differ across physical-vocabulary arms")

    with torch.no_grad():
        for name, small_p in small_params.items():
            full_p = full_params[name]
            if name.endswith("embedding.weight"):
                if tuple(small_p.shape) != (COMMON_VOCAB, 256):
                    raise ValueError("unexpected small embedding shape")
                if tuple(full_p.shape) != (FULL_VOCAB, 256):
                    raise ValueError("unexpected full embedding shape")
                small_p.copy_(full_p[:COMMON_VOCAB])
            else:
                if tuple(small_p.shape) != tuple(full_p.shape):
                    raise ValueError(f"shared tensor shape differs: {name}")
                small_p.copy_(full_p)

    shared_hashes: dict[str, str] = {}
    for name, small_p in small.named_parameters():
        full_p = dict(full.named_parameters())[name]
        if name.endswith("embedding.weight"):
            left = _tensor_sha(small_p, torch)
            right = _tensor_sha(full_p[:COMMON_VOCAB], torch)
        else:
            left = _tensor_sha(small_p, torch)
            right = _tensor_sha(full_p, torch)
        if left != right:
            raise ValueError(f"shared initialization mismatch: {name}")
        shared_hashes[name] = left

    receipt = {
        "schema": "anra-cs-transfer-001-matched-init/v1",
        "seed": int(seed),
        "small_model_spec_sha256": small.spec.sha256(),
        "full_model_spec_sha256": full.spec.sha256(),
        "small_parameters": sum(int(p.numel()) for p in small.parameters()),
        "full_parameters": sum(int(p.numel()) for p in full.parameters()),
        "shared_embedding_rows": COMMON_VOCAB,
        "full_extra_embedding_rows": FULL_VOCAB - COMMON_VOCAB,
        "shared_parameter_hashes": shared_hashes,
        "shared_initialization_exact": True,
    }
    return small, full, receipt


def select_arm(pair, arm: str):
    small, full, receipt = pair
    if arm == "PHYS_4096":
        return small, receipt
    if arm == "PHYS_24576":
        return full, receipt
    raise ValueError(f"unknown arm {arm}")


__all__ = [
    "COMMON_VOCAB", "FULL_VOCAB", "build_matched_pair", "select_arm", "spec_for",
]
