"""HORM-002 regression: runtime patch applies bounded effect and restores cleanly."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch

from v5_contracts.model_spec import ModelSpec
from v5_identity import HORMONES, HormonalProjection
from v5_identity.attention_patch import HormonalAttentionPatch
from v5_model.core import initialize, packed_layout


def _projection(alpha: float) -> HormonalProjection:
    return HormonalProjection(
        weights=tuple(0.1 if n in ("cortisol", "adrenaline") else 0.05 for n in HORMONES),
        bound=0.2,
        raw_alpha=alpha,
    )


def _spec() -> ModelSpec:
    return ModelSpec(
        schema="anra-v5-model-spec/v1", family="dense-decoder-transformer",
        vocabulary_size=64, width=16, layers=2, query_heads=2, kv_heads=1,
        head_dimension=8, ffn_width=32, context_length=16, rope_base=10000.0,
        norm_epsilon=1e-5, tied_embeddings=True, qk_norm=True, qk_norm_affine=True,
        linear_bias=False, dropout=0.0)


def test_patch_changes_forward_and_restores_bitwise() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(1)
    model = initialize(_spec(), seed=1).eval()
    tokens = torch.randint(4, 64, (2, 8))
    segment = torch.zeros((2, 8), dtype=torch.int32)
    positions, mask = packed_layout(segment, torch_module=torch)
    with torch.no_grad():
        baseline = model(tokens, positions, mask)
        patch = HormonalAttentionPatch(model, projection=_projection(1.0))
        patched = model(tokens, positions, mask)
        patch.restore()
        restored = model(tokens, positions, mask)
    assert (patched - baseline).abs().max().item() > 1e-6
    assert torch.equal(restored, baseline)


def test_inert_projection_is_noop() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(1)
    model = initialize(_spec(), seed=1).eval()
    tokens = torch.randint(4, 64, (2, 8))
    segment = torch.zeros((2, 8), dtype=torch.int32)
    positions, mask = packed_layout(segment, torch_module=torch)
    with torch.no_grad():
        baseline = model(tokens, positions, mask)
        patch = HormonalAttentionPatch(model, projection=_projection(0.0))
        inert = model(tokens, positions, mask)
        patch.restore()
    assert torch.equal(inert, baseline)
