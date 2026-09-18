"""HORM-001 regression tests (CPU-only, tiny, thread-capped)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch

from v5_identity import (
    HORMONES,
    HormonalProjection,
    HormonalState,
    V5A_250M_HORMONAL_V1,
)
from v5_contracts.model_spec import V5A_250M


pytestmark = pytest.mark.filterwarnings("ignore")


def _projection(alpha: float) -> HormonalProjection:
    return HormonalProjection(
        weights=tuple(0.1 if n in ("cortisol", "adrenaline") else 0.05
                      for n in HORMONES),
        bound=V5A_250M_HORMONAL_V1.bound,
        raw_alpha=alpha,
    )


class TestHormonalState:
    def test_baseline_in_bounds_and_ordered(self) -> None:
        state = HormonalState.baseline()
        assert tuple(state.values) == HORMONES
        assert all(0.0 <= v <= 2.0 for v in state.values.values())

    def test_appraise_moves_verified_outcomes_only(self) -> None:
        state = HormonalState.baseline()
        state.appraise("success")
        assert state.values["dopamine"] > 0.10
        state.appraise("failure")
        assert state.values["cortisol"] > 0.10
        with pytest.raises(ValueError):
            state.appraise("vibes")

    def test_decay_moves_toward_baseline_and_cortisol_inherits_adrenaline(self) -> None:
        state = HormonalState.baseline()
        state.appraise("failure")
        pre_adrenaline = state.values["adrenaline"]
        pre_cortisol = state.values["cortisol"]
        state.decay()
        assert state.values["adrenaline"] < pre_adrenaline
        assert state.values["cortisol"] >= pre_cortisol

    def test_bounds_are_enforced(self) -> None:
        with pytest.raises(ValueError):
            HormonalState(values={name: 5.0 for name in HORMONES})


class TestProjection:
    def test_inert_at_zero_alpha(self) -> None:
        projection = _projection(0.0)
        assert projection.scale(HormonalState.baseline().vector()) == 1.0

    def test_bounded_scale_range(self) -> None:
        projection = _projection(1.0)
        for vector in (tuple(2.0 for _ in HORMONES), tuple(0.0 for _ in HORMONES)):
            assert 0.8 <= projection.scale(vector) <= 1.2

    def test_rejects_out_of_range_alpha(self) -> None:
        with pytest.raises(ValueError):
            _projection(-0.1)


class TestOverlay:
    def test_overlay_binds_frozen_base_spec(self) -> None:
        V5A_250M_HORMONAL_V1.assert_sibling()
        assert V5A_250M_HORMONAL_V1.base_spec_sha256 == V5A_250M.sha256()


class TestAttentionWiringCanary:
    def test_attention_scale_canary(self) -> None:
        torch.set_num_threads(2)
        torch.manual_seed(707)
        from types import SimpleNamespace

        from v5_model.attention import build_attention

        config = SimpleNamespace(
            width=16, query_heads=2, kv_heads=1, head_dimension=8,
            qk_norm=True, qk_norm_affine=True, qk_norm_epsilon=1e-6,
            rope_base=10000.0,
        )
        attention = build_attention(config, torch_module=torch).eval()
        hidden = torch.randn(2, 6, 16)
        positions = torch.arange(6).expand(2, -1)
        mask = torch.ones(6, 6, dtype=torch.bool).tril()[None, None]
        with torch.no_grad():
            baseline = attention(hidden, positions, mask)
            q = attention.query(hidden).view(2, 6, 2, 8).transpose(1, 2)
            k = attention.key(hidden).view(2, 6, 1, 8).transpose(1, 2)
            v = attention.value(hidden).view(2, 6, 1, 8).transpose(1, 2)
            q = attention.rope(attention.normalize(q, attention.query_scale), positions)
            k = attention.rope(attention.normalize(k, attention.key_scale), positions)
            k = k.repeat_interleave(2, dim=1)
            v = v.repeat_interleave(2, dim=1)
            noop = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask)
            torch.testing.assert_close(
                attention.output(noop.transpose(1, 2).contiguous().view(2, 6, 16)),
                baseline, rtol=0, atol=0)
            scale = _projection(1.0).scale(HormonalState.baseline().vector())
            treated = torch.nn.functional.scaled_dot_product_attention(
                q * scale, k, v, attn_mask=mask)
            difference = (attention.output(
                treated.transpose(1, 2).contiguous().view(2, 6, 16)) - baseline
            ).abs().max().item()
        assert difference > 1e-6
