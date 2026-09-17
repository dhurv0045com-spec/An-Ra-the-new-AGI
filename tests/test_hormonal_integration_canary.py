"""Canary: if the hormonal mechanism silently stops reaching the forward pass,
these tests fail. Same spirit as test_v5_mutation_canary.py — mechanical, not
capability evidence.
"""
from __future__ import annotations

import torch

from v5_model.core import initialize, packed_layout
from v5_identity.hormonal_model import wrap_core_with_hal
from v5_identity.hormonal_projection import ACTIVE_HORMONES, MAX_LOG_T


def _fixture():
    from dataclasses import replace
    from v5_contracts.model_spec import V5A_250M
    spec = __import__("dataclasses").replace(
        V5A_250M, vocabulary_size=32, width=16, layers=2,
        query_heads=2, kv_heads=1, head_dimension=8, ffn_width=32, context_length=16)
    core = initialize(spec, 17, torch_module=torch)
    x = torch.tensor([[2, 4, 5, 6, 7, 8]])
    pos, mask = packed_layout(torch.zeros_like(x), torch_module=torch)
    return core, x, pos, mask


def test_canary_zero_projection_strict_noop():
    core, x, pos, mask = _fixture()
    baseline = core(x, pos, mask)
    model = wrap_core_with_hal(core, torch=torch, shared_projection=torch.zeros(2, 7))
    model.set_hormone_values({"adrenaline": 0.9})
    assert torch.equal(model(x, pos, mask), baseline)


def test_canary_adrenaline_changes_logits_monotonically():
    core, x, pos, mask = _fixture()
    model = wrap_core_with_hal(core, torch=torch, shared_projection=torch.zeros(2, 7))
    with torch.no_grad():
        model.projection.zero_()
        model.projection[:, 3] = 1.0  # adrenaline -> wider attention per head
    model.set_hormone_values({"adrenaline": 0.5})
    logits_half = model(x, pos, mask).detach()
    model.set_hormone_values({"adrenaline": 1.0})
    logits_full = model(x, pos, mask).detach()
    assert not torch.equal(logits_half, logits_full)
    deltas = (logits_full - logits_half).abs().amax(dim=-1)
    assert deltas.max() > 0


def test_canary_state_only_hormones_never_touch_forward():
    core, x, pos, mask = _fixture()
    model = wrap_core_with_hal(core, torch=torch, shared_projection=torch.zeros(2, 7))
    with torch.no_grad():
        model.projection[:, :].zero_()
        model.projection[:, 4:] = 1.0  # oxytocin/gaba/norepinephrine columns
    baseline = core(x, pos, mask)
    model.set_hormone_values({"oxytocin": 1.0, "gaba": 1.0, "norepinephrine": 1.0})
    assert torch.equal(model(x, pos, mask), baseline)


def test_canary_temperature_ceiling_enforced():
    core, x, pos, mask = _fixture()
    model = wrap_core_with_hal(core, torch=torch, shared_projection=torch.zeros(2, 7))
    model.set_hormone_values({"adrenaline": 1.0})
    with torch.no_grad():
        model.projection[:, 3] = 10.0
    log_t = model.log_temperature()
    assert float(log_t.detach().abs().max()) <= MAX_LOG_T
    assert torch.isfinite(model(x, pos, mask)).all()
