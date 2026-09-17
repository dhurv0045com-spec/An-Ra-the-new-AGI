"""Small CPU regression tests: HORM mechanisms, never capability evidence."""
from dataclasses import replace

import pytest
import torch

from v5_contracts.model_spec import V5A_250M
from v5_model.core import initialize, packed_layout
from v5_identity.hormonal_model import wrap_core_with_hal


def fixture_model():
    spec = replace(V5A_250M, vocabulary_size=32, width=16, layers=2,
                   query_heads=2, kv_heads=1, head_dimension=8,
                   ffn_width=32, context_length=16)
    core = initialize(spec, 17, torch_module=torch)
    x = torch.tensor([[2, 4, 5, 6]])
    positions, mask = packed_layout(torch.zeros_like(x), torch_module=torch)
    return core, x, positions, mask


def test_registered_projection_reaches_logits_and_gradient():
    core, x, pos, mask = fixture_model()
    baseline = core(x, pos, mask).detach()
    model = wrap_core_with_hal(core, torch=torch, shared_projection=torch.zeros(2, 7))
    model.set_hormone_values({'adrenaline': 0.9})
    assert torch.equal(model(x, pos, mask).detach(), baseline)
    # Even at zero initialization, the active feature must reach the gradient.
    model(x, pos, mask).square().mean().backward()
    assert model.projection.grad is not None
    assert model.projection.grad.abs().max() > 0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.projection.fill_(0.1)
    active = model(x, pos, mask)
    assert not torch.equal(active.detach(), baseline)
    assert torch.isfinite(active).all()
    active.square().mean().backward()
    assert model.projection.grad.abs().max() > 0


def test_noop_matches_baseline_gradients():
    core, x, pos, mask = fixture_model()
    core(x, pos, mask).square().mean().backward()
    expected = {n: p.grad.clone() for n, p in core.named_parameters()}
    core.zero_grad(set_to_none=True)
    model = wrap_core_with_hal(core, torch=torch, shared_projection=torch.zeros(2, 7))
    model.set_hormone_values({'adrenaline': 0.9})
    model(x, pos, mask).square().mean().backward()
    for n, p in model.core.named_parameters():
        torch.testing.assert_close(p.grad, expected[n], atol=0, rtol=0)


def test_checkpoint_roundtrip_and_recomputation(tmp_path):
    core, x, pos, mask = fixture_model()
    model = wrap_core_with_hal(core, torch=torch, shared_projection=torch.ones(2, 7) * 0.1)
    model.set_hormone_values({'adrenaline': 0.9})
    ordinary = model(x, pos, mask)
    ordinary.square().mean().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    recomputed = model(x, pos, mask, use_activation_checkpointing=True)
    torch.testing.assert_close(recomputed, ordinary, atol=0, rtol=0)
    recomputed.square().mean().backward()
    for n, p in model.named_parameters():
        torch.testing.assert_close(p.grad, grads[n], atol=0, rtol=0)
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    other, _, _, _ = fixture_model()
    restored = wrap_core_with_hal(other, torch=torch, shared_projection=torch.zeros(2, 7))
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert restored.hormone_state() == model.hormone_state()
    torch.testing.assert_close(restored(x, pos, mask), ordinary, atol=0, rtol=0)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 1.1])
def test_invalid_state_rejected(value):
    core, _, _, _ = fixture_model()
    model = wrap_core_with_hal(core, torch=torch, shared_projection=torch.zeros(2, 7))
    with pytest.raises(ValueError):
        model.set_hormone_values({'adrenaline': value})


def test_packed_segments_and_causality():
    core, _, _, _ = fixture_model()
    model = wrap_core_with_hal(core, torch=torch, shared_projection=torch.ones(2, 7) * 0.1)
    model.set_hormone_values({'adrenaline': 0.9})
    x = torch.tensor([[2, 4, 5, 2, 7, 8]])
    pos, mask = packed_layout(torch.tensor([[0, 0, 0, 1, 1, 1]]), torch_module=torch)
    original = model(x, pos, mask)
    changed = x.clone()
    changed[0, 1] = 9
    changed[0, 5] = 10
    actual = model(changed, pos, mask)
    torch.testing.assert_close(original[:, 3:5], actual[:, 3:5], atol=0, rtol=0)
    torch.testing.assert_close(original[:, :1], actual[:, :1], atol=0, rtol=0)
    assert model.core.embedding.weight is core.embedding.weight


def test_temperature_bound_and_inactive_channels():
    core, x, pos, mask = fixture_model()
    model = wrap_core_with_hal(core, torch=torch, shared_projection=torch.ones(2, 7) * 100)
    model.set_hormone_values({'oxytocin': 1, 'gaba': 1, 'norepinephrine': 1})
    expected = core(x, pos, mask)
    torch.testing.assert_close(model(x, pos, mask), expected, atol=0, rtol=0)
    model.set_hormone_values({'adrenaline': 1})
    assert model.log_temperature().abs().max() <= 1.5
    assert torch.isfinite(model(x, pos, mask)).all()
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in core.parameters()) + 14


def test_forward_state_snapshot_survives_update_before_backward():
    core, x, pos, mask = fixture_model()
    model = wrap_core_with_hal(core, torch=torch, shared_projection=torch.ones(2, 7) * 0.1)
    model.set_hormone_values({'adrenaline': 0.9})
    model(x, pos, mask, use_activation_checkpointing=True).square().mean().backward()
    expected = model.projection.grad.clone()
    model.zero_grad(set_to_none=True)
    out = model(x, pos, mask, use_activation_checkpointing=True)
    model.set_hormone_values({'adrenaline': 0.1})
    out.square().mean().backward()
    torch.testing.assert_close(model.projection.grad, expected, atol=0, rtol=0)
