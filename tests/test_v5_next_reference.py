"""V5-Next reference implementation tests: tiny CPU fixtures, no training, no GPU.

Run with a torch-capable interpreter, e.g.:
  ark014-cu126-venv/Scripts/python.exe -m pytest tests/test_v5_next_reference.py -q
Skips cleanly when torch is unavailable so non-GPU CI still passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

torch = pytest.importorskip("torch")

from v5_next.contracts import (  # noqa: E402
    CANONICAL_OUTPUT_MODE,
    TINY_REFERENCE_GEOMETRY,
    NextCoreContract,
    output_training_logits,
)
from v5_next.reference import build_reference_model, checkpoint_round_trip  # noqa: E402
from v5_model.core import packed_layout  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return build_reference_model(TINY_REFERENCE_GEOMETRY, seed=7)


def _inputs(model, batch=2, length=8):
    generator = torch.Generator().manual_seed(1234)
    tokens = torch.randint(0, TINY_REFERENCE_GEOMETRY.vocabulary_size,
                           (batch, length), generator=generator)
    segments = torch.zeros(batch, length, dtype=torch.int32)
    positions, mask = packed_layout(segments, torch_module=torch)
    return tokens, positions, mask


def test_parameter_count_matches_contract_receipt(model):
    expected = TINY_REFERENCE_GEOMETRY.parameter_receipt()["total"]
    actual = sum(int(p.numel()) for p in model.parameters())
    assert actual == expected


def test_tied_weight_identity(model):
    weight = model.embedding.weight
    names = [name for name, _ in model.named_parameters()]
    assert names.count("embedding.weight") == 1
    assert not any("lm_head" in name or "output_head" in name for name in names)
    assert weight.shape == (TINY_REFERENCE_GEOMETRY.vocabulary_size,
                            TINY_REFERENCE_GEOMETRY.width)


def test_affine_qk_norm_scales_present(model):
    names = [name for name, _ in model.named_parameters()]
    assert sum(name.endswith("query_scale") for name in names) == TINY_REFERENCE_GEOMETRY.layers
    assert sum(name.endswith("key_scale") for name in names) == TINY_REFERENCE_GEOMETRY.layers


def test_causal_masking_prefix_invariance(model):
    tokens, positions, mask = _inputs(model)
    with torch.no_grad():
        base = model(tokens, positions, mask)
    perturbed_tokens = tokens.clone()
    perturbed_tokens[:, -1] = (perturbed_tokens[:, -1] + 1) % TINY_REFERENCE_GEOMETRY.vocabulary_size
    with torch.no_grad():
        changed = model(perturbed_tokens, positions, mask)
    assert torch.equal(base[:, :-1], changed[:, :-1])
    assert not torch.equal(base[:, -1], changed[:, -1])


def test_packed_segment_isolation(model):
    generator = torch.Generator().manual_seed(99)
    tokens = torch.randint(0, TINY_REFERENCE_GEOMETRY.vocabulary_size,
                           (1, 8), generator=generator)
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.int32)
    positions, mask = packed_layout(segments, torch_module=torch)
    with torch.no_grad():
        base = model(tokens, positions, mask)
    other = tokens.clone()
    other[0, 4:] = (other[0, 4:] + 7) % TINY_REFERENCE_GEOMETRY.vocabulary_size
    with torch.no_grad():
        changed = model(other, positions, mask)
    assert torch.equal(base[0, :4], changed[0, :4])


def test_deterministic_initialization():
    a = build_reference_model(TINY_REFERENCE_GEOMETRY, seed=11)
    b = build_reference_model(TINY_REFERENCE_GEOMETRY, seed=11)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb)


def test_finite_forward_and_backward(model):
    tokens, positions, mask = _inputs(model)
    out = model(tokens, positions, mask)
    assert torch.isfinite(out).all()
    loss = out.float().square().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_optimizer_ownership_and_update(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    tokens, positions, mask = _inputs(model)
    (model(tokens, positions, mask).float().square().mean()).backward()
    optimizer.step()
    changed = any(not torch.equal(p.detach(), b) for p, b in zip(model.parameters(), before))
    assert changed


def test_checkpoint_round_trip(model):
    assert checkpoint_round_trip(model) is True


def test_contract_rejects_unsupervised_eos():
    with pytest.raises(ValueError):
        NextCoreContract(**{**TINY_REFERENCE_GEOMETRY.__dict__, "eos_supervised": False})


def test_experimental_output_modes_are_gated():
    with pytest.raises(ValueError):
        NextCoreContract(**{**TINY_REFERENCE_GEOMETRY.__dict__,
                            "output_mode": "participating_mask"})
    experimental = NextCoreContract(**{**TINY_REFERENCE_GEOMETRY.__dict__,
                                       "output_mode": "participating_mask",
                                       "allow_experimental": True})
    assert experimental.component_statuses()["output_mode"] == "EXPERIMENT_ONLY"
    assert experimental.identity_sha256() != TINY_REFERENCE_GEOMETRY.identity_sha256()


def test_canonical_output_mode_untouched_by_treatments():
    tokens = torch.randn(1, 4, TINY_REFERENCE_GEOMETRY.vocabulary_size)
    active = tuple(range(19))
    canonical = output_training_logits(
        tokens, output_mode=CANONICAL_OUTPUT_MODE,
        active_ids=active, inactive_offset_log_value=1.0)
    assert torch.equal(canonical, tokens)
    masked = output_training_logits(
        tokens, output_mode="participating_mask",
        active_ids=active, inactive_offset_log_value=1.0)
    assert torch.equal(masked[..., :19], tokens[..., :19])
    offset = output_training_logits(
        tokens, output_mode="inactive_offset",
        active_ids=active, inactive_offset_log_value=0.5)
    assert torch.equal(offset[..., :19], tokens[..., :19])
    assert torch.allclose(offset[..., 19:], tokens[..., 19:] - 0.5)
