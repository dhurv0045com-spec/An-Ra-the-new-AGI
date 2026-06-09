"""Unit tests for CausalTransformerV2 — the production PyTorch model.
These test the REAL model in anra_brain.py, not the archived NumPy reference in core/.
All tests run on CPU. No GPU required. Must complete in under 30 seconds total.
"""
from __future__ import annotations

import pytest
import torch
from anra_brain import CausalTransformerV2


@pytest.fixture(scope="module")
def tiny() -> CausalTransformerV2:
    return CausalTransformerV2(
        vocab_size=256, n_embd=64, n_head=4, n_kv_head=2,
        n_layer=4, block_size=64, mod_layers={1, 2},
    )

def test_forward_logits_shape(tiny):
    idx = torch.randint(0, 256, (2, 32))
    logits, loss = tiny(idx)
    assert logits.shape == (2, 32, 256), f"Expected (2,32,256) got {logits.shape}"
    assert loss is None

def test_forward_with_targets_gives_scalar_loss(tiny):
    idx = torch.randint(0, 256, (2, 32))
    tgt = torch.randint(0, 256, (2, 32))
    _, loss = tiny(idx, targets=tgt)
    assert loss is not None
    assert loss.shape == (), "Loss must be a scalar"
    assert loss.item() > 0
    assert not torch.isnan(loss).item()

def test_no_nan_gradients_after_backward(tiny):
    idx = torch.randint(0, 256, (1, 32))
    tgt = torch.randint(0, 256, (1, 32))
    _, loss = tiny(idx, targets=tgt)
    loss.backward()
    for name, param in tiny.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any().item(), f"NaN gradient in {name}"
    tiny.zero_grad()

def test_mod_router_gate_gradient_nonzero(tiny):
    idx = torch.randint(0, 256, (1, 32))
    tgt = torch.randint(0, 256, (1, 32))
    _, loss = tiny(idx, targets=tgt)
    loss.backward()
    for layer_idx, router in tiny.mod_routers.items():
        g = router.gate.weight.grad
        assert g is not None, f"MoD layer {layer_idx}: gate has no gradient"
        assert g.abs().max().item() > 1e-9, f"MoD layer {layer_idx}: gate gradient is zero"
    tiny.zero_grad()


def test_gradient_checkpointing_gate_gradient_not_stale():
    """Verify that MoD gate gradients survive checkpoint recomputation."""
    model = CausalTransformerV2(
        vocab_size=256,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        n_layer=4,
        block_size=64,
        mod_layers={1, 2},
    )
    model.gradient_checkpointing_enable()
    model.train()
    idx = torch.randint(0, 256, (1, 32))
    tgt = torch.randint(0, 256, (1, 32))
    _, loss = model(idx, targets=tgt)
    assert loss is not None
    loss.backward()
    for layer_idx, router in model.mod_routers.items():
        gradient = router.gate.weight.grad
        assert gradient is not None, f"MoD layer {layer_idx}: no gradient with checkpointing"
        assert gradient.abs().max().item() > 1e-9, (
            f"MoD layer {layer_idx}: zero gradient with checkpointing - closure bug"
        )
    model.zero_grad()
    model.gradient_checkpointing_disable()

def test_kv_cache_matches_no_cache(tiny):
    tiny.eval()
    idx = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        logits_plain, _ = tiny(idx)
    tiny.enable_kv_cache()
    with torch.no_grad():
        logits_cached, _ = tiny(idx)
    tiny.disable_kv_cache()
    torch.testing.assert_close(logits_plain, logits_cached, atol=1e-4, rtol=1e-4)
    tiny.train()

def test_tied_embeddings_share_memory(tiny):
    assert tiny.token_embedding.weight.data_ptr() == tiny.lm_head.weight.data_ptr(), \
        "Embeddings and lm_head must share weight tensor (tied embeddings)"

def test_sequence_exceeds_block_size_raises(tiny):
    idx = torch.randint(0, 256, (1, 65))  # block_size = 64
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        tiny(idx)

def test_generate_extends_sequence_by_n_tokens(tiny):
    tiny.eval()
    idx = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        out = tiny.generate(idx, max_new_tokens=10, temperature=1.0, top_k=20)
    assert out.shape[1] == 18, f"Expected length 18 (8+10), got {out.shape[1]}"
    tiny.train()

def test_loss_decreases_with_overfit_step(tiny):
    """Single overfit step on one batch — loss must decrease."""
    tiny.train()
    idx = torch.randint(0, 256, (1, 32))
    tgt = torch.randint(0, 256, (1, 32))
    opt = torch.optim.AdamW(tiny.parameters(), lr=1e-2)
    _, loss_before = tiny(idx, targets=tgt)
    loss_before.backward()
    opt.step()
    opt.zero_grad()
    with torch.no_grad():
        _, loss_after = tiny(idx, targets=tgt)
    assert loss_after.item() < loss_before.item(), \
        f"Loss did not decrease: {loss_before.item():.4f} → {loss_after.item():.4f}"

def test_model_registered_in_registry():
    """CausalTransformerV2 must be discoverable via MODEL_REGISTRY."""
    from anra.core.registry import MODEL_REGISTRY
    assert "causal_transformer_v2" in MODEL_REGISTRY, (
        "CausalTransformerV2 is not registered. "
        "Add @MODEL_REGISTRY.register('causal_transformer_v2') above the class definition."
    )
    model = MODEL_REGISTRY.build(
        "causal_transformer_v2",
        vocab_size=256, n_embd=64, n_head=4, n_kv_head=2,
        n_layer=2, block_size=64,
    )
    idx = torch.randint(0, 256, (1, 16))
    logits, _ = model(idx)
    assert logits.shape == (1, 16, 256)


def test_model_config_roundtrip(tiny: CausalTransformerV2):
    """model_config() must contain every parameter needed for reconstruction."""
    import json

    cfg = tiny.model_config()
    required = {
        "vocab_size",
        "n_embd",
        "n_head",
        "n_layer",
        "block_size",
        "n_kv_head",
        "use_hal",
        "use_layer_temperature_bias",
    }
    missing = required - set(cfg)
    assert not missing, f"model_config() missing keys: {missing}"
    json.dumps(cfg)
