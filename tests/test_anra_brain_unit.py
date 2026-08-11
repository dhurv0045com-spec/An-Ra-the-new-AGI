"""Unit tests for CausalTransformerV2 — the production PyTorch model.
These test the REAL model in anra_brain.py, not the archived NumPy reference in core/.
All tests run on CPU. No GPU required. Must complete in under 30 seconds total.
"""

from __future__ import annotations

import copy

import pytest
import torch
from anra_brain import (
    CausalTransformerV2,
    MoDRouter,
    MultiHeadAttentionV2,
    RouterContext,
    SparseUpcycledMoE,
    SwiGLU,
)


@pytest.fixture(scope="module")
def tiny() -> CausalTransformerV2:
    return CausalTransformerV2(
        vocab_size=256,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        n_layer=4,
        block_size=64,
        mod_layers={1, 2},
    )


def test_forward_logits_shape(tiny: CausalTransformerV2) -> None:
    idx = torch.randint(0, 256, (2, 32))
    logits, loss = tiny(idx)
    assert logits.shape == (2, 32, 256), f"Expected (2,32,256) got {logits.shape}"
    assert loss is None


def test_forward_with_targets_gives_scalar_loss(tiny: CausalTransformerV2) -> None:
    idx = torch.randint(0, 256, (2, 32))
    tgt = torch.randint(0, 256, (2, 32))
    _, loss = tiny(idx, targets=tgt)
    assert loss is not None
    assert loss.shape == (), "Loss must be a scalar"
    assert loss.item() > 0
    assert not torch.isnan(loss).item()


def test_no_nan_gradients_after_backward(tiny: CausalTransformerV2) -> None:
    idx = torch.randint(0, 256, (1, 32))
    tgt = torch.randint(0, 256, (1, 32))
    _, loss = tiny(idx, targets=tgt)
    loss.backward()
    for name, param in tiny.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any().item(), f"NaN gradient in {name}"
    tiny.zero_grad()


def test_mod_router_gate_gradient_nonzero(tiny: CausalTransformerV2) -> None:
    idx = torch.randint(0, 256, (1, 32))
    tgt = torch.randint(0, 256, (1, 32))
    _, loss = tiny(idx, targets=tgt)
    loss.backward()
    for layer_idx, router in tiny.mod_routers.items():
        g = router.gate.weight.grad
        assert g is not None, f"MoD layer {layer_idx}: gate has no gradient"
        assert g.abs().max().item() > 1e-9, f"MoD layer {layer_idx}: gate gradient is zero"
    tiny.zero_grad()


def test_mod_router_uses_hard_forward_and_straight_through_backward() -> None:
    router = MoDRouter(4, capacity=0.5)
    router.train()
    with torch.no_grad():
        router.gate.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    x = torch.tensor(
        [
            [
                [4.0, 1.0, 0.0, 0.0],
                [3.0, 1.0, 0.0, 0.0],
                [-3.0, 1.0, 0.0, 0.0],
                [-4.0, 1.0, 0.0, 0.0],
            ]
        ],
        requires_grad=True,
    )
    output = router(x, torch.nn.Identity(), RouterContext())

    assert not torch.equal(output[:, :2], x[:, :2])
    torch.testing.assert_close(output[:, 2:], x[:, 2:])
    output.sum().backward()
    assert router.gate.weight.grad is not None
    assert router.gate.weight.grad.abs().sum() > 0


def test_mod_router_dispatches_only_selected_tokens_with_train_eval_parity() -> None:
    """MoD must skip unselected FFN rows without changing its forward semantics."""

    class CountingFFN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rows_seen = 0

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            self.rows_seen += int(values.shape[0] * values.shape[1])
            return values * 3.0

    router = MoDRouter(2, capacity=0.5)
    with torch.no_grad():
        router.gate.weight.copy_(torch.tensor([[1.0, 0.0]]))
        router.capacity_control.zero_()
    x = torch.tensor(
        [[[4.0, 1.0], [3.0, 1.0], [-3.0, 1.0], [-4.0, 1.0]]],
        requires_grad=True,
    )
    ffn = CountingFFN()

    router.train()
    train_out = router(x, ffn)
    assert ffn.rows_seen == 2  # 50% capacity, not the old dense 4-row FFN.
    train_out.square().mean().backward()
    assert router.gate.weight.grad is not None
    assert router.gate.weight.grad.abs().sum() > 0

    ffn.rows_seen = 0
    router.eval()
    with torch.no_grad():
        eval_out = router(x.detach(), ffn)
    assert ffn.rows_seen == 2
    torch.testing.assert_close(train_out.detach(), eval_out)


def test_router_context_civ_changes_gate_strength() -> None:
    router = MoDRouter(2, capacity=1.0)
    router.eval()
    with torch.no_grad():
        router.gate.weight.zero_()
        router.context_weights.copy_(torch.tensor([0.0, 0.0, 2.0]))
    x = torch.ones(1, 3, 2)
    low = router(x, torch.nn.Identity(), RouterContext(civ_similarity=0.0))
    high = router(x, torch.nn.Identity(), RouterContext(civ_similarity=1.0))
    assert not torch.allclose(low, high)


def test_diagnostic_mode_records_no_native_subsystem_execution() -> None:
    model = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=16,
        mod_layers={1},
    )
    prior = model.configure_runtime_mode("diagnostic")
    assert model.use_layer_temperature_bias is False
    model.begin_subsystem_trace(civ_similarity=0.25)
    model.eval()
    with torch.no_grad():
        model(torch.randint(0, 64, (1, 8)))
    execution = model.subsystem_telemetry()["execution"]
    model.restore_runtime_mode(prior)
    assert execution == {
        "mod": 0,
        "rim": 0,
        "dstp": 0,
        "esv": 0,
        "esv_features": 0,
        "hal": 0,
    }


def test_explicit_subsystem_policy_is_dense_or_isolated() -> None:
    model = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_layer=2,
        block_size=16,
        mod_layers={1},
    )
    dense = model.configure_subsystems(set())
    assert not any(dense.values())
    assert model.use_layer_temperature_bias is False

    isolated = model.configure_subsystems({"mod"})
    assert isolated == {
        "mod": True,
        "rim": False,
        "dstp": False,
        "esv": False,
        "hal": False,
    }


def test_runtime_can_activate_only_the_checkpoint_approved_recipe() -> None:
    model = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_layer=2,
        block_size=16,
        mod_layers={1},
    )

    prior = model.configure_runtime_mode("full_system")
    assert not model.use_mod
    assert not model.use_rim
    assert not model.use_dstp
    assert not model.use_esv_control
    assert not model.use_hal
    model.restore_runtime_mode(prior)

    model.configure_subsystems({"mod", "esv"})
    prior = model.configure_runtime_mode("native")
    assert model.use_mod
    assert model.use_esv_control
    assert not model.use_rim
    assert not model.use_dstp
    assert not model.use_hal
    model.restore_runtime_mode(prior)


def test_esv_control_ablation_keeps_rim_dependency_visible_without_false_esv_trace() -> None:
    model = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=16,
        mod_layers={1},
    )
    model.configure_subsystems({"mod", "rim", "dstp", "esv"})
    prior = model.configure_runtime_mode("native")
    model.neutralize_subsystem("esv")
    model.begin_subsystem_trace()
    model.eval()
    with torch.no_grad():
        model(torch.randint(0, 64, (1, 8)))
    execution = model.subsystem_telemetry()["execution"]
    model.restore_runtime_mode(prior)
    assert model.use_layer_temperature_bias is True

    assert execution["esv"] == 0
    assert execution["esv_features"] > 0
    assert execution["rim"] > 0


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


def test_layer_temperature_bias_is_trainable_positive_bounded_and_reported() -> None:
    model = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=16,
        mod_layers={1},
    )
    assert isinstance(model.layer_temperature_bias_log, torch.nn.Parameter)
    assert model.layer_temperature_bias_log.requires_grad
    torch.testing.assert_close(model._layer_temperature_bias(0), torch.tensor(1.0))

    with torch.no_grad():
        model.layer_temperature_bias_log.copy_(torch.tensor([-100.0, 100.0]))
    torch.testing.assert_close(model._layer_temperature_bias(0), torch.tensor(0.5))
    torch.testing.assert_close(model._layer_temperature_bias(1), torch.tensor(2.0))
    assert model.subsystem_telemetry()["layer_temperature_biases"] == [0.5, 2.0]

    with torch.no_grad():
        model.layer_temperature_bias_log.zero_()
    logits, loss = model(torch.randint(0, 64, (1, 12)), torch.randint(0, 64, (1, 12)))
    assert logits.shape == (1, 12, 64)
    assert loss is not None
    (loss + model.native_regularization_loss()).backward()
    gradient = model.layer_temperature_bias_log.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max().item() > 0.0


def test_gradient_checkpointing_matches_plain_forward_and_gradients() -> None:
    """Checkpoint recomputation must represent the exact same layer function."""
    torch.manual_seed(20260710)
    plain = CausalTransformerV2(
        vocab_size=128,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=4,
        block_size=32,
        mod_layers={1, 2},
    )
    checkpointed = copy.deepcopy(plain)
    plain.gradient_checkpointing_disable()
    checkpointed.gradient_checkpointing_enable()
    plain.train()
    checkpointed.train()
    idx = torch.randint(0, 128, (1, 24))
    targets = torch.randint(0, 128, (1, 24))

    plain_logits, plain_loss = plain(idx, targets=targets)
    checkpointed_logits, checkpointed_loss = checkpointed(idx, targets=targets)
    assert plain_loss is not None and checkpointed_loss is not None
    plain_total = plain_loss + plain.native_regularization_loss()
    checkpointed_total = checkpointed_loss + checkpointed.native_regularization_loss()
    plain_total.backward()
    checkpointed_total.backward()

    torch.testing.assert_close(plain_logits, checkpointed_logits, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(plain_total, checkpointed_total, atol=1e-6, rtol=1e-6)
    checkpointed_parameters = dict(checkpointed.named_parameters())
    for name, parameter in plain.named_parameters():
        other = checkpointed_parameters[name]
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is other.grad, f"gradient presence differs for {name}"
            continue
        torch.testing.assert_close(
            parameter.grad,
            other.grad,
            atol=2e-6,
            rtol=2e-5,
            msg=lambda message, param=name: f"gradient mismatch for {param}: {message}",
        )


def test_kv_cache_matches_no_cache(tiny: CausalTransformerV2) -> None:
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


def test_incremental_kv_cache_uses_absolute_rotary_position(
    tiny: CausalTransformerV2,
) -> None:
    tiny.eval()
    mode = tiny.configure_runtime_mode("diagnostic")
    prompt = torch.randint(0, 256, (1, 12))
    next_token = torch.randint(0, 256, (1, 1))
    with torch.no_grad():
        full_logits, _ = tiny(torch.cat([prompt, next_token], dim=1))
    tiny.enable_kv_cache()
    tiny.clear_kv_cache()
    with torch.no_grad():
        tiny(prompt)
        cached_logits, _ = tiny(next_token)
    tiny.disable_kv_cache()
    tiny.restore_runtime_mode(mode)
    torch.testing.assert_close(
        full_logits[:, -1, :],
        cached_logits[:, -1, :],
        atol=1e-4,
        rtol=1e-4,
    )


def test_tied_embeddings_share_memory(tiny: CausalTransformerV2) -> None:
    assert tiny.token_embedding.weight.data_ptr() == tiny.lm_head.weight.data_ptr(), (
        "Embeddings and lm_head must share weight tensor (tied embeddings)"
    )


def test_residual_projections_use_depth_scaled_initialization() -> None:
    torch.manual_seed(17)
    model = CausalTransformerV2(
        vocab_size=128,
        n_embd=256,
        n_head=8,
        n_layer=8,
        block_size=16,
        mod_layers=(),
    )
    expected = 0.02 / (16**0.5)
    assert model.initialization_scheme == "depth_scaled_residual_v1"
    assert abs(model.blocks[0].attn.out_proj.weight.std().item() - expected) < 0.0002
    assert abs(model.blocks[0].mlp.down_proj.weight.std().item() - expected) < 0.0002
    assert model.blocks[0].attn.q_proj.weight.std().item() > expected * 3


def test_sequence_exceeds_block_size_raises(tiny: CausalTransformerV2) -> None:
    idx = torch.randint(0, 256, (1, 65))  # block_size = 64
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        tiny(idx)


def test_generate_extends_sequence_by_n_tokens(tiny: CausalTransformerV2) -> None:
    tiny.eval()
    idx = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        out = tiny.generate(idx, max_new_tokens=10, temperature=1.0, top_k=20)
    assert out.shape[1] == 18, f"Expected length 18 (8+10), got {out.shape[1]}"
    tiny.train()


def test_loss_decreases_with_overfit_step() -> None:
    """Single overfit step on one batch — loss must decrease."""
    torch.manual_seed(1701)
    tiny = CausalTransformerV2(
        vocab_size=256,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=64,
        mod_layers={1},
    )
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
    assert loss_after.item() < loss_before.item(), (
        f"Loss did not decrease: {loss_before.item():.4f} → {loss_after.item():.4f}"
    )


def test_model_registered_in_registry():
    """CausalTransformerV2 must be discoverable via MODEL_REGISTRY."""
    from anra.core.registry import MODEL_REGISTRY

    assert "causal_transformer_v2" in MODEL_REGISTRY, (
        "CausalTransformerV2 is not registered. "
        "Add @MODEL_REGISTRY.register('causal_transformer_v2') above the class definition."
    )
    model = MODEL_REGISTRY.build(
        "causal_transformer_v2",
        vocab_size=256,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=64,
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


def test_get_hidden_states_shape(tiny: CausalTransformerV2) -> None:
    idx = torch.randint(0, 256, (1, 16))
    states = tiny.get_hidden_states(idx)
    assert len(states) == 4
    assert all(state.shape == (1, 16, 64) for state in states)


def test_layer_norms_are_positive(tiny: CausalTransformerV2) -> None:
    norms = tiny.layer_norms()
    assert len(norms) == 4
    assert all(norm > 0 for norm in norms)


def test_model_config_serializable(tiny: CausalTransformerV2) -> None:
    import json

    json.dumps(tiny.model_config())


def test_qk_norm_and_hybrid_attention_are_explicit_model_contracts() -> None:
    model = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=4,
        block_size=16,
        mod_layers=(),
        use_qk_norm=True,
        sliding_window=4,
        full_attention_every=4,
    )
    assert [block.attn.sliding_window for block in model.blocks] == [4, 4, 4, None]
    assert all(block.attn.use_qk_norm for block in model.blocks)
    assert model.model_config()["use_qk_norm"] is True
    assert model.model_config()["sliding_window"] == 4
    assert model.model_config()["full_attention_every"] == 4


def test_sliding_attention_excludes_tokens_older_than_its_window() -> None:
    torch.manual_seed(19)
    attention = MultiHeadAttentionV2(
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        target_seq_len=16,
        sliding_window=3,
        use_qk_norm=True,
    ).eval()
    original = torch.randn(1, 6, 32)
    changed = original.clone()
    changed[:, 0, :] += 100.0
    with torch.no_grad():
        original_last = attention(original)[:, -1, :]
        changed_last = attention(changed)[:, -1, :]
    torch.testing.assert_close(original_last, changed_last, atol=1e-6, rtol=1e-6)


def test_mtp_heads_have_finite_weighted_future_token_gradients() -> None:
    torch.manual_seed(23)
    model = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=16,
        mod_layers=(),
        use_mtp=True,
        mtp_depth=2,
        mtp_loss_weight=0.2,
    )
    inputs = torch.randint(1, 64, (2, 12))
    targets = torch.randint(1, 64, (2, 12))
    model(inputs)
    mtp_loss = model.multi_token_prediction_loss(targets)
    assert torch.isfinite(mtp_loss)
    assert mtp_loss.item() > 0.0
    mtp_loss.backward()
    assert all(
        projection.weight.grad is not None
        and torch.isfinite(projection.weight.grad).all()
        and projection.weight.grad.abs().sum() > 0
        for projection in model.mtp_projections
    )


def test_sparse_moe_upcycle_starts_with_exact_dense_function_parity() -> None:
    torch.manual_seed(29)
    dense = SwiGLU(16, 32)
    moe = SparseUpcycledMoE(dense, routed_experts=8, top_k=2).eval()
    inputs = torch.randn(3, 7, 16)
    with torch.no_grad():
        expected = dense(inputs)
        actual = moe(inputs)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_sparse_moe_balances_with_persistent_bias_not_auxiliary_loss() -> None:
    torch.manual_seed(31)
    moe = SparseUpcycledMoE(SwiGLU(16, 32), routed_experts=8, top_k=2).train()
    with torch.no_grad():
        moe.router.weight.zero_()
    output = moe(torch.randn(2, 5, 16))
    output.square().mean().backward()
    before = moe.expert_bias.clone()
    moe.update_balance()
    assert not torch.equal(before, moe.expert_bias)
    assert torch.isclose(moe.expert_load_ema.sum(), torch.tensor(1.0))
