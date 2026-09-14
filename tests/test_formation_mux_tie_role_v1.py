from __future__ import annotations


def test_tie_role_protocol_is_prospective_and_complete():
    from v5_experiments import tie_role_protocol_v1 as proto

    assert proto.total_official_arms() == 24
    assert proto.A_UPDATES == 2000
    assert proto.B_PROCESSED_TOKEN_BUDGET == 500_000
    assert proto.A_CHECKPOINT_EVERY_UPDATES == 200
    assert proto.B_CHECKPOINT_EVERY_TOKENS == 10_000
    assert proto.gradient_scales("T0_CANONICAL") == (1.0, 1.0)
    assert proto.gradient_scales("T1_INPUT_X4") == (4.0, 1.0)
    assert proto.gradient_scales("T2_OUTPUT_X025") == (1.0, 0.25)
    assert proto.gradient_scales("T3_BALANCED_X4_X025") == (4.0, 0.25)
    assert proto.gradient_scales("X1_BALANCED_R0") == (4.0, 0.25)
    assert len(proto.protocol_sha(proto.EXPERIMENT_A)) == 64
    assert len(proto.protocol_sha(proto.EXPERIMENT_B)) == 64


def test_gradient_scaled_view_preserves_value_and_scales_gradient():
    import torch
    from anra_v5.tie_role_model_v1 import gradient_scaled_view

    for scale in (0.0, 0.25, 1.0, 4.0):
        w = torch.tensor([1.5, -2.0, 3.25], dtype=torch.float32, requires_grad=True)
        view = gradient_scaled_view(w, scale)
        torch.testing.assert_close(view.detach(), w.detach(), rtol=0.0, atol=0.0)
        view.sum().backward()
        torch.testing.assert_close(w.grad, torch.full_like(w, scale), rtol=0.0, atol=0.0)


def test_forward_equivalence_matched_seed_cpu():
    import torch
    from anra_v5 import tie_role_model_v1 as fxm
    from v5_model.core import packed_layout

    seed = 12345
    canonical = fxm.build_model(seed, "T0_CANONICAL", torch=torch, device=torch.device("cpu"))
    balanced = fxm.build_model(seed, "T3_BALANCED_X4_X025", torch=torch, device=torch.device("cpu"))

    for (name_a, p_a), (name_b, p_b) in zip(canonical.named_parameters(), balanced.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(p_a, p_b, rtol=0.0, atol=0.0)

    tokens = torch.tensor([[2, 260, 269, 3]], dtype=torch.long)
    segments = torch.zeros_like(tokens)
    positions, mask = packed_layout(segments, torch_module=torch)
    with torch.no_grad():
        a = canonical(tokens, positions, mask)
        b = balanced(tokens, positions, mask)
    torch.testing.assert_close(a, b, rtol=0.0, atol=1e-7)


def test_role_gradients_are_additive_on_tied_matrix():
    import torch
    from anra_v5 import tie_role_model_v1 as fxm
    from v5_model.core import packed_layout
    from v5_objectives.causal_lm import causal_lm_loss

    model = fxm.build_model(777, "T0_CANONICAL", torch=torch, device=torch.device("cpu"))
    tokens = torch.tensor([[2, 260, 269, 3]], dtype=torch.long)
    segments = torch.zeros_like(tokens)
    eligible = torch.tensor([[False, False, True, True]], dtype=torch.bool)
    positions, mask = packed_layout(segments, torch_module=torch)

    def grad(inp, out):
        model.zero_grad(set_to_none=True)
        fxm.set_gradient_scales(model, input_scale=inp, output_scale=out)
        logits = model(tokens, positions, mask)
        loss, _ = causal_lm_loss(logits, tokens, segments, eligible=eligible, torch_module=torch)
        loss.backward()
        return model.embedding.weight.grad.detach().clone()

    g_in = grad(1.0, 0.0)
    g_out = grad(0.0, 1.0)
    g_full = grad(1.0, 1.0)
    torch.testing.assert_close(g_full, g_in + g_out, rtol=1e-4, atol=1e-6)


def test_factorial_primary_verdict_thresholds():
    from v5_experiments import tie_role_protocol_v1 as proto

    auc = {73011: 0.08, 73012: 0.07, 73013: 0.06, 73014: -0.01}
    sealed = {73011: 0.10, 73012: 0.08, 73013: 0.07, 73014: -0.01}
    verdict = proto.paired_verdict(formation_auc_deltas=auc, sealed_endpoint_gaps=sealed)
    assert verdict["verdict"] == "SUCCESS"
