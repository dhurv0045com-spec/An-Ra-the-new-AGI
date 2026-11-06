"""Mechanical forward/backward checks for candidate-space recurrent copying."""

from dataclasses import replace

import torch

from anra_v5.elendril.associative import AssociativeDecoder
from anra_v5.elendril.codec import BOS, EOS, encode
from anra_v5.elendril.training import MODEL, generate, generate_batched, logits_for
from v5_model.core import initialize, packed_layout


def _model(mode):
    return AssociativeDecoder(initialize(replace(MODEL, vocabulary_size=1024), 7), 7, mode=mode)


def _packed():
    first = [BOS, *encode("1:A:B;B:C?A>"), EOS]
    second = [BOS, *encode("4:D:E;E:F?D>"), EOS]
    tokens = torch.tensor([[*first, *second, 0]])
    segments = torch.tensor([[*([0] * len(first)), *([1] * len(second)), -1]])
    return tokens, segments, len(first)


def test_grouped_intermediate_matches_dense_forward_and_backward():
    torch.set_num_threads(1)
    tokens, segments, _ = _packed()
    sparse, dense = _model("recurrent"), _model("recurrent-dense")
    dense.load_state_dict(sparse.state_dict())
    sparse_logits = logits_for(sparse, tokens, segments)
    dense_logits = logits_for(dense, tokens, segments)
    assert torch.isfinite(sparse_logits).all()
    torch.testing.assert_close(sparse_logits, dense_logits, rtol=1e-6, atol=1e-7)
    # Every vocabulary coordinate participates in the comparison.
    weights = torch.linspace(-0.1, 0.1, sparse_logits.numel()).reshape_as(sparse_logits)
    (sparse_logits * weights).mean().backward()
    (dense_logits * weights).mean().backward()
    for left, right in zip(sparse.parameters(), dense.parameters()):
        assert left.grad is not None and right.grad is not None
        assert torch.isfinite(left.grad).all()
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)
    assert sparse.join_offsets.grad.abs().sum() > 0


def test_public_depth_is_causal_segment_local_and_bounded():
    torch.set_num_threads(1)
    tokens, segments, boundary = _packed()
    model = _model("recurrent")
    pos, _ = packed_layout(segments, torch_module=torch)
    depth = model._requested_depth(tokens, pos)
    assert depth[0, :boundary].tolist() == [1] * boundary
    assert depth[0, boundary].item() == 1
    assert depth[0, boundary + 1:-1].tolist() == [4] * (tokens.size(1) - boundary - 2)
    expected = logits_for(model, tokens, segments)
    changed = tokens.clone()
    changed[0, boundary + 1:] = 3
    actual = logits_for(model, changed, segments)
    torch.testing.assert_close(actual[0, :boundary], expected[0, :boundary], rtol=0, atol=0)


def test_batched_generation_respects_legacy_order_eos_and_work_counters():
    torch.set_num_threads(1)
    model = AssociativeDecoder(initialize(MODEL, 7), 7, mode="recurrent")
    prompts = (("first", "1:A:B?A>"), ("long", "2:A:B;B:C?A>"), ("last", "1:D:E?D>"))
    expected, sequential = generate(model, prompts, max_new_tokens=2)
    actual, batched = generate_batched(model, prompts, max_new_tokens=2)
    assert actual == expected
    assert batched["forward_calls"] <= sequential["forward_calls"]
    assert batched["read_invocations"] == 4 * batched["forward_calls"]
