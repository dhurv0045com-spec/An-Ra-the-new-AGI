"""Small differential/adversarial checks; no research training campaign."""

import pytest
import torch

from anra_v5.elendril.associative import AssociativeDecoder
from anra_v5.elendril.backend import BackendIdentity, NeuralBackend, set_attention_backend
from anra_v5.elendril.codec import BOS, CODEC_SHA, EOS, encode
from anra_v5.elendril.training import MODEL, configure_cpu, generate, logits_for
from v5_model.core import initialize


def adapter(model=None, spec=MODEL):
    configure_cpu()
    model = initialize(spec, 17) if model is None else model
    return NeuralBackend(model, BackendIdentity(spec.sha256(), CODEC_SHA,
                                               spec.vocabulary_size, spec.context_length))


def test_adapter_matches_reference_tensor_generation_and_counts():
    backend = adapter()
    samples = (("short", "A>"), ("long", "A:B;B:C?A>"))
    expected, counters = generate(backend.model, samples, max_new_tokens=3)
    actual = backend.generate(tuple((BOS, *encode(prompt)) for _, prompt in samples), max_new_tokens=3)
    assert [row.tokens for row in actual] == [row.tokens for row in expected]
    assert sum(row.forward_calls for row in actual) == counters["forward_calls"]
    assert sum(row.evaluated_positions for row in actual) == counters["evaluated_positions"]
    inventory = backend.parameter_inventory()
    assert inventory["total"] == MODEL.parameter_receipt().total
    assert list(name for name in inventory["parameters"] if "embedding.weight" in name) == ["embedding.weight"]


def test_adapter_accepts_associative_log_probabilities_and_preserves_mode():
    model = AssociativeDecoder(initialize(MODEL, 7), 7, mode="two-read").train()
    backend = adapter(model)
    model.core.blocks[0].eval()  # Preserve mixed module modes as well.
    modes = [module.training for module in model.modules()]
    tokens = torch.tensor([[BOS, *encode("A:B?A>")]])
    segments = torch.zeros_like(tokens)
    torch.testing.assert_close(backend.logits(tokens, segments), logits_for(model, tokens, segments), rtol=0, atol=0)
    scores = backend.next_distribution(tokens[0].tolist())
    assert torch.isfinite(scores).all() and abs(scores.sum().item() - 1) < 1e-6
    assert modes == [module.training for module in model.modules()]
    with pytest.raises(ValueError, match="not supported"):
        backend.logits(tokens, segments, activation_checkpointing=True)


def test_native_gqa_matches_packed_reference_forward_backward_and_checkpointing():
    configure_cpu()
    reference = adapter()
    native = adapter()
    assert set_attention_backend(native.model, "native-gqa") == MODEL.layers
    assert reference.model.state_dict().keys() == native.model.state_dict().keys()
    tokens = torch.tensor([[BOS, 7, 8, 9, BOS, 10, 11, 0],
                           [BOS, 12, 13, 0, 0, 0, 0, 0]])
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, -1], [0, 0, 0, -1, -1, -1, -1, -1]])
    expected = reference.logits(tokens, segments)
    actual = native.logits(tokens, segments, activation_checkpointing=True)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
    # A nonsymmetric target probes every logit and parameter path.
    weights = torch.linspace(-1, 1, expected.numel()).reshape_as(expected)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    for left, right in zip(reference.model.parameters(), native.model.parameters()):
        assert left.grad is not None and right.grad is not None
        assert torch.isfinite(right.grad).all()
        torch.testing.assert_close(left.grad, right.grad, rtol=3e-4, atol=2e-5)
    changed = tokens.clone()
    changed[0, 4:7] = 20
    torch.testing.assert_close(native.logits(changed, segments)[0, :4], actual[0, :4], rtol=0, atol=0)
    assert set_attention_backend(native.model, "repeat-sdpa") == MODEL.layers
    torch.testing.assert_close(native.logits(tokens, segments), expected, rtol=0, atol=0)


def test_invalid_layouts_vocabulary_nonfinite_and_context_rejected_before_generation():
    backend = adapter()
    tokens = torch.tensor([[BOS, 7, 8, 0]])
    for segments in (torch.tensor([[0, 0, -1, -1]]), torch.tensor([[0, 1, 0, -1]]),
                     torch.tensor([[-1, -1, -1, -1]])):
        with pytest.raises(ValueError):
            backend.logits(tokens, segments)
    with pytest.raises(ValueError, match="vocabulary"):
        backend.logits(torch.tensor([[999]]), torch.tensor([[0]]))
    with pytest.raises(ValueError, match="native context"):
        backend.generate(((BOS, *([3] * (MODEL.context_length - 1))),), max_new_tokens=2)
    with pytest.raises(ValueError):
        BackendIdentity("not-a-sha", CODEC_SHA, 44, 32)
    with pytest.raises(ValueError, match="dimensions"):
        NeuralBackend(backend.model, BackendIdentity(MODEL.sha256(), CODEC_SHA, MODEL.vocabulary_size + 1, 32))
    with torch.no_grad():
        backend.model.embedding.weight[7].fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        backend.next_distribution((BOS, 7))


def test_eos_and_invalid_special_are_raw_predictions_not_repaired():
    class Emitting(torch.nn.Module):
        config = type("Config", (), {"vocabulary_size": MODEL.vocabulary_size, "context_length": 32})()

        def __init__(self, token):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(1))
            self.token = token

        def forward(self, tokens, positions, mask):
            scores = torch.full((*tokens.shape, MODEL.vocabulary_size), -100.)
            scores[:, :, self.token] = 100.
            return scores

    for token, reason in ((EOS, "eos"), (0, "invalid-special"), (BOS, "invalid-special")):
        backend = adapter(Emitting(token))
        result, = backend.generate(((BOS, 7),), max_new_tokens=4)
        assert result.tokens == (token,)
        assert result.ended_eos == (token == EOS)
        assert result.stop_reason == reason
        assert result.forward_calls == 1 and result.evaluated_positions == 2


def test_native_context_limit_and_cpu_bfloat16_normalization_remain_finite():
    backend = adapter()
    tokens = torch.full((1, MODEL.context_length), 7, dtype=torch.long)
    tokens[:, 0] = BOS
    result = backend.logits(tokens, torch.zeros_like(tokens))
    assert result.shape[1] == MODEL.context_length
    with pytest.raises(ValueError, match="native context"):
        backend.logits(torch.cat((tokens, tokens[:, :1]), 1), torch.zeros(1, MODEL.context_length + 1, dtype=torch.long))
    # This qualifies CPU BF16 numerics only; no CUDA or AMP speed claim.
    backend.model.to(dtype=torch.bfloat16)
    set_attention_backend(backend.model, "native-gqa")
    bf16 = backend.logits(tokens, torch.zeros_like(tokens))
    assert bf16.dtype == torch.bfloat16 and torch.isfinite(bf16).all()
    bf16.float().square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in backend.model.parameters())
