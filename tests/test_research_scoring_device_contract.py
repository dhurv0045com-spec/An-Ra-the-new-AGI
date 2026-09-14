"""Focused device and validation contracts for the differentiable scorers."""

import pytest
import torch

from bramastra_lab.research.config import BuildConfig, seed_everything
from bramastra_lab.research.learning.k8_scoring import (
    ScoringError,
    score_candidates_trainable,
    value_estimate_trainable,
    world_transition_token_loss,
)
from bramastra_lab.research.models import IntegratedModel


@pytest.fixture
def tiny_model():
    seed_everything(314)
    config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
    return config, IntegratedModel(config)


def test_candidate_scoring_uses_parameter_device_and_backward(tiny_model):
    config, model = tiny_model
    seen = {}
    original = model.forward_hidden

    def recording_forward(tokens, padding=None, *args, **kwargs):
        seen["tokens"] = tokens
        seen["padding"] = padding
        return original(tokens, padding, *args, **kwargs)

    model.forward_hidden = recording_forward
    result = score_candidates_trainable(
        model, config, [1, 2], [[70], [80]],
        legal_mask=torch.tensor([True, False]))
    assert seen["tokens"].device == next(model.parameters()).device
    assert seen["padding"].device == next(model.parameters()).device
    assert result["legal_mask"].device == next(model.parameters()).device
    result["log_probs"].sum().backward()
    assert model.action_head.weight.grad is not None


def test_value_and_world_inputs_are_on_parameter_device_and_backward(tiny_model):
    config, model = tiny_model
    seen = []
    handle = model.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].device))
    try:
        value = value_estimate_trainable(model, config, [1, 2])
        loss = world_transition_token_loss(
            model, config, [1, 2], action={"kind": "press"},
            target_feedback={"result": "ok"})
    finally:
        handle.remove()
    parameter_device = next(model.parameters()).device
    assert seen and all(device == parameter_device for device in seen)
    (value + loss).backward()
    assert model.value_head.weight.grad is not None


def test_candidate_mask_shape_and_all_illegal_fail_before_model_call(tiny_model):
    config, model = tiny_model

    def fail_if_called(*args, **kwargs):
        raise AssertionError("model must not run for an invalid legal mask")

    model.forward_hidden = fail_if_called
    with pytest.raises(ScoringError, match="shape mismatch"):
        score_candidates_trainable(
            model, config, [1], [[70], [80]],
            legal_mask=torch.ones(2, 1, dtype=torch.bool))
    with pytest.raises(ScoringError, match="at least one legal"):
        score_candidates_trainable(
            model, config, [1], [[70], [80]],
            legal_mask=torch.zeros(2, dtype=torch.bool))


def test_value_context_is_rejected_before_model_call(tiny_model):
    config, model = tiny_model
    model.forward = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("model must not run for an overlong value prefix"))
    with pytest.raises(ScoringError, match="value prefix exceeds"):
        value_estimate_trainable(model, config, [1] * (config.model.max_seq + 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="owner CUDA hardware not available")
def test_candidate_scoring_cuda_device_and_backward():
    config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
    model = IntegratedModel(config).cuda()
    result = score_candidates_trainable(
        model, config, [1, 2], [[70], [80]],
        legal_mask=torch.tensor([True, False]))
    assert result["scores"].device.type == "cuda"
    result["log_probs"].sum().backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="owner CUDA hardware not available")
def test_value_and_world_cuda_device_and_backward():
    config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
    model = IntegratedModel(config).cuda()
    value = value_estimate_trainable(model, config, [1, 2])
    loss = world_transition_token_loss(
        model, config, [1, 2], action={"kind": "press"},
        target_feedback={"result": "ok"})
    assert value.device.type == "cuda"
    assert loss.device.type == "cuda"
    (value + loss).backward()
