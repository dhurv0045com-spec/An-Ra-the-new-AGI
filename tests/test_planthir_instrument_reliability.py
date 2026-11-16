import random

import pytest
import torch

from anra_v5.planthir.contracts import UniversalObservation
from anra_v5.planthir.instrument import validate_environment_kernel, validate_model_and_gradients
from anra_v5.planthir.adapters.v8_adapter import V8EnvironmentKernel
from anra_v5.elendril.v8.data import build_d1_training_dataset


def test_gradient_probe_restores_state_even_on_exception():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Dropout(), torch.nn.Linear(3, 1))
    model.eval()
    model[1].train()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    grads = [p.grad for p in model.parameters()]
    buffers = [b.clone() for b in model.buffers()]
    modes = [m.training for m in model.modules()]
    torch_state, python_state = torch.get_rng_state(), random.getstate()

    def loss(m):
        random.random()
        output = m(torch.randn(4, 3))
        raise RuntimeError("probe failure")

    with pytest.raises(RuntimeError, match="probe failure"):
        validate_model_and_gradients(model, loss)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    assert [m.training for m in model.modules()] == modes
    assert all(torch.equal(a, b) for a, b in zip(model.buffers(), buffers))
    assert all(p.grad is old and torch.all(old == 1) for p, old in zip(model.parameters(), grads))


def test_zero_gradient_is_not_a_passing_probe():
    model = torch.nn.Linear(1, 1)
    report = validate_model_and_gradients(model, lambda m: m(torch.ones(1, 1)).sum() * 0)
    assert not report["passed"]
    assert "no_nonzero_gradients" in report["defects"]


def test_nested_metadata_and_tool_types_checked():
    obs = UniversalObservation("id", "Action:", {"query": 1}, frozenset(), 0, 1, {"nested": [{"Expected": "A"}]})
    errors = obs.validate_invariants()
    assert "invalid_available_tools" in errors
    assert "forbidden_metadata_key:Expected" in errors


def test_requested_v8_parity_requires_eligible_world():
    worlds = build_d1_training_dataset(quota_per_family=16).worlds
    world = next(w for w in worlds if w["renderer_template"] != 0)
    report = validate_environment_kernel(V8EnvironmentKernel(), [world], check_v8_evaluator_parity=True)
    assert not report["passed"]
    assert not report["checks"]["v8_evaluator_parity"]["all_passed"]


def test_evaluator_unknown_action_terminates_as_invalid():
    from types import SimpleNamespace as PolicyProposal
    from anra_v5.elendril.v8.evaluation import run_episode

    class Actor:
        mode = "raw"
        model_identity = "test"
        def propose(self, obs):
            proposal = PolicyProposal(operation="BOGUS", argument=None, joint_log_probability=0.0,
                                      raw_output="BOGUS", model_calls=0, evaluated_positions=0,
                                      mode="raw", model_identity="test")
            return proposal, proposal

    world = build_d1_training_dataset(quota_per_family=16).worlds[0]
    result = run_episode(Actor(), world, max_actions=2)
    assert len(result.steps) == 1
    assert result.outcome.terminal_status == "INVALID_ACTION"
