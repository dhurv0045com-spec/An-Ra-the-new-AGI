from dataclasses import replace

import numpy as np
import pytest
import torch

from bramastra_lab.discovery.curriculum import build_demonstrations, teacher_gains
from bramastra_lab.discovery.evaluation import evaluate
from bramastra_lab.discovery.learner import Investigator, LearnerConfig
from bramastra_lab.discovery.training import (
    TrainConfig,
    Trainer,
    continuation_probe,
    fingerprint,
)
from bramastra_lab.discovery.worlds import RuleWorld, information_gains, inputs, make_worlds


@pytest.fixture(scope="module")
def small_worlds():
    return make_worlds(bits=3)


@pytest.fixture
def model():
    torch.manual_seed(41)
    return Investigator(LearnerConfig(bits=3, width=16))


def demonstrations(small_worlds, *, seed=7):
    return build_demonstrations(small_worlds, episodes=8, budget=2, seed=seed)


def test_encoding_ignores_padding_and_heads_depend_on_context_and_target(model):
    history = torch.tensor(
        [[[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    altered_padding = history.clone()
    altered_padding[:, 1:] = torch.tensor(
        [[[19.0, -7.0, 5.0, 3.0], [-11.0, 2.0, 8.0, -4.0]]]
    )
    state = model.encode(history, torch.tensor([1]))
    assert torch.equal(state, model.encode(altered_padding, torch.tensor([1])))

    state = state.detach().requires_grad_()
    target = torch.tensor([[0.0, 1.0, 0.0]], requires_grad=True)
    outcome = model.predict(state, target).sum()
    state_gradient, target_gradient = torch.autograd.grad(outcome, (state, target))
    assert state_gradient.abs().sum() > 0
    assert target_gradient.abs().sum() > 0

    state = state.detach().requires_grad_()
    target = target.detach().requires_grad_()
    candidates = torch.tensor(inputs(3), dtype=torch.float32)
    policy = model.select(state, target, candidates, torch.ones(1, 8, dtype=torch.bool)).sum()
    state_gradient, target_gradient = torch.autograd.grad(policy, (state, target))
    assert state_gradient.abs().sum() > 0
    assert target_gradient.abs().sum() > 0


def test_vectorized_teacher_matches_independent_exact_teacher(small_worlds):
    subset = small_worlds[:11]
    table = np.asarray([world.table for world in subset], dtype=np.uint8)
    target = 5
    legal = np.ones(8, dtype=bool)
    legal[[target, 2]] = False
    actual = teacher_gains(table, target, legal)
    expected = information_gains(subset, [], target)
    for action in range(8):
        wanted = expected.get(action, 0.0) if legal[action] else 0.0
        assert actual[action] == pytest.approx(wanted, abs=2e-6)


def test_one_step_teacher_exposes_preparatory_parity_limitation():
    hypotheses = [
        RuleWorld("00", "parity-diagnostic", 2, (0, 0, 0, 0)),
        RuleWorld("10", "parity-diagnostic", 2, (0, 1, 0, 1)),
        RuleWorld("01", "parity-diagnostic", 2, (0, 0, 1, 1)),
        RuleWorld("11", "parity-diagnostic", 2, (0, 1, 1, 0)),
    ]
    table = np.asarray([world.table for world in hypotheses], dtype=np.uint8)
    legal = np.asarray([False, True, True, False])
    gains = teacher_gains(table, target=3, legal=legal)
    assert gains[1] == pytest.approx(0.0, abs=1e-7)
    assert gains[2] == pytest.approx(0.0, abs=1e-7)
    assert information_gains(hypotheses, [], target=3)[1] == pytest.approx(0.0)
    # After either unrewarded preparatory query, the other query is worth one bit.
    assert information_gains(hypotheses, [(1, 0)], target=3)[2] == pytest.approx(1.0)


def test_every_policy_avoids_target_and_repeated_queries(model, small_worlds):
    policies = ("random", "coverage", "learned", "uncertainty", "no_memory")
    rows = evaluate(
        model,
        small_worlds[:5],
        policies=policies,
        budgets=(0, 1, 2),
        targets_per_world=2,
        seed=29,
    )
    assert {row["policy"] for row in rows} == set(policies)
    for row in rows:
        assert len(row["actions"]) == row["queries"] == row["budget"]
        assert row["target"] not in row["actions"]
        assert len(row["actions"]) == len(set(row["actions"]))


def test_training_mutates_parameters_and_checkpoint_continues_exactly(
    model, small_worlds, tmp_path
):
    data = demonstrations(small_worlds)
    trainer = Trainer(
        model,
        data,
        TrainConfig(steps=2, batch_size=8, learning_rate=0.001, seed=17),
    )
    before = fingerprint(model)
    metrics = [trainer.update() for _ in range(2)]
    assert fingerprint(model) != before
    assert trainer.step == 2
    assert all(np.isfinite(item["loss"]) and item["grad_norm"] > 0 for item in metrics)

    checkpoint = tmp_path / "continuation.pt"
    result = continuation_probe(trainer, checkpoint)
    assert result["status"] == "PASS"
    assert result["next_step"] == 3

    mismatched = demonstrations(small_worlds, seed=8)
    with pytest.raises(ValueError, match="mismatch"):
        Trainer.restore(checkpoint, mismatched)

    # Even tensor-identical data must be rejected if its declared identity differs.
    mismatched_manifest = replace(data, manifest={**data.manifest, "seed": -1})
    with pytest.raises(ValueError, match="mismatch"):
        Trainer.restore(checkpoint, mismatched_manifest)

    changed = data.observations.clone()
    changed[0, 0, 0] = 1 - changed[0, 0, 0]
    changed_data = replace(data, observations=changed)
    with pytest.raises(ValueError, match="tensor/manifest mismatch"):
        Trainer.restore(checkpoint, changed_data)
