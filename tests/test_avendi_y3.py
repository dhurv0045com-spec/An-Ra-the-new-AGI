from itertools import product

import numpy as np
import pytest

from experiments.avendi_y3.plan import build_plan
from experiments.avendi_y3.tasks import law_bank, lookup_cube, public_action_values, update_rule_belief


def test_campaign_denominators():
    plan = build_plan()
    jobs = plan["training_jobs"]
    ids = {j["job_id"] for j in jobs}
    assert len(jobs) == len(ids) == 54
    assert len(plan["frozen_lineages"]) == 6
    assert set(plan["reused_scale_controls"]) <= ids
    assert [s["study_id"] for s in plan["studies"]] == [f"U{i:02}" for i in range(1, 51)]
    assert all(j["status"] == "PLANNED" for j in jobs)


def test_lookup_and_positional_shortcut_on_the_same_cube():
    trials = list(lookup_cube())
    assert len({(rows, key) for rows, key, _ in trials}) == 8
    assert sum(dict(rows)[key] == gold for rows, key, gold in trials) == 8
    assert sum(rows[-1][1] == gold for rows, _, gold in trials) == 4
    assert sum(rows[0][1] == gold for rows, _, gold in trials) == 4
    with pytest.raises(ValueError):
        list(lookup_cube(("A", "A")))


def test_exact_version_space_and_unseen_transfer():
    laws, tensor = law_bank()
    assert set(map(tuple, laws)) == set(product(range(4), repeat=4))
    np.testing.assert_array_equal(tensor.argmax(axis=2), laws)
    assert tensor.sum(axis=2).min() == tensor.sum(axis=2).max() == 1
    pair = laws[[0, 5]]  # 0000 and 0011 agree at 0, differ at both 2 and 3
    prior = np.array([0.5, 0.5])
    np.testing.assert_allclose(update_rule_belief(prior, pair, 0, 0), prior)
    for true_law in (0, 1):
        posterior = update_rule_belief(prior, pair, 2, pair[true_law, 2])
        assert posterior[pair[:, 3] == pair[true_law, 3]].sum() == 1.0
    with pytest.raises(ValueError, match="impossible"):
        update_rule_belief(prior, pair, 0, 3)


def test_public_bellman_against_enumerated_one_tool_policy_trees():
    rewards = np.array([[1, -2], [-2, 1], [0, 0]])
    prior = np.array([0.5, 0.5])
    observations = np.array([np.eye(2), [[0.5, 0.5], [0.5, 0.5]]])
    costs = [0.1, 0.05]
    values = public_action_values(prior, rewards, observations, costs)
    direct = rewards @ prior
    for t, cost in enumerate(costs):
        tree_values = [
            sum(prior[s] * observations[t, s, o] * rewards[policy[o], s]
                for s, o in product(range(2), repeat=2)) - cost
            for policy in product(range(3), repeat=2)
        ]
        assert values[len(direct) + t] == pytest.approx(max(tree_values))
    np.testing.assert_allclose(values, [-0.5, -0.5, 0, 0.9, -0.05])


def test_two_tool_horizon_and_consumption():
    states = list(product(range(2), repeat=2))
    truth = np.array([a ^ b for a, b in states])
    rewards = np.stack((np.where(truth == 0, 1, -2), np.where(truth == 1, 1, -2), np.zeros(4)))
    signals = np.eye(2)[np.array(states).T]
    for horizon, expected in [(0, -np.inf), (1, -0.1), (2, 0.8), (3, 0.8)]:
        values = public_action_values([0.25] * 4, rewards, signals, [0.1, 0.1], horizon)
        np.testing.assert_allclose(values[-2:], [expected, expected])
