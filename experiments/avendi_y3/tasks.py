"""Exact finite references for Avendi Y3 lookup, public feedback and tool choice."""

from itertools import product

import numpy as np


def lookup_cube(keys=("A", "B"), values=("C", "D")):
    """Yield public rows, requested key and correct answer for the eight interventions."""
    if len(keys) != 2 or len(values) != 2 or len(set((*keys, *values))) != 4:
        raise ValueError("expected two disjoint keys and two distinct values")
    for swap, reverse_order, query in product(range(2), repeat=3):
        assignment = values[::-1] if swap else values
        rows = tuple(zip(keys, assignment))
        yield (rows[::-1] if reverse_order else rows), keys[query], assignment[query]


def law_bank():
    """Return the 256 maps on four symbols and their one-hot transition tensors."""
    laws = np.indices((4, 4, 4, 4), dtype=np.uint8).reshape(4, -1).T
    return laws, np.eye(4, dtype=bool)[laws]


def update_rule_belief(prior, laws, input_symbol, observed_symbol, reliability=1.0):
    """Update a version space using symmetric observation noise over the four outputs."""
    laws = np.asarray(laws)
    if laws.ndim != 2 or laws.shape[1] != 4 or not np.isin(laws, range(4)).all():
        raise ValueError("invalid law table")
    if input_symbol not in range(4) or observed_symbol not in range(4):
        raise ValueError("symbol outside the four-symbol alphabet")
    if not 0 <= reliability <= 1:
        raise ValueError("invalid feedback reliability")
    belief = _belief(prior, len(laws))
    matches = laws[:, input_symbol] == observed_symbol
    likelihood = np.where(matches, reliability, (1.0 - reliability) / 3)
    mass = belief * likelihood
    if mass.sum() == 0:
        raise ValueError("observation impossible under the public model")
    return mass / mass.sum()


def _belief(prior, states):
    belief = np.asarray(prior, dtype=float)
    if belief.shape != (states,) or not np.isfinite(belief).all() or (belief < 0).any():
        raise ValueError("invalid public belief")
    if not np.isclose(belief.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("public belief must sum to one")
    return belief


def public_action_values(prior, terminal_rewards, observation_likelihood, costs, horizon=1):
    """Exact static-world Bellman values for terminal actions and consumable tools.

    Rewards have axes [terminal action, hidden state]. Likelihood has axes
    [tool, hidden state, observation]. Output order is terminal actions then tools.
    Impossible observations contribute zero. Illegal tools have value -infinity.
    """
    reward = np.asarray(terminal_rewards, dtype=float)
    likelihood = np.asarray(observation_likelihood, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if reward.ndim != 2 or min(reward.shape) == 0 or not np.isfinite(reward).all():
        raise ValueError("invalid terminal rewards")
    belief = _belief(prior, reward.shape[1])
    if likelihood.ndim != 3 or likelihood.shape[1] != len(belief) or likelihood.shape[2] == 0:
        raise ValueError("invalid tool observation axes")
    if not np.isfinite(likelihood).all() or (likelihood < 0).any():
        raise ValueError("invalid observation probabilities")
    if not np.allclose(likelihood.sum(axis=2), 1.0, rtol=0, atol=1e-12):
        raise ValueError("tool observation probabilities must sum to one")
    n_tools = len(likelihood)
    if costs.shape != (n_tools,) or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("invalid tool costs")
    if not isinstance(horizon, int) or horizon < 0:
        raise ValueError("invalid action horizon")

    def evaluate(b, available, remaining):
        terminal = reward @ b
        tool_values = np.full(n_tools, -np.inf)
        if remaining:
            for tool in available:
                joint = likelihood[tool].T * b
                masses = joint.sum(axis=1)
                next_tools = tuple(t for t in available if t != tool)
                future = sum(
                    p * evaluate(row / p, next_tools, remaining - 1).max()
                    for row, p in zip(joint, masses) if p > 0
                )
                tool_values[tool] = future - costs[tool]
        return np.concatenate((terminal, tool_values))

    return evaluate(belief, tuple(range(n_tools)), min(horizon, n_tools))
