"""Policies (observed-only), the isolated oracle, and canonical metrics.

Policy interface: choose(TaskInstance -> ObservedFailureFeatures visible
only) -> intervention name. The oracle consumes the latent truth and is
type-excluded from the policy interface: ``PolicyDecision`` refuses an
oracle argument, so no composed "oracle-assisted" policy can pass review.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from x_factor.contracts import NO_CHANGE, REGISTRY, ObservedFailureFeatures
from x_factor.world import TaskInstance


class Policy(ABC):
    name: str = "policy"

    @abstractmethod
    def choose(self, task: TaskInstance) -> str: ...


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    policy: str
    task_id: str
    intervention: str
    observed: ObservedFailureFeatures  # provenance: what the policy saw

    def __post_init__(self) -> None:
        if self.intervention not in REGISTRY:
            raise ValueError(f"decision references unknown intervention {self.intervention}")


class Oracle:
    """Evaluator-side upper bound. Deliberately NOT a Policy subclass."""

    name = "ORACLE"

    @staticmethod
    def choose(task: TaskInstance) -> str:
        best = min(
            (name for name, it in REGISTRY.items() if it.supplies >= task.required),
            key=lambda n: (REGISTRY[n].cost, n),
        )
        return best


class AlwaysOne(Policy):
    def __init__(self, intervention: str) -> None:
        if intervention not in REGISTRY:
            raise ValueError("unknown fixed intervention")
        self.intervention = intervention
        self.name = f"ALWAYS_{intervention}"

    def choose(self, task: TaskInstance) -> str:
        return self.intervention


class RandomPolicy(Policy):
    name = "RANDOM"

    def __init__(self, seed: int = 0) -> None:
        import random
        self._rng = random.Random(seed)

    def choose(self, task: TaskInstance) -> str:
        return self._rng.choice(sorted(REGISTRY))


class FamilyShortcut(Policy):
    """NEGATIVE CONTROL: memorizes family -> intervention from training
    outcomes. Must dominate in-domain and collapse cross-family."""
    name = "FAMILY_SHORTCUT"

    def __init__(self, table: dict[str, str]) -> None:
        self.table = dict(table)

    def choose(self, task: TaskInstance) -> str:
        return self.table.get(task.family, NO_CHANGE)


class LearnedFingerprintPolicy(Policy):
    """Observed-only learner: per-intervention logistic heads over the legal
    feature vector; decision is cost-adjusted argmax of predicted repair
    probability. Sees nothing evaluator-side (enforced by the features type).
    """

    name = "LEARNED_FINGERPRINT"

    def __init__(self, weights: dict[str, list[float]], bias: dict[str, float],
                 cost_weight: float = 0.0) -> None:
        self.weights = weights
        self.bias = bias
        self.cost_weight = cost_weight

    def predict(self, features: ObservedFailureFeatures, intervention: str) -> float:
        x = features.vector()
        z = self.bias[intervention] + sum(w * xi for w, xi in zip(self.weights[intervention], x))
        return 1.0 / (1.0 + math.exp(-z))

    def choose(self, task: TaskInstance) -> str:
        best, best_value = NO_CHANGE, -1e18
        for name, it in REGISTRY.items():
            value = self.predict(task.features, name) - self.cost_weight * it.cost
            if value > best_value:
                best, best_value = name, value
        return best


def train_fingerprint(train: list[TaskInstance], *, epochs: int = 300,
                      lr: float = 0.5, cost_weight: float = 0.05,
                      seed: int = 0) -> LearnedFingerprintPolicy:
    """Pure-python logistic training on OBSERVED features only. The label is
    the intervention's own outcome on the training task — outcomes of
    TRAINING tasks are legal (that is the intervention evidence), outcomes
    of evaluation tasks are not."""
    import random

    rng = random.Random(seed)
    names = sorted(REGISTRY)
    weights = {n: [0.0] * 7 for n in names}
    bias = {n: 0.0 for n in names}
    rows = []
    for t in train:
        for n in names:
            rows.append((t, n, 1.0 if t.outcome(n).repaired else 0.0))
    for _ in range(epochs):
        rng.shuffle(rows)
        for t, n, y in rows:
            p = 1.0 / (1.0 + math.exp(-(
                bias[n] + sum(w * x for w, x in zip(weights[n], t.features.vector())))))
            g = (p - y) * 0.05  # dampen per-example step for stability
            bias[n] -= lr * g
            weights[n] = [w - lr * g * x for w, x in zip(weights[n], t.features.vector())]
    return LearnedFingerprintPolicy(weights, bias, cost_weight=cost_weight)


# ---------------------------------------------------------------------------
# Canonical metrics (small, fixed set).
# ---------------------------------------------------------------------------

def evaluate(policy: Policy, tasks: list[TaskInstance]) -> dict[str, float]:
    decisions = [PolicyDecision(policy.name, t.task_id, policy.choose(t), t.features)
                 for t in tasks]
    repairs = [t.outcome(d.intervention).repaired for t, d in zip(tasks, decisions)]
    oracle = [Oracle.choose(t) for t in tasks]
    oracle_repairs = [t.outcome(o).repaired for t, o in zip(tasks, oracle)]
    n = len(tasks)
    top1 = sum(repairs) / n
    oracle_rate = sum(oracle_repairs) / n
    # Pairwise ranking accuracy over cost-ordered candidate pairs.
    pair_hits = pair_total = 0
    for t, d in zip(tasks, decisions):
        chosen = REGISTRY[d.intervention]
        better_exists = any(
            other.supplies >= t.required and not chosen.supplies >= t.required
            and REGISTRY[other_name].cost <= chosen.cost
            for other_name, other in REGISTRY.items())
        pair_total += 1
        pair_hits += 0 if better_exists else 1
    costs = [REGISTRY[d.intervention].cost for d in decisions]
    return {
        "top1_repair_accuracy": round(top1, 4),
        "oracle_repair_rate": round(oracle_rate, 4),
        "regret_vs_oracle": round(oracle_rate - top1, 4),
        "ranking_accuracy": round(pair_hits / pair_total, 4) if pair_total else 1.0,
        "mean_cost": round(sum(costs) / n, 3),
        "cost_adjusted_score": round(top1 - 0.02 * (sum(costs) / n), 4),
    }
