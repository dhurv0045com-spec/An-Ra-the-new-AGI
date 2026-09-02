"""Deterministic latent-factor world: the benchmark's causal physics.

Task instances require a subset of latent factors. A baseline attempt fails
iff at least one required factor is unsatisfied. Intervention j repairs the
task iff it supplies every missing required factor. Under this physics the
task x intervention outcome matrix is low-rank (rank <= |factors| + 1):
that is the structure X0 tests for, and the structure the learner must
exploit WITHOUT being told the factor names.

The negative control is structural: surface family (template) is sampled
INDEPENDENTLY of the required-factor set, so a family-ID shortcut policy
wins on in-family development data and collapses cross-family. A learned
observed-feature policy cannot rely on family and must read the observed
factor-gap signals.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from x_factor.contracts import (
    FACTORS,
    REGISTRY,
    NO_CHANGE,
    ObservedFailureFeatures,
)
FACTORS_SORTED = sorted(FACTORS)
FAMILIES = ("ledger", "gazetteer", "telemetry", "manifest")


@dataclass(frozen=True, slots=True)
class TaskInstance:
    task_id: str
    family: str                      # evaluator-side only (shortcut control)
    required: frozenset[str]         # evaluator-side only (latent truth)
    features: ObservedFailureFeatures  # policy-visible

    def outcome(self, intervention: str) -> "InterventionOutcomeLite":
        it = REGISTRY[intervention]
        repaired = it.supplies >= self.required
        # Effect magnitude: number of missing factors removed (signed, and
        # NO_CHANGE on a failing task is always 0).
        missing = len(self.required - it.supplies)
        effect = 0.0 if intervention == NO_CHANGE else float((it.supplies >= self.required)) - 0.25 * missing
        from x_factor.contracts import InterventionOutcome
        return InterventionOutcome(self.task_id, intervention, repaired, effect)


def make_split(seed: int, n_tasks: int, *, split: str) -> list[TaskInstance]:
    rng = random.Random(seed)
    tasks = []
    for i in range(n_tasks):
        family = FAMILIES[rng.randrange(len(FAMILIES))]  # independent of truth
        k = rng.choice((1, 1, 2, 2, 3))
        required = frozenset(rng.sample(FACTORS_SORTED, k))
        # Observed gaps: monotone, noisy evidence of which factors are missing.
        gaps = {}
        for f in FACTORS_SORTED:
            missing = f in required
            gaps[f] = (0.55 + 0.45 * rng.random()) if missing else (0.45 * rng.random())
        features = ObservedFailureFeatures(
            task_id=f"{split}-{i:05d}",
            observed_retrieval_gap=gaps["retrieve"],
            observed_binding_gap=gaps["bind"],
            observed_composition_gap=gaps["compose"],
            n_candidates=rng.randrange(2, 6),
            output_arity=rng.randrange(1, 4),
            format_code=rng.randrange(3),
            confidence_signal=round(0.5 - 0.4 * (len(required) - 1) + 0.1 * rng.random(), 3),
        )
        tasks.append(TaskInstance(f"{split}-{i:05d}", family, required, features))
    return tasks


def outcome_matrix(tasks: list[TaskInstance]) -> dict[str, dict[str, bool]]:
    """tasks x interventions repair matrix — the X0 object whose rank
    structure exposes latent factors."""
    return {
        t.task_id: {name: t.outcome(name).repaired for name in REGISTRY}
        for t in tasks
    }
