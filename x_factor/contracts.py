"""Causal Cognitive Self-Modeling: contracts, leakage law, intervention registry.

Central hypothesis (SPEC.md): a failed execution's OBSERVED state carries
enough information to predict which cognitive intervention will repair it,
because failures have low-rank latent causal structure that interventions
address factor-wise.

Leakage law, enforced in code: a policy may observe only
``ObservedFailureFeatures``. Correctness labels, gold answers, hidden
required-factor sets, family identity, and any intervention outcome are
evaluator-side only. ``assert_observation_legality`` rejects any feature
record containing forbidden fields, and the policy interface accepts only
the legal type — a shortcut cannot smuggle the answer through the type
system without a greppable bypass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Intervention registry. Semantics: each intervention supplies a set of
# latent cognitive factors. NO_CHANGE supplies nothing and is always present
# (the do-nothing control that makes every claim counterfactual).
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Intervention:
    name: str
    supplies: frozenset[str]
    cost: int


REGISTRY: dict[str, Intervention] = {
    it.name: it
    for it in (
        Intervention("NO_CHANGE", frozenset(), 0),
        Intervention("RETRIEVAL_HELP", frozenset({"retrieve"}), 1),
        Intervention("BINDING_SUPPORT", frozenset({"bind"}), 1),
        Intervention("DECOMPOSITION", frozenset({"compose"}), 2),
        Intervention("FULL_REPLAY", frozenset({"retrieve", "bind", "compose"}), 4),
    )
}
NO_CHANGE = "NO_CHANGE"

# The latent factor alphabet. NOT a failure taxonomy: task generators sample
# requirements from this alphabet, and the learned representation is free to
# discover any structure over it (or over outcomes directly).
FACTORS = frozenset({"retrieve", "bind", "compose"})

# ---------------------------------------------------------------------------
# Leakage law.
# ---------------------------------------------------------------------------

ALLOWED_FEATURES = frozenset({
    "observed_retrieval_gap", "observed_binding_gap", "observed_composition_gap",
    "n_candidates", "output_arity", "format_code", "confidence_signal",
})
FORBIDDEN_FEATURES = frozenset({
    "family_id", "family", "required_factors", "gold", "gold_answer",
    "gold_rank", "correct", "correctness", "outcome", "intervention_outcome",
    "hidden_label", "task_family", "difficulty_label", "template_id",
})


@dataclass(frozen=True, slots=True)
class ObservedFailureFeatures:
    """Everything an intervention policy may see BEFORE outcomes. Field-level
    legality: adding a forbidden attribute here is a constructor-time error,
    not a silent leak."""

    task_id: str
    observed_retrieval_gap: float
    observed_binding_gap: float
    observed_composition_gap: float
    n_candidates: int
    output_arity: int
    format_code: int
    confidence_signal: float

    def __post_init__(self) -> None:
        illegal = FORBIDDEN_FEATURES & {f for f in self.__dataclass_fields__}
        if illegal:
            raise ValueError(f"observed features contain forbidden fields: {sorted(illegal)}")

    def vector(self) -> list[float]:
        return [self.observed_retrieval_gap, self.observed_binding_gap,
                self.observed_composition_gap, float(self.n_candidates),
                float(self.output_arity), float(self.format_code),
                self.confidence_signal]


def assert_observation_legality(record: dict) -> None:
    """Schema-level audit for serialized evidence: any forbidden key anywhere
    in the record (including nested evaluator blocks) fails the audit."""
    keys = {str(k).lower() for k in _walk_keys(record)}
    leaked = FORBIDDEN_FEATURES & keys
    if leaked:
        raise ValueError(f"observation legality violated: {sorted(leaked)}")


def _walk_keys(record):
    if isinstance(record, dict):
        for key, value in record.items():
            yield key
            yield from _walk_keys(value)
    elif isinstance(record, (list, tuple)):
        for item in record:
            yield from _walk_keys(item)


# ---------------------------------------------------------------------------
# Outcome record: evaluator-side only. Never handed to a policy.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class InterventionOutcome:
    task_id: str
    intervention: str
    repaired: bool
    effect: float  # verifier-defined delta; sign and magnitude matter

    def __post_init__(self) -> None:
        if self.intervention not in REGISTRY:
            raise ValueError(f"unknown intervention {self.intervention}")
