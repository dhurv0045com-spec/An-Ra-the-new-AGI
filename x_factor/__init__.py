"""X FACTOR: Causal Cognitive Self-Modeling.

Thesis: a failed execution's observed state predicts which cognitive
intervention repairs it, because failures carry low-rank latent causal
structure. The learned representation must predict FUTURE intervention
outcomes better than chance, fixed policies, confidence, and family
identity — and the family shortcut must fail cross-family by design.

See SPEC.md for the canonical specification and the falsification law.
"""

from x_factor.contracts import (
    FORBIDDEN_FEATURES,
    REGISTRY,
    assert_observation_legality,
)
from x_factor.ladder import build_ladder

__all__ = ["REGISTRY", "FORBIDDEN_FEATURES", "assert_observation_legality", "build_ladder"]
