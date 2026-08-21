"""Cognitive credit assignment through controlled interventions.

Connector-owned research experiment. Core is treated as an opaque
``f(context, decode) -> tokens`` function; this package plants controlled
failures, generates interventions from *observed* information only, and
measures which single-variable change flips the verifier.

Structural no-leakage contract
------------------------------
``HiddenGroundTruth`` (planted cause, gold solution, gold knowledge, gold
plan) lives in a separate frozen dataclass from ``ObservedCase``. The
intervention generator's signature accepts only ``ObservedCase``; the hidden
evaluator is the only component that reads ``HiddenGroundTruth``. A focused
test proves that permuting the hidden label while holding the observed case
fixed cannot change the generated intervention set.
"""

from connector.experiments.cognitive_credit.case import (
    Attempt,
    DecodePolicy,
    HiddenGroundTruth,
    ObservedCase,
    ToolBehavior,
)
from connector.experiments.cognitive_credit.diagnose import (
    Diagnosis,
    classify_from_outcomes,
)
from connector.experiments.cognitive_credit.interventions import (
    InterventionSpec,
    build_interventions,
)

__all__ = [
    "Attempt",
    "DecodePolicy",
    "Diagnosis",
    "HiddenGroundTruth",
    "InterventionSpec",
    "ObservedCase",
    "ToolBehavior",
    "build_interventions",
    "classify_from_outcomes",
]
