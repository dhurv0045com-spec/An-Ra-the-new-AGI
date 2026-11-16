"""ELENDRIL V8 mechanism discovery and cumulative learning package.

Importing this package performs no CUDA work, dataset generation, or training.
"""

from .contracts import (
    ACTION_19_LABELS,
    ARCHITECTURES,
    OBSERVATION_VERSION,
    OP_VOCABULARY,
    POLICY_MODES,
    SYMBOLS,
    VERIFIER_IDENTITY,
    EvaluationOutcome,
    PolicyProposal,
    PublicInput,
    V8ModelSpec,
    build_v8_model_spec,
    choose_fcap_bottleneck,
)
from .protocol import (
    ADVANCEMENT_RULES,
    EVALUATION_ALLOCATION,
    OPTIMIZER_DEFAULTS,
    PHASES,
    PROTOCOL_SHA,
    REPLICATION_SEEDS,
    SCENARIO_LEDGER,
    SEEDS,
    jobs,
    protocol,
    validate_job,
)

__all__ = [
    "ACTION_19_LABELS",
    "ADVANCEMENT_RULES",
    "ARCHITECTURES",
    "EVALUATION_ALLOCATION",
    "EvaluationOutcome",
    "OBSERVATION_VERSION",
    "OPTIMIZER_DEFAULTS",
    "OP_VOCABULARY",
    "PHASES",
    "POLICY_MODES",
    "PROTOCOL_SHA",
    "PolicyProposal",
    "PublicInput",
    "REPLICATION_SEEDS",
    "SCENARIO_LEDGER",
    "SEEDS",
    "SYMBOLS",
    "V8ModelSpec",
    "VERIFIER_IDENTITY",
    "build_v8_model_spec",
    "choose_fcap_bottleneck",
    "jobs",
    "protocol",
    "validate_job",
]
