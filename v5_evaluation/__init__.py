"""V5 evaluation: adapter contract, fixtures, metrics, statistics, receipts."""

from .adapter import ADAPTER_SCHEMA, ModelAdapter
from .fixture import FIXTURE_SCHEMA, TaskFixtureBatch
from .metrics import (
    METRIC_REGISTRY,
    accuracy,
    balanced_accuracy,
    candidate_margin,
    candidate_rank1,
    conditional_realization,
    exact_accuracy,
    gold_suffix_nll,
    invariance_stability,
    loss_regression,
    sensitivity_flip_rate,
    wilson_lcb,
)
from .protocol import (
    EvaluationProtocol,
    EvaluationReceipt as ProtocolEvaluationReceipt,
    TaskLevelEvidence,
    run_evaluation,
    verify_evidence_artifact,
    write_evidence_artifact,
)
from .receipt import RECEIPT_SCHEMA, EvaluationReceipt
from .stats import STATISTICAL_RULES, cluster_bootstrap_delta, exact_mcnemar, wilson_binomial

__all__ = [
    "ADAPTER_SCHEMA",
    "FIXTURE_SCHEMA",
    "METRIC_REGISTRY",
    "RECEIPT_SCHEMA",
    "STATISTICAL_RULES",
    "EvaluationProtocol",
    "EvaluationReceipt",
    "ModelAdapter",
    "ProtocolEvaluationReceipt",
    "TaskFixtureBatch",
    "TaskLevelEvidence",
    "accuracy",
    "balanced_accuracy",
    "candidate_margin",
    "candidate_rank1",
    "cluster_bootstrap_delta",
    "conditional_realization",
    "exact_accuracy",
    "exact_mcnemar",
    "gold_suffix_nll",
    "invariance_stability",
    "loss_regression",
    "run_evaluation",
    "sensitivity_flip_rate",
    "verify_evidence_artifact",
    "wilson_binomial",
    "wilson_lcb",
    "write_evidence_artifact",
]
