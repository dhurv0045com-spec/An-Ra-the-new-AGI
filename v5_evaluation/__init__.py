"""V5 evaluation: adapter contract, pure metrics, bound receipts."""

from .adapter import ADAPTER_SCHEMA, ModelAdapter
from .metrics import (
    accuracy,
    conditional_realization,
    invariance_stability,
    loss_regression,
    sensitivity_flip_rate,
    wilson_lcb,
)
from .receipt import RECEIPT_SCHEMA, EvaluationReceipt

__all__ = [
    "ADAPTER_SCHEMA",
    "RECEIPT_SCHEMA",
    "EvaluationReceipt",
    "ModelAdapter",
    "accuracy",
    "conditional_realization",
    "invariance_stability",
    "loss_regression",
    "sensitivity_flip_rate",
    "wilson_lcb",
]
