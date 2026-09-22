"""Evidence-bounded 100M-class An-Ra research core contract."""

from .spec import (
    MODEL_SPEC,
    TPU_DEPTH_PRESERVING_CHALLENGER,
    TPU_TILED_CHALLENGER,
    candidate_receipts,
    parameter_receipt,
    resource_estimate,
)

__all__ = [
    "MODEL_SPEC", "TPU_TILED_CHALLENGER", "TPU_DEPTH_PRESERVING_CHALLENGER", "candidate_receipts",
    "parameter_receipt", "resource_estimate",
]
