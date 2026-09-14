"""FORMATION-MUX-001 Science S3 training binding.

Reuses the audited Amendment-1 training implementation while rebinding its
model/protocol globals to Amendment-1B. This keeps exposure matching,
identity-only primary scoring, exact resume, and model-only sealed restore,
while preserving frozen-row gradients through the global clip.
"""

from anra_v5 import formation_mux_train_v2 as _base
from anra_v5 import formation_mux_model_v3 as fxm
from v5_experiments import formation_mux_protocol_v3 as proto

_base.fxm = fxm
_base.proto = proto

build_batch = _base.build_batch
evaluate_development = _base.evaluate_development
load_model_for_evaluation = _base.load_model_for_evaluation
_encode_row = _base._encode_row
_row_processed_tokens = _base._row_processed_tokens
_greedy_rates = _base._greedy_rates


def train_arm(**kwargs):
    # Rebind immediately before every invocation in case another module imported
    # the v2 training surface in the same interpreter.
    _base.fxm = fxm
    _base.proto = proto
    return _base.train_arm(**kwargs)
