"""FORMATION-MUX-001 Science-S4 training binding.

S4 keeps the S3 mathematics exactly, but a training worker is now incapable of
receiving raw sealed rows.  The historical v2 implementation internally
expects a `sealed` key only for a pre-training R0-id presence check, so this
binding supplies an empty compatibility split after proving that the public
manifest itself contains training + development only.
"""

from __future__ import annotations

from typing import Any, Mapping

from anra_v5 import formation_mux_train_v2 as _base
from anra_v5 import formation_mux_model_v3 as fxm
from v5_experiments import formation_mux_protocol_v4 as proto
from v5_experiments.formation_mux_surface_v4 import validate_public_surface

_base.fxm = fxm
_base.proto = proto

build_batch = _base.build_batch
evaluate_development = _base.evaluate_development
load_model_for_evaluation = _base.load_model_for_evaluation
_encode_row = _base._encode_row
_row_processed_tokens = _base._row_processed_tokens
_greedy_rates = _base._greedy_rates


def _worker_surface(public_surface: Mapping[str, Any]) -> dict[str, Any]:
    validate_public_surface(public_surface)
    splits = dict(public_surface["splits"])
    if "sealed" in splits:
        raise RuntimeError(
            "SEALED_FIREWALL_BREACH: training worker received a sealed split"
        )
    # Compatibility only: the v2 mathematical loop checks R0 tokenization in
    # all named splits but never trains/evaluates this key.  Keeping it empty
    # exposes zero sealed examples while avoiding a fork of the audited loop.
    return {**dict(public_surface), "splits": {**splits, "sealed": []}}


def train_arm(**kwargs):
    _base.fxm = fxm
    _base.proto = proto
    if "surface" not in kwargs:
        raise RuntimeError("public worker surface missing")
    kwargs = dict(kwargs)
    kwargs["surface"] = _worker_surface(kwargs["surface"])
    return _base.train_arm(**kwargs)
