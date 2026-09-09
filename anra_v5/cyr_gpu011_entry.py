"""Canonical operator entry points for CYR-GPU-011.

The scientific runner was written against an early research call-site that
redundantly supplied AdamW hyperparameters already frozen by Cymek's canonical
optimizer constructor. These wrappers apply a scoped, fail-closed compatibility
context only for the duration of CYR-GPU-011 calibration/campaign execution.
The production optimizer API is restored immediately afterwards.
"""
from __future__ import annotations

from typing import Any

from anra_v5 import cyr_gpu011_run as _runner
from anra_v5.cyr_gpu011_optimizer_compat import canonical_optimizer_compat

BUNDLE_NAME = _runner.BUNDLE_NAME
production_tokenizer = _runner.production_tokenizer
write_json = _runner.write_json
read_json = _runner.read_json


def calibrate_all(*args: Any, **kwargs: Any):
    with canonical_optimizer_compat():
        return _runner.calibrate_all(*args, **kwargs)


def run_campaign(*args: Any, **kwargs: Any):
    with canonical_optimizer_compat():
        return _runner.run_campaign(*args, **kwargs)


__all__ = [
    "BUNDLE_NAME",
    "calibrate_all",
    "production_tokenizer",
    "read_json",
    "run_campaign",
    "write_json",
]
