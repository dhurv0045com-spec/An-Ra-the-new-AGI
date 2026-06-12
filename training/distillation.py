"""License- and verification-gated distillation intake."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistillationSource:
    model: str
    model_license: str
    output_use_allowed: bool
    dataset_license: str
    attribution_required: bool
    redistribution_allowed: bool


@dataclass(frozen=True)
class DistillationExample:
    prompt: str
    output: str
    verified: bool
    identity_score: float
    source: DistillationSource


def accept_distillation_example(example: DistillationExample) -> tuple[bool, tuple[str, ...]]:
    failures: list[str] = []
    if not example.verified:
        failures.append("unverified")
    if example.identity_score < 0.30:
        failures.append("identity_filter")
    if not example.source.output_use_allowed:
        failures.append("output_use_not_allowed")
    if not example.source.redistribution_allowed:
        failures.append("redistribution_not_allowed")
    if not example.source.model_license or not example.source.dataset_license:
        failures.append("license_metadata_missing")
    return not failures, tuple(failures)
