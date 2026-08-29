"""Machine-readable statistical and promotion procedure identities."""

from __future__ import annotations

import hashlib
import json


PROTOCOL = {
    "schema": "esoes-e0-statistical-protocol/v1",
    "binary_accuracy": {
        "single_system_interval": "wilson-95-two-sided",
        "paired_comparison": "exact-two-sided-sign-test-on-discordant-pairs",
        "alpha": 0.05,
    },
    "continuous_score_delta": {
        "comparison": "paired-percentile-bootstrap",
        "resamples": 10_000,
        "confidence": 0.95,
        "seed_source": "evaluation-receipt-preregistered-seed",
    },
    "query_flip": {
        "interval": "wilson-95-two-sided",
        "gate": "lower-bound-versus-preregistered-threshold",
    },
    "conditional_realization": {
        "denominator": "cases-with-correct-unassisted-selection",
        "interval": "wilson-95-two-sided",
        "minimum_eligible_cases": 100,
    },
    "family_policy": {
        "promotion": "conjunctive-worst-family-gates-no-averaging",
        "exploratory_multiple_tests": "holm-bonferroni",
    },
    "sealed_consumption": "any outcome-guided decision consumes the sealed fixture",
}


def protocol_sha256() -> str:
    encoded = json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
