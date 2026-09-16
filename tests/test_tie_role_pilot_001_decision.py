"""TIE-ROLE-PILOT-001 decision-integrity tests (CPU, no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.tie_role_pilot_001_colab_v1 import SEEDS, _decide  # noqa: E402


def _pair(seed: int, auc_delta: float = 0.10, endpoint_delta: float = 0.20, endpoint: float = 0.5) -> dict:
    return {
        "seed": seed,
        "formation_auc_delta_treatment_minus_control": auc_delta,
        "endpoint_delta_treatment_minus_control": endpoint_delta,
        "endpoint_control": endpoint,
        "endpoint_treatment": endpoint + endpoint_delta,
    }


def test_decision_requires_both_registered_fresh_seeds():
    """Two pairs from the SAME seed must not yield a GO."""
    pairs = [_pair(SEEDS[0]), _pair(SEEDS[0])]
    decision = _decide(pairs)
    assert decision["decision"] == "DO_NOT_RUN_FULL_TIE_ROLE"
    assert decision["expensive_experiment_worth_running"] is False


def test_decision_rejects_unregistered_seed():
    assert _decide([_pair(SEEDS[0]), _pair(12345)])["decision"] == "DO_NOT_RUN_FULL_TIE_ROLE"


def test_decision_accepts_reversed_seed_order():
    assert _decide([_pair(SEEDS[1]), _pair(SEEDS[0])])["decision"] == "RUN_FULL_TIE_ROLE"


def test_decision_rejects_incomplete_pairs():
    for pairs in ([], [_pair(SEEDS[0])]):
        assert _decide(pairs)["decision"] == "DO_NOT_RUN_FULL_TIE_ROLE"


def test_decision_retains_threshold_and_ceiling_rules():
    for weak in (
        _pair(SEEDS[1], auc_delta=0.049),
        _pair(SEEDS[1], endpoint_delta=0.099),
        _pair(SEEDS[1], endpoint_delta=-0.10),
        _pair(SEEDS[1], endpoint=0.75),
    ):
        assert _decide([_pair(SEEDS[0]), weak])["decision"] == "DO_NOT_RUN_FULL_TIE_ROLE"


def test_decision_go_with_correct_registered_seeds():
    pairs = [_pair(SEEDS[0]), _pair(SEEDS[1])]
    decision = _decide(pairs)
    assert decision["decision"] == "RUN_FULL_TIE_ROLE"
    assert decision["expensive_experiment_worth_running"] is True
