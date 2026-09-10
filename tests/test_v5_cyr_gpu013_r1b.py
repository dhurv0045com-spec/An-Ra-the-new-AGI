from __future__ import annotations

import pytest

from v5_experiments import cyr_gpu013_r1b as core


def _cal(vocab: int, ups: float = 20.0, eps: float = 1000.0) -> dict:
    return {
        "status": "PASS",
        "regime": f"R1B_CHAR_V{vocab}",
        "batch_rows": 64,
        "training_updates_per_sec": ups,
        "generation_examples_per_sec": eps,
    }


def test_grid_and_endpoint_are_frozen() -> None:
    assert core.VOCABS == (19, 1024, 4096, 8192, 16384, 24576)
    assert core.BATCH_ROWS == 64
    assert core.UPDATES == 2000
    assert core.ROW_PRESENTATIONS == 128000
    assert len(core.MODEL_SEEDS) == 3
    assert len(core.ORDER_SEEDS) == 3


def test_arm_orders_are_exact_permutations() -> None:
    target = sorted(core.VOCABS)
    assert len(core.ARM_ORDERS) == 3
    for order in core.ARM_ORDERS:
        assert sorted(order) == target


def test_resolver_requires_two_curves_and_may_allow_three() -> None:
    cals = {f"V{v}": _cal(v, ups=20.0) for v in core.VOCABS}
    out = core.resolve_from_calibrations(cals)
    assert out["curves_to_run"] in (2, 3)
    assert out["mandatory_curves"] == 2
    assert out["updates_per_arm"] == 2000


def test_resolver_fails_closed_when_two_curves_do_not_fit() -> None:
    cals = {f"V{v}": _cal(v, ups=0.2, eps=10.0) for v in core.VOCABS}
    with pytest.raises(RuntimeError):
        core.resolve_from_calibrations(cals)


def _arm(score: float) -> dict:
    return {
        "acquisition": {
            "row_presentations": core.ROW_PRESENTATIONS,
            "reasoning_battery_final": {
                "STANDARD": {"complete_exact_with_valid_stop": score}
            },
        }
    }


def test_replicated_intermediate_advantage_decision() -> None:
    arms = {}
    for i in range(2):
        scores = {19: 0.10, 1024: 0.72, 4096: 0.95, 8192: 0.84, 16384: 0.65, 24576: 0.02}
        for vocab, score in scores.items():
            arms[core.arm_label(i, vocab)] = _arm(score)
    d = core.decision(arms, 2)
    assert d["verdict"] == "REPLICATED_INTERMEDIATE_CLASS_SPACE_ADVANTAGE"
    assert d["pre500m_authorized"] is False


def test_mixed_seed_sensitive_decision() -> None:
    arms = {}
    first = {19: 0.10, 1024: 0.70, 4096: 0.95, 8192: 0.85, 16384: 0.60, 24576: 0.0}
    second = {19: 0.55, 1024: 0.52, 4096: 0.56, 8192: 0.50, 16384: 0.48, 24576: 0.50}
    for i, scores in enumerate((first, second)):
        for vocab, score in scores.items():
            arms[core.arm_label(i, vocab)] = _arm(score)
    assert core.decision(arms, 2)["verdict"] == "MIXED_OR_SEED_SENSITIVE_RESPONSE_CURVE"


def test_incomplete_curve_is_not_scored() -> None:
    arms = {core.arm_label(0, v): _arm(0.8) for v in core.VOCABS}
    d = core.decision(arms, 2)
    assert d["verdict"] == "INCONCLUSIVE_NO_TWO_COMPLETE_CURVES"
