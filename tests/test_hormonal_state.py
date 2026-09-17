"""HALState: decay, appraisal deltas, bounds, cortisol hangover, legibility."""
import math

import pytest

from v5_identity.hormonal_state import (
    CORTISOL_HANGOVER_FROM_ADRENALINE,
    HALState,
    HORMONE_DYNAMICS,
    HORMONE_NAMES,
    neutral_state,
)

def test_decay_returns_toward_baseline_at_hormone_specific_rates():
    state = HALState(adrenaline=1.0, cortisol=0.0, serotonin=0.9)
    after = state.decay()
    for name in HORMONE_NAMES:
        baseline = HORMONE_DYNAMICS[name][0]
        rate = HORMONE_DYNAMICS[name][1]
        expected = baseline + (getattr(state, name) - baseline) * (1 - rate)
        if name == "cortisol":
            dropped = state.adrenaline - (
                HORMONE_DYNAMICS["adrenaline"][0]
                + (state.adrenaline - HORMONE_DYNAMICS["adrenaline"][0]) * (1 - HORMONE_DYNAMICS["adrenaline"][1]))
            expected += CORTISOL_HANGOVER_FROM_ADRENALINE * max(0.0, dropped)
        assert after.__getattribute__(name) == pytest.approx(expected)
    assert after.step == state.step + 1


def test_decay_repeated_converges_to_baselines():
    state = neutral_state().appraise(adrenaline=0.95).appraise(cortisol=0.5)
    for _ in range(2000):
        state = state.decay()
    for name in HORMONE_NAMES:
        assert state.__getattribute__(name) == pytest.approx(HORMONE_DYNAMICS[name][0], abs=1e-6)


def test_bounds_clamp_deltas_within_range():
    state = neutral_state().appraise(dopamine=5.0).appraise(serotonin=-5.0)
    assert state.dopamine == pytest.approx(HORMONE_DYNAMICS["dopamine"][3])  # clamped to high
    assert state.serotonin == pytest.approx(HORMONE_DYNAMICS["serotonin"][2])  # clamped to low


def test_cortisol_hangover_from_adrenaline_decay():
    state = HALState(adrenaline=0.5)
    baseline_adrenaline = HORMONE_DYNAMICS["adrenaline"][0]
    after = state.decay()
    dropped = state.adrenaline - after.adrenaline
    expected_cortisol = min(
        HORMONE_DYNAMICS["cortisol"][3],
        HORMONE_DYNAMICS["cortisol"][0]
        + (state.cortisol - HORMONE_DYNAMICS["cortisol"][0]) * (1 - HORMONE_DYNAMICS["cortisol"][1])
        + CORTISOL_HANGOVER_FROM_ADRENALINE * dropped,
    )
    assert after.cortisol == pytest.approx(expected_cortisol)


def test_appraise_rejects_unknown_signals_and_nonfinite():
    state = neutral_state()
    with pytest.raises(KeyError):
        state.appraise(venom=1.0)
    with pytest.raises(ValueError):
        state.appraise(dopamine=float("nan"))


def test_state_is_immutable_value_type():
    state = neutral_state()
    with pytest.raises(Exception):
        state.dopamine = 0.5


def test_summary_is_legible_and_receipt_roundtrips():
    state = neutral_state().appraise(adrenaline=0.6)
    text = state.summary()
    assert "adrenaline" in text
    receipt = state.as_receipt()
    assert set(receipt) == {"step", *HORMONE_NAMES}
    assert all(isinstance(v, float) for k, v in receipt.items() if k != "step")


def test_decay_log_grows_and_records_history():
    state = neutral_state()
    for _ in range(3):
        state = state.decay()
    assert len(state.log) == 3
    assert state.log[-1]["step"] == 3.0
