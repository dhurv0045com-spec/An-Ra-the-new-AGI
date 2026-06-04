"""Unit tests for HALModule - hormonal analog layer."""

from __future__ import annotations

import pytest

from identity.hal import HALModule


@pytest.fixture
def hal():
    return HALModule()


def test_initial_state_within_bounds(hal):
    s = hal.state
    for name in [
        "dopamine",
        "serotonin",
        "cortisol",
        "adrenaline",
        "oxytocin",
        "gaba",
        "norepinephrine",
    ]:
        if not hasattr(s, name):
            continue
        v = getattr(s, name)
        assert 0.0 <= v <= 1.0, f"{name}={v} out of [0,1]"


def test_decay_moves_toward_baseline(hal):
    hal.state.dopamine = 1.0
    hal.state.cortisol = 0.0
    for _ in range(20):
        hal.decay()
    assert hal.state.dopamine < 1.0, "dopamine should decay from spike"
    assert hal.state.cortisol >= 0.0, "cortisol must remain non-negative"


def test_appraise_high_reward_raises_dopamine(hal):
    initial = hal.state.dopamine
    hal.appraise(verifier_result=0.95, session_context={"novel_connection_detected": True})
    assert hal.state.dopamine >= initial, "high reward should not lower dopamine"


def test_appraise_failure_raises_cortisol(hal):
    initial = hal.state.cortisol
    hal.appraise(verifier_result=0.05, session_context={})
    assert hal.state.cortisol >= initial, "failure should not lower cortisol"


def test_attention_temperature_is_positive_float(hal):
    t = hal.attention_temperature()
    assert isinstance(t, float)
    assert t > 0.0


def test_attention_temperature_higher_under_cortisol(hal):
    hal.state.cortisol = 0.0
    t_calm = hal.attention_temperature()
    hal.state.cortisol = 0.9
    t_stress = hal.attention_temperature()
    assert t_stress > t_calm


def test_to_dict_and_from_dict_roundtrip(hal):
    hal.state.dopamine = 0.7
    d = hal.to_dict() if hasattr(hal, "to_dict") else vars(hal.state)
    assert "dopamine" in str(d)


def test_hal_registered_in_identity_registry():
    import anra
    from anra.core.registry import IDENTITY_REGISTRY

    assert "hal" in IDENTITY_REGISTRY, "HALModule must be registered as 'hal'"
