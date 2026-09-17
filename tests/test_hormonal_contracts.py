"""Fail-closed hormonal contracts, independent of training evidence."""
from dataclasses import replace

import pytest

from v5_identity.hormonal_state import HALState


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 1.1])
def test_constructor_validates_levels(value):
    with pytest.raises(ValueError):
        HALState(adrenaline=value)


def test_direct_delta_rejects_nonfinite():
    with pytest.raises(ValueError):
        HALState().with_delta('adrenaline', float('nan'))


def test_history_has_bounded_memory():
    state = HALState()
    for _ in range(200):
        state = state.decay()
    assert len(state.log) <= 64
    assert state.step == 200


def test_verified_outcomes_map_to_legible_deltas():
    from v5_identity.hormonal_state import VerifiedOutcome
    state = HALState()
    success = state.appraise(VerifiedOutcome('success', 'symbolic-check-1'))
    assert success.dopamine > state.dopamine
    failure = state.appraise(VerifiedOutcome('failure', 'symbolic-check-2'))
    assert failure.cortisol > state.cortisol
    assert state.appraise(VerifiedOutcome('coherence', 'check-3')).serotonin > state.serotonin
    with pytest.raises(ValueError):
        VerifiedOutcome('success', '')
    with pytest.raises(ValueError):
        VerifiedOutcome('confidence', 'self-report')


def test_serotonin_dampens_and_stress_triggers_gaba():
    from v5_identity.hormonal_state import VerifiedOutcome
    event = VerifiedOutcome('surprise', 'loss-spike')
    low, high = HALState(serotonin=0), HALState(serotonin=1)
    assert high.appraise(event).adrenaline - high.adrenaline < low.appraise(event).adrenaline - low.adrenaline
    stressed = HALState(cortisol=1, adrenaline=1)
    assert stressed.appraise(VerifiedOutcome('unresolved_stress', 'measured-errors')).gaba > stressed.gaba


def test_sibling_full_receipt_and_config_bound_to_hash():
    import torch
    from v5_contracts.model_spec import V5A_250M
    from v5_contracts.model_spec_hormonal import V5A_250M_HORMONAL_V1
    from v5_identity.hormonal_model import initialize_hormonal
    variant = replace(V5A_250M_HORMONAL_V1, base=replace(
        V5A_250M, vocabulary_size=32, width=16, layers=1, query_heads=2,
        kv_heads=1, head_dimension=8, ffn_width=32, context_length=16))
    model = initialize_hormonal(variant, 17, torch_module=torch)
    assert sum(p.numel() for p in model.parameters()) == variant.parameter_receipt()['total']
    assert variant.parameter_receipt()['projection'] == 14
    assert model.spec.sha256() == variant.sha256()
    different = replace(variant, hormonal=replace(variant.hormonal, max_log_temperature=0.25))
    assert variant.sha256() != different.sha256()
    changed = initialize_hormonal(different, 17, torch_module=torch)
    with torch.no_grad():
        changed.projection.fill_(100)
    changed.set_hormones(HALState())
    assert changed.log_temperature().abs().max().item() <= 0.25
    # A checkpoint from a different conditioning contract must fail closed.
    with pytest.raises(ValueError):
        changed.load_state_dict(model.state_dict())


@pytest.mark.parametrize('step', [float('nan'), 0.5, -1, True])
def test_step_must_be_nonnegative_integer(step):
    with pytest.raises(ValueError):
        HALState(step=step)


def test_numeric_levels_normalized_for_decay():
    assert HALState(adrenaline='0.5').decay().adrenaline < 0.5


def test_packaging_includes_identity():
    import pathlib
    assert '"v5_identity*"' in pathlib.Path('pyproject.toml').read_text()
