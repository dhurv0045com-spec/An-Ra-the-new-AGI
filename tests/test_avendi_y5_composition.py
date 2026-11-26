import pytest
import torch
from experiments.avendi_y5 import composition as c


def test_donor_validation_rejects_original_bug_and_corruption():
    p = c.pairs(3, 12)[0]
    assert c.validate_pair(p)
    with pytest.raises(ValueError):
        c.validate_pair(dict(p, donor_rows=p['original_donor_rows']))
    with pytest.raises(ValueError):
        c.validate_pair(dict(p, donor_on_recipient_answer=p['recipient_answer']))
    assert c.preflight()['passed']


def test_panels_are_valid_and_exclude_duplicate_edges():
    for depth in (3, 4, 5, 6):
        for x in c.panel(12, 5, depth):
            assert c.walk(x['rows'], x['query'], depth) == x['gold']


def test_matched_losses_use_different_real_controls():
    torch.set_num_threads(1)
    model = c.Y4TypedCompositionModel('categorical', 16, seed=7)
    batch = c.pairs(4, 9)
    values = [c.intervention_loss(model, batch, arm) for arm in ('ordinary', 'iit', 'sham')]
    assert all(torch.isfinite(x) for x in values)
    assert values[1].item() != values[2].item()
    values[1].backward()
    assert model.k_proj.weight.grad is not None


def test_job_dependencies_and_historical_six():
    rows = c.jobs()
    ids = {r['id'] for r in rows}
    assert len(ids) == len(rows)
    assert all(set(r['depends']) <= ids for r in rows)
    assert sum(r['stage'] == 'G' for r in rows) == 6

def test_baseline_and_scout_cannot_reuse_development_panels_for_confirmation():
    from types import SimpleNamespace
    ctx = SimpleNamespace(split='confirm')
    for job in c.jobs()[:2]:
        result = c.run(job, ctx)
        assert result['status'] == 'BLOCKED'
        assert result['verification'] == 'BASELINE_SCOUT_DEV_ONLY'

def test_intervention_records_real_predictions_and_identity_agreement():
    torch.set_num_threads(1)
    model = c.Y4TypedCompositionModel('categorical', 16, seed=7)
    fixtures = c.pairs(4, 321, depth=3)
    result = c.measure_interventions(model, fixtures, 'cpu')
    assert result['total'] == 4
    assert result['identity_same_prediction'] == 4
    for row in result['cases']:
        assert row['outcomes']['actual_donor']['gold'] == row['fixture']['donor_on_recipient_answer']
        assert row['outcomes']['wrong_donor']['gold'] == row['fixture']['wrong_donor_actual_answer']
        assert row['outcomes']['rewired']['gold'] == row['fixture']['rewired_donor_on_recipient_answer']
        assert all(x['prediction'] in c.ENTITY_64 for x in row['outcomes'].values())
    panel = c.measure(model, c.panel(4, 45, 3), 'cpu')
    assert all(row['prediction'] in c.ENTITY_64 and c.walk(row['rows'], row['query'], row['depth']) == row['gold'] for row in panel['cases'])


def test_matched_arms_use_same_forward_count_and_batch_sizes():
    torch.set_num_threads(1)
    model = c.Y4TypedCompositionModel('categorical', 16, seed=7)
    original = model.forward_trace
    batch = c.pairs(4, 77)
    traces = {}
    for arm in ('ordinary', 'iit', 'sham'):
        calls = []
        def capture(rk, rv, q, mask, **kwargs):
            calls.append((tuple(rk.shape), kwargs['steps'], kwargs.get('patch_at')))
            return original(rk, rv, q, mask, **kwargs)
        model.forward_trace = capture
        c.intervention_loss(model, batch, arm)
        traces[arm] = calls
    assert traces['ordinary'] == traces['iit'] == traces['sham']
