import copy
import pytest
from anra_v5.elendril.v6_data import _case
from anra_v5.elendril.cognition import belief
from anra_v5.elendril.v6_evaluation import ControlPolicy, evaluate
from anra_v5.elendril.v6_supplement import condition_specs, run_experiment, verify_experiment, summarize

@pytest.fixture
def data():
    return tuple(_case(i, 606, 'validation') for i in range(32))

def test_permutation_and_replay(data):
    specs = condition_specs(data, 'S17')
    for mode in ('fixed', 'oracle'):
        result = run_experiment(data, ControlPolicy(mode), 'S17')
        assert result['summary']['paired']['baseline_vs_permuted']['regressions'] == 0
        assert result['summary']['paired']['baseline_vs_permuted']['gains'] == 0
        broken = copy.deepcopy(result)
        broken['records']['baseline']['episodes'][0]['outcome']['correct'] ^= True
        with pytest.raises(ValueError): verify_experiment(data, 'S17', broken)
    assert specs['baseline']['cases'][0].goal.subject != specs['permuted']['cases'][0].goal.subject

def test_budget_and_stale(data):
    result = run_experiment(data, ControlPolicy('fixed'), 'S16')
    small = result['records']['1']['episodes']
    assert all(not r['outcome']['correct'] for r in small if r['kind'] in ('missing', 'memory'))
    result = run_experiment(data, ControlPolicy('fixed'), 'S18')
    assert result['summary']['paired']['baseline_vs_stale']['regressions'] == 0
    for c in condition_specs(data, 'S18')['stale']['cases']:
        old = c.memory.events[0]
        assert belief(c.environment.initial + (old,), c.goal.subject).value != old.value
    assert all(r['trace']['requests'] == 1 for rec in result['records'].values() for r in rec['episodes'])

def test_missing_cohort(data):
    result = run_experiment(data[:1], ControlPolicy('fixed'), 'S16')
    assert result['status'] == 'NOT_QUALIFIED'

def test_cost_and_mismatch(data):
    # Use a bound synthetic model record only to exercise arithmetic, not as an evaluation receipt.
    controls = {name: evaluate(data, ControlPolicy(name)) for name in ('fixed', 'always-query', 'no-query')}
    model = copy.deepcopy(controls['fixed']); model['policy'] = 'model'; model['model_identity'] = 'bound'
    result = summarize('S19', {'model': model}, controls)
    totals = model['metrics']['all']
    assert result['summary']['mean_utility_by_query_cost']['model']['0.5'] == (totals['correct'] - .5 * totals['verified_queries']) / totals['episodes']
    controls['fixed']['budget']['decisions'] += 1
    with pytest.raises(ValueError): summarize('S19', {'model': model}, controls)


def test_binding_and_report_endpoints(data):
    result = run_experiment(data, ControlPolicy('fixed'), 'S17')
    pair = result['summary']['paired']['baseline_vs_permuted']
    assert pair['both_correct'] <= pair['pairs']
    assert pair['changed_outcome_rate'] == 0
    assert result['summary']['output_equivariance'] == 1
    for field, value in [('schema', 'bad'), ('source_sha256', 'bad')]:
        damaged = copy.deepcopy(result); damaged[field] = value
        with pytest.raises(ValueError): verify_experiment(data, 'S17', damaged)
    damaged = copy.deepcopy(result); damaged['records']['permuted']['split'] = 'other'
    with pytest.raises(ValueError): verify_experiment(data, 'S17', damaged)
    damaged = copy.deepcopy(result); damaged['records']['permuted']['model_identity'] = 'other'
    with pytest.raises(ValueError): summarize('S17', damaged['records'])

def test_s19_missing_control_with_extra(data):
    controls = {name: evaluate(data, ControlPolicy(name)) for name in ('fixed', 'no-query')}
    controls['extra'] = controls['fixed']
    model = copy.deepcopy(controls['fixed']); model['policy'] = 'model'; model['model_identity'] = 'bound'
    with pytest.raises(ValueError): summarize('S19', {'model': model}, controls)
