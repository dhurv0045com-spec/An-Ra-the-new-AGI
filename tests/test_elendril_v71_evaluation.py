"""Scripted engineering controls only; these are not model research results."""
import copy
import pytest
from anra_v5.elendril.v6_evaluation import ControlPolicy
from anra_v5.elendril.v71_data import build_suite
from anra_v5.elendril.v71_evaluation import evaluate_suite, verify_suite, acquisition_gate, retention


@pytest.fixture(scope='module')
def fixture_suite():
    return build_suite(worlds_per_kind=2)


def test_verified_controls_and_raw_metric_tampering(fixture_suite):
    report = evaluate_suite(fixture_suite, ControlPolicy('fixed'), conditions=['base', 'balanced-payload'])
    assert verify_suite(fixture_suite, report)['status'] == 'VERIFIED'
    assert report['cause_status'] == 'CAUSE_UNRESOLVED'
    assert report['metrics']['base']['all']['utility_denominator'] == 10
    assert report['engineered_reference']['base']
    assert any(x['paired_denominator'] > 0 for x in report['within_cohort_comparisons'])
    altered = copy.deepcopy(report)
    altered['metrics']['base']['all']['invalid'] += 1
    with pytest.raises(ValueError): verify_suite(fixture_suite, altered)
    altered = copy.deepcopy(report)
    altered['records']['base']['raw']['episodes'][0]['outcome']['correct'] = False
    with pytest.raises(ValueError): verify_suite(fixture_suite, altered)


def test_precision_recall_and_unavailable():
    from anra_v5.elendril.v6_data import make_dataset
    cases = make_dataset('validation', worlds=14).cases
    report = evaluate_suite({'empty': (), 'base': cases}, ControlPolicy('query-blind'))
    all_metrics = report['metrics']['base']['all']
    assert report['metrics']['empty'] is None
    assert all_metrics['query_attempt_episodes'] == 0 and all_metrics['query_precision'] is None
    assert all_metrics['query_needed'] > 0 and all_metrics['query_recall'] == 0
    verify_suite({'empty': (), 'base': cases}, report)


def test_memory_intervention_matched_panels(fixture_suite):
    report = evaluate_suite(fixture_suite, ControlPolicy('fixed'), conditions=['memory-manipulations'])
    matched = report['within_cohort_comparisons']
    assert any(x['left'] == 'memory-original' and x['right'] == 'memory-removed' and x['paired_denominator'] for x in matched)
    assert verify_suite(fixture_suite, report)['status'] == 'VERIFIED'


def test_retention_requires_both_acquisition_gates():
    acquired = {'nonnull_correct': 9, 'nonnull_total': 10}
    failed = {'nonnull_correct': 1, 'nonnull_total': 10}
    assert acquisition_gate(acquired)['status'] == 'ACQUIRED'
    assert retention(acquired, {'new_skill': failed, 'old_skill': acquired})['status'] == 'NOT_QUALIFIED'
    qualified = retention(acquired, {'new_skill': acquired, 'old_skill': failed})
    assert qualified['status'] == 'OBSERVED' and qualified['delta'] == pytest.approx(-.8)
    assert qualified['forward_transfer'].startswith('NOT_QUALIFIED')


def test_fixed_repricing_and_confidence_are_descriptive():
    from dataclasses import replace
    from anra_v5.elendril.v6_data import make_dataset
    class Confident(ControlPolicy):
        def for_case(self, case, *, memory_available=None):
            base = super().for_case(case, memory_available=memory_available)
            class Actor:
                origin = base.origin
                model_identity = None
                def propose(self, state, legal, channels, positions):
                    return replace(base.propose(state, legal, channels, positions), predicted_success=.8)
            return Actor()
    suite = {'base': make_dataset('validation', worlds=14).cases}
    report = evaluate_suite(suite, Confident('fixed'))
    m = report['metrics']['base']['all']
    assert m['calibration']['count'] == m['episodes'] and m['calibration']['brier'] is not None
    assert sum(x['count'] for x in m['calibration']['bins']) == m['episodes']
    assert [x['query_cost_numerator']/x['query_cost_denominator'] for x in m['fixed_policy_repricing']['values']] == [0, .1, .5, 1]
    assert 'NOT cost adaptation' in m['fixed_policy_repricing']['qualification']
    verify_suite(suite, report)
