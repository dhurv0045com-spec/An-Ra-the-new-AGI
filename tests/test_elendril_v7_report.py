import builtins
import csv
import json

from tools.elendril_v7_report import summarize, write_summary


def receipt(root, method, seed, name, update, data='monitor-id'):
    folder = root / 'analysis' / 'jobs' / f'{method}-s{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    findings = {'interactive': {'nonnull_correct': 3, 'nonnull_total': 4,
                                'query_needed': 2, 'needed_queried': 1, 'answer_entropy_bits': 0.5},
                'supplied': {'nonnull_correct': 4, 'nonnull_total': 4,
                             'query_needed': None, 'needed_queried': None, 'answer_entropy_bits': None}}
    (folder / name).write_text(json.dumps({'update': update, 'data_identity': data,
        'checkpoint': method, 'diagnosis': {'findings': findings}}), encoding='utf8')


def test_matched_blocks_and_cohorts(tmp_path):
    for method in ('uniform', 'blocked', 'interleaved'):
        receipt(tmp_path, method, 7, 'diagnostic-u10.json', 10)
    receipt(tmp_path, 'uniform', 7, 'development-u10.json', 10, 'development-id')
    receipt(tmp_path, 'blocked', 19, 'training-u10.json', 10, 'training-id')
    receipt(tmp_path, 'uniform', 7, 'confirmation.json', 40, 'confirmation-id')
    report = summarize(tmp_path)
    assert report['status'] == 'COMPLETE'
    assert set(r['cohort'] for r in report['rows']) == {'monitor', 'development', 'training', 'confirmation'}
    monitor = [b for b in report['paired_blocks'] if b['cohort'] == 'monitor']
    assert len(monitor) == 2 and all(b['complete'] for b in monitor)
    assert all(not b['complete'] for b in report['paired_blocks'] if b['cohort'] != 'monitor')
    interactive = next(r for r in report['rows'] if r['condition'] == 'interactive')
    supplied = next(r for r in report['rows'] if r['condition'] == 'supplied')
    assert interactive['nonnull_accuracy'] == .75 and interactive['query_recall'] == .5
    assert supplied['query_recall'] is None and supplied['answer_entropy_bits'] is None


def test_empty_and_missing_plot_library_preserves_csv(tmp_path, monkeypatch):
    assert summarize(tmp_path)['status'] == 'NOT_RUN'
    assert write_summary(tmp_path)['status'] == 'NOT_RUN'
    receipt(tmp_path, 'uniform', 7, 'diagnostic-u10.json', 10)
    actual_import = builtins.__import__
    def no_matplotlib(name, *args, **kwargs):
        if name == 'matplotlib': raise ImportError('not available')
        return actual_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_matplotlib)
    report = write_summary(tmp_path)
    assert report['charts']['status'] == 'NOT_QUALIFIED'
    with (tmp_path / 'analysis' / 'trajectory.csv').open(encoding='utf8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert next(r for r in rows if r['condition'] == 'supplied')['query_recall'] == ''
    assert json.loads((tmp_path / 'analysis' / 'summary.json').read_text(encoding='utf8'))['charts']['status'] == 'NOT_QUALIFIED'


def test_scientific_observations_are_paired_and_cause_unresolved(tmp_path):
    # Scripted engineering fixture: these numbers are not research outcomes.
    from tools.elendril_v7_report import scientific_report
    for update, interactive in ((10, False), (20, True)):
        receipt(tmp_path, 'uniform', 7, f'diagnostic-u{update}.json', update)
        path = tmp_path / 'analysis' / 'jobs' / 'uniform-s7' / f'diagnostic-u{update}.json'
        data = json.loads(path.read_text(encoding='utf8'))
        data['diagnosis']['episodes'] = [
            {'cohort': 'interactive', 'case_id': 'fixture-1', 'symptom': 'outcomes_differ',
             'first_observable_divergence': 'action_1_after_release_alignment',
             'queried_then_prompt_identical': None, 'interactive_correct': interactive, 'supplied_correct': True},
            {'cohort': 'interactive', 'case_id': 'fixture-2', 'symptom': 'same_terminal_behavior',
             'first_observable_divergence': None, 'queried_then_prompt_identical': True,
             'interactive_correct': None, 'supplied_correct': False}]
        path.write_text(json.dumps(data), encoding='utf8')
    result = scientific_report(tmp_path)
    assert result['cause_status'] == 'UNRESOLVED' and result['demonstrated_cause'] is None
    first = result['paired_interventions'][0]
    assert (first['paired_denominator'], first['unavailable_outcome_pairs'], first['supplied_only']) == (1, 1, 1)
    assert result['case_observations'][0]['first_observed_divergence'] == 'action_1_after_release_alignment'
    assert result['case_observations'][0]['queried_then_prompt_identical'] is None
    assert any(t['case_id'] == 'fixture-1' and t['condition'] == 'interactive'
               and t['observation'] == 'ACQUISITION_OBSERVED' for t in result['case_transitions'])
    assert any(t['case_id'] == 'fixture-2' and t['observation'] == 'UNAVAILABLE'
               for t in result['case_transitions'])
    write_summary(tmp_path)
    assert json.loads((tmp_path / 'analysis/scientific_report.json').read_text(encoding='utf8'))['cause_status'] == 'UNRESOLVED'
    assert 'CAUSE UNRESOLVED' in (tmp_path / 'analysis/scientific_report.md').read_text(encoding='utf8')


def test_scientific_trajectories_do_not_mix_dataset_or_cohort(tmp_path):
    from tools.elendril_v7_report import scientific_report
    for filename, update, identity in [('diagnostic-u10.json', 10, 'id-a'),
                                       ('diagnostic-u20.json', 20, 'id-b'),
                                       ('training-u20.json', 20, 'id-a')]:
        receipt(tmp_path, 'uniform', 7, filename, update, identity)
        path = tmp_path / 'analysis/jobs/uniform-s7' / filename
        data = json.loads(path.read_text(encoding='utf8'))
        data['diagnosis']['episodes'] = [{'cohort': 'interactive', 'case_id': 'fixture-1',
            'interactive_correct': True, 'supplied_correct': False}]
        path.write_text(json.dumps(data), encoding='utf8')
    assert scientific_report(tmp_path)['case_transitions'] == []


def test_engineered_reference_uses_observed_public_state(tmp_path):
    # Small generated dataset and scripted baseline are an engineering fixture.
    from anra_v5.elendril.v6_data import make_dataset, write_dataset
    from anra_v5.elendril.v6_evaluation import ControlPolicy
    from anra_v5.elendril.v7_diagnostics import diagnose
    from tools.elendril_v7_report import scientific_report
    reference = tmp_path / 'reference'
    (reference / 'data').mkdir(parents=True)
    dataset = make_dataset('validation', worlds=14)
    write_dataset(dataset, reference / 'data' / 'monitor.json')
    diagnosis = diagnose(dataset, ControlPolicy('always-query'))
    folder = tmp_path / 'analysis/jobs/uniform-s7'
    folder.mkdir(parents=True)
    (folder / 'diagnostic-u10.json').write_text(json.dumps({'update': 10, 'data_identity': dataset.identity,
        'checkpoint': 'engineering-fixture', 'diagnosis': diagnosis}), encoding='utf8')
    report = scientific_report(tmp_path, reference_root=reference)
    refs = [v for row in report['case_observations'] for v in row['engineered_reference'].values()
            if v['status'] == 'AVAILABLE']
    assert refs and any(r['first_disagreement'] is not None for r in refs)
    assert any(r['evidence_release_observed'] and r['post_release_step_observed'] for r in refs)
    disagreement = next(r['first_disagreement'] for r in refs if r['first_disagreement'] is not None)
    assert disagreement['expected'] != disagreement['actual'] and disagreement['decision_index'] >= 0
    no_reference = scientific_report(tmp_path, reference_root=tmp_path)
    assert all(v['status'] == 'UNAVAILABLE' for r in no_reference['case_observations']
               for v in r['engineered_reference'].values())
