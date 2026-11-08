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
