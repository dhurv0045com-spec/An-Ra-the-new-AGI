"""Tiny scripted engineering artifacts, not scientific results."""
import csv
import json
from pathlib import Path
from tools.elendril_v71_report import write_report


def artifact(root, job, epoch, grammar=False, correct=True, checkpoint='ckpt'):
    folder = root / 'analysis/jobs' / job; folder.mkdir(parents=True, exist_ok=True)
    outcome = {'case_id': 'case', 'world_id': 'world', 'outcome': {'correct': correct}}
    metrics = {'episodes': 1, 'correct': int(correct), 'accuracy': float(correct)}
    data = {'grammar': grammar, 'checkpoint': checkpoint, 'update': 20,
            'evaluation': {'cause_status': 'CAUSE_UNRESOLVED', 'metrics': {'base': {'all': metrics}},
                           'records': {'base': {'raw': {'cases_sha256': 'dataset', 'episodes': [outcome]}}}}}
    (folder / ('grammar-final.json' if grammar else f'epoch-{epoch}.json')).write_text(json.dumps(data), encoding='utf8')


def test_summary_matches_free_grammar_and_seed_blocks(tmp_path):
    for method in ('uniform', 'blocked'):
        job = f'S3-{method}-ce-s7'
        artifact(tmp_path, job, 4, correct=False)
        artifact(tmp_path, job, 4, grammar=True, correct=True)
    result = write_report(tmp_path)
    assert result['status'] == 'AVAILABLE'
    assert len(result['rows']) == 4 and result['paired_seed_blocks'][0]['complete']
    assert len(result['free_grammar']) == 2 and result['free_grammar'][0]['grammar_only'] == 1
    assert 'CAUSE UNRESOLVED' in (tmp_path / 'analysis/report.md').read_text(encoding='utf8')
    with (tmp_path / 'analysis/report.csv').open(encoding='utf8', newline='') as handle:
        assert len(list(csv.DictReader(handle))) == 4


def test_missing_grammar_and_unknown_identity_are_partial(tmp_path):
    artifact(tmp_path, 'unknown', 1)
    result = write_report(tmp_path)
    assert result['status'] == 'PARTIAL' and len(result['missing_inputs']) == 2
    assert result['rows'][0]['seed'] is None


def test_mismatched_checkpoints_do_not_pair(tmp_path):
    artifact(tmp_path, 'S3-uniform-ce-s7', 4, checkpoint='a')
    artifact(tmp_path, 'S3-uniform-ce-s7', 4, grammar=True, checkpoint='b')
    result = write_report(tmp_path)
    assert result['status'] == 'PARTIAL' and not result['free_grammar']
    assert not result['paired_seed_blocks'][0]['complete']


def test_no_jobs_is_partial_not_fabricated(tmp_path):
    result = write_report(tmp_path)
    assert result['status'] == 'PARTIAL' and not result['rows']
    assert result['forward_transfer'].startswith('NOT_QUALIFIED')


def test_frozen_original_job_identity_and_zero_epoch(tmp_path):
    artifact(tmp_path, 'originaluniform-s7', 0)
    artifact(tmp_path, 'originaluniform-s7', 0, grammar=True)
    result = write_report(tmp_path)
    assert result['status'] == 'AVAILABLE'
    assert result['rows'][0]['profile'] == 'S3' and result['rows'][0]['method'] == 'uniform'
    assert result['rows'][0]['epoch'] == 0
def test_continual_summary_retains_qualification_and_baseline_boundaries(tmp_path):
    from tools.elendril_v71_report import write_report
    raw={'phase':'continual','results':[{'job':{'job_id':'S3-fresh-b-ce-s7'},'status':'COMPLETE',
        'B_before':{'status':'NOT_ACQUIRED'},'B_after':{'status':'ACQUIRED'},
        'retention':{'status':'NOT_QUALIFIED','reason':'fresh baseline never acquired A'}}]}
    report=write_report(tmp_path,campaign=raw)
    assert report['retention'][0]['retention']['status']=='NOT_QUALIFIED'
    assert 'NOT_QUALIFIED' in report['forward_transfer']
