import json
from pathlib import Path
import zipfile

from tools import elendril_v6_export as v6
from tools import elendril_v7_export as v7

SOURCE = 'a' * 40
PROTOCOL = 'b' * 64


def run(root, **kw):
    return v7.export_results(root, source=SOURCE, protocol_sha=PROTOCOL,
                             campaign={'status': 'PARTIAL'}, **kw)


def test_dual_archives_and_rescore(tmp_path):
    analysis = tmp_path / 'analysis'
    analysis.mkdir()
    (analysis / 'failed.json').write_text('{"status":"FAILED"}', encoding='utf8')
    (analysis / 'checkpoint_manifest.json').write_text('{"sha256":"abc"}', encoding='utf8')
    weights = tmp_path / 'checkpoints' / 'weights.bin'
    weights.parent.mkdir()
    weights.write_bytes(b'\x00model')
    seen = []
    def rescore(root, campaign):
        seen.append(root)
        return {'status': 'FAILED', 'reason': 'mismatch'}
    reports = run(tmp_path, rescore=rescore)
    assert seen == [tmp_path.resolve()]
    for report in reports.values():
        assert report['status'] == 'EXPORT_VERIFIED'
        assert report['scientific_rescore']['status'] == 'FAILED'
        assert v6.verify_zip(Path(report['zip_path']))['status'] == 'EXPORT_VERIFIED'
    with zipfile.ZipFile(reports['analysis']['zip_path']) as z:
        assert z.read('failed.json') == b'{"status":"FAILED"}'
        assert 'checkpoint_manifest.json' in z.namelist()
        assert not any('weights.bin' in n or n.endswith('.zip') for n in z.namelist())
    with zipfile.ZipFile(reports['full']['zip_path']) as z:
        assert z.read('checkpoints/weights.bin') == b'\x00model'
        assert not any(n.endswith('.zip') for n in z.namelist())


def test_failure_preserves_previous_and_raw(tmp_path, monkeypatch):
    analysis = tmp_path / 'analysis'
    analysis.mkdir()
    raw = analysis / 'failed.json'
    raw.write_text('{"status":"FAILED"}', encoding='utf8')
    first = run(tmp_path)
    archives = [Path(first[k]['zip_path']) for k in ('analysis', 'full')]
    prior = [p.read_bytes() for p in archives]
    def fail(*args, **kwargs):
        raise v6.SnapshotChanged('worker changed output')
    monkeypatch.setattr(v6, 'export_bundle', fail)
    result = run(tmp_path)
    assert all(result[k]['status'] == 'EXPORT_DEFERRED' for k in ('analysis', 'full'))
    assert [p.read_bytes() for p in archives] == prior
    assert raw.read_text(encoding='utf8') == '{"status":"FAILED"}'


def test_analysis_rejects_weights_and_binary(tmp_path):
    a = tmp_path / 'analysis'
    a.mkdir()
    (a / 'weights.pt').write_bytes(b'weights')
    result = run(tmp_path, full=False)
    assert result['analysis']['status'] == 'EXPORT_DEFERRED'
    assert result['full']['status'] == 'SKIPPED'
    assert (a / 'weights.pt').exists()


def test_audit_journal(tmp_path):
    p = tmp_path / 'ledger.zip'
    events = [{'kind': 'EXPORT_DEFERRED', 'error_type': 'ValueError', 'message': 'checkpoint writer lock unavailable for this lineage'},
              {'kind': 'EXPORT_VERIFIED'}, {'status': 'COMPLETE'}]
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('experiment_ledger.jsonl', '\n'.join(json.dumps({'event': e}) for e in events))
    audit = v7.audit_journal(p)
    assert audit['events'] == 3
    assert audit['counts'] == {'EXPORT_DEFERRED': 1, 'EXPORT_VERIFIED': 1, 'COMPLETE': 1}
    assert audit['deferred_reasons'] == {'lock_unavailable': 1}


def test_export_lock_defers_without_touching_archives(tmp_path):
    from v5_training.checkpoint_lock import checkpoint_writer
    analysis = tmp_path / 'analysis'
    analysis.mkdir()
    (analysis / 'partial.json').write_text('{"status":"PARTIAL"}', encoding='utf8')
    first = run(tmp_path)
    archives = [Path(first[k]['zip_path']) for k in ('analysis', 'full')]
    before = [p.read_bytes() for p in archives]
    with checkpoint_writer(tmp_path, filename='.pending-v7-export-writer'):
        result = run(tmp_path)
    assert [result[k]['status'] for k in ('analysis', 'full')] == ['EXPORT_DEFERRED'] * 2
    assert [p.read_bytes() for p in archives] == before
    assert (analysis / 'partial.json').exists()
