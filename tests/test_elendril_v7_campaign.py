"""Small admission/cache tests; never launch research training or require CUDA."""
import json
from types import SimpleNamespace

import pytest

from tools import elendril_v7_worker as worker
from anra_v5.elendril.v7_protocol import PROTOCOL_SHA


def test_worker_requires_owner_and_valid_cap(tmp_path, monkeypatch):
    request = tmp_path / 'request.json'
    receipt = tmp_path / 'out' / 'receipt.json'
    request.write_text(json.dumps({'timecap_seconds': float('nan')}))
    with pytest.raises(SystemExit):
        worker.main(['train', '--request', str(request), '--receipt', str(receipt)])
    with pytest.raises(ValueError, match='wall cap'):
        worker.main(['train', '--owner-enabled', '--request', str(request), '--receipt', str(receipt)])
    assert not receipt.exists()


def test_worker_protocol_and_source_path_admission(tmp_path, monkeypatch):
    request = tmp_path / 'request.json'
    receipt = tmp_path / 'receipt.json'
    req = {'timecap_seconds': 10, 'protocol_sha': 'wrong', 'source': 'a' * 40,
           'output': str(tmp_path), 'checkpoint_root': str(tmp_path / 'cp')}
    request.write_text(json.dumps(req))
    with pytest.raises(ValueError, match='protocol mismatch'):
        worker.main(['canary', '--owner-enabled', '--request', str(request), '--receipt', str(receipt)])
    req['protocol_sha'] = PROTOCOL_SHA
    req['output'] = str(worker.REPO / 'unsafe-output')
    request.write_text(json.dumps(req))
    monkeypatch.setattr(worker, 'check_source', lambda *_: None)
    with pytest.raises(ValueError, match='outside source'):
        worker.main(['canary', '--owner-enabled', '--request', str(request), '--receipt', str(receipt)])
    assert not receipt.exists()


def test_train_rejects_unregistered_job_before_cuda(tmp_path, monkeypatch):
    request = tmp_path / 'request.json'
    receipt = tmp_path / 'receipt.json'
    request.write_text(json.dumps({'timecap_seconds': 10, 'protocol_sha': PROTOCOL_SHA,
        'source': 'a' * 40, 'output': str(tmp_path),
        'checkpoint_root': str(tmp_path / 'cp'),
        'job': {'job_id': 'arbitrary', 'method': 'uniform', 'seed': 7}}))
    monkeypatch.setattr(worker, 'check_source', lambda *_: None)
    with pytest.raises(ValueError, match='outside frozen'):
        worker.main(['train', '--owner-enabled', '--request', str(request), '--receipt', str(receipt)])
    assert not receipt.exists()


def test_cached_checkpoint_and_update_must_match(tmp_path, monkeypatch):
    import anra_v5.elendril.v7_diagnostics as diagnostics
    monkeypatch.setattr(worker, '_weights', lambda _: 'weights')
    monkeypatch.setattr(diagnostics, 'verify_diagnosis', lambda *_: {})
    trainer = SimpleNamespace(model=object(), parent='expected', state=SimpleNamespace(global_update=4))
    data = SimpleNamespace(identity='dataset')
    path = tmp_path / 'diagnostic.json'
    result = {'checkpoint': 'wrong', 'data_identity': 'dataset', 'update': 4,
              'weight_sha256': 'weights', 'diagnosis': {'records': {}}}
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='binding'):
        worker._evaluate(trainer, data, path)


def test_campaign_owner_disabled_has_no_side_effects(tmp_path):
    from tools import elendril_v7_campaign as campaign
    target = tmp_path / 'untouched'
    with pytest.raises(ValueError, match='owner'):
        campaign.run(output=target, source='a' * 40)
    assert not target.exists()


def test_campaign_preflight_failure_exports_receipt_and_zip(tmp_path, monkeypatch):
    from tools import elendril_v7_campaign as campaign
    import zipfile
    target = tmp_path / 'run'
    def fail_source(*_):
        raise ValueError('deliberate source failure')
    monkeypatch.setattr(campaign, 'check_source', fail_source)
    result = campaign.run(output=target, source='a' * 40, owner_enabled=True, minutes=1)
    assert result['status'] == 'NO_GO'
    assert 'deliberate source failure' in result['message']
    receipts = list((target / 'analysis').glob('campaign-*.json'))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_bytes())['status'] == 'NO_GO'
    archive = target / 'exports' / 'ELENDRIL_V7_ANALYSIS.zip'
    assert archive.is_file()
    with zipfile.ZipFile(archive) as z:
        assert any(name.endswith(receipts[0].name) for name in z.namelist())


@pytest.mark.parametrize('failure', ['timeout', 'missing'])
def test_invoke_preserves_error_receipt(tmp_path, monkeypatch, failure):
    from tools import elendril_v7_campaign as campaign
    import subprocess
    monkeypatch.setattr(campaign, 'device_env', lambda _: {})
    def invoke_process(*args, **kwargs):
        if failure == 'timeout': raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(campaign.subprocess, 'run', invoke_process)
    result = campaign.invoke('train', {'output': str(tmp_path), 'source': 'a'*40,
                                      'timecap_seconds': 1}, 'unused')
    assert result['status'] == 'ERROR'
    receipt = json.loads(__import__('pathlib').Path(result['receipt_path']).read_bytes())
    assert receipt['status'] == 'ERROR'
    assert receipt['error_type'] == ('TimeoutExpired' if failure == 'timeout' else 'RuntimeError')


def test_rescore_rejects_diagnostic_checkpoint_mismatch(tmp_path, monkeypatch):
    from tools import elendril_v7_campaign as campaign
    import anra_v5.elendril.v6_data as v6_data
    import anra_v5.elendril.v7_diagnostics as diag
    monkeypatch.setattr(v6_data, 'read_dataset', lambda path: SimpleNamespace(identity=path.stem))
    monkeypatch.setattr(diag, 'verify_diagnosis', lambda *_: {})
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    for name in ('canary', 'monitor', 'validation', 'confirmation'):
        (data_dir / (name + '.json')).write_text('{}')
    folder = tmp_path / 'analysis' / 'jobs' / 'one'
    folder.mkdir(parents=True)
    (folder / 'diagnostic-u1.json').write_text(json.dumps({'data_identity':'validation',
        'checkpoint':'expected', 'diagnosis':{'records':{'interactive': {'raw': {'model_identity':'other'}}}}}))
    with pytest.raises(ValueError, match='checkpoint identity'):
        campaign.rescore(tmp_path, {})


def test_rescore_rejects_confirmation_checkpoint_mismatch(tmp_path, monkeypatch):
    from tools import elendril_v7_campaign as campaign
    import anra_v5.elendril.v6_data as v6_data
    import anra_v5.elendril.v7_diagnostics as diag
    monkeypatch.setattr(v6_data, 'read_dataset', lambda path: SimpleNamespace(identity=path.stem))
    monkeypatch.setattr(diag, 'verify_diagnosis', lambda *_: {})
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    for name in ('canary', 'monitor', 'validation', 'confirmation'):
        (data_dir / (name + '.json')).write_text('{}')
    folder = tmp_path / 'analysis' / 'jobs' / 'one'
    folder.mkdir(parents=True)
    (folder / 'confirmation.json').write_text(json.dumps({'data_identity':'confirmation',
        'checkpoint':'expected', 'diagnosis':{'records':{'interactive': {'raw': {'model_identity':'other'}}}}}))
    with pytest.raises(ValueError, match='checkpoint identity'):
        campaign.rescore(tmp_path, {})
