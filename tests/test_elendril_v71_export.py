"""Small scripted archive fixtures; no model training."""
import json
import os
import zipfile
import pytest
from hashlib import sha256
from pathlib import Path

from tools import elendril_v71_export as export
from v5_training.checkpoint import REQUIRED_COMPONENTS

SOURCE = 'a'*40
PROTOCOL = 'b'*64


def setup_evidence(root):
    job = 'S3-uniform-ce-s7'
    directory = root / 'analysis/jobs' / job
    directory.mkdir(parents=True)
    (directory / 'epoch-4.json').write_text('{"status":"fixture"}', encoding='utf8')
    (directory / 'grammar-final.json').write_text('{"status":"fixture"}', encoding='utf8')
    components = [(name, ('fixture '+name).encode()) for name in sorted(REQUIRED_COMPONENTS)]
    manifest = json.dumps({'components': [{'name': n, 'sha256': sha256(b).hexdigest(), 'byte_size': len(b)} for n,b in components]}, sort_keys=True).encode()
    identity = sha256(manifest).hexdigest()
    prefix = Path('checkpoints/jobs') / job / job / 'objects' / identity
    store = root / prefix
    store.mkdir(parents=True)
    (store / 'manifest.json').write_bytes(manifest)
    for name, contents in components: (store/name).write_bytes(contents)
    return job, (prefix/'manifest.json').as_posix()


def campaign():
    return {'run_id': 'fixture', 'status': 'EXECUTED', 'source': SOURCE,
            'independent_replay': {'status': 'PASS'}, 'summary': {'status': 'AVAILABLE'}}


def test_finalizer_checks_dual_archives_and_checkpoint_components(tmp_path):
    job, manifest = setup_evidence(tmp_path)
    result = export.finalize(tmp_path, SOURCE, PROTOCOL, campaign(), [job], [manifest], True)
    assert result['status'] == 'COMPLETE' and len(result['verified']) == 2
    assert Path(result['delivery_receipt']).is_file()
    assert all(Path(result['exports'][k]['zip_path']).name == export._NAMES[k] for k in ('analysis', 'full'))
    assert json.loads((tmp_path/'exports/FINAL_STATUS.json').read_bytes())['status'] == 'COMPLETE'


def test_missing_results_and_incomplete_study_fail_closed(tmp_path):
    job, manifest = setup_evidence(tmp_path)
    (tmp_path/'analysis/jobs'/job/'grammar-final.json').unlink()
    result = export.finalize(tmp_path, SOURCE, PROTOCOL, campaign(), [job], [manifest], True)
    assert result['status'] == 'EXPORT_FAILED'
    assert any('grammar report absent' in e for e in result['errors'])
    result = export.finalize(tmp_path, SOURCE, PROTOCOL, campaign(), [], [], False)
    assert result['status'] == 'PARTIAL'


def test_failed_copy_preserves_previous_v71_archives(tmp_path, monkeypatch):
    setup_evidence(tmp_path)
    first = export.export_results(tmp_path, source=SOURCE, protocol_sha=PROTOCOL, campaign=campaign())
    archives = [Path(first[k]['zip_path']) for k in ('analysis', 'full')]
    before = [p.read_bytes() for p in archives]
    original = os.replace
    def fail(src, dst):
        if Path(dst).name in export._NAMES.values(): raise OSError('injected publication failure')
        return original(src, dst)
    monkeypatch.setattr(export.os, 'replace', fail)
    second = export.export_results(tmp_path, source=SOURCE, protocol_sha=PROTOCOL, campaign=campaign())
    assert all(second[k]['status'] == 'EXPORT_DEFERRED' for k in ('analysis', 'full'))
    assert [p.read_bytes() for p in archives] == before


def test_changed_checkpoint_component_rejected(tmp_path):
    job, manifest = setup_evidence(tmp_path)
    component = next(p for p in (tmp_path / Path(manifest).parent).iterdir() if p.name != 'manifest.json')
    component.write_bytes(b'changed after checkpoint publication')
    result = export.finalize(tmp_path, SOURCE, PROTOCOL, campaign(), [job], [manifest], True)
    assert result['status'] == 'EXPORT_FAILED'
    assert any('checkpoint component differs' in e for e in result['errors'])


def test_partial_missing_jobs_keeps_verified_archives(tmp_path):
    (tmp_path/'analysis').mkdir()
    result = export.finalize(tmp_path, SOURCE, PROTOCOL, campaign(), ['S3-uniform-ce-s7'], [], False)
    assert result['status'] == 'PARTIAL' and len(result['verified']) == 2


def test_verified_restore_and_bound(tmp_path, monkeypatch):
    src = tmp_path/'src'; src.mkdir(); setup_evidence(src)
    archives = export.export_results(src, source=SOURCE, protocol_sha=PROTOCOL, campaign=campaign())
    zip_path = archives['full']['zip_path']
    result = export.extract_verified(zip_path, tmp_path/'restored', reserve_bytes=0)
    assert result['status'] == 'EXTRACTION_VERIFIED'
    assert (tmp_path/'restored/analysis/jobs/S3-uniform-ce-s7/epoch-4.json').exists()
    monkeypatch.setattr(export, 'MAX_EXTRACT_BYTES', 100)
    with pytest.raises(ValueError): export.extract_verified(zip_path, tmp_path/'oversized', reserve_bytes=0)
    assert not (tmp_path/'oversized').exists()


@pytest.mark.parametrize('kind', ['traversal', 'tamper'])
def test_restore_rejects_untrusted_archive(tmp_path, kind):
    src = tmp_path/'src'; src.mkdir(); setup_evidence(src)
    archives = export.export_results(src, source=SOURCE, protocol_sha=PROTOCOL, campaign=campaign())
    source = Path(archives['analysis']['zip_path'])
    bad = tmp_path/'bad.zip'
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(bad, 'w') as target:
        for info in original.infolist():
            content = original.read(info.filename)
            if kind == 'tamper' and info.filename == 'README_RESULTS.md': content += b'tampered'
            target.writestr(info.filename, content)
        if kind == 'traversal': target.writestr('../escape', b'unsafe')
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        export.extract_verified(bad, tmp_path/'restored', reserve_bytes=0)
    assert not (tmp_path/'restored').exists() and not (tmp_path/'escape').exists()


def test_restore_admits_full_matching_identity_and_rejects_wrong_inputs(tmp_path):
    from anra_v5.elendril.v71_protocol import PROTOCOL_SHA
    src = tmp_path/'src'; src.mkdir(); setup_evidence(src)
    (src/'analysis/identity.json').write_text(json.dumps({'source':SOURCE, 'protocol_sha256':PROTOCOL_SHA, 'phase':'core'}), encoding='utf8')
    bundle = export.export_results(src, source=SOURCE, protocol_sha=PROTOCOL_SHA, campaign=campaign())
    full = bundle['full']['zip_path']; analysis = bundle['analysis']['zip_path']
    admitted = export.restore_archive(full, tmp_path/'restored', source=SOURCE, phase='core')
    assert admitted['status'] == 'EXTRACTION_VERIFIED' and admitted['restore_kind'] == 'resume'
    assert (tmp_path/'restored/analysis/identity.json').is_file()
    for candidate, source, phase, parent in [(analysis, SOURCE, 'core', False),
                                             (full, 'c'*40, 'core', False),
                                             (full, SOURCE, 'scale', False),
                                             (full, SOURCE, 'core', True)]:
        dest = tmp_path / ('rejected-' + str(len(list(tmp_path.glob('rejected-*')))))
        with pytest.raises(ValueError):
            export.restore_archive(candidate, dest, source=source, phase=phase, parent=parent)
        assert not dest.exists()
    parent = export.restore_archive(full, tmp_path/'parent', source=SOURCE, phase='continual', parent=True)
    assert parent['archive_identity']['phase'] == 'core'


def test_restore_frozen_requires_original_v7_identity(tmp_path):
    from anra_v5.elendril.v7_protocol import PROTOCOL_SHA as V7_PROTOCOL
    old_source = '413a3eccf4344ef2a39051b870e92feb6271b769'
    src = tmp_path/'src'; src.mkdir(); setup_evidence(src)
    (src/'analysis/identity.json').write_text(json.dumps({'source': old_source, 'protocol_sha256': V7_PROTOCOL}), encoding='utf8')
    bundle = export.export_results(src, source=old_source, protocol_sha=V7_PROTOCOL, campaign=campaign())
    admitted = export.restore_archive(bundle['full']['zip_path'], tmp_path/'parent', source=SOURCE,
                                      phase='frozen', parent=True)
    assert admitted['archive_identity']['source'] == old_source
    assert admitted['requested_phase'] == 'frozen'
