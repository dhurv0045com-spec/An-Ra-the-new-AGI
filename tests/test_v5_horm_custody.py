"""Read-only custody inventory, not scientific-result validation."""
import hashlib
import json
from pathlib import Path

import pytest

from anra_v5 import horm_custody_audit as audit


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _fixture(root, body=b'{"a": 1}\n', expected=None, missing=False):
    results = root / 'results'
    results.mkdir(parents=True)
    (results / 'lane_A.json').write_bytes(body)
    artifacts = {'lane_A.json': expected or _sha(body)}
    if missing:
        artifacts['lane_A.pt'] = _sha(b'weights')
    (results / 'RESULT.json').write_bytes(json.dumps(
        {'artifact_sha256': artifacts, 'verdict': 'X', 'error': None}).encode())
    return results


def test_raw_match_is_only_declared_byte_consistency(tmp_path):
    _fixture(tmp_path)
    report = audit.audit_line('LINE', tmp_path)
    assert report['artifacts']['lane_A.json']['classification'] == audit.RAW_MATCH
    assert report['status'] == audit.INTEGRITY_OK
    assert report['scientific_validation'] == 'NOT_PERFORMED'


def test_missing_checkpoint_has_exact_recovery_identity(tmp_path):
    _fixture(tmp_path, missing=True)
    report = audit.audit_line('LINE', tmp_path)
    entry = report['artifacts']['lane_A.pt']
    assert entry['classification'] == audit.MISSING
    assert report['status'] == audit.ARTIFACT_INTEGRITY_BLOCKED
    assert entry['recovery']['declared_sha256'] == _sha(b'weights')
    assert entry['recovery']['expected_path'] == 'results/lane_A.pt'


def test_eol_reconstruction_never_passes_raw_verification(tmp_path):
    original = b'{"a": 1}\n'
    results = _fixture(tmp_path, expected=_sha(b'{"a": 1}\r\n'))
    report = audit.audit_line('LINE', tmp_path)
    entry = report['artifacts']['lane_A.json']
    assert entry['classification'] == audit.EOL_RECONSTRUCTED
    assert entry['raw_sha256'] == _sha(original)
    assert entry['eol_reconstructed_sha256'] == _sha(b'{"a": 1}\r\n')
    assert report['status'] == audit.ARTIFACT_INTEGRITY_BLOCKED
    assert (results / 'lane_A.json').read_bytes() == original


def test_unexplained_hash_mismatch(tmp_path):
    _fixture(tmp_path, expected=_sha(b'other'))
    report = audit.audit_line('LINE', tmp_path)
    assert report['artifacts']['lane_A.json']['classification'] == audit.HASH_MISMATCH
    assert report['status'] == audit.ARTIFACT_INTEGRITY_BLOCKED


def test_real_repository_horm_lines():
    root = Path(__file__).resolve().parents[1]
    for line, count, missing in [('HORM-001', 9, 4), ('HORM-002', 20, 9)]:
        report = audit.audit_line(line, root / 'experiments' / line)
        assert len(report['artifacts']) == count
        assert report['status'] == audit.ARTIFACT_INTEGRITY_BLOCKED
        assert sum(a['classification'] == audit.MISSING
                   for a in report['artifacts'].values()) == missing
        assert all(a['classification'] in (audit.RAW_MATCH, audit.EOL_RECONSTRUCTED)
                   for n, a in report['artifacts'].items() if n.endswith('.json'))


@pytest.mark.parametrize('manifest', [
    '{"artifact_sha256": {"a.pt": "' + 'a' * 64 + '", "a.pt": "' + 'b' * 64 + '"}}',
    '{"artifact_sha256": {}}',
    '{"artifact_sha256": {"../outside.pt": "' + 'a' * 64 + '"}}',
    '{"artifact_sha256": {"C:/outside.pt": "' + 'a' * 64 + '"}}',
    '{"artifact_sha256": {"a.pt": "bad-hash"}}',
    '{"artifact_sha256": {"a.pt": "' + 'a' * 64 + '", "A.pt": "' + 'b' * 64 + '"}}',
    '{"artifact_sha256": {"a.pt:stream": "' + 'a' * 64 + '"}}',
    '{"artifact_sha256": null}',
    '{"artifact_sha256": {"a.pt": NaN}}',
    'not json',
])
def test_rejects_invalid_or_ambiguous_manifest(tmp_path, manifest):
    results = tmp_path / 'results'
    results.mkdir()
    (results / 'RESULT.json').write_text(manifest, encoding='utf-8')
    with pytest.raises(ValueError):
        audit.audit_line('LINE', tmp_path)


def test_hash_matching_malformed_json_is_not_valid_evidence(tmp_path):
    _fixture(tmp_path, body=b'not json')
    with pytest.raises(ValueError):
        audit.audit_line('LINE', tmp_path)


def test_resolved_path_escape_is_rejected(tmp_path):
    results = tmp_path / 'results'
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'payload.pt').write_bytes(b'weights')
    link = results / 'link'
    link.mkdir(parents=True)
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('symlink creation unsupported on this host')
    escape = 'link/payload.pt'
    manifest = {'artifact_sha256': {escape: _sha(b'weights')},
                'verdict': 'X', 'error': None}
    (results / 'RESULT.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError):
        audit.audit_line('LINE', tmp_path)


def test_cli_exit_zero_on_fully_matching_synthetic_tree(tmp_path):
    for line in ('HORM-001', 'HORM-002'):
        results = tmp_path / 'experiments' / line / 'results'
        results.mkdir(parents=True)
        body = b'{"lane": 1}\n'
        (results / 'lane_A.json').write_bytes(body)
        (results / 'RESULT.json').write_text(json.dumps(
            {'artifact_sha256': {'lane_A.json': _sha(body)},
             'verdict': 'X', 'error': None}), encoding='utf-8')
    out = tmp_path / 'synthetic_report.json'
    assert audit.main(['--root', str(tmp_path), '--out', str(out)]) == 0
    report = json.loads(out.read_text(encoding='utf-8'))
    assert all(r['status'] == audit.INTEGRITY_OK
               for r in report['lines'].values())


def test_cli_never_overwrites_existing_evidence(tmp_path):
    out = tmp_path / 'existing.json'
    out.write_bytes(b'preserved evidence')
    with pytest.raises(FileExistsError):
        audit.main(['--out', str(out)])
    assert out.read_bytes() == b'preserved evidence'


def test_cli_reports_blocked_with_nonzero_exit(tmp_path):

    out = tmp_path / 'audit.json'
    assert audit.main(['--out', str(out)]) == 2
    data = json.loads(out.read_text())
    assert data['schema'] == audit.SCHEMA
    assert set(data['lines']) == {'HORM-001', 'HORM-002'}
    assert data['fail_closed_notes']
    assert data['diagnostic_sha256'] == _sha(Path(audit.__file__).read_bytes())
