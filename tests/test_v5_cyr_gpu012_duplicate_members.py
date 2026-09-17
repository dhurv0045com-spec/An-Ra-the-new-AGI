"""CPU-only archive identity regression tests."""
import hashlib
from zipfile import ZipFile

import pytest

from anra_v5.cyr_gpu012_evidence import verify_bundle


def declaration(path, names):
    return dict(bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                json_files=names)


def test_duplicate_members_rejected(tmp_path):
    path = tmp_path / 'ambiguous.zip'
    with ZipFile(path, 'w') as archive:
        archive.writestr('result.json', 'not JSON')
        with pytest.warns(UserWarning, match='Duplicate name'):
            archive.writestr('result.json', '{"valid": true}')
    with pytest.raises(ValueError, match='duplicate bundle member'):
        verify_bundle(path, declaration(path, ['result.json', 'result.json']))


def test_unique_members_validate(tmp_path):
    path = tmp_path / 'valid.zip'
    with ZipFile(path, 'w') as archive:
        archive.writestr('result.json', '{"valid": true}')
    assert verify_bundle(path, declaration(path, ['result.json'])) is True


def test_unique_invalid_json_rejected(tmp_path):
    path = tmp_path / 'invalid.zip'
    with ZipFile(path, 'w') as archive:
        archive.writestr('result.json', 'not JSON')
    with pytest.raises(ValueError):
        verify_bundle(path, declaration(path, ['result.json']))


def test_duplicate_declaration_rejected(tmp_path):
    path = tmp_path / 'valid.zip'
    with ZipFile(path, 'w') as archive:
        archive.writestr('result.json', '{}')
    with pytest.raises(ValueError):
        verify_bundle(path, declaration(path, ['result.json', 'result.json']))
