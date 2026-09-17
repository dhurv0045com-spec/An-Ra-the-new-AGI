"""CPU-only evidence audit; no model construction or scientific training."""
import importlib
import importlib.util
import json
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest


def test_verify_bundle_rejects_bad_hash(tmp_path):
    assert importlib.util.find_spec('anra_v5.cyr_gpu012_evidence') is not None, 'evidence verifier missing'
    module = importlib.import_module('anra_v5.cyr_gpu012_evidence')
    path = tmp_path / 'evidence.zip'
    with ZipFile(path, 'w', ZIP_DEFLATED) as z:
        z.writestr('x.json', json.dumps({'x': 1}))
    with pytest.raises(ValueError, match='bundle SHA'):
        module.verify_bundle(path, {'sha256': '0' * 64, 'bytes': path.stat().st_size,
                                    'json_files': ['x.json']})


def test_verify_bundle_requires_declared_members(tmp_path):
    from anra_v5.cyr_gpu012_evidence import verify_bundle
    path = tmp_path / 'evidence.zip'
    with ZipFile(path, 'w', ZIP_DEFLATED) as z:
        z.writestr('x.json', json.dumps({'x': 1}))
    declared = dict(bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='declared JSON file list'):
        verify_bundle(path, declared)
    declared['json_files'] = ['x.json']
    assert verify_bundle(path, declared) is True
    declared['json_files'] = ['other.json']
    with pytest.raises(ValueError, match='file list mismatch'):
        verify_bundle(path, declared)

