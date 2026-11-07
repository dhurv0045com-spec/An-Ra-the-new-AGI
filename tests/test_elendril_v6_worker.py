import json
from pathlib import Path
import pytest
from tools.elendril_v6_worker import admit, copy_checkpoint


def test_requires_owner(tmp_path):
    request = tmp_path / 'request.json'
    request.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='owner-enabled'):
        admit(request, tmp_path / 'receipt.json', False)


def test_copy_checkpoint_verified_and_no_overwrite(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'LATEST').write_text('abc', encoding='ascii')
    destination = tmp_path / 'copy'
    copy_checkpoint(source, destination)
    assert (destination / 'LATEST').read_text(encoding='ascii') == 'abc'
    with pytest.raises(ValueError, match='new'):
        copy_checkpoint(source, destination)
