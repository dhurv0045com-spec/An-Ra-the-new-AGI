"""Verify the published notebook against its own immutable source tree."""
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile


def test_published_notebook_rebuilds_from_pinned_commit(tmp_path):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / 'notebooks/ELENDRIL_V7_KAGGLE_T4X2.ipynb').read_text(encoding='utf8'))
    source = notebook['metadata']['source_reference']
    assert re.fullmatch(r'[0-9a-f]{40}', source)
    subprocess.run(['git', 'merge-base', '--is-ancestor', source, 'HEAD'], cwd=root, check=True)
    archive = subprocess.check_output(['git', 'archive', '--format=tar', source], cwd=root)
    checkout = tmp_path / 'source'
    checkout.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(checkout, filter='data')
    rebuilt = tmp_path / 'rebuilt.ipynb'
    subprocess.run([sys.executable, '-m', 'tools.build_elendril_v7_notebook',
                    '--source-sha', source, '--output', str(rebuilt)], cwd=checkout,
                   env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, check=True,
                   capture_output=True, timeout=60)
    assert notebook == json.loads(rebuilt.read_text(encoding='utf8'))
    assert notebook['nbformat'] == 4
    intro = ''.join(notebook['cells'][0]['source'])
    config = ''.join(notebook['cells'][1]['source'])
    assert 'RUN_STUDY = True' in intro and 'RUN_STUDY = True' in config
