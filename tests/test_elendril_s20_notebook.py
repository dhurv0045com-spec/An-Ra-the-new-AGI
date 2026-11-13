"""Exercise the S20 notebook's normal launch and immutable publication."""
import ast
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import types
from unittest.mock import Mock, patch

import pytest

from tools.build_elendril_s20_notebook import build, DEST
from tools.elendril_v71_campaign import selected_jobs


def test_owner_run_all_selects_all_s20_jobs_and_budget():
    pin = 'a' * 40
    notebook = build(pin)
    scope = {}
    for cell in notebook['cells'][1:]:
        ast.parse(''.join(cell['source']))
    exec(''.join(notebook['cells'][1]['source']), scope)
    assert scope['RUN_STUDY'] is True and scope['PARENT_ARCHIVE'] is None
    module = types.ModuleType('IPython.display')
    module.FileLink = str
    module.display = Mock()
    with patch.dict('sys.modules', {'IPython.display': module}), \
         patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.is_symlink', return_value=False), \
         patch('pathlib.Path.is_dir', return_value=True), \
         patch('pathlib.Path.is_file', return_value=False), \
         patch('subprocess.check_output', side_effect=[pin+'\n', '']), \
         patch('subprocess.run') as run:
        exec(''.join(notebook['cells'][2]['source']), scope)
    command = run.call_args.args[0]
    assert command[command.index('--minutes')+1] == '200'
    assert command[command.index('--profile-subset')+1] == 'S20'
    assert command[command.index('--phase')+1] == 'scale'
    assert '--seed-subset' not in command
    jobs = selected_jobs('scale', 'S20')
    assert len(jobs) == 6 and {j['seed'] for j in jobs} == {7,19,41}
    assert {j['method'] for j in jobs} == {'uniform','blocked'}
    assert all(j['epochs'] == 4 for j in jobs)


def test_invalid_pin_is_rejected():
    with pytest.raises(ValueError):
        build('Elendril')


def test_published_s20_rebuilds_from_immutable_source(tmp_path):
    if not DEST.exists():
        pytest.skip('immutable source commit must precede publication')
    notebook = json.loads(DEST.read_bytes())
    pin = notebook['metadata']['source_reference']
    root = DEST.parents[1]
    archive = subprocess.check_output(['git','archive','--format=tar',pin],cwd=root)
    checkout = tmp_path/'source'
    checkout.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(checkout,filter='data')
    target = tmp_path/'rebuilt.ipynb'
    subprocess.run([sys.executable,'-m','tools.build_elendril_s20_notebook','--source-sha',pin,'--output',str(target)],
        cwd=checkout,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},timeout=60)
    assert notebook == json.loads(target.read_bytes()) == build(pin)
