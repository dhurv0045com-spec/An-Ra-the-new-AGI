import ast
import json
from pathlib import Path
from unittest.mock import Mock, patch
import types
import pytest

from tools.build_elendril_v71_notebook import build

PIN = 'a' * 40


def scope_for():
    notebook = build(PIN)
    scope = {}
    exec(''.join(notebook['cells'][1]['source']), scope)
    return notebook, scope


def module():
    display = Mock()
    ipython = types.ModuleType('IPython.display')
    ipython.display, ipython.FileLink = display, lambda path: path
    return ipython, display


def test_pin_and_syntax():
    notebook, scope = scope_for()
    assert scope['RUN_STUDY'] is True and scope['PHASE'] == 'core' and scope['MINUTES'] == 330
    assert all(scope[k] is None for k in ('PARENT_OUTPUT', 'RESTORE_ARCHIVE', 'PARENT_ARCHIVE'))
    assert notebook['metadata']['source_reference'] == PIN
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
            assert cell['outputs'] == [] and cell['execution_count'] is None
    text = '\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert 'pip install' not in text
    assert 'ELENDRIL_V71_ANALYSIS.zip' in text and 'ELENDRIL_V71_FULL.zip' in text
    assert 'restore_archive' in text


@pytest.mark.parametrize('pin', ['a'*39, 'A'*40, 'z'*40, '', None])
def test_invalid_pin(pin):
    with pytest.raises(ValueError):
        build(pin)


def test_disabled_has_no_side_effects():
    notebook, scope = scope_for()
    scope['RUN_STUDY'] = False
    mod, display = module()
    with patch.dict('sys.modules', {'IPython.display': mod}), \
         patch('subprocess.run', side_effect=AssertionError('unexpected subprocess')), \
         patch('pathlib.Path.mkdir', side_effect=AssertionError('unexpected directory')):
        exec(''.join(notebook['cells'][2]['source']), scope)
    assert display.call_count == 2


def test_enabled_pinned_cli_and_parent():
    notebook, scope = scope_for()
    scope['PARENT_OUTPUT'] = '/tmp/parent'
    mod, display = module()
    with patch.dict('sys.modules', {'IPython.display': mod}), \
         patch('pathlib.Path.exists', return_value=True), patch('pathlib.Path.is_symlink', return_value=False), \
         patch('pathlib.Path.is_dir', return_value=True), patch('pathlib.Path.is_file', return_value=False), \
         patch('subprocess.check_output', side_effect=[PIN+'\n', '']), patch('subprocess.run') as run:
        exec(''.join(notebook['cells'][2]['source']), scope)
    command = run.call_args.args[0]
    assert command[1:3] == ['-m', 'tools.elendril_v71_campaign']
    assert command[command.index('--source') + 1] == PIN
    assert command[command.index('--phase') + 1] == 'core'
    assert command[command.index('--minutes') + 1] == '330'
    assert command[-2:] == ['--parent-output', '/tmp/parent']
    assert '--owner-enabled' in command and str(scope['OUTPUT']).endswith(PIN[:12] + '-core')
    assert display.call_count == 3


def test_wrong_source_fails_before_launch():
    notebook, scope = scope_for()
    mod, _ = module()
    with patch.dict('sys.modules', {'IPython.display': mod}), \
         patch('pathlib.Path.exists', return_value=True), patch('pathlib.Path.is_symlink', return_value=False), \
         patch('pathlib.Path.is_dir', return_value=True), patch('pathlib.Path.is_file', return_value=False), \
         patch('subprocess.check_output', side_effect=['b'*40+'\n', '']), patch('subprocess.run') as run:
        with pytest.raises(RuntimeError, match='Campaign failed'):
            exec(''.join(notebook['cells'][2]['source']), scope)
    run.assert_not_called()


def test_archive_extraction_and_existing_destination_rejection():
    notebook, scope = scope_for()
    scope['RESTORE_ARCHIVE'], scope['PARENT_ARCHIVE'] = '/input/restore.zip', '/input/parent.zip'
    mod, _ = module()
    def exists(path):
        return path == scope.get('REPO')
    with patch.dict('sys.modules', {'IPython.display': mod}), \
         patch('pathlib.Path.exists', exists), patch('pathlib.Path.is_symlink', return_value=False), \
         patch('pathlib.Path.is_dir', return_value=True), patch('pathlib.Path.is_file', return_value=False), \
         patch('subprocess.check_output', side_effect=[PIN+'\n', '']), patch('subprocess.run') as run:
        exec(''.join(notebook['cells'][2]['source']), scope)
    calls = [c.args[0] for c in run.call_args_list]
    assert len(calls) == 3 and all('restore_archive' in c[2] for c in calls[:2])
    assert calls[0][3:5] == ['/input/restore.zip', str(scope['OUTPUT'])]
    assert calls[0][5:] == [PIN,'core','resume']
    assert calls[1][3] == '/input/parent.zip'
    assert calls[2][-2:] == ['--parent-output', calls[1][4]]
    notebook, scope = scope_for(); scope['RESTORE_ARCHIVE'] = '/input/restore.zip'
    with patch.dict('sys.modules', {'IPython.display': mod}), \
         patch('pathlib.Path.exists', return_value=True), patch('pathlib.Path.is_symlink', return_value=False), \
         patch('pathlib.Path.is_dir', return_value=True), patch('pathlib.Path.is_file', return_value=False), \
         patch('subprocess.check_output', side_effect=[PIN+'\n', '']), patch('subprocess.run') as run:
        with pytest.raises(RuntimeError, match='Campaign failed'):
            exec(''.join(notebook['cells'][2]['source']), scope)
    run.assert_not_called()


def test_published_notebook_matches_when_present():
    path = Path(__file__).resolve().parents[1] / 'notebooks' / 'ELENDRIL_V71_KAGGLE_T4X2.ipynb'
    if not path.exists():
        pytest.skip('publication awaits immutable source commit')
    notebook = json.loads(path.read_text(encoding='utf8'))
    assert notebook == build(notebook['metadata']['source_reference'])


def test_published_notebook_rebuilds_from_immutable_source(tmp_path):
    import io
    import os
    import subprocess
    import sys
    import tarfile
    root=Path(__file__).resolve().parents[1]
    path=root/'notebooks/ELENDRIL_V71_KAGGLE_T4X2.ipynb'
    if not path.exists(): pytest.skip('publication awaits immutable source commit')
    notebook=json.loads(path.read_bytes()); pin=notebook['metadata']['source_reference']
    subprocess.run(['git','merge-base','--is-ancestor',pin,'HEAD'],cwd=root,check=True)
    archive=subprocess.check_output(['git','archive','--format=tar',pin],cwd=root)
    checkout=tmp_path/'source'; checkout.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar: tar.extractall(checkout,filter='data')
    target=tmp_path/'rebuilt.ipynb'
    subprocess.run([sys.executable,'-m','tools.build_elendril_v71_notebook','--source-sha',pin,'--output',str(target)],
        cwd=checkout,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},check=True,capture_output=True,timeout=60)
    assert notebook==json.loads(target.read_bytes())
    import nbformat
    nbformat.validate(nbformat.from_dict(notebook))
