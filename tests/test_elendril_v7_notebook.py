import ast
import json
from pathlib import Path
from unittest.mock import patch, Mock
import types

import pytest

from tools.build_elendril_v7_notebook import build

PIN = 'a' * 40


def test_pin_and_code_syntax():
    notebook = build(PIN)
    assert notebook['metadata']['source_reference'] == PIN
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
            assert cell['outputs'] == [] and cell['execution_count'] is None
    content = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
    assert 'RUN_STUDY = False' in content
    assert 'tools.elendril_v7_campaign' in content
    assert "'--owner-enabled', '--source', SOURCE_SHA" in content
    assert 'ELENDRIL_V7_ANALYSIS.zip' in content and 'ELENDRIL_V7_FULL.zip' in content
    assert 'pip install' not in content


@pytest.mark.parametrize('invalid', ['a' * 39, 'A' * 40, 'z' * 40, '', None])
def test_rejects_invalid_pin(invalid):
    with pytest.raises(ValueError):
        build(invalid)


def test_disabled_does_not_run_campaign_or_create_directories():
    notebook = build(PIN)
    scope = {}
    display = Mock()
    ipython = types.ModuleType('IPython')
    display_module = types.ModuleType('IPython.display')
    display_module.display = display
    display_module.FileLink = lambda path: path
    with patch.dict('sys.modules', {'IPython': ipython, 'IPython.display': display_module}), \
         patch('subprocess.run', side_effect=AssertionError('subprocess launched')), \
         patch('pathlib.Path.mkdir', side_effect=AssertionError('directory created')):
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                exec(''.join(cell['source']), scope)
    assert display.call_count == 2


def test_generated_notebook_matches_generator():
    path = Path(__file__).resolve().parents[1] / 'notebooks' / 'ELENDRIL_V7_KAGGLE_T4X2.ipynb'
    notebook = json.loads(path.read_text(encoding='utf8'))
    assert notebook == build(notebook['metadata']['source_reference'])


def test_enabled_existing_checkout_launches_exact_pinned_cli_and_displays_failure_outputs():
    notebook=build(PIN); scope={}
    exec(''.join(notebook['cells'][1]['source']),scope)
    scope['RUN_STUDY']=True
    display=Mock(); module=types.ModuleType('IPython.display')
    module.display=display; module.FileLink=lambda path:path
    import subprocess
    with patch.dict('sys.modules',{'IPython.display':module}), \
         patch('pathlib.Path.exists',return_value=True), \
         patch('pathlib.Path.is_symlink',return_value=False), \
         patch('pathlib.Path.is_dir',return_value=True), \
         patch('pathlib.Path.is_file',return_value=False), \
         patch('subprocess.check_output',side_effect=[PIN+'\n','']), \
         patch('subprocess.run',side_effect=subprocess.CalledProcessError(1,'campaign')) as run:
        with pytest.raises(RuntimeError,match='Campaign failed or is incomplete'):
            exec(''.join(notebook['cells'][2]['source']),scope)
    command=run.call_args.args[0]
    assert command[1:3]==['-m','tools.elendril_v7_campaign']
    assert command[-3:]==['--owner-enabled','--source',PIN]
    assert display.call_count==2
