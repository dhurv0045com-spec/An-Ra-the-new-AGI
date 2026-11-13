import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.build_elendril_v717_notebook import build
from tools.elendril_v717_campaign import run, _measurement_coverage, _profile_forecast, storage_admission
from anra_v5.elendril.v717_protocol import jobs


def test_notebook_source_launch_and_safety_contract():
    nb = build('a'*40)
    for c in nb['cells']:
        if c['cell_type']=='code': ast.parse(''.join(c['source']))
    config = {}; exec(''.join(nb['cells'][1]['source']),config)
    assert config['RUN_STUDY'] is True
    assert config['PHASE']=='optimization' and config['RESTORE_ARCHIVE'] is None
    launch=''.join(nb['cells'][2]['source'])
    assert 'tools.elendril_v717_campaign' in launch and '--owner-enabled' in launch
    assert "['git','fetch','--depth=1','origin',SOURCE_SHA]" in launch
    assert 'pip install' not in launch
    for invalid in ('Elendril','z'*40,'a'*39):
        with pytest.raises(ValueError): build(invalid)


def test_owner_gate_and_session_bounds(tmp_path):
    with pytest.raises(ValueError,match='owner'):
        run(output=tmp_path,source='a'*40)
    for minutes in (0,331,float('nan'),True):
        with pytest.raises(ValueError,match='budget'):
            run(output=tmp_path,source='a'*40,owner_enabled=True,minutes=minutes)


def test_storage_stages_full_archive_and_fails_before_training(tmp_path, monkeypatch):
    from types import SimpleNamespace
    with patch('tools.elendril_v717_campaign.shutil.disk_usage',return_value=SimpleNamespace(free=20*(1<<30))):
        row=storage_admission(tmp_path,jobs('optimization'))
    assert row['existing_raw_bytes']==0
    assert 2*row['projected_raw_bytes'] < row['required_additional_bytes'] < 20*(1<<30)
    with patch('tools.elendril_v717_campaign.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
        with pytest.raises(ValueError,match='storage needs'):
            storage_admission(tmp_path,jobs('optimization'))


def test_missing_measurement_never_complete(tmp_path):
    with pytest.raises(ValueError,match='required measurement'):
        _measurement_coverage(tmp_path,jobs('optimization'))


def test_forecast_matches_actual_two_lanes_and_requires_audit_timing():
    registered=jobs('optimization')
    profiles=[dict(status='PASS',measurement_seconds_per_audit=2) for _ in range(4)]
    with patch('tools.elendril_v71_campaign.forecast',return_value={
            'seconds':1,'job_seconds':[10]*8,'policy':'test'}):
        estimate=_profile_forecast(registered,profiles,{}, {})
        assert estimate['seconds']==4*(10+4*2*2)+600
    profiles[0]['measurement_seconds_per_audit']=None
    with pytest.raises(ValueError,match='timing'):
        _profile_forecast(registered,profiles,{}, {})


def test_preflight_failure_exports_real_partial_archives(tmp_path):
    from tools.elendril_v6_export import verify_zip
    with patch('tools.elendril_v717_campaign.check_source',side_effect=ValueError('pin mismatch')):
        result=run(output=tmp_path/'campaign',source='a'*40,owner_enabled=True)
    assert result['status']=='PARTIAL'
    assert not result['delivery']['completion_eligible']
    assert result['delivery']['execution']['message']=='pin mismatch'
    for item in result['exports'].values():
        assert item['status']=='EXPORT_VERIFIED'
        assert verify_zip(Path(item['zip_path']))['source']=='a'*40


def test_published_notebook_exact_reconstruction():
    path=Path(__file__).resolve().parents[1]/'notebooks'/'ELENDRIL_V717_KAGGLE_T4X2.ipynb'
    if not path.exists(): pytest.skip('publication follows immutable runtime commit')
    published=json.loads(path.read_text(encoding='utf-8'))
    assert published==build(published['metadata']['source_reference'])


def test_actual_launch_cell_invokes_pinned_campaign(tmp_path, monkeypatch):
    import sys
    import types
    import subprocess
    sha='a'*40; nb=build(sha); cfg={}
    exec(''.join(nb['cells'][1]['source']),cfg)
    repo=tmp_path/('elendril-v717-source-'+sha); repo.mkdir()
    cfg.update(WORKING=str(tmp_path),OUTPUT=tmp_path/'output')
    display=types.ModuleType('IPython.display')
    display.FileLink=str; display.display=lambda value:None
    monkeypatch.setitem(sys.modules,'IPython.display',display)
    calls=[]
    monkeypatch.setattr(subprocess,'run',lambda args,**kwargs:calls.append((args,kwargs)))
    monkeypatch.setattr(subprocess,'check_output',lambda args,**kwargs:sha if 'rev-parse' in args else '')
    launch=''.join(nb['cells'][2]['source']).replace("Path('/kaggle/working')",'Path(WORKING)')
    exec(launch,cfg)
    assert len(calls)==1
    cmd,kwargs=calls[0]
    assert cmd[1:3]==['-m','tools.elendril_v717_campaign']
    assert cmd[cmd.index('--source')+1]==sha
    assert cmd[cmd.index('--phase')+1]=='optimization'
    assert kwargs['cwd']==repo and kwargs['check'] is True


def test_v717_full_archive_restore_identity(tmp_path):
    from tools import elendril_v71_export as export
    from anra_v5.elendril.v717_protocol import PROTOCOL_SHA
    src=tmp_path/'source'; (src/'analysis').mkdir(parents=True)
    identity={'source':'a'*40,'protocol_sha256':PROTOCOL_SHA,'phase':'optimization'}
    (src/'analysis'/'identity.json').write_text(json.dumps(identity))
    bundle=export.export_results(src,source='a'*40,protocol_sha=PROTOCOL_SHA,
                                 campaign={'status':'PARTIAL'},rescore=lambda *_:{'status':'NOT_RUN'})
    restored=export.restore_archive(bundle['full']['zip_path'],tmp_path/'restored',source='a'*40,phase='optimization')
    assert restored['archive_identity']==identity
    for phase in ('representation','core'):
        with pytest.raises(ValueError):
            export.restore_archive(bundle['full']['zip_path'],tmp_path/phase,source='a'*40,phase=phase)
