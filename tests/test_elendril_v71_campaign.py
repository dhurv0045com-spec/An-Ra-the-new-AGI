"""CPU-only campaign admission and delivery integration; no research jobs."""
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

import pytest
import torch

from anra_v5.elendril.v6_data import make_dataset
from anra_v5.elendril.v6_evaluation import NeuralPolicy
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.v71_data import suite_to_record
from anra_v5.elendril.v71_evaluation import evaluate_suite
from anra_v5.elendril.v71_protocol import PROTOCOL_SHA, jobs
from tools import elendril_v71_campaign as campaign
from tools import elendril_v71_export as export
from v5_training.checkpoint import REQUIRED_COMPONENTS

SOURCE = 'a'*40


def test_owner_and_failed_gpu_admission_never_invoke(tmp_path, monkeypatch):
    target = tmp_path/'run'
    with pytest.raises(ValueError, match='owner'):
        campaign.run(output=target, source=SOURCE)
    assert not target.exists()
    monkeypatch.setattr(campaign, 'check_source', lambda *_: None)
    monkeypatch.setattr(campaign.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100<<30))
    monkeypatch.setattr(campaign, '_gpu_records', lambda: (_ for _ in ()).throw(ValueError('no GPUs')))
    monkeypatch.setattr(campaign, 'invoke', lambda *a, **k: pytest.fail('worker started after admission failure'))
    result = campaign.run(output=target, source=SOURCE, owner_enabled=True, minutes=20)
    assert result['status'] != 'COMPLETE'
    assert 'no GPUs' in result['message']
    assert Path(result['delivery']['delivery_receipt']).is_file()


def test_forecast_charges_all_jobs_to_two_lanes():
    phase = [{'profile':'S3','mechanism':'ce','epochs':4} for _ in range(3)]
    profiles = [{'profile':'S3','job':{'mechanism':'ce'},'seconds_per_update':2,
                 'evaluation_seconds_per_episode':3}]
    data = {'train':SimpleNamespace(samples=list(range(32)))}
    suite = {'base':[1,2], 'hop-3':[1], 'other':[1,2,3]}
    forecast = campaign.forecast(phase, profiles, data, suite)
    per_job = 2*(8*2 + (3*3 + 2*6)*3)
    assert forecast['job_seconds'] == [per_job]*3
    assert forecast['seconds'] == 2*per_job + 600


class FakeBackend:
    def __init__(self, identity):
        self.binding=SimpleNamespace(tokenizer_sha256=UTF8.identity, vocabulary_size=UTF8.vocabulary_size,
            checkpoint_sha256=identity, identity=identity, context_length=512)
    def next_distribution(self, ids):
        result=torch.zeros(UTF8.vocabulary_size)
        result[UTF8.eos_id if ids[-1] == UTF8.encode('A')[0] else UTF8.encode('A')[0]] = 1
        return result


def synthetic_checkpoint(root, job_id):
    components=[(name, ('fixture '+name).encode()) for name in sorted(REQUIRED_COMPONENTS)]
    manifest=json.dumps({'components':[{'name':n,'sha256':sha256(b).hexdigest(),'byte_size':len(b)}
        for n,b in components]},sort_keys=True).encode()
    identity=sha256(manifest).hexdigest()
    directory=root/'checkpoints'/'jobs'/job_id/job_id/'objects'/identity
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'manifest.json').write_bytes(manifest)
    for name,content in components: (directory/name).write_bytes(content)
    return identity


@pytest.fixture
def short_output():
    # Windows default MAX_PATH is shorter than pytest's descriptive directory.
    with tempfile.TemporaryDirectory(prefix='v71-') as directory:
        yield Path(directory)/'run'


@pytest.mark.parametrize('scenario', ['complete','missing_grammar','zip_failure'])
def test_campaign_finalizer_gates_completion(short_output, monkeypatch, scenario):
    output=short_output
    data=make_dataset(worlds=4)
    suite={'base':data.cases[:1], 'hop-3':data.cases[1:2]}
    identity=synthetic_checkpoint(output, 'template')
    full=evaluate_suite(suite, NeuralPolicy(FakeBackend(identity)))
    short=evaluate_suite(suite, NeuralPolicy(FakeBackend(identity)), conditions=['base','hop-3'])
    def prepare(root):
        folder=Path(root)/'data'; folder.mkdir(parents=True,exist_ok=True)
        (folder/'suite.json').write_text(json.dumps(suite_to_record(suite)))
        return {'train':data, 'replay':data, 'new-b':data},suite
    monkeypatch.setattr(campaign,'prepare_data',prepare)
    monkeypatch.setattr(campaign,'check_source',lambda *_: None)
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda _:SimpleNamespace(free=100<<30))
    monkeypatch.setattr(campaign,'_gpu_records',lambda:['fake0','fake1'])
    monkeypatch.setattr(campaign,'check_torch',lambda *_:{'fixture':True})
    monkeypatch.setattr(campaign,'forecast',lambda phase,*_: {'seconds':1,'job_seconds':[1]*len(phase)})
    original_export=export.export_results
    def controlled_export(*args,**kwargs):
        if not kwargs.get('full',True): return {}
        if scenario=='zip_failure':
            return {'analysis':{'status':'EXPORT_DEFERRED'},'full':{'status':'EXPORT_DEFERRED'}}
        return original_export(*args,**kwargs)
    monkeypatch.setattr(export,'export_results',controlled_export)
    def invoke(mode,req,gpu,**kwargs):
        if mode=='canary': return {'status':'PASS','attempt_wall_seconds':0}
        if mode=='profile': return {'status':'PASS','profile':req['job']['profile'],
            'seconds_per_update':1,'evaluation_seconds_per_episode':1,'attempt_wall_seconds':0}
        j=req['job']; folder=Path(req['output']); folder.mkdir(parents=True,exist_ok=True)
        assert synthetic_checkpoint(output,j['job_id']) == identity
        for epoch in (1,2,4):
            row={'checkpoint':identity,'update':epoch,'grammar':False,'evaluation':full if epoch==4 else short}
            (folder/f'epoch-{epoch}.json').write_text(json.dumps(row))
        if scenario != 'missing_grammar' or j != jobs('core')[0]:
            (folder/'grammar-final.json').write_text(json.dumps({'checkpoint':identity,'update':4,'grammar':True,'evaluation':full}))
        return {'status':'COMPLETE','attempt_wall_seconds':0}
    monkeypatch.setattr(campaign,'invoke',invoke)
    result=campaign.run(output=output,source=SOURCE,phase='core',owner_enabled=True,minutes=60)
    if scenario=='complete':
        assert result['status']=='COMPLETE'
        assert result['completed_jobs']==len(jobs('core'))
        assert result['independent_replay']['status']=='PASS'
        assert len(result['delivery']['verified'])==2
    else:
        assert result['status']!='COMPLETE'
    assert json.loads((output/'exports'/'FINAL_STATUS.json').read_bytes())['status']==result['status']


def test_frozen_coverage_requires_matching_full_cohort_and_checkpoint(tmp_path):
    data=make_dataset(worlds=4)
    suite={'base':data.cases[:1], 'hop-3':data.cases[1:2]}
    identity='a'*64
    report=evaluate_suite(suite, NeuralPolicy(FakeBackend(identity)))
    j={'job_id':'uniform-s7','profile':'S3','method':'uniform','seed':7,'epochs':0,'mechanism':'ce'}
    folder=tmp_path/'analysis'/'jobs'/j['job_id']; folder.mkdir(parents=True)
    row={'checkpoint':identity,'update':0,'grammar':False,'evaluation':report}
    (folder/'epoch-0.json').write_text(json.dumps(row))
    grammar={**row,'grammar':True}
    (folder/'grammar-final.json').write_text(json.dumps(grammar))
    required=campaign.coverage(tmp_path,[j],suite)
    assert len(required)==1 and identity in required[0]
    grammar['checkpoint']='b'*64
    (folder/'grammar-final.json').write_text(json.dumps(grammar))
    with pytest.raises(ValueError, match='grammar'):
        campaign.coverage(tmp_path,[j],suite)


def test_chain_surface_roundtrip_uses_canonical_records_and_rejects_tampering():
    import copy
    from dataclasses import asdict, replace
    from anra_v5.elendril.v71_data import ChainSurface, suite_manifest, suite_from_record
    data=make_dataset(worlds=7)
    case=next(c for c in data.cases if c.family=='route')
    case=replace(case, surface=ChainSurface(**asdict(case.surface)))
    suite={'hop-2':(case,)}
    raw=suite_to_record(suite)
    restored=suite_from_record(raw)
    assert restored != suite  # canonical hop-2 reader uses the V6 Surface class
    assert suite_manifest(restored) == suite_manifest(suite)
    assert suite_to_record(restored) == raw
    broken=copy.deepcopy(raw)
    broken['cohorts']['hop-2'][0]['truth']['expected']='A' if case.truth.expected!='A' else 'B'
    with pytest.raises(ValueError): suite_from_record(broken)


def test_v715_explicit_parent_source_is_registered():
    import inspect
    body = inspect.getsource(campaign.run)
    assert "6104ffd7134cb4cf4b3f340bd060d24a1e2dd200" in body
    assert "parent_core_source" in body
