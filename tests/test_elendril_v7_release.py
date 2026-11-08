"""Delivery-gate engineering fixtures; no model results or CUDA execution."""
import json
from types import SimpleNamespace
import pytest
from tools import elendril_v7_release as release
from anra_v5.elendril.v7_protocol import jobs,PROTOCOL_SHA


def execution():
    return {'source':'a'*40,'status':'EXECUTED','summary':{'status':'COMPLETE'},
            'independent_replay':{'status':'PASS'},'results':[]}


def isolate_coverage(monkeypatch):
    # Coverage is tested separately; these tests exercise the real ZIP path.
    monkeypatch.setattr(release,'required_results',lambda *_:{'status':'PASS','analysis_members':[],
                                                            'checkpoint_members':[]})


def test_complete_jobs_without_mandatory_reports_are_not_complete(tmp_path):
    (tmp_path/'analysis').mkdir()
    (tmp_path/'analysis'/'data_manifest.json').write_text(json.dumps({'train':{'manifest':{'examples':9472}}}))
    campaign=execution()
    campaign['canaries']=[{'status':'PASS','gpu_uuid':f'GPU-{i}'} for i in (1,2)]
    campaign['results']=[{'job':j,'status':'COMPLETE','global_update':2368,'checkpoint':'b'*64} for j in jobs()]
    assert release.required_results(tmp_path,campaign)['status']=='FAILED'
    release.finalize(tmp_path,campaign,'missing-results')
    assert campaign['status']=='INVALID_RESULTS'


def test_real_current_archives_required_and_tampering_rejected(tmp_path,monkeypatch):
    isolate_coverage(monkeypatch); (tmp_path/'analysis').mkdir()
    campaign=release.finalize(tmp_path,execution(),'current-run')
    assert campaign['status']=='COMPLETE' and campaign['delivery_status']=='PASS'
    assert json.loads((tmp_path/'exports'/'FINAL_STATUS.json').read_bytes())['status']=='COMPLETE'
    # Old verified archives do not constitute delivery of another attempt.
    assert release.verify_delivery(tmp_path,campaign['exports'],campaign,'another-run')['status']=='FAILED'
    with (tmp_path/'exports'/'ELENDRIL_V7_FULL.zip').open('ab') as handle: handle.write(b'changed')
    assert release.verify_delivery(tmp_path,campaign['exports'],campaign,'current-run')['status']=='FAILED'


@pytest.mark.parametrize('kind',['analysis','full'])
def test_deferred_or_missing_export_prevents_success(tmp_path,monkeypatch,kind):
    isolate_coverage(monkeypatch); (tmp_path/'analysis').mkdir()
    real=release.export_results
    def deferred(*args,**kwargs):
        result=real(*args,**kwargs); result[kind]={'status':'EXPORT_DEFERRED'}; return result
    monkeypatch.setattr(release,'export_results',deferred)
    campaign=release.finalize(tmp_path,execution(),'deferred')
    assert campaign['execution_status']=='EXECUTED' and campaign['status']=='EXPORT_FAILED'
    assert campaign['delivery_status']=='FAILED'


def test_interrupted_study_keeps_partial_status_and_downloads(tmp_path):
    (tmp_path/'analysis').mkdir(); campaign=execution(); campaign['status']='PARTIAL'
    release.finalize(tmp_path,campaign,'interrupted')
    assert campaign['status']=='PARTIAL' and campaign['delivery_status']=='PASS'
    assert (tmp_path/'exports'/'ELENDRIL_V7_ANALYSIS.zip').is_file()


@pytest.mark.parametrize('devices',[
    [], [SimpleNamespace(uuid='GPU-a')],
    [SimpleNamespace(uuid='GPU-a'),SimpleNamespace(uuid='GPU-a')],
    [SimpleNamespace(uuid='GPU-a',name='P100',free_mib=15000,total_mib=16000),
     SimpleNamespace(uuid='GPU-b',name='T4',free_mib=15000,total_mib=16000)]])
def test_gpu_admission_fails_closed_without_querying_local_hardware(monkeypatch,devices):
    from tools import elendril_v6_campaign as v6
    monkeypatch.setattr(v6,'inventory',lambda:devices)
    with pytest.raises(ValueError,match='NO-GO'): v6._gpu_records()


def test_worker_interruption_preserves_committed_state_and_error_receipt(tmp_path,monkeypatch):
    from tools import elendril_v7_worker as worker
    cp=tmp_path/'checkpoints'; cp.mkdir(); (cp/'committed').write_bytes(b'unchanged')
    request=tmp_path/'request.json'; receipt=tmp_path/'analysis'/'error.json'
    request.write_text(json.dumps({'source':'a'*40,'protocol_sha':PROTOCOL_SHA,'timecap_seconds':1,
        'output':str(receipt.parent),'checkpoint_root':str(cp),'job':jobs()[0]}))
    monkeypatch.setattr(worker,'check_source',lambda *_:None)
    def interrupted(*_): raise RuntimeError('injected interruption')
    monkeypatch.setattr(worker,'run',interrupted)
    assert worker.main(['train','--owner-enabled','--request',str(request),'--receipt',str(receipt)])==1
    assert json.loads(receipt.read_bytes())['status']=='ERROR'
    assert (cp/'committed').read_bytes()==b'unchanged'


@pytest.mark.parametrize('status,code',[('COMPLETE',0),('EXECUTED',1),('EXPORT_FAILED',1),('PARTIAL',1)])
def test_cli_success_requires_delivered_completion(tmp_path,monkeypatch,status,code):
    from tools import elendril_v7_campaign as campaign
    monkeypatch.setattr(campaign,'run',lambda **_:{'status':status})
    assert campaign.main(['--output',str(tmp_path),'--source','a'*40,'--owner-enabled'])==code


@pytest.mark.parametrize('corrupt',[False,True])
def test_packaged_checkpoint_remains_bound_to_committed_components(tmp_path,corrupt):
    from hashlib import sha256
    import zipfile
    from tools.elendril_v6_export import MANIFEST_NAME,canonical
    from v5_training.checkpoint import REQUIRED_COMPONENTS
    # Byte-integrity fixture only, not a usable or trained model checkpoint.
    payloads={name:b'original-'+name.encode() for name in REQUIRED_COMPONENTS}
    cp=canonical({'components':[{'name':n,'sha256':sha256(b).hexdigest(),'byte_size':len(b)} for n,b in payloads.items()]})
    prefix='checkpoints/jobs/fixture/fixture/objects/'+sha256(cp).hexdigest()
    if corrupt: payloads['model.bin']=b'corrupted after checkpoint publication'
    files={prefix+'/'+n:b for n,b in payloads.items()}; files[prefix+'/manifest.json']=cp
    path=tmp_path/'fixture.zip'
    with zipfile.ZipFile(path,'w') as archive:
        for name,body in files.items(): archive.writestr(name,body)
        # Packaging hashes are internally consistent even in the corrupt case.
        archive.writestr(MANIFEST_NAME,canonical({'files':[{'name':n,'sha256':sha256(b).hexdigest(),'byte_size':len(b)} for n,b in files.items()]}))
    with zipfile.ZipFile(path) as archive:
        if corrupt:
            with pytest.raises(ValueError,match='committed manifest'):
                release.verify_checkpoint_members(archive,[prefix+'/manifest.json'])
        else: release.verify_checkpoint_members(archive,[prefix+'/manifest.json'])
