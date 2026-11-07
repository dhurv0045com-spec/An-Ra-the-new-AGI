from copy import deepcopy
from dataclasses import asdict
import math

import pytest

from anra_v5.elendril.v6_training import Job
from tools.elendril_v6_campaign import estimate_job, select_candidates, _receipt_gate


def test_measured_budget_rejects_nonfinite_and_counts_complete_epochs():
    j=Job('screen','screen','A1',7)
    measurements={'S3': {'seconds_per_update': .5,'evaluation_seconds_per_episode': .1}}
    expected=1.75*.5*math.ceil(9472/16)*2+2*.1*112+60
    assert estimate_job(j,measurements,9472)==pytest.approx(expected)
    for bad in (float('nan'),float('inf'),0,-1,True):
        m=deepcopy(measurements);m['S3']['seconds_per_update']=bad
        with pytest.raises(ValueError):estimate_job(j,m,9472)


def _result(seed,arm,correct=112,split='validation'):
    metric={'episodes':112,'correct':correct,'invalid':0,'requests':32,'utility':correct-3.2}
    return {'status':'COMPLETE','job':asdict(Job(f'{arm}-{seed}','screen',arm,seed)),
            'evaluation':{'split':split,'metrics':{'all':metric,'by_kind':{'known':metric}}},
            'controls':{'fixed':{'metrics':{'all':metric}}}}


def test_both_seeds_and_validation_only_selection():
    assert select_candidates([_result(7,'A1')])['winner'] is None
    assert select_candidates([_result(7,'A1'),_result(19,'A1')])['winner']=='A1'
    with pytest.raises(ValueError,match='confirmation'):
        select_candidates([_result(7,'A1',split='confirmation'),_result(19,'A1')])
    assert select_candidates([_result(7,'A1',20),_result(19,'A1',20)])['winner'] is None


def test_recovery_requires_real_new_vm_and_both_lane_proofs(monkeypatch):
    from tools import elendril_v6_campaign as c
    monkeypatch.setattr(c,'session_identity',lambda:'new')
    receipt={'status':'PASS','source':'a'*40,'protocol_sha256':'b'*64,'session':'new','results':[
        {'status':'PASS','source':'a'*40,'session':'new','fresh_session':False},
        {'status':'PASS','source':'a'*40,'session':'new','fresh_session':True,'original_session':'old','challenge_sha256':'c'*64}]}
    with pytest.raises(ValueError,match='new-VM'):
        _receipt_gate(receipt,'a'*40,'b'*64,'recovery')


@pytest.mark.parametrize('worker_failure',[False,True])
def test_coordinated_scheduler_reaches_final_export_even_after_worker_failure(tmp_path,monkeypatch,worker_failure):
    """Mock GPU boundary only; actual stages, selection and finally path execute."""
    import json,time
    from types import SimpleNamespace
    from tools import elendril_v6_campaign as c
    from tools import verify_elendril_v6 as verifier
    from anra_v5.elendril import v6_data
    from anra_v5.elendril.v6_registry import REGISTRY_SHA
    devices=[SimpleNamespace(uuid='GPU-aa'),SimpleNamespace(uuid='GPU-bb')]
    monkeypatch.setattr(c,'session_identity',lambda:'mock-linux-vm')
    monkeypatch.setattr(c,'_validate_environment',lambda *a:devices)
    monkeypatch.setattr(c,'_receipt_gate',lambda *a:None)
    monkeypatch.setattr(c,'_measurements',lambda *a:{p:{'seconds_per_update':.01,'evaluation_seconds_per_episode':.01,'peak_cuda_allocated_bytes':1} for p in ('S3','S8','S20')})
    monkeypatch.setattr(v6_data,'read_dataset',lambda *a:SimpleNamespace(manifest={'examples':32}))
    data={k:{'file':str(tmp_path/'data'/f'{k}.json')} for k in ('train','validation','confirmation')}
    monkeypatch.setattr(c,'prepare_data',lambda *a,**k:data)
    monkeypatch.setattr(c,'checkpoint_export',lambda *a,**k:{'status':'EXPORT_VERIFIED'})
    monkeypatch.setattr(c,'_render_outputs',lambda *a:None)
    monkeypatch.setattr(verifier,'verify_result',lambda *a:None)
    exports=[]
    monkeypatch.setattr(c,'independent_rescore_and_export',lambda **k:exports.append(k['campaign']['status']) or {'status':'EXPORT_VERIFIED','zip_path':'unit-test-only.zip'})
    def fake_invoke(mode,request,gpu,timeout):
        if mode=='diagnose':return {'status':'PARTIAL','source':'a'*40,'job':request['job']}
        if worker_failure and request['job']['arm']=='A2':raise RuntimeError('mock CUDA boundary failure')
        r=_result(request['job']['seed'],request['job']['arm'])
        r.update(job=request['job'],source='a'*40,protocol_sha256=REGISTRY_SHA)
        # Replication consumes reserved confirmation; never used to nominate.
        r['evaluation']['split']='confirmation' if request['job']['stage']=='replicate' else 'validation'
        path=tmp_path/(request['job']['job_id']+'.json');path.write_text(json.dumps(r))
        r.update(receipt_file=str(path),receipt_sha256=c.fingerprint(path))
        return r
    monkeypatch.setattr(c,'_invoke',fake_invoke)
    preflight={'output':str(tmp_path),'data':data,'available_seconds':21600,'created_monotonic':time.monotonic()}
    result=c.run_campaign(source='a'*40,protocol_sha=REGISTRY_SHA,preflight=preflight,canary={'results':[{}]},
                          recovery={'physical_uuids':['GPU-aa','GPU-bb']},run_id='unit',output=tmp_path)
    assert exports and result['export']['status']=='EXPORT_VERIFIED'
    assert result['status']==('FAILED_WITH_RECEIPTS' if worker_failure else 'COMPLETE')
    assert list((tmp_path/'campaigns').glob('*.json'))
    if not worker_failure:
        assert result['selection']['winner']=='A1'
        assert any(r['job']['representation']=='legacy-ascii' for r in result['results'])
        assert any(r['evaluation']['split']=='confirmation' for r in result['results'])
        assert any(r.get('profile')=='S20' and r['status']=='SKIPPED' for r in result['skipped'])
