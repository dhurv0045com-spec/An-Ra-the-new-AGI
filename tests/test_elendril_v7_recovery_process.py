"""Complete recovery worker on tiny CPU weights; only CUDA admission is simulated."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from anra_v5.elendril.foundation_training import Recipe
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.training import MODEL,configure_device
from anra_v5.elendril.v7_training import V7Trainer
from anra_v5.elendril.v7_protocol import PROTOCOL_SHA
from anra_v5.elendril.v6_data import make_dataset,write_dataset
from tools import elendril_v7_worker as worker


def tiny(data,job,*,source,root,resume=False,updates=4):
    configure_device('cpu')
    spec=replace(MODEL,vocabulary_size=260,width=16,layers=1,query_heads=2,
                 kv_heads=1,head_dimension=8,ffn_width=32,context_length=512)
    return V7Trainer(data.samples,spec=spec,recipe=Recipe(seed=7,total_updates=updates,
        batch_size=4,micro_batch_size=2,lr=.0003,sampling='uniform',schedule='cosine'),
        method=job['method'],source_commit=source,root=Path(root),lineage=job['job_id'],
        data_provenance=data.manifest,tokenizer_manifest=UTF8.manifest.record(),device='cpu',resume=resume)


@pytest.mark.parametrize('method,corrupt_reference',[('uniform',False),('blocked',False),('interleaved',False),('uniform',True)])
def test_entire_recovery_worker_in_fresh_process(tmp_path,method,corrupt_reference):
    data=make_dataset(worlds=4); path=tmp_path/'data.json'; write_dataset(data,path)
    job={'job_id':'mechanical-'+method,'method':method,'seed':7}
    expected=tmp_path/'expected'; trainer=tiny(data,job,source='a'*40,root=expected)
    for u in range(1,5):
        trainer.step()
        if u in (2,4):
            trainer.save()
            if u==2:
                worker.copy_checkpoint(expected,tmp_path/'head2'); head2=trainer.parent
    challenge={'source':'a'*40,'protocol_sha256':PROTOCOL_SHA,'gpu_uuid':'GPU-test-only',
        'head2_path':str(tmp_path/'head2'),'expected_path':str(expected),'head2':head2,'head4':trainer.parent}
    (tmp_path/'challenge.json').write_text(json.dumps(challenge))
    if corrupt_reference:
        component=expected/job['job_id']/'objects'/trainer.parent/'model.bin'
        component.write_bytes(component.read_bytes()+b'corrupt-reference-fixture')
    req={'source':'a'*40,'protocol_sha':PROTOCOL_SHA,'job':job,'data_file':str(path),
        'challenge':str(tmp_path/'challenge.json'),'output':str(tmp_path/'result'),
        'checkpoint_root':str(tmp_path/'restored'),'timecap_seconds':60}
    request=tmp_path/'request.json'; request.write_text(json.dumps(req)); receipt=tmp_path/'result'/'receipt.json'
    run=subprocess.run([sys.executable,str(Path(__file__)),'--child',str(request),str(receipt)],
        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'GPU-test-only'},
        capture_output=True,text=True,timeout=60)
    assert receipt.exists(),run.stdout+run.stderr
    result=json.loads(receipt.read_bytes())
    if corrupt_reference:
        assert run.returncode==1 and result['status']=='ERROR',result
        assert 'component' in result['message'].lower(),result
        return
    assert run.returncode==0,result
    assert result['status']=='PASS' and result['update']==4 and result['checkpoint']==trainer.parent


if __name__=='__main__':
    import torch
    import anra_v5.elendril.training as training
    import tools.elendril_action_cloud as cloud
    runtime=training.runtime_identity
    training.runtime_identity=lambda device: runtime(device) if device.type=='cpu' else {'test':'simulated admission only'}
    cloud.session_identity=lambda:'cpu-mechanical-fixture'
    torch.cuda.is_available=lambda:True
    torch.cuda.device_count=lambda:1
    torch.cuda.synchronize=lambda:None
    torch.cuda.max_memory_allocated=lambda:0
    torch.cuda.max_memory_reserved=lambda:0
    def cpu_create(*args,**kwargs):
        # Admission was simulated; all subsequent tensor/RNG work is genuinely
        # CPU, including fork_rng's automatic device enumeration.
        torch.cuda.is_available=lambda:False
        torch.cuda.device_count=lambda:0
        return tiny(*args,**kwargs)
    worker.create=cpu_create
    worker.check_source=lambda *_:None
    raise SystemExit(worker.main(['recover','--owner-enabled','--request',sys.argv[2],'--receipt',sys.argv[3]]))
