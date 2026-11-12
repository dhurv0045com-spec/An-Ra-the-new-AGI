"""Owner-operated V7.1 workers, using the qualified neural/checkpoint foundation."""
import argparse
from dataclasses import asdict,replace
import gc
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from uuid import uuid4

from anra_v5.elendril.codec import digest
from anra_v5.elendril.v71_protocol import PROTOCOL_SHA,validate,trainer_job
from anra_v5.elendril.v71_training import create_trainer
from anra_v5.elendril.v71_data import suite_from_record
from anra_v5.elendril.v71_evaluation import evaluate_suite,verify_suite
from anra_v5.elendril.v6_data import read_dataset
from anra_v5.elendril.v6_evaluation import NeuralPolicy
from anra_v5.elendril.v71_policy import GrammarPolicy
from tools.elendril_v6_export import immutable_json
from tools.elendril_v6_worker import _weights,append_json
from tools.elendril_cloud import check_source
from v5_training.checkpoint import CheckpointStore

REPO=Path(__file__).resolve().parents[1]


def create(data,job,*,source,root,resume=False,transition=None,parent_store=None):
    return create_trainer(data,trainer_job(job),source=source,root=Path(root),resume=resume,
                          transition=transition,parent_store=parent_store)


def bound(trainer):
    trainer.backend.binding=replace(trainer.backend.binding,checkpoint_sha256=trainer.parent or _weights(trainer.model))
    return NeuralPolicy(trainer.backend)


def evaluation(trainer,suite,path,*,conditions=None,grammar=False):
    before=_weights(trainer.model); bound(trainer)
    expected=trainer.backend.binding.checkpoint_sha256
    if path.exists():
        result=json.loads(path.read_bytes())
        if result['checkpoint']!=expected or result['update']!=trainer.state.global_update or result['grammar']!=grammar:
            raise ValueError('cached evaluation identity mismatch')
        verify_suite(suite,result['evaluation'])
        if result['evaluation']['conditions']!=(sorted(suite) if conditions is None else list(conditions)):
            raise ValueError('cached evaluation coverage mismatch')
    else:
        policy=GrammarPolicy(trainer.backend) if grammar else bound(trainer)
        started=time.monotonic(); report=evaluate_suite(suite,policy,conditions=conditions)
        verify_suite(suite,report)
        result={'checkpoint':expected,'update':trainer.state.global_update,'grammar':grammar,
                'weight_sha256':before,'wall_seconds':time.monotonic()-started,'evaluation':report}
        immutable_json(path,result)
    if any(r['raw']['model_identity']!=expected for r in result['evaluation']['records'].values() if r):
        raise ValueError('inner evaluation checkpoint mismatch')
    if result['weight_sha256']!=before or _weights(trainer.model)!=before: raise ValueError('evaluation changed weights')
    return result


def recover(req):
    from tools.elendril_action_cloud import compare_payloads
    job=req['job']; challenge=json.loads(Path(req['challenge']).read_bytes())
    if (challenge['source']!=req['source'] or challenge['protocol_sha256']!=PROTOCOL_SHA
            or challenge['gpu_uuid']!=os.environ['CUDA_VISIBLE_DEVICES']): raise ValueError('challenge mismatch')
    # Import the verified generation, not an entire tree with a mutable LATEST.
    # It preserves exact content addressing and avoids precreated-copy errors.
    reference=CheckpointStore(Path(challenge['expected_root']),job['job_id'])
    state,payloads=reference.restore(challenge['head2'])
    destination=CheckpointStore(Path(req['checkpoint_root']),job['job_id'])
    published=destination.publish(state=state,payloads=payloads,expected_parent_sha256=None)
    if published!=challenge['head2']: raise ValueError('republished generation differs')
    trainer=create(read_dataset(req['data_file']),job,source=req['source'],root=Path(req['checkpoint_root']),resume=True)
    if trainer.state.global_update!=2: raise ValueError('recovery cursor mismatch')
    while trainer.state.global_update<4: trainer.step()
    trainer.save()
    _,left=reference.restore(challenge['head4']); _,right=trainer.store.restore(trainer.parent)
    compare_payloads(left,right,exact=True)
    return {'status':'PASS','checkpoint':trainer.parent,'global_update':4,'comparison':'exact full payload comparison'}


def profile(req,deadline,*,worker_module='tools.elendril_v71_worker'):
    import torch
    job=req['job']; root=Path(req['checkpoint_root']); output=Path(req['output'])
    trainer=create(read_dataset(req['data_file']),job,source=req['source'],root=root/'expected')
    timings=[]; start_weights=_weights(trainer.model)
    for u in range(1,9):
        if time.monotonic()>deadline: raise TimeoutError('profile deadline')
        t=time.monotonic(); trainer.step(); torch.cuda.synchronize(); timings.append(time.monotonic()-t)
        if u in (2,4):
            trainer.save()
            if u==2: head2=trainer.parent
            else: head4=trainer.parent
    if _weights(trainer.model)==start_weights: raise ValueError('profile weights did not update')
    suite=suite_from_record(json.loads(Path(req['suite_file']).read_bytes()))
    trainer.backend.binding=replace(trainer.backend.binding,checkpoint_sha256=_weights(trainer.model))
    t=time.monotonic(); report=evaluate_suite({'base':suite['base']},NeuralPolicy(trainer.backend)); verify_suite({'base':suite['base']},report)
    episode_seconds=(time.monotonic()-t)/len(suite['base'])
    immutable_json(output/'timing-evaluation.json',report)
    challenge={'source':req['source'],'protocol_sha256':PROTOCOL_SHA,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],
        'expected_root':str(root/'expected'),'head2':head2,'head4':head4}
    immutable_json(output/'challenge.json',challenge)
    del trainer; gc.collect(); torch.cuda.empty_cache()
    child={**req,'checkpoint_root':str(root/'recovery'),'output':str(output/'recovery'),
           'challenge':str(output/'challenge.json'),'timecap_seconds':max(1,deadline-time.monotonic())}
    request=output/'recovery-request.json'; receipt=output/'recovery'/'receipt.json'; immutable_json(request,child)
    process=subprocess.run([sys.executable,'-m',worker_module,'recover','--owner-enabled',
        '--request',str(request),'--receipt',str(receipt)],cwd=REPO,timeout=max(1,deadline-time.monotonic()))
    recovered=json.loads(receipt.read_bytes())
    if process.returncode or recovered['status']!='PASS': raise ValueError('profile recovery failed: '+str(recovered))
    # Remove only redundant temporary copies after exact comparison. Original
    # generation 2 and 4, manifests, challenge and all receipts remain retained.
    transient=(root/'recovery').resolve()
    if not transient.is_relative_to(root.resolve()) or transient==root.resolve(): raise ValueError('unsafe transient root')
    shutil.rmtree(transient)
    return {'status':'PASS','profile':job['profile'],'seconds_per_update':sum(timings)/8,
            'evaluation_seconds_per_episode':episode_seconds,'recovery':recovered,
            'checkpoint_root':str(root/'expected'),'head2':head2,'head4':head4,
            'qualification':'production-shaped timing and exact fresh-process recovery; no skill claim'}


def train(req,deadline):
    from anra_v5.elendril.foundation_training import StageTransition
    job=req['job']; data=read_dataset(req['data_file']); root=Path(req['checkpoint_root']); output=Path(req['output'])
    suite=suite_from_record(json.loads(Path(req['suite_file']).read_bytes()))
    resume=(root/job['job_id']/'LATEST').exists(); transition=None; parent_store=None
    if 'parent' in req:
        parent=req['parent']; parent_store=CheckpointStore(Path(parent['root']),parent['lineage'])
        _,payloads=parent_store.restore(parent['checkpoint'])
        transition_file=output/'transition.json'
        if transition_file.exists(): transition=StageTransition(**json.loads(transition_file.read_bytes()))
        else:
            prototype=create(data,job,source=req['source'],root=root)
            transition=StageTransition(parent['checkpoint'],digest(json.loads(payloads['scheduler.json'])['metadata']['recipe']),
                digest(asdict(prototype.recipe)),prototype.data_sha)
            del prototype; gc.collect()
            immutable_json(transition_file,asdict(transition))
    trainer=create(data,job,source=req['source'],root=root,resume=resume,transition=transition,parent_store=parent_store)
    before=trainer.state.global_update; epoch_updates=math.ceil(len(trainer.samples)/trainer.recipe.batch_size)
    marks={epoch_updates*e for e in range(1,job['epochs']+1) if e in (1,2,4)}
    initial='initial.json' if not req.get('parent') else 'after-transfer-before-B.json'
    if not resume: evaluation(trainer,suite,output/initial,conditions=['base','hop-3'])
    def inspect():
        u=trainer.state.global_update
        if u not in marks: return
        trainer.store.record_milestone(threshold_tokens=trainer.state.cumulative_tokens,checkpoint_sha256=trainer.parent)
        final=u==trainer.recipe.total_updates
        evaluation(trainer,suite,output/f'epoch-{u//epoch_updates}.json',conditions=None if final else ['base','hop-3'])
    if resume: inspect()
    log=output/('updates-'+uuid4().hex+'.jsonl')
    interrupted=None
    try:
        while not trainer.state.complete and time.monotonic()<deadline-180:
            t=time.monotonic(); row=trainer.step(); row['wall_seconds']=time.monotonic()-t
            chosen=[trainer.samples[i] for i in trainer.plan[row['update']-1]]
            chunks=[chosen[i:i+trainer.recipe.micro_batch_size] for i in range(0,len(chosen),trainer.recipe.micro_batch_size)]
            row['padded_input_positions']=sum(len(c)*max(len(s.tokens) for s in c) for c in chunks)
            row['padded_attention_pairs']=sum(len(c)*max(len(s.tokens) for s in c)**2 for c in chunks)
            append_json(log,row)
            if trainer.state.global_update%64==0 or trainer.state.global_update in marks:
                trainer.save(); inspect(); trainer.store.prune(keep={trainer.parent})
    except (KeyboardInterrupt,InterruptedError) as exc:
        interrupted=type(exc).__name__
    # A mid-step interruption can leave partially mutated weights/optimizer.
    # Only complete transactions may be published; otherwise recover last disk generation.
    if not trainer.faulted and trainer.state.global_update>trainer.published_update: trainer.save()
    if trainer.state.complete and not interrupted: evaluation(trainer,suite,output/'grammar-final.json',grammar=True)
    return {'status':'COMPLETE' if trainer.state.complete and not interrupted else 'PARTIAL','interrupted':interrupted,
        'recovery_required':bool(trainer.faulted),'job':job,'source':req['source'],
        'checkpoint':trainer.parent,'checkpoint_root':str(root),'start_update':before,
        'global_update':trainer.state.global_update,'total_updates':trainer.recipe.total_updates,
        'supervised_tokens':trainer.state.cumulative_tokens,'metadata':trainer.metadata,
        'parameter_count':sum(p.numel() for p in trainer.model.parameters())}


def frozen(req):
    import torch
    from v5_model.core import initialize
    from anra_v5.elendril.training import configure_device,_load
    from anra_v5.elendril.v6_registry import MODEL_PROFILES
    from anra_v5.elendril.representation import UTF8
    from anra_v5.elendril.backend import NeuralBackend,BackendIdentity
    parent=req['parent']; store=CheckpointStore(Path(parent['root']),parent['lineage'])
    state,payloads=store.restore(parent['checkpoint']); spec=MODEL_PROFILES['S3']; configure_device('cuda:0')
    if state.identities.model_spec_sha256!=spec.sha256() or state.identities.tokenizer_sha256!=UTF8.identity:
        raise ValueError('frozen checkpoint incompatible with S3 UTF8')
    model=initialize(spec,7).to('cuda:0'); weights=_load(payloads['model.bin'])
    if any(not torch.isfinite(t).all() for t in weights.values()): raise ValueError('nonfinite frozen weights')
    model.load_state_dict(weights,strict=True)
    backend=NeuralBackend(model,BackendIdentity(spec.sha256(),UTF8.identity,260,spec.context_length,parent['checkpoint']),
                          pad_id=UTF8.pad_id,bos_id=UTF8.bos_id,eos_id=UTF8.eos_id)
    suite=suite_from_record(json.loads(Path(req['suite_file']).read_bytes())); before=_weights(model)
    if state.identities.source_commit!='413a3eccf4344ef2a39051b870e92feb6271b769': raise ValueError('frozen source is not the audited owner V7 source')
    for name,policy in (('epoch-0',NeuralPolicy(backend)),('grammar-final',GrammarPolicy(backend))):
        path=Path(req['output'])/(name+'.json')
        if path.exists():
            saved=json.loads(path.read_bytes()); report=saved['evaluation']
            if (saved['checkpoint']!=parent['checkpoint'] or saved['weight_sha256']!=before
                or saved['grammar']!=(name=='grammar-final')): raise ValueError('frozen cached identity mismatch')
        else:
            report=evaluate_suite(suite,policy)
            immutable_json(path,{'original_source':state.identities.source_commit,
                'checkpoint':parent['checkpoint'],'weight_sha256':before,'update':state.global_update,
                'grammar':name=='grammar-final','evaluation':report})
        verify_suite(suite,report)
        if report['conditions']!=sorted(suite) or any(r['raw']['model_identity']!=parent['checkpoint'] for r in report['records'].values() if r):
            raise ValueError('frozen report coverage or checkpoint mismatch')
    if _weights(model)!=before: raise ValueError('frozen evaluation altered weights')
    return {'status':'COMPLETE','original_source':state.identities.source_commit,'checkpoint':parent['checkpoint'],
            'qualification':'weights-only reevaluation; no original optimizer continuation or cross-runtime equivalence claim'}


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('profile','recover','train','frozen'))
    p.add_argument('--owner-enabled',action='store_true'); p.add_argument('--request',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True); a=p.parse_args(argv)
    if not a.owner_enabled: p.error('owner execution required')
    req=json.loads(a.request.read_bytes()); cap=req['timecap_seconds']
    if req['protocol_sha']!=PROTOCOL_SHA or type(cap) not in (int,float) or not math.isfinite(cap) or not 1<=cap<=21600:
        raise ValueError('protocol or wall cap mismatch')
    check_source(REPO,req['source'])
    out=Path(req['output']).resolve(); cp=Path(req.get('checkpoint_root',out/'unused')).resolve()
    if out.is_relative_to(REPO) or cp.is_relative_to(REPO) or not a.receipt.resolve().is_relative_to(out) or a.receipt.exists():
        raise ValueError('outside-source output and new contained receipt required')
    out.mkdir(parents=True,exist_ok=True)
    if a.mode in ('train','profile','recover'): validate(req['job'],req['phase'])
    started=time.monotonic()
    def stop(_signum,_frame): raise InterruptedError('worker termination requested; preserve verified checkpoint')
    signal.signal(signal.SIGTERM,stop)
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise ValueError('one isolated CUDA GPU required')
        fn={'profile':profile,'recover':lambda r,d:recover(r),'train':train,'frozen':lambda r,d:frozen(r)}[a.mode]
        result=fn(req,started+cap)
        result.update(gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],wall_seconds=time.monotonic()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as exc:
        result={'status':'ERROR','error_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
                'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES')}
    result.update(source=req['source'],protocol_sha256=PROTOCOL_SHA,mode=a.mode)
    immutable_json(a.receipt,result)
    return 0 if result['status'] in ('PASS','COMPLETE','PARTIAL') else 1


if __name__=='__main__': raise SystemExit(main())
