"""Owner-run, phase-admitted V7.1 research; imports never launch training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
from queue import Queue, Empty
import shutil
import subprocess
import sys
import time
import traceback
from uuid import uuid4

from anra_v5.elendril.codec import digest
from anra_v5.elendril.v71_protocol import PROTOCOL_SHA, PHASES, protocol, jobs
from anra_v5.elendril.v71_data import (build_suite, suite_manifest, suite_to_record,
                                      suite_from_record, assert_training_disjoint, make_route_training)
from anra_v5.elendril.v71_evaluation import verify_suite, acquisition_gate, retention
from tools.elendril_cloud import check_source, check_torch, device_env, fingerprint
from tools.elendril_v6_campaign import _gpu_records
from tools.elendril_v6_export import immutable_json, append_event
from tools.elendril_v7_campaign import write_once

REPO=Path(__file__).resolve().parents[1]


def prepare_data(output):
    """Automatically generate and audit input under writable working, never /input."""
    from anra_v5.elendril.v6_data import make_dataset, read_dataset, write_dataset, _dataset, assert_disjoint
    root=Path(output)/'data'; root.mkdir(parents=True,exist_ok=True)
    def publish(name,build):
        path=root/(name+'.json')
        if path.exists(): return read_dataset(path)
        pending=root/('.pending-'+uuid4().hex)
        try:
            write_dataset(build(),pending)
            with pending.open('r+b') as f: f.flush(); os.fsync(f.fileno())
            os.link(pending,path)
            return read_dataset(path)
        finally: pending.unlink(missing_ok=True)
    data={}
    for name,seed in (('train',606),('validation',707),('confirmation',909)):
        d=publish(name,lambda name=name,seed=seed:make_dataset(name,seed=seed,
            target_bytes=2200000,worlds=None if name=='train' else 112))
        if d.split!=name or d.manifest['seed']!=seed or (name=='train' and d.manifest['target_bytes']!=2200000) or (name!='train' and len(d.cases)!=112):
            raise ValueError('cached data recipe mismatch')
        data[name]=d
    assert_disjoint(*data.values())
    if not 2000000<=data['train'].manifest['text_bytes']<=3000000: raise ValueError('training text budget mismatch')
    for name,parent,quota in (('canary','train',2),('monitor','validation',4)):
        counts={}; selected=[]
        cases=data[parent].cases if name=='canary' else sorted(data[parent].cases,key=lambda c:(c.world_id,c.case_id))
        for c in cases:
            if counts.get(c.kind,0)<quota: selected.append(c); counts[c.kind]=counts.get(c.kind,0)+1
        if len(counts)!=7 or min(counts.values())!=quota: raise ValueError('incomplete control cohort')
        expected=_dataset(data[parent].split,tuple(selected),seed=data[parent].manifest['seed'],parent=data[parent].identity)
        data[name]=publish(name,lambda:expected)
        if data[name].identity!=expected.identity: raise ValueError('control cohort mismatch')
    suite=build_suite(); suite['confirmation']=data['confirmation'].cases
    write_once(root/'suite.json',suite_to_record(suite))
    reconstructed=suite_from_record(json.loads((root/'suite.json').read_bytes()))
    if suite_manifest(reconstructed)!=suite_manifest(suite): raise ValueError('suite roundtrip mismatch')
    audit=assert_training_disjoint(data['train'],suite)
    data['new-b']=publish('new-b',make_route_training)
    assert_training_disjoint(data['new-b'],suite)
    selected=[]; counts={}
    for c in sorted(data['train'].cases,key=lambda c:(c.world_id,c.case_id)):
        if c.kind in ('missing','memory') and counts.get(c.kind,0)<64:
            selected.append(c); counts[c.kind]=counts.get(c.kind,0)+1
    if counts!={'memory':64,'missing':64}: raise ValueError('replay cohort incomplete')
    expected=_dataset('train',tuple((*data['new-b'].cases,*selected)),seed=27171,parent=digest([data['train'].identity,data['new-b'].identity]))
    data['replay']=publish('replay',lambda:expected)
    if data['replay'].identity!=expected.identity: raise ValueError('replay recipe mismatch')
    assert_training_disjoint(data['replay'],suite)
    manifest={k:{'file':k+'.json','identity':v.identity,'file_sha256':fingerprint(root/(k+'.json')),'manifest':v.manifest} for k,v in data.items()}
    write_once(Path(output)/'analysis'/'data_manifest.json',manifest)
    write_once(Path(output)/'analysis'/'suite_manifest.json',{'suite':suite_manifest(suite),'training_disjoint':audit,
        'shared_authorship':'Independent generator implementation, shared AI authorship; not a sealed external test.'})
    return data,suite


def invoke(mode,request,gpu,*,module='tools.elendril_v71_worker'):
    directory=Path(request['output']); directory.mkdir(parents=True,exist_ok=True)
    attempt=uuid4().hex; reqfile=directory/(attempt+'-request.json'); receipt=directory/(attempt+'-receipt.json')
    immutable_json(reqfile,request); started=time.monotonic()
    with (directory/(attempt+'.txt')).open('xb') as log:
        try:
            result=subprocess.run([sys.executable,'-m',module,mode,'--owner-enabled','--request',str(reqfile),'--receipt',str(receipt)],
                cwd=REPO,env=device_env(gpu),stdout=log,stderr=subprocess.STDOUT,timeout=request['timecap_seconds']+30)
            if not receipt.exists(): raise RuntimeError('worker exited without terminal receipt; see '+str(directory/(attempt+'.txt')))
            report=json.loads(receipt.read_bytes())
            if result.returncode and report.get('status') not in ('ERROR','FAIL'): raise ValueError('exit code/receipt conflict')
        except Exception as exc:
            report={'status':'ERROR','error_type':type(exc).__name__,'message':str(exc),'source':request['source']}
            if not receipt.exists(): immutable_json(receipt,report)
    return {**report,'receipt_path':str(receipt),'attempt_wall_seconds':time.monotonic()-started}


def require_pass(rows,label):
    failed=[r for r in rows if r.get('status')!='PASS']
    for row in failed:
        print(json.dumps({'gate':label,**{k:row.get(k) for k in ('status','error_type','message','reason','traceback','receipt_path')}},indent=2),flush=True)
    if failed:
        first=failed[0]
        raise ValueError(label+' failed: '+str(first.get('message',first.get('reason',first.get('status')))))


def rescore(output,campaign=None):
    path=Path(output)/'data'/'suite.json'
    if not path.exists(): return {'status':'NOT_RUN','reason':'suite unavailable'}
    suite=suite_from_record(json.loads(path.read_bytes())); checked=0
    analysis=Path(output)/'analysis'
    paths=list((analysis/'controls').glob('*.json'))
    for pattern in ('*/epoch-*.json','*/grammar-final.json','*/initial.json','*/after-transfer-before-B.json'):
        paths.extend((analysis/'jobs').glob(pattern))
    paths.extend((analysis/'profiles').glob('*/timing-evaluation.json'))
    for path in sorted(set(paths)):
        if path.is_symlink(): raise ValueError('linked scientific report')
        if path.stat().st_size>512<<20: raise ValueError('oversized report')
        report=json.loads(path.read_bytes())
        if path.name=='timing-evaluation.json': verify_suite({'base':suite['base']},report); checked+=1
        else:
            if not isinstance(report.get('evaluation'),dict): raise ValueError('malformed mandatory evaluation '+str(path))
            verify_suite(suite,report['evaluation'])
            if any(r['raw']['model_identity']!=report['checkpoint'] for r in report['evaluation']['records'].values() if r):
                raise ValueError('inner checkpoint binding mismatch')
            checked+=1
    return {'status':'PASS' if checked else 'NOT_RUN','reports_replayed':checked,
        'limitation':'Recorded public actions replayed without neural inference; no probability regeneration or causal promotion.'}


def forecast(phase_jobs,profiles,data,suite):
    """Greedy two-lane forecast, whole protocol, 2x timing margin plus snapshots."""
    work=[]
    for job in phase_jobs:
        measurements=[r for r in profiles if r['profile']==job['profile'] and r['job']['mechanism']==job['mechanism']]
        if not measurements: raise ValueError('missing production profile')
        step=max(r['seconds_per_update'] for r in measurements); episode=max(r['evaluation_seconds_per_episode'] for r in measurements)
        dataset=data['replay' if job.get('stage')=='replay' else 'new-b' if job.get('stage') else 'train']
        samples=len(dataset.samples)
        if job['mechanism']=='intermediate':
            from anra_v5.elendril.v6_data import transform_dataset
            samples=len(transform_dataset(dataset,scratch=True).samples)
        updates=math.ceil(samples/16)*job['epochs']
        intermediate=sum(len(suite[n]) for n in ('base','hop-3'))
        episodes=job['epochs']//2*intermediate+intermediate+2*sum(map(len,suite.values()))
        work.append(2*(updates*step+episodes*episode))
    lanes=[0.,0.]
    for seconds in work: lanes[lanes.index(min(lanes))]+=seconds
    return {'seconds':max(lanes)+600,'job_seconds':work,'policy':'whole phase, 2x measured headroom, ten-minute export reserve; no score-based pruning'}


def old_skill(metrics):
    rows=[metrics['by_kind'][k] for k in ('missing','memory')]
    return {k:sum(r[k] for r in rows) for k in ('nonnull_correct','nonnull_total')}


def coverage(output,phase_jobs,suite):
    required=[]
    for job in phase_jobs:
        folder=Path(output)/'analysis'/'jobs'/job['job_id']; final=None
        for epoch in ((0,) if job['epochs']==0 else (1,2,4)):
            if epoch>job['epochs']: continue
            p=folder/f'epoch-{epoch}.json'
            if not p.is_file(): raise ValueError('missing mandatory report '+str(p))
            row=json.loads(p.read_bytes()); verify_suite(suite,row['evaluation'])
            conditions=sorted(suite) if epoch==job['epochs'] else ['base','hop-3']
            if row['evaluation']['conditions']!=conditions or row['grammar'] is not False: raise ValueError('checkpoint evaluation coverage differs')
            required.append(str(Path('checkpoints')/'jobs'/job['job_id']/job['job_id']/'objects'/row['checkpoint']/'manifest.json'))
            if epoch==job['epochs']: final=row
        row=json.loads((folder/'grammar-final.json').read_bytes()); verify_suite(suite,row['evaluation'])
        if row['checkpoint']!=final['checkpoint'] or row['grammar'] is not True or row['evaluation']['conditions']!=sorted(suite):
            raise ValueError('grammar comparison coverage differs')
    return required


def frozen_phase(output,parent_output,base,devices,deadline,campaign):
    """Reevaluate exactly the nine audited V7 final checkpoints, without updates."""
    from anra_v5.elendril.v7_protocol import jobs as old_jobs
    from v5_training.checkpoint import CheckpointStore
    if parent_output is None: raise ValueError('frozen requires --parent-output extracted from owner V7 FULL ZIP')
    parent_output=Path(parent_output).resolve(); registered=[]; requests=[]; required=[]
    for old in old_jobs():
        job={**old,'profile':'S3','mechanism':'ce','epochs':0}; registered.append(job)
        proof=json.loads((parent_output/'analysis'/'jobs'/old['job_id']/'confirmation.json').read_bytes())
        reference=CheckpointStore(parent_output/'checkpoints'/'jobs'/old['job_id'],old['job_id'])
        head=proof['checkpoint']; state,_=reference.restore(head)
        if state.identities.source_commit!='413a3eccf4344ef2a39051b870e92feb6271b769' or not state.complete:
            raise ValueError('frozen checkpoint is not a complete audited V7 generation')
        target=output/'checkpoints'/'jobs'/old['job_id']/old['job_id']/'objects'/head
        if not target.exists(): shutil.copytree(reference.lineage_root/'objects'/head,target)
        # A read-only historical generation is copied byte-for-byte, not migrated
        # into a fictitious optimizer lineage. Explicit SHA restore needs no LATEST.
        destination=CheckpointStore(output/'checkpoints'/'jobs'/old['job_id'],old['job_id']); destination.restore(head)
        required.append(str(target.relative_to(output)/'manifest.json'))
        requests.append({**base,'job':job,'output':str(output/'analysis'/'jobs'/old['job_id']),
            'parent':{'root':str(output/'checkpoints'/'jobs'/old['job_id']),'lineage':old['job_id'],'checkpoint':head}})
    pending=Queue()
    for req in requests: pending.put(req)
    def lane(index):
        rows=[]
        while time.monotonic()<deadline-600:
            try: req=pending.get_nowait()
            except Empty: break
            req['timecap_seconds']=max(1,min(21600,deadline-time.monotonic()-600))
            rows.append({**invoke('frozen',req,devices[index]),'job':req['job']})
        return rows
    with ThreadPoolExecutor(max_workers=2) as pool: campaign['results']=[r for rows in pool.map(lane,range(2)) for r in rows]
    campaign['completed_jobs']=sum(r['status']=='COMPLETE' for r in campaign['results'])
    campaign['status']='EXECUTED' if campaign['completed_jobs']==9 else 'PARTIAL'
    return registered,required


def run(*,output,source,phase='core',owner_enabled=False,minutes=330,parent_output=None):
    if owner_enabled is not True: raise ValueError('explicit owner launch required')
    if phase not in PHASES or type(minutes) not in (int,float) or not math.isfinite(minutes) or not 1<=minutes<=330: raise ValueError('invalid phase/budget')
    requested=Path(output)
    if requested.is_symlink(): raise ValueError('output symlink forbidden')
    output=requested.resolve()
    if output.is_relative_to(REPO) or output==Path(output.anchor): raise ValueError('dedicated output outside source required')
    output.mkdir(parents=True,exist_ok=True); started=time.monotonic(); deadline=started+minutes*60; run_id=uuid4().hex
    campaign={'schema':'elendril-v71-campaign-v1','run_id':run_id,'phase':phase,'source':source,'protocol_sha256':PROTOCOL_SHA,
        'status':'PREFLIGHT','results':[],'promotion':'NONE','cause_status':'CAUSE_UNRESOLVED','fresh_vm_recovery':'NOT_QUALIFIED until owner restoration'}
    required=[]; registered=jobs(phase)
    try:
        check_source(REPO,source)
        write_once(output/'analysis'/'identity.json',{'source':source,'protocol_sha256':PROTOCOL_SHA,'phase':phase})
        write_once(output/'analysis'/'protocol.json',protocol())
        # Conservative raw+snapshot budget. The exact available volume is recorded.
        free=shutil.disk_usage(output).free; minimum=(19 if phase=='scale' else 12)*(1<<30)
        campaign['free_disk_bytes']=free
        if free<minimum: raise ValueError(f'NO-GO: {minimum/(1<<30):g} GiB free required; use a fresh session, preserve previous outputs')
        devices=_gpu_records(); runtimes=check_torch(REPO,devices)
        immutable_json(output/'analysis'/('environment-'+run_id+'.json'),runtimes)
        print('PREFLIGHT PASS. Generating and auditing data automatically under working; no input dataset required.',flush=True)
        data,suite=prepare_data(output)
        base={'source':source,'protocol_sha':PROTOCOL_SHA,'phase':phase,'suite_file':str(output/'data'/'suite.json')}
        if phase=='frozen':
            registered,required=frozen_phase(output,parent_output,base,devices,deadline,campaign)
            if campaign['status']=='EXECUTED': required=coverage(output,registered,suite)
            return campaign
        if phase=='continual' and parent_output is None: raise ValueError('continual requires verified V7.1 core output via --parent-output')
        if phase=='continual':
            parent_output=Path(parent_output).resolve()
            identity=json.loads((parent_output/'analysis'/'identity.json').read_bytes())
            admitted_parents=(
                {'source':source,'protocol_sha256':PROTOCOL_SHA,'phase':'core'},
                {'source':'6104ffd7134cb4cf4b3f340bd060d24a1e2dd200','protocol_sha256':PROTOCOL_SHA,'phase':'core'},
            )
            if identity not in admitted_parents: raise ValueError('parent core identity mismatch')
            campaign['parent_core_source']=identity['source']
            campaign['parent_lineage_note']='Explicit V7.15 cross-source fresh-stage migration, never exact optimizer continuation.'
            coverage(parent_output,jobs('core'),suite)
        from anra_v5.elendril.v6_evaluation import ControlPolicy
        from anra_v5.elendril.v71_evaluation import evaluate_suite
        for method in ('fixed','always-query','no-query','random','recency','query-blind'):
            path=output/'analysis'/'controls'/(method+'.json')
            if not path.exists():
                record=evaluate_suite(suite,ControlPolicy(method)); verify_suite(suite,record)
                identity=next(r['raw']['model_identity'] for r in record['records'].values() if r)
                immutable_json(path,{'evaluation':record,'checkpoint':identity,
                    'attribution':'Engineered comparator; no learned capability evidence.'})
        # Reuse the existing positively qualified memorization/recovery control,
        # explicitly under its original V7 schema, then profile V7.1 full shapes.
        from anra_v5.elendril.v7_protocol import PROTOCOL_SHA as V7_SHA
        def positive(index):
            return invoke('canary',{'source':source,'protocol_sha':V7_SHA,'output':str(output/'analysis'/'positive'/f'{run_id}-lane{index}'),
                'checkpoint_root':str(output/'checkpoints'/'positive'/f'{run_id}-lane{index}'),
                'job':{'job_id':'canary'+str(index),'method':'uniform','seed':7},
                'data_file':str(output/'data'/'canary.json'),'eval_file':str(output/'data'/'monitor.json'),
                'production_file':str(output/'data'/'train.json'),'timecap_seconds':max(1,min(1200,deadline-time.monotonic()-180))},
                devices[index],module='tools.elendril_v7_worker')
        with ThreadPoolExecutor(max_workers=2) as pool: canaries=list(pool.map(positive,range(2)))
        campaign['positive_controls']=canaries
        require_pass(canaries,'positive learning/recovery canary')
        profiles=[]
        conditions=sorted({(j['profile'],j['mechanism']) for j in registered})
        for profile,mechanism in conditions:
            job=next(j for j in registered if j['profile']==profile and j['mechanism']==mechanism)
            def qualify(index):
                r=invoke('profile',{**base,'job':job,'data_file':str(output/'data'/('replay.json' if phase=='continual' else 'train.json')),
                    'output':str(output/'analysis'/'profiles'/f'{run_id}-{profile}-{mechanism}-lane{index}'),
                    'checkpoint_root':str(output/'checkpoints'/'profiles'/f'{run_id}-{profile}-{mechanism}-lane{index}'),
                    'timecap_seconds':max(1,min(1800,deadline-time.monotonic()-180))},devices[index])
                return {**r,'job':job}
            with ThreadPoolExecutor(max_workers=2) as pool: profiles.extend(pool.map(qualify,range(2)))
            campaign['profiles']=profiles
            require_pass(profiles,'production-shaped profile/recovery')
        campaign['profiles']=profiles
        estimate=forecast(registered,profiles,data,suite); campaign['capacity_estimate']=estimate
        if estimate['seconds']>deadline-time.monotonic(): raise ValueError('whole phase exceeds measured remaining capacity; DEFERRED, no silent matrix reduction')
        pending=Queue()
        for job in registered: pending.put(job)
        campaign['status']='RUNNING'
        from tools.elendril_v71_export import export_results
        def lane(index):
            rows=[]
            while time.monotonic()<deadline-600:
                try: job=pending.get_nowait()
                except Empty: break
                request={**base,'job':job,'output':str(output/'analysis'/'jobs'/job['job_id']),
                    'checkpoint_root':str(output/'checkpoints'/'jobs'/job['job_id']),
                    'data_file':str(output/'data'/('replay.json' if job.get('stage')=='replay' else 'new-b.json' if phase=='continual' else 'train.json')),
                    'timecap_seconds':max(1,min(21600,deadline-time.monotonic()-600))}
                if phase=='continual':
                    parentjob=f'S3-blocked-ce-s{job["seed"]}'
                    parent=json.loads((parent_output/'analysis'/'jobs'/parentjob/'epoch-4.json').read_bytes())
                    prior=old_skill(parent['evaluation']['metrics']['base']); gate=acquisition_gate(prior)
                    if gate['status']!='ACQUIRED' and job['stage']!='fresh-b':
                        row={'status':'NOT_QUALIFIED','job':job,'acquisition_gate':gate,'reason':'A not acquired; retention comparison prohibited'}
                        immutable_json(Path(request['output'])/'acquisition-not-qualified.json',row); rows.append(row); continue
                    if job['stage']!='fresh-b': request['parent']={'root':str(parent_output/'checkpoints'/'jobs'/parentjob),'lineage':parentjob,'checkpoint':parent['checkpoint']}
                row=invoke('train',request,devices[index]); row['job']=job
                if phase=='continual' and row['status']=='COMPLETE':
                    final=json.loads((Path(request['output'])/'epoch-2.json').read_bytes())['evaluation']['metrics']
                    initial_file='initial.json' if job['stage']=='fresh-b' else 'after-transfer-before-B.json'
                    initial=json.loads((Path(request['output'])/initial_file).read_bytes())['evaluation']['metrics']
                    row['B_before']=acquisition_gate(initial['hop-3']['all'])
                    row['B_after']=acquisition_gate(final['hop-3']['all'])
                    row['novel_B_acquisition']='NOT_QUALIFIED: B already acquired before training' if row['B_before']['status']=='ACQUIRED' else 'OBSERVABLE before/after acquisition contrast only'
                    row['retention']=({'status':'NOT_QUALIFIED','reason':'fresh-B model did not acquire parent A; baseline only'} if job['stage']=='fresh-b'
                        else retention(old_skill(initial['base']),{'new_skill':final['hop-3']['all'],'old_skill':old_skill(final['base'])}))
                rows.append(row); append_event(output/'analysis',{'event':'JOB_'+row['status'],'job':job})
                print(job['job_id']+': '+row['status'],flush=True)
                if row['status']=='ERROR': print(json.dumps(row,indent=2),flush=True)
                export_results(output,source=source,protocol_sha=PROTOCOL_SHA,campaign=campaign,full=False,rescore=rescore)
            return rows
        with ThreadPoolExecutor(max_workers=2) as pool: campaign['results']=[r for rows in pool.map(lane,range(2)) for r in rows]
        campaign['completed_jobs']=sum(r['status']=='COMPLETE' for r in campaign['results'])
        campaign['status']='EXECUTED' if campaign['completed_jobs']==len(registered) else 'PARTIAL'
        if campaign['status']=='EXECUTED': required=coverage(output,registered,suite)
        campaign['gpu_worker_reservation_seconds']=sum(r.get('attempt_wall_seconds',0) for r in canaries+profiles+campaign['results'])
    except (Exception,KeyboardInterrupt) as exc:
        campaign.update(status='PARTIAL' if isinstance(exc,KeyboardInterrupt) else 'NO_GO',error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        print('Campaign stopped: '+str(exc),flush=True)
    finally:
        campaign['wall_seconds']=time.monotonic()-started
        try:
            from tools.elendril_v71_report import write_report
            campaign['summary']=write_report(output,campaign=campaign)
            campaign['independent_replay']=rescore(output,campaign)
        except Exception as exc:
            campaign['independent_replay']={'status':'FAILED','message':str(exc)}; campaign['status']='INVALID_RESULTS'
        from tools.elendril_v71_export import finalize
        delivery=finalize(output,source,PROTOCOL_SHA,campaign,[j['job_id'] for j in registered],required,
            campaign['status']=='EXECUTED' and campaign['independent_replay']['status']=='PASS' and campaign.get('summary',{}).get('status')=='AVAILABLE')
        campaign.update(status=delivery['status'],exports=delivery['exports'],delivery=delivery)
        print(json.dumps({'status':campaign['status'],'exports':campaign.get('exports'),'message':campaign.get('message')},indent=2),flush=True)
    return campaign


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); p.add_argument('--source',required=True)
    p.add_argument('--phase',choices=PHASES,default='core'); p.add_argument('--owner-enabled',action='store_true'); p.add_argument('--minutes',type=float,default=330)
    p.add_argument('--parent-output',type=Path)
    a=p.parse_args(argv); result=run(output=a.output,source=a.source,phase=a.phase,owner_enabled=a.owner_enabled,minutes=a.minutes,parent_output=a.parent_output)
    return 0 if result['status']=='COMPLETE' else 1


if __name__=='__main__': raise SystemExit(main())
