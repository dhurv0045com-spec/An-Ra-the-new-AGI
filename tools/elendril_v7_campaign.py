"""Owner-run V7 paired-order diagnostic campaign on two physical T4s."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
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

from anra_v5.elendril.v7_protocol import PROTOCOL_SHA, protocol, jobs
from tools.elendril_cloud import check_source, check_torch, device_env, fingerprint
from tools.elendril_v6_campaign import _gpu_records
from tools.elendril_v6_export import immutable_json, append_event
from tools.elendril_v7_export import export_results

REPO = Path(__file__).resolve().parents[1]


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_bytes()) != json.loads(json.dumps(value)):
            raise ValueError('immutable campaign artifact differs: '+str(path))
    else: immutable_json(path,value)


def prepare_data(output):
    """Procedural data is generated automatically in working, never input."""
    from anra_v5.elendril.v6_data import make_dataset, read_dataset, write_dataset, _dataset, assert_disjoint
    directory = Path(output)/'data'; directory.mkdir(parents=True,exist_ok=True)
    datasets = {}

    def publish(name, build):
        path = directory/(name+'.json')
        if path.exists(): return read_dataset(path)
        pending = directory/('.pending-'+uuid4().hex)
        try:
            data = build(); write_dataset(data,pending)
            # Windows requires a writable descriptor for FlushFileBuffers.
            with pending.open('r+b') as h: h.flush(); os.fsync(h.fileno())
            os.link(pending,path)
            return read_dataset(path)
        finally: pending.unlink(missing_ok=True)

    p = protocol()
    for name,seed in (('train',606),('validation',707),('confirmation',909)):
        datasets[name] = publish(name,lambda name=name,seed=seed: make_dataset(name,
            seed=seed,target_bytes=p['train_bytes'],worlds=None if name=='train' else 112))
        d = datasets[name]
        if (d.split != name or d.manifest['seed'] != seed
                or (name=='train' and d.manifest['target_bytes'] != p['train_bytes'])
                or (name!='train' and len(d.cases) != 112)):
            raise ValueError('cached dataset recipe mismatch')
    assert_disjoint(*datasets.values())
    for name,parent,quota in (('canary','train',2),('monitor','validation',4),('train-monitor','train',4)):
        counts = {}; selected = []
        cases = datasets[parent].cases
        if name!='canary': cases = sorted(cases,key=lambda c:(c.world_id,c.case_id))
        for case in cases:
            if counts.get(case.kind,0) < quota:
                selected.append(case); counts[case.kind] = counts.get(case.kind,0)+1
        if len(counts)!=7 or min(counts.values())!=quota: raise ValueError('incomplete per-kind cohort')
        expected = _dataset(datasets[parent].split,tuple(selected),seed=datasets[parent].manifest['seed'],parent=datasets[parent].identity)
        datasets[name] = publish(name,lambda expected=expected: expected)
        if datasets[name].identity != expected.identity: raise ValueError('cached cohort mismatch')
    if not 2_000_000 <= datasets['train'].manifest['text_bytes'] <= 3_000_000:
        raise ValueError('train text outside registered range')
    manifest = {name:{'file':str(directory/(name+'.json')),'identity':d.identity,
                     'file_sha256':fingerprint(directory/(name+'.json')),'manifest':d.manifest}
                for name,d in datasets.items()}
    write_once(Path(output)/'analysis'/'data_manifest.json',manifest)
    return manifest


def invoke(mode, request, gpu):
    directory = Path(request['output']); directory.mkdir(parents=True,exist_ok=True)
    attempt = uuid4().hex
    reqfile = directory/(attempt+'-request.json'); receipt = directory/(attempt+'-receipt.json')
    immutable_json(reqfile,request)
    started=time.monotonic()
    with (directory/(attempt+'.txt')).open('xb') as log:
        try:
            result = subprocess.run([sys.executable,'-m','tools.elendril_v7_worker',mode,
                '--owner-enabled','--request',str(reqfile),'--receipt',str(receipt)],
                cwd=REPO,env=device_env(gpu),stdout=log,stderr=subprocess.STDOUT,
                timeout=request['timecap_seconds']+30)
            if not receipt.exists(): raise RuntimeError('worker exited without terminal receipt')
            report = json.loads(receipt.read_bytes())
            if result.returncode and report.get('status') not in ('ERROR','FAIL'):
                raise ValueError('worker exit code conflicts with receipt')
        except Exception as exc:
            report = {'status':'ERROR','error_type':type(exc).__name__,'message':str(exc),
                      'source':request['source'],'protocol_sha256':PROTOCOL_SHA}
            if not receipt.exists(): immutable_json(receipt,report)
    return {**report,'receipt_path':str(receipt),'attempt_wall_seconds':time.monotonic()-started}


def rescore(root, campaign):
    """Reconstruct public cases and replay actions without neural inference."""
    from anra_v5.elendril.v6_data import read_dataset
    from anra_v5.elendril.v7_diagnostics import verify_diagnosis
    from anra_v5.elendril.v7_measurement import verify_record
    if not all((Path(root)/'data'/(name+'.json')).exists() for name in ('canary','monitor','validation','confirmation')):
        return {'status':'NOT_RUN','reason':'Preflight did not publish complete data; no scientific replay.'}
    datasets = {name:read_dataset(Path(root)/'data'/(name+'.json'))
                for name in ('canary','monitor','validation','confirmation','train-monitor')}
    checked = 0
    paths=[]
    for pattern in ('*/diagnostic-u*.json','*/development-u*.json','*/training-u*.json'):
        paths.extend((Path(root)/'analysis'/'jobs').glob(pattern))
    for path in sorted(paths):
        report=json.loads(path.read_bytes())
        data=next((d for d in datasets.values() if d.identity==report['data_identity']),None)
        if data is None: raise ValueError('unknown diagnostic dataset')
        verify_diagnosis(data,report['diagnosis'])
        if any(r['raw']['model_identity'] != report['checkpoint'] for r in report['diagnosis']['records'].values() if r):
            raise ValueError('diagnostic checkpoint identity mismatch')
        checked+=1
    for path in sorted((Path(root)/'analysis'/'jobs').glob('*/confirmation.json')):
        report=json.loads(path.read_bytes())
        if report['data_identity'] != datasets['confirmation'].identity:
            raise ValueError('confirmation dataset identity mismatch')
        verify_diagnosis(datasets['confirmation'],report['diagnosis'])
        if any(r['raw']['model_identity'] != report['checkpoint'] for r in report['diagnosis']['records'].values() if r):
            raise ValueError('confirmation checkpoint identity mismatch')
        checked+=1
    for path in sorted((Path(root)/'analysis'/'canaries').glob('*/positive.json')):
        report=json.loads(path.read_bytes()); verify_record(datasets['canary'],report['raw'])
        verify_record(datasets['monitor'],report['distinct_validation']); checked+=1
    for path in sorted((Path(root)/'analysis'/'controls').glob('*.json')):
        report=json.loads(path.read_bytes())
        data=datasets[report['cohort']]
        if report['data_identity'] != data.identity: raise ValueError('control dataset mismatch')
        for record in report['records'].values(): verify_record(data,record); checked+=1
    return {'status':'PASS' if checked else 'NOT_RUN','replayed_records':checked,
            'limitation':'Replays recorded actions; token probabilities require original backend. No promotion.'}


def run(*, output, source, owner_enabled=False, minutes=330):
    if owner_enabled is not True: raise ValueError('explicit owner execution required')
    if type(minutes) not in (int,float) or not math.isfinite(minutes) or not 1<=minutes<=330:
        raise ValueError('wall budget must be within 1..330 minutes')
    requested=Path(output)
    if requested.is_symlink(): raise ValueError('output cannot be a symlink')
    output=requested.resolve()
    if output.is_relative_to(REPO) or output==Path(output.anchor):
        raise ValueError('output must be a dedicated directory outside source')
    output.mkdir(parents=True,exist_ok=True); (output/'analysis').mkdir(exist_ok=True)
    started=time.monotonic(); deadline=started+60*minutes
    campaign={'schema':'elendril-v7-campaign-v1','source':source,'protocol_sha256':PROTOCOL_SHA,
              'status':'PREFLIGHT','results':[],'promotion':'NONE',
              'fresh_vm_recovery':'NOT_QUALIFIED; same-session fresh-process control only',
              'original_v6_checkpoints':'NOT_AVAILABLE; archived traces audited separately'}
    run_id=uuid4().hex
    try:
        check_source(REPO,source)
        write_once(output/'analysis'/'identity.json',{'source':source,'protocol_sha256':PROTOCOL_SHA})
        write_once(output/'analysis'/'protocol.json',protocol())
        if shutil.disk_usage(output).free < 12*(1<<30): raise ValueError('at least 12 GiB free required for checkpoints and snapshots')
        devices=_gpu_records(); runtimes=check_torch(REPO,devices)
        immutable_json(output/'analysis'/('environment-'+run_id+'.json'),runtimes)
        print('PREFLIGHT PASS: two isolated T4s; generating pinned data automatically.',flush=True)
        data=prepare_data(output)
        # Deterministic competitors are engineered controls, never learned cognition.
        from anra_v5.elendril.v6_data import read_dataset
        from anra_v5.elendril.v6_evaluation import ControlPolicy
        from anra_v5.elendril.v7_measurement import evaluate, verify_record
        for name in ('monitor','validation','confirmation'):
            path=output/'analysis'/'controls'/(name+'.json')
            cohort=read_dataset(data[name]['file'])
            if not path.exists():
                records={method:evaluate(cohort,ControlPolicy(method))
                    for method in ('fixed','always-query','no-query','random','recency','query-blind','oracle')}
                for record in records.values(): verify_record(cohort,record)
                immutable_json(path,{'cohort':name,'data_identity':cohort.identity,'records':records,
                    'attribution':'Engineered policies, including privileged oracle; no learned success.'})
        base={'source':source,'protocol_sha':PROTOCOL_SHA}
        def qualify(index):
            return invoke('canary',{**base,'output':str(output/'analysis'/'canaries'/('lane'+str(index)+'-'+run_id)),
                'checkpoint_root':str(output/'checkpoints'/'canaries'/('lane'+str(index)+'-'+run_id)),
                'job':{'job_id':'canary'+str(index),'method':'uniform','seed':7},
                'data_file':data['canary']['file'],'eval_file':data['monitor']['file'],
                'production_file':data['train']['file'],
                'timecap_seconds':max(1,min(1200,deadline-time.monotonic()-60))},devices[index])
        print('CANARIES: positive learning, production timing and fresh-process exact recovery on each GPU.',flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool: canaries=list(pool.map(qualify,range(2)))
        campaign['canaries']=canaries
        if any(c.get('status')!='PASS' for c in canaries): raise ValueError('NO-GO: learning/recovery canary failed')
        per_epoch=math.ceil(data['train']['manifest']['examples']/16)
        update_seconds=max(c['production_seconds_per_update'] for c in canaries)
        eval_seconds=max(c['evaluation_seconds_per_episode'] for c in canaries)
        # Five two-GPU waves in the worst lane. Diagnostics: four monitors,
        # three full development cohorts and one confirmation, each <=4x size.
        estimate=5*(4*per_epoch*update_seconds + (11*28+4*112)*4*eval_seconds)*2
        campaign['capacity_estimate']={'seconds':estimate,'policy':'2x timing headroom; five waves; no outcome dependence',
            'production_seconds_per_update':update_seconds,'evaluation_seconds_per_episode':eval_seconds}
        if estimate > deadline-time.monotonic()-120:
            raise ValueError('NO-GO: measured complete-study forecast exceeds remaining budget; no silently reduced design')
        pending=Queue()
        for job in jobs(): pending.put(job)
        campaign['status']='RUNNING'
        print('CANARIES PASS. Running nine matched 4-epoch jobs; checkpoints and reports saved incrementally.',flush=True)
        def lane(index):
            results=[]
            while time.monotonic()<deadline-180:
                try: job=pending.get_nowait()
                except Empty: break
                folder=output/'analysis'/'jobs'/job['job_id']
                request={**base,'job':job,'output':str(folder),
                    'checkpoint_root':str(output/'checkpoints'/'jobs'/job['job_id']),
                    'data_file':data['train']['file'],'monitor_file':data['monitor']['file'],
                    'train_monitor_file':data['train-monitor']['file'],
                    'eval_file':data['validation']['file'],'confirmation_file':data['confirmation']['file'],
                    'timecap_seconds':max(1,min(3600,deadline-time.monotonic()-120))}
                row=invoke('train',request,devices[index]); results.append(row)
                append_event(output/'analysis',{'event':'JOB_'+row['status'],'job':job,'receipt':row['receipt_path']})
                print(job['job_id']+': '+row['status'],flush=True)
                # Snapshot races are deferred, never treated as erased results.
                export_results(output,source=source,protocol_sha=PROTOCOL_SHA,campaign=campaign,full=False)
            return results
        with ThreadPoolExecutor(max_workers=2) as pool:
            campaign['results']=[r for rows in pool.map(lane,range(2)) for r in rows]
        complete=sum(r['status']=='COMPLETE' for r in campaign['results'])
        campaign['status']='EXECUTED' if complete==9 else 'PARTIAL'
        campaign['completed_jobs']=complete
        campaign['gpu_worker_reservation_seconds']=sum(r['attempt_wall_seconds'] for r in canaries+campaign['results'])
    except Exception as exc:
        campaign.update(status='NO_GO',error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        print('NO-GO: '+str(exc),flush=True)
    finally:
        campaign['wall_seconds']=time.monotonic()-started
        try:
            from tools.elendril_v7_report import write_summary
            campaign['summary']=write_summary(output)
        except Exception as exc: campaign['summary']={'status':'FAILED','message':str(exc)}
        try: campaign['independent_replay']=rescore(output,campaign)
        except Exception as exc:
            campaign['independent_replay']={'status':'FAILED','message':str(exc)}
            campaign['status']='INVALID_RESULTS'
        immutable_json(output/'analysis'/('campaign-'+run_id+'.json'),campaign)
        exports=export_results(output,source=source,protocol_sha=PROTOCOL_SHA,campaign=campaign,
            rescore=lambda _root,_campaign: campaign['independent_replay'])
        immutable_json(output/'analysis'/('export-'+run_id+'.json'),exports)
        print(json.dumps({'status':campaign['status'],'exports':exports},indent=2),flush=True)
    return campaign


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--source',required=True)
    p.add_argument('--owner-enabled',action='store_true'); p.add_argument('--minutes',type=float,default=330)
    args=p.parse_args(argv)
    result=run(output=args.output,source=args.source,owner_enabled=args.owner_enabled,minutes=args.minutes)
    return 0 if result['status']=='EXECUTED' else 1


if __name__=='__main__': raise SystemExit(main())
