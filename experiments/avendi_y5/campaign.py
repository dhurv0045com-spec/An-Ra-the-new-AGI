"""Run development, seal each independent cohort, then evaluate its frozen models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from anra_v5.palantiri import CampaignOrchestrator, SessionJob, export_session_archives
from anra_v5.planthir.archives import safe_extract_planthir_archive, verify_session_archive
from . import composition, information, reader
from .runtime import Context, digest, publish, read

MODULES = {'AB': reader, 'F': reader, 'C': composition, 'G': composition, 'D': information, 'E': information}
HERE = Path(__file__).resolve().parent


def registry():
    rows = reader.jobs() + composition.jobs() + information.jobs()
    out = {j['id']: j for j in rows}
    if len(out) != len(rows) or any(p not in out for j in rows for p in j['depends']):
        raise ValueError('invalid experiment dependencies')
    return out


def identity():
    source = subprocess.check_output(['git','rev-parse','HEAD'], cwd=HERE, text=True).strip()
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py'))}
    return source, digest([files, registry()])


def checks():
    res = {m.__name__: m.preflight() for m in dict.fromkeys(MODULES.values())}
    if not all(r.get('passed', r.get('ok', False)) for r in res.values()):
        raise ValueError(f'engineering fixture failed: {res}')
    return res


def receipt(root, split, jid):
    return root / split / (jid+'.json')


def worker(args, jobs, source, protocol):
    job = jobs[args.worker]
    if args.device.startswith('cuda') and torch.cuda.device_count() > 1:
        # A host-wide CUDA_VISIBLE_DEVICES can survive the scheduler's setdefault.
        # Allocation is relative to that original logical ordering.
        args.device = os.environ.get('PLANTHIR_ASSIGNED_DEVICE',args.device)
    if job['id'] not in read(args.work_root/'DESIGN.json')['jobs']:
        raise ValueError('job outside sealed design')
    ctx = Context(args.work_root, job, args.split, args.device, args.canary, args.full, args.inputs, source)
    info = read(ctx.root / 'DESIGN.json')
    if info['protocol'] != protocol or info['source'] != source or info['canary'] != args.canary:
        raise ValueError('worker design mismatch')
    if args.split == 'confirm':
        lock = read(ctx.root / 'locks' / (job['stage']+'.json'))
        sha = lock['eligible'][job['id']]
        if sha != digest(read(receipt(ctx.root,'dev',job['id']))):
            raise ValueError('development selection changed after seal')
    torch.set_num_threads(2)
    torch.manual_seed(job['seed'])
    start = time.monotonic()
    result = MODULES[job['stage']].run(job, ctx)
    status = result.get('status', 'COMPLETED')
    if status not in ('COMPLETE','COMPLETED','GATED','BLOCKED'):
        raise ValueError(f'unrecognized job status: {status}')
    if args.split == 'dev' and job['updates'] and status in ('COMPLETE','COMPLETED') and not ctx.trained:
        raise ValueError('missing final checkpoint')
    publish(receipt(ctx.root,args.split,job['id']), dict(job=job, source=source, protocol=protocol,
        split=args.split, result=result, status=status, weights=ctx.trained,
        seconds=time.monotonic()-start, canary=args.canary))


def restored(root, split, job, source, protocol, canary):
    path = receipt(root,split,job['id'])
    if not path.exists():
        return None
    row = read(path)
    if (row['job'],row['source'],row['protocol'],row['split'],row['canary']) != (job,source,protocol,split,canary):
        raise ValueError(f'incompatible receipt: {path}')
    if row['weights']:
        from v5_training.checkpoint import CheckpointStore
        state,_ = CheckpointStore(root/'weights',job['id']).restore(row['weights']['checkpoint_sha256'])
        if state.global_update != row['weights']['updates'] or state.identities.source_commit != source:
            raise ValueError('inconsistent completed checkpoint')
    return row


def cohort(root, stage, group, dev):
    if any(j not in dev for j in group):
        return None
    eligible = {j:digest(dev[j]) for j in group if dev[j]['weights']}
    lock = dict(stage=stage, eligible=eligible, dispositions={j:dev[j]['status'] for j in group},
                dev_hashes={j:digest(dev[j]) for j in group})
    publish(root/'locks'/(stage+'.json'), lock)
    return eligible


def execute(args, jobs, source, protocol):
    root = args.work_root
    args.output.mkdir(parents=True,exist_ok=True)
    if args.resume:
        if (root/'DESIGN.json').exists():
            raise ValueError('use an empty work root to import FULL, or omit --resume for in-place continuation')
        verify_session_archive(args.resume,expected_source_commit=source,require_sidecar_sha256=False,
                               required_members=['DESIGN.json','CAMPAIGN.json'])
        safe_extract_planthir_archive(args.resume,root)
    if not args.canary:
        dirty = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=HERE,text=True)
        if dirty.strip():
            raise ValueError('production requires clean tracked sources')
        if torch.cuda.device_count() < 2:
            raise ValueError('owner campaign requires two GPUs')
    full_hash = None
    if args.full is not None:
        with open(args.full,'rb') as f:
            sha = hashlib.sha256()
            for block in iter(lambda:f.read(4*1024**2),b''):
                sha.update(block)
        full_hash = sha.hexdigest()
        if full_hash != '7d006b2e72c18ed5a763d873cb23c5bba053e6ce8426c79ad13cb1857c2bb4a1':
            raise ValueError('Y4 FULL differs from the verified owner baseline')
    design = dict(y4_full_sha256=full_hash,source=source,protocol=protocol,canary=args.canary,jobs=jobs)
    publish(root/'DESIGN.json', design)
    publish(root/'PREFLIGHT.json', checks())
    devices = ['cpu'] if args.canary else ['cuda:0','cuda:1']
    stages = sorted({j['stage'] for j in jobs.values()})
    start = time.monotonic()
    reserve = min(30, args.minutes*.15)
    os.environ['AVENDI_Y5_STOP_EPOCH'] = str(time.time()+60*(args.minutes-reserve))
    coordinator = CampaignOrchestrator(campaign_id='avendi_y5', source_commit=source, work_dir=root/'coordinator',
                   max_minutes=args.minutes, reserve_minutes=reserve, min_disk_bytes=0 if args.canary else 1024**3)
    rows = {s:{} for s in ('dev','confirm')}
    for split, bucket in rows.items():
        for jid, job in jobs.items():
            r = restored(root,split,job,source,protocol,args.canary)
            if r:
                bucket[jid] = r
    errors = []
    attempted = set()
    admission = []
    while True:
        queue = [('dev',j) for j,v in jobs.items() if j not in rows['dev'] and all(p in rows['dev'] for p in v['depends'])]
        for stage in stages:
            group = [j for j,v in jobs.items() if v['stage']==stage]
            eligible = cohort(root,stage,group,rows['dev'])
            if eligible is not None:
                if any(j not in eligible for j in rows['confirm'] if jobs[j]['stage']==stage):
                    raise ValueError('confirm receipt outside frozen eligibility')
                queue.extend(('confirm',j) for j in eligible if j not in rows['confirm'])
        queue = [x for x in queue if x not in attempted]
        if not queue:
            break
        # Interleave stages across workers, prioritizing inexpensive prerequisites.
        queue.sort(key=lambda x:(bool(jobs[x[1]]['updates']),sum(jobs[k]['stage']==jobs[x[1]]['stage'] for k in rows[x[0]]), x[1]))
        if time.monotonic()-start >= (args.minutes-reserve)*60:
            break
        wave = queue[:len(devices)]
        if len(wave)==2 and jobs[wave[0][1]]['stage']==jobs[wave[1][1]]['stage']:
            wave[1] = next((x for x in queue[1:] if jobs[x[1]]['stage']!=jobs[wave[0][1]]['stage']),wave[1])
        specs = [SessionJob(jid,jobs[jid]['stage'],jobs[jid]['seed'],jobs[jid]['seed'],
                            max(1, jobs[jid]['updates'] if split=='dev' and not args.canary else 1))
                 for split,jid in wave]
        remaining = args.minutes-(time.monotonic()-start)/60
        estimates = {}
        for spec, (split, jid) in zip(specs, wave):
            samples = [r['seconds']/max(1,r['job']['updates']) for r in rows['dev'].values()
                       if r['weights'] and r['job']['arm']==jobs[jid]['arm']]
            estimates[jid] = max(samples,default=.2) if split=='dev' else 5.
        weight_bytes = sum(p.stat().st_size for p in (root/'weights').rglob('*') if p.is_file())
        plan = coordinator.admit_slice(specs,measured_seconds_per_update=.2, per_job_seconds_per_update=estimates,
                 per_job_disk_bytes=100000 if args.canary else 210_000_000,parallel_workers=len(devices),
                 outer_remaining_minutes=remaining,startup_overhead_seconds=5,
                 export_reserve_bytes=weight_bytes,available_disk_bytes=min(shutil.disk_usage(root).free,shutil.disk_usage(args.output).free))
        admission.append(plan)
        allowed = set(plan['admitted_job_ids'])
        wave = [x for x in wave if x[1] in allowed]
        specs = [s for s in specs if s.job_id in allowed]
        if not specs:
            break
        attempted.update(wave)
        split_map = {j:s for s,j in wave}
        def command(s, _):
            flags = ['--canary'] if args.canary else []
            full = ['--full',str(args.full)] if args.full else []
            return (sys.executable,'-m','experiments.avendi_y5.campaign','--worker',s.job_id,
                    '--split',split_map[s.job_id],'--device',devices[0],'--work-root',str(root),
                    *flags,*full,'--inputs',*[str(p) for p in args.inputs])
        wave_id = str(time.time_ns())
        runner = CampaignOrchestrator(campaign_id='avendi_y5',source_commit=source,work_dir=root/'waves'/wave_id)
        runner.run_waves(specs,command_builder=command,devices=devices)
        for split,jid in wave:
            row = restored(root,split,jobs[jid],source,protocol,args.canary)
            if row is None:
                errors.append(dict(split=split,job=jid,wave=wave_id))
            else:
                rows[split][jid] = row
    pending = sorted(set(jobs)-set(rows['dev']))
    confirm_pending = sorted(j for j,r in rows['dev'].items() if r['weights'] and j not in rows['confirm'])
    from .analysis import summarize
    analysis = summarize({s:{j:dict(r['result'],engineering_canary=args.canary,stage=jobs[j]['stage'],arm=jobs[j]['arm'],seed=jobs[j]['seed'])
                             for j,r in bucket.items()} for s,bucket in rows.items()})
    dispositions = {j:r['status'] for j,r in rows['dev'].items()}
    status = ('PARTIAL' if pending or confirm_pending or errors else
              'FINISHED_WITH_BLOCKS' if 'BLOCKED' in dispositions.values() else 'FINISHED')
    data = dict(execution=status, scientific_status='SEE_PER_CONTRAST',engineering_canary=args.canary,
                jobs=len(jobs),dev=len(rows['dev']),confirm=len(rows['confirm']),dispositions=dispositions,
                pending=pending,confirm_pending=confirm_pending,errors=errors,admission=admission)
    for name, value in [('CAMPAIGN',data),('ANALYSIS',analysis)]:
        (root/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    summary = {name:root/name for name in ('DESIGN.json','PREFLIGHT.json','CAMPAIGN.json','ANALYSIS.json')}
    evidence = dict(summary)
    for folder in ('dev','confirm','locks','exposure','waves'):
        for p in (root/folder).rglob('*'):
            if p.is_file() and (p.suffix in ('.json','.log') or p.name.endswith('.txt')):
                evidence[p.relative_to(root).as_posix()] = p
    full = dict(evidence)
    full.update({p.relative_to(root).as_posix():p for p in (root/'weights').rglob('*') if p.is_file()})
    export_session_archives(root,args.output,prefix='AVENDI_Y5',summary_files=summary,evidence_files=evidence,
                            full_files=full,status=status,source_commit=source,protocol_sha256=protocol,
                            rescore={'status':'NOT_INDEPENDENTLY_RESCORED'})
    print(json.dumps({k:data[k] for k in ('execution','jobs','dev','confirm','errors')}))
    return int(bool(errors))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--worker')
    parser.add_argument('--split',choices=['dev','confirm'],default='dev')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--work-root',type=Path,default=Path('work/avendi_y5'))
    parser.add_argument('--output',type=Path,default=Path('.'))
    parser.add_argument('--inputs',nargs='*',type=Path,default=[Path('/kaggle/input')])
    parser.add_argument('--full',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--canary',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='exercise representative paths with tiny models and 1-2 updates')
    parser.add_argument('--minutes',type=float,default=float(os.environ.get('PLANTHIR_SESSION_MINUTES',540)))
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--inventory',action='store_true')
    args = parser.parse_args()
    args.work_root = args.work_root.resolve()
    if args.preflight:
        print(json.dumps(checks())); return 0
    jobs = registry()
    if args.smoke:
        args.canary = True
        wanted = {'AB-reader-fixed-ce-s7','AB-answer-fixed-ce-s7','F-new_cap-s7','D-informative-s7','E-relevant-s7',
                  'C-baseline-s7','C-scout-s7','C-iit-s7',next(k for k,v in jobs.items() if v['stage']=='G')}
        jobs = {k:v for k,v in jobs.items() if k in wanted}
    if args.inventory:
        print(json.dumps(jobs,indent=2)); return 0
    source,protocol = identity()
    if not args.full:
        matches = {p.resolve() for root in args.inputs if root.exists() for p in
                   ([root] if root.is_file() and root.name=='AVENDI_Y4_FULL.zip' else root.rglob('AVENDI_Y4_FULL.zip') if root.is_dir() else [])}
        if len(matches)>1:
            raise ValueError('multiple Y4 archives, select --full')
        args.full = next(iter(matches),None)
    if args.worker:
        worker(args,jobs,source,protocol); return 0
    if os.environ.get('PLANTHIR_RESUME_INPUT') and not args.resume:
        args.resume = Path(os.environ['PLANTHIR_RESUME_INPUT'])
    return execute(args,jobs,source,protocol)


if __name__ == '__main__':
    raise SystemExit(main())
