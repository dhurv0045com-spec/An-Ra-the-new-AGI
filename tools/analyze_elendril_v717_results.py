"""Offline V7.17 owner archive audit: no inference, training or ZIP extraction."""
import argparse
from collections import Counter
from hashlib import file_digest
import json
import math
from pathlib import Path
import subprocess
from zipfile import ZipFile

from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.v6_data import make_dataset, read_dataset
from anra_v5.elendril.v71_data import build_suite, suite_manifest
from anra_v5.elendril.v717_protocol import PROTOCOL_SHA, jobs
from anra_v5.elendril.v717_measurement import representative_indices, audit_targets
from tools.analyze_elendril_v71_results import portable_replay, compact
from tools.analyze_elendril_v715_s8 import outcome_pairs
from tools.elendril_v6_export import verify_zip

ROOT=Path(__file__).resolve().parents[1]
SOURCE='724af4cb508173e0eb3f4397d5dcb4a42a66712f'
STAGES=(('initial',0),('epoch-1',592),('epoch-2',1184),('epoch-4',2368))


def close(actual, expected):
    if type(actual) not in (int,float) or not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('measurement arithmetic differs')


def audit_measurement(report, data, indices, *, decoded):
    """Recompute target marginals, recorded confusions and decoded exact matches.

    Loss/probabilities are not regenerated: the analysis ZIP contains no model.
    """
    if report['examples']!=len(indices) or report['audit']!=audit_targets(data):
        raise ValueError('measurement cohort or target audit differs')
    expected=[]
    value_ids=set(UTF8.encode('ABCDEFGHIJKLMNOP'))
    for index in indices:
        sample=data.samples[index]; start=sample.eligible.index(True)
        for position in range(start,len(sample.tokens)):
            token=sample.tokens[position]
            expected.append((sample.category,token,position==start,token in value_ids,token==UTF8.eos_id))
    groups=[(report['teacher_forced'],expected)]
    kinds=sorted({s[0] for s in expected})
    if sorted(report['by_kind'])!=kinds: raise ValueError('measurement task kinds differ')
    groups += [(report['by_kind'][kind],[r for r in expected if r[0]==kind]) for kind in kinds]
    for metrics, rows in groups:
        for role, position in (('all',None),('first_action',2),('value',3),('eos',4)):
            targets=Counter(str(r[1]) for r in rows if position is None or r[position])
            m=metrics[role]
            if m['target_distribution']!=dict(sorted(targets.items())) or m['tokens']!=sum(targets.values()):
                raise ValueError('teacher target marginal differs')
            observed=Counter(); correct=0
            for key, count in m['confusion'].items():
                if type(count) is not int or count<=0: raise ValueError('invalid confusion count')
                left,right=key.split('->'); int(left); int(right)
                observed[left]+=count
                if left==right: correct+=count
            if observed!=targets or m['correct']!=correct:
                raise ValueError('teacher confusion/correct count differs')
            if m['tokens']:
                close(m['accuracy'],correct/m['tokens'])
                if not math.isfinite(m['cross_entropy']) or m['cross_entropy']<0:
                    raise ValueError('nonfinite/negative loss')
            elif m['accuracy'] is not None or m['cross_entropy'] is not None:
                raise ValueError('empty role has a metric')
    exact=report['teacher_forced_exact_actions']
    if exact['examples']!=len(indices) or not 0<=exact['correct']<=len(indices): raise ValueError('teacher exact count differs')
    close(exact['accuracy'],exact['correct']/len(indices))
    d=report['decoded']
    if not decoded:
        if d['status']!='NOT_RUN' or d['examples']!=0 or d['records']: raise ValueError('unexpected decoded coverage')
        return
    if d['status']!='MEASURED' or len(d['records'])!=len(indices) or d['examples']!=len(indices):
        raise ValueError('decoded cohort differs')
    for row,index in zip(d['records'],indices):
        s=data.samples[index]; target=list(s.tokens[s.eligible.index(True):])
        if (row['task_id']!=s.task_id or row['category']!=s.category or row['expected']!=target
                or type(row['exact']) is not bool or row['exact']!=(row['generated']==target)):
            raise ValueError('decoded target or exact match differs')
    correct=sum(r['exact'] for r in d['records'])
    if d['correct']!=correct: raise ValueError('decoded correct count differs')
    close(d['accuracy'],correct/len(indices))
    close(report['teacher_minus_decode_exact_accuracy'],exact['accuracy']-d['accuracy'])


def score(metrics):
    return {k:metrics[k] for k in ('episodes','correct','accuracy','nonnull_correct','nonnull_total',
                                   'invalid','query_needed','needed_queried','query_recall')}


def small_measurement(record, *, final=False, datasets=None):
    result={}
    for label in ('train','heldout'):
        t=record[label]['teacher']; d=record[label]['decoded']
        result[label]={
            'examples':t['examples'],'teacher_exact':t['teacher_forced_exact_actions'],
            'roles':{k:{f:v for f,v in m.items() if f in ('tokens','correct','accuracy','cross_entropy')}
                     for k,m in t['teacher_forced'].items()},
            'decoded':{k:v for k,v in d['decoded'].items() if k!='records'},
            'teacher_decode_gap':d['teacher_minus_decode_exact_accuracy']}
        if final:
            result[label]['by_kind']={k:{r:{f:v for f,v in m.items() if f in ('tokens','correct','accuracy')}
                         for r,m in value.items() if r!='all'} for k,value in t['by_kind'].items()}
            if datasets is not None:
                examples={e.task_id:e for e in datasets[label].examples}
                grouped={}
                for r in d['decoded']['records']:
                    e=examples[r['task_id']]; key=e.kind+'/'+e.stage
                    item=grouped.setdefault(key,{'examples':0,'correct':0})
                    item['examples']+=1; item['correct']+=r['exact']
                result[label]['decoded_by_observation_stage']=grouped
    return result


def analyze(path, *, train_file=None):
    path=Path(path); integrity=verify_zip(path)
    if (integrity['source'],integrity['protocol_sha256'],integrity['campaign_status'])!=(SOURCE,PROTOCOL_SHA,'EXECUTED'):
        raise ValueError('unexpected owner study identity')
    subprocess.run(['git','diff','--exit-code',SOURCE,'--','anra_v5/elendril'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    train=read_dataset(Path(train_file)) if train_file else make_dataset('train',seed=606,target_bytes=2200000)
    validation=make_dataset('validation',seed=707,worlds=112)
    confirmation=make_dataset('confirmation',seed=909,worlds=112)
    suite=build_suite(); suite['confirmation']=confirmation.cases
    registered=jobs('optimization'); expected={j['job_id']:j for j in registered}
    checked=measurements=0; rounding=[]; models={}; finals={}; initials={}
    previous=json.loads((ROOT/'Experiment Result/V7.16/51f3aeb7cb604c6d8c80f6ae4cd9f3ff/METRICS.json').read_bytes())
    with ZipFile(path) as z:
        read=lambda n:json.loads(z.read(n))
        markers=[n for n in z.namelist() if n.startswith('campaign-') and n.endswith('.json')]
        if len(markers)!=1: raise ValueError('expected single campaign marker')
        marker=read(markers[0]); campaign=marker['campaign']
        if (campaign['source'],campaign['protocol_sha256'],campaign['phase'],campaign['status'],campaign['completed_jobs'])!=(SOURCE,PROTOCOL_SHA,'optimization','EXECUTED',8):
            raise ValueError('campaign inventory differs')
        if (marker['source']!=SOURCE or marker['protocol_sha256']!=PROTOCOL_SHA or marker['run_id']!=campaign['run_id']
                or marker['required_jobs']!=list(expected)): raise ValueError('marker binding differs')
        if read('identity.json')!={'source':SOURCE,'protocol_sha256':PROTOCOL_SHA,'phase':'optimization'}:
            raise ValueError('archive identity differs')
        dm=read('data_manifest.json')
        for name,data in (('train',train),('validation',validation),('confirmation',confirmation)):
            if dm[name]['identity']!=data.identity: raise ValueError('dataset identity differs')
        if read('suite_manifest.json')['suite']!=suite_manifest(suite): raise ValueError('evaluation reconstruction differs')
        if len(campaign['profiles'])!=4: raise ValueError('profile coverage differs')
        for p in campaign['profiles']:
            if (p['status'],p['gradient_probe']['status'],p['recovery']['status'])!=('PASS','PASS','PASS'):
                raise ValueError('profile or recovery failed')
        for n in z.namelist():
            if n.startswith('profiles/') and n.endswith('/timing-evaluation.json'):
                rounding.extend(portable_replay({'base':suite['base']},read(n),name=n)); checked+=1
        workers=campaign['results']
        if len(workers)!=len(expected) or len({r['job']['job_id'] for r in workers})!=len(expected):
            raise ValueError('worker inventory differs')
        for worker in workers:
            job=worker['job']; jid=job['job_id']; folder='jobs/'+jid+'/'
            if expected.get(jid)!=job or worker['status']!='COMPLETE' or worker['parameter_count']!=19871616:
                raise ValueError('worker qualification differs')
            trajectory=[]
            for stage,update in STAGES:
                r=read(folder+stage+'.json'); m=read(folder+'measurements/'+stage+'.json')
                if (r['update']!=update or m['update']!=update or r['checkpoint']!=m['checkpoint']
                        or r['weight_sha256']!=m['weight_sha256'] or m['job']!=job or r['grammar'] is not False):
                    raise ValueError('checkpoint measurement binding differs')
                conditions=sorted(suite) if update==2368 else ['base','hop-3']
                if r['evaluation']['conditions']!=conditions: raise ValueError('evaluation conditions differ')
                if any(v['raw']['model_identity']!=r['checkpoint'] for v in r['evaluation']['records'].values() if v):
                    raise ValueError('inner model binding differs')
                rounding.extend(portable_replay(suite,r['evaluation'],name=folder+stage)); checked+=1
                for label,data in (('train',train),('heldout',validation)):
                    if m[label]['dataset_identity']!=data.identity: raise ValueError('measurement dataset binding differs')
                    audit_measurement(m[label]['teacher'],data,tuple(range(len(data.samples))),decoded=False)
                    audit_measurement(m[label]['decoded'],data,representative_indices(data),decoded=True)
                measurements+=1
                if update==0:
                    if m['gradient_probe']['status']!='PASS' or not all(p['passed'] for p in m['gradient_probe']['parameters']):
                        raise ValueError('initial gradient parity differs')
                    initials[jid]=m['weight_sha256']
                trajectory.append({'stage':stage,'update':update,'checkpoint':m['checkpoint'],
                    **small_measurement(m,final=update==2368,datasets={'train':train,'heldout':validation}),
                    'interactive':{k:score(v['all']) for k,v in r['evaluation']['metrics'].items() if v and k in ('base','hop-3')}})
            final=r; grammar=read(folder+'grammar-final.json')
            if (grammar['checkpoint']!=final['checkpoint'] or grammar['weight_sha256']!=final['weight_sha256']
                    or grammar['update']!=2368 or grammar['grammar'] is not True or grammar['evaluation']['conditions']!=sorted(suite)
                    or worker['checkpoint']!=final['checkpoint']): raise ValueError('final decoder/worker binding differs')
            rounding.extend(portable_replay(suite,grammar['evaluation'],name=folder+'grammar-final')); checked+=1
            logs=sorted((json.loads(line) for n in z.namelist() if n.startswith(folder+'updates-') and n.endswith('.jsonl')
                         for line in z.read(n).splitlines()),key=lambda r:r['update'])
            if [r['update'] for r in logs]!=list(range(1,2369)) or any(not math.isfinite(r['loss']) for r in logs):
                raise ValueError('training trajectory incomplete or nonfinite')
            if worker['global_update']!=2368 or worker['total_updates']!=2368:
                raise ValueError('worker cursor differs')
            work={k:math.fsum(r[k] for r in logs) for k in ('supervised_tokens','examples','input_positions','wall_seconds')}
            if work['supervised_tokens']!=worker['supervised_tokens']: raise ValueError('training exposure differs')
            finals[jid]=final
            failure=next((r for r in final['evaluation']['records']['hop-3']['raw']['episodes'] if not r['outcome']['correct']),None)
            models[jid]=dict(job=job,checkpoint=final['checkpoint'],initial_weights_sha256=initials[jid],
                plan_sha256=worker['metadata']['plan_sha256'],work=work,
                last64_mean_loss=math.fsum(r['loss'] for r in logs[-64:])/64,
                trajectory=trajectory,
                cohorts={k:{'raw':score(v['all']),'grammar':score(grammar['evaluation']['metrics'][k]['all']),
                            **({'by_kind':{kind:score(value) for kind,value in v['by_kind'].items()}} if k in ('base','confirmation','hop-3') else {})}
                         for k,v in final['evaluation']['metrics'].items()},
                hop3_first_failure=({k:failure[k] for k in ('case_id','prompts','outcome','confidence')}|
                    {'decisions':failure['trace']['decisions'],'status':failure['trace']['status']}) if failure else None)
            models[jid]['hop3_first_actions']=dict(Counter(e['trace']['decisions'][0]['operation']
                for e in final['evaluation']['records']['hop-3']['raw']['episodes']))
            if job['arm']=='baseline':
                prior=previous['models'][f'S20-blocked-ce-s{job["seed"]}']
                models[jid]['historical_V716_comparison']={
                    'compared_cohorts':len(suite),'compared_metric_keys':sorted(compact(final['evaluation']['metrics']['base']['all'])),
                    'different_cohorts':[k for k,v in final['evaluation']['metrics'].items() if compact(v['all'])!=prior['cohorts'][k]['raw']],
                    'scope':'Recorded aggregate metrics only; model tensors and optimizer not independently restored.'}
        pairs=[]
        for job in registered:
            if job['arm']=='baseline': continue
            base=f'S20-baseline-i{job["seed"]}-o{job["seed"]}'; jid=job['job_id']
            if initials[base]!=initials[jid]: raise ValueError('paired initialization weights differ')
            if job['arm']=='low-lr' and models[base]['plan_sha256']!=models[jid]['plan_sha256']:
                raise ValueError('LR intervention changed sampling plan')
            if job['arm']=='crossed' and models[base]['plan_sha256']==models[jid]['plan_sha256']:
                raise ValueError('order intervention did not change plan')
            pairs.append({'baseline':base,'intervention':jid,
                'confirmation_pairs':outcome_pairs(finals[base]['evaluation']['records']['confirmation']['raw']['episodes'],
                                                    finals[jid]['evaluation']['records']['confirmation']['raw']['episodes']),
                'base_pairs':outcome_pairs(finals[base]['evaluation']['records']['base']['raw']['episodes'],
                                            finals[jid]['evaluation']['records']['base']['raw']['episodes'])})
    with path.open('rb') as stream: archive_sha=file_digest(stream,'sha256').hexdigest()
    return dict(schema='elendril-v717-owner-audit-v1',run_id=campaign['run_id'],source=SOURCE,protocol_sha256=PROTOCOL_SHA,
        archive={'filename':path.name,'bytes':path.stat().st_size,'sha256':archive_sha,**integrity},
        verification={'suite_reports_replayed':checked,'measurement_milestones_audited':measurements,
                      'brier_cross_platform_rounding_adjustments':len(rounding),
                      'scope':'Recorded-action replay, target/confusion/decoded arithmetic and provenance. No neural probability regeneration.',
                      'final_delivery':'NOT_VERIFIED: separate FINAL_STATUS and FULL ZIP not supplied'},
        execution={k:campaign[k] for k in ('completed_jobs','wall_seconds','gpu_worker_reservation_seconds')},
        models=models,paired_interventions=pairs,cause='UNRESOLVED',
        interpretation={
            'supported':'Lower LR improves fit and standard raw episode outcomes in all three matched seeds; order effects depend on initialization.',
            'unresolved':'No demonstrated internal mechanism for continued one-hop/three-hop or novel-format failures.',
            'regression':'Low LR reduces raw alias-cohort correct count from 55 to 40 /336.',
            'confirmation_scope':'Public development set used for reporting; not independent sealed evidence.'})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--zip',required=True,type=Path)
    p.add_argument('--train-file',type=Path); p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(argv)
    if a.output.exists(): raise FileExistsError('refuse to replace existing curated metrics')
    result=analyze(a.zip,train_file=a.train_file)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'run_id':result['run_id'],**result['verification']},indent=2))


if __name__=='__main__': main()
