"""Offline owner-result audit. Replays actions, never trains or infers weights.

python -m tools.analyze_elendril_v71_results --zip ANALYSIS.zip --final FINAL_STATUS.json --output NEW_DIR [--v7-zip V7_ANALYSIS.zip] [--data-dir FULL_OUTPUT/data]
"""
import argparse
import copy
from collections import Counter,defaultdict
import csv
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import statistics
import subprocess

from anra_v5.elendril.v71_data import build_suite,suite_manifest
from anra_v5.elendril.v71_evaluation import verify_suite
from anra_v5.elendril.v71_protocol import PROTOCOL_SHA,jobs
from anra_v5.elendril.v6_data import make_dataset,read_dataset
from anra_v5.elendril.v7_measurement import verify_record
from tools.elendril_v6_export import verify_zip
import zipfile

SOURCE='6104ffd7134cb4cf4b3f340bd060d24a1e2dd200'
KEYS=('episodes','correct','accuracy','nonnull_correct','nonnull_total','nonnull_accuracy','null_total',
      'correct_abstentions','query_needed','needed_queried','query_precision','query_recall','invalid',
      'verified_queries','model_calls','model_positions','answer_entropy_bits','both_correct_pairs','complete_pairs')


def bind_status(path,status):
    """Full byte/hash validation before scientific interpretation."""
    receipt=status['exports']['analysis']; identity=verify_zip(path)
    if (Path(path).stat().st_size!=receipt['zip_bytes'] or sha256(Path(path).read_bytes()).hexdigest()!=receipt['zip_sha256']
            or identity['manifest_sha256']!=receipt['manifest_sha256'] or identity['source']!=status['source']
            or identity['protocol_sha256']!=status['protocol_sha256']): raise ValueError('archive/final-status mismatch')
    if status['source']!=SOURCE or status['protocol_sha256']!=PROTOCOL_SHA: raise ValueError('unsupported study identity')
    return identity


def compact(metrics):
    out={k:metrics[k] for k in KEYS}
    out['brier']=metrics['calibration']['brier']
    return out


def portable_replay(suite,report,*,name=''):
    """Only derived Brier rounding may differ by <=2 binary64 ULPs.

    Linux/Windows pow can round a squared residual differently. Preserve the
    original receipt; all actions, confidence values, counts and other fields
    remain strict. Entropy retains the historical verifier's declared bound.
    """
    candidate=copy.deepcopy(report); differences=[]
    for cohort,record in report['records'].items():
        if record is None: continue
        episodes=record['raw']['episodes']
        groups=[('all',None,episodes)]+[('by_kind',kind,[r for r in episodes if r['kind']==kind]) for kind in report['metrics'][cohort]['by_kind']]
        for group,kind,rows in groups:
            valid=[(r['confidence'],int(r['outcome']['correct'])) for r in rows
                if type(r['confidence']) in (int,float) and math.isfinite(r['confidence']) and 0<=r['confidence']<=1]
            computed=math.fsum((p-y)**2 for p,y in valid)/len(valid) if valid else None
            target=candidate['metrics'][cohort][group] if kind is None else candidate['metrics'][cohort][group][kind]
            old=target['calibration']['brier']
            if old!=computed:
                if type(old) is not float or type(computed) is not float or not math.isfinite(old): raise ValueError('invalid Brier value')
                tolerance=2*max(math.ulp(old),math.ulp(computed))
                if abs(old-computed)>tolerance: raise ValueError('derived Brier difference exceeds two ULPs')
                differences.append({'receipt':name,'cohort':cohort,'group':group,'kind':kind,'recorded':old,'recomputed':computed,'absolute_difference':abs(old-computed),'tolerance':tolerance})
                target['calibration']['brier']=computed
    verify_suite(suite,candidate)
    return differences


def panels(record,cases):
    index={c.case_id:c for c in cases}
    groups=defaultdict(list)
    for r in record['raw']['episodes']:
        label=r['panel']
        if label=='row-order': label+='-'+index[r['case_id']].surface.order
        groups[label].append(r)
    return {name:{'episodes':len(rows),'correct':sum(r['outcome']['correct'] for r in rows),
        'invalid':sum(r['trace']['status']=='invalid_action' for r in rows),
        'first_actions':dict(Counter(r['trace']['decisions'][0]['operation'] if r['trace']['decisions'] else 'NONE' for r in rows))}
        for name,rows in groups.items()}


def trace_signature(row):
    return {'prompts':row['prompts'],'decisions':row['trace']['decisions'],'correct':row['outcome']['correct'],
            'answer':row['trace']['answer'],'status':row['trace']['status']}


def analyze(archive,final,*,data_dir=None,v7_zip=None):
    status=json.loads(Path(final).read_text(encoding='utf-8-sig')); integrity=bind_status(archive,status)
    # Bind reconstructors to the actual owner source, not merely this branch name.
    repo=Path(__file__).resolve().parents[1]; source_files={}
    for name in ('v71_data.py','v71_evaluation.py','v6_data.py','v6_evaluation.py','v7_measurement.py','cognition.py'):
        rel='anra_v5/elendril/'+name; original=subprocess.check_output(['git','show',SOURCE+':'+rel],cwd=repo)
        current=(repo/rel).read_bytes().replace(b'\r\n',b'\n')
        if current!=original: raise ValueError('reconstruction source differs: '+name)
        source_files[rel]=sha256(current).hexdigest()
    suite=build_suite(); confirmation=make_dataset('confirmation',worlds=112,seed=909)
    suite['confirmation']=confirmation.cases
    shortcut_audit={name:{'episodes':len(cases),
        'last_visible_value_correct':sum((c.environment.initial if c.surface.order=='original' else tuple(reversed(c.environment.initial)))[-1].value==c.truth.expected for c in cases),
        'maximum_tick_value_correct':sum(max(c.environment.initial,key=lambda e:e.tick).value==c.truth.expected for c in cases)}
        for name,cases in suite.items() if name.startswith('hop-') and name!='hop-2-clamped'}
    checked=0; positive=0; rows=[]; trajectories=[]; examples=[]; training=[]; comparisons=[]; control_rows=[]; rounding=[]
    final_reports={}
    with zipfile.ZipFile(archive) as z:
        get=lambda name:json.loads(z.read(name))
        if get('suite_manifest.json')['suite']!=suite_manifest(suite): raise ValueError('reconstructed suite identity differs')
        if get('data_manifest.json')['confirmation']['identity']!=confirmation.identity: raise ValueError('confirmation identity differs')
        marker=get(next(n for n in z.namelist() if n.startswith('campaign-')))
        campaign=marker['campaign']
        if marker['run_id']!=status['run_id'] or campaign['completed_jobs']!=9 or campaign['phase']!='core': raise ValueError('campaign identity/count differs')
        for name in sorted(z.namelist()):
            if name.startswith('jobs/') and (name.split('/')[-1] in ('initial.json','grammar-final.json') or '/epoch-' in name):
                r=get(name); rounding.extend(portable_replay(suite,r['evaluation'],name=name)); checked+=1
                if any(v['raw']['model_identity']!=r['checkpoint'] for v in r['evaluation']['records'].values() if v): raise ValueError('inner checkpoint binding differs')
            elif name.startswith('controls/') and name.endswith('.json'):
                r=get(name); rounding.extend(portable_replay(suite,r['evaluation'],name=name)); checked+=1
                control_rows.append({'name':name,'cohorts':{k:compact(v['all']) for k,v in r['evaluation']['metrics'].items()}})
            elif name.startswith('profiles/') and name.endswith('/timing-evaluation.json'):
                rounding.extend(portable_replay({'base':suite['base']},get(name),name=name)); checked+=1
        for job in jobs('core'):
            directory='jobs/'+job['job_id']+'/'
            required={directory+f'epoch-{e}.json' for e in (1,2,4)}|{directory+'grammar-final.json',directory+'initial.json'}
            if not required<=set(z.namelist()): raise ValueError('missing registered reports')
            final_report=get(directory+'epoch-4.json'); grammar=get(directory+'grammar-final.json')
            final_reports[job['job_id']]=final_report
            if grammar['checkpoint']!=final_report['checkpoint'] or grammar['weight_sha256']!=final_report['weight_sha256']: raise ValueError('grammar weights differ')
            if not grammar['grammar'] or final_report['grammar']: raise ValueError('decoder mode mismatch')
            for epoch,name in [(0,'initial.json'),(1,'epoch-1.json'),(2,'epoch-2.json'),(4,'epoch-4.json')]:
                r=get(directory+name)
                if r['evaluation']['conditions']!=(sorted(suite) if epoch==4 else ['base','hop-3']): raise ValueError('epoch coverage differs')
                for cohort,m in r['evaluation']['metrics'].items():
                    trajectories.append({'job_id':job['job_id'],'seed':job['seed'],'method':job['method'],'mechanism':job['mechanism'],
                        'epoch':epoch,'update':r['update'],'cohort':cohort,'checkpoint':r['checkpoint'],**compact(m['all'])})
            for cohort,m in final_report['evaluation']['metrics'].items():
                e=final_report['evaluation']; raw=e['records'][cohort]; gm=grammar['evaluation']['metrics'][cohort]
                refs=e['engineered_reference'][cohort]; first=Counter()
                for event in refs.values():
                    mismatch=event.get('first_disagreement')
                    if mismatch: first[f"{mismatch['expected']['operation']}->{mismatch['actual']['operation']}"]+=1
                row={'job_id':job['job_id'],'seed':job['seed'],'method':job['method'],'mechanism':job['mechanism'],'cohort':cohort,
                    'raw':compact(m['all']),'grammar':compact(gm['all']),
                    'by_kind':{k:compact(v) for k,v in m['by_kind'].items()},'panels':panels(raw,suite[cohort]),'first_reference_disagreements':dict(first)}
                if cohort=='memory-manipulations':
                    index={c.case_id:c for c in suite[cohort]}
                    wrong=[r for r in raw['raw']['episodes'] if r['panel']=='memory-stale-corrected' and not r['outcome']['correct']]
                    row['stale_error_breakdown']={'wrong':len(wrong),'answered_stale':sum(r['trace']['answer']==index[r['case_id']].memory.events[0].value for r in wrong),
                        'qualification':'A stale-answer match is descriptive, not an internal mechanism diagnosis.'}
                rows.append(row)
                a=raw['raw']['episodes']; b=grammar['evaluation']['records'][cohort]['raw']['episodes']
                if [v['case_id'] for v in a]!=[v['case_id'] for v in b]: raise ValueError('decoder pairing differs')
                comparisons.append({'job_id':job['job_id'],'cohort':cohort,'episodes':len(a),
                    'raw_only':sum(x['outcome']['correct'] and not y['outcome']['correct'] for x,y in zip(a,b)),
                    'grammar_only':sum(y['outcome']['correct'] and not x['outcome']['correct'] for x,y in zip(a,b)),
                    'changed_actions':sum([(d['operation'],d['argument']) for d in x['trace']['decisions']]!=[(d['operation'],d['argument']) for d in y['trace']['decisions']] for x,y in zip(a,b)),
                    'raw_calls':m['all']['model_calls'],'grammar_calls':gm['all']['model_calls'],
                    'raw_positions':m['all']['model_positions'],'grammar_positions':gm['all']['model_positions']})
                # Bound retained examples while preserving full receipts in the ZIP.
                if cohort in ('base','hop-2','hop-4','memory-manipulations','query-crossed','robustness-row-order'):
                    for success in (True,False):
                        selected=next((v for v in a if v['outcome']['correct']==success),None)
                        if selected: examples.append({'job_id':job['job_id'],'cohort':cohort,'case_id':selected['case_id'],'world_id':selected['world_id'],
                            **trace_signature(selected),'reference':refs.get(selected['case_id'])})
            log_names=[n for n in z.namelist() if n.startswith(directory+'updates-') and n.endswith('.jsonl')]
            logs=sorted([json.loads(line) for n in log_names for line in z.read(n).decode().splitlines()],key=lambda r:r['update'])
            if [r['update'] for r in logs]!=list(range(1,final_report['update']+1)): raise ValueError('incomplete/duplicate update trajectory')
            cost=next(r for r in campaign['results'] if r['job']==job)
            training.append({'job':job,'updates':len(logs),'reported_weight_sha256':final_report['weight_sha256'],
                **{k:sum(r[k] for r in logs) for k in ('examples','supervised_tokens','input_positions','padded_input_positions','padded_attention_pairs','wall_seconds')},
                'worker_reservation_seconds':cost['attempt_wall_seconds'],'peak_cuda_allocated_bytes':cost['peak_cuda_allocated_bytes'],
                'last64_mean_loss':statistics.mean(r['loss'] for r in logs[-64:]),
                'qualification':'teacher-forced training loss, not accuracy; mechanism objectives differ'})
            if training[-1]['supervised_tokens']!=cost['supervised_tokens']: raise ValueError('worker/log token count differs')
        if data_dir:
            for name in z.namelist():
                if name.startswith('positive/') and name.endswith('/positive.json'):
                    control=get(name)
                    for label,filename in (('raw','canary'),('distinct_validation','monitor')):
                        data=read_dataset(Path(data_dir)/(filename+'.json'))
                        if data.identity!=get('data_manifest.json')[filename]['identity']: raise ValueError('positive-control data differs')
                        verify_record(data,control[label]); positive+=1
        old_comparison=[]
        if v7_zip:
            old_status=json.loads((repo/'Experiment Result/V7/c9981309a94847a4a97d8115c56d75c8/FINAL_STATUS.json').read_bytes())
            if sha256(Path(v7_zip).read_bytes()).hexdigest()!=old_status['exports']['analysis']['zip_sha256']: raise ValueError('V7 archive hash differs')
            verify_zip(v7_zip)
            with zipfile.ZipFile(v7_zip) as old:
                for job in jobs('core'):
                    if job['mechanism']!='ce': continue
                    prior=json.loads(old.read(f"jobs/{job['method']}-s{job['seed']}/confirmation.json")); current=final_reports[job['job_id']]
                    a=prior['diagnosis']['records']['interactive']['raw']['episodes']; b=current['evaluation']['records']['confirmation']['raw']['episodes']
                    left={r['case_id']:trace_signature(r) for r in a}; right={r['case_id']:trace_signature(r) for r in b}
                    old_comparison.append({'job_id':job['job_id'],'same_reported_weight_fingerprint':prior['weight_sha256']==current['weight_sha256'],
                        'same_confirmation_actions_prompts_probabilities':left==right,'cases':len(a),
                        'qualification':'Receipt comparison; original/new tensors not independently retrieved. Deterministic repetitions are not additional independent seeds.'})
    return {'schema':'elendril-v71-owner-analysis-v1','run_id':status['run_id'],'cause':'UNRESOLVED','integrity':integrity,
        'independent_replay':{'status':'PASS','suite_reports':checked,'positive_records':positive,'limitation':'Actions/prompts/outcomes replayed; probabilities and weights not regenerated.'},
        'derived_brier_rounding':rounding,
        'shortcut_audit':{'qualification':'Deterministic public row-copy controls, not model behavior or composition.', 'cohorts':shortcut_audit},
        'source_files':source_files,'suite':{'identity':suite_manifest(suite)['suite_sha256'],'records':sum(map(len,suite.values())),
            'unique_worlds':len({c.world_id for cs in suite.values() for c in cs}),'cohort_sizes':{k:len(v) for k,v in suite.items()}},
        'rows':rows,'trajectories':trajectories,'examples':examples,'training':training,'grammar_comparisons':comparisons,'controls':control_rows,
        'V7_comparison':old_comparison,'within_cohort_comparisons':{k:v['evaluation']['within_cohort_comparisons'] for k,v in final_reports.items()},
        'cost':{k:campaign[k] for k in ('wall_seconds','gpu_worker_reservation_seconds','capacity_estimate')},
        'canaries':{'positive':[{'status':r['status'],'gpu_uuid':r.get('gpu_uuid'),'recovery':r.get('recovery')} for r in campaign['positive_controls']],
            'production':[{'status':r['status'],'profile':r['profile'],'recovery':r['recovery'],'gpu_uuid':r['gpu_uuid']} for r in campaign['profiles']]},
        'full_archive':{'owner_report':status['exports']['full'],'independent_custody':'NOT_AVAILABLE: analysis ZIP has no tensor payloads'}}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--zip',type=Path,required=True); p.add_argument('--final',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--data-dir',type=Path); p.add_argument('--v7-zip',type=Path)
    a=p.parse_args(); result=analyze(a.zip,a.final,data_dir=a.data_dir,v7_zip=a.v7_zip)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'METRICS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    buffer=io.StringIO(newline=''); w=csv.DictWriter(buffer,fieldnames=list(result['trajectories'][0]),lineterminator='\n'); w.writeheader(); w.writerows(result['trajectories'])
    (a.output/'TRAJECTORIES.csv').write_text(buffer.getvalue(),encoding='utf8')
    print(json.dumps({'run_id':result['run_id'],'replay':result['independent_replay'],'rows':len(result['rows']),'output':str(a.output)},indent=2))


if __name__=='__main__': main()
