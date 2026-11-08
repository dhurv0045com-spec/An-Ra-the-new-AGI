"""Reproduce compact owner-result analysis. No inference or training.

python analyze.py --zip ANALYSIS.zip --final FINAL_STATUS.json --output NEW_DIRECTORY [--data-dir FULL_OUTPUT/data]
Datasets enable independent recorded-action replay; archive integrity is always checked.
"""
import argparse
import copy
from collections import Counter
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from tools.elendril_v6_export import verify_zip
from anra_v5.elendril.v6_data import read_dataset
from anra_v5.elendril.v7_diagnostics import verify_diagnosis, _summary, _cohorts
from anra_v5.elendril.v7_measurement import verify_record


def analyze(archive,final,data_dir=None):
    status=json.loads(Path(final).read_text(encoding='utf-8-sig'))
    identity=verify_zip(archive)
    actual=sha256(Path(archive).read_bytes()).hexdigest()
    if actual!=status['exports']['analysis']['zip_sha256']: raise ValueError('ZIP differs from owner final status')
    if identity['source']!=status['source']: raise ValueError('Source mismatch')
    if identity['manifest_sha256']!=status['exports']['analysis']['manifest_sha256']: raise ValueError('Manifest mismatch')
    datasets={}
    if data_dir:
        for name in ('monitor','validation','confirmation','train-monitor','canary'):
            datasets[name]=read_dataset(Path(data_dir)/(name+'.json'))
    checked=0; rows=[]; trajectories=[]; examples=[]; entropy_differences=[]
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('data_manifest.json'))
        for name,data in datasets.items():
            if data.identity!=manifest[name]['identity']: raise ValueError('Dataset mismatch: '+name)
        job_names=sorted({n.split('/')[1] for n in z.namelist() if n.startswith('jobs/')})
        if len(job_names)!=9: raise ValueError('Expected nine jobs')
        for name in z.namelist():
            if name.startswith('jobs/') and any('/'+prefix in name for prefix in ('diagnostic-u','development-u','training-u','confirmation.json')):
                report=json.loads(z.read(name)); diagnosis=report['diagnosis']
                if datasets:
                    data=next(d for d in datasets.values() if d.identity==report['data_identity'])
                    # Cross-Python reduction can change derived entropy's last
                    # bits. Preserve original bytes; permit only this derived
                    # field within a declared binary64 tolerance. All raw action
                    # records, outcomes, counts and other summaries stay strict.
                    cloned=copy.deepcopy(diagnosis)
                    recomputed=_summary(_cohorts(data),diagnosis['records'])
                    for label,fields in recomputed.items():
                        old=diagnosis['findings'][label].get('answer_entropy_bits')
                        value=fields.get('answer_entropy_bits')
                        if old!=value:
                            tolerance=32*sys.float_info.epsilon*(1+abs(value))*max(1,len(fields['answer_counts']))
                            if old is None or not math.isfinite(old) or abs(old-value)>tolerance:
                                raise ValueError('Entropy discrepancy exceeds declared tolerance')
                            entropy_differences.append({'receipt':name,'cohort':label,'difference':abs(old-value),'tolerance':tolerance})
                            cloned['findings'][label]['answer_entropy_bits']=value
                    verify_diagnosis(data,cloned); checked+=1
            elif name.startswith('controls/'):
                if datasets:
                    report=json.loads(z.read(name))
                    for record in report['records'].values(): verify_record(datasets[report['cohort']],record); checked+=1
            elif name.startswith('canaries/') and name.endswith('/positive.json') and datasets:
                report=json.loads(z.read(name)); verify_record(datasets['canary'],report['raw'])
                verify_record(datasets['monitor'],report['distinct_validation']); checked+=1
        for job in job_names:
            def report(file): return json.loads(z.read('jobs/'+job+'/'+file))['diagnosis']
            final_report=report('confirmation.json'); f=final_report['findings']['interactive']
            metrics=final_report['records']['interactive']['metrics']
            supplies=final_report['findings']['supplied']
            swaps=final_report['records']['interactive-swap']['metrics']['pairs']
            pairs=Counter()
            for e in final_report['episodes']:
                if e['cohort']=='interactive':
                    pairs['supplied_only']+=int(not e['interactive_correct'] and e['supplied_correct'])
                    pairs['interactive_only']+=int(e['interactive_correct'] and not e['supplied_correct'])
                    if e['queried_then_prompt_identical'] is not None:
                        pairs['prompt_comparisons']+=1
                        pairs['identical_postquery_prompts']+=int(e['queried_then_prompt_identical'])
            row={'job':job,'confirmation_correct':metrics['all']['correct'],'episodes':112,
                 'nonnull_correct':f['nonnull_correct'],'nonnull_total':f['nonnull_total'],
                 'correct_abstentions':f['correct_abstentions'],'null_total':f['null_total'],
                 'needed_queried':f['needed_queried'],'query_needed':f['query_needed'],
                 'needless_queries':f['needless_query'],'invalid':metrics['all']['invalid'],
                 'supplied_nonnull_correct':supplies['nonnull_correct'],
                 'swap_both_correct':swaps['both_correct'],'swap_pairs':swaps['complete_pairs'],
                 'by_kind':{k:v['correct'] for k,v in metrics['by_kind'].items()},
                 'paired_supply_effect':dict(pairs)}
            rows.append(row)
            for prefix in ('development','training','diagnostic'):
                updates=(592,1184,2368) if prefix!='diagnostic' else (0,74,148,222,296,592,1184,2368)
                for u in updates:
                    d=report(prefix+'-u'+str(u)+'.json'); m=d['records']['interactive']['metrics']['all']; f=d['findings']['interactive']
                    trajectories.append({'job':job,'cohort':prefix,'update':u,'correct':m['correct'],
                        'episodes':m['episodes'],'nonnull_correct':f['nonnull_correct'],
                        'nonnull_total':f['nonnull_total'],'needed_queried':f['needed_queried'],
                        'query_needed':f['query_needed'],'answer_entropy_bits':f['answer_entropy_bits']})
            if job=='blocked-s19':
                lookup={c.case_id:c for c in datasets.get('confirmation',()).cases} if datasets else {}
                for kind,wanted in [('route',False),('known',False),('missing',True)]:
                    r=next(r for r in final_report['records']['interactive']['raw']['episodes'] if r['kind']==kind and r['outcome']['correct']==wanted)
                    examples.append({'job':job,'kind':kind,'case_id':r['case_id'],'correct':r['outcome']['correct'],
                        'expected':lookup[r['case_id']].truth.expected if lookup else 'UNAVAILABLE',
                        'answer':r['trace']['answer'],'status':r['trace']['status'],
                        'decisions':r['trace']['decisions'],'prompts':r['prompts']})
        controls=json.loads(z.read('controls/confirmation.json'))
        control_scores={k:v['metrics']['all']['correct'] for k,v in controls['records'].items()}
        science=json.loads(z.read('scientific_report.json'))
        aggregates={}
        for method in ('uniform','blocked','interleaved'):
            selected=[r for r in rows if r['job'].startswith(method+'-')]
            aggregates[method]={'seed_scores':[next(r['confirmation_correct'] for r in selected if r['job'].endswith('-s'+str(s))) for s in (7,19,41)],
                'mean_accuracy':sum(r['confirmation_correct'] for r in selected)/336,
                'nonnull_correct_across_model_seeds':sum(r['nonnull_correct'] for r in selected),
                'nonnull_denominator_across_model_seeds':228,
                'swap_both_correct_across_model_seeds':sum(r['swap_both_correct'] for r in selected),
                'swap_pair_denominator_across_model_seeds':96,
                'by_kind_across_model_seeds':{k:sum(r['by_kind'][k] for r in selected) for k in selected[0]['by_kind']}}
    return {'source':status['source'],'analysis_zip_sha256':actual,'archive_integrity':'VERIFIED',
            'owner_status':status['status'],'owner_delivery_status':status['delivery_status'],
            'independent_recorded_action_replay':{'status':'PASS' if datasets else 'NOT_RUN','records':checked},
            'derived_entropy_compatibility':{'affected_fields':len(entropy_differences),
                'max_absolute_difference':max((r['difference'] for r in entropy_differences),default=0),
                'policy':'Only derived entropy normalized in a copy within 32*epsilon*(1+abs(entropy))*label_count; all raw traces unchanged.'},
            'limits':'No model inference; full checkpoint ZIP not supplied. Seed aggregates reuse the same worlds; not independent world counts.',
            'scientific_conclusion':'CAUSE UNRESOLVED','jobs':rows,'method_aggregates':aggregates,
            'confirmation_controls':control_scores,'trajectories':trajectories,'selected_cases':examples,
            'pre_finalization_controller_wall_seconds':status['wall_seconds'],
            'worker_reservation_seconds':status['gpu_worker_reservation_seconds'],
            'canaries':[{k:c.get(k) for k in ('gpu_uuid','status','wall_seconds')} for c in status['canaries']],
            'reported_scientific_report_errors':science['errors']}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--zip',type=Path,required=True); p.add_argument('--final',type=Path,required=True)
    p.add_argument('--data-dir',type=Path); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    result=analyze(a.zip,a.final,a.data_dir); destination=a.output
    destination.mkdir(parents=True,exist_ok=False)
    (destination/'METRICS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    with (destination/'TRAJECTORIES.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(result['trajectories'][0])); writer.writeheader(); writer.writerows(result['trajectories'])
    print(json.dumps({k:result[k] for k in ('archive_integrity','independent_recorded_action_replay','method_aggregates','confirmation_controls')},indent=2))
