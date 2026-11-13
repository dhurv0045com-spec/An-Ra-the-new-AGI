"""Explain measured divergences without promoting correlation to root cause."""
import json
from pathlib import Path
from tools.elendril_v6_export import _write_derived


def write_diagnosis(output, registered):
    root = Path(output); rows = []; missing = []
    (root/'analysis').mkdir(parents=True,exist_ok=True)
    for job in registered:
        folder = root/'analysis'/'jobs'/job['job_id']
        for epoch, name in ((0,'initial'),(1,'epoch-1'),(2,'epoch-2'),(4,'epoch-4')):
            path = folder/'measurements'/(name+'.json')
            if not path.is_file():
                missing.append(f'{job["job_id"]}/{name}'); continue
            from tools.elendril_v717_worker import verify_measurement
            m = verify_measurement(path,job)
            row = dict(job_id=job['job_id'],arm=job['arm'],initialization_seed=job['seed'],
                order_seed=job['order_seed'],lr=job['lr'],epoch=epoch,
                update=m['update'],checkpoint=m['checkpoint'],weight_sha256=m['weight_sha256'])
            for label in ('train','heldout'):
                t = m[label]['teacher']; d = m[label]['decoded']
                row[label] = dict(
                    examples=t['examples'],
                    token_ce=t['teacher_forced']['all']['cross_entropy'],
                    first_action_accuracy=t['teacher_forced']['first_action']['accuracy'],
                    value_accuracy=t['teacher_forced']['value']['accuracy'],
                    eos_accuracy=t['teacher_forced']['eos']['accuracy'],
                    teacher_exact=t['teacher_forced_exact_actions']['accuracy'],
                    # This gap uses the SAME bounded cohort, avoiding train-set size confounding.
                    bounded_teacher_exact=d['teacher_forced_exact_actions']['accuracy'],
                    decoded_exact=d['decoded']['accuracy'],
                    teacher_decode_gap=d['teacher_minus_decode_exact_accuracy'],
                    decoded_examples=d['decoded']['examples'],
                    by_kind=t['by_kind'])
            evaluation = folder/(name+'.json')
            if evaluation.is_file():
                metrics = json.loads(evaluation.read_bytes())['evaluation']['metrics']
                row['interactive'] = {k:v['all'] for k,v in metrics.items() if k in ('base','hop-3','confirmation') and v is not None}
            rows.append(row)
    comparisons = []
    final = {r['job_id']:r for r in rows if r['epoch']==4}
    for job in registered:
        baseline = next((j for j in registered if j['arm']=='baseline' and j['seed']==job['seed']),None)
        if job['arm']=='baseline' or baseline is None: continue
        a,b = final.get(baseline['job_id']),final.get(job['job_id'])
        if a and b:
            comparisons.append(dict(baseline=a['job_id'], intervention=b['job_id'],
                initialization_seed=job['seed'],
                train_decoded_delta=b['train']['decoded_exact']-a['train']['decoded_exact'],
                heldout_decoded_delta=b['heldout']['decoded_exact']-a['heldout']['decoded_exact'],
                attribution='Paired observation; no automatic causal or capability promotion.'))
    result = dict(schema='elendril-v717-diagnosis-v1',cause='CAUSE UNRESOLVED',
        observations=rows,paired_comparisons=comparisons,missing=missing,
        limitations=['Full teacher-forced and bounded decoded cohorts have different denominators.',
                    'Decoded actions use teacher observation states; raw interactive rollouts are separate.',
                    'Train/validation share procedural generator authorship.',
                    'Initial gradient parity verifies mechanics, not positive skill acquisition.',
                    'Fixed four epochs do not test whether substantially longer training solves the task.'])
    text = ['# S20 failure investigation', '', '**CAUSE UNRESOLVED.** These are measurements, not a root-cause verdict.', '',
        '| Run | Epoch | Train CE | Train decoded | Held-out decoded | Held-out teacher/decode gap |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f'| {r["job_id"]} | {r["epoch"]} | {r["train"]["token_ce"]:.4f} | {r["train"]["decoded_exact"]:.3f} | {r["heldout"]["decoded_exact"]:.3f} | {r["heldout"]["teacher_decode_gap"]:.3f} |')
    text += ['', 'Interpretation: weak training-set acquisition keeps optimization, exposure and representation hypotheses open. '
             'Training success with held-out failure localizes a generalization gap. Teacher-forced success with decoded failure '
             'localizes an inference gap. Decoded teacher-state success with raw interactive failure localizes a rollout/policy gap. '
             'None alone proves a mechanism; inspect per-kind/token confusions and raw episode traces before deciding the next intervention.',
             '', f'Missing diagnostic milestones: {len(missing)}. See JSON for paired changes and exact denominators.', '']
    _write_derived(root/'analysis'/'S20_DIAGNOSIS.json',(json.dumps(result,indent=2,sort_keys=True)+'\n').encode())
    _write_derived(root/'analysis'/'S20_DIAGNOSIS.md','\n'.join(text).encode())
    return {'status':'AVAILABLE' if rows else 'NO_MEASUREMENTS','observations':len(rows),'missing':len(missing),'cause':'UNRESOLVED'}
