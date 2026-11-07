"""Owner-enabled isolated CUDA V6 campaign worker."""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]


def _inside(path, root):
    return path == root or root in path.parents


def admit(request, receipt, owner):
    if not owner:
        raise ValueError('explicit --owner-enabled required')
    request, receipt = Path(request).resolve(strict=True), Path(receipt).resolve()
    data = json.loads(request.read_text(encoding='utf-8'))
    cap = data.get('timecap_seconds', 21600)
    if type(cap) not in (int, float) or not math.isfinite(cap) or not 1 <= cap <= 21600:
        raise ValueError('worker wall cap must be finite and within1..21600 seconds')
    reserve=data.get('evaluation_reserve_seconds',300)
    if type(reserve) not in (int,float) or not math.isfinite(reserve) or not 0 < reserve <= 600:
        raise ValueError('invalid evaluation/export reserve')
    output = Path(data['output']).resolve()
    if _inside(output, REPO) or not _inside(receipt, output) or receipt.exists():
        raise ValueError('output must be outside repository and receipt must be new within output')
    for key in ('data_file', 'eval_file', 'canary_file', 'challenge', 'parent_result'):
        if key in data:
            path = Path(data[key]).resolve(strict=True)
            if _inside(path, REPO):
                raise ValueError(f'{key} must be outside source repository')
    if 'job' in data:
        from anra_v5.elendril.v6_training import Job
        Job(**data['job']).validate()
    from anra_v5.elendril.v6_registry import REGISTRY_SHA
    if data['protocol_sha'] != REGISTRY_SHA:
        raise ValueError('invalid protocol hash')
    from tools.elendril_cloud import check_source
    check_source(REPO, data['source'])
    import re
    if not re.fullmatch(r'GPU-[a-fA-F0-9-]+', os.environ.get('CUDA_VISIBLE_DEVICES', '')):
        raise ValueError('exactly one physical GPU UUID required')
    return data, receipt


def write_json(path, value, *, exclusive=False):
    if exclusive:
        from tools.elendril_v6_export import immutable_json
        immutable_json(Path(path), value)
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, ensure_ascii=False, default=str)
        f.flush(); os.fsync(f.fileno())


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, sort_keys=True, default=str) + '\n')
        f.flush(); os.fsync(f.fileno())


def copy_checkpoint(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or _inside(destination, source):
        raise ValueError('checkpoint copy destination must be new')
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('linked checkpoint payloads are not accepted')
    shutil.copytree(source, destination)
    def inventory(root):
        from hashlib import file_digest
        result = {}
        for p in root.rglob('*'):
            if p.is_file():
                with p.open('rb') as handle:
                    result[str(p.relative_to(root))] = file_digest(handle, 'sha256').hexdigest()
        return result
    if inventory(source) != inventory(destination):
        raise ValueError('checkpoint copy verification failed')


def _weights(model):
    from anra_v5.elendril.backend import update_tensor_fingerprint
    h = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); update_tensor_fingerprint(h, tensor)
    return h.hexdigest()


def run_supplemental(data, policy, base, controls, output, *, deadline, seconds_per_episode=None):
    """Bounded frozen-checkpoint probes; commit each condition before proceeding."""
    from anra_v5.elendril.v6_supplement import SPECS, condition_specs, summarize, verify_experiment
    from anra_v5.elendril.v6_evaluation import evaluate, independent_rescore
    from anra_v5.elendril.codec import canonical
    started = time.monotonic()
    expires = min(deadline - 15, started + 90)
    if seconds_per_episode is not None and (type(seconds_per_episode) not in (float,int)
            or not math.isfinite(seconds_per_episode) or seconds_per_episode <= 0):
        raise ValueError('invalid supplemental timing forecast')
    reports = {}
    reused = []
    # S19 only rescales existing behavior; it spends no additional model forwards.
    reports['S19'] = summarize('S19', {'model': base}, controls)
    verify_experiment(data, 'S19', reports['S19'], controls)
    write_path = Path(output) / 'supplement-S19.json'
    if write_path.exists():
        if canonical(json.loads(write_path.read_bytes())) != canonical(reports['S19']):
            raise ValueError('cached cost experiment differs from frozen parent')
    else:
        write_json(write_path, reports['S19'], exclusive=True)
    for experiment in SPECS:
        experiment_id = experiment['id']
        if experiment_id == 'S19':
            continue
        specs = condition_specs(data, experiment_id)
        records = {}
        for name, spec in specs.items():
            cache = Path(output) / f'supplement-{experiment_id}-{name}.json'
            if cache.exists():
                record = json.loads(cache.read_bytes())
                if (record['model_identity'] != base['model_identity']
                        or record['budget'] != asdict(spec['budget'])
                        or record['memory_mode'] != spec['memory_mode']
                        or record['split'] != base['split']):
                    raise ValueError('cached supplemental condition binding differs')
                independent_rescore(spec['cases'], record)
                reused.append(f'{experiment_id}/{name}')
            elif time.monotonic() + (2 * seconds_per_episode * len(spec['cases']) + 5
                                     if seconds_per_episode is not None else 0) < expires:
                record = evaluate(spec['cases'], policy, split=base['split'],
                                  budget=spec['budget'], memory_mode=spec['memory_mode'])
                write_json(cache, record, exclusive=True)
            else:
                break
            records[name] = record
        if set(records) == set(specs):
            result = summarize(experiment_id, records, controls)
            verify_experiment(data, experiment_id, result, controls)
        else:
            result = {'schema': 'elendril-v6-supplement-partial-v1', 'experiment_id': experiment_id,
                      'status': 'PARTIAL' if records else 'SKIPPED', 'records': records,
                      'missing_conditions': sorted(set(specs) - set(records)),
                      'reason': 'supplemental time allowance or campaign deadline'}
        reports[experiment_id] = result
    return {'schema': 'elendril-v6-supplement-bundle-v1', 'experiments': reports,
            'elapsed_seconds': time.monotonic() - started, 'limit_seconds': 90,
            'reused_conditions': reused,
            'qualification': 'exploratory fixed-checkpoint probes; no extra training or promotion'}


def run(mode, req):
    import torch
    from anra_v5.elendril.v6_training import Job, create_trainer, policy_for
    from anra_v5.elendril.v6_data import read_dataset, _dataset
    from anra_v5.elendril.v6_evaluation import evaluate, diagnose, ControlPolicy, panel_report
    from anra_v5.elendril.training import runtime_identity
    from tools.elendril_action_cloud import compare_payloads, session_identity
    from v5_training.checkpoint import CheckpointStore
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('one isolated CUDA device required')
    output = Path(req['output']).resolve(); output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic(); deadline = start + float(req.get('timecap_seconds', 21600))
    session = session_identity(); gpu = os.environ['CUDA_VISIBLE_DEVICES']
    cpu_started = time.process_time()
    report = {'status': 'PARTIAL', 'mode': mode, 'source': req['source'], 'protocol_sha256': req['protocol_sha'],
              'session': session, 'gpu_uuid': gpu, 'environment': runtime_identity(torch.device('cuda:0'))}
    if mode == 'recover':
        cp = Path(req['challenge']).resolve(strict=True); challenge = json.loads(cp.read_text(encoding='utf-8'))
        if challenge['source'] != req['source'] or challenge['protocol_sha256'] != req['protocol_sha']:
            raise ValueError('challenge source/protocol mismatch')
        if req.get('fresh_session') and session == challenge['session']:
            raise ValueError('fresh VM session required')
        job = Job(**challenge['job']); data = read_dataset(cp.parent / 'data.json')
        recovery = output / 'recovery'; copy_checkpoint(cp.parent / 'head2', recovery)
        trainer = create_trainer(data, job, source=req['source'], root=recovery, resume=True, updates_override=4)
        if trainer.parent != challenge['head2'] or trainer.state.global_update != 2:
            raise ValueError('recovery head mismatch')
        while trainer.state.global_update < 4: trainer.step()
        trainer.save()
        expected = CheckpointStore(cp.parent / 'expected', job.job_id)
        _, left = expected.restore(challenge['expected']); _, right = trainer.store.restore(trainer.parent)
        compare_payloads(left, right, exact=challenge['gpu_uuid'] == gpu)
        report.update(status='PASS', checkpoint=trainer.parent, job=asdict(job), comparison='exact' if challenge['gpu_uuid'] == gpu else 'numerical', fresh_session=session != challenge['session'],
                      challenge_sha256=sha256(cp.read_bytes()).hexdigest(), original_session=challenge['session'], original_gpu_uuid=challenge['gpu_uuid'],
                      challenge_source=challenge['source'], checkpoint_root=str(recovery), global_update=trainer.state.global_update)
    elif mode == 'benchmark':
        base = Job(**req['job']); measurements = {}; challenges = []
        tiny = read_dataset(req['canary_file']); heldout = read_dataset(req['eval_file'])
        heldout = _dataset(heldout.split, heldout.cases[:14], seed=heldout.manifest['seed'], parent=heldout.identity)
        positive_job = replace(base, job_id=base.job_id + '-positive', profile='S3')
        positive = create_trainer(tiny, positive_job, source=req['source'], root=output / 'positive', updates_override=64)
        history = []
        for _ in range(64):
            if time.monotonic() >= deadline: raise TimeoutError('positive control deadline')
            history.append(positive.step())
            append_json(output/'positive-updates.jsonl',history[-1])
        positive.save(); positive.backend.binding = replace(positive.backend.binding, checkpoint_sha256=positive.parent)
        pos_eval = evaluate(tiny, policy_for(positive, positive_job))
        metrics = pos_eval['metrics']['all']
        passed = (metrics['correct'] == metrics['episodes'] and metrics['invalid'] == 0
                  and metrics['query_needed'] > 0 and metrics['useful_first_query'] == metrics['query_needed']
                  and metrics['first_query'] == metrics['query_needed']
                  and history[-1]['loss'] < .2 * history[0]['loss'])
        report['positive_control'] = {'source': req['source'], 'passed': passed, 'rawtrainval': pos_eval, 'updates': history,
                                      'distinct_validation': evaluate(heldout, policy_for(positive, positive_job)), 'checkpoint': positive.parent}
        write_json(output/'positive-control.json',report['positive_control'],exclusive=True)
        if not passed:
            report.update(status='FAIL', reason='positive control failed; no scale benchmarks admitted',
                          wall_seconds=time.monotonic()-start, process_cpu_seconds=time.process_time()-cpu_started)
            return report
        del positive
        import gc
        gc.collect(); torch.cuda.empty_cache()
        for profile in ('S3', 'S8', 'S20'):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            job = replace(base, job_id=base.job_id + '-' + profile, profile=profile)
            folder = output / profile; folder.mkdir()
            trainer = create_trainer(tiny, job, source=req['source'], root=folder / 'expected', updates_override=4)
            t = time.monotonic()
            for u in range(1, 5):
                if time.monotonic() >= deadline: raise TimeoutError('profile benchmark deadline')
                row = trainer.step(); append_json(folder/'updates.jsonl',row)
                if u in (2, 4):
                    trainer.save()
                    if u == 2: copy_checkpoint(folder / 'expected', folder / 'head2'); head2 = trainer.parent
            torch.cuda.synchronize(); seconds = (time.monotonic() - t) / 4
            trainer.backend.binding = replace(trainer.backend.binding, checkpoint_sha256=trainer.parent)
            t = time.monotonic(); evaluation = evaluate(heldout, policy_for(trainer, job)); torch.cuda.synchronize()
            per_episode = (time.monotonic()-t) / len(heldout.cases)
            shutil.copy2(req['canary_file'], folder / 'data.json')
            challenge = {'source': req['source'], 'protocol_sha256': req['protocol_sha'], 'job': asdict(job), 'head2': head2, 'expected': trainer.parent, 'gpu_uuid': gpu, 'session': session}
            challenge_path = folder / 'challenge.json'; write_json(challenge_path, challenge, exclusive=True)
            parameters = sum(p.numel() for p in trainer.model.parameters())
            peak = torch.cuda.max_memory_allocated()
            del trainer
            gc.collect(); torch.cuda.empty_cache()
            recovery_req = {'source': req['source'], 'protocol_sha': req['protocol_sha'], 'output': str(folder / 'same-gpu-recovery'), 'challenge': str(challenge_path), 'timecap_seconds': max(1, deadline-time.monotonic())}
            request_path = folder / 'same-gpu-request.json'; receipt_path = folder / 'same-gpu-recovery' / 'receipt.json'; write_json(request_path, recovery_req, exclusive=True)
            process = subprocess.run([sys.executable, '-m', 'tools.elendril_v6_worker', 'recover', '--owner-enabled', '--request', str(request_path), '--receipt', str(receipt_path)], cwd=REPO, timeout=max(1, deadline-time.monotonic()))
            if process.returncode or json.loads(receipt_path.read_text(encoding='utf-8')).get('status') != 'PASS': raise ValueError('same GPU recovery failed')
            measurements[profile] = {'seconds_per_update': seconds, 'evaluation_seconds_per_episode': per_episode, 'peak_cuda_allocated_bytes': peak, 'parameter_count': parameters, 'evaluation': evaluation}
            write_json(folder/'measurement.json',{'source':req['source'],'protocol_sha256':req['protocol_sha'],
                       'profile':profile,**measurements[profile]},exclusive=True)
            challenges.append(str(challenge_path))
        report.update(status='PASS' if passed else 'FAIL', measurements=measurements, challenges=challenges)
    else:
        job = Job(**req['job']); report['job'] = asdict(job)
        data, eval_data = read_dataset(req['data_file']), read_dataset(req['eval_file'])
        root = output / 'checkpoints'
        if mode == 'diagnose':
            parent = json.loads(Path(req['parent_result']).read_text(encoding='utf-8'))
            if parent.get('status') != 'COMPLETE' or parent['job'] != asdict(job): raise ValueError('invalid parent result')
            root = Path(parent['checkpoint_root']).resolve(strict=True)
        resume = mode == 'diagnose' or (root / job.job_id / 'LATEST').exists()
        trainer = create_trainer(data, job, source=req['source'], root=root, resume=resume)
        report.update(metadata=trainer.metadata, checkpoint_root=str(root), parameter_count=sum(p.numel() for p in trainer.model.parameters()),
                      output=str(output), training_worlds_sha256=data.manifest['worlds_sha256'], updates=[])
        if mode == 'train':
            from uuid import uuid4
            updates_path = output / ('updates-' + uuid4().hex + '.jsonl')
            report['start_update'] = trainer.state.global_update
            report['updates_log'] = str(updates_path)
            while not trainer.state.complete and time.monotonic() < deadline - req.get('evaluation_reserve_seconds',300):
                t = time.monotonic(); row = trainer.step(); row['wall_seconds'] = time.monotonic()-t; append_json(updates_path, row)
                report['updates'].append(row)
                if trainer.state.global_update % 64 == 0 or trainer.state.complete:
                    trainer.save(); trainer.store.prune(keep={trainer.parent})
            if trainer.state.global_update > trainer.published_update: trainer.save(); trainer.store.prune(keep={trainer.parent})
        if trainer.parent:
            trainer.store.record_milestone(threshold_tokens=trainer.state.cumulative_tokens,checkpoint_sha256=trainer.parent)
            trainer.backend.binding = replace(trainer.backend.binding, checkpoint_sha256=trainer.parent)
            report.update(checkpoint=trainer.parent, optimizer_updates=trainer.state.global_update, global_update=trainer.state.global_update,
                          supervised_tokens=trainer.state.cumulative_tokens, input_positions=sum(r['input_positions'] for r in report.get('updates', [])),
                          recipe=asdict(trainer.recipe), data_sha256=trainer.data_sha, cursor=asdict(trainer.state.cursor))
        if mode == 'train' and trainer.state.complete and time.monotonic() < deadline:
            from anra_v5.elendril.v6_evaluation import fit_calibration
            if eval_data.split != 'validation':
                validation_file = Path(req['eval_file']).with_name('validation.json')
                validation = read_dataset(validation_file)
                report['calibration_validation'] = evaluate(validation, policy_for(trainer, job))
                report['calibration'] = fit_calibration(report['calibration_validation'])
                calibration_path=output/'calibration-before-confirmation.json'
                if calibration_path.exists():
                    if json.loads(calibration_path.read_bytes()) != report['calibration']:
                        raise ValueError('resumed calibration differs from frozen pre-confirmation fit')
                else: write_json(calibration_path,report['calibration'],exclusive=True)
            report['evaluation'] = evaluate(eval_data, policy_for(trainer, job))
            report['controls'] = {name: evaluate(eval_data, ControlPolicy(name)) for name in ('fixed','always-query','no-query','random','recency','query-blind','oracle')}
            if eval_data.split == 'validation':
                report['calibration'] = fit_calibration(report['evaluation'])
            report['status'] = 'COMPLETE'
        elif mode == 'diagnose':
            from anra_v5.elendril.v6_evaluation import independent_rescore
            before = _weights(trainer.model); variants = {}; base = parent['evaluation']
            if base['model_identity'] != trainer.parent:
                raise ValueError('diagnostic parent checkpoint binding differs')
            independent_rescore(eval_data, base)
            completed_names=[]; reused=[]
            controls = parent.get('controls', {})
            supplemental = run_supplemental(eval_data, policy_for(trainer, job), base, controls,
                                             output, deadline=deadline,
                                             seconds_per_episode=req.get('supplement_evaluation_seconds_per_episode', 1.))
            for name in ('payload-swap','query-swap','time-order','row-order','alias','template','distractor','combined','three-hop','memory','memory-swap'):
                if time.monotonic() >= deadline - 15: break
                t = time.monotonic()
                panel_path=output/('panel-'+name+'.json')
                if panel_path.exists():
                    from anra_v5.elendril.v6_evaluation import independent_rescore,diagnostic_cases
                    part=json.loads(panel_path.read_bytes())['result']
                    if part['base']['model_identity'] != trainer.parent:raise ValueError('cached panel checkpoint differs')
                    independent_rescore(eval_data,part['base'])
                    for variant,raw in part['variants'].items():independent_rescore(diagnostic_cases(eval_data,variant),raw)
                    reused.append(name)
                else:
                    part = diagnose(eval_data, policy_for(trainer, job), variant_names=(name,), positive_control=req.get('positive_control'), controls=controls, weight_identity={'before': before, 'after': before, 'checkpoint': trainer.parent}, base_record=base,
                                    calibration=parent.get('calibration') if eval_data.split!='validation' else None)
                    write_json(panel_path, {'elapsed_seconds':time.monotonic()-t, 'result':part}, exclusive=True)
                base = part['base']; variants.update(part['variants'])
                completed_names.append(name)
            after = _weights(trainer.model)
            if before != after: raise ValueError('diagnostics altered weights')
            if base is not None:
                report['diagnostics'] = {'base': base, 'variants': variants, 'supplemental': supplemental,
                                        'panels': panel_report(base, variants, positive_control=req.get('positive_control'), controls=controls, weight_identity={'before': before, 'after': after, 'checkpoint': trainer.parent},
                                        calibration=parent.get('calibration') if eval_data.split!='validation' else None)}
            report.update(weight_sha256_before=before, weight_sha256_after=after, reused_panels=reused,
                          status='COMPLETE' if len(completed_names) == 11 and all(
                              r['status'] in ('COMPLETE','NOT_QUALIFIED') for r in supplemental['experiments'].values()) else 'PARTIAL')
    report.update(wall_seconds=time.monotonic()-start, process_cpu_seconds=time.process_time()-cpu_started,
                  gpu_wall_reservation_seconds=time.monotonic()-start,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    return report


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=('benchmark','train','recover','diagnose')); p.add_argument('--owner-enabled', action='store_true'); p.add_argument('--request', required=True); p.add_argument('--receipt', required=True)
    args = p.parse_args(argv)
    receipt = Path(args.receipt).resolve()
    if not args.owner_enabled:
        p.error('explicit --owner-enabled required; no device access or output writes')
    req = None
    try:
        req, receipt = admit(args.request, receipt, args.owner_enabled)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        result = run(args.mode, req)
    except BaseException as exc:
        result = {'status':'FAILED', 'mode':args.mode, 'error':repr(exc), 'traceback':traceback.format_exc()}
        if req is not None:
            result.update(source=req['source'], protocol_sha256=req['protocol_sha'], job=req.get('job'), output=req['output'])
    try:
        write_json(receipt, result, exclusive=True)
    except Exception:
        traceback.print_exc(); return 1
    return 0 if result['status'] in ('COMPLETE', 'PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
