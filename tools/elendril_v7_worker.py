"""Isolated owner-operated CUDA worker; no CPU research fallback."""
import argparse
from dataclasses import asdict, replace
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from uuid import uuid4

from anra_v5.elendril.v7_protocol import PROTOCOL_SHA, protocol, milestones, validate_job
from tools.elendril_v6_export import immutable_json
from tools.elendril_v6_worker import copy_checkpoint, _weights, append_json
from tools.elendril_cloud import check_source

REPO = Path(__file__).resolve().parents[1]


def create(data, job, *, source, root, resume=False, updates=None):
    from anra_v5.elendril.v7_training import V7Trainer
    from anra_v5.elendril.v6_registry import MODEL_PROFILES
    from anra_v5.elendril.foundation_training import Recipe
    from anra_v5.elendril.representation import UTF8
    from anra_v5.elendril.training import configure_device
    configure_device('cuda:0')
    batch = min(16, len(data.samples))
    recipe = Recipe(seed=job['seed'], total_updates=updates or math.ceil(len(data.samples)/batch)*4,
                    batch_size=batch, micro_batch_size=min(2,batch), lr=.0003,
                    objective='ce', sampling='uniform', schedule='cosine', precision='fp32')
    return V7Trainer(data.samples, spec=MODEL_PROFILES['S3'], recipe=recipe,
        method=job['method'], source_commit=source, root=Path(root), lineage=job['job_id'],
        data_provenance={**data.manifest, 'v7_protocol_sha256': PROTOCOL_SHA, 'v7_job': job},
        tokenizer_manifest=UTF8.manifest.record(), device='cuda:0', resume=resume)


def _bound(trainer):
    from anra_v5.elendril.v6_evaluation import NeuralPolicy
    trainer.backend.binding = replace(trainer.backend.binding,
        checkpoint_sha256=trainer.parent or _weights(trainer.model))
    return NeuralPolicy(trainer.backend)


def _evaluate(trainer, data, path):
    from anra_v5.elendril.v7_diagnostics import diagnose, verify_diagnosis
    before = _weights(trainer.model)
    identity = trainer.parent or before
    if path.exists():
        result = json.loads(path.read_bytes())
        if result['checkpoint'] != identity or result['data_identity'] != data.identity:
            raise ValueError('cached checkpoint diagnostic binding mismatch')
        if result['update'] != trainer.state.global_update:
            raise ValueError('cached diagnostic update mismatch')
        _bound(trainer)
        verify_diagnosis(data, result['diagnosis'], backend=trainer.backend)
    else:
        started = time.monotonic()
        diagnosis = diagnose(data, _bound(trainer), backend=trainer.backend)
        verify_diagnosis(data, diagnosis, backend=trainer.backend)
        result = {'checkpoint': identity, 'data_identity': data.identity,
                  'update': trainer.state.global_update, 'diagnosis': diagnosis,
                  'state_kind':'committed_checkpoint' if trainer.parent else 'untrained_weight_fingerprint',
                  'weight_sha256': before, 'wall_seconds': time.monotonic()-started}
        immutable_json(path, result)
    if any(r['raw']['model_identity'] != identity
           for r in result['diagnosis']['records'].values() if r is not None):
        raise ValueError('inner diagnostic checkpoint identity mismatch')
    if before != _weights(trainer.model) or before != result['weight_sha256']:
        raise ValueError('diagnostic mutated weights or cached weight identity differs')
    return result


def run(mode, req):
    import torch
    from anra_v5.elendril.v6_data import read_dataset
    from anra_v5.elendril.v7_measurement import evaluate, verify_record
    from anra_v5.elendril.training import runtime_identity
    from tools.elendril_action_cloud import session_identity, compare_payloads
    from v5_training.checkpoint import CheckpointStore
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('one isolated CUDA GPU required; no CPU fallback')
    started = time.monotonic(); cpu = time.process_time()
    deadline = started + req['timecap_seconds']
    output = Path(req['output']); output.mkdir(parents=True, exist_ok=True)
    cp_root = Path(req['checkpoint_root']); cp_root.mkdir(parents=True, exist_ok=True)
    job = req['job']
    data = read_dataset(req['data_file'])
    report = {'mode': mode, 'source': req['source'], 'protocol_sha256': PROTOCOL_SHA,
              'job': job, 'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'],
              'session': session_identity(), 'runtime': runtime_identity(torch.device('cuda:0'))}
    if mode == 'recover':
        challenge = json.loads(Path(req['challenge']).read_bytes())
        if challenge['source'] != req['source'] or challenge['protocol_sha256'] != PROTOCOL_SHA:
            raise ValueError('recovery challenge identity mismatch')
        if challenge['gpu_uuid'] != os.environ['CUDA_VISIBLE_DEVICES']:
            raise ValueError('same-device recovery challenge ran on different GPU')
        copy_checkpoint(challenge['head2_path'], cp_root)
        trainer = create(data, job, source=req['source'], root=cp_root, resume=True, updates=4)
        if trainer.state.global_update != 2 or trainer.parent != challenge['head2']:
            raise ValueError('recovery cursor/head mismatch')
        while not trainer.state.complete: trainer.step()
        trainer.save()
        expected = CheckpointStore(challenge['expected_path'], job['job_id'])
        _, left = expected.restore(challenge['head4'])
        _, right = trainer.store.restore(trainer.parent)
        compare_payloads(left, right, exact=True)
        report.update(status='PASS', comparison='all model/optimizer/RNG/cursor/scheduler exact',
                      checkpoint=trainer.parent, update=trainer.state.global_update,
                      qualification='fresh process, same session and same physical GPU')
    elif mode == 'canary':
        positive = create(data, job, source=req['source'], root=cp_root/'positive', updates=64)
        history = []
        for _ in range(64):
            if time.monotonic() > deadline: raise TimeoutError('positive control deadline')
            history.append(positive.step())
        positive.save()
        record = evaluate(data, _bound(positive)); verify_record(data, record)
        m = record['metrics']['all']
        passed = (m['correct'] == m['episodes'] and m['invalid'] == 0 and m['query_needed'] > 0
                  and m['useful_first_query'] == m['query_needed'] == m['first_query']
                  and history[-1]['loss'] < .2*history[0]['loss'])
        heldout = read_dataset(req['eval_file'])
        generalization = evaluate(heldout, _bound(positive)); verify_record(heldout, generalization)
        report.update(positive_control={'passed': passed, 'raw': record, 'updates': history,
            'distinct_validation': generalization, 'checkpoint': positive.parent})
        immutable_json(output/'positive.json', report['positive_control'])
        if not passed:
            report.update(status='FAIL', reason='positive learning control failed; research prohibited')
        else:
            del positive; gc.collect(); torch.cuda.empty_cache()
            trainer = create(data, job, source=req['source'], root=cp_root/'expected', updates=4)
            times = []
            for u in range(1,5):
                if time.monotonic() > deadline: raise TimeoutError('recovery canary deadline')
                t = time.monotonic(); trainer.step(); torch.cuda.synchronize(); times.append(time.monotonic()-t)
                if u in (2,4):
                    trainer.save()
                    if u == 2:
                        copy_checkpoint(cp_root/'expected', cp_root/'head2'); head2 = trainer.parent
            challenge = {'source': req['source'], 'protocol_sha256': PROTOCOL_SHA,
                'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'head2': head2, 'head4': trainer.parent,
                'head2_path': str(cp_root/'head2'), 'expected_path': str(cp_root/'expected')}
            immutable_json(output/'challenge.json', challenge)
            del trainer; gc.collect(); torch.cuda.empty_cache()
            recovery = {**req, 'output': str(output/'recovery'), 'checkpoint_root': str(cp_root/'recovery'),
                        'challenge': str(output/'challenge.json'), 'timecap_seconds': max(1,deadline-time.monotonic())}
            request = output/'recover-request.json'; receipt = output/'recovery'/'receipt.json'
            immutable_json(request,recovery)
            result = subprocess.run([sys.executable,'-m','tools.elendril_v7_worker','recover',
                '--owner-enabled','--request',str(request),'--receipt',str(receipt)], cwd=REPO,
                timeout=max(1, deadline-time.monotonic()))
            recovered = json.loads(receipt.read_bytes())
            if result.returncode or recovered.get('status') != 'PASS':
                raise ValueError('fresh-process exact recovery failed')
            # A production-shaped, eight-update timing probe admits capacity;
            # its weights are discarded and are never reported as study results.
            production = read_dataset(req['production_file'])
            probe = create(production, job, source=req['source'], root=cp_root/'timing', updates=8)
            timings = []
            for _ in range(8):
                if time.monotonic() > deadline: raise TimeoutError('production timing deadline')
                t = time.monotonic(); probe.step(); torch.cuda.synchronize(); timings.append(time.monotonic()-t)
            t = time.monotonic()
            timing_eval = evaluate(heldout, _bound(probe)); verify_record(heldout,timing_eval)
            eval_seconds = time.monotonic()-t
            report.update(status='PASS', recovery=recovered, canary_seconds_per_update=sum(times)/4,
                production_seconds_per_update=sum(timings)/len(timings),
                evaluation_seconds_per_episode=eval_seconds/len(heldout.cases),
                timing_updates=8, timing_weights_used_for_science=False)
    else:
        validate_job(job)
        resume = (cp_root/job['job_id']/'LATEST').exists()
        trainer = create(data, job, source=req['source'], root=cp_root, resume=resume)
        report.update(start_update=trainer.state.global_update, metadata=trainer.metadata,
            parameter_count=sum(p.numel() for p in trainer.model.parameters()),
            checkpoint_root=str(cp_root), data_identity=data.identity)
        monitor = read_dataset(req['monitor_file']); validation = read_dataset(req['eval_file'])
        train_monitor = read_dataset(req['train_monitor_file'])
        confirmation = read_dataset(req['confirmation_file'])
        marks = milestones(len(data.samples)); per_epoch = math.ceil(len(data.samples)/16)
        log = output/('updates-'+uuid4().hex+'.jsonl')
        report['updates_log'] = str(log)

        def inspect_checkpoint():
            u = trainer.state.global_update
            if u not in marks: return
            trainer.store.record_milestone(threshold_tokens=trainer.state.cumulative_tokens,
                                          checkpoint_sha256=trainer.parent)
            _evaluate(trainer, monitor, output/f'diagnostic-u{u}.json')
            if u in (per_epoch,2*per_epoch,4*per_epoch):
                _evaluate(trainer,validation,output/f'development-u{u}.json')
                _evaluate(trainer,train_monitor,output/f'training-u{u}.json')

        if trainer.parent: inspect_checkpoint()  # finish interrupted diagnostic before any next update
        else: _evaluate(trainer,monitor,output/'diagnostic-u0.json')
        while not trainer.state.complete and time.monotonic() < deadline-90:
            t = time.monotonic(); row = trainer.step(); row['wall_seconds'] = time.monotonic()-t
            chosen=[trainer.samples[i] for i in trainer.plan[row['update']-1]]
            chunks=[chosen[i:i+trainer.recipe.micro_batch_size]
                    for i in range(0,len(chosen),trainer.recipe.micro_batch_size)]
            row['padded_input_positions']=sum(len(chunk)*max(len(s.tokens) for s in chunk) for chunk in chunks)
            row['padded_attention_pairs']=sum(len(chunk)*max(len(s.tokens) for s in chunk)**2 for chunk in chunks)
            append_json(log,row)
            u = trainer.state.global_update
            if u % 64 == 0 or u in marks or trainer.state.complete:
                trainer.save(); inspect_checkpoint(); trainer.store.prune(keep={trainer.parent})
        if trainer.state.global_update > trainer.published_update: trainer.save()
        report.update(status='COMPLETE' if trainer.state.complete else 'PARTIAL',
            checkpoint=trainer.parent, global_update=trainer.state.global_update,
            cumulative_supervised_tokens=trainer.state.cumulative_tokens,
            cumulative_input_positions=sum(sum(len(trainer.samples[i].tokens) for i in batch)
                                           for batch in trainer.plan[:trainer.state.global_update]),
            cursor=asdict(trainer.state.cursor), plan_sha256=trainer.metadata['plan_sha256'])
        if trainer.state.complete:
            # All registered arms receive the same untouched confirmation; no winner selection.
            _evaluate(trainer, confirmation, output/'confirmation.json')
    torch.cuda.synchronize()
    report.update(wall_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu,
        gpu_wall_reservation_seconds=time.monotonic()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('canary','recover','train'))
    p.add_argument('--owner-enabled', action='store_true')
    p.add_argument('--request', type=Path, required=True); p.add_argument('--receipt', type=Path, required=True)
    args = p.parse_args(argv)
    if not args.owner_enabled: p.error('explicit owner execution required')
    req = json.loads(args.request.read_bytes())
    cap = req['timecap_seconds']
    if type(cap) not in (int,float) or not math.isfinite(cap) or not 1 <= cap <= 21600:
        raise ValueError('invalid wall cap')
    if req['protocol_sha'] != PROTOCOL_SHA: raise ValueError('protocol mismatch')
    check_source(REPO, req['source'])
    output = Path(req['output']).resolve(); receipt = args.receipt.resolve()
    cp = Path(req['checkpoint_root']).resolve()
    if (output.is_relative_to(REPO) or cp.is_relative_to(REPO)
            or not receipt.is_relative_to(output) or receipt.exists()):
        raise ValueError('new receipt and output/checkpoints outside source required')
    if args.mode == 'train': validate_job(req['job'])
    try:
        report = run(args.mode,req)
    except Exception as exc:
        report = {'status':'ERROR', 'source':req['source'], 'protocol_sha256':PROTOCOL_SHA,
                  'job':req.get('job'), 'checkpoint_root':str(cp), 'request_path':str(args.request),
                  'recovery_policy':'Resume last committed generation; in-flight updates are not committed.',
                  'error_type':type(exc).__name__, 'message':str(exc), 'traceback':traceback.format_exc()}
    immutable_json(receipt,report)
    return 0 if report['status'] in ('PASS','COMPLETE','PARTIAL') else 1


if __name__ == '__main__': raise SystemExit(main())
