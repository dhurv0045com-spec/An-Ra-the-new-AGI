import argparse
import json

import pytest
import torch

from experiments.avendi_y5 import campaign as c
from experiments.avendi_y5.runtime import Context, publish, read

SOURCE = '1' * 40
JOB = dict(id='tiny',stage='AB',arm='test',seed=7,depends=[],updates=4)


def context(root, split='dev'):
    return Context(root, JOB, split, 'cpu', True, None, [], SOURCE)


def exercise(ctx, exposed):
    torch.manual_seed(7)
    m = torch.nn.Linear(1,1)
    def loss(model, batch):
        exposed.extend(batch)
        return model(torch.tensor(batch,dtype=torch.float32).reshape(-1,1)).square().mean()
    result = ctx.train(m,[1,2,3],[[1,2],[2,0],[0,1],[2,2]],loss,.001,
                       on_step=lambda u,m: dict(update=u))
    return m,result


def test_actual_exposure_and_confirm_restore(tmp_path):
    ctx = context(tmp_path)
    samples = []
    model,info = exercise(ctx,samples)
    assert samples == [2,3,3,1,1,2,3,3]
    assert [x['update'] for x in info['milestones']] == [0,1,2,4]
    publish(tmp_path/'dev/tiny.json',dict(weights=ctx.trained,source=SOURCE,result={}))
    seen = []
    replay,_ = exercise(context(tmp_path,'confirm'),seen)
    assert not seen and torch.equal(model.weight,replay.weight)
    obj = tmp_path/'weights'/'tiny'/'objects'/info['checkpoint_sha256']/'model.bin'
    obj.write_bytes(obj.read_bytes()[:-1]+b'!')
    with pytest.raises(ValueError):
        exercise(context(tmp_path,'confirm'),[])


def test_interrupted_training_and_history(tmp_path,monkeypatch):
    expected,_ = exercise(context(tmp_path/'reference'),[])
    monkeypatch.setenv('AVENDI_Y5_STOP_EPOCH','0')
    with pytest.raises(TimeoutError):
        exercise(context(tmp_path/'resume'),[])
    monkeypatch.delenv('AVENDI_Y5_STOP_EPOCH')
    actual,result = exercise(context(tmp_path/'resume'),[])
    assert torch.equal(expected.weight,actual.weight)
    assert [x['update'] for x in result['milestones']] == [0,1,2,4]


def test_seal_tampering_and_conflicting_selection(tmp_path):
    first = {'a':{'weights':{'checkpoint_sha256':'a'},'status':'COMPLETE'}}
    assert c.cohort(tmp_path,'AB',['a','b'],first) is None
    assert 'a' in c.cohort(tmp_path,'AB',['a'],first)
    with pytest.raises(ValueError):
        c.cohort(tmp_path,'AB',['a'],{'a':dict(first['a'],status='BLOCKED')})
    path = tmp_path/'locks/AB.json'
    doc = json.loads(path.read_text())
    doc['content']['eligible'] = {}
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        read(path)


def test_registry_and_preflight():
    jobs = c.registry()
    assert set(j['stage'] for j in jobs.values()) == {'AB','C','D','E','F','G'}
    assert all(j['id'] not in j['depends'] for j in jobs.values())
    assert len(c.checks()) == 3


def test_global_confirmation_embargo_blocks_premature_confirm(tmp_path, monkeypatch):
    fake_jobs = {
        'ab_job': {'id': 'ab_job', 'stage': 'AB', 'arm': 'a', 'seed': 7, 'depends': [], 'updates': 2, 'module': 'reader'},
        'f_job': {'id': 'f_job', 'stage': 'F', 'arm': 'f', 'seed': 7, 'depends': ['ab_job'], 'updates': 2, 'module': 'composition'},
    }
    monkeypatch.setattr(c, 'registry', lambda: dict(fake_jobs))
    monkeypatch.setattr(c, 'checks', lambda: {'ok': True})
    monkeypatch.setattr(c, 'benchmark_hardware', lambda canary=False: {'canary': canary, 'max_working_bytes': c.MAX_WORKING_BYTES, 'models': {}})
    proto = c.identity()[1]
    monkeypatch.setattr(c, 'identity', lambda: ('0' * 40, proto))
    dispatched = []

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            pass

        def admit_slice(self, specs, **kwargs):
            return {'admitted_job_ids': [s.job_id for s in specs], 'deferred_job_ids': []}

        def run_waves(self, specs, command_builder=None, devices=None):
            for idx, spec in enumerate(specs):
                cmd = command_builder(spec, (devices or ['cpu'])[idx % max(1, len(devices or ['cpu']))])
                split = cmd[cmd.index('--split') + 1]
                dispatched.append((split, spec.job_id))
                if split == 'dev' and spec.job_id == 'f_job':
                    # Defer/crash f_job during dev: global embargo MUST prevent ANY confirm dispatch.
                    continue
                ctx_w = Context(tmp_path / 'work', dict(fake_jobs[spec.job_id], updates=4), split, 'cpu', True, None, [], '0' * 40)
                if split == 'dev':
                    exercise(ctx_w, [])
                weights_info = ctx_w.trained if split == 'dev' else read(tmp_path / 'work' / 'dev' / f'{spec.job_id}.json')['weights']
                payload = {
                    'job': fake_jobs[spec.job_id],
                    'split': split,
                    'source': '0' * 40,
                    'protocol': proto,
                    'canary': True,
                    'status': 'COMPLETE',
                    'seconds': 0.1,
                    'weights': weights_info,
                    'result': {'metric': 1.0},
                }
                publish(tmp_path / 'work' / split / f'{spec.job_id}.json', payload)

    monkeypatch.setattr(c, 'CampaignOrchestrator', FakeOrchestrator)
    args = argparse.Namespace(
        work_root=tmp_path / 'work',
        output=tmp_path / 'out',
        minutes=30,
        canary=True,
        full=None,
        resume=None,
        inputs=[],
    )
    c.execute(args, fake_jobs, '0' * 40, proto)
    confirm_dispatches = [d for d in dispatched if d[0] == 'confirm']
    assert not confirm_dispatches, f'Premature confirm opened before all dev resolved: {confirm_dispatches}'
    assert not (tmp_path / 'work' / 'CONFIRM_LOCK.json').exists()
    # Second session completes remaining dev job, seals CONFIRM_LOCK.json globally, then opens confirm ONCE.
    def _complete_all(specs, command_builder=None, devices=None):
        for idx, spec in enumerate(specs):
            cmd = command_builder(spec, (devices or ['cpu'])[idx % max(1, len(devices or ['cpu']))])
            split = cmd[cmd.index('--split') + 1]
            ctx_w = Context(tmp_path / 'work', dict(fake_jobs[spec.job_id], updates=4), split, 'cpu', True, None, [], '0' * 40)
            if split == 'dev':
                exercise(ctx_w, [])
            w_info = ctx_w.trained if split == 'dev' else read(tmp_path / 'work' / 'dev' / f'{spec.job_id}.json')['weights']
            publish(tmp_path / 'work' / split / f'{spec.job_id}.json', {
                'job': fake_jobs[spec.job_id], 'split': split, 'source': '0' * 40,
                'protocol': proto, 'canary': True, 'status': 'COMPLETE', 'seconds': 0.1,
                'weights': w_info, 'result': {'metric': 1.0},
            })
    FakeOrchestrator.run_waves = lambda self, specs, command_builder=None, devices=None: _complete_all(specs, command_builder, devices)
    c.execute(args, fake_jobs, '0' * 40, proto)
    assert (tmp_path / 'work' / 'CONFIRM_LOCK.json').exists()
    lock = read(tmp_path / 'work' / 'CONFIRM_LOCK.json')
    assert sorted(lock['eligible']) == ['ab_job', 'f_job']
    # Confirm worker fails closed if dev selection is mutated after global seal.
    (tmp_path / 'work' / 'dev' / 'ab_job.json').write_text('{"tampered": true}')
    with pytest.raises(ValueError):
        c.verify_global_confirmation_lock(tmp_path / 'work', 'ab_job', fake_jobs, '0' * 40, proto)


def test_mocked_2gpu_66job_dispatch_nonstarving_and_leaf_compaction(tmp_path, monkeypatch):
    jobs = {k: {**v, 'updates': 4 if v['updates'] else 0} for k, v in c.registry().items()}
    assert len(jobs) == 66
    manifest = c.registry_manifest(jobs)
    assert set(manifest) == set(jobs)
    monkeypatch.setattr(c, 'registry', lambda: jobs)
    monkeypatch.setattr(c, 'checks', lambda: {'ok': True})
    monkeypatch.setattr(c, 'benchmark_hardware', lambda canary=False: {'canary': canary, 'max_working_bytes': c.MAX_WORKING_BYTES, 'models': {}})
    proto = c.identity()[1]
    monkeypatch.setattr(c, 'identity', lambda: ('0' * 40, proto))
    monkeypatch.setattr(c.torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(c.subprocess, 'check_output', lambda *a, **k: '')
    seen_devices = set()
    class Fake2GPUOrchestrator:
        def __init__(self, **kwargs):
            pass
        def admit_slice(self, specs, **kwargs):
            # Reject AB on wave 0 when mixed with smaller independent jobs -> smaller jobs still run (non-starving)!
            if len(specs) > 1 and any(s.job_id.startswith('AB-') for s in specs) and any(s.job_id.startswith('D-') for s in specs):
                return {'admitted_job_ids': [s.job_id for s in specs if not s.job_id.startswith('AB-')], 'deferred_job_ids': [s.job_id for s in specs if s.job_id.startswith('AB-')]}
            return {'admitted_job_ids': [s.job_id for s in specs], 'deferred_job_ids': []}
        def run_waves(self, specs, command_builder=None, devices=None):
            assert list(devices or ()) == ['cuda:0', 'cuda:1'] or not specs
            for idx, spec in enumerate(specs):
                cmd = command_builder(spec, tmp_path / 'worker' / spec.job_id)
                seen_devices.add(cmd[cmd.index('--device') + 1])
                split = cmd[cmd.index('--split') + 1]
                job = jobs[spec.job_id]
                if split == 'dev' and job['stage'] == 'G':
                    publish(tmp_path / 'work' / split / f'{spec.job_id}.json', {
                        'job': job, 'split': split, 'source': '0' * 40, 'protocol': proto, 'canary': False,
                        'status': 'BLOCKED', 'seconds': 0.05, 'weights': None, 'result': {'status': 'BLOCKED', 'verification': 'MISSING_OPTIONAL_Y1'},
                    })
                    continue
                if split == 'dev' and job['updates'] > 0:
                    ctx_w = Context(tmp_path / 'work', job, 'dev', 'cpu', True, None, [], '0' * 40)
                    exercise(ctx_w, [])
                    w_info = ctx_w.trained
                elif split == 'confirm':
                    w_info = read(tmp_path / 'work' / 'dev' / f'{spec.job_id}.json')['weights']
                else:
                    w_info = None
                res_obj = {'status': 'COMPLETED', 'panels': {'5': {'total': 1, 'correct': 1, 'cases': [{'case_id': 'c0', 'gold': 'A0', 'prediction': 'A0', 'correct': True}]}}, 'interventions': {}} if job['stage'] == 'C' else {'status': 'COMPLETE'}
                publish(tmp_path / 'work' / split / f'{spec.job_id}.json', {
                    'job': job, 'split': split, 'source': '0' * 40, 'protocol': proto, 'canary': False,
                    'status': 'COMPLETE', 'seconds': 0.1, 'weights': w_info, 'result': res_obj,
                })
    monkeypatch.setattr(c, 'CampaignOrchestrator', Fake2GPUOrchestrator)
    args = argparse.Namespace(work_root=tmp_path / 'work', output=tmp_path / 'out', minutes=480, canary=False, full=None, resume=None, inputs=[])
    rc = c.execute(args, jobs, '0' * 40, proto)
    assert rc == 0
    assert seen_devices == {'cuda:0', 'cuda:1'}
    lock = read(tmp_path / 'work' / 'CONFIRM_LOCK.json')
    g_id = next(k for k in jobs if k.startswith('G-'))
    assert g_id in lock['diagnostic_dispositions']
    assert lock['diagnostic_dispositions'][g_id]['status'] == 'BLOCKED'
    assert 'AB-reader-fixed-ce-s7' in lock['eligible']
    d_latest = list((tmp_path / 'work' / 'weights' / 'D-informative-s7' / 'objects').iterdir())[0]
    assert (d_latest / 'model.bin').is_file()
    assert not (d_latest / 'optimizer.bin').exists()
    ab_latest = list((tmp_path / 'work' / 'weights' / 'AB-answer-fixed-ce-s7' / 'objects').iterdir())[0]
    assert (ab_latest / 'optimizer.bin').is_file()
    from experiments.avendi_y5.analysis import rescore_evidence
    ev_zip = tmp_path / 'out' / 'AVENDI_Y5_EVIDENCE.zip'
    assert ev_zip.is_file()
    rescored = rescore_evidence(ev_zip)
    assert rescored['verified'] and rescored['rescored_count'] > 0


def test_post_lock_crash_resume_scores_only_missing_and_rejects_swapped_sha(tmp_path, monkeypatch):
    fake_jobs = {
        'j1': {'id': 'j1', 'stage': 'D', 'arm': 'informative', 'seed': 7, 'depends': [], 'updates': 4, 'role': 'primary'},
        'j2': {'id': 'j2', 'stage': 'E', 'arm': 'relevant', 'seed': 7, 'depends': [], 'updates': 4, 'role': 'primary'},
    }
    monkeypatch.setattr(c, 'registry', lambda: dict(fake_jobs))
    monkeypatch.setattr(c, 'checks', lambda: {'ok': True})
    monkeypatch.setattr(c, 'benchmark_hardware', lambda canary=False: {'canary': canary, 'max_working_bytes': c.MAX_WORKING_BYTES, 'models': {}})
    proto = c.identity()[1]
    monkeypatch.setattr(c, 'identity', lambda: ('0' * 40, proto))
    confirm_calls = []
    fail_confirm_j2 = [True]
    class CrashAfterLockOrchestrator:
        def __init__(self, **kwargs):
            pass
        def admit_slice(self, specs, **kwargs):
            return {'admitted_job_ids': [s.job_id for s in specs], 'deferred_job_ids': []}
        def run_waves(self, specs, command_builder=None, devices=None):
            for idx, spec in enumerate(specs):
                cmd = command_builder(spec, 'cpu')
                split = cmd[cmd.index('--split') + 1]
                if split == 'confirm':
                    confirm_calls.append(spec.job_id)
                    if spec.job_id == 'j2' and fail_confirm_j2[0]:
                        continue
                ctx_w = Context(tmp_path / 'work', fake_jobs[spec.job_id], split, 'cpu', True, None, [], '0' * 40)
                if split == 'dev':
                    exercise(ctx_w, [])
                w_info = ctx_w.trained if split == 'dev' else read(tmp_path / 'work' / 'dev' / f'{spec.job_id}.json')['weights']
                publish(tmp_path / 'work' / split / f'{spec.job_id}.json', {
                    'job': fake_jobs[spec.job_id], 'split': split, 'source': '0' * 40, 'protocol': proto,
                    'canary': True, 'status': 'COMPLETE', 'seconds': 0.1, 'weights': w_info, 'result': {'metric': 1.0},
                })
    monkeypatch.setattr(c, 'CampaignOrchestrator', CrashAfterLockOrchestrator)
    args = argparse.Namespace(work_root=tmp_path / 'work', output=tmp_path / 'out', minutes=30, canary=True, full=None, resume=None, inputs=[])
    assert c.execute(args, fake_jobs, '0' * 40, proto) == 1
    assert confirm_calls == ['j1', 'j2']
    # Resume after lock scores ONLY missing j2, never re-scoring j1!
    confirm_calls.clear()
    fail_confirm_j2[0] = False
    assert c.execute(args, fake_jobs, '0' * 40, proto) == 0
    assert confirm_calls == ['j2']
    # Swapped checkpoint SHA on dev receipt must fail closed.
    p1 = tmp_path / 'work' / 'dev' / 'j1.json'
    d1 = read(p1)
    d1['weights'] = {'checkpoint_sha256': 'f' * 64, 'updates': 4}
    p1.unlink()
    publish(p1, d1)
    with pytest.raises(ValueError, match='checkpoint'):
        c.execute(args, fake_jobs, '0' * 40, proto)



