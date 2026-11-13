from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from anra_v5.elendril.training import MODEL
from anra_v5.elendril.v6_data import make_dataset, write_dataset
from anra_v5.elendril.v71_protocol import PROTOCOL_SHA
from anra_v5.elendril.v71_training import create_trainer
from tools import elendril_v71_worker as worker


def spec():
    return replace(MODEL, vocabulary_size=260, width=16, layers=1, query_heads=2,
        kv_heads=1, head_dimension=8, ffn_width=32, context_length=512)


def job(name='tiny', epochs=4):
    return {'job_id':name, 'profile':'S3', 'method':'blocked', 'seed':7,
            'epochs':epochs, 'mechanism':'ce'}


def cpu_create(data, job, *, source, root, resume=False, transition=None, parent_store=None):
    return create_trainer(data, worker.trainer_job(job), source=source, root=root, resume=resume,
        transition=transition, parent_store=parent_store, device='cpu', cpu_canary=True,
        spec_override=spec(), updates_override=4)


def test_recover_fresh_process_exact_payloads(tmp_path):
    data = make_dataset(worlds=4)
    data_file = tmp_path / 'data.json'; write_dataset(data, data_file)
    expected = tmp_path / 'expected'; j = job()
    trainer = cpu_create(data, j, source='a'*40, root=expected)
    for _ in range(2): trainer.step()
    trainer.save(); head2 = trainer.parent
    for _ in range(2): trainer.step()
    trainer.save(); head4 = trainer.parent
    challenge = tmp_path / 'challenge.json'
    challenge.write_text(json.dumps({'source':'a'*40,'protocol_sha256':PROTOCOL_SHA,
        'gpu_uuid':'fake-cpu-canary','expected_root':str(expected),'head2':head2,'head4':head4}))
    request = {'source':'a'*40,'job':j,'challenge':str(challenge),'data_file':str(data_file),
               'checkpoint_root':str(tmp_path / 'recovered')}
    reqfile = tmp_path / 'request.json'; reqfile.write_text(json.dumps(request))
    script = '''
import json, os, sys
from dataclasses import replace
from anra_v5.elendril.training import MODEL
from anra_v5.elendril.v71_training import create_trainer
from tools import elendril_v71_worker as w
spec=replace(MODEL,vocabulary_size=260,width=16,layers=1,query_heads=2,kv_heads=1,head_dimension=8,ffn_width=32,context_length=512)
def create(data,job,*,source,root,resume=False,transition=None,parent_store=None):
    return create_trainer(data,w.trainer_job(job),source=source,root=root,resume=resume,transition=transition,parent_store=parent_store,device='cpu',cpu_canary=True,spec_override=spec,updates_override=4)
w.create=create
os.environ['CUDA_VISIBLE_DEVICES']='fake-cpu-canary'
result=w.recover(json.load(open(sys.argv[1])))
print(json.dumps(result))
'''
    process = subprocess.run([sys.executable, '-c', script, str(reqfile)], cwd=worker.REPO,
                             text=True, capture_output=True, timeout=60)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result['status'] == 'PASS' and result['global_update'] == 4


def test_worker_continual_transition_then_resume(tmp_path, monkeypatch):
    data = make_dataset(worlds=4)
    parent_job = job('parent')
    parent = cpu_create(data, parent_job, source='a'*40, root=tmp_path/'parent')
    parent.step(); parent.save()
    parent_model = [p.detach().clone() for p in parent.model.parameters()]
    data_file = tmp_path/'data.json'; write_dataset(data,data_file)
    suite_file = tmp_path/'suite.json'; suite_file.write_text('{}')
    output = tmp_path/'output'; output.mkdir()
    child = job('child', 2)
    req = {'job':child,'data_file':str(data_file),'checkpoint_root':str(tmp_path/'child'),
           'output':str(output),'suite_file':str(suite_file),'source':'a'*40,
           'parent':{'root':str(tmp_path/'parent'),'lineage':'parent','checkpoint':parent.parent}}
    monkeypatch.setattr(worker, 'create', cpu_create)
    monkeypatch.setattr(worker, 'suite_from_record', lambda _: {})
    seen = []
    def evaluate(t, suite, path, **kwargs):
        if path.name == 'after-transfer-before-B.json':
            seen.append(all(torch.equal(a,b) for a,b in zip(parent_model,t.model.parameters())))
        return {}
    monkeypatch.setattr(worker, 'evaluation', evaluate)
    clock = [0]
    monkeypatch.setattr(worker.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(worker, 'append_json', lambda *args: clock.__setitem__(0, 1000))
    first = worker.train(req, 200)
    assert first['status'] == 'PARTIAL' and first['global_update'] == 1 and seen == [True]
    transition = json.loads((output/'transition.json').read_bytes())
    assert transition['parent_checkpoint'] == parent.parent
    # Replay the next update independently from the saved child checkpoint.
    from anra_v5.elendril.foundation_training import StageTransition
    reference = cpu_create(data, child, source='a'*40, root=tmp_path/'child', resume=True,
                           transition=StageTransition(**transition), parent_store=parent.store)
    expected_row = reference.step()
    clock[0] = 0
    second = worker.train(req, 200)
    assert second['start_update'] == 1 and second['global_update'] == 2
    restored = cpu_create(data, child, source='a'*40, root=tmp_path/'child', resume=True,
                          transition=StageTransition(**transition), parent_store=parent.store)
    assert restored.state.cursor == reference.state.cursor
    assert restored.state.cumulative_tokens == reference.state.cumulative_tokens
    assert restored.rng_bytes() == reference.rng_bytes()
    assert all(torch.equal(a,b) for a,b in zip(restored.model.parameters(),reference.model.parameters()))
    assert restored.optimizer.state_dict()['param_groups'] == reference.optimizer.state_dict()['param_groups']
    for key, state in reference.optimizer.state_dict()['state'].items():
        for name, value in state.items():
            assert torch.equal(value, restored.optimizer.state_dict()['state'][key][name]) if isinstance(value, torch.Tensor) else value == restored.optimizer.state_dict()['state'][key][name]


def test_interrupt_after_completed_update_publishes_resumable_state(tmp_path, monkeypatch):
    data = make_dataset(worlds=4)
    data_file = tmp_path/'data.json'; write_dataset(data,data_file)
    suite_file = tmp_path/'suite.json'; suite_file.write_text('{}')
    output = tmp_path/'output'; output.mkdir()
    req = {'job':job(), 'data_file':str(data_file), 'checkpoint_root':str(tmp_path/'ckpt'),
           'output':str(output), 'suite_file':str(suite_file), 'source':'a'*40}
    monkeypatch.setattr(worker, 'create', cpu_create)
    monkeypatch.setattr(worker, 'suite_from_record', lambda _: {})
    monkeypatch.setattr(worker, 'evaluation', lambda *a, **k: {})
    def interrupt(*_): raise InterruptedError('simulated stop between updates')
    monkeypatch.setattr(worker, 'append_json', interrupt)
    result = worker.train(req, worker.time.monotonic() + 300)
    assert result['status'] == 'PARTIAL' and result['global_update'] == 1
    restored = cpu_create(data, req['job'], source=req['source'], root=tmp_path/'ckpt', resume=True)
    assert restored.state.global_update == 1
    assert restored.state.cursor.sequence_ordinal == restored.sample_cursor(1)
    assert not restored.faulted


def test_interrupt_mid_update_keeps_last_committed_checkpoint(tmp_path, monkeypatch):
    from anra_v5.elendril.backend import NeuralBackend
    data = make_dataset(worlds=4)
    data_file = tmp_path/'data.json'; write_dataset(data,data_file)
    suite_file = tmp_path/'suite.json'; suite_file.write_text('{}')
    output = tmp_path/'output'; output.mkdir()
    req = {'job':job(), 'data_file':str(data_file), 'checkpoint_root':str(tmp_path/'ckpt'),
           'output':str(output), 'suite_file':str(suite_file), 'source':'a'*40}
    trainer = cpu_create(data, req['job'], source=req['source'], root=tmp_path/'ckpt')
    trainer.step(); trainer.save(); committed = trainer.parent
    state, payloads = trainer.store.restore(committed)
    monkeypatch.setattr(worker, 'create', cpu_create)
    monkeypatch.setattr(worker, 'suite_from_record', lambda _: {})
    monkeypatch.setattr(worker, 'evaluation', lambda *a, **k: {})
    def interrupt(*args, **kwargs): raise InterruptedError('simulated interrupted forward')
    monkeypatch.setattr(NeuralBackend, 'logits', interrupt)
    result = worker.train(req, worker.time.monotonic() + 300)
    assert result['status'] == 'PARTIAL' and result['recovery_required'] is True
    assert result['checkpoint'] == committed
    restored = cpu_create(data, req['job'], source=req['source'], root=tmp_path/'ckpt', resume=True)
    after_state, after_payloads = restored.store.restore(committed)
    assert after_state == state and after_payloads == payloads
    assert restored.state.global_update == 1
