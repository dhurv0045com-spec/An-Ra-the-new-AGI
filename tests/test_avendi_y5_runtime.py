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

