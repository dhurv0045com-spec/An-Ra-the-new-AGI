from dataclasses import replace
import json

import pytest
import torch

from anra_v5.elendril.training import MODEL
from anra_v5.elendril.v6_data import make_dataset
from anra_v5.elendril.v717_protocol import jobs, validate, protocol
from anra_v5.elendril.v717_training import create_trainer, prepare_variant
from anra_v5.elendril.v71_training import create_trainer as original


def spec():
    return replace(MODEL, vocabulary_size=260, width=16, layers=1, query_heads=2,
                   kv_heads=1, head_dimension=8, ffn_width=32, context_length=512)


def make(data, job, root, **kwargs):
    return create_trainer(data, job, source='a'*40, root=root, device='cpu',
        cpu_canary=True, spec_override=spec(), updates_override=4, **kwargs)


def test_registry_and_no_silent_reduction():
    assert len(jobs('optimization')) == 8
    assert len(jobs('representation')) == 6
    for phase in ('optimization', 'representation'):
        rows = jobs(phase)
        assert len({j['job_id'] for j in rows}) == len(rows)
        assert all(j['profile'] == 'S20' and j['epochs'] == 4 for j in rows)
        for row in rows: validate(row, phase)
    with pytest.raises(ValueError): validate({**jobs('optimization')[0], 'epochs': 2}, 'optimization')
    assert protocol()['scientific_status'].startswith('CAUSE UNRESOLVED')


def test_baseline_reproduces_old_factory_and_crossed_seeds(tmp_path):
    data = make_dataset(worlds=16)
    job = jobs('optimization')[0]
    new = make(data, job, tmp_path/'new')
    oldjob = {k:job[k] for k in ('job_id','profile','method','seed','epochs','mechanism')}
    old = original(data, oldjob, source='a'*40, root=tmp_path/'old', device='cpu',
        cpu_canary=True, spec_override=spec(), updates_override=4)
    assert new.plan == old.plan
    assert all(torch.equal(a,b) for a,b in zip(new.model.parameters(), old.model.parameters()))
    assert new.step() == old.step()
    assert all(torch.equal(a,b) for a,b in zip(new.model.parameters(), old.model.parameters()))
    crossed = make(data, jobs('optimization')[6], tmp_path/'crossed')
    initial = make(data, job, tmp_path/'initial')
    assert all(torch.equal(a,b) for a,b in zip(initial.model.parameters(), crossed.model.parameters()))
    assert initial.plan != crossed.plan
    assert initial.data_sha != crossed.data_sha


def test_exact_recovery_and_wrong_order_rejected(tmp_path):
    data = make_dataset(worlds=8)
    job = jobs('optimization')[0]
    trainer = make(data, job, tmp_path)
    trainer.step(); trainer.step(); trainer.save()
    restored = make(data, job, tmp_path, resume=True)
    for _ in range(2):
        assert trainer.step() == restored.step()
    assert all(torch.equal(a,b) for a,b in zip(trainer.model.parameters(), restored.model.parameters()))
    # A different order seed cannot continue this lineage under the same identity.
    wrong = {**jobs('optimization')[6], 'job_id': job['job_id']}
    with pytest.raises(ValueError, match='unregistered'):
        make(data, wrong, tmp_path, resume=True)


def test_row_order_preserves_targets_and_exposure():
    data = make_dataset(worlds=112)
    changed = prepare_variant(data)
    assert changed.identity == prepare_variant(data).identity != data.identity
    assert [e.target for e in changed.examples] == [e.target for e in data.examples]
    assert [s.eligible for s in changed.samples] == [s.eligible for s in data.samples]
    assert any(a.tokens != b.tokens for a,b in zip(changed.samples,data.samples))
    assert sorted(len(s.tokens) for s in changed.samples) == sorted(len(s.tokens) for s in data.samples)


def test_measurement_identity_validation(tmp_path):
    from tools.elendril_v717_worker import verify_measurement
    path = tmp_path/'row.json'; job = jobs('optimization')[0]
    metric = dict(schema='elendril-v717-measurement-v1',examples=1,
                  audit={'conflicting_targets':0},teacher_forced={'all':{'tokens':2}},
                  decoded={'status':'MEASURED'})
    cohort = dict(dataset_identity='d'*64, teacher=metric, decoded=metric)
    row = dict(checkpoint='c'*64, update=0, job=job, train=cohort, heldout=cohort,
               gradient_probe={'status':'PASS'})
    path.write_text(json.dumps(row))
    verify_measurement(path, job, checkpoint='c'*64, update=0)
    with pytest.raises(ValueError, match='identity'):
        verify_measurement(path, job, checkpoint='b'*64)
    row['gradient_probe']['status'] = 'FAIL'; path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='gradient'):
        verify_measurement(path, job)


def test_real_measurement_worker_sidecar_and_diagnosis(tmp_path, monkeypatch):
    from anra_v5.elendril.v6_data import write_dataset
    from tools import elendril_v717_worker as worker
    from tools.elendril_v717_report import write_diagnosis
    data=make_dataset(worlds=8); job=jobs('optimization')[0]
    trainer=make(data,job,tmp_path/'weights')
    worker.inherited.bound(trainer)
    output=tmp_path/'output'; (output/'data').mkdir(parents=True)
    write_dataset(make_dataset('validation',worlds=8),output/'data'/'validation.json')
    path=output/'analysis'/'jobs'/job['job_id']/'initial.json'
    path.parent.mkdir(parents=True)
    # Only expensive interactive rollouts are substituted. The full diagnostic,
    # decoding, gradient control, file placement, cache checks and report are real.
    report=dict(checkpoint=trainer.backend.binding.checkpoint_sha256,update=0,
                evaluation={'metrics':{'base':{'all':{'correct':0}},'confirmation':None}})
    monkeypatch.setattr(worker,'_evaluation',lambda *args,**kwargs:report)
    path.write_text(json.dumps(report))
    worker.evaluation(trainer,{},path)
    sidecar=path.parent/'measurements'/'initial.json'
    row=json.loads(sidecar.read_text())
    assert row['gradient_probe']['status']=='PASS'
    assert row['train']['teacher']['examples']==len(data.samples)
    assert trainer.state.global_update==0
    # Cache re-entry must not redo diagnostic inference.
    monkeypatch.setattr(worker,'measure',lambda *args,**kwargs:pytest.fail('cached measurement recomputed'))
    worker.evaluation(trainer,{},path)
    summary=write_diagnosis(output,[job])
    assert summary['observations']==1 and summary['missing']==3
    saved=json.loads((output/'analysis'/'S20_DIAGNOSIS.json').read_text())
    assert saved['cause']=='CAUSE UNRESOLVED'
    assert saved['observations'][0]['interactive']['base']['correct']==0
