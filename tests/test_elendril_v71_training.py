from dataclasses import replace
import math

import pytest
import torch

from anra_v5.elendril.training import MODEL
from anra_v5.elendril.v6_data import make_dataset, transform_dataset
from anra_v5.elendril.v71_training import create_trainer


def job(profile='S3', mechanism='ce', method='uniform'):
    return {'job_id': profile + '-' + mechanism, 'profile': profile, 'method': method,
            'seed': 7, 'epochs': 4, 'mechanism': mechanism}


def micro_spec():
    return replace(MODEL, vocabulary_size=260, width=16, layers=1, query_heads=2,
                   kv_heads=1, head_dimension=8, ffn_width=32, context_length=512)


@pytest.fixture(scope='module')
def data():
    return make_dataset(worlds=8)


@pytest.mark.parametrize('profile', ['S3', 'S8', 'S20'])
def test_four_update_cpu_checkpoint_recovery(profile, data, tmp_path):
    args = dict(data=data, job=job(profile), source='a'*40, root=tmp_path,
                device='cpu', cpu_canary=True, spec_override=micro_spec(), updates_override=4)
    trainer = create_trainer(**args)
    assert trainer.recipe.total_updates == 4
    assert trainer.metadata['provenance']['v71_job']['profile'] == profile
    assert trainer.metadata['provenance']['v71_supervision']['samples'] == len(data.samples)
    trainer.step(); trainer.step(); trainer.save()
    resumed = create_trainer(**args, resume=True)
    assert resumed.metadata['plan_sha256'] == trainer.metadata['plan_sha256']
    for _ in range(2):
        assert trainer.step() == resumed.step()
    assert all(torch.equal(a, b) for a,b in zip(trainer.model.parameters(), resumed.model.parameters()))


def test_intermediate_is_same_worlds_with_explicit_extra_supervision(data, tmp_path):
    transformed = transform_dataset(data, scratch=True)
    assert [c.world_id for c in transformed.cases] == [c.world_id for c in data.cases]
    intermediate = [e for e in transformed.examples if e.target.startswith('~')]
    assert intermediate
    assert all(next(c for c in transformed.cases if c.case_id == e.case_id).family == 'route' for e in intermediate)
    trainer = create_trainer(data, job(mechanism='intermediate', method='blocked'), source='a'*40,
        root=tmp_path, device='cpu', cpu_canary=True, spec_override=micro_spec(), updates_override=4)
    summary = trainer.metadata['provenance']['v71_supervision']
    assert summary['samples'] == len(transformed.samples) > len(data.samples)
    assert summary['ce_parent_samples'] == len(data.samples)
    assert sum(v for k,v in summary['target_actions'].items() if k.startswith('~')) == len(intermediate)
    assert not summary['compute_matched_to_ce']
    per_epoch = math.ceil(len(transformed.samples) / trainer.recipe.batch_size)
    assert sorted(i for b in trainer.plan[:per_epoch] for i in b) == list(range(len(transformed.samples)))


def test_rejects_unqualified_architecture_and_cpu(data, tmp_path):
    with pytest.raises(NotImplementedError, match='pointer'):
        create_trainer(data, job(mechanism='pointer'), source='a'*40, root=tmp_path)
    with pytest.raises(ValueError, match='no CPU fallback'):
        create_trainer(data, job(), source='a'*40, root=tmp_path, device='cpu')
    with pytest.raises(ValueError, match='1..8'):
        create_trainer(data, job(), source='a'*40, root=tmp_path,
                       device='cpu', cpu_canary=True, spec_override=micro_spec(), updates_override=9)
