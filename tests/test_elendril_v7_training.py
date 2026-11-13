from collections import Counter
from dataclasses import replace
import math

import pytest
import torch

from anra_v5.elendril.foundation_training import Recipe
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.training import MODEL, configure_device
from anra_v5.elendril.v6_data import make_dataset
from anra_v5.elendril.v7_training import V7Trainer, ordered_indices


def test_orders_cover_same_samples_and_match_v6_block():
    data = make_dataset(worlds=8)
    samples = data.samples
    uniform = ordered_indices(samples, 7, 0, 'uniform')
    blocked = ordered_indices(samples, 7, 0, 'blocked')
    interleaved = ordered_indices(samples, 7, 0, 'interleaved')
    for method, order in [('uniform', uniform), ('blocked', blocked), ('interleaved', interleaved)]:
        assert sorted(order) == list(range(len(samples)))
        assert order == ordered_indices(samples, 7, 0, method)
    assert blocked == tuple(sorted(uniform, key=lambda i: samples[i].category not in ('missing', 'conflict')))
    assert ordered_indices(samples, 7, 1, 'blocked') == ordered_indices(samples, 7, 1, 'uniform')
    assert Counter(samples[i].category for i in interleaved) == Counter(s.category for s in samples)
    assert interleaved != blocked
    with pytest.raises(ValueError): ordered_indices(samples, 7, 0, 'bad')


def test_plan_work_and_exact_cpu_resume(tmp_path):
    data = make_dataset(worlds=4)
    samples = data.samples
    batch = min(4, len(samples))
    updates = math.ceil(len(samples) / batch)
    recipe = Recipe(seed=7, total_updates=updates, batch_size=batch, micro_batch_size=2,
                    lr=3e-4, objective='ce', sampling='uniform', schedule='cosine',
                    precision='fp32', activation_checkpointing=False)
    spec = replace(MODEL, vocabulary_size=260, width=16, layers=1, query_heads=2,
                   kv_heads=1, head_dimension=8, ffn_width=32, context_length=512)
    configure_device('cpu')
    def make(method, resume=False):
        return V7Trainer(samples, spec=spec, recipe=recipe, source_commit='a'*40,
                         root=tmp_path / method, lineage='tiny',
                         data_provenance={**data.manifest, 'split': 'train'},
                         tokenizer_manifest=UTF8.manifest.record(), device='cpu',
                         method=method, resume=resume)
    trainers = [make(method) for method in ('uniform', 'blocked', 'interleaved')]
    assert len({sum(t.counts) for t in trainers}) == 1
    for t in trainers:
        assert sorted(i for batch_indices in t.plan for i in batch_indices) == list(range(len(samples)))
        assert t.metadata['provenance']['v7_sampling_method'] == t.method
    trainer = trainers[-1]
    trainer.step(); trainer.save()
    restored = make('interleaved', resume=True)
    assert trainer.plan == restored.plan
    if not trainer.state.complete:
        assert trainer.step() == restored.step()
        assert all(torch.equal(a, b) for a, b in zip(trainer.model.parameters(), restored.model.parameters()))
    with pytest.raises(ValueError):
        V7Trainer(samples, spec=spec, recipe=recipe, source_commit='a'*40,
                  root=tmp_path / 'interleaved', lineage='tiny',
                  data_provenance={**data.manifest, 'split': 'train'},
                  tokenizer_manifest=UTF8.manifest.record(), device='cpu',
                  method='uniform', resume=True)
