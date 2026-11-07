from dataclasses import replace
import math

import pytest
import torch

from anra_v5.elendril.v6_data import make_dataset, transform_dataset
from anra_v5.elendril.v6_training import Job, create_trainer, represented_samples
from anra_v5.elendril.v6_registry import MODEL_PROFILES, LEGACY_MODEL_PROFILES
from anra_v5.elendril.training import MODEL


def micro_spec(vocab=260):
    return replace(MODEL, vocabulary_size=vocab, width=16, layers=1, query_heads=2,
                   kv_heads=1, head_dimension=8, ffn_width=32, context_length=512)


def test_full_epoch_partial_batch_cursor_and_fresh_instance_resume(tmp_path):
    data = make_dataset(worlds=20)
    job = Job('tiny', 'qualification', 'A1', 7, epochs=1)
    args = dict(data=data, job=job, source='a'*40, root=tmp_path, device='cpu', cpu_canary=True, spec_override=micro_spec())
    trainer = create_trainer(**args)
    assert len(trainer.plan) == math.ceil(len(data.samples)/16)
    assert sorted(i for batch in trainer.plan for i in batch) == list(range(len(data.samples)))
    original = trainer.model.embedding.weight.clone()
    trainer.step(); trainer.save()
    restored = create_trainer(**args, resume=True)
    assert trainer.sample_cursor(1) == restored.state.cursor.sequence_ordinal == 16
    if not trainer.state.complete:
        assert trainer.step() == restored.step()
        for p, q in zip(trainer.model.parameters(), restored.model.parameters()): assert torch.equal(p,q)
    assert not torch.equal(original, trainer.model.embedding.weight)
    assert trainer.sample_cursor(len(trainer.plan)) == len(data.samples)


def test_three_objective_arms_and_representation_pairs_preserve_semantics():
    data = make_dataset(worlds=16)
    legacy, lp, lt = represented_samples(data, 'legacy-ascii')
    byte, bp, bt = represented_samples(data, 'utf8-ascii')
    assert len(legacy) == len(byte)
    assert [s.world_id for s in legacy] == [s.world_id for s in byte]
    for left, right in zip(legacy, byte):
        # Compare model-visible ASCII body and target text independently of
        # vocab IDs / the byte tokenizer's structural ANSWER delimiter.
        li = left.eligible.index(True); ri = right.eligible.index(True)
        assert lt.decode(left.tokens[1:li]) == bt.decode(right.tokens[1:ri-1])
        assert lt.decode(left.tokens[li:-1]) == bt.decode(right.tokens[ri:-1])
    assert lp['semantic_parent'] == bp['semantic_parent'] == data.identity
    for changed in (transform_dataset(data, augmentation=True), transform_dataset(data,scratch=True)):
        assert [c.world_id for c in changed.cases] == [c.world_id for c in data.cases]
    assert MODEL_PROFILES['S3'].parameter_receipt().total == 3084800
    assert MODEL_PROFILES['S8'].parameter_receipt().total == 7970944
    assert MODEL_PROFILES['S20'].parameter_receipt().total == 19871616
    assert LEGACY_MODEL_PROFILES['S3'].vocabulary_size == 46


def test_no_campaign_cpu_fallback_and_recipe_limits(tmp_path):
    data = make_dataset(worlds=4)
    with pytest.raises(ValueError,match='no CPU fallback'):
        create_trainer(data, Job('no','screen','A1',7),source='a'*40,root=tmp_path,device='cpu')
    with pytest.raises(ValueError): Job('bad','screen','R0',7).validate()
