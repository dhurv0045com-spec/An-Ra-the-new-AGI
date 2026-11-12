"""Tiny CPU mechanical checks; these do not establish a learned capability."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from anra_v5.elendril.training import MODEL, batch_tensors
from anra_v5.elendril.v6_data import make_dataset
from anra_v5.elendril.v71_training import create_trainer
from anra_v5.elendril.v717_measurement import (
    audit_targets, diagnostic, gradient_probe, representative_indices,
)


@pytest.fixture
def instrument(tmp_path):
    data = make_dataset(worlds=8)
    spec = replace(MODEL, vocabulary_size=260, width=16, layers=1, query_heads=2,
                   kv_heads=1, head_dimension=8, ffn_width=32, context_length=512)
    trainer = create_trainer(data, dict(job_id='measure', profile='S20', method='uniform',
        seed=7, epochs=4, mechanism='ce'), source='a' * 40, root=tmp_path,
        device='cpu', cpu_canary=True, spec_override=spec, updates_override=4)
    return trainer, data


def test_teacher_ce_matches_independent_shifted_reference(instrument):
    trainer, data = instrument
    report = diagnostic(trainer, data, decode=False)
    total_loss = 0.
    correct = count = 0
    exact = 0
    # Each unpadded example independently scored: this intentionally does not use
    # the diagnostic's grouping or accumulation code.
    with torch.inference_mode():
        for sample in data.samples:
            tokens, segments, _ = batch_tensors([(sample.tokens, sample.eligible)], trainer.device)
            logits = trainer.backend.logits(tokens, segments)[0].float()
            eligible = [i for i, flag in enumerate(sample.eligible) if flag]
            targets = tokens[0, eligible]
            predictions = logits[[i - 1 for i in eligible]]
            total_loss += float(torch.nn.functional.cross_entropy(predictions, targets, reduction='sum'))
            matches = predictions.argmax(-1).eq(targets)
            correct += int(matches.sum())
            count += len(eligible)
            exact += bool(matches.all())
    assert report['teacher_forced']['all']['cross_entropy'] == pytest.approx(total_loss / count, abs=2e-6)
    assert report['teacher_forced']['all']['correct'] == correct
    assert report['teacher_forced']['all']['tokens'] == count
    assert report['teacher_forced']['first_action']['tokens'] == len(data.samples)
    assert report['teacher_forced']['eos']['tokens'] == len(data.samples)
    assert report['teacher_forced_exact_actions']['correct'] == exact
    assert sum(r['all']['tokens'] for r in report['by_kind'].values()) == count
    assert report['decoded']['status'] == 'NOT_RUN'
    json.dumps(report, allow_nan=False)


def test_decoding_is_actual_backend_output_and_preserves_state(instrument):
    trainer, data = instrument
    trainer.model.train()
    # Preserve deliberately heterogeneous module modes and pre-existing grads.
    list(trainer.model.modules())[-1].eval()
    modes = [m.training for m in trainer.model.modules()]
    before = [p.detach().clone() for p in trainer.model.parameters()]
    for p in trainer.model.parameters():
        p.grad = torch.ones_like(p)
    grads = [p.grad.clone() for p in trainer.model.parameters()]
    state = trainer.state
    report = diagnostic(trainer, data, indices=(0,), decode=True)
    sample = data.samples[0]
    first = sample.eligible.index(True)
    with torch.inference_mode():
        expected = trainer.backend.generate((sample.tokens[:first],), max_new_tokens=4)[0]
    assert report['decoded']['records'][0]['generated'] == list(expected.tokens)
    assert report['teacher_minus_decode_exact_accuracy'] == (
        report['teacher_forced_exact_actions']['accuracy'] - report['decoded']['accuracy'])
    assert [m.training for m in trainer.model.modules()] == modes
    assert trainer.state == state
    assert all(torch.equal(a, p) for a, p in zip(before, trainer.model.parameters()))
    assert all(torch.equal(g, p.grad) for g, p in zip(grads, trainer.model.parameters()))
    assert not trainer.optimizer.state


@pytest.mark.parametrize('objective', ['ce', 'action-weighted'])
def test_gradient_reference_matches_microbatches_without_update(instrument, objective):
    trainer, _ = instrument
    trainer.recipe = replace(trainer.recipe, objective=objective)
    weights = [p.detach().clone() for p in trainer.model.parameters()]
    for p in trainer.model.parameters():
        p.grad = torch.full_like(p, .125)
    grads = [p.grad.clone() for p in trainer.model.parameters()]
    modes = [m.training for m in trainer.model.modules()]
    result = gradient_probe(trainer)
    assert result['status'] == 'PASS'
    assert result['examples'] > 1
    assert result['reference_loss'] == pytest.approx(result['microbatch_loss'], abs=1e-6)
    assert len(result['parameters']) == len(weights)
    assert all(torch.equal(w, p) for w, p in zip(weights, trainer.model.parameters()))
    assert all(torch.equal(g, p.grad) for g, p in zip(grads, trainer.model.parameters()))
    assert [m.training for m in trainer.model.modules()] == modes
    assert trainer.state.global_update == 0 and not trainer.optimizer.state
    json.dumps(result, allow_nan=False)


def test_prompt_conflicts_fail_before_scoring(instrument):
    trainer, data = instrument
    original = data.examples[0]
    corrupt = SimpleNamespace(examples=(original, replace(original, target='contradiction')))
    with pytest.raises(ValueError, match='conflicting targets'):
        diagnostic(trainer, corrupt)
    repeated = SimpleNamespace(examples=(original, original))
    assert audit_targets(repeated)['duplicate_same_target'] == 1


def test_target_alignment_failure_restores_modes(instrument):
    trainer, data = instrument
    corrupt = SimpleNamespace(examples=(replace(data.examples[0], target='X'),),
                              samples=(data.samples[0],))
    modes = [m.training for m in trainer.model.modules()]
    with pytest.raises(ValueError, match='framed loss span'):
        diagnostic(trainer, corrupt, decode=False)
    assert modes == [m.training for m in trainer.model.modules()]


def test_representative_selection_retains_every_world_transition():
    data = make_dataset(worlds=24)
    selected = representative_indices(data, per_kind=1)
    worlds = {data.examples[i].world_id for i in selected}
    assert selected == representative_indices(data, per_kind=1)
    assert selected == tuple(i for i, e in enumerate(data.examples) if e.world_id in worlds)
    assert {data.examples[i].kind for i in selected} == {e.kind for e in data.examples}
    for kind in {e.kind for e in data.examples}:
        assert len({data.examples[i].world_id for i in selected if data.examples[i].kind == kind}) == 1
    with pytest.raises(ValueError):
        representative_indices(data, per_kind=0)
