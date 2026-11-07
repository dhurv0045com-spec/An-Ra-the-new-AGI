from dataclasses import replace
import copy
import pytest

from anra_v5.elendril.cognition import CognitiveState, Truth
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.v6_data import (
    CHANNEL, MAX_SEQUENCE, _state_key, assert_disjoint, dataset_from_record,
    dataset_to_record, make_dataset, panel_cases, public_action, render_public,
    transform_dataset,
)


def test_split_framing_and_roundtrip():
    train, validation, confirmation = (make_dataset(split, worlds=40) for split in
                                       ("train", "validation", "confirmation"))
    assert_disjoint(train, validation, confirmation)
    assert dataset_from_record(dataset_to_record(train)) == train
    record = dataset_to_record(train)
    record["samples"][0]["tokens"] = [0]
    with pytest.raises(ValueError):
        dataset_from_record(record)
    for e, sample in zip(train.examples, train.samples):
        assert sample.tokens == UTF8.frame(e.prompt, e.target, max_length=MAX_SEQUENCE).ids
        assert sample.tokens[0] == UTF8.bos_id and sample.tokens[-1] == UTF8.eos_id
        assert sample.tokens[1 + len(e.prompt.encode())] == 3


def test_truth_and_hidden_payload_do_not_change_initial_observation_or_action():
    data = make_dataset(worlds=48)
    case = next(c for c in data.cases if CHANNEL in c.environment.channels)
    state = CognitiveState(case.goal).observe(case.environment.initial)
    prompt = render_public(state, tuple(case.environment.channels), case.surface)
    label = public_action(state, tuple(case.environment.channels), case.surface)
    changed = replace(case, truth=Truth(case.truth.episode, None, ()))
    assert render_public(state, tuple(changed.environment.channels), changed.surface) == prompt
    assert public_action(state, tuple(changed.environment.channels), changed.surface) == label
    assert 'derived' not in prompt and 'contradicted' not in prompt


def test_transform_preserves_worlds_and_panels_are_usable():
    base = make_dataset(worlds=120)
    transformed = transform_dataset(base, augmentation=True, scratch=True)
    assert [c.world_id for c in base.cases] == [c.world_id for c in transformed.cases]
    assert transformed.manifest['parent_dataset'] == base.identity
    for panel in ('query-swap', 'time-order', 'alias', 'template', 'distractor',
                  'combined', 'three-hop', 'memory'):
        cases = panel_cases(base, panel)
        assert cases, panel
        for case in cases:
            state = CognitiveState(case.goal).observe(case.environment.initial)
            render_public(state, tuple(case.environment.channels), case.surface, case.memory is not None)
    for panel in ('query-swap', 'time-order'):
        pairs = panel_cases(base, panel)
        assert len(pairs) % 2 == 0
        assert all(pairs[i].pair_id == pairs[i+1].pair_id and
                   pairs[i].truth.expected != pairs[i+1].truth.expected
                   for i in range(0, len(pairs), 2))


def test_timestamp_shift_is_same_semantic_split_key():
    case = make_dataset(worlds=1).cases[0]
    state = CognitiveState(case.goal).observe(case.environment.initial)
    shifted = replace(state, events=tuple(replace(e, tick=e.tick + 100) for e in state.events))
    assert _state_key(state, tuple(case.environment.channels), bool(case.memory)) == _state_key(
        shifted, tuple(case.environment.channels), bool(case.memory))


def test_balanced_heldout_quota_and_disjointness():
    train = make_dataset("train", worlds=8)
    validation = make_dataset("validation", worlds=112)
    confirmation = make_dataset("confirmation", worlds=112)
    assert set(validation.manifest["kinds"].values()) == {16}
    assert set(confirmation.manifest["kinds"].values()) == {16}
    assert validation.manifest["sampling"] == "fixed-per-kind-quotas-v1"
    assert_disjoint(train, validation, confirmation)
