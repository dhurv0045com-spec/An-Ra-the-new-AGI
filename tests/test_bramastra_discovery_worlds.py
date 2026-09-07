import math

import pytest

from bramastra_lab.discovery.worlds import (
    Episode,
    RuleWorld,
    information_gains,
    inputs,
    make_worlds,
    posterior_indices,
    split_worlds,
    target_probability,
)


def world(name, table):
    return RuleWorld(name, "fixture", 2, tuple(table))


def test_inputs_are_little_endian_integer_order():
    assert inputs(2) == ((0, 0), (1, 0), (0, 1), (1, 1))


def test_generation_is_deterministic_nonconstant_and_semantically_unique():
    first = make_worlds(3)
    second = make_worlds(3)
    assert first == second
    assert len({item.table for item in first}) == len(first)
    assert all(set(item.table) == {0, 1} for item in first)
    assert {item.family for item in first} == {"parity", "conjunction", "threshold"}


def test_splits_are_complete_disjoint_stable_and_semantic():
    worlds = make_worlds(6)
    split = split_worlds(worlds)
    assert set(split) == {"train", "dev", "test"}
    assert sum(map(len, split.values())) == len(worlds)
    assert split == split_worlds(list(reversed(worlds)))
    assert all(split[name] for name in split)
    locations = {item.table: name for name, items in split.items() for item in items}
    duplicate = RuleWorld("alias", "other", worlds[0].bits, worlds[0].table)
    alias_split = split_worlds([duplicate])
    assert duplicate in alias_split[locations[duplicate.table]]


def test_episode_enforces_query_contract_without_revealing_target():
    episode = Episode(world("w", (0, 1, 1, 0)), target=2, budget=1)
    assert episode.legal_actions() == (0, 1, 3)
    assert episode.observe(1) == 1
    assert episode.history == [(1, 1)]
    assert episode.legal_actions() == ()
    with pytest.raises(RuntimeError):
        episode.observe(0)
    with pytest.raises(ValueError):
        Episode(world("w", (0, 1, 1, 0)), 2, 2).observe(2)
    with pytest.raises(IndexError):
        Episode(world("w", (0, 1, 1, 0)), 2, 2).observe(4)
    repeated = Episode(world("w", (0, 1, 1, 0)), 2, 2)
    repeated.observe(1)
    with pytest.raises(ValueError):
        repeated.observe(1)


def test_exact_posterior_probability_and_information_gain():
    hypotheses = [
        world("a", (0, 0, 0, 0)),
        world("b", (0, 1, 1, 0)),
        world("c", (0, 0, 0, 0)),
        world("d", (0, 1, 1, 0)),
    ]
    assert posterior_indices(hypotheses, [(1, 1)]) == [1, 3]
    assert target_probability(hypotheses, [], target=2) == 0.5
    gains = information_gains(hypotheses, [], target=2)
    assert math.isclose(gains[1], 1.0)
    assert gains[3] == 0.0
    assert 2 not in gains
    with pytest.raises(ValueError):
        target_probability(hypotheses, [(0, 1)], 2)
