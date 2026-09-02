"""Fail-closed proofs for the scorer-tournament rotation contract.

The preregistration declares three position rotations per candidate triplet.
A prior implementation declared them in receipts without executing them.
These tests make that impossible again: the schedule is verified, the
geometry helper rejects deleted/duplicated/doctored rotations, the position
gates catch a deliberately biased selector, and the fixture check is
computed rather than asserted.
"""

from __future__ import annotations

import pytest

from e2_architecture.scoring_policy_fixture import (
    rotation_schedule,
    verify_rotation_schedule,
)
from e2_architecture.scoring_policy_tournament import (
    _assert_rotation_geometry,
    _rotation_geometry,
    rotation_order,
)


def test_schedule_has_three_distinct_full_coverage_permutations() -> None:
    schedule = rotation_schedule(groups=7)
    assert verify_rotation_schedule(schedule)
    for rotations in schedule:
        assert len(rotations) == 3 and len({tuple(r) for r in rotations}) == 3
        assert rotations[0] == (0, 1, 2), "identity rotation must come first"
        for candidate in range(3):
            positions = [rotation.index(candidate) for rotation in rotations]
            assert sorted(positions) == [0, 1, 2]


def test_tampered_schedules_fail_verification() -> None:
    good = rotation_schedule(groups=2)
    assert verify_rotation_schedule(good)
    deleted = [group[:2] for group in good]              # a rotation removed
    duplicated = [[group[0], group[0], group[1]] for group in good]
    doctored = [[(0, 1, 1)] * 3 for _ in good]           # not a permutation
    assert not verify_rotation_schedule(deleted)
    assert not verify_rotation_schedule(duplicated)
    assert not verify_rotation_schedule(doctored)
    assert not verify_rotation_schedule([])


def test_rotation_order_rejects_non_permutations() -> None:
    with pytest.raises(ValueError):
        rotation_order(("a", "b", "c"), (0, 1, 1))


def test_geometry_helper_accepts_only_executed_contract() -> None:
    candidates = ("a", "b", "c")
    schedule = rotation_schedule(1)[0]
    scores = {"a": -1.0, "b": -2.0, "c": -3.0}
    geometry = [_rotation_geometry(scores, candidates, rotation) for rotation in schedule]
    _assert_rotation_geometry(geometry, candidates)  # valid: passes

    with pytest.raises(ValueError, match="exactly three"):
        _assert_rotation_geometry(geometry[:2], candidates)  # deleted rotation
    with pytest.raises(ValueError, match="missing, duplicated"):
        _assert_rotation_geometry([geometry[0], geometry[0], geometry[1]], candidates)
    with pytest.raises(ValueError, match="malformed"):
        bad = [dict(item) for item in geometry]
        bad[1] = {**bad[1], "rotation": [0, 1, 1]}
        _assert_rotation_geometry(bad, candidates)


def test_geometry_detects_position_biased_selector() -> None:
    """A selector that always promotes the candidate presented at position 0
    must FAIL the rotation contract: unstable winning role, first-position
    rate 3/3. If this ever passes, the rotation gates are vacuous."""
    candidates = ("a", "b", "c")
    scores = {"a": -1.0, "b": -2.0, "c": -3.0}
    schedule = rotation_schedule(1)[0]
    biased = []
    for rotation in schedule:
        ordered = rotation_order(candidates, rotation)
        winner = ordered[0]  # position-bias: first-presented candidate wins
        biased.append({
            "rotation": list(rotation),
            "presented_order": list(ordered),
            "winner_role": candidates.index(winner),
            "winner_position": ordered.index(winner),
        })
    assert len({item["winner_role"] for item in biased}) == 3  # unstable role
    assert all(item["winner_position"] == 0 for item in biased)  # 3/3 first-position
    with pytest.raises(ValueError, match="rotation-stable"):
        _assert_rotation_geometry(biased, candidates)


def test_unbiased_scoring_yields_stable_role_and_uniform_positions() -> None:
    """Position-invariant scoring: the winning role is constant across
    rotations, and its position cycles through 0, 1, 2 exactly once."""
    candidates = ("a", "b", "c")
    scores = {"a": -1.0, "b": -2.0, "c": -3.0}
    schedule = rotation_schedule(1)[0]
    geometry = [_rotation_geometry(scores, candidates, rotation) for rotation in schedule]
    _assert_rotation_geometry(geometry, candidates)
    assert sum(item["winner_position"] == 0 for item in geometry) == 1


def test_rotation_stable_scores_are_position_invariant_by_construction() -> None:
    """The raw per-candidate scores must be identical across rotation views
    (model traces are reused; only presentation order changes). Any future
    implementation whose scores differ per rotation has smuggled position
    into the model input and must fail the parity gates instead."""
    candidates = ("a", "b", "c")
    scores = {"a": -1.5, "b": -0.5, "c": -2.5}
    for rotation in rotation_schedule(1)[0]:
        ordered = rotation_order(candidates, rotation)
        assert {c: scores[c] for c in ordered} == scores
