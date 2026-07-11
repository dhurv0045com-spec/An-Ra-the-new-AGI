from __future__ import annotations

import pytest

from training.curriculum_sampler import (
    ScheduledCurriculumSampler,
    curriculum_multipliers,
)


def test_curriculum_curves_match_pre_registered_order() -> None:
    assert curriculum_multipliers("code-before-prose", 0.0) == {
        "permissive_code": 3.0,
        "fineweb_edu": 0.35,
    }
    assert curriculum_multipliers("math-density-ramp", 0.0)["finemath"] == 0.5
    assert curriculum_multipliers("math-density-ramp", 1.0)["finemath"] == 2.0
    assert curriculum_multipliers("identity-mix-late", 0.69)["identity_replay"] == 0.0
    assert curriculum_multipliers("identity-mix-late", 1.0)[
        "identity_replay"
    ] == pytest.approx(2.0)


def test_sampler_is_deterministic_and_code_first() -> None:
    ranges = {"permissive_code": ((0, 100),), "fineweb_edu": ((100, 200),)}
    first = list(
        ScheduledCurriculumSampler(
            ranges, curriculum="code-before-prose", num_samples=1000, seed=41
        )
    )
    replay = list(
        ScheduledCurriculumSampler(
            ranges, curriculum="code-before-prose", num_samples=1000, seed=41
        )
    )
    assert first == replay
    early = first[:300]
    assert sum(index < 100 for index in early) / len(early) > 0.80


def test_identity_late_emits_no_identity_windows_before_boundary() -> None:
    ranges = {"fineweb_edu": ((0, 100),), "identity_replay": ((100, 200),)}
    samples = list(
        ScheduledCurriculumSampler(
            ranges, curriculum="identity-mix-late", num_samples=1000, seed=43
        )
    )
    assert all(index < 100 for index in samples[:700])
    assert any(index >= 100 for index in samples[700:])
