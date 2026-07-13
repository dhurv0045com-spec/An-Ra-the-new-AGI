from __future__ import annotations

import math

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


def test_none_curriculum_materializes_declared_target_mix() -> None:
    ranges = {"fineweb_edu": ((0, 100),), "identity_replay": ((100, 200),)}
    samples = list(
        ScheduledCurriculumSampler(
            ranges,
            curriculum="none",
            num_samples=10_000,
            seed=47,
            target_mass={"fineweb_edu": 0.9, "identity_replay": 0.1},
        )
    )
    fineweb_share = sum(index < 100 for index in samples) / len(samples)
    assert fineweb_share == pytest.approx(0.9, abs=0.02)


@pytest.mark.parametrize("weight", [math.nan, math.inf, -0.1])
def test_declared_target_mix_rejects_invalid_weights(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        ScheduledCurriculumSampler(
            {"fineweb_edu": ((0, 10),)},
            curriculum="none",
            num_samples=10,
            seed=1,
            target_mass={"fineweb_edu": weight},
        )


def test_curriculum_rejects_nonfinite_runtime_modifier() -> None:
    sampler = ScheduledCurriculumSampler(
        {"fineweb_edu": ((0, 10),)},
        curriculum="none",
        num_samples=10,
        seed=1,
        multiplier_fn=lambda _name, _progress: {"fineweb_edu": math.nan},
    )
    with pytest.raises(RuntimeError, match="modifier must be finite"):
        list(sampler)
