from __future__ import annotations

import math

import pytest

from training.curriculum_sampler import (
    PERMUTATION_SAMPLER_ALGORITHM,
    DeterministicPermutationSampler,
    RankStridedSampler,
    ScheduledCurriculumSampler,
    curriculum_multipliers,
    source_replay_budget_violations,
    validate_sampler_resume_contract,
)


def test_compact_pack_sampler_is_unique_deterministic_and_exactly_resumable() -> None:
    full = list(
        DeterministicPermutationSampler(257, num_samples=257, seed=1301)
    )
    replay = list(
        DeterministicPermutationSampler(257, num_samples=257, seed=1301)
    )
    resumed = list(
        DeterministicPermutationSampler(
            257, num_samples=257, seed=1301, start_position=93
        )
    )

    assert full == replay
    assert len(set(full)) == 257
    assert resumed == full[93:]
    assert DeterministicPermutationSampler(
        257, num_samples=257, seed=1301
    ).state_dict()["algorithm"] == PERMUTATION_SAMPLER_ALGORITHM


def test_compact_pack_resume_rejects_changed_dataset_size() -> None:
    with pytest.raises(RuntimeError, match="dataset size changed"):
        validate_sampler_resume_contract(
            {
                "algorithm": PERMUTATION_SAMPLER_ALGORITHM,
                "seed": 1301,
                "curriculum": "none",
                "position": 32,
                "num_samples": 64,
                "dataset_size": 64,
            },
            seed=1301,
            curriculum="none",
            active_num_samples=64,
            algorithm=PERMUTATION_SAMPLER_ALGORITHM,
            dataset_size=65,
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


def test_foundation_replay_budget_rejects_tiny_fixed_share() -> None:
    violations = source_replay_budget_violations(
        {"fineweb_edu": 5_000_000, "identity_replay": 2},
        {"fineweb_edu": 0.98, "identity_replay": 0.02},
        num_samples=500_000,
    )
    assert set(violations) == {"identity_replay"}
    assert violations["identity_replay"]["expected_draws"] == 10_000
    assert violations["identity_replay"]["allowed_draws"] == 8


def test_foundation_replay_budget_accepts_broad_native_mix() -> None:
    assert not source_replay_budget_violations(
        {
            "fineweb_edu": 3_000_000,
            "permissive_code": 800_000,
            "finemath": 700_000,
            "science_technical": 500_000,
        },
        {
            "fineweb_edu": 11 / 18,
            "permissive_code": 1 / 6,
            "finemath": 2 / 15,
            "science_technical": 4 / 45,
        },
        num_samples=500_000,
    )


def test_none_curriculum_can_extend_its_resume_horizon() -> None:
    assert (
        validate_sampler_resume_contract(
            {
                "algorithm": "counter_based_sha256_v1",
                "seed": 1301,
                "curriculum": "none",
                "position": 32,
                "num_samples": 32,
            },
            seed=1301,
            curriculum="none",
            active_num_samples=64,
        )
        == 32
    )


def test_scheduled_curriculum_cannot_extend_its_resume_horizon() -> None:
    with pytest.raises(RuntimeError, match="scheduled curriculum"):
        validate_sampler_resume_contract(
            {
                "algorithm": "counter_based_sha256_v1",
                "seed": 1301,
                "curriculum": "math-density-ramp",
                "position": 32,
                "num_samples": 32,
            },
            seed=1301,
            curriculum="math-density-ramp",
            active_num_samples=64,
        )


def test_rank_strided_sampler_union_is_exact_canonical_suffix() -> None:
    base = DeterministicPermutationSampler(64, num_samples=64, seed=1301)
    expected = [base.index_at(position) for position in range(8, 64)]
    ranks = [
        list(RankStridedSampler(base, rank=rank, world_size=4, global_cursor=8))
        for rank in range(4)
    ]
    reconstructed = [
        ranks[position % 4][position // 4] for position in range(len(expected))
    ]
    assert reconstructed == expected
    assert all(
        set(ranks[left]).isdisjoint(ranks[right])
        for left in range(4)
        for right in range(left + 1, 4)
    )


def test_rank_strided_sampler_rejects_unequal_collective_horizon() -> None:
    base = DeterministicPermutationSampler(10, num_samples=10, seed=1301)
    with pytest.raises(ValueError, match="divide evenly"):
        RankStridedSampler(base, rank=0, world_size=4, global_cursor=4)


def test_rank_strided_sampler_rejects_partial_per_rank_microbatch() -> None:
    base = DeterministicPermutationSampler(12, num_samples=12, seed=1301)
    with pytest.raises(ValueError, match="global microbatch"):
        RankStridedSampler(
            base,
            rank=0,
            world_size=2,
            global_cursor=0,
            micro_batch_size_per_rank=4,
        )


def test_rank_strided_sampler_rejects_cursor_inside_global_microbatch() -> None:
    base = DeterministicPermutationSampler(14, num_samples=14, seed=1301)
    with pytest.raises(ValueError, match="complete DDP global microbatch boundary"):
        RankStridedSampler(
            base,
            rank=0,
            world_size=2,
            global_cursor=2,
            micro_batch_size_per_rank=2,
        )
