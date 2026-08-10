"""Deterministic source-aware sampling for pre-registered curriculum pilots."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator, Sequence

from torch.utils.data import Sampler

CURRICULUMS = {"none", "code-before-prose", "math-density-ramp", "identity-mix-late"}
SAMPLER_ALGORITHM = "counter_based_sha256_v1"
PERMUTATION_SAMPLER_ALGORITHM = "global_affine_permutation_v1"
MAX_FOUNDATION_SOURCE_EPOCHS = 4.0


class DeterministicPermutationSampler(Sampler[int]):
    """Direct-addressable shuffled epochs without replacement.

    Compact cloud packs should not pay to transfer unused tokens or silently
    resample the same windows. Each epoch is an affine permutation over the
    complete dataset. The mapping is deterministic at every absolute position,
    so a resumed worker can start at its saved cursor without rebuilding state.
    """

    def __init__(
        self,
        dataset_size: int,
        *,
        num_samples: int,
        seed: int,
        start_position: int = 0,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.start_position = int(start_position)
        if self.dataset_size < 1:
            raise ValueError("permutation sampler requires a non-empty dataset")
        if self.num_samples < 1:
            raise ValueError("permutation sampler requires a positive sample budget")
        if not 0 <= self.start_position <= self.num_samples:
            raise ValueError("start_position must be within the declared sample budget")

    def __len__(self) -> int:
        return self.num_samples - self.start_position

    def state_dict(self, *, position: int | None = None) -> dict[str, object]:
        cursor = self.start_position if position is None else int(position)
        if not 0 <= cursor <= self.num_samples:
            raise ValueError("sampler cursor is outside its sample budget")
        return {
            "schema_version": 1,
            "algorithm": PERMUTATION_SAMPLER_ALGORITHM,
            "seed": self.seed,
            "position": cursor,
            "num_samples": self.num_samples,
            "dataset_size": self.dataset_size,
            "curriculum": "none",
        }

    def _parameters(self, epoch: int) -> tuple[int, int]:
        digest = hashlib.sha256(
            f"{PERMUTATION_SAMPLER_ALGORITHM}:{self.seed}:{epoch}".encode("ascii")
        ).digest()
        if self.dataset_size == 1:
            return 1, 0
        multiplier = int.from_bytes(digest[:8], "big") % self.dataset_size
        multiplier = max(1, multiplier)
        while math.gcd(multiplier, self.dataset_size) != 1:
            multiplier = (multiplier + 1) % self.dataset_size
            if multiplier == 0:
                multiplier = 1
        offset = int.from_bytes(digest[8:16], "big") % self.dataset_size
        return multiplier, offset

    def __iter__(self) -> Iterator[int]:
        for position in range(self.start_position, self.num_samples):
            yield self.index_at(position)

    def index_at(self, global_position: int) -> int:
        """Return the canonical sample at one absolute campaign position."""

        position = int(global_position)
        if not 0 <= position < self.num_samples:
            raise IndexError("global sampler position is outside its sample budget")
        epoch, local_position = divmod(position, self.dataset_size)
        multiplier, offset = self._parameters(epoch)
        return (multiplier * local_position + offset) % self.dataset_size


def source_replay_budget_violations(
    bucket_counts: dict[str, int],
    target_mass: dict[str, float],
    *,
    num_samples: int,
    max_source_epochs: float = MAX_FOUNDATION_SOURCE_EPOCHS,
) -> dict[str, dict[str, float | int]]:
    """Return sources whose requested raw sampling would exceed safe replay.

    Small supervised corpora may be valuable, but assigning them a fixed share
    of a billion-token raw-causal phase silently turns a few unique examples
    into millions of repeats. Structured continuation owns that upweighting;
    foundation sampling is capped by unique-window capacity.
    """
    positive = {
        str(name): float(weight)
        for name, weight in target_mass.items()
        if float(weight) > 0.0
    }
    total_mass = sum(positive.values())
    if total_mass <= 0.0:
        return {}
    violations: dict[str, dict[str, float | int]] = {}
    for name, weight in positive.items():
        unique_windows = int(bucket_counts.get(name, 0))
        expected_draws = int(math.ceil(int(num_samples) * weight / total_mass))
        allowed_draws = int(math.floor(unique_windows * float(max_source_epochs)))
        if unique_windows <= 0 or expected_draws > allowed_draws:
            violations[name] = {
                "unique_windows": unique_windows,
                "expected_draws": expected_draws,
                "allowed_draws": allowed_draws,
                "expected_epochs": (
                    float("inf")
                    if unique_windows <= 0
                    else expected_draws / unique_windows
                ),
            }
    return violations


def validate_sampler_resume_contract(
    state: dict[str, object],
    *,
    seed: int,
    curriculum: str,
    active_num_samples: int,
    algorithm: str = SAMPLER_ALGORITHM,
    dataset_size: int | None = None,
) -> int:
    """Validate a raw sampler cursor and return its restart position."""
    expected = {
        "algorithm": str(algorithm),
        "seed": int(seed),
        "curriculum": str(curriculum),
    }
    mismatches = {
        key: {"checkpoint": state.get(key), "active": value}
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Raw V4 sampler contract changed across resume: {mismatches}")
    saved_budget = int(state.get("num_samples", -1))
    position = int(state.get("position", -1))
    if saved_budget <= 0:
        raise RuntimeError("Raw V4 resume has an invalid sampler budget")
    if not 0 <= position <= int(active_num_samples):
        raise RuntimeError("Raw V4 sampler cursor is outside its campaign budget")
    if int(active_num_samples) < saved_budget:
        raise RuntimeError("Raw V4 sampler budget cannot shrink across resume")
    if curriculum != "none" and saved_budget != int(active_num_samples):
        raise RuntimeError(
            "A scheduled curriculum cannot change its sample horizon across resume"
        )
    if algorithm == PERMUTATION_SAMPLER_ALGORITHM:
        saved_size = int(state.get("dataset_size", -1))
        if dataset_size is None or saved_size != int(dataset_size):
            raise RuntimeError(
                "Raw V4 permutation dataset size changed across resume: "
                f"checkpoint={saved_size} active={dataset_size}"
            )
    return position


def curriculum_multipliers(name: str, progress: float) -> dict[str, float]:
    """Return relative source multipliers at campaign progress in [0, 1]."""
    if name not in CURRICULUMS:
        raise ValueError(f"unknown curriculum: {name}")
    p = max(0.0, min(1.0, float(progress)))
    if name == "none":
        return {}
    if name == "code-before-prose":
        if p < 0.30:
            return {"permissive_code": 3.0, "fineweb_edu": 0.35}
        blend = min(1.0, (p - 0.30) / 0.30)
        return {
            "permissive_code": 3.0 - 2.0 * blend,
            "fineweb_edu": 0.35 + 0.65 * blend,
        }
    if name == "math-density-ramp":
        return {"finemath": 0.5 + 1.5 * p}
    # Identity enters only after general capability has a foundation.
    if p < 0.70:
        return {"identity_replay": 0.0}
    return {"identity_replay": 2.0 * (p - 0.70) / 0.30}


class ScheduledCurriculumSampler(Sampler[int]):
    """Sample source buckets with deterministic progress-dependent weights.

    Base probability is the realized window share, so a multiplier of 1.0
    preserves the immutable corpus distribution. Sampling with replacement is
    deliberate for matched-token paired pilots; window-consumption telemetry
    separately reports repeat rate.
    """

    def __init__(
        self,
        bucket_ranges: dict[str, Sequence[tuple[int, int]]],
        *,
        curriculum: str,
        num_samples: int,
        seed: int,
        start_position: int = 0,
        target_mass: dict[str, float] | None = None,
        multiplier_fn: Callable[[str, float], dict[str, float]] = curriculum_multipliers,
    ) -> None:
        if curriculum not in CURRICULUMS:
            raise ValueError("Scheduled sampler requires a registered curriculum")
        self.curriculum = curriculum
        self.num_samples = max(1, int(num_samples))
        self.start_position = int(start_position)
        if self.start_position < 0 or self.start_position > self.num_samples:
            raise ValueError("start_position must be within the declared sample budget")
        self.seed = int(seed)
        self.multiplier_fn = multiplier_fn
        normalized = {
            str(name): tuple(
                (int(start), int(stop))
                for start, stop in ranges
                if int(stop) > int(start)
            )
            for name, ranges in bucket_ranges.items()
        }
        normalized = {name: ranges for name, ranges in normalized.items() if ranges}
        if not normalized:
            raise ValueError("curriculum sampler requires at least one source bucket")
        self.bucket_ranges = normalized
        self.names = tuple(sorted(normalized))
        self.bucket_counts = {
            name: sum(stop - start for start, stop in normalized[name]) for name in self.names
        }
        total = sum(self.bucket_counts.values())
        if target_mass is None:
            self.base_mass = {
                name: self.bucket_counts[name] / total for name in self.names
            }
        else:
            declared_targets: dict[str, float] = {}
            for name, raw_weight in target_mass.items():
                weight = float(raw_weight)
                if not math.isfinite(weight) or weight < 0.0:
                    raise ValueError(
                        "target source mix weight must be finite and non-negative: "
                        f"{name}={raw_weight}"
                    )
                declared_targets[str(name)] = weight
            missing = sorted(set(declared_targets) - set(self.names))
            if missing:
                raise ValueError(
                    f"target source mix has no immutable windows for: {missing}"
                )
            positive_targets = {
                name: weight for name, weight in declared_targets.items() if weight > 0.0
            }
            target_total = sum(positive_targets.values())
            if not math.isfinite(target_total) or target_total <= 0.0:
                raise ValueError("target source mix must assign positive mass")
            self.base_mass = {
                name: positive_targets.get(name, 0.0) / target_total
                for name in self.names
            }

    def __len__(self) -> int:
        return self.num_samples - self.start_position

    def state_dict(self, *, position: int | None = None) -> dict[str, object]:
        cursor = self.start_position if position is None else int(position)
        if cursor < 0 or cursor > self.num_samples:
            raise ValueError("sampler cursor is outside its sample budget")
        return {
            "schema_version": 1,
            "algorithm": SAMPLER_ALGORITHM,
            "seed": self.seed,
            "position": cursor,
            "num_samples": self.num_samples,
            "curriculum": self.curriculum,
        }

    def _counter_values(self, position: int) -> tuple[float, int]:
        payload = f"{SAMPLER_ALGORITHM}:{self.seed}:{position}".encode("ascii")
        digest = hashlib.sha256(payload).digest()
        unit = int.from_bytes(digest[:8], "big") / float(2**64)
        offset = int.from_bytes(digest[8:16], "big")
        return unit, offset

    def index_at(self, global_position: int) -> int:
        """Return the canonical scheduled sample at an absolute position."""

        position = int(global_position)
        if not 0 <= position < self.num_samples:
            raise IndexError("global sampler position is outside its sample budget")
        unit, offset_key = self._counter_values(position)
        progress = position / max(1, self.num_samples - 1)
        modifiers = self.multiplier_fn(self.curriculum, progress)
        weights: list[float] = []
        for name in self.names:
            modifier = float(modifiers.get(name, 1.0))
            if not math.isfinite(modifier) or modifier < 0.0:
                raise RuntimeError(
                    "curriculum modifier must be finite and non-negative: "
                    f"{name}={modifier}"
                )
            weights.append(self.base_mass[name] * modifier)
        total = sum(weights)
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("curriculum schedule assigned zero mass to every source")
        threshold = unit * total
        cumulative = 0.0
        selected = next(
            name
            for name, weight in reversed(tuple(zip(self.names, weights, strict=True)))
            if weight > 0.0
        )
        for name, weight in zip(self.names, weights, strict=True):
            cumulative += weight
            if threshold < cumulative:
                selected = name
                break
        offset = offset_key % self.bucket_counts[selected]
        for start, stop in self.bucket_ranges[selected]:
            width = stop - start
            if offset < width:
                return start + offset
            offset -= width
        raise AssertionError("scheduled sampler failed to resolve a declared bucket range")

    def __iter__(self) -> Iterator[int]:
        for position in range(self.start_position, self.num_samples):
            yield self.index_at(position)


class RankStridedSampler(Sampler[int]):
    """Partition one absolute sampler sequence across equal DDP ranks.

    This wrapper never invents padding samples.  The declared global suffix
    must divide evenly across ranks so every worker executes identical numbers
    of collectives and the union remains exactly the single-GPU sequence.
    """

    def __init__(
        self,
        base_sampler: DeterministicPermutationSampler | ScheduledCurriculumSampler,
        *,
        rank: int,
        world_size: int,
        global_cursor: int,
    ) -> None:
        self.base_sampler = base_sampler
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.global_cursor = int(global_cursor)
        if self.world_size < 2:
            raise ValueError("rank-strided sampling requires at least two ranks")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be within world_size")
        if not 0 <= self.global_cursor <= base_sampler.num_samples:
            raise ValueError("global cursor is outside the base sample budget")
        remaining = base_sampler.num_samples - self.global_cursor
        if remaining % self.world_size:
            raise ValueError(
                "remaining global sample budget must divide evenly across DDP ranks"
            )
        self.local_samples = remaining // self.world_size

    def __len__(self) -> int:
        return self.local_samples

    def __iter__(self) -> Iterator[int]:
        for local_offset in range(self.local_samples):
            position = self.global_cursor + self.rank + local_offset * self.world_size
            yield self.base_sampler.index_at(position)
