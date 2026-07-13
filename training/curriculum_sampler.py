"""Deterministic source-aware sampling for pre-registered curriculum pilots."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator, Sequence

from torch.utils.data import Sampler

CURRICULUMS = {"none", "code-before-prose", "math-density-ramp", "identity-mix-late"}


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
    deliberate for matched-token factorial cells; window-consumption telemetry
    separately reports repeat rate.
    """

    def __init__(
        self,
        bucket_ranges: dict[str, Sequence[tuple[int, int]]],
        *,
        curriculum: str,
        num_samples: int,
        seed: int,
        target_mass: dict[str, float] | None = None,
        multiplier_fn: Callable[[str, float], dict[str, float]] = curriculum_multipliers,
    ) -> None:
        if curriculum not in CURRICULUMS:
            raise ValueError("Scheduled sampler requires a registered curriculum")
        self.curriculum = curriculum
        self.num_samples = max(1, int(num_samples))
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
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        for position in range(self.num_samples):
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
            threshold = rng.random() * total
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
            offset = rng.randrange(self.bucket_counts[selected])
            for start, stop in self.bucket_ranges[selected]:
                width = stop - start
                if offset < width:
                    yield start + offset
                    break
                offset -= width
