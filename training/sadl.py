"""Scale-adjusted owner-dominant data law."""

from __future__ import annotations

import math


def owner_weight(
    parameter_count: int,
    *,
    reference_parameters: int = 904_535_040,
    base_weight: float = 0.65,
    scale_gain: float = 0.05,
    floor: float = 0.50,
    ceiling: float = 0.85,
) -> float:
    if parameter_count <= 0 or reference_parameters <= 0:
        raise ValueError("Parameter counts must be positive.")
    value = base_weight + scale_gain * math.log10(parameter_count / reference_parameters)
    return max(floor, min(ceiling, value))


def normalized_mix(parameter_count: int) -> dict[str, float]:
    owner = owner_weight(parameter_count)
    remaining = 1.0 - owner
    proportions = {
        "teacher": 0.431,
        "symbolic": 0.288,
        "identity": 0.1405,
        "replay": 0.1405,
    }
    mix = {"owner": owner}
    mix.update({name: remaining * ratio for name, ratio in proportions.items()})
    correction = 1.0 - sum(mix.values())
    mix["replay"] += correction
    return mix
