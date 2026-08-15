"""Seeded simulation domain randomization contract."""

from __future__ import annotations

import random


def sample_domain(seed: int) -> dict[str, float]:
    rng = random.Random(int(seed))
    return {
        "mass_scale": rng.uniform(0.8, 1.2),
        "position_offset_m": rng.uniform(-0.05, 0.05),
        "lighting_scale": rng.uniform(0.3, 1.5),
        "sensor_noise_std": rng.uniform(0.0, 0.02),
        "timing_scale": rng.uniform(0.9, 1.1),
    }
