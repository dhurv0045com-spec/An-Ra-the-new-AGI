"""Frozen construction contract; appraisal rules are versioned in source."""
from dataclasses import dataclass
import math

ACTIVE_HORMONES = ("dopamine", "cortisol", "serotonin", "adrenaline")
MAX_LOG_T = 1.5


@dataclass(frozen=True, slots=True)
class HormonalConfig:
    max_log_temperature: float = MAX_LOG_T
    active_hormones: tuple[str, ...] = ACTIVE_HORMONES
    schema: str = "anra-hal/v1"

    def __post_init__(self):
        if not math.isfinite(self.max_log_temperature) or not 0 < self.max_log_temperature <= MAX_LOG_T:
            raise ValueError("log-temperature bound must be finite in (0, 1.5]")
        if self.active_hormones != ACTIVE_HORMONES or self.schema != "anra-hal/v1":
            raise ValueError("v1 fixes active channels and appraisal schema")
