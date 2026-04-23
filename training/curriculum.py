"""Curriculum learning phases for An-Ra training."""
from dataclasses import dataclass
from typing import List


@dataclass
class CurriculumPhase:
    name: str
    epoch_start: int
    epoch_end: int
    lr_multiplier: float
    data_fraction: float
    ouroboros_passes: int
    description: str


SCHEDULE: List[CurriculumPhase] = [
    CurriculumPhase("warmup", 0, 2, 0.1, 0.3, 1, "LM only, easy samples"),
    CurriculumPhase("ramp", 2, 5, 0.5, 0.6, 2, "Consistency enabled"),
    CurriculumPhase("main", 5, 9, 1.0, 1.0, 3, "Full Ouroboros"),
    CurriculumPhase("refinement", 9, 12, 0.3, 1.0, 3, "Verification focus"),
]


def get_phase(epoch: int) -> CurriculumPhase:
    for phase in reversed(SCHEDULE):
        if epoch >= phase.epoch_start:
            return phase
    return SCHEDULE[0]


def is_valid_schedule() -> bool:
    for i, p in enumerate(SCHEDULE):
        if p.epoch_start >= p.epoch_end:
            return False
        if i > 0 and p.epoch_start != SCHEDULE[i - 1].epoch_end:
            return False
    return True
