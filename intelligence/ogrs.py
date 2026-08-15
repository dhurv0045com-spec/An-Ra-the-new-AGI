"""Online goal regulation without live weight mutation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Regulation:
    identity_retrieval: int
    reflection_depth: int
    candidate_count: int
    rim_scale: float
    weight_updates_allowed: bool = False


class OnlineGoalRegulationSystem:
    def regulate(self, civ_similarity: float) -> Regulation:
        drift = max(0.0, min(1.0, 1.0 - float(civ_similarity)))
        return Regulation(
            identity_retrieval=1 + min(4, int(drift * 10)),
            reflection_depth=1 + min(4, int(drift * 8)),
            candidate_count=1 + min(7, int(drift * 12)),
            rim_scale=min(1.0, 0.25 + drift),
            weight_updates_allowed=False,
        )
