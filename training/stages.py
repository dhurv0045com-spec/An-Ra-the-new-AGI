"""Resumable four-stage AN-RA V3 campaign state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path


class TrainingStage(str, Enum):
    FOUNDATION = "foundation"
    OWNER_ADAPTATION = "owner_adaptation"
    AGENCY = "agency"
    VERIFIED_REASONING = "verified_reasoning"


@dataclass(frozen=True)
class StageConfig:
    stage: TrainingStage
    objective: str
    owner_ratio: float
    max_steps: int
    verifier_required: bool = False


DEFAULT_STAGES = (
    StageConfig(TrainingStage.FOUNDATION, "next_token", 0.05, 50_000),
    StageConfig(TrainingStage.OWNER_ADAPTATION, "owner_sft", 0.6754, 100_000),
    StageConfig(TrainingStage.AGENCY, "tool_trajectory", 0.60, 20_000),
    StageConfig(TrainingStage.VERIFIED_REASONING, "grpo_star", 0.50, 50_000, True),
)


class CampaignState:
    def __init__(self, path: str | Path, stages: tuple[StageConfig, ...] = DEFAULT_STAGES) -> None:
        self.path = Path(path)
        self.stages = stages
        self.state = {
            config.stage.value: {"step": 0, "status": "pending", "checkpoint": None}
            for config in stages
        }
        if self.path.exists():
            self.state.update(json.loads(self.path.read_text(encoding="utf-8")))

    def update(
        self,
        stage: TrainingStage,
        *,
        step: int,
        status: str,
        checkpoint: str | None,
    ) -> None:
        if status not in {"pending", "running", "complete", "blocked"}:
            raise ValueError(f"Invalid stage status: {status}")
        self.state[stage.value] = {
            "step": int(step),
            "status": status,
            "checkpoint": checkpoint,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")

    def next_stage(self) -> StageConfig | None:
        for config in self.stages:
            if self.state[config.stage.value]["status"] != "complete":
                return config
        return None

    def manifest(self) -> dict[str, object]:
        return {
            "stages": [asdict(config) for config in self.stages],
            "state": self.state,
            "foundation_token_target": 60_000_000_000,
            "single_t4_role": "smoke_profile_adapter_pilot_inference",
        }
