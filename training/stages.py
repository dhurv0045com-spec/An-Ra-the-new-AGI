"""Resumable, gate-driven four-stage AN-RA training campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Callable

from training.v2_config import V2_FRONTIER_PARAMETER_COUNT


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
    token_target: int
    verifier_required: bool = False


DEFAULT_STAGES = (
    StageConfig(TrainingStage.FOUNDATION, "next_token", 0.05, 50_000, 5_000_000_000),
    StageConfig(
        TrainingStage.OWNER_ADAPTATION,
        "owner_sft",
        0.6754,
        100_000,
        10_000_000_000,
    ),
    StageConfig(TrainingStage.AGENCY, "tool_trajectory", 0.60, 20_000, 3_000_000_000),
    StageConfig(
        TrainingStage.VERIFIED_REASONING,
        "rlvr_star",
        0.50,
        50_000,
        3_000_000_000,
        True,
    ),
)


@dataclass(frozen=True)
class CampaignConfig:
    model_size: str
    data_path: str
    output_dir: str


@dataclass(frozen=True)
class StageResult:
    stage: str
    passed_gate: bool
    gate_failures: tuple[str, ...]
    checkpoint_path: str | None
    metrics: dict[str, object]
    exit_code: int


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
            "frontier_reference_tokens": V2_FRONTIER_PARAMETER_COUNT * 20,
            "three_b_from_scratch_reference_tokens": 58_365_030_400,
            "three_b_continuation_minimum_tokens": 21_000_000_000,
            "single_t4_role": "smoke_profile_adapter_pilot_inference",
        }


class StagedTrainingCampaign:
    def __init__(self, config: CampaignConfig) -> None:
        self.config = config
        output = Path(config.output_dir)
        self.state = CampaignState(output / f"campaign_{config.model_size}.json")
        self.results_dir = output / "stage_results"

    @staticmethod
    def _gate(config: StageConfig, metrics: dict[str, object]) -> tuple[str, ...]:
        failures: list[str] = []
        ibs = metrics.get("ibs", {})
        dimensions = ibs.get("dimensions", {}) if isinstance(ibs, dict) else {}
        training_tokens = int(metrics.get("training_tokens", 0))
        if training_tokens < config.token_target:
            failures.append(
                f"training tokens {training_tokens:,} < stage target {config.token_target:,}"
            )
        if config.stage == TrainingStage.FOUNDATION:
            perplexity = float(metrics.get("perplexity", float("inf")))
            if perplexity >= 12.0:
                failures.append(f"perplexity {perplexity:.3f} >= 12")
            if not bool(metrics.get("numerically_stable", False)):
                failures.append("numerical stability evidence missing")
            if not bool(metrics.get("tokenizer_schema_valid", False)):
                failures.append("tokenizer and checkpoint schema validation missing")
        elif config.stage == TrainingStage.OWNER_ADAPTATION:
            if float(metrics.get("civ_similarity", 0.0)) < 0.85:
                failures.append("CIV similarity below 0.85")
            if float(ibs.get("overall", ibs.get("overall_score", 0.0))) < 0.50:
                failures.append("IBS overall below 0.50")
            if bool(metrics.get("protected_regression", False)):
                failures.append("reasoning, identity, or safety regressed")
        elif config.stage == TrainingStage.AGENCY:
            if int(metrics.get("verified_trajectories", 0)) < 1000:
                failures.append("fewer than 1,000 verified trajectories")
            if float(dimensions.get("tool_use", 0.0)) < 0.60:
                failures.append("IBS tool_use below 0.60")
        elif config.stage == TrainingStage.VERIFIED_REASONING:
            if float(dimensions.get("reasoning", 0.0)) < 0.70:
                failures.append("IBS reasoning below 0.70")
            if float(metrics.get("star_verification_rate", 0.0)) < 0.90:
                failures.append("STaR verification rate below 0.90")
            if float(metrics.get("truth_checking_coverage", 0.0)) <= 0.95:
                failures.append("truth-checking coverage is not above 0.95")
        return tuple(failures)

    def run_stage(
        self,
        stage_name: str,
        *,
        execute: Callable[[StageConfig], tuple[int, str | None]],
        load_metrics: Callable[[StageConfig], dict[str, object]],
    ) -> StageResult:
        aliases = {
            "stage_a": TrainingStage.FOUNDATION,
            "stage_b": TrainingStage.OWNER_ADAPTATION,
            "stage_c": TrainingStage.AGENCY,
            "stage_d": TrainingStage.VERIFIED_REASONING,
            "owner_sft": TrainingStage.OWNER_ADAPTATION,
            "rlvr": TrainingStage.VERIFIED_REASONING,
        }
        stage = aliases[stage_name] if stage_name in aliases else TrainingStage(stage_name)
        config = next(item for item in self.state.stages if item.stage == stage)
        if stage == TrainingStage.AGENCY:
            preflight_metrics = load_metrics(config)
            if int(preflight_metrics.get("verified_trajectories", 0)) < 1000:
                result = StageResult(
                    stage=stage.value,
                    passed_gate=False,
                    gate_failures=(
                        "agency stage cannot start before 1,000 verified trajectories",
                    ),
                    checkpoint_path=None,
                    metrics=preflight_metrics,
                    exit_code=0,
                )
                self.state.update(
                    stage,
                    step=0,
                    status="blocked",
                    checkpoint=None,
                )
                self.results_dir.mkdir(parents=True, exist_ok=True)
                (self.results_dir / f"{stage.value}.json").write_text(
                    json.dumps(asdict(result), indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                return result
        self.state.update(stage, step=0, status="running", checkpoint=None)
        exit_code, checkpoint = execute(config)
        metrics = load_metrics(config)
        failures = list(self._gate(config, metrics))
        if exit_code != 0:
            failures.insert(0, f"training exited with code {exit_code}")
        result = StageResult(
            stage=stage.value,
            passed_gate=not failures,
            gate_failures=tuple(failures),
            checkpoint_path=checkpoint,
            metrics=metrics,
            exit_code=exit_code,
        )
        self.state.update(
            stage,
            step=config.max_steps,
            status="complete" if result.passed_gate else "blocked",
            checkpoint=checkpoint,
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / f"{stage.value}.json").write_text(
            json.dumps(asdict(result), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return result
