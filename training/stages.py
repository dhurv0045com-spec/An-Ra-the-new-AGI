"""Resumable, gate-driven four-stage AN-RA training campaigns."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from training.v2_config import V2_FRONTIER_PARAMETER_COUNT


class TrainingStage(str, Enum):
    FOUNDATION = "foundation"
    OWNER_ADAPTATION = "owner_adaptation"
    AGENCY = "agency"
    VERIFIED_REASONING = "verified_reasoning"
    VERIFIER_REPLAY = "verifier_replay"


@dataclass(frozen=True)
class StageConfig:
    stage: TrainingStage
    objective: str
    owner_ratio: float
    max_steps: int
    token_target: int
    verifier_required: bool = False
    continuation_phase: str = "D"
    training_layout: str = "bucket_packed_v1"


DEFAULT_STAGES = (
    StageConfig(
        TrainingStage.FOUNDATION,
        "raw_next_token_frozen_native",
        0.0,
        50_000,
        1_000_000_000,
        continuation_phase="A",
        training_layout="raw_causal_shards_v1",
    ),
    StageConfig(
        TrainingStage.OWNER_ADAPTATION,
        "raw_next_token_staged_native",
        0.0,
        50_000,
        1_000_000_000,
        continuation_phase="B",
        training_layout="raw_causal_shards_v1",
    ),
    StageConfig(
        TrainingStage.AGENCY,
        "mixed_code_math_science_dfc",
        0.05,
        20_000,
        200_000_000,
        continuation_phase="C",
        training_layout="raw_causal_shards_v1",
    ),
    StageConfig(
        TrainingStage.VERIFIED_REASONING,
        "conversation_instruction",
        0.05,
        20_000,
        100_000_000,
        continuation_phase="D",
    ),
    StageConfig(
        TrainingStage.VERIFIER_REPLAY,
        "verifier_replay_tools",
        0.05,
        10_000,
        10_000_000,
        True,
        continuation_phase="E",
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
            "draft_proof_tokens": 32_000_000,
            "frontier_rescue_tokens": 110_000_000,
            "frontier_recovery_floor_tokens": 2_310_000_000,
            "single_t4_role": "smoke_profile_adapter_pilot_inference",
        }


def training_progress_report(
    *,
    phase: str,
    phase_tokens_seen: int,
    tokens_per_second: float,
    session_minutes: int = 180,
) -> dict[str, object]:
    targets = {
        "DRAFT": 32_000_000,
        "RESCUE": 110_000_000,
        **{config.continuation_phase: config.token_target for config in DEFAULT_STAGES},
    }
    normalized = phase.strip().upper()
    target = int(targets.get(normalized, 0))
    seen = max(0, int(phase_tokens_seen))
    remaining = max(0, target - seen)
    throughput = max(0.0, float(tokens_per_second))
    session_tokens = int(throughput * max(1, session_minutes) * 60)
    sessions_remaining = (
        math.ceil(remaining / session_tokens) if remaining and session_tokens else None
    )
    return {
        "schema_version": 1,
        "phase": normalized,
        "tokens_seen": seen,
        "target_tokens": target,
        "completion": min(1.0, seen / target) if target else 0.0,
        "tokens_per_second": throughput,
        "session_minutes": int(session_minutes),
        "tokens_per_session": session_tokens,
        "sessions_remaining": sessions_remaining,
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
            if not bool(metrics.get("subsystem_trace_complete", False)):
                failures.append("isolated native subsystem trace is incomplete")
            if bool(metrics.get("protected_regression", False)):
                failures.append("short-context or core-language validation regressed")
        elif config.stage == TrainingStage.AGENCY:
            if float(metrics.get("validation_regression", 1.0)) > 0.02:
                failures.append("native integration regressed validation loss by more than 2%")
        elif config.stage == TrainingStage.VERIFIED_REASONING:
            if float(metrics.get("coherence_rate", 0.0)) < 0.90:
                failures.append("chat coherence below 0.90")
            if float(metrics.get("format_compliance", 0.0)) < 0.85:
                failures.append("instruction format compliance below 0.85")
        elif config.stage == TrainingStage.VERIFIER_REPLAY:
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
            "stage_e": TrainingStage.VERIFIER_REPLAY,
            "owner_sft": TrainingStage.OWNER_ADAPTATION,
            "rlvr": TrainingStage.VERIFIER_REPLAY,
        }
        stage = aliases[stage_name] if stage_name in aliases else TrainingStage(stage_name)
        config = next(item for item in self.state.stages if item.stage == stage)
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
