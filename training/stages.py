"""Resumable, gate-driven four-stage AN-RA training campaigns."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
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


def build_validation_regression_gate(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    max_relative_regression: float = 0.02,
    require_answer: bool = False,
) -> dict[str, object]:
    """Compare immutable, domain-stratified validation evidence fail-closed."""
    failures: list[str] = []
    baseline_identity = str(baseline.get("validation_identity", ""))
    candidate_identity = str(candidate.get("validation_identity", ""))
    if not baseline_identity or baseline_identity != candidate_identity:
        failures.append("validation identity is missing or changed")
    baseline_step = int(baseline.get("step", -1))
    candidate_step = int(candidate.get("step", -1))
    if candidate_step <= baseline_step:
        failures.append("candidate validation is not newer than the baseline")

    def finite_loss(report: Mapping[str, object], key: str, label: str) -> float | None:
        raw = report.get(key)
        if raw is None:
            failures.append(f"{label} is missing")
            return None
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            failures.append(f"{label} must be finite and non-negative")
            return None
        return value

    comparisons: dict[str, dict[str, float]] = {}

    def compare(label: str, base_value: float | None, candidate_value: float | None) -> None:
        if base_value is None or candidate_value is None:
            return
        regression = (candidate_value - base_value) / max(base_value, 1e-12)
        comparisons[label] = {
            "baseline": base_value,
            "candidate": candidate_value,
            "relative_regression": regression,
        }
        if regression > max_relative_regression:
            failures.append(
                f"{label} regressed by {regression:.2%} > {max_relative_regression:.2%}"
            )

    compare(
        "overall.loss",
        finite_loss(baseline, "loss", "baseline overall loss"),
        finite_loss(candidate, "loss", "candidate overall loss"),
    )
    baseline_domains = baseline.get("domain_losses", {})
    candidate_domains = candidate.get("domain_losses", {})
    baseline_domains = baseline_domains if isinstance(baseline_domains, Mapping) else {}
    candidate_domains = candidate_domains if isinstance(candidate_domains, Mapping) else {}
    if not baseline_domains:
        failures.append("baseline domain losses are missing")
    for domain in sorted(baseline_domains):
        base_domain = baseline_domains[domain]
        candidate_domain = candidate_domains.get(domain)
        if not isinstance(base_domain, Mapping):
            failures.append(f"baseline domain {domain} is invalid")
            continue
        if not isinstance(candidate_domain, Mapping):
            failures.append(f"candidate domain {domain} is missing")
            continue
        compare(
            f"domain.{domain}.loss",
            finite_loss(base_domain, "loss", f"baseline {domain} loss"),
            finite_loss(candidate_domain, "loss", f"candidate {domain} loss"),
        )
        if require_answer:
            compare(
                f"domain.{domain}.answer_loss",
                finite_loss(
                    base_domain,
                    "answer_loss",
                    f"baseline {domain} answer loss",
                ),
                finite_loss(
                    candidate_domain,
                    "answer_loss",
                    f"candidate {domain} answer loss",
                ),
            )
    return {
        "schema_version": 1,
        "passed": not failures,
        "max_relative_regression": float(max_relative_regression),
        "require_answer": bool(require_answer),
        "validation_identity": baseline_identity or None,
        "comparisons": comparisons,
        "failures": failures,
    }


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
        if config.stage != TrainingStage.FOUNDATION:
            baseline = metrics.get("validation_baseline", {})
            candidate = metrics.get("validation_candidate", {})
            baseline = baseline if isinstance(baseline, Mapping) else {}
            candidate = candidate if isinstance(candidate, Mapping) else {}
            validation_gate = build_validation_regression_gate(
                baseline,
                candidate,
                require_answer=config.stage
                in {TrainingStage.VERIFIED_REASONING, TrainingStage.VERIFIER_REPLAY},
            )
            failures.extend(str(value) for value in validation_gate["failures"])
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
        elif config.stage == TrainingStage.AGENCY:
            pass
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
