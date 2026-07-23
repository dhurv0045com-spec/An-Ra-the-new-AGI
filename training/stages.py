"""Milestone orchestration for the canonical dense V4 foundation lineage.

Post-training, architecture pilots, and model growth are separate signed
lineages.  They are intentionally not disguised as continuation phases in the
foundation campaign.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from training.foundation_campaign import (
    ARCHITECTURE_PILOT_TOKENS,
    FOUNDATION_MILESTONES,
    MAX_WINDOW_TOKENS,
    MIN_WINDOW_TOKENS,
    evaluate_foundation_milestone,
)
from training.v2_config import (
    ANRA_V4_MODEL_PARAMETER_COUNT,
    CANONICAL_MODEL_PROFILE,
    CANONICAL_TRAINING_SEED,
    TOKENIZER_SCHEMA_VERSION,
)

FOUNDATION_STATE_CONTRACT = "anra-v4-foundation-state/v1"


class FoundationMilestone(StrEnum):
    TOKENS_200M = "foundation_200m"
    TOKENS_500M = "foundation_500m"
    TOKENS_1B = "foundation_1b"
    TOKENS_3_6B = "foundation_3_6b"


_MILESTONE_TARGETS = dict(zip(FoundationMilestone, FOUNDATION_MILESTONES, strict=True))


@dataclass(frozen=True)
class FoundationStageConfig:
    milestone: FoundationMilestone
    token_target: int
    objective: str = "dense_v4_next_token"
    continuation_phase: str = "A"
    training_layout: str = "raw_causal_shards_v1"


FOUNDATION_STAGES = tuple(
    FoundationStageConfig(milestone, target)
    for milestone, target in _MILESTONE_TARGETS.items()
)


@dataclass(frozen=True)
class FoundationCampaignConfig:
    model_size: str
    data_path: str
    output_dir: str


@dataclass(frozen=True)
class MilestoneResult:
    milestone: str
    target_tokens: int
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
    """Compare immutable, source-stratified validation evidence fail closed."""

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
                finite_loss(base_domain, "answer_loss", f"baseline {domain} answer loss"),
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


class FoundationCampaignState:
    """Atomic state for one V4 foundation lineage; legacy states are rejected."""

    def __init__(
        self,
        path: str | Path,
        stages: tuple[FoundationStageConfig, ...] = FOUNDATION_STAGES,
    ) -> None:
        self.path = Path(path)
        self.stages = stages
        self.state: dict[str, object] = {
            "contract_id": FOUNDATION_STATE_CONTRACT,
            "model_profile": CANONICAL_MODEL_PROFILE,
            "seed": CANONICAL_TRAINING_SEED,
            "milestones": {
                config.milestone.value: {
                    "target_tokens": config.token_target,
                    "tokens_seen": 0,
                    "status": "pending",
                    "checkpoint": None,
                }
                for config in stages
            },
        }
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("contract_id") != FOUNDATION_STATE_CONTRACT:
                raise RuntimeError(
                    "Legacy campaign state cannot resume the V4 foundation lineage; "
                    "start with a new foundation state file"
                )
            if payload.get("model_profile") != CANONICAL_MODEL_PROFILE:
                raise RuntimeError("Foundation campaign model profile changed")
            if int(payload.get("seed", -1)) != CANONICAL_TRAINING_SEED:
                raise RuntimeError("Foundation campaign seed changed")
            self.state = payload

    @property
    def milestones(self) -> dict[str, dict[str, object]]:
        value = self.state.get("milestones", {})
        if not isinstance(value, dict):
            raise RuntimeError("Foundation campaign milestones are malformed")
        return value  # type: ignore[return-value]

    def update(
        self,
        milestone: FoundationMilestone,
        *,
        tokens_seen: int,
        status: str,
        checkpoint: str | None,
    ) -> None:
        if status not in {"pending", "running", "complete", "blocked"}:
            raise ValueError(f"Invalid milestone status: {status}")
        target = _MILESTONE_TARGETS[milestone]
        self.milestones[milestone.value] = {
            "target_tokens": target,
            "tokens_seen": max(0, int(tokens_seen)),
            "status": status,
            "checkpoint": checkpoint,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def next_milestone(self) -> FoundationStageConfig | None:
        for config in self.stages:
            row = self.milestones.get(config.milestone.value, {})
            if row.get("status") != "complete":
                return config
        return None

    def manifest(self) -> dict[str, object]:
        return {
            "contract_id": FOUNDATION_STATE_CONTRACT,
            "model_profile": CANONICAL_MODEL_PROFILE,
            "model_parameters": ANRA_V4_MODEL_PARAMETER_COUNT,
            "tokenizer": "v4-32768",
            "tokenizer_schema_version": TOKENIZER_SCHEMA_VERSION,
            "seed": CANONICAL_TRAINING_SEED,
            "milestones": [asdict(config) for config in self.stages],
            "window_tokens": {"minimum": MIN_WINDOW_TOKENS, "maximum": MAX_WINDOW_TOKENS},
            "architecture_pilot_tokens": ARCHITECTURE_PILOT_TOKENS,
            "post_training_is_separate_lineage": True,
            "state": self.state,
        }


def training_progress_report(
    *,
    milestone: str,
    tokens_seen: int,
    tokens_per_second: float,
    session_minutes: int = 180,
) -> dict[str, object]:
    """Estimate sessions to one named cumulative V4 foundation milestone."""

    key = milestone.strip().lower().replace("-", "_")
    targets = {item.value: _MILESTONE_TARGETS[item] for item in FoundationMilestone}
    targets["foundation"] = FOUNDATION_MILESTONES[-1]
    if key not in targets:
        raise ValueError(f"Unknown V4 foundation milestone: {milestone!r}")
    target = int(targets[key])
    seen = max(0, int(tokens_seen))
    remaining = max(0, target - seen)
    throughput = max(0.0, float(tokens_per_second))
    session_tokens = int(throughput * max(1, session_minutes) * 60)
    sessions_remaining = (
        math.ceil(remaining / session_tokens) if remaining and session_tokens else None
    )
    return {
        "schema": "anra-v4-foundation-progress/v1",
        "milestone": key,
        "tokens_seen": seen,
        "target_tokens": target,
        "completion": min(1.0, seen / target),
        "tokens_per_second": throughput,
        "session_minutes": int(session_minutes),
        "tokens_per_session": session_tokens,
        "sessions_remaining": sessions_remaining,
    }


class FoundationTrainingCampaign:
    def __init__(self, config: FoundationCampaignConfig) -> None:
        if config.model_size != CANONICAL_MODEL_PROFILE:
            raise ValueError(
                "The dense foundation campaign accepts only "
                f"{CANONICAL_MODEL_PROFILE!r}"
            )
        self.config = config
        output = Path(config.output_dir)
        self.state = FoundationCampaignState(
            output / f"v4_foundation_{config.model_size}.json"
        )
        self.results_dir = output / "foundation_milestones"

    @staticmethod
    def _gate(
        config: FoundationStageConfig,
        metrics: Mapping[str, object],
    ) -> tuple[str, ...]:
        validation = metrics.get("validation_candidate", {})
        validation = validation if isinstance(validation, Mapping) else {}
        raw_window = metrics.get("raw_window_consumption", {})
        raw_window = raw_window if isinstance(raw_window, Mapping) else {}
        behavior = metrics.get("behavior", {})
        behavior = behavior if isinstance(behavior, Mapping) else {}
        gate = evaluate_foundation_milestone(
            {
                "tokens_seen": int(metrics.get("training_tokens", 0)),
                "durability_state": str(metrics.get("durability_state", "")),
                "numerically_stable": bool(metrics.get("numerically_stable", False)),
                "duplicate_windows": int(raw_window.get("repeated_windows", -1)),
                "validation": validation,
                "behavior": behavior,
            },
            target_tokens=config.token_target,
        )
        failures = [str(value) for value in gate["failures"]]
        if int(metrics.get("tokenizer_schema_version", -1)) != TOKENIZER_SCHEMA_VERSION:
            failures.append("canonical tokenizer schema 4 evidence is missing")
        baseline = metrics.get("validation_baseline", {})
        baseline = baseline if isinstance(baseline, Mapping) else {}
        if baseline:
            regression = build_validation_regression_gate(baseline, validation)
            failures.extend(str(value) for value in regression["failures"])
        return tuple(failures)

    def run_milestone(
        self,
        milestone_name: str,
        *,
        execute: Callable[[FoundationStageConfig], tuple[int, str | None]],
        load_metrics: Callable[[FoundationStageConfig], dict[str, object]],
    ) -> MilestoneResult:
        try:
            milestone = FoundationMilestone(milestone_name)
        except ValueError as exc:
            raise ValueError(
                f"Unknown foundation milestone {milestone_name!r}; legacy Stage A-E "
                "aliases are retired"
            ) from exc
        config = next(item for item in self.state.stages if item.milestone == milestone)
        self.state.update(
            milestone,
            tokens_seen=int(
                self.state.milestones[milestone.value].get("tokens_seen", 0)
            ),
            status="running",
            checkpoint=None,
        )
        exit_code, checkpoint = execute(config)
        metrics = load_metrics(config)
        failures = list(self._gate(config, metrics))
        if exit_code != 0:
            failures.insert(0, f"training exited with code {exit_code}")
        result = MilestoneResult(
            milestone=milestone.value,
            target_tokens=config.token_target,
            passed_gate=not failures,
            gate_failures=tuple(failures),
            checkpoint_path=checkpoint,
            metrics=dict(metrics),
            exit_code=exit_code,
        )
        self.state.update(
            milestone,
            tokens_seen=int(metrics.get("training_tokens", 0)),
            status="complete" if result.passed_gate else "blocked",
            checkpoint=checkpoint,
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        target = self.results_dir / f"{milestone.value}.json"
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(asdict(result), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(target)
        return result
