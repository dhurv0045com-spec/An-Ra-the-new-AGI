"""Token-based execution policy for the canonical V4 foundation campaign."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

FOUNDATION_MILESTONES = (200_000_000, 500_000_000, 1_000_000_000, 3_600_000_000)
MIN_WINDOW_TOKENS = 50_000_000
MAX_WINDOW_TOKENS = 170_000_000
ARCHITECTURE_PILOT_TOKENS = 20_000_000


@dataclass(frozen=True)
class FoundationWindow:
    start_token: int
    end_token: int
    target_tokens: int
    next_milestone: int
    estimated_minutes: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def next_milestone(tokens_seen: int) -> int | None:
    seen = max(0, int(tokens_seen))
    return next((target for target in FOUNDATION_MILESTONES if seen < target), None)


def plan_foundation_window(
    *,
    tokens_seen: int,
    tokens_per_second: float,
    session_budget_minutes: int,
    drain_reserve_minutes: int = 30,
) -> FoundationWindow | None:
    """Plan one non-overlapping 50M-170M window without crossing a milestone."""

    seen = max(0, int(tokens_seen))
    milestone = next_milestone(seen)
    if milestone is None:
        return None
    usable_minutes = max(0, int(session_budget_minutes) - int(drain_reserve_minutes))
    rate = max(0.0, float(tokens_per_second))
    estimate = int(rate * usable_minutes * 60) if rate else MIN_WINDOW_TOKENS
    remaining = milestone - seen
    if remaining <= MIN_WINDOW_TOKENS:
        window_tokens = remaining
    else:
        window_tokens = min(remaining, max(MIN_WINDOW_TOKENS, min(MAX_WINDOW_TOKENS, estimate)))
    estimated_minutes = window_tokens / rate / 60.0 if rate else None
    return FoundationWindow(
        start_token=seen,
        end_token=seen + window_tokens,
        target_tokens=window_tokens,
        next_milestone=milestone,
        estimated_minutes=estimated_minutes,
    )


def evaluate_foundation_milestone(
    evidence: Mapping[str, object],
    *,
    target_tokens: int,
) -> dict[str, object]:
    """Fail closed on durability, sampler, numerical and per-source evidence."""

    failures: list[str] = []
    seen = int(evidence.get("tokens_seen", 0))
    if target_tokens not in FOUNDATION_MILESTONES:
        failures.append(f"unknown foundation milestone {target_tokens:,}")
    if seen < int(target_tokens):
        failures.append(f"tokens seen {seen:,} < milestone {target_tokens:,}")
    if str(evidence.get("durability_state", "")) not in {
        "canonical_verified",
        "protected",
    }:
        failures.append("milestone checkpoint is not remotely verified")
    if not bool(evidence.get("numerically_stable")):
        failures.append("numerical stability evidence is missing")
    if int(evidence.get("duplicate_windows", -1)) != 0:
        failures.append("sampler duplicate-window evidence is missing or nonzero")
    validation = evidence.get("validation", {})
    validation = validation if isinstance(validation, Mapping) else {}
    if not str(validation.get("validation_identity", "")):
        failures.append("immutable validation identity is missing")
    domain_losses = validation.get("domain_losses", {})
    domain_losses = domain_losses if isinstance(domain_losses, Mapping) else {}
    if not domain_losses:
        failures.append("source-stratified validation losses are missing")
    for source, row in domain_losses.items():
        if not isinstance(row, Mapping):
            failures.append(f"validation source {source} is invalid")
            continue
        loss = float(row.get("loss", float("inf")))
        if not math.isfinite(loss) or loss < 0.0:
            failures.append(f"validation source {source} has invalid loss")
    behavior = evidence.get("behavior", {})
    behavior = behavior if isinstance(behavior, Mapping) else {}
    required_behavior = {
        "generation_noncollapse",
        "copy",
        "uncertainty",
        "reasoning",
        "math",
        "code",
        "context_use",
    }
    if not required_behavior <= set(behavior):
        missing = sorted(required_behavior - set(behavior))
        failures.append(f"milestone behavior evidence is missing: {missing}")
    return {
        "schema": "anra-foundation-milestone/v1",
        "target_tokens": int(target_tokens),
        "tokens_seen": seen,
        "passed": not failures,
        "failures": failures,
    }


def compare_architecture_pilot(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """Choose promote/reject/replicate for one matched 20M-token pilot."""

    failures: list[str] = []
    for key in ("parent_checkpoint_sha256", "window_id", "seed", "training_tokens"):
        if baseline.get(key) != candidate.get(key):
            failures.append(f"pilot mismatch: {key}")
    if int(candidate.get("training_tokens", 0)) != ARCHITECTURE_PILOT_TOKENS:
        failures.append("candidate pilot does not contain exactly 20M training tokens")
    baseline_score = float(baseline.get("capability_score", 0.0))
    candidate_score = float(candidate.get("capability_score", 0.0))
    capability_delta = candidate_score - baseline_score
    baseline_throughput = float(baseline.get("tokens_per_second", 0.0))
    candidate_throughput = float(candidate.get("tokens_per_second", 0.0))
    throughput_ratio = (
        candidate_throughput / baseline_throughput if baseline_throughput > 0 else 0.0
    )
    candidate_domains = candidate.get("domain_regressions", {})
    candidate_domains = candidate_domains if isinstance(candidate_domains, Mapping) else {}
    worst_regression = max(
        (float(value) for value in candidate_domains.values()),
        default=float("inf"),
    )
    if not candidate_domains:
        failures.append("candidate source-regression evidence is missing")
    if worst_regression > 0.02:
        failures.append(f"worst source regression {worst_regression:.2%} exceeds 2%")
    if throughput_ratio < 0.85:
        failures.append(f"throughput ratio {throughput_ratio:.3f} is below 0.85")
    if bool(candidate.get("oom")) or not bool(candidate.get("numerically_stable")):
        failures.append("candidate is unstable or exceeded memory")

    if failures:
        decision = "reject"
    elif capability_delta >= 0.01:
        decision = "promote"
    elif capability_delta > -0.01:
        decision = "replicate_once"
    else:
        decision = "reject"
    return {
        "schema": "anra-architecture-pilot-decision/v1",
        "decision": decision,
        "capability_delta": capability_delta,
        "throughput_ratio": throughput_ratio,
        "worst_source_regression": worst_regression,
        "failures": failures,
    }

