"""Collection orchestration (B2.1): environments -> experience ledger.

Runs real episodes with a chosen policy and appends hash-chained receipts to
the experience ledger, closing the loop
``environment -> observation/action history -> episode receipt -> replay``.
Every failed episode receives one primary observable failure category per the
system failure taxonomy; unrun or no-event regimes are reported, never
silently dropped.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Mapping, Sequence

from bramastra_lab.research.contracts.core import content_identity
from bramastra_lab.research.environments.base import BaseEnvironment
from bramastra_lab.research.environments.oracles import EpisodeRecord, rollout
from bramastra_lab.research.experience.ledger import EpisodeReceipt, ExperienceLedger

FAILURE_CATEGORIES = frozenset({
    "none", "invalid_action", "exhausted_budget", "wrong_world_prediction",
    "exhausted_budget_after_invalid", "runtime_failure", "unknown",
})

BudgetUnit = "action"  # every action, including final submission, consumes budget


def classify_failure(record: EpisodeRecord) -> str:
    """Assign one primary observable failure category to a finished episode."""
    if record.success:
        return "none"
    invalid = any(step["feedback"].get("kind") == "invalid_action"
                  for step in record.steps)
    if record.truncated and invalid:
        return "exhausted_budget_after_invalid"
    if record.truncated:
        return "exhausted_budget"
    if record.terminated:
        submissions = [step for step in record.steps if step["action"].get("kind") == "submit"]
        if submissions:
            return "wrong_world_prediction"
    return "unknown"


def episode_transcript(record: EpisodeRecord) -> dict[str, Any]:
    """Public transcript of an episode: actions and feedback, never hidden state."""
    return {
        "environment": record.environment,
        "steps": [{"action": dict(step["action"]), "feedback": dict(step["feedback"])}
                  for step in record.steps],
    }


def collect(
    environments: Sequence[BaseEnvironment],
    ledger: ExperienceLedger,
    policy: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    policy_identity: str,
    episodes_per_environment: int = 1,
    episode_prefix: str = "collect",
) -> dict[str, Any]:
    """Run real episodes and append one receipt each. Returns an exact report.

    Failed episodes are valid experience: they are recorded with
    ``success=False`` and a failure category, never dropped or relabeled.
    """
    if episodes_per_environment <= 0:
        raise ValueError("episodes_per_environment must be positive")
    failures_by_category: dict[str, int] = {}
    successes = 0
    total_cost = 0.0
    episode_count = 0
    for environment in environments:
        for index in range(episodes_per_environment):
            episode_id = f"{episode_prefix}-{environment.name}-{index}-{uuid.uuid4().hex[:8]}"
            record = rollout(environment, episode_id, policy,
                             policy_identity=policy_identity)
            transcript = episode_transcript(record)
            receipt = EpisodeReceipt(
                episode_id=record.episode_id,
                task_semantic_id=environment.name,
                family=environment.name,
                collection_policy=policy_identity,
                success=record.success,
                quality="accepted",
                quality_reason=None,
                transition_costs=tuple(step["cost"] for step in record.steps),
                episode_content_identity=content_identity(transcript),
                recorded_at_unix=time.time(),
                notes={"transcript": transcript,
                       "failure_category": classify_failure(record),
                       "budget_unit": BudgetUnit,
                       "inquiry_cost_total": record.inquiry_cost_total,
                       "submission_cost_total": record.submission_cost_total,
                       "terminated": record.terminated,
                       "truncated": record.truncated,
                       "policy_identity": policy_identity},
            )
            ledger.append(receipt)
            episode_count += 1
            total_cost += sum(step["cost"] for step in record.steps)
            if record.success:
                successes += 1
            else:
                category = classify_failure(record)
                failures_by_category[category] = failures_by_category.get(category, 0) + 1
    return {
        "episodes": episode_count,
        "successes": successes,
        "failures_by_category": dict(sorted(failures_by_category.items())),
        "total_action_cost": total_cost,
        "policy_identity": policy_identity,
        "budget_unit": BudgetUnit,
    }
