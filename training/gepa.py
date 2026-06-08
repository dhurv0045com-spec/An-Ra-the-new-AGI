from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


GEPA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GEPATrace:
    trace_id: str
    source: str
    component: str
    prompt: str
    response: str = ""
    failure: str = ""
    score: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GEPACandidate:
    candidate_id: str
    target: str
    edit_type: str
    proposed_text: str
    evidence_trace_ids: list[str]
    predicted_delta: dict[str, float]
    rollout_cost_estimate: int
    owner_approval_required: bool = True
    status: str = "proposed_owner_review"


def _category_failure(category: str, score: float) -> str:
    if category == "identity":
        return "identity_or_owner_voice_weak"
    if category == "symbolic":
        return "verifier_grounding_weak"
    if category == "continuity":
        return "memory_continuity_weak"
    if category == "instruction":
        return "instruction_following_weak"
    if category == "reasoning":
        return "reasoning_explanation_weak"
    if score < 0.5:
        return "low_eval_score"
    return "needs_margin"


def traces_from_eval_summary(eval_summary: dict[str, object]) -> list[GEPATrace]:
    results = eval_summary.get("results", [])
    if not isinstance(results, list):
        return []
    traces: list[GEPATrace] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        score = float(item.get("score", 0.0) or 0.0)
        if score >= 0.75:
            continue
        category = str(item.get("category", "general"))
        trace_id = f"eval:{item.get('id', len(traces))}"
        traces.append(
            GEPATrace(
                trace_id=trace_id,
                source="compact_eval",
                component=category,
                prompt=str(item.get("prompt", "")),
                response=str(item.get("response", "")),
                failure=_category_failure(category, score),
                score=score,
                metadata={
                    "reason": item.get("reason", ""),
                    "expected": item.get("expected", ""),
                    "category": category,
                },
            )
        )
    return traces


def traces_from_hard_examples(hard_examples: Iterable[dict[str, object]]) -> list[GEPATrace]:
    traces: list[GEPATrace] = []
    for idx, item in enumerate(hard_examples):
        if not isinstance(item, dict):
            continue
        loss = float(item.get("loss", 0.0) or 0.0)
        preview = str(item.get("preview", ""))
        if not preview:
            continue
        traces.append(
            GEPATrace(
                trace_id=f"hard:{idx}",
                source="hard_examples",
                component="training_loop",
                prompt=preview[:500],
                failure="high_loss_training_example",
                score=max(0.0, 1.0 - loss / 10.0),
                metadata={"loss": loss, "sample_index": item.get("sample_index")},
            )
        )
    return traces


def traces_from_rlvr_report(rlvr_report: dict[str, object]) -> list[GEPATrace]:
    if not rlvr_report:
        return []
    pass_rate = float(rlvr_report.get("verifier_pass_rate", 0.0) or 0.0)
    if pass_rate >= 0.8:
        return []
    return [
        GEPATrace(
            trace_id=f"rlvr:{rlvr_report.get('task_id', 'latest')}",
            source="rlvr_report",
            component=str(rlvr_report.get("task_type", "rlvr")),
            prompt=str(rlvr_report.get("task_id", "")),
            failure="low_verifier_pass_rate",
            score=pass_rate,
            metadata={
                "mean_reward": rlvr_report.get("mean_reward", 0.0),
                "reward_stats": rlvr_report.get("reward_stats", {}),
                "dapo_config": rlvr_report.get("dapo_config", {}),
            },
        )
    ]


def reflect_on_trace(trace: GEPATrace) -> dict[str, object]:
    failure = trace.failure
    if failure == "verifier_grounding_weak":
        cause = "The response likely needs a verifier-first instruction before free-form explanation."
        edit = "Require symbolic/tool verification before final answer when a verifier exists."
        target = "symbolic_tool_policy"
    elif failure == "identity_or_owner_voice_weak":
        cause = "The response did not preserve enough An-Ra identity and owner-shaped voice."
        edit = "Add an identity anchor before self-description and purpose answers."
        target = "identity_prompt"
    elif failure == "memory_continuity_weak":
        cause = "The response did not retain or reuse the task-local memory key."
        edit = "Require explicit recall of user-provided keys before explaining context."
        target = "memory_prompt"
    elif failure == "high_loss_training_example":
        cause = "A training example remains difficult and should become hard replay with a corrected target."
        edit = "Route this example into hard-replay review before increasing its training weight."
        target = "replay_policy"
    elif failure == "low_verifier_pass_rate":
        cause = "RLVR rollouts are producing too many verifier failures."
        edit = "Increase verifier-guided reflection before the next rollout group."
        target = "rlvr_prompt_policy"
    else:
        cause = "The trace shows weak task performance but needs more evidence before action."
        edit = "Collect another eval sample before changing default policy."
        target = "eval_policy"
    return {
        "trace_id": trace.trace_id,
        "failure": failure,
        "cause": cause,
        "proposed_direction": edit,
        "target": target,
    }


def candidates_from_reflections(reflections: list[dict[str, object]]) -> list[GEPACandidate]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for reflection in reflections:
        target = str(reflection.get("target", "eval_policy"))
        grouped.setdefault(target, []).append(reflection)

    candidates: list[GEPACandidate] = []
    for idx, (target, items) in enumerate(sorted(grouped.items())):
        trace_ids = [str(item.get("trace_id", "")) for item in items if item.get("trace_id")]
        directions = []
        for item in items:
            direction = str(item.get("proposed_direction", ""))
            if direction and direction not in directions:
                directions.append(direction)
        proposed_text = " ".join(directions)
        evidence_count = len(trace_ids)
        candidates.append(
            GEPACandidate(
                candidate_id=f"gepa_{target}_{idx}",
                target=target,
                edit_type="prompt_or_policy_rule",
                proposed_text=proposed_text,
                evidence_trace_ids=trace_ids,
                predicted_delta={
                    "eval_success": round(min(0.12, 0.03 * evidence_count), 4),
                    "rollout_cost": -float(max(1, evidence_count)),
                },
                rollout_cost_estimate=max(1, evidence_count),
            )
        )
    return candidates


def score_candidate(candidate: GEPACandidate) -> dict[str, object]:
    eval_delta = float(candidate.predicted_delta.get("eval_success", 0.0))
    cost = max(1, int(candidate.rollout_cost_estimate))
    evidence = len(candidate.evidence_trace_ids)
    score = round((eval_delta * 100.0) + min(10.0, evidence * 2.0) - min(5.0, cost * 0.5), 4)
    return {
        "candidate_id": candidate.candidate_id,
        "score": score,
        "decision": "owner_review" if score >= 4.0 else "collect_more_evidence",
        "pareto": {
            "predicted_eval_delta": eval_delta,
            "rollout_cost_estimate": cost,
            "evidence_traces": evidence,
        },
    }


def build_gepa_report(
    *,
    eval_summary: dict[str, object] | None = None,
    hard_examples: Iterable[dict[str, object]] | None = None,
    rlvr_report: dict[str, object] | None = None,
) -> dict[str, object]:
    traces = []
    traces.extend(traces_from_eval_summary(eval_summary or {}))
    traces.extend(traces_from_hard_examples(hard_examples or []))
    traces.extend(traces_from_rlvr_report(rlvr_report or {}))
    reflections = [reflect_on_trace(trace) for trace in traces]
    candidates = candidates_from_reflections(reflections)
    scores = [score_candidate(candidate) for candidate in candidates]
    return {
        "schema_version": GEPA_SCHEMA_VERSION,
        "generated_at": time.time(),
        "stage": "gepa_reflection_v1",
        "training_enabled": False,
        "auto_apply_enabled": False,
        "required_gate": "owner_review_and_sovereignty_audit",
        "traces": [asdict(trace) for trace in traces],
        "reflections": reflections,
        "candidates": [asdict(candidate) for candidate in candidates],
        "scores": scores,
        "accepted": [],
        "notes": [
            "GEPA candidates are proposals only; no prompt or tool policy is changed automatically.",
            "A candidate can promote only after eval comparison shows no identity, verifier, or safety regression.",
        ],
    }


def write_gepa_report(report: dict[str, object], output_path: Path | None = None) -> dict[str, object]:
    from training.v2_runtime import v2_report_path, write_json

    path = output_path or v2_report_path("gepa_report")
    report["report_path"] = str(path)
    write_json(path, report)
    return report
