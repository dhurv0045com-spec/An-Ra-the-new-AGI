"""Fail-closed plan → act → verify execution with ledger evidence."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

from runtime.answer_contracts import scan_untrusted_context
from runtime.experience_ledger import content_hash, record_experience
from verification.registry import DEFAULT_VERIFIER_REGISTRY, VerifierRegistry


@dataclass(frozen=True)
class PlanActStep:
    step_id: str
    action: Callable[[Mapping[str, object]], Mapping[str, object]]
    verifier_name: str
    verifier_payload: Callable[[Mapping[str, object]], Mapping[str, object]]
    irreversible: bool = False
    authorization_verifier_name: str | None = None
    authorization_payload: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None


@dataclass(frozen=True)
class PlanActStepResult:
    step_id: str
    passed: bool
    verifier_name: str
    score: float
    reason: str
    result_hash: str


@dataclass(frozen=True)
class PlanActReport:
    trace_id: str
    goal_id: str
    passed: bool
    stopped_at: str | None
    steps: tuple[PlanActStepResult, ...]


class PlanActVerifyRunner:
    """Executes declared actions only when their verifier passes each step."""

    def __init__(self, registry: VerifierRegistry | None = None, *, threshold: float = 0.8) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.registry = registry or DEFAULT_VERIFIER_REGISTRY
        self.threshold = float(threshold)

    def run(
        self,
        *,
        goal_id: str,
        steps: Sequence[PlanActStep],
        context: Mapping[str, object] | None = None,
        untrusted_spans: Sequence[Mapping[str, object]] = (),
        trace_id: str | None = None,
    ) -> PlanActReport:
        if not goal_id or not steps:
            raise ValueError("goal_id and at least one plan step are required")
        trace_id = trace_id or str(uuid.uuid4())
        findings = scan_untrusted_context(untrusted_spans)
        if any(finding.tainted for finding in findings):
            record_experience(
                trace_id=trace_id,
                kind="plan_act_verify",
                inputs={"goal_id": goal_id, "step_count": len(steps)},
                output={"blocked": "tainted_context"},
                gate_record={"allowed": False, "gate": "prompt_injection"},
                source="agents.plan_act_verify",
                metadata={"context_findings": [asdict(finding) for finding in findings]},
            )
            return PlanActReport(trace_id, goal_id, False, "context_scan", ())

        state: dict[str, object] = dict(context or {})
        results: list[PlanActStepResult] = []
        for step in steps:
            if step.irreversible:
                authorization_name = step.authorization_verifier_name
                authorization_payload = step.authorization_payload
                if not authorization_name or authorization_payload is None:
                    result = PlanActStepResult(
                        step_id=step.step_id,
                        passed=False,
                        verifier_name=authorization_name or "authorization_required",
                        score=0.0,
                        reason="missing_irreversible_action_authorization",
                        result_hash=content_hash({"authorized": False}),
                    )
                    results.append(result)
                    record_experience(
                        trace_id=trace_id,
                        kind="plan_act_verify",
                        inputs={"goal_id": goal_id, "step_id": step.step_id},
                        output={"result_hash": result.result_hash, "irreversible": True},
                        verifier_verdicts=[
                            {
                                "name": result.verifier_name,
                                "score": 0.0,
                                "passed": False,
                                "tier": 1,
                                "reason": result.reason,
                            }
                        ],
                        gate_record={
                            "allowed": False,
                            "gate": "irreversible_action_authorization",
                        },
                        source="agents.plan_act_verify",
                    )
                    return PlanActReport(
                        trace_id, goal_id, False, step.step_id, tuple(results)
                    )

                authorization = self.registry.verify(
                    authorization_name,
                    authorization_payload(state),
                )
                authorized = float(authorization.score) >= self.threshold
                record_experience(
                    trace_id=trace_id,
                    kind="plan_act_verify",
                    inputs={"goal_id": goal_id, "step_id": step.step_id},
                    output={"irreversible": True, "authorization_checked": True},
                    verifier_verdicts=[
                        {
                            "name": authorization_name,
                            "score": float(authorization.score),
                            "passed": authorized,
                            "tier": int(authorization.tier),
                            "reason": str(authorization.reason),
                        }
                    ],
                    gate_record={
                        "allowed": authorized,
                        "gate": "irreversible_action_authorization",
                    },
                    source="agents.plan_act_verify",
                )
                if not authorized:
                    result = PlanActStepResult(
                        step_id=step.step_id,
                        passed=False,
                        verifier_name=authorization_name,
                        score=float(authorization.score),
                        reason=str(authorization.reason),
                        result_hash=content_hash({"authorized": False}),
                    )
                    results.append(result)
                    return PlanActReport(
                        trace_id, goal_id, False, step.step_id, tuple(results)
                    )

            output = dict(step.action(state))
            verdict = self.registry.verify(step.verifier_name, step.verifier_payload(output))
            passed = float(verdict.score) >= self.threshold
            result = PlanActStepResult(
                step_id=step.step_id,
                passed=passed,
                verifier_name=step.verifier_name,
                score=float(verdict.score),
                reason=str(verdict.reason),
                result_hash=content_hash(output),
            )
            results.append(result)
            record_experience(
                trace_id=trace_id,
                kind="plan_act_verify",
                inputs={"goal_id": goal_id, "step_id": step.step_id},
                output={"result_hash": result.result_hash, "irreversible": step.irreversible},
                verifier_verdicts=[
                    {
                        "name": result.verifier_name,
                        "score": result.score,
                        "passed": result.passed,
                        "tier": int(verdict.tier),
                        "reason": result.reason,
                    }
                ],
                gate_record={"allowed": passed, "gate": "plan_step_verification"},
                source="agents.plan_act_verify",
            )
            if not passed:
                return PlanActReport(trace_id, goal_id, False, step.step_id, tuple(results))
            state.update(output)
        return PlanActReport(trace_id, goal_id, True, None, tuple(results))


def run_plan_act_suite(
    runner: PlanActVerifyRunner,
    cases: Sequence[tuple[str, Sequence[PlanActStep], Mapping[str, object]]],
) -> dict[str, object]:
    """Run a named suite; a 50-goal promotion claim needs 50 real cases."""
    reports = [
        runner.run(goal_id=goal_id, steps=steps, context=context)
        for goal_id, steps, context in cases
    ]
    return {
        "schema_version": 1,
        "goals": len(reports),
        "passed_goals": sum(report.passed for report in reports),
        "all_passed": bool(reports) and all(report.passed for report in reports),
        "meets_50_goal_gate": len(reports) >= 50 and all(report.passed for report in reports),
        "reports": [
            {
                "trace_id": report.trace_id,
                "goal_id": report.goal_id,
                "passed": report.passed,
                "stopped_at": report.stopped_at,
                "steps": [asdict(step) for step in report.steps],
            }
            for report in reports
        ],
    }
