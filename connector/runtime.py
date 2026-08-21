"""An-Ra reference runtime: the one executable path from task to verified result.

This module binds the live pieces into a single closed loop:

    task -> attempt -> Core -> verify -> (success | intervene -> diagnose
    -> repair -> verify) -> structured execution record

It reuses the cognitive-credit machinery (``Attempt.render``, real tool
adapters, ``build_interventions``, ``classify_from_outcomes``) so the runtime
and the experiment share one causal vocabulary. Success is decided only by the
verifier. Every step is recorded as operational state — prompts, outputs,
verifications, interventions, costs — never private neural internals.

Usage:

    from connector.runtime import run
    result = run("What is the capital of Portugal?",
                 executor=executor, expected="Lisbon")
    print(result.status, result.answer)
    print(result.to_json())

    # or the thin façade:
    #   import anra; anra.run(task, checkpoint=path, expected="Lisbon")
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field

from connector.experiments.cognitive_credit.case import (
    Attempt,
    CompletionResult,
    DecodePolicy,
    ObservedCase,
    ToolBehavior,
)
from connector.experiments.cognitive_credit.diagnose import (
    ArmOutcome,
    Diagnosis,
    classify_from_outcomes,
)
from connector.experiments.cognitive_credit.interventions import build_interventions
from connector.experiments.cognitive_credit.run_real import make_core_completer
from connector.experiments.cognitive_credit.runner import contains_answer


@dataclass(frozen=True, slots=True)
class Step:
    """One observable execution step (operational state, not neural internals)."""

    role: str  # "baseline" | "intervention:<name>" | "repair" | "self_report"
    prompt: str
    outputs: tuple[str, ...]
    verified: bool
    error: str | None = None
    n_executions: int = 0
    seconds: float = 0.0


@dataclass(slots=True)
class RunResult:
    task: str
    status: str  # "success" | "repaired" | "failed" | "error"
    answer: str | None
    diagnosis: str | None
    selected_intervention: str | None
    changed_variable: str | None
    repair_success: bool | None
    steps: list[Step] = field(default_factory=list)
    interventions: list[ArmOutcome] = field(default_factory=list)
    learning_candidate: dict[str, str] | None = None
    core_executions: int = 0
    wall_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["interventions"] = [asdict(a) for a in self.interventions]
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _case_id(question: str) -> str:
    return "run-" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]


def run(
    task: str,
    *,
    executor=None,
    checkpoint: str | None = None,
    device: str = "cpu",
    expected: str | None = None,
    verifier=None,
    knowledge: tuple[str, ...] | list[str] = (),
    plan_candidates: tuple[str, ...] | list[str] = (),
    tools: tuple[ToolBehavior, ...] | list[ToolBehavior] = (),
    decode: DecodePolicy | None = None,
    complete=None,
) -> RunResult:
    """Run one task to a verified conclusion with a full execution record.

    Either pass a loaded ``CoreExecutor`` or a ``checkpoint`` path. Success is
    decided by ``verifier(text) -> bool`` or, when ``expected`` is given, by
    tolerant token match against it. On failure the runtime executes the
    single-variable intervention battery, diagnoses from measured flips only,
    retries the winner, and — when a repair verifies — emits a learning
    candidate (verified corrective experience) for later offline training.
    ``complete`` overrides Core execution (used by oracle-physics tests and
    replay); it must return ``CompletionResult`` and never a success label.
    """
    if executor is None and complete is None:
        if not checkpoint:
            raise ValueError("run() needs an executor, a completer, or a checkpoint path")
        from anra_core.executor import CoreExecutor

        executor = CoreExecutor.from_checkpoint(checkpoint, device=device)

    if verifier is None:
        if expected is None:
            raise ValueError("run() needs `expected` or a custom `verifier`")

        def verifier(text: str) -> bool:  # noqa: F811 - task-derived verifier
            return contains_answer(text, expected)

    policy = decode or DecodePolicy()
    # The baseline attempt starts with NO retrieved knowledge: the corpus is
    # what interventions may retrieve from, mirroring the experiment's design.
    attempt0 = Attempt(
        question=task,
        knowledge="",
        plan=plan_candidates[0] if plan_candidates else "",
        context_blocks=(),
        tool=tools[0] if tools else None,
        decode=policy,
    )
    observed = ObservedCase(
        case_id=_case_id(task),
        question=task,
        success_criterion=f"answer contains {expected!r}" if expected else "custom verifier",
        initial_attempt=attempt0,
        corpus=tuple(knowledge),
        plan_candidates=tuple(plan_candidates),
        tools=tuple(tools),
    )

    if complete is not None:
        stats = {"generations": 0}
    else:
        if executor is None or executor.tokenizer is None:
            raise ValueError("executor has no bound tokenizer; cannot run")
        complete, stats = make_core_completer(executor, executor.tokenizer)
    started = time.time()
    steps: list[Step] = []

    def execute(role: str, attempt: Attempt) -> tuple[CompletionResult, str | None]:
        t0 = time.time()
        result = complete(attempt)
        passing = next((t for t in result.texts if verifier(t)), None)
        steps.append(Step(
            role=role,
            prompt=attempt.render(),
            outputs=tuple(result.texts),
            verified=passing is not None,
            error=result.error,
            n_executions=result.n_executions,
            seconds=round(time.time() - t0, 3),
        ))
        return result, passing

    baseline, passing = execute("baseline", attempt0)
    if baseline.error is not None and not baseline.texts:
        return RunResult(
            task=task, status="error", answer=None, diagnosis=None,
            selected_intervention=None, changed_variable=None, repair_success=None,
            steps=steps, interventions=[], learning_candidate=None,
            core_executions=stats["generations"], wall_seconds=round(time.time() - started, 3),
        )
    if passing is not None:
        return RunResult(
            task=task, status="success", answer=passing, diagnosis=None,
            selected_intervention=None, changed_variable=None, repair_success=None,
            steps=steps, interventions=[], learning_candidate=None,
            core_executions=stats["generations"], wall_seconds=round(time.time() - started, 3),
        )

    # Failure: controlled single-variable interventions, diagnosis from flips.
    specs = build_interventions(observed)
    outcomes: list[ArmOutcome] = []
    spec_by_name = {}
    for spec in specs:
        arm, arm_passing = execute(f"intervention:{spec.name}", spec.attempt)
        if arm.error is not None and not arm.texts:
            continue  # arm never executed: cannot count as a measured non-flip
        outcomes.append(ArmOutcome(spec.name, spec.changed, arm_passing is not None))
        spec_by_name[spec.name] = spec

    expected_names = frozenset(s.name for s in specs)
    diagnosis: Diagnosis = classify_from_outcomes(False, tuple(outcomes), expected_arm_names=expected_names)

    repair_success: bool | None = None
    winner_text: str | None = None
    winner_attempt: Attempt | None = None
    if diagnosis.selected_intervention is not None and diagnosis.changed_variable:
        spec = spec_by_name[diagnosis.selected_intervention]
        import dataclasses as _dc

        retry_attempt = (
            _dc.replace(spec.attempt, decode=_dc.replace(spec.attempt.decode, seed=spec.attempt.decode.seed + 17))
            if diagnosis.changed_variable == "decode"
            else spec.attempt
        )
        retry, passing = execute("repair", retry_attempt)
        repair_success = passing is not None
        if repair_success:
            winner_text = passing
            winner_attempt = retry_attempt

    learning_candidate = None
    if repair_success and winner_text and winner_attempt is not None:
        # Verified corrective experience: exactly what a future protocol SFT
        # should consume. Only ever produced from verifier-confirmed output.
        learning_candidate = {
            "task": task,
            "prompt": winner_attempt.render(),
            "verified_output": winner_text,
        }

    return RunResult(
        task=task,
        status="repaired" if repair_success else "failed",
        answer=winner_text,
        diagnosis=diagnosis.label,
        selected_intervention=diagnosis.selected_intervention,
        changed_variable=diagnosis.changed_variable,
        repair_success=repair_success,
        steps=steps,
        interventions=outcomes,
        learning_candidate=learning_candidate,
        core_executions=stats["generations"],
        wall_seconds=round(time.time() - started, 3),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="An-Ra reference runtime")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--knowledge", action="append", default=[])
    parser.add_argument("--plan", action="append", default=[])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run(
        args.task, checkpoint=args.checkpoint, device=args.device,
        expected=args.expected, knowledge=args.knowledge, plan_candidates=args.plan,
    )
    print(result.to_json())
