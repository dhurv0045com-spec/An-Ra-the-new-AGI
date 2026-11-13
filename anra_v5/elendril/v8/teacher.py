"""Separate V8 teacher for public supervision and shortcut-baseline audits.

The teacher operates only on visible public observations and tool availability;
it never inspects evaluator Truth or hidden channel payloads prior to a tool call.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re

from anra_v5.elendril.codec import digest
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.v6_data import action_text

from .contracts import OBSERVATION_VERSION, SYMBOLS, PublicInput

_MAX_SCREEN_BYTES = 512


@dataclass(frozen=True, slots=True)
class PublicFact:
    subject: str
    source: str
    tick: int
    value: str | None


@dataclass(frozen=True, slots=True)
class TeacherStep:
    task_id: str
    world_id: str
    family: str
    stage: str
    public_input: PublicInput
    gold_op: str
    gold_arg: str | None
    target_text: str
    prompt_bytes: int
    is_answer_step: bool


def resolve_public_belief(facts: tuple[PublicFact, ...], subject: str) -> tuple[str, str | None]:
    """Resolve one subject from public facts under V8 timestamp/source rules.

    Later observations from the same source supersede earlier ones.
    Equal-time contradictions or cross-source disagreements remain unresolved.
    """

    relevant = [f for f in facts if f.subject == subject]
    if not relevant:
        return "missing", None
    latest_tick_by_source: dict[str, int] = {}
    for fact in relevant:
        prev = latest_tick_by_source.get(fact.source, -1)
        if fact.tick > prev:
            latest_tick_by_source[fact.source] = fact.tick
    latest_facts = [f for f in relevant if f.tick == latest_tick_by_source[f.source]]
    # Check equal-time contradiction within a single source or across sources
    values = {f.value for f in latest_facts}
    if len(values) != 1:
        return "contradicted", None
    val = next(iter(values))
    if val is None:
        return "null", None
    return "derived", val


def teacher_decide(
    facts: tuple[PublicFact, ...],
    goal_subject: str,
    hops: int,
    query_available: bool,
    memory_available: bool,
) -> tuple[str, str | None]:
    """Compute teacher operation/argument strictly from public facts and availability."""

    current = goal_subject
    for _ in range(max(1, hops)):
        status, val = resolve_public_belief(facts, current)
        if status == "null":
            return "ABSTAIN", None
        if status != "derived":
            if memory_available:
                return "RETRIEVE", None
            if query_available:
                return "QUERY", None
            return "ABSTAIN", None
        assert val is not None
        current = val
    return "ANSWER", current


def audit_teacher_samples(steps: tuple[TeacherStep, ...] | list[TeacherStep]) -> dict[str, object]:
    """Verify zero contradictory visible prompts and report target/length statistics."""

    by_prompt: dict[str, str] = {}
    duplicates = 0
    for step in steps:
        prev = by_prompt.get(step.public_input.text_utf8)
        if prev is not None:
            if prev != step.target_text:
                raise ValueError(
                    f"contradictory teacher targets for identical prompt: {prev!r} vs {step.target_text!r}"
                )
            duplicates += 1
        by_prompt[step.public_input.text_utf8] = step.target_text
    target_counts = dict(sorted(Counter(s.target_text for s in steps).items()))
    answer_counts = dict(sorted(Counter(s.gold_arg for s in steps if s.gold_op == "ANSWER").items()))
    max_ans = max(answer_counts.values()) if answer_counts else 0
    min_ans = min(answer_counts.values()) if answer_counts else 0
    return {
        "total_steps": len(steps),
        "unique_prompts": len(by_prompt),
        "duplicate_same_target": duplicates,
        "contradictory_targets": 0,
        "total_utf8_bytes": sum(s.prompt_bytes for s in steps),
        "target_distribution": target_counts,
        "answer_symbol_distribution": answer_counts,
        "answer_max_min_ratio": (max_ans / min_ans) if min_ans > 0 else None,
    }


def compute_shortcut_baselines(worlds: list[dict[str, object]] | tuple[dict[str, object], ...]) -> dict[str, object]:
    """Measure non-neural heuristic baselines on world records to verify absence of shortcuts."""

    if not worlds:
        return {"worlds": 0}
    correct = defaultdict(int)
    evaluated = 0
    for world in worlds:
        expected = world["hidden"]["expected"]
        if expected is None:
            continue
        evaluated += 1
        facts = world["initial_facts"]
        non_null_vals = [f["value"] for f in facts if f["value"] is not None]
        if non_null_vals:
            if non_null_vals[0] == expected:
                correct["first_row_value"] += 1
            if non_null_vals[-1] == expected:
                correct["last_row_value"] += 1
            max_tick_val = max((f for f in facts if f["value"] is not None), key=lambda f: f["tick"])["value"]
            if max_tick_val == expected:
                correct["highest_tick_value"] += 1
            if min(non_null_vals) == expected:
                correct["lex_min_value"] += 1
            if max(non_null_vals) == expected:
                correct["lex_max_value"] += 1
        if expected == "A":
            correct["constant_A"] += 1
    return {
        "answerable_worlds": evaluated,
        "heuristics": {k: (v / evaluated if evaluated else None) for k, v in sorted(correct.items())},
    }


def _target_from_op(op: str, arg: str | None) -> str:
    if op == "ANSWER":
        return action_text("ANSWER", arg)
    if op == "QUERY":
        return "Q"
    if op == "RETRIEVE":
        return "R"
    if op == "ABSTAIN":
        return "!"
    raise ValueError(f"unsupported V8 op: {op!r}")


def build_teacher_steps_for_world(
    world: dict[str, object],
    *,
    render_fn,
) -> tuple[TeacherStep, ...]:
    """Generate all supervised public decision steps for a world without touching Truth."""

    facts = tuple(PublicFact(**row) for row in world["initial_facts"])
    query_avail = bool(world["allowed_public_tools"]["query_available"])
    mem_avail = bool(world["allowed_public_tools"]["memory_available"])
    # Hidden channels/memory are accessed ONLY when the teacher chooses QUERY/RETRIEVE,
    # mirroring real environment execution.
    query_payload = tuple(PublicFact(**row) for row in world["hidden"]["query_payload"])
    memory_payload = tuple(PublicFact(**row) for row in world["hidden"]["memory_payload"])
    hops = int(world["hops"])
    goal_subject = str(world["goal_subject"])
    steps: list[TeacherStep] = []
    seen_facts = set(facts)
    ordered_facts = list(facts)
    for decision in range(4):
        text = render_fn(
            ordered_facts=tuple(ordered_facts),
            goal_subject=goal_subject,
            hops=hops,
            query_available=query_avail,
            memory_available=mem_avail,
            template_id=int(world["renderer_template"]),
            alias_prefix=str(world["alias_prefix"]),
            order_seed=int(world["row_order_seed"]),
            decision=decision,
        )
        raw_bytes = len(text.encode("utf-8"))
        # Reserve framing tokens (BOS, ANSWER, target byte, EOS) <= 512
        if raw_bytes + 5 > _MAX_SCREEN_BYTES:
            raise ValueError(f"rendered V8 prompt exceeds {_MAX_SCREEN_BYTES}-byte limit ({raw_bytes} bytes)")
        op, arg = teacher_decide(tuple(ordered_facts), goal_subject, hops, query_avail, mem_avail)
        target = _target_from_op(op, arg)
        # Validate UTF8 framing fits
        UTF8.frame(text, target, max_length=_MAX_SCREEN_BYTES)
        steps.append(
            TeacherStep(
                task_id=f"{world['world_id']}/{decision}",
                world_id=str(world["world_id"]),
                family=str(world["family"]),
                stage="initial" if decision == 0 else f"observed-{decision}",
                public_input=PublicInput(
                    text_utf8=text,
                    query_available=query_avail,
                    memory_available=mem_avail,
                    action_budget=4 - decision,
                    observation_version=OBSERVATION_VERSION,
                ),
                gold_op=op,
                gold_arg=arg,
                target_text=target,
                prompt_bytes=raw_bytes,
                is_answer_step=(op == "ANSWER"),
            )
        )
        if op == "QUERY":
            for f in query_payload:
                if f not in seen_facts:
                    seen_facts.add(f)
                    ordered_facts.append(f)
            query_avail = False
        elif op == "RETRIEVE":
            for f in memory_payload:
                if f not in seen_facts:
                    seen_facts.add(f)
                    ordered_facts.append(f)
            mem_avail = False
        else:
            break
    else:
        raise ValueError("teacher failed to terminate within 4 steps")
    return tuple(steps)


__all__ = [
    "PublicFact",
    "TeacherStep",
    "audit_teacher_samples",
    "build_teacher_steps_for_world",
    "compute_shortcut_baselines",
    "resolve_public_belief",
    "teacher_decide",
]
