"""Intervention generation from observed information only.

``build_interventions`` accepts an ``ObservedCase`` — the type system makes it
impossible to pass ``HiddenGroundTruth``. Every arm is derived from material
the diagnostician actually has:

- knowledge arms retrieve each corpus document into <k> (may fail to help);
- plan arms come from ``case.plan_candidates`` (the system's own heuristics);
- decode arm changes the real inference policy; the completer is responsible
  for executing every candidate the policy requests (honest best-of-N);
- tool arm toggles a real adapter's availability — the runner invokes
  ``ToolBehavior.run()`` and injects its actual output or error.

Three clean variables (knowledge, plan, decode, tool). A context-reposition
arm was removed deliberately: in this suite every case with repositionable
knowledge has empty baseline knowledge by construction, so the arm could
never make a genuine single-variable change.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
from typing import Literal

from connector.experiments.cognitive_credit.case import (
    Attempt,
    DecodePolicy,
    ObservedCase,
)

ChangedVariable = Literal[
    "knowledge",
    "plan",
    "decode",
    "tool",
]


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    """One controlled intervention: baseline with exactly one variable changed."""

    name: str
    changed: ChangedVariable
    attempt: Attempt


def build_interventions(case: ObservedCase) -> tuple[InterventionSpec, ...]:
    """Generate the one-variable-at-a-time battery from observed data only.

    Deterministic in the case. Contains no branch on any hidden label; there
    is no parameter through which one could enter.
    """
    base = case.initial_attempt
    sampled = DecodePolicy(
        temperature=0.8,
        top_p=0.92,
        candidates=4,
        seed=base.decode.seed + 1,
        max_new_tokens=base.decode.max_new_tokens,
    )
    specs: list[InterventionSpec] = []

    # Knowledge: place each corpus document in <k>. Retrieval can fail.
    for index, doc in enumerate(case.corpus):
        if doc == base.knowledge:
            continue
        specs.append(
            InterventionSpec(
                name=f"retrieve_{index}",
                changed="knowledge",
                attempt=_with(base, knowledge=doc),
            )
        )

    # Planning: try each alternative decomposition the system itself holds.
    for index, plan in enumerate(case.plan_candidates):
        if plan == base.plan:
            continue
        specs.append(
            InterventionSpec(
                name=f"plan_alt_{index}",
                changed="plan",
                attempt=_with(base, plan=plan),
            )
        )

    # Decode/search sensitivity: same prompt, real sampling policy with N
    # candidates. The completer must execute all of them.
    if base.decode.temperature == 0.0 or base.decode.candidates == 1:
        specs.append(
            InterventionSpec(
                name="decode_search_sensitivity",
                changed="decode",
                attempt=_with(base, decode=sampled),
            )
        )

    # Tool: adopt each catalog adapter whose availability differs from the
    # baseline. The runner executes the adapter and feeds the attempt its
    # real output (or explicit failure). No arm if nothing would change.
    for tool in case.tools:
        baseline_tool = base.tool
        if (
            baseline_tool is not None
            and tool.name == baseline_tool.name
            and tool.available == baseline_tool.available
        ):
            continue
        specs.append(
            InterventionSpec(
                name=f"tool_toggle_{tool.name}",
                changed="tool",
                attempt=_with(base, tool=tool),
            )
        )

    # Context: no arm in this suite. Repositioning requires non-empty baseline
    # knowledge, which no case here has (empty knowledge IS the injected fault
    # of the knowledge family). See module docstring.

    return tuple(specs)


def _with(attempt: Attempt, **changes) -> Attempt:
    return _dc_replace(attempt, **changes)


def _repack_near(attempt: Attempt) -> Attempt:
    """Move knowledge into context at the front; identical total content."""
    blocks = (attempt.knowledge, *attempt.context_blocks)
    return Attempt(
        question=attempt.question,
        knowledge="",
        plan=attempt.plan,
        context_blocks=blocks,
        tool=attempt.tool,
        decode=attempt.decode,
    )
