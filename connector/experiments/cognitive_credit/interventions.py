"""Intervention generation from observed information only.

``build_interventions`` accepts an ``ObservedCase`` — the type system makes it
impossible to pass ``HiddenGroundTruth``. Every arm is derived from material
the diagnostician actually has:

- knowledge arms retrieve from ``case.corpus`` (may fail to find gold);
- plan arms come from ``case.plan_candidates`` (the system's own heuristics);
- decode arm changes the real inference policy;
- tool arm toggles a real adapter behavior;
- context arms re-pack the same information (near vs distractor-packed).

The returned specs record *which variable changed* so outcomes can later be
represented as intervention -> changed variables -> outcome, without hard-
coding label semantics into the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    "context",
]


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    """One controlled intervention: baseline with exactly one variable changed."""

    name: str
    changed: ChangedVariable
    attempt: Attempt


def _repack_near(attempt: Attempt) -> Attempt:
    """Move existing knowledge to the front of context, drop nothing else."""
    blocks = (attempt.knowledge, *attempt.context_blocks) if attempt.knowledge else attempt.context_blocks
    return Attempt(
        question=attempt.question,
        knowledge="",
        plan=attempt.plan,
        context_blocks=blocks,
        tool=attempt.tool,
        decode=attempt.decode,
    )


def _repack_distract(attempt: Attempt) -> Attempt:
    """Bury existing knowledge under distractor blocks (same total content)."""
    if not attempt.knowledge:
        return attempt
    filler = tuple(
        f"background note {index}: general domain chatter without task facts"
        for index in range(1, 7)
    )
    blocks = (*filler, attempt.knowledge, *attempt.context_blocks)
    return Attempt(
        question=attempt.question,
        knowledge="",
        plan=attempt.plan,
        context_blocks=blocks,
        tool=attempt.tool,
        decode=attempt.decode,
    )


def build_interventions(case: ObservedCase) -> tuple[InterventionSpec, ...]:
    """Generate the full one-variable-at-a-time battery from observed data only.

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

    # Knowledge: retrieve from the provided corpus. The corpus may or may not
    # contain the needed fact; retrieval can legitimately fail.
    for index, doc in enumerate(case.corpus):
        specs.append(
            InterventionSpec(
                name=f"retrieve_{index}",
                changed="knowledge",
                attempt=Attempt(
                    question=base.question,
                    knowledge=doc,
                    plan=base.plan,
                    context_blocks=base.context_blocks,
                    tool=base.tool,
                    decode=base.decode,
                ),
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
                attempt=Attempt(
                    question=base.question,
                    knowledge=base.knowledge,
                    plan=plan,
                    context_blocks=base.context_blocks,
                    tool=base.tool,
                    decode=base.decode,
                ),
            )
        )

    # Decode/search sensitivity: same prompt, real sampling + best-of-N.
    specs.append(
        InterventionSpec(
            name="decode_search_sensitivity",
            changed="decode",
            attempt=Attempt(
                question=base.question,
                knowledge=base.knowledge,
                plan=base.plan,
                context_blocks=base.context_blocks,
                tool=base.tool,
                decode=sampled,
            ),
        )
    )

    # Tool: enable each available adapter (real behavior change).
    for tool in case.tools:
        if base.tool is not None and tool.name == base.tool.name and tool.available == base.tool.available:
            continue
        specs.append(
            InterventionSpec(
                name=f"tool_enable_{tool.name}",
                changed="tool",
                attempt=Attempt(
                    question=base.question,
                    knowledge=base.knowledge,
                    plan=base.plan,
                    context_blocks=base.context_blocks,
                    tool=tool,
                    decode=base.decode,
                ),
            )
        )

    # Context packing: same information, different position/noise.
    specs.append(
        InterventionSpec(
            name="context_repack_near",
            changed="context",
            attempt=_repack_near(base),
        )
    )
    specs.append(
        InterventionSpec(
            name="context_repack_distract",
            changed="context",
            attempt=_repack_distract(base),
        )
    )

    return tuple(specs)
