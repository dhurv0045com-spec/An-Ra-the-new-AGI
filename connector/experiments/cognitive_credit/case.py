"""Case types: observed surface vs hidden ground truth.

``ObservedCase`` is everything the diagnostic system may see. It contains no
field derived from the planted cause. ``HiddenGroundTruth`` is evaluator-only.
The two types are structurally separate so an intervention generator that
accepts only ``ObservedCase`` cannot receive privileged information without a
deliberate, greppable bypass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Scaffolding labels for this experiment's four clean failure families.
FailureFamily = Literal[
    "missing_knowledge",
    "bad_planning",
    "decode_search_sensitivity",
    "tool_failure",
    "context_failure",
    "model_limitation",
    "unknown",
]

# Diagnosis vocabulary: uncertainty is first-class (Rule 3).
DiagnosisLabel = Literal[
    "missing_knowledge",
    "bad_planning",
    "decode_search_sensitivity",
    "tool_failure",
    "context_failure",
    "model_limitation",
    "multiple_plausible",
    "unresolved",
]


@dataclass(frozen=True, slots=True)
class DecodePolicy:
    """Actual inference policy passed to the Core."""

    temperature: float = 0.0
    top_p: float = 0.92
    candidates: int = 1
    seed: int = 0
    max_new_tokens: int = 24


@dataclass(frozen=True, slots=True)
class ToolBehavior:
    """A real controlled tool adapter behavior, not a status string.

    ``available=False`` means the tool call fails at the adapter level; the
    attempt then sees the adapter's error output, never a fabricated OK.
    """

    name: str
    available: bool = True


@dataclass(frozen=True, slots=True)
class Attempt:
    """One executable attempt: what the Core actually receives and how it decodes.

    This is the unit of intervention. Changing exactly one field relative to
    the baseline attempt changes exactly one causal variable.
    """

    question: str
    knowledge: str = ""
    plan: str = ""
    context_blocks: tuple[str, ...] = ()
    tool: ToolBehavior | None = None
    decode: DecodePolicy = field(default_factory=DecodePolicy)

    def render(self) -> str:
        parts: list[str] = []
        if self.context_blocks:
            parts.append("<context>\n" + "\n".join(self.context_blocks) + "\n</context>")
        if self.knowledge:
            parts.append(f"<k>{self.knowledge}</k>")
        if self.plan:
            parts.append(f"<plan>{self.plan}</plan>")
        if self.tool is not None:
            parts.append(f"<tool>{self.tool.name}</tool>")
        parts.append(f"<q>{self.question}</q>")
        parts.append("<answer>")
        return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class ObservedCase:
    """Everything the diagnostic system may legitimately observe.

    Built by the case factory from public task material only. No planted-cause
    information enters this object: the factory derives the observed surface
    from the *task*, and the hidden evaluator separately records which fault
    was injected into that surface.
    """

    case_id: str
    question: str
    success_criterion: str  # human-readable description of the verifier
    initial_attempt: Attempt
    # A small retrieval corpus the system may search during interventions.
    corpus: tuple[str, ...] = ()
    # Alternative plans the system may try (its own heuristics, not gold).
    plan_candidates: tuple[str, ...] = ()
    tools: tuple[ToolBehavior, ...] = ()


@dataclass(frozen=True, slots=True)
class HiddenGroundTruth:
    """Evaluator-only privileged data. Never given to the diagnostician."""

    family: FailureFamily
    gold_solution: str
    gold_knowledge: str = ""
    gold_plan: str = ""
    notes: str = ""
