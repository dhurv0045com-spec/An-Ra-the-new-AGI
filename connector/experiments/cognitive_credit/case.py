"""Case types: observed surface vs hidden ground truth.

``ObservedCase`` is everything the diagnostic system may see. It contains no
field derived from the planted cause. ``HiddenGroundTruth`` is evaluator-only.
The two types are structurally separate so an intervention generator that
accepts only ``ObservedCase`` cannot receive privileged information without a
deliberate, greppable bypass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

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
    """A real controlled tool adapter owned by the outer system.

    ``execute`` is the adapter: calling it performs the actual operation and
    returns its real output (or raises). When ``available=False`` the attempt
    sees an explicit adapter error instead of a result. The diagnostician may
    toggle availability; it never fabricates outputs.
    """

    name: str
    available: bool = True
    # Adapter identity is (name, availability); the callable is excluded from
    # equality so structurally identical adapters compare equal.
    execute: Callable[[], str] | None = field(default=None, compare=False)

    def run(self) -> str:
        """Actually execute the adapter. Raises on real failure."""
        if not self.available:
            raise ToolUnavailableError(self.name)
        if self.execute is None:
            raise ToolUnavailableError(f"{self.name} has no executable adapter")
        return self.execute()


class ToolUnavailableError(RuntimeError):
    """Raised when a disabled adapter is invoked."""


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """What a completer returns: raw outputs, never success labels.

    Success is decided exclusively by the runner's verifier. ``texts`` holds
    one entry per executed candidate (best-of-N yields N entries);
    ``n_executions`` counts actual Core/tool invocations for cost metrics;
    ``error`` records a typed execution fault, if any.
    """

    texts: tuple[str, ...] = ()
    n_executions: int = 0
    error: str | None = None


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
