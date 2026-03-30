"""
================================================================================
FILE: agent/core/goal.py
PROJECT: Agent Loop — 45K
PURPOSE: Goal interpreter — receive, parse, validate, and structure any goal
================================================================================

The goal interpreter is the first thing the agent runs.
It takes raw natural language and turns it into a structured GoalSpec:
  - Clear objective statement
  - Measurable success criteria
  - Constraints (time, resources, safety)
  - Decomposition hints for the planner
  - Safety classification (safe / flagged / rejected)

Constitutional rules from 45I are checked here — unsafe goals are
blocked before any planning or execution begins.
================================================================================
"""

import re
import time
import json
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    PENDING   = "pending"
    APPROVED  = "approved"
    ACTIVE    = "active"
    COMPLETED = "completed"
    FAILED    = "failed"
    REJECTED  = "rejected"   # Failed constitutional check
    FLAGGED   = "flagged"    # Needs clarification or approval


class GoalRisk(Enum):
    LOW    = "low"     # Fully safe, auto-approved
    MEDIUM = "medium"  # Needs review before high-impact steps
    HIGH   = "high"    # Needs explicit user approval at each step
    UNSAFE = "unsafe"  # Rejected outright


@dataclass
class GoalSpec:
    """Structured representation of a goal after interpretation."""
    goal_id:       str
    raw_goal:      str
    objective:     str                      # Cleaned, unambiguous objective
    success_criteria: List[str]             # Measurable conditions for success
    constraints:   List[str]                # What NOT to do / limits
    deadline:      Optional[str]            # Time limit if specified
    resources:     List[str]                # Allowed tools / resources
    risk:          GoalRisk                 # Safety classification
    status:        GoalStatus
    created_at:    float = field(default_factory=time.time)
    clarifications_needed: List[str] = field(default_factory=list)
    metadata:      Dict             = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "goal_id":              self.goal_id,
            "raw_goal":             self.raw_goal,
            "objective":            self.objective,
            "success_criteria":     self.success_criteria,
            "constraints":          self.constraints,
            "deadline":             self.deadline,
            "resources":            self.resources,
            "risk":                 self.risk.value,
            "status":               self.status.value,
            "created_at":           self.created_at,
            "clarifications_needed": self.clarifications_needed,
        }

    def summary(self) -> str:
        lines = [
            f"GOAL: {self.objective}",
            f"Risk: {self.risk.value.upper()}  |  Status: {self.status.value}",
        ]
        if self.success_criteria:
            lines.append("Success criteria:")
            lines.extend(f"  ✓ {c}" for c in self.success_criteria)
        if self.constraints:
            lines.append("Constraints:")
            lines.extend(f"  ✗ {c}" for c in self.constraints)
        if self.deadline:
            lines.append(f"Deadline: {self.deadline}")
        if self.clarifications_needed:
            lines.append("Needs clarification:")
            lines.extend(f"  ? {q}" for q in self.clarifications_needed)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CONSTITUTIONAL RULES (from 45I)
# Hard-coded rules that override any goal
# ──────────────────────────────────────────────────────────────────────────────

_UNSAFE_PATTERNS = [
    # Physical harm
    r"\b(kill|murder|harm|injure|hurt|attack|shoot|bomb|poison)\b.*\b(person|people|human|someone)\b",
    # Weapons
    r"\b(make|build|create|synthesize|manufacture)\b.*\b(weapon|bomb|explosive|poison|bioweapon|malware|virus)\b",
    # Illegal activity
    r"\b(hack|crack|steal|fraud|scam|phish|launder|bypass|exploit)\b.*\b(password|system|account|bank|money|credential)\b",
    # Privacy violations
    r"\b(dox|doxx|stalk|track|spy on|surveillance)\b.*\b(person|someone|individual|user)\b",
    # CSAM
    r"\b(child|minor|underage)\b.*\b(sexual|explicit|nude|naked)\b",
    # Self-harm
    r"\bhow to\b.*\b(suicide|self.harm|cut myself|overdose)\b",
]

_FLAGGED_PATTERNS = [
    # High-stakes irreversible
    r"\b(delete|wipe|format|destroy)\b.*\b(database|all files|production|server)\b",
    r"\bsend.*email.*everyone\b",
    r"\bpost.*public(ly)?\b",
    r"\bpurchase|buy|pay\b",
    r"\b(real money|credit card|bank account)\b",
    # Potential misuse
    r"\b(impersonate|pretend to be|pose as)\b.*\b(CEO|admin|police|government)\b",
]


def _check_constitutional(raw_goal: str) -> tuple:
    """
    Check goal against constitutional rules.
    Returns (GoalRisk, reason_string).
    """
    text = raw_goal.lower()

    for pattern in _UNSAFE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return GoalRisk.UNSAFE, f"Goal matches unsafe pattern: {pattern}"

    flagged_reasons = []
    for pattern in _FLAGGED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            flagged_reasons.append(pattern)

    if flagged_reasons:
        return GoalRisk.HIGH, f"Potentially high-risk action detected: {flagged_reasons[0]}"

    # Medium risk: involves external systems
    if re.search(r'\b(web|internet|email|API|external|server|database)\b', text, re.IGNORECASE):
        return GoalRisk.MEDIUM, "Goal involves external systems"

    return GoalRisk.LOW, "No risk factors detected"


# ──────────────────────────────────────────────────────────────────────────────
# GOAL EXTRACTION PATTERNS
# ──────────────────────────────────────────────────────────────────────────────

_DEADLINE_PATTERNS = [
    r'\b(by|before|within|in)\s+([\w\s]+(?:hours?|days?|weeks?|minutes?|tomorrow|tonight|end of day))\b',
    r'\b(asap|immediately|urgent|as soon as possible)\b',
]

_CONSTRAINT_KEYWORDS = [
    "without", "do not", "don't", "avoid", "never", "no more than",
    "must not", "cannot", "only use", "only using", "limited to",
    "budget", "under $", "less than $",
]

_SUCCESS_PATTERNS = [
    r'\b(produce|create|write|generate|build|make|find|discover|deliver)\b.*\b(report|document|file|list|summary|plan|code|answer)\b',
    r'\b(answer|determine|figure out|calculate|find out)\b',
    r'\b(compare|analyze|evaluate|assess|rank)\b',
    r'\b(working|functional|complete|finished|done)\b',
]


def _extract_deadline(text: str) -> Optional[str]:
    for pattern in _DEADLINE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def _extract_constraints(text: str) -> List[str]:
    constraints = []
    sentences = re.split(r'[.;,]', text)
    for s in sentences:
        s_lower = s.lower()
        for kw in _CONSTRAINT_KEYWORDS:
            if kw in s_lower and len(s.strip()) > 5:
                constraints.append(s.strip())
                break
    return list(set(constraints))[:5]


def _infer_success_criteria(text: str, objective: str) -> List[str]:
    """Generate reasonable success criteria from the goal text."""
    criteria = []

    # Explicit criteria
    if re.search(r'\b(report|document|write-up|writeup|summary)\b', text, re.IGNORECASE):
        criteria.append("A written report or document is produced")
    if re.search(r'\b(comparison|compare|vs\.?|versus)\b', text, re.IGNORECASE):
        criteria.append("Items are compared with clear differences noted")
    if re.search(r'\b(code|script|program|function)\b', text, re.IGNORECASE):
        criteria.append("Working code is produced and tested")
    if re.search(r'\b(find|discover|identify|determine)\b', text, re.IGNORECASE):
        criteria.append("A clear answer or finding is stated")
    if re.search(r'\b(list|enumerate|catalog)\b', text, re.IGNORECASE):
        criteria.append("A complete list is produced")
    if re.search(r'\b(under|less than|budget|affordable|cheap)\b.*\$?\d+', text, re.IGNORECASE):
        criteria.append("Budget constraint is respected")

    # Always include
    criteria.append("Goal is fully completed — no partial outputs")
    criteria.append("All stated constraints are respected")

    return criteria[:5]


def _infer_resources(text: str) -> List[str]:
    """Identify which tools the goal likely needs."""
    resources = []
    text_l = text.lower()
    if any(k in text_l for k in ["research", "search", "find out", "current", "latest", "best"]):
        resources.append("web_search")
    if any(k in text_l for k in ["calculate", "compute", "math", "number", "price", "cost"]):
        resources.append("calculator")
    if any(k in text_l for k in ["write", "report", "document", "save", "file"]):
        resources.append("file_manager")
    if any(k in text_l for k in ["code", "script", "program", "python", "run"]):
        resources.append("code_executor")
    if any(k in text_l for k in ["summarize", "summary", "condense", "extract"]):
        resources.append("summarizer")
    resources.append("memory_tool")    # always available
    resources.append("task_manager")  # always available
    return resources


def _clarifications_needed(text: str, risk: GoalRisk) -> List[str]:
    """Identify what needs clarification before the agent can proceed."""
    questions = []

    # Vague scope
    if len(text.split()) < 5:
        questions.append("The goal is very short — could you provide more detail?")

    # Ambiguous targets
    if re.search(r'\ball\b|\beverything\b|\bthe system\b', text, re.IGNORECASE):
        questions.append("What specifically does 'all' / 'everything' refer to?")

    # Undefined budget
    if re.search(r'\baffordable|cheap|reasonable|budget\b', text, re.IGNORECASE) and \
       not re.search(r'\$\d+|\d+\s*dollars?', text, re.IGNORECASE):
        questions.append("What is the specific budget or price range?")

    # High-risk: always ask before proceeding
    if risk == GoalRisk.HIGH:
        questions.append(
            "This goal involves potentially irreversible or high-impact actions. "
            "Please confirm you want to proceed."
        )

    return questions


# ──────────────────────────────────────────────────────────────────────────────
# GOAL INTERPRETER
# ──────────────────────────────────────────────────────────────────────────────

class GoalInterpreter:
    """
    Converts raw natural language goals into structured GoalSpec objects.

    Pipeline:
      1. Constitutional safety check — unsafe goals are rejected immediately
      2. Extract structured fields (objective, criteria, constraints, deadline)
      3. Identify required resources / tools
      4. Flag ambiguities that need clarification
      5. Return GoalSpec ready for the planner

    Thread-safe: stateless (all state is in the returned GoalSpec).
    """

    def __init__(self):
        self._history: List[GoalSpec] = []

    def interpret(self, raw_goal: str, override_safety: bool = False) -> GoalSpec:
        """
        Parse and validate a natural language goal.

        Args:
            raw_goal:        The raw goal string from the user.
            override_safety: If True, allow HIGH-risk goals without extra questions.
                             UNSAFE goals are ALWAYS rejected regardless.

        Returns:
            GoalSpec with status=APPROVED, FLAGGED, or REJECTED.
        """
        raw_goal = raw_goal.strip()
        if not raw_goal:
            spec = GoalSpec(
                goal_id="empty",
                raw_goal="",
                objective="(empty goal)",
                success_criteria=[],
                constraints=[],
                deadline=None,
                resources=[],
                risk=GoalRisk.LOW,
                status=GoalStatus.REJECTED,
                clarifications_needed=["Goal cannot be empty. Please provide a goal."],
            )
            return spec

        # Generate stable ID from content
        goal_id = "G_" + hashlib.md5(raw_goal.encode()).hexdigest()[:8]

        # ── 1. Constitutional check ───────────────────────────────────────
        risk, risk_reason = _check_constitutional(raw_goal)
        logger.info(f"Goal {goal_id} risk: {risk.value} — {risk_reason}")

        if risk == GoalRisk.UNSAFE:
            spec = GoalSpec(
                goal_id=goal_id,
                raw_goal=raw_goal,
                objective=raw_goal[:200],
                success_criteria=[],
                constraints=[],
                deadline=None,
                resources=[],
                risk=GoalRisk.UNSAFE,
                status=GoalStatus.REJECTED,
                clarifications_needed=[
                    f"This goal was rejected by the constitutional safety filter: {risk_reason}. "
                    "I cannot help with this."
                ],
            )
            logger.warning(f"Goal REJECTED (unsafe): {raw_goal[:100]}")
            self._history.append(spec)
            return spec

        # ── 2. Extract fields ─────────────────────────────────────────────
        # Clean objective: first sentence or up to 200 chars
        first_sentence = re.split(r'[.!?]\s', raw_goal)[0].strip()
        objective = first_sentence[:200] if first_sentence else raw_goal[:200]

        deadline    = _extract_deadline(raw_goal)
        constraints = _extract_constraints(raw_goal)
        criteria    = _infer_success_criteria(raw_goal, objective)
        resources   = _infer_resources(raw_goal)
        questions   = _clarifications_needed(raw_goal, risk)

        # ── 3. Determine status ───────────────────────────────────────────
        if questions and not override_safety:
            status = GoalStatus.FLAGGED
        else:
            status = GoalStatus.APPROVED

        spec = GoalSpec(
            goal_id=goal_id,
            raw_goal=raw_goal,
            objective=objective,
            success_criteria=criteria,
            constraints=constraints,
            deadline=deadline,
            resources=resources,
            risk=risk,
            status=status,
            clarifications_needed=questions,
        )

        logger.info(
            f"Goal {goal_id} interpreted: status={status.value}, "
            f"risk={risk.value}, resources={resources}"
        )
        self._history.append(spec)
        return spec

    def clarify(self, spec: GoalSpec, answers: Dict[str, str]) -> GoalSpec:
        """
        Apply user answers to flagged clarifications and re-evaluate status.

        Args:
            spec:    A FLAGGED GoalSpec.
            answers: Mapping of question → answer from the user.

        Returns:
            Updated GoalSpec — APPROVED if all clarifications resolved.
        """
        if spec.status != GoalStatus.FLAGGED:
            return spec

        # Append answers as additional context
        extra = " ".join(f"{q}: {a}" for q, a in answers.items())
        augmented_goal = spec.raw_goal + " " + extra

        # Re-interpret with the augmented goal
        return self.interpret(augmented_goal, override_safety=True)

    def get_history(self) -> List[GoalSpec]:
        return list(self._history)

    def get_goal(self, goal_id: str) -> Optional[GoalSpec]:
        for g in self._history:
            if g.goal_id == goal_id:
                return g
        return None
