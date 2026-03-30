"""
response.py — Unified response envelope for 45Q Symbolic Logic Bridge.

Every engine returns a VerifiedResult. This ensures callers always receive
a consistent structure: mode, confidence, primary answer, verification
status, steps, and any warnings.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Mode(str, Enum):
    MATH = "MATH"
    LOGIC = "LOGIC"
    CODE = "CODE"
    NATURAL = "NATURAL"
    UNKNOWN = "UNKNOWN"


class Verdict(str, Enum):
    VERIFIED = "VERIFIED"           # Both passes agree, confidence = 1.0
    SYMBOLIC_ONLY = "SYMBOLIC_ONLY" # Numeric check could not run
    UNCERTAIN = "UNCERTAIN"         # Passes disagree or delta > threshold
    UNSAT = "UNSAT"                 # SAT: formula unsatisfiable
    SAT = "SAT"                     # SAT: formula satisfiable
    TAUTOLOGY = "TAUTOLOGY"         # Logic: always true
    CONTRADICTION = "CONTRADICTION" # Logic: always false
    SATISFIABLE = "SATISFIABLE"     # Logic: sometimes true
    VALID_PROOF = "VALID_PROOF"     # Natural deduction: proof correct
    INVALID_PROOF = "INVALID_PROOF" # Natural deduction: error found
    BUG_FREE = "BUG_FREE"           # Code: no issues found
    BUGS_FOUND = "BUGS_FOUND"       # Code: issues detected
    ERROR = "ERROR"                 # Engine internal error


@dataclass
class VerificationPass:
    """One of the two verification passes in the dual-pass layer."""
    method: str                     # e.g. "SymPy symbolic", "scipy.quad numeric"
    result: Any                     # Raw result value
    success: bool                   # Did this pass complete without error?
    error: Optional[str] = None     # If not success, what went wrong


@dataclass
class VerifiedResult:
    """
    The single return type for every engine in 45Q.

    Parameters
    ----------
    mode : Mode
        Which engine produced this result.
    verdict : Verdict
        High-level classification of the outcome.
    confidence : float
        0.0–1.0. Below config.CONFIDENCE_THRESHOLD → UNCERTAIN.
    answer : Any
        The primary answer (expression, bool, list of bugs, etc.).
    answer_latex : str
        LaTeX representation of the answer where applicable.
    answer_text : str
        Plain-text representation of the answer.
    steps : list[str]
        Ordered list of intermediate reasoning/computation steps.
    passes : list[VerificationPass]
        The two verification passes and their outcomes.
    warnings : list[str]
        Non-fatal issues, caveats, or flagged edge cases.
    counterexample : Any
        For logic: a falsifying assignment; for code: a failing test.
    raw_input : str
        The original input as received.
    parsed_repr : str
        The parsed representation (expression tree, AST dump, etc.).
    delta : Optional[float]
        Absolute difference between symbolic and numeric passes.
    debug_log : list[str]
        Internal trace messages for debugging.
    """
    mode: Mode
    verdict: Verdict
    confidence: float
    answer: Any = None
    answer_latex: str = ""
    answer_text: str = ""
    steps: list[str] = field(default_factory=list)
    passes: list[VerificationPass] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counterexample: Any = None
    raw_input: str = ""
    parsed_repr: str = ""
    delta: Optional[float] = None
    debug_log: list[str] = field(default_factory=list)

    # ── Display helpers ────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a compact human-readable summary of this result."""
        lines = [
            f"[45Q | Mode={self.mode.value} | Verdict={self.verdict.value} | "
            f"Confidence={self.confidence:.2f}]",
            f"Answer: {self.answer_text or str(self.answer)}",
        ]
        if self.answer_latex:
            lines.append(f"LaTeX:  {self.answer_latex}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"⚠  {w}")
        if self.verdict == Verdict.UNCERTAIN:
            lines.append("⚡ UNCERTAIN — see passes for detail")
            for p in self.passes:
                lines.append(f"   [{p.method}] → {p.result}")
            if self.delta is not None:
                lines.append(f"   delta = {self.delta:.2e}")
        return "\n".join(lines)

    def full_report(self) -> str:
        """Return a detailed, structured report of this result."""
        sep = "─" * 72
        lines = [
            sep,
            f"45Q RESULT  mode={self.mode.value}  verdict={self.verdict.value}",
            sep,
        ]
        if self.raw_input:
            lines += ["INPUT:", f"  {self.raw_input}"]
        if self.parsed_repr:
            lines += ["PARSED:", f"  {self.parsed_repr}"]
        if self.steps:
            lines += ["STEPS:"]
            for i, s in enumerate(self.steps, 1):
                lines.append(f"  {i:>3}. {s}")
        lines += [
            "ANSWER:",
            f"  {self.answer_text or str(self.answer)}",
        ]
        if self.answer_latex:
            lines.append(f"  LaTeX: {self.answer_latex}")
        lines += [
            "VERIFICATION:",
            f"  Confidence : {self.confidence:.4f}",
        ]
        for p in self.passes:
            status = "✓" if p.success else "✗"
            lines.append(f"  {status} [{p.method}] → {p.result}")
            if p.error:
                lines.append(f"      Error: {p.error}")
        if self.delta is not None:
            lines.append(f"  Delta      : {self.delta:.2e}")
        if self.counterexample is not None:
            lines += ["COUNTEREXAMPLE:", f"  {self.counterexample}"]
        if self.warnings:
            lines += ["WARNINGS:"]
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")
        lines.append(sep)
        return "\n".join(lines)


def error_result(mode: Mode, raw_input: str, error_msg: str) -> VerifiedResult:
    """
    Construct a VerifiedResult representing an internal engine error.

    Parameters
    ----------
    mode : Mode
        The engine mode that encountered the error.
    raw_input : str
        The original input that triggered the error.
    error_msg : str
        Description of what went wrong.

    Returns
    -------
    VerifiedResult
        Result with verdict=ERROR, confidence=0.0.
    """
    return VerifiedResult(
        mode=mode,
        verdict=Verdict.ERROR,
        confidence=0.0,
        answer_text=f"ERROR: {error_msg}",
        raw_input=raw_input,
        warnings=[error_msg],
    )
