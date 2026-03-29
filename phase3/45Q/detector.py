"""
detector.py — Mode Detection Engine for 45Q Symbolic Logic Bridge.

Pure rule-based, regex-driven mode detector. No LLM. No neural network.
Deterministic. Target latency: < 30ms on any input ≤ 10,000 characters.

Detection hierarchy:
    MATH  (M) → mathematical expressions, equations, calculations
    LOGIC (L) → propositional/predicate logic, proofs, arguments
    CODE  (C) → Python code snippets requiring verification
    NATURAL(N) → everything else, passed through untouched
"""

from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from .response import Mode


# ── Compiled regex patterns (compiled once at import time) ─────────────────────

# Mathematical operators (Unicode + ASCII)
_RE_MATH_OPERATORS = re.compile(
    r'[=≠≤≥∫∑∏√^]|'
    r'\b(mod|modulo)\b',
    re.IGNORECASE | re.UNICODE,
)

# Mathematical keywords
_RE_MATH_KEYWORDS = re.compile(
    r'\b(solve|integrate|differentiate|derive|calculate|simplify|'
    r'factor(?:ise|ize)?|expand|limit|series|matrix|determinant|'
    r'eigenvalue|eigenvector|modulo|prime|factorial|'
    r'find\s+x|evaluate|compute|roots?\s+of|'
    r'gcd|lcm|totient|diophantine|'
    r'taylor|laurent|gradient|jacobian|hessian|'
    r'linear\s+algebra|linear\s+system|'
    r'prove\s+that\s+\d|'
    r'what\s+is\s+\d+\s*[\+\-\*\/\^])\b',
    re.IGNORECASE,
)

# Equation pattern: something = something (not == in code context)
_RE_EQUATION = re.compile(
    r'[a-zA-Z0-9\.\(\)]+\s*=\s*[a-zA-Z0-9\.\(\)]+',
)

# Function notation: f(x), g(x, y)
_RE_FUNCTION_NOTATION = re.compile(
    r'\b[a-zA-Z_]\w*\s*\([^)]*\)',
)

# Interval notation: [a,b] or (a,b)
_RE_INTERVAL = re.compile(
    r'[\[\(]-?\d[\d\s,\.]*-?\d[\]\)]',
)

# Set builder notation: {x | ...} or {x : ...}
_RE_SET_BUILDER = re.compile(
    r'\{[^}]+[\|:][^}]+\}',
)

# Integral / derivative expression forms
_RE_CALCULUS = re.compile(
    r'd[a-zA-Z]/d[a-zA-Z]|'         # dy/dx style
    r'∂[a-zA-Z]/∂[a-zA-Z]|'         # partial
    r'\\int\b|'                       # LaTeX \int
    r'\\sum\b|'                       # LaTeX \sum
    r'\\prod\b|'                      # LaTeX \prod
    r'\bintegral\b|\bderivative\b',
    re.IGNORECASE,
)

# ── Logic patterns ─────────────────────────────────────────────────────────────

_RE_LOGIC_SYMBOLS = re.compile(
    r'[→↔∧∨¬⊕∀∃⊢⊨]|'
    r'<->|->|<=>|=>',
    re.UNICODE,
)

_RE_LOGIC_KEYWORDS = re.compile(
    r'\b(tautology|satisfiable|contradiction|valid|entails|'
    r'implies|implication|equivalent|'
    r'modus\s+ponens|modus\s+tollens|syllogism|'
    r'for\s+all|there\s+exists|'
    r'counterexample|countermodel|'
    r'propositional|predicate\s+logic|'
    r'disjunctive|conjunctive|biconditional|'
    r'truth\s+table|cnf|dnf|'
    r'prove|disprove|refute)\b',
    re.IGNORECASE,
)

_RE_LOGIC_CONNECTIVES = re.compile(
    r'\b(AND|OR|NOT|IFF|XOR|NOR|NAND)\b|'
    r'\bIF\b.{1,80}\bTHEN\b',
    re.IGNORECASE,
)

# Structured argument: P1: ... P2: ... C: ...
_RE_ARGUMENT_STRUCTURE = re.compile(
    r'\b(P\d+|premise\s*\d*|conclusion|C\s*:)',
    re.IGNORECASE,
)

# ── Code patterns ──────────────────────────────────────────────────────────────

_RE_CODE_KEYWORDS = re.compile(
    r'^\s*(def |class |import |from |for |while |if |return |'
    r'with |try:|except|raise |assert |lambda )',
    re.MULTILINE,
)

_RE_CODE_QUERY_WORDS = re.compile(
    r'\b(verify|debug|test|find\s+the\s+bug|correct\s+this|'
    r'analyse|analyze|review|does\s+this\s+work|'
    r"what.s\s+wrong|is\s+this\s+correct|"
    r'check\s+this\s+(function|code|script)|'
    r'bug|error\s+in|fix\s+this)\b',
    re.IGNORECASE,
)

_RE_BACKTICK_CODE = re.compile(
    r'`{1,3}[^`]+`{1,3}',
    re.DOTALL,
)

_RE_INDENTED_BLOCK = re.compile(
    r'(?m)^(    |\t)[^\n]+\n(?:(?:    |\t)[^\n]+\n)*',
)


@dataclass
class DetectionResult:
    """
    Outcome of mode detection for a single input string.

    Attributes
    ----------
    mode : Mode
        The detected mode (MATH, LOGIC, CODE, NATURAL).
    scores : dict[str, float]
        Keyword density scores for each candidate mode.
    rationale : str
        Human-readable explanation of why this mode was chosen.
    elapsed_ms : float
        How long detection took in milliseconds.
    debug_log : list[str]
        Detailed trace of each rule that fired.
    """
    mode: Mode
    scores: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    elapsed_ms: float = 0.0
    debug_log: list[str] = field(default_factory=list)


def detect(text: str) -> DetectionResult:
    """
    Detect the mode of a natural language or code query.

    Runs purely rule-based pattern matching. Never calls an LLM.
    Designed to complete in < 30ms for inputs ≤ 10,000 characters.

    Parameters
    ----------
    text : str
        The raw input query from the user or calling system.

    Returns
    -------
    DetectionResult
        Detected mode with confidence scores and rationale.

    Notes
    -----
    Ambiguity resolution:
      - M + L: pick higher keyword density score
      - M + C: run Code first, then extract math sub-queries
      - All decisions logged to debug_log
    """
    t0 = time.perf_counter()
    log: list[str] = []

    # Normalise whitespace but preserve structure
    normalised = text.strip()

    m_score = _score_math(normalised, log)
    l_score = _score_logic(normalised, log)
    c_score = _score_code(normalised, log)

    log.append(f"Scores → MATH={m_score:.3f}  LOGIC={l_score:.3f}  CODE={c_score:.3f}")

    # Determine primary mode
    mode, rationale = _resolve(m_score, l_score, c_score, log)

    elapsed = (time.perf_counter() - t0) * 1000
    log.append(f"Detection completed in {elapsed:.2f} ms → {mode.value}")

    return DetectionResult(
        mode=mode,
        scores={"MATH": m_score, "LOGIC": l_score, "CODE": c_score},
        rationale=rationale,
        elapsed_ms=elapsed,
        debug_log=log,
    )


# ── Scoring functions ──────────────────────────────────────────────────────────

def _score_math(text: str, log: list[str]) -> float:
    """
    Compute a keyword-density score for Mode MATH.

    Returns a float where higher → more likely to be a math query.
    Scores are additive hits normalised by text length (chars/100).
    """
    score = 0.0
    length_factor = max(1, len(text) / 100)

    # Operator presence — each unique match type scores 2 pts
    op_matches = _RE_MATH_OPERATORS.findall(text)
    if op_matches:
        pts = min(len(op_matches), 5) * 2.0
        score += pts
        log.append(f"  MATH: operators found ({len(op_matches)}) +{pts:.1f}")

    # Keyword hits — each keyword 3 pts
    kw_matches = _RE_MATH_KEYWORDS.findall(text)
    if kw_matches:
        pts = min(len(kw_matches), 4) * 3.0
        score += pts
        log.append(f"  MATH: keywords {kw_matches[:3]} +{pts:.1f}")

    # Equation pattern — 4 pts
    if _RE_EQUATION.search(text):
        score += 4.0
        log.append("  MATH: equation pattern detected +4.0")

    # Calculus notation — 5 pts
    if _RE_CALCULUS.search(text):
        score += 5.0
        log.append("  MATH: calculus notation detected +5.0")

    # Interval or set notation — 2 pts each
    if _RE_INTERVAL.search(text):
        score += 2.0
        log.append("  MATH: interval notation +2.0")
    if _RE_SET_BUILDER.search(text):
        score += 2.0
        log.append("  MATH: set builder notation +2.0")

    # Explicit numeric expression (e.g. 2^10, 3*x+1)
    if re.search(r'\d\s*[\+\-\*\/\^]\s*\d', text):
        score += 3.0
        log.append("  MATH: numeric arithmetic expression +3.0")

    # Normalise by text length to get density
    return score / length_factor


def _score_logic(text: str, log: list[str]) -> float:
    """
    Compute a keyword-density score for Mode LOGIC.
    """
    score = 0.0
    length_factor = max(1, len(text) / 100)

    if _RE_LOGIC_SYMBOLS.search(text):
        pts = 6.0
        score += pts
        log.append(f"  LOGIC: symbolic connectives/quantifiers +{pts:.1f}")

    kw_matches = _RE_LOGIC_KEYWORDS.findall(text)
    if kw_matches:
        pts = min(len(kw_matches), 4) * 3.5
        score += pts
        log.append(f"  LOGIC: keywords {kw_matches[:3]} +{pts:.1f}")

    conn_matches = _RE_LOGIC_CONNECTIVES.findall(text)
    if conn_matches:
        pts = min(len(conn_matches), 3) * 2.5
        score += pts
        log.append(f"  LOGIC: connectives {conn_matches[:3]} +{pts:.1f}")

    if _RE_ARGUMENT_STRUCTURE.search(text):
        score += 4.0
        log.append("  LOGIC: structured argument pattern +4.0")

    # Numbered proof lines pattern: "1. P [Premise]"
    if re.search(r'^\s*\d+\.\s+\S.+\[', text, re.MULTILINE):
        score += 6.0
        log.append("  LOGIC: numbered proof steps pattern +6.0")

    return score / length_factor


def _score_code(text: str, log: list[str]) -> float:
    """
    Compute a keyword-density score for Mode CODE.
    """
    score = 0.0
    length_factor = max(1, len(text) / 100)

    kw_matches = _RE_CODE_KEYWORDS.findall(text)
    if kw_matches:
        pts = min(len(kw_matches), 5) * 4.0
        score += pts
        log.append(f"  CODE: Python keywords ({len(kw_matches)}) +{pts:.1f}")

    if _RE_CODE_QUERY_WORDS.search(text):
        score += 5.0
        log.append("  CODE: code-review query words +5.0")

    if _RE_BACKTICK_CODE.search(text):
        score += 4.0
        log.append("  CODE: backtick code block +4.0")

    if _RE_INDENTED_BLOCK.search(text):
        score += 3.0
        log.append("  CODE: indented code block +3.0")

    # Function signature
    if re.search(r'\bdef\s+\w+\s*\(', text):
        score += 5.0
        log.append("  CODE: function definition +5.0")

    return score / length_factor


def _resolve(
    m: float,
    l: float,
    c: float,
    log: list[str],
) -> tuple[Mode, str]:
    """
    Resolve competing scores into a single mode and rationale string.

    Ambiguity rules (from spec):
      M + L  → pick higher score
      M + C  → MODE_CODE (C subsumes M for code queries containing math)
      all low → NATURAL

    Parameters
    ----------
    m, l, c : float
        Scores for MATH, LOGIC, CODE modes.
    log : list[str]
        Mutable log to append resolution trace.

    Returns
    -------
    tuple[Mode, str]
        Selected mode and rationale string.
    """
    THRESHOLD = 1.5  # Minimum score to trigger a non-NATURAL mode

    m_active = m >= THRESHOLD
    l_active = l >= THRESHOLD
    c_active = c >= THRESHOLD

    log.append(f"  Active flags → MATH={m_active} LOGIC={l_active} CODE={c_active}")

    if not m_active and not l_active and not c_active:
        return Mode.NATURAL, "No domain-specific patterns detected; treating as natural language."

    # CODE wins over MATH when both active (spec: run C first, extract math)
    if c_active and m_active and not l_active:
        log.append("  M+C conflict → CODE wins (will extract math sub-queries)")
        return Mode.CODE, (
            f"CODE ({c:.2f}) + MATH ({m:.2f}) both triggered. "
            "Routing to Code engine; math sub-expressions will be extracted."
        )

    # M + L: pick higher density
    if m_active and l_active and not c_active:
        if m >= l:
            log.append(f"  M+L conflict → MATH wins ({m:.2f} >= {l:.2f})")
            return Mode.MATH, (
                f"MATH ({m:.2f}) and LOGIC ({l:.2f}) both triggered. "
                "MATH score higher → routing to Math engine."
            )
        else:
            log.append(f"  M+L conflict → LOGIC wins ({l:.2f} > {m:.2f})")
            return Mode.LOGIC, (
                f"MATH ({m:.2f}) and LOGIC ({l:.2f}) both triggered. "
                "LOGIC score higher → routing to Logic engine."
            )

    # All three active: highest wins
    if m_active and l_active and c_active:
        best = max([(m, Mode.MATH), (l, Mode.LOGIC), (c, Mode.CODE)], key=lambda x: x[0])
        log.append(f"  All three active → {best[1].value} wins ({best[0]:.2f})")
        return best[1], f"All three modes triggered; {best[1].value} has highest score ({best[0]:.2f})."

    # Single mode active
    if c_active:
        return Mode.CODE, f"CODE patterns detected (score={c:.2f})."
    if m_active:
        return Mode.MATH, f"MATH patterns detected (score={m:.2f})."
    if l_active:
        return Mode.LOGIC, f"LOGIC patterns detected (score={l:.2f})."

    return Mode.NATURAL, "Fallback: no strong signal detected."
