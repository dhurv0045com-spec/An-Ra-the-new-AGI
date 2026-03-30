"""
self_check.py — Dual-Pass Self-Check Layer for 45Q Symbolic Logic Bridge.

Every result from Mode MATH and Mode LOGIC passes through this layer
before being returned to the caller. It is not optional and cannot
be bypassed.

For MATH:
  Pass A → SymPy symbolic result
  Pass B → scipy/numpy numerical evaluation at NUMERIC_VERIFY_POINTS points
  Confidence rules:
    |A - B| < NUMERIC_DELTA_THRESHOLD at all points → 1.0
    delta > threshold at any point → 0.85, flag UNCERTAIN
    Pass B fails to evaluate → 0.90, flag SYMBOLIC_ONLY

For LOGIC:
  Pass A → Truth table verdict (or DPLL for large formulas)
  Pass B → DPLL SAT solver verdict
  Confidence rules:
    Both agree → 1.0
    Disagree → 0.0, flag CONTRADICTION_IN_VERIFIERS, full trace logged
"""

from __future__ import annotations
import random
from typing import Any, Optional

import sympy as sp
import numpy as np

from .response import (
    Mode, Verdict, VerifiedResult, VerificationPass, error_result
)
from . import config


def self_check_math(result: VerifiedResult, expr: Any, var_symbol: Any) -> VerifiedResult:
    """
    Apply the dual-pass self-check to a MATH result.

    Numerically evaluates the symbolic answer at NUMERIC_VERIFY_POINTS
    random test points and compares with scipy/numpy evaluation of the
    original expression.

    Parameters
    ----------
    result : VerifiedResult
        The Math engine result to check.
    expr : Any
        The original SymPy expression (integrand, equation, etc.).
    var_symbol : Any
        The primary SymPy Symbol (e.g., x).

    Returns
    -------
    VerifiedResult
        Updated result with self-check applied (confidence may decrease).
    """
    if result.verdict == Verdict.ERROR:
        return result

    # If symbolic answer is a single number (definite integral, etc.),
    # we verify differently
    answer = result.answer
    if answer is None:
        return result

    try:
        # Generate random test points avoiding 0 (possible singularity)
        test_points = [random.uniform(0.1, 2.0) for _ in range(config.NUMERIC_VERIFY_POINTS)]
        deltas: list[float] = []
        sc_pass_details: list[str] = []

        for pt in test_points:
            try:
                sym_val = complex(sp.N(answer.subs(var_symbol, pt)))
                orig_val = complex(sp.N(expr.subs(var_symbol, pt)))
                delta = abs(sym_val - orig_val)
                deltas.append(delta)
                sc_pass_details.append(
                    f"x={pt:.3f}: answer_val={sym_val.real:.6f}, expr_val={orig_val.real:.6f}, δ={delta:.2e}"
                )
            except Exception as e:
                sc_pass_details.append(f"x={pt:.3f}: evaluation failed ({e})")

        if not deltas:
            # Numeric check couldn't run
            result.confidence = min(result.confidence, 0.90)
            result.warnings.append("Numeric self-check could not evaluate at test points")
            result.passes.append(VerificationPass(
                method="Self-check: numeric eval",
                result="FAILED",
                success=False,
                error="No test points evaluated",
            ))
            if result.confidence < config.CONFIDENCE_THRESHOLD:
                result.verdict = Verdict.SYMBOLIC_ONLY
        else:
            max_delta = max(deltas)
            result.delta = max_delta
            result.steps.extend(["Self-check numerical evaluation:"] + sc_pass_details)

            if max_delta < config.NUMERIC_DELTA_THRESHOLD:
                confidence = 1.0
                result.passes.append(VerificationPass(
                    method=f"Self-check: numeric eval at {len(deltas)} points",
                    result=f"max_δ={max_delta:.2e} ✓",
                    success=True,
                ))
            else:
                confidence = 0.85
                result.warnings.append(
                    f"Symbolic and numeric results differ: max δ = {max_delta:.2e} "
                    f"(threshold = {config.NUMERIC_DELTA_THRESHOLD}). "
                    "Possible branch cut, singularity, or integration constant."
                )
                result.passes.append(VerificationPass(
                    method=f"Self-check: numeric eval at {len(deltas)} points",
                    result=f"max_δ={max_delta:.2e} ✗ (exceeds threshold)",
                    success=False,
                ))

            result.confidence = min(result.confidence, confidence)
            if result.confidence < config.CONFIDENCE_THRESHOLD:
                result.verdict = Verdict.UNCERTAIN

    except Exception as e:
        result.warnings.append(f"Self-check failed with error: {e}")

    return result


def self_check_logic(
    pass_a_verdict: str,
    pass_b_verdict: str,
    pass_a: VerificationPass,
    pass_b: VerificationPass,
    raw_input: str,
    trace_a: list[str],
    trace_b: list[str],
) -> VerifiedResult:
    """
    Apply the dual-pass self-check to a LOGIC result.

    Compares the Truth Table verdict (Pass A) with DPLL verdict (Pass B).
    If they disagree, returns confidence=0.0 and flags CONTRADICTION_IN_VERIFIERS.

    Parameters
    ----------
    pass_a_verdict : str
        Verdict from Pass A (Truth Table or primary DPLL).
    pass_b_verdict : str
        Verdict from Pass B (DPLL verifier).
    pass_a : VerificationPass
        Pass A details.
    pass_b : VerificationPass
        Pass B details.
    raw_input : str
        The original formula string.
    trace_a : list[str]
        Full trace from Pass A.
    trace_b : list[str]
        Full trace from Pass B.

    Returns
    -------
    VerifiedResult
        Cross-checked result.
    """
    from .response import Verdict as V

    if pass_a_verdict == pass_b_verdict:
        # Happy path: both agree
        verdict_map = {
            "TAUTOLOGY": V.TAUTOLOGY,
            "CONTRADICTION": V.CONTRADICTION,
            "SATISFIABLE": V.SATISFIABLE,
            "SAT": V.SAT,
            "UNSAT": V.UNSAT,
        }
        verdict = verdict_map.get(pass_a_verdict, V.VERIFIED)
        return VerifiedResult(
            mode=Mode.LOGIC,
            verdict=verdict,
            confidence=1.0,
            answer=pass_a_verdict,
            answer_text=pass_a_verdict,
            raw_input=raw_input,
            steps=trace_a + ["---", "PASS B CONFIRMATION:"] + trace_b,
            passes=[pass_a, pass_b],
        )
    else:
        # Disagreement: CONTRADICTION_IN_VERIFIERS
        debug_trace = (
            ["=== PASS A TRACE ==="] + trace_a +
            ["=== PASS B TRACE ==="] + trace_b
        )
        return VerifiedResult(
            mode=Mode.LOGIC,
            verdict=V.UNCERTAIN,
            confidence=0.0,
            answer=None,
            answer_text=(
                f"CONTRADICTION_IN_VERIFIERS: "
                f"Pass A={pass_a_verdict}, Pass B={pass_b_verdict}. "
                "Manual review required."
            ),
            raw_input=raw_input,
            steps=debug_trace,
            passes=[pass_a, pass_b],
            warnings=[
                f"Verifiers disagree: Truth Table={pass_a_verdict}, DPLL={pass_b_verdict}",
                "This should never happen; indicates a bug in 45Q itself.",
                "Manual verification step: enumerate all rows by hand.",
            ],
            debug_log=debug_trace,
        )


def self_check_uncertain_response(result: VerifiedResult) -> VerifiedResult:
    """
    Format a result below CONFIDENCE_THRESHOLD as a proper UNCERTAIN response.

    UNCERTAIN responses must include:
    - Both results side by side
    - The delta or disagreement
    - Which method is more likely correct and why
    - A suggested manual verification step for the user

    Parameters
    ----------
    result : VerifiedResult
        A result with confidence below CONFIDENCE_THRESHOLD.

    Returns
    -------
    VerifiedResult
        Same result, updated with proper UNCERTAIN formatting.
    """
    if result.confidence >= config.CONFIDENCE_THRESHOLD:
        return result

    result.verdict = Verdict.UNCERTAIN

    # Add standard UNCERTAIN guidance
    guidance = []

    if result.passes:
        guidance.append("Both method results:")
        for p in result.passes:
            status = "✓" if p.success else "✗"
            guidance.append(f"  {status} [{p.method}]: {p.result}")

    if result.delta is not None:
        guidance.append(f"Absolute difference: {result.delta:.2e}")
        if len(result.passes) >= 2:
            a, b = result.passes[0], result.passes[1]
            if a.success and not b.success:
                guidance.append(f"More likely correct: [{a.method}] (Pass B failed to evaluate)")
            elif b.success and not a.success:
                guidance.append(f"More likely correct: [{b.method}] (Pass A returned an error)")
            else:
                guidance.append(
                    "Cannot determine which pass is more reliable automatically. "
                    "The symbolic result is typically more exact if the function is analytic."
                )

    guidance.append("Suggested manual verification:")
    if result.mode == Mode.MATH:
        guidance.append("  - Substitute the solution into the original equation manually")
        guidance.append("  - Use a CAS (WolframAlpha, Mathematica) as a third check")
    elif result.mode == Mode.LOGIC:
        guidance.append("  - Enumerate truth table rows manually for small formulas")
        guidance.append("  - Check the CNF conversion by hand")

    result.steps.extend(["--- UNCERTAIN GUIDANCE ---"] + guidance)
    return result
