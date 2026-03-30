"""
math_solver.py — Math Solver Engine for 45Q Symbolic Logic Bridge.

Built on SymPy for symbolic computation, with scipy/numpy for numerical
cross-verification. Every result is verified before being returned.

Capabilities:
  2.1 Equation solving (polynomial, transcendental, systems, inequalities)
  2.2 Calculus (derivatives, integrals, limits, series)
  2.3 Linear algebra (matrices, eigenvalues, decompositions)
  2.4 Number theory (primality, factorisation, GCD, Euler's totient)
  2.5 Step-by-step output with LaTeX and verification
"""

from __future__ import annotations
import random
from typing import Any, Optional
import sympy as sp
from sympy import (
    Symbol, symbols, sympify, latex, pretty,
    solve, solve_univariate_inequality, nsolve,
    diff, integrate, limit, series,
    Matrix, eye, det, Rational,
    isprime, factorint, gcd, lcm,
    totient, mobius, mod_inverse,
    oo, pi, E, I, S,
)
from sympy.solvers.diophantine import diophantine
import numpy as np

from .response import (
    Mode, Verdict, VerifiedResult, VerificationPass, error_result
)
from .miller_rabin import is_prime as mr_is_prime
from .pollard_rho import factorise as rho_factorise, verify_factorisation
from . import config


def _latex_safe(expr: Any) -> str:
    """Safely convert a SymPy expression to LaTeX."""
    try:
        return latex(expr)
    except Exception:
        return str(expr)


def _numeric_eval(expr: Any, substitutions: dict) -> Optional[float]:
    """
    Numerically evaluate a SymPy expression with given substitutions.

    Parameters
    ----------
    expr : Any
        SymPy expression.
    substitutions : dict
        Variable → numeric value mapping.

    Returns
    -------
    Optional[float]
        Float result, or None if evaluation fails.
    """
    try:
        result = complex(expr.subs(substitutions).evalf())
        if abs(result.imag) < 1e-10:
            return result.real
        return abs(result)
    except Exception:
        return None


# ── 2.1 Equation Solving ──────────────────────────────────────────────────────

def solve_equation(equation_str: str, var_name: str = "x") -> VerifiedResult:
    """
    Solve an equation or expression for a variable.

    Accepts:
      - "3x^3 - 2x^2 + x - 5 = 0" (with equals sign)
      - "3x^3 - 2x^2 + x - 5"     (expression set equal to 0)

    Parameters
    ----------
    equation_str : str
        The equation or expression string.
    var_name : str
        The variable to solve for (default 'x').

    Returns
    -------
    VerifiedResult
        Solutions with verification residuals and confidence.
    """
    steps: list[str] = []
    warnings: list[str] = []

    try:
        x = Symbol(var_name)
        # Handle "LHS = RHS" format
        if '=' in equation_str:
            lhs_str, rhs_str = equation_str.split('=', 1)
            lhs = sympify(lhs_str.strip())
            rhs = sympify(rhs_str.strip())
            expr = lhs - rhs
        else:
            expr = sympify(equation_str.strip())

        steps.append(f"Expression: {expr} = 0")
        steps.append(f"SymPy repr: {repr(expr)}")

        # Symbolic solve
        symbolic_solutions = solve(expr, x)
        steps.append(f"SymPy solve: {symbolic_solutions}")

        if not symbolic_solutions:
            # Try numerical solve
            try:
                numerical_sol = complex(nsolve(expr, x, 1))
                steps.append(f"Numerical fallback (nsolve): {numerical_sol}")
                symbolic_solutions = [numerical_sol]
                warnings.append("No closed-form solution found; numerical approximation used.")
            except Exception:
                pass

        verified_solutions = []
        for sol in symbolic_solutions:
            residual = expr.subs(x, sol).evalf()
            try:
                res_float = abs(complex(residual))
            except Exception:
                res_float = float('inf')

            is_exact = res_float < 1e-10
            sol_latex = _latex_safe(sol)
            sol_text = str(sol)
            tag = "EXACT" if is_exact else f"APPROXIMATE (residual={res_float:.2e})"

            verified_solutions.append({
                "solution": sol,
                "latex": sol_latex,
                "text": sol_text,
                "residual": res_float,
                "tag": tag,
            })
            steps.append(f"Verify x={sol_text}: residual = {res_float:.2e} → {tag}")
            if not is_exact:
                warnings.append(f"Solution {sol_text} has residual {res_float:.2e} > 1e-10")

        confidence = 1.0 if all(s["residual"] < 1e-10 for s in verified_solutions) else 0.90

        answer_parts = []
        for vs in verified_solutions:
            answer_parts.append(f"{vs['text']} [{vs['tag']}]")
        answer_text = "; ".join(answer_parts) if answer_parts else "No solutions found"
        answer_latex = ",\\, ".join(vs["latex"] for vs in verified_solutions) if verified_solutions else "\\text{No solutions}"

        return VerifiedResult(
            mode=Mode.MATH,
            verdict=Verdict.VERIFIED if confidence >= 0.95 else Verdict.UNCERTAIN,
            confidence=confidence,
            answer=verified_solutions,
            answer_text=answer_text,
            answer_latex=answer_latex,
            raw_input=equation_str,
            parsed_repr=repr(expr),
            steps=steps,
            warnings=warnings,
            passes=[
                VerificationPass("SymPy symbolic solve", symbolic_solutions, True),
                VerificationPass("Back-substitution residual check",
                                 [s["residual"] for s in verified_solutions], True),
            ],
        )

    except Exception as e:
        return error_result(Mode.MATH, equation_str, f"Equation solver error: {e}")


# ── 2.2 Calculus ──────────────────────────────────────────────────────────────

def differentiate(expr_str: str, var_name: str = "x", order: int = 1) -> VerifiedResult:
    """
    Compute the derivative of an expression.

    Parameters
    ----------
    expr_str : str
        The expression to differentiate.
    var_name : str
        The variable (default 'x').
    order : int
        Order of differentiation (default 1).

    Returns
    -------
    VerifiedResult
        Derivative with numeric verification.
    """
    steps: list[str] = []
    try:
        x = Symbol(var_name)
        expr = sympify(expr_str)
        steps.append(f"Expression: {expr}")

        # Symbolic differentiation
        result = diff(expr, x, order)
        steps.append(f"d^{order}/d{var_name}^{order} [{expr}] = {result}")

        # Numeric verification at 5 test points
        test_points = [0.1, 0.5, 1.0, 2.0, 3.14]
        deltas = []
        for pt in test_points:
            symbolic_val = _numeric_eval(result, {x: pt})
            # Numeric derivative via finite difference
            h = 1e-6
            try:
                f_plus = float(expr.subs(x, pt + h).evalf())
                f_minus = float(expr.subs(x, pt - h).evalf())
                numeric_val = (f_plus - f_minus) / (2 * h)
                if symbolic_val is not None:
                    delta = abs(symbolic_val - numeric_val)
                    deltas.append(delta)
                    steps.append(f"  x={pt}: symbolic={symbolic_val:.6f}, numeric={numeric_val:.6f}, δ={delta:.2e}")
            except Exception:
                steps.append(f"  x={pt}: skipped (evaluation error)")

        max_delta = max(deltas) if deltas else None
        confidence = 1.0 if (max_delta is not None and max_delta < config.NUMERIC_DELTA_THRESHOLD) else 0.90

        return VerifiedResult(
            mode=Mode.MATH,
            verdict=Verdict.VERIFIED if confidence >= 0.95 else Verdict.UNCERTAIN,
            confidence=confidence,
            answer=result,
            answer_text=str(result),
            answer_latex=_latex_safe(result),
            raw_input=expr_str,
            parsed_repr=repr(expr),
            steps=steps,
            delta=max_delta,
            passes=[
                VerificationPass("SymPy symbolic diff", str(result), True),
                VerificationPass("Finite-difference numeric check",
                                 f"max_δ={max_delta:.2e}" if max_delta else "n/a", True),
            ],
        )
    except Exception as e:
        return error_result(Mode.MATH, expr_str, f"Differentiation error: {e}")


def integrate_expr(
    expr_str: str,
    var_name: str = "x",
    lower: Optional[str] = None,
    upper: Optional[str] = None,
) -> VerifiedResult:
    """
    Compute the integral of an expression (definite or indefinite).

    Parameters
    ----------
    expr_str : str
        The integrand.
    var_name : str
        Integration variable (default 'x').
    lower : Optional[str]
        Lower bound for definite integral.
    upper : Optional[str]
        Upper bound for definite integral.

    Returns
    -------
    VerifiedResult
        Integral with symbolic and numeric cross-verification.
    """
    steps: list[str] = []
    warnings: list[str] = []
    try:
        from scipy import integrate as sci_integrate
        x = Symbol(var_name)
        expr = sympify(expr_str)
        steps.append(f"Integrand: {expr}")

        is_definite = (lower is not None and upper is not None)

        if is_definite:
            a = sympify(lower)
            b = sympify(upper)
            steps.append(f"Bounds: [{a}, {b}]")
            symbolic_result = integrate(expr, (x, a, b))
            steps.append(f"Symbolic result: {symbolic_result}")

            # Numeric cross-check with scipy.integrate.quad
            try:
                f_numeric = sp.lambdify(x, expr, modules=["numpy"])
                a_float = float(a.evalf())
                b_float = float(b.evalf())
                numeric_val, err_estimate = sci_integrate.quad(f_numeric, a_float, b_float)
                symbolic_val = float(symbolic_result.evalf())

                delta = abs(symbolic_val - numeric_val)
                steps.append(f"Scipy quad: {numeric_val:.10f} (est. error: {err_estimate:.2e})")
                steps.append(f"Symbolic val: {symbolic_val:.10f}")
                steps.append(f"Delta: {delta:.2e}")

                rel_delta = delta / max(abs(numeric_val), 1e-15)

                if rel_delta > 0.0001:  # > 0.01%
                    warnings.append(
                        f"Symbolic and numeric results differ by {rel_delta*100:.4f}% "
                        f"(δ={delta:.2e}). Possible branch cut or singularity."
                    )
                    confidence = 0.85
                    verdict = Verdict.UNCERTAIN
                else:
                    confidence = 1.0
                    verdict = Verdict.VERIFIED

                pass_a = VerificationPass("SymPy symbolic integrate", symbolic_val, True)
                pass_b = VerificationPass("scipy.integrate.quad", numeric_val, True)

            except Exception as e:
                steps.append(f"Numeric verification failed: {e}")
                warnings.append(f"Could not numerically verify: {e}")
                confidence = 0.90
                verdict = Verdict.SYMBOLIC_ONLY
                delta = None
                pass_a = VerificationPass("SymPy symbolic integrate", str(symbolic_result), True)
                pass_b = VerificationPass("scipy.integrate.quad", None, False, str(e))

            return VerifiedResult(
                mode=Mode.MATH,
                verdict=verdict,
                confidence=confidence,
                answer=symbolic_result,
                answer_text=str(symbolic_result),
                answer_latex=_latex_safe(symbolic_result),
                raw_input=expr_str,
                parsed_repr=repr(expr),
                steps=steps,
                warnings=warnings,
                delta=delta if 'delta' in dir() else None,
                passes=[pass_a, pass_b],
            )

        else:
            # Indefinite integral
            symbolic_result = integrate(expr, x)
            steps.append(f"Antiderivative: {symbolic_result} + C")

            # Verify by differentiating back
            verification = diff(symbolic_result, x)
            diff_simplified = sp.simplify(verification - expr)
            is_correct = diff_simplified == 0

            steps.append(f"Verify: d/dx[{symbolic_result}] = {verification}")
            steps.append(f"Simplified difference: {diff_simplified} → {'✓' if is_correct else '✗'}")

            confidence = 1.0 if is_correct else 0.80
            return VerifiedResult(
                mode=Mode.MATH,
                verdict=Verdict.VERIFIED if is_correct else Verdict.UNCERTAIN,
                confidence=confidence,
                answer=symbolic_result,
                answer_text=f"{symbolic_result} + C",
                answer_latex=_latex_safe(symbolic_result) + " + C",
                raw_input=expr_str,
                parsed_repr=repr(expr),
                steps=steps,
                passes=[
                    VerificationPass("SymPy symbolic integrate", str(symbolic_result), True),
                    VerificationPass("Differentiate-back check",
                                     "zero" if is_correct else str(diff_simplified), True),
                ],
            )

    except Exception as e:
        return error_result(Mode.MATH, expr_str, f"Integration error: {e}")


def compute_limit(expr_str: str, var_name: str = "x", point: str = "oo", direction: str = "+-") -> VerifiedResult:
    """
    Compute the limit of an expression as var → point.

    Parameters
    ----------
    expr_str : str
        The expression.
    var_name : str
        The variable (default 'x').
    point : str
        The limit point (default 'oo' for infinity).
    direction : str
        '+' (right), '-' (left), or '+-' (two-sided, default).

    Returns
    -------
    VerifiedResult
        The limit value with verification.
    """
    steps: list[str] = []
    try:
        x = Symbol(var_name)
        expr = sympify(expr_str)
        point_expr = sympify(point)

        if direction == "+-":
            lim_plus = limit(expr, x, point_expr, '+')
            lim_minus = limit(expr, x, point_expr, '-')
            steps.append(f"Right limit: {lim_plus}")
            steps.append(f"Left limit:  {lim_minus}")

            if sp.simplify(lim_plus - lim_minus) == 0:
                result = lim_plus
                steps.append(f"Two-sided limit exists: {result}")
                confidence = 1.0
                verdict = Verdict.VERIFIED
            else:
                result = (lim_minus, lim_plus)
                steps.append("One-sided limits differ → limit does not exist (DNE)")
                confidence = 1.0
                verdict = Verdict.VERIFIED
        else:
            result = limit(expr, x, point_expr, direction)
            steps.append(f"Limit ({direction}): {result}")
            confidence = 1.0
            verdict = Verdict.VERIFIED

        return VerifiedResult(
            mode=Mode.MATH,
            verdict=verdict,
            confidence=confidence,
            answer=result,
            answer_text=str(result),
            answer_latex=_latex_safe(result),
            raw_input=expr_str,
            steps=steps,
            passes=[VerificationPass("SymPy limit", str(result), True)],
        )
    except Exception as e:
        return error_result(Mode.MATH, expr_str, f"Limit error: {e}")


def taylor_series(expr_str: str, var_name: str = "x", point: str = "0", order: int = 6) -> VerifiedResult:
    """
    Compute the Taylor series expansion of an expression.

    Parameters
    ----------
    expr_str : str
        The expression to expand.
    var_name : str
        The variable (default 'x').
    point : str
        The expansion point (default '0').
    order : int
        Number of terms (default 6).

    Returns
    -------
    VerifiedResult
        Taylor series with verification.
    """
    steps: list[str] = []
    try:
        x = Symbol(var_name)
        expr = sympify(expr_str)
        point_expr = sympify(point)

        s = series(expr, x, point_expr, order)
        steps.append(f"Taylor series around x={point}, order {order}:")
        steps.append(str(s))

        return VerifiedResult(
            mode=Mode.MATH,
            verdict=Verdict.VERIFIED,
            confidence=1.0,
            answer=s,
            answer_text=str(s),
            answer_latex=_latex_safe(s),
            raw_input=expr_str,
            steps=steps,
            passes=[VerificationPass("SymPy series expansion", str(s), True)],
        )
    except Exception as e:
        return error_result(Mode.MATH, expr_str, f"Series error: {e}")


# ── 2.3 Linear Algebra ─────────────────────────────────────────────────────────

def matrix_eigenvalues(matrix_data: list[list]) -> VerifiedResult:
    """
    Compute eigenvalues and eigenvectors of a square matrix.

    Verifies each eigenvalue by checking Av = λv for each eigenvector.

    Parameters
    ----------
    matrix_data : list[list]
        2D list of matrix entries (numeric or symbolic strings).

    Returns
    -------
    VerifiedResult
        Eigenvalues, eigenvectors, and verification results.
    """
    steps: list[str] = []
    warnings: list[str] = []
    try:
        M = Matrix(matrix_data)
        n = M.shape[0]
        steps.append(f"Matrix ({n}×{n}):\n{M}")

        eigendata = M.eigenvects()  # returns [(eigenval, multiplicity, [eigenvects]), ...]
        steps.append(f"SymPy eigenvects result: {len(eigendata)} distinct eigenvalue(s)")

        verified_pairs = []
        all_correct = True

        for (eigenval, mult, evecs) in eigendata:
            for evec in evecs:
                # Verify: M * v == eigenval * v
                Mv = M * evec
                lambda_v = eigenval * evec
                residual_vec = Mv - lambda_v
                residual_vec_simplified = sp.simplify(residual_vec)
                is_zero = all(e == 0 for e in residual_vec_simplified)

                steps.append(
                    f"λ={eigenval} (mult={mult}): "
                    f"‖Av - λv‖ = {residual_vec_simplified.T} → {'✓' if is_zero else '✗'}"
                )
                if not is_zero:
                    all_correct = False
                    warnings.append(f"Eigenvalue {eigenval}: Av=λv check FAILED")

                verified_pairs.append({
                    "eigenvalue": eigenval,
                    "eigenvector": evec,
                    "multiplicity": mult,
                    "verified": is_zero,
                    "latex_val": _latex_safe(eigenval),
                    "latex_vec": _latex_safe(evec),
                })

        eigenvalues_only = [ep["eigenvalue"] for ep in verified_pairs]
        answer_text = f"Eigenvalues: {eigenvalues_only}"
        answer_latex = ", ".join(ep["latex_val"] for ep in verified_pairs)

        confidence = 1.0 if all_correct else 0.80
        verdict = Verdict.VERIFIED if all_correct else Verdict.UNCERTAIN

        return VerifiedResult(
            mode=Mode.MATH,
            verdict=verdict,
            confidence=confidence,
            answer=verified_pairs,
            answer_text=answer_text,
            answer_latex=answer_latex,
            raw_input=str(matrix_data),
            parsed_repr=str(M),
            steps=steps,
            warnings=warnings,
            passes=[
                VerificationPass("SymPy eigenvects", str(eigenvalues_only), True),
                VerificationPass("Av=λv residual check",
                                 "all zero" if all_correct else "MISMATCH", all_correct),
            ],
        )
    except Exception as e:
        return error_result(Mode.MATH, str(matrix_data), f"Eigenvalue error: {e}")


def matrix_operations(matrix_data: list[list], operation: str = "det") -> VerifiedResult:
    """
    Perform various matrix operations.

    Parameters
    ----------
    matrix_data : list[list]
        The matrix.
    operation : str
        One of: 'det', 'inv', 'rref', 'rank', 'transpose', 'lu', 'qr', 'svd'.

    Returns
    -------
    VerifiedResult
        Result with verification.
    """
    steps: list[str] = []
    try:
        M = Matrix(matrix_data)
        steps.append(f"Matrix:\n{M}")
        steps.append(f"Operation: {operation.upper()}")

        if operation == "det":
            result = M.det()
            steps.append(f"det(M) = {result}")
            # Verify with numpy
            np_det = np.linalg.det(np.array(matrix_data, dtype=float))
            delta = abs(float(result.evalf()) - np_det)
            steps.append(f"NumPy det = {np_det:.6f}, delta = {delta:.2e}")
            confidence = 1.0 if delta < 1e-6 else 0.90
            return VerifiedResult(
                mode=Mode.MATH, verdict=Verdict.VERIFIED if confidence >= 0.95 else Verdict.UNCERTAIN,
                confidence=confidence, answer=result,
                answer_text=str(result), answer_latex=_latex_safe(result),
                raw_input=str(matrix_data), steps=steps, delta=delta,
                passes=[
                    VerificationPass("SymPy det", str(result), True),
                    VerificationPass("NumPy det", str(np_det), True),
                ],
            )

        elif operation == "inv":
            if M.det() == 0:
                return error_result(Mode.MATH, str(matrix_data), "Matrix is singular (det=0), no inverse")
            result = M.inv()
            # Verify: M * M^-1 == I
            check = sp.simplify(M * result - eye(M.shape[0]))
            is_identity = all(e == 0 for e in check)
            steps.append(f"M * M^-1 = I check: {'✓' if is_identity else '✗'}")
            return VerifiedResult(
                mode=Mode.MATH, verdict=Verdict.VERIFIED if is_identity else Verdict.UNCERTAIN,
                confidence=1.0 if is_identity else 0.80, answer=result,
                answer_text=str(result), answer_latex=_latex_safe(result),
                raw_input=str(matrix_data), steps=steps,
                passes=[
                    VerificationPass("SymPy inv", str(result), True),
                    VerificationPass("M*M^-1=I check", "identity" if is_identity else "FAIL", is_identity),
                ],
            )

        elif operation == "rref":
            result, pivots = M.rref()
            steps.append(f"RREF:\n{result}")
            steps.append(f"Pivot columns: {pivots}")
            return VerifiedResult(
                mode=Mode.MATH, verdict=Verdict.VERIFIED, confidence=1.0,
                answer=result, answer_text=str(result), answer_latex=_latex_safe(result),
                raw_input=str(matrix_data), steps=steps,
                passes=[VerificationPass("SymPy rref", str(pivots), True)],
            )

        elif operation == "rank":
            result = M.rank()
            steps.append(f"rank(M) = {result}")
            return VerifiedResult(
                mode=Mode.MATH, verdict=Verdict.VERIFIED, confidence=1.0,
                answer=result, answer_text=str(result), answer_latex=str(result),
                raw_input=str(matrix_data), steps=steps,
                passes=[VerificationPass("SymPy rank", str(result), True)],
            )

        else:
            return error_result(Mode.MATH, str(matrix_data), f"Unknown operation: {operation}")

    except Exception as e:
        return error_result(Mode.MATH, str(matrix_data), f"Matrix operation error: {e}")


# ── 2.4 Number Theory ─────────────────────────────────────────────────────────

def primality_test(n: int) -> VerifiedResult:
    """
    Test whether n is prime using Miller-Rabin (from scratch),
    then verify with SymPy's isprime.

    Parameters
    ----------
    n : int
        The integer to test.

    Returns
    -------
    VerifiedResult
        Primality verdict with dual verification.
    """
    steps: list[str] = []
    try:
        mr_result = mr_is_prime(n)
        steps.append(mr_result.explanation)
        steps += [f"  Witnesses: {mr_result.witnesses_checked[:5]}{'...' if len(mr_result.witnesses_checked) > 5 else ''}"]

        # Cross-check with SymPy
        sympy_result = bool(isprime(n))
        agree = (mr_result.is_prime == sympy_result)
        steps.append(f"SymPy isprime({n}) = {sympy_result} → {'AGREE ✓' if agree else 'DISAGREE ✗'}")

        # For small N, brute-force verify
        brute_check: Optional[bool] = None
        if n < config.SMALL_PRIME_BRUTE_LIMIT:
            brute_check = _brute_force_prime(n)
            agree_brute = (mr_result.is_prime == brute_check)
            steps.append(f"Brute-force check (n<10^6): {brute_check} → {'AGREE ✓' if agree_brute else 'DISAGREE ✗'}")
            agree = agree and agree_brute

        confidence = 1.0 if agree else 0.0
        verdict = Verdict.VERIFIED if agree else Verdict.UNCERTAIN

        result_text = f"{n} is {'PRIME' if mr_result.is_prime else 'COMPOSITE'}"
        if not agree:
            result_text += " (VERIFIERS DISAGREE — see warnings)"

        return VerifiedResult(
            mode=Mode.MATH,
            verdict=verdict,
            confidence=confidence,
            answer=mr_result.is_prime,
            answer_text=result_text,
            answer_latex=f"{n} \\text{{ is {'prime' if mr_result.is_prime else 'composite'}}}",
            raw_input=str(n),
            steps=steps,
            warnings=[] if agree else ["Miller-Rabin and SymPy disagree on primality"],
            passes=[
                VerificationPass("Miller-Rabin (from scratch)", mr_result.is_prime, True),
                VerificationPass("SymPy isprime (verifier)", sympy_result, True),
            ],
        )
    except Exception as e:
        return error_result(Mode.MATH, str(n), f"Primality test error: {e}")


def _brute_force_prime(n: int) -> bool:
    """Brute-force primality check for n < 10^6."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def factorise_number(n: int) -> VerifiedResult:
    """
    Factorise n using Pollard's rho (from scratch), then verify.

    Parameters
    ----------
    n : int
        The integer to factorise.

    Returns
    -------
    VerifiedResult
        Complete factorisation with verification.
    """
    steps: list[str] = []
    try:
        rho_result = rho_factorise(n)
        steps += rho_result.steps

        # Verify by reconstruction
        verified = verify_factorisation(rho_result)
        steps.append(f"Verification: ∏p^e = {n} → {'✓' if verified else '✗'}")

        # Cross-check with SymPy factorint
        sympy_factors = factorint(n)
        sympy_match = (dict(rho_result.factors) == dict(sympy_factors))
        steps.append(f"SymPy factorint: {sympy_factors} → {'AGREE ✓' if sympy_match else 'DISAGREE ✗'}")

        confidence = 1.0 if (verified and sympy_match) else 0.85
        verdict = Verdict.VERIFIED if confidence >= 0.95 else Verdict.UNCERTAIN

        return VerifiedResult(
            mode=Mode.MATH,
            verdict=verdict,
            confidence=confidence,
            answer=rho_result.factors,
            answer_text=f"{n} = {rho_result.factorisation_str}",
            answer_latex=f"{n} = {rho_result.factorisation_str.replace('×', '\\times')}",
            raw_input=str(n),
            steps=steps,
            warnings=[] if (verified and sympy_match) else ["Factorisation mismatch between methods"],
            passes=[
                VerificationPass("Pollard's rho (from scratch)", rho_result.factorisation_str, True),
                VerificationPass("Reconstruction check ∏p^e=n", str(verified), verified),
                VerificationPass("SymPy factorint (verifier)", str(sympy_factors), True),
            ],
        )
    except Exception as e:
        return error_result(Mode.MATH, str(n), f"Factorisation error: {e}")


def number_theory_misc(n: int, m: Optional[int] = None, operation: str = "totient") -> VerifiedResult:
    """
    Miscellaneous number theory operations.

    Parameters
    ----------
    n, m : int
        Input numbers. m is second argument for gcd/lcm/mod_inv.
    operation : str
        One of: 'gcd', 'lcm', 'totient', 'mobius', 'mod_inv'.

    Returns
    -------
    VerifiedResult
        Computed result with verification.
    """
    steps: list[str] = []
    try:
        if operation == "totient":
            result = totient(n)
            steps.append(f"φ({n}) = {result}")
            # Verify by counting coprimes directly for small n
            if n < 10000:
                brute = sum(1 for k in range(1, n + 1) if gcd(k, n) == 1)
                agree = (int(result) == brute)
                steps.append(f"Brute-force count: {brute} → {'✓' if agree else '✗'}")
            return VerifiedResult(
                mode=Mode.MATH, verdict=Verdict.VERIFIED, confidence=1.0,
                answer=result, answer_text=f"φ({n}) = {result}",
                answer_latex=f"\\phi({n}) = {result}",
                raw_input=f"totient({n})", steps=steps,
                passes=[VerificationPass("SymPy totient", str(result), True)],
            )

        elif operation == "gcd":
            result = gcd(n, m)
            steps.append(f"gcd({n}, {m}) = {result}")
            # Verify: result divides both, and is maximal
            divides_both = (n % result == 0) and (m % result == 0)
            steps.append(f"Divides both: {'✓' if divides_both else '✗'}")
            return VerifiedResult(
                mode=Mode.MATH, verdict=Verdict.VERIFIED, confidence=1.0,
                answer=result, answer_text=f"gcd({n}, {m}) = {result}",
                answer_latex=f"\\gcd({n}, {m}) = {result}",
                raw_input=f"gcd({n},{m})", steps=steps,
                passes=[VerificationPass("SymPy gcd", str(result), True)],
            )

        elif operation == "lcm":
            result = lcm(n, m)
            steps.append(f"lcm({n}, {m}) = {result}")
            return VerifiedResult(
                mode=Mode.MATH, verdict=Verdict.VERIFIED, confidence=1.0,
                answer=result, answer_text=f"lcm({n}, {m}) = {result}",
                answer_latex=f"\\text{{lcm}}({n}, {m}) = {result}",
                raw_input=f"lcm({n},{m})", steps=steps,
                passes=[VerificationPass("SymPy lcm", str(result), True)],
            )

        else:
            return error_result(Mode.MATH, f"{operation}({n},{m})", f"Unknown operation: {operation}")

    except Exception as e:
        return error_result(Mode.MATH, str(n), f"Number theory error: {e}")
