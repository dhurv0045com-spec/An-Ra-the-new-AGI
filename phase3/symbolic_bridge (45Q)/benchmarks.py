"""
benchmarks.py — Internal Performance Benchmarks for 45Q.

Measures the performance characteristics of each engine component.
Run with: python -m symbolic_bridge.benchmarks

Expected results:
  - Mode detection:   < 30ms per query
  - Miller-Rabin:     < 10ms for 64-bit numbers
  - Pollard's rho:    < 1s for N < 10^15
  - DPLL 50-var 3SAT: < 2s
  - Truth table 12v:  < 5s
"""

from __future__ import annotations
import time
import random
from typing import Callable, Any


def _time_it(label: str, fn: Callable, *args, **kwargs) -> tuple[Any, float]:
    """Run fn(*args, **kwargs) and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def _print_result(label: str, elapsed: float, target: float, extra: str = "") -> None:
    """Print a benchmark result with PASS/FAIL indicator."""
    status = "✓ PASS" if elapsed <= target else "✗ FAIL"
    bar = "█" * min(40, int(elapsed / target * 20))
    print(f"  {status}  {label}")
    print(f"         elapsed={elapsed*1000:.2f}ms  target={target*1000:.0f}ms  {bar}")
    if extra:
        print(f"         {extra}")
    print()


def bench_detection() -> None:
    """Benchmark the mode detection engine."""
    from symbolic_bridge.detector import detect

    print("── Mode Detection ─────────────────────────────────────────────")

    queries = [
        ("Math equation", "solve 3x^3 - 2x^2 + x - 5 = 0"),
        ("Logic formula", "Is (A→B) ∧ (B→C) → (A→C) a tautology?"),
        ("Code snippet", "def find_max(lst): return max(lst[0:len(lst)-1])"),
        ("Natural text", "What is the capital of France?"),
        ("Long query", "Please integrate the function x squared times sin of x from 0 to pi " * 10),
    ]

    total_elapsed = 0.0
    for label, query in queries:
        result, elapsed = _time_it(label, detect, query)
        total_elapsed += elapsed
        _print_result(
            f"{label} → {result.mode.value}",
            elapsed,
            target=0.030,  # 30ms
            extra=f"score={result.scores}",
        )

    print(f"  Average: {total_elapsed / len(queries) * 1000:.2f}ms per query\n")


def bench_miller_rabin() -> None:
    """Benchmark Miller-Rabin primality testing."""
    from symbolic_bridge.miller_rabin import is_prime

    print("── Miller-Rabin Primality ─────────────────────────────────────")

    test_cases = [
        ("Small prime (7)", 7),
        ("Medium prime (982,451,653)", 982_451_653),
        ("Large prime (2^61 - 1)", 2**61 - 1),
        ("Large composite (2^62)", 2**62),
        ("Random large number", random.randint(10**18, 10**19)),
    ]

    for label, n in test_cases:
        result, elapsed = _time_it(label, is_prime, n)
        _print_result(
            f"{label}: {n} → {'PRIME' if result.is_prime else 'COMPOSITE'}",
            elapsed,
            target=0.010,  # 10ms
            extra=f"rounds={result.rounds}, deterministic={result.is_deterministic}",
        )


def bench_pollard_rho() -> None:
    """Benchmark Pollard's rho factorisation."""
    from symbolic_bridge.pollard_rho import factorise

    print("── Pollard's Rho Factorisation ────────────────────────────────")

    test_cases = [
        ("Semiprime 15", 15),
        ("360 = 2^3 * 3^2 * 5", 360),
        ("Large semiprime (p*q ~10^10)", 9_999_999_967 * 3),  # ~3*10^10
        ("N = 10^15 composite", 999_999_999_999_937 * 2),
        ("Prime 982,451,653", 982_451_653),
    ]

    for label, n in test_cases:
        result, elapsed = _time_it(label, factorise, n)
        _print_result(
            f"{label}: {n} = {result.factorisation_str}",
            elapsed,
            target=1.0,  # 1 second
        )


def bench_dpll() -> None:
    """Benchmark DPLL SAT solver on random 3-SAT instances."""
    from symbolic_bridge.dpll_solver import solve_cnf
    from symbolic_bridge.cnf_converter import Clause, Literal

    print("── DPLL SAT Solver ────────────────────────────────────────────")

    def random_3sat(num_vars: int, num_clauses: int) -> tuple[list[Clause], set[str]]:
        """Generate a random 3-SAT instance."""
        variables = [f"v{i}" for i in range(num_vars)]
        clauses: list[Clause] = []
        for _ in range(num_clauses):
            chosen = random.sample(variables, 3)
            clause: Clause = frozenset(
                (v, random.choice([True, False])) for v in chosen
            )
            clauses.append(clause)
        return clauses, set(variables)

    test_cases = [
        ("10-var, 40-clause 3-SAT", 10, 40),
        ("20-var, 80-clause 3-SAT", 20, 80),
        ("50-var, 200-clause 3-SAT", 50, 200),  # Target: < 2 seconds
        ("100-var, 400-clause 3-SAT", 100, 400),
    ]

    for label, n_vars, n_clauses in test_cases:
        clauses, variables = random_3sat(n_vars, n_clauses)
        result, elapsed = _time_it(label, solve_cnf, clauses, variables)
        target = 2.0 if n_vars == 50 else 5.0
        _print_result(
            f"{label}: {'SAT' if result.satisfiable else 'UNSAT'} "
            f"(decisions={result.decisions}, propagations={result.propagations})",
            elapsed,
            target=target,
        )


def bench_truth_table() -> None:
    """Benchmark truth table generation."""
    from symbolic_bridge.cnf_converter import parse_formula
    from symbolic_bridge.logic_checker import build_truth_table

    print("── Truth Table ────────────────────────────────────────────────")

    formulas = [
        ("4-var formula", "((A AND B) OR (C AND D)) AND NOT (A AND C)"),
        ("8-var formula", "((A OR B) AND (C OR D)) AND ((E OR F) AND (G OR H))"),
        ("12-var tautology", "(A OR NOT A) AND (B OR NOT B) AND (C OR NOT C) AND (D OR NOT D)"),
    ]

    for label, formula_str in formulas:
        try:
            formula = parse_formula(formula_str)
            result, elapsed = _time_it(label, build_truth_table, formula)
            _print_result(
                f"{label}: {result.classification} ({result.num_rows} rows)",
                elapsed,
                target=5.0,
            )
        except Exception as e:
            print(f"  ✗ {label}: ERROR — {e}\n")


def bench_math_solver() -> None:
    """Benchmark core math solver operations."""
    from symbolic_bridge.math_solver import (
        solve_equation, integrate_expr, matrix_eigenvalues
    )

    print("── Math Solver ────────────────────────────────────────────────")

    ops = [
        ("Polynomial solve (degree 3)", lambda: solve_equation("3*x**3 - 2*x**2 + x - 5")),
        ("Definite integral ∫x²sin(x)dx [0,π]", lambda: integrate_expr("x**2 * sin(x)", lower="0", upper="pi")),
        ("Eigenvalues 2×2", lambda: matrix_eigenvalues([[4, 1], [2, 3]])),
        ("Eigenvalues 4×4", lambda: matrix_eigenvalues([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])),
    ]

    for label, fn in ops:
        result, elapsed = _time_it(label, fn)
        _print_result(
            f"{label}: confidence={result.confidence:.2f}",
            elapsed,
            target=5.0,
            extra=f"verdict={result.verdict.value}",
        )


def run_all() -> None:
    """Run all benchmarks and print a summary."""
    print("=" * 68)
    print("  45Q SYMBOLIC LOGIC BRIDGE — PERFORMANCE BENCHMARKS")
    print("=" * 68)
    print()

    benches = [
        ("Mode Detection", bench_detection),
        ("Miller-Rabin", bench_miller_rabin),
        ("Pollard's Rho", bench_pollard_rho),
        ("DPLL SAT Solver", bench_dpll),
        ("Truth Table", bench_truth_table),
        ("Math Solver", bench_math_solver),
    ]

    for name, fn in benches:
        try:
            fn()
        except Exception as e:
            print(f"  ✗ {name} benchmark FAILED: {e}\n")

    print("=" * 68)
    print("  Benchmarks complete.")
    print("=" * 68)


if __name__ == "__main__":
    run_all()
