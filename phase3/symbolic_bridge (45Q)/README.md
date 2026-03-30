# 45Q — Symbolic Logic Bridge

A production-grade, deterministic, formally verified reasoning engine for
mathematics, formal logic, and code correctness.

Every result is either:
- **(a) Formally verified** — confidence = 1.0
- **(b) Numerically cross-checked** — confidence = 0.95–0.99
- **(c) Explicitly flagged UNCERTAIN** — confidence < 0.95, both results shown

There is no fourth option. The system never returns an unverified answer silently.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         query(text)                                  │
│                       Public API (__init__.py)                       │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
           ┌─────────────────────┐
           │   detector.py       │  ← pure regex, < 30ms
           │  Mode Detection     │
           └──────┬──────┬───────┘
                  │      │
        ┌─────────┘      └──────────┐
        ▼                           ▼
┌──────────────┐         ┌──────────────────┐
│ math_solver  │         │  logic_checker   │
│  SymPy +     │         │  Truth Table +   │
│  scipy/numpy │         │  DPLL SAT solver │
└──────┬───────┘         └────────┬─────────┘
       │                          │
       │   ┌──────────────────────┤
       │   │  code_verifier       │
       │   │  AST static analysis │
       │   │  test_generator      │
       │   │  sandbox_runner      │
       │   └──────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      self_check.py                                   │
│           Dual-Pass Verification (mandatory, cannot bypass)          │
│   Pass A: Symbolic/Primary   Pass B: Numeric/DPLL Verification       │
│   If delta > 1e-8 → UNCERTAIN    If passes disagree → UNCERTAIN      │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      response.py                                     │
│                     VerifiedResult                                   │
│   mode | verdict | confidence | answer | steps | passes | warnings  │
└─────────────────────────────────────────────────────────────────────┘

From-scratch implementations:
  miller_rabin.py    — Miller-Rabin primality (Brent witnesses)
  pollard_rho.py     — Pollard's rho factorisation (Brent variant)
  cnf_converter.py   — Tseitin CNF transformation (linear size)
  dpll_solver.py     — DPLL + VSIDS heuristic SAT solver
  natural_deduction.py — Proof step rule checker
```

---

## Installation

```bash
pip install sympy scipy numpy
```

Or with pinned versions:
```bash
pip install -r symbolic_bridge/requirements.txt
```

---

## Quick Start (10 lines)

```python
from symbolic_bridge import query

# Auto-detect mode and solve
result = query("solve 3x^3 - 2x^2 + x - 5 = 0")
print(result.full_report())

# Logic
result = query("Is (A→B) ∧ (B→C) → (A→C) a tautology?")
print(result.answer_text)   # → TAUTOLOGY

# Code verification
result = query("def find_max(lst): return max(lst[0:len(lst)-1])")
print(result.warnings)      # → ['Slice ... excludes last element']
```

---

## Examples

### MATH: Solve an equation
```python
from symbolic_bridge import solve_equation

result = solve_equation("x**2 - 4 = 0")
# result.answer_text → "2; -2"
# result.confidence  → 1.0
# result.verdict     → VERIFIED
```

### MATH: Definite integral
```python
from symbolic_bridge import integrate_expr

result = integrate_expr("x**2 * sin(x)", lower="0", upper="pi")
# result.answer_text → "pi**2 - 4"
# result.passes[1]   → scipy.quad cross-check
# result.delta       → < 1e-10
```

### MATH: Primality (from scratch)
```python
from symbolic_bridge import miller_rabin_is_prime, pollard_rho_factorise

mr = miller_rabin_is_prime(982_451_653)
# mr.is_prime → True, mr.is_deterministic → True

rho = pollard_rho_factorise(360)
# rho.factorisation_str → "2^3 × 3^2 × 5"
```

### LOGIC: Formula classification
```python
from symbolic_bridge import check_formula

result = check_formula("(A -> B) AND (B -> C) -> (A -> C)")
# result.answer  → "TAUTOLOGY"
# result.passes  → [TruthTable, DPLL] — both agree → confidence 1.0
```

### LOGIC: Proof verification
```python
from symbolic_bridge import verify_proof

proof = """
1. P -> Q    [Premise]
2. Q -> R    [Premise]
3. P         [Premise]
4. Q         [MP: 1, 3]
5. R         [MP: 2, 4]
"""
result = verify_proof(proof)
# result.verdict → VALID_PROOF
```

### CODE: Static analysis
```python
from symbolic_bridge import analyse_code

result = analyse_code("""
def fib(n):
    return fib(n-1) + fib(n-2)
""")
# result.issues[0].category  → INFINITE_LOOP
# result.issues[0].severity  → CRITICAL
```

---

## Run the Demo

```bash
python -m symbolic_bridge.demo
```

## Run Benchmarks

```bash
python -m symbolic_bridge.benchmarks
```

---

## Performance Characteristics

| Component          | Target     | Typical    |
|--------------------|------------|------------|
| Mode detection     | < 30ms     | 0.5–2ms    |
| Miller-Rabin (64b) | < 10ms     | 1–3ms      |
| Pollard's rho 10^12| < 1s       | 50–200ms   |
| DPLL 50-var 3-SAT  | < 2s       | 100ms–1.5s |
| Truth table 12-var | < 5s       | 0.5–2s     |
| SymPy integral     | < 5s       | 0.2–3s     |

---

## Limitations and Known Boundaries

- **SAT**: DPLL without clause learning (CDCL) is slower than industrial solvers
  for very hard instances. The VSIDS heuristic helps significantly.
- **Math parsing**: The equation parser strips natural language prefixes but is
  not a full NLP parser. Complex phrasings may need simplification.
- **Logic parser**: Supports propositional logic. Full first-order predicate logic
  with quantifiers is partially supported (verification only, not automated proof).
- **Sandbox**: Memory/network limits rely on OS support. On Windows, `resource`
  module limits are not enforced.
- **Branch cuts**: Definite integrals involving `sqrt(x²)`, `log(x)` etc. may
  produce UNCERTAIN due to SymPy's handling of branch cuts. This is intentional
  — the self-check correctly identifies the discrepancy.
- **Confidence 1.0**: Achieved only when both symbolic and numeric passes agree
  within 1e-8. Transcendental equations may yield APPROXIMATE solutions.
