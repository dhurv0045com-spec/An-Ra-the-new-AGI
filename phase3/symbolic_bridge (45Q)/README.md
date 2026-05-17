# 45Q — Symbolic Bridge

**Component 18/19 · `symbolic_bridge`**

LLMs are fluent. Fluency is not truth. This module answers checkable questions with **verdicts** — math, logic, code — before and after generation when the detector says it should.

---

## Modes

| Mode | Engine | Example |
| --- | --- | --- |
| **MATH** | SymPy + numeric cross-check | `solve x^2 - 9 = 0` |
| **LOGIC** | Truth tables, DPLL, natural deduction | Tautology / satisfiability |
| **CODE** | Sandbox + static checks | "Does this function return the right type?" |
| **NATURAL** | Pass-through | Low symbolic signal |

---

## Entry points

```bash
# From repo root (paths injected)
python anra.py --symbolic "factor 360"
python anra.py --symbolic "Is (A->B) and (B->C) -> (A->C) a tautology?"

# Local demo
cd "phase3/symbolic_bridge (45Q)"
python demo.py
python benchmarks.py
```

**Python API:** `from symbolic_bridge import query` → `result.verdict`, `result.confidence`, `result.steps`

---

## Verdicts you care about

| Verdict | Meaning |
| --- | --- |
| `VERIFIED` | Check passed |
| `VERIFIED_INCORRECT` | Check failed — retry / flag |
| `UNCERTAIN` | Symbolic/numeric disagree or incomplete |
| `UNVERIFIABLE` | No deterministic check — do not fake confidence |

---

## Key files

| File | Role |
| --- | --- |
| `symbolic_bridge.py` | Router + `query()` |
| `detector.py` | MATH / LOGIC / CODE scoring |
| `math_solver.py` | Equations, calculus, linear algebra |
| `logic_checker.py` | Propositional / predicate |
| `code_verifier.py` | Execution + analysis |
| `response.py` | Formatted answers |
| `self_check.py` | Dual-pass numeric/symbolic agreement |

**Deps:** `sympy`, `scipy`, `numpy` (install via `requirements.txt`)

---

## Integration

Orchestrator task kind `symbolic` → this component. Respect feature flags: `set_flag("symbolic_bridge", False)` skips cleanly.

Feeds DFC training: verified samples in the 5% symbolic bucket. See root [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
