# 45Q - Symbolic Bridge

**Layer 18/19: `symbolic_bridge`**

45Q is the deterministic verification layer for math, propositional logic, and code analysis. Its job is to stop fluent guesses from being treated as truth when the system can check the answer.

## Result Contract

Every result should be one of:

| Verdict | Meaning |
| --- | --- |
| Verified | Symbolic and/or formal checks agree |
| Cross-checked | Numeric or secondary pass agrees within tolerance |
| Uncertain | The bridge exposes disagreement or insufficient confidence |

There should be no silent fourth option.

## Architecture

```text
query(text)
  -> detector.py
  -> math_solver.py / logic_checker.py / code_verifier.py
  -> self_check.py
  -> response.py VerifiedResult
```

From-scratch or local reasoning helpers include:

- `miller_rabin.py`
- `pollard_rho.py`
- `cnf_converter.py`
- `dpll_solver.py`
- `natural_deduction.py`
- `sandbox_runner.py`
- `test_generator.py`

## Quick Start

From the repo root:

```bash
python anra.py --symbolic "solve x^2 - 9 = 0"
python anra.py --symbolic "factor 360"
```

From this folder:

```bash
python demo.py
python benchmarks.py
```

Python use from inside this folder:

```python
from symbolic_bridge import query

result = query("Is (A -> B) AND (B -> C) -> (A -> C) a tautology?")
print(result.verdict)
print(result.answer_text)
```

## Dependencies

Best experience:

```bash
pip install sympy scipy numpy
```

The source layer is still present without optional packages, but advanced math and numeric checks need them.

## Boundary

45Q should augment generation, not pretend to be a general language model. Use it where determinism matters:

- equations
- integrals
- primality/factorization
- propositional formulas
- proof-step validation
- code smell and static bug checks

For natural language judgment, route back to the model and evaluator.
