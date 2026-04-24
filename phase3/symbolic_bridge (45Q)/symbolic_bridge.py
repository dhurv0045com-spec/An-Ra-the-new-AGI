"""
symbolic_bridge.py
==================
Compatibility shim for the 45Q Symbolic Logic Bridge package.

The 45Q directory name starts with a digit, so Python cannot import it
directly as a package from sys.path. This file sits inside phase3/symbolic_bridge (45Q)/
and acts as the importable `symbolic_bridge` module when that directory
is on sys.path.

It loads the real __init__.py as a package using importlib so that all
relative imports inside 45Q (.config, .detector, .math_solver, etc.)
resolve correctly.

Usage — after adding phase3/45Q to sys.path:
    from symbolic_bridge import query, detect
    from symbolic_bridge import solve_equation, check_formula
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# ── Load phase3/symbolic_bridge (45Q)/__init__.py as a properly-named package ──────────────────
# submodule_search_locations tells importlib that relative imports like
# `.config` and `.detector` should look in the same directory as __init__.py.

_PKG_DIR    = Path(__file__).parent                  # phase3/symbolic_bridge (45Q)/
_INIT_PY    = _PKG_DIR / "__init__.py"
_PKG_NAME   = "symbolic_bridge"                      # the module name we expose
# Private key used DURING loading to avoid colliding with this shim module
_PRIV_KEY   = "_symbolic_bridge_pkg_internal"

if _PKG_NAME not in sys.modules or sys.modules[_PKG_NAME] is sys.modules.get(__name__):
    _spec = importlib.util.spec_from_file_location(
        _PRIV_KEY,
        str(_INIT_PY),
        submodule_search_locations=[str(_PKG_DIR)],
    )
    _mod = importlib.util.module_from_spec(_spec)
    # Register under the PRIVATE key so relative sub-imports resolve correctly,
    # but NOT yet as symbolic_bridge (that would point back to this shim file).
    sys.modules[_PRIV_KEY] = _mod

    # Also register submodule paths the __init__ will load via relative import
    # so that .config, .detector etc. are found under the right package name.
    _spec.loader.exec_module(_mod)

    # Now promote to the canonical public name
    sys.modules[_PKG_NAME] = _mod

_mod = sys.modules[_PKG_NAME]

# ── Re-export the public API ──────────────────────────────────────────────────
# Top-level unified interface
query       = _mod.query
query_math  = _mod.query_math
query_logic = _mod.query_logic
query_code  = _mod.query_code

# Detection
detect          = _mod.detect
DetectionResult = _mod.DetectionResult

# Response types
Mode             = _mod.Mode
Verdict          = _mod.Verdict
VerifiedResult   = _mod.VerifiedResult
VerificationPass = _mod.VerificationPass

# Math
solve_equation      = _mod.solve_equation
differentiate       = _mod.differentiate
integrate_expr      = _mod.integrate_expr
compute_limit       = _mod.compute_limit
taylor_series       = _mod.taylor_series
matrix_eigenvalues  = _mod.matrix_eigenvalues
matrix_operations   = _mod.matrix_operations
primality_test      = _mod.primality_test
factorise_number    = _mod.factorise_number

# Logic
check_formula     = _mod.check_formula
verify_syllogism  = _mod.verify_syllogism
verify_proof      = _mod.verify_proof
build_truth_table = _mod.build_truth_table

# Code
analyse_code   = _mod.analyse_code
generate_tests = _mod.generate_tests
run_tests      = _mod.run_tests

# From-scratch implementations
miller_rabin_is_prime  = _mod.miller_rabin_is_prime
pollard_rho_factorise  = _mod.pollard_rho_factorise
to_cnf                 = _mod.to_cnf
parse_logic_formula    = _mod.parse_logic_formula
solve_cnf              = _mod.solve_cnf
check_proof            = _mod.check_proof

__version__ = getattr(_mod, "__version__", "1.0.0")

__all__ = [
    "query", "query_math", "query_logic", "query_code",
    "detect", "DetectionResult",
    "Mode", "Verdict", "VerifiedResult", "VerificationPass",
    "solve_equation", "differentiate", "integrate_expr",
    "compute_limit", "taylor_series",
    "matrix_eigenvalues", "matrix_operations",
    "primality_test", "factorise_number",
    "check_formula", "verify_syllogism", "verify_proof", "build_truth_table",
    "analyse_code", "generate_tests", "run_tests",
    "miller_rabin_is_prime", "pollard_rho_factorise",
    "to_cnf", "parse_logic_formula", "solve_cnf", "check_proof",
    "__version__",
]


def health_check() -> dict:
    try:
        import sympy

        result = query("What is 2 + 2?")
        return {
            "status": "ok",
            "module": "symbolic_bridge",
            "sympy_version": sympy.__version__,
            "verdict": getattr(result.verdict, "value", str(result.verdict)),
        }
    except ImportError:
        return {"status": "degraded", "module": "symbolic_bridge", "reason": "sympy not installed"}
    except Exception as exc:
        return {"status": "degraded", "module": "symbolic_bridge", "reason": str(exc)}
