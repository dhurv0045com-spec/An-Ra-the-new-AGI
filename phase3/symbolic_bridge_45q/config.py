"""
config.py — Central configuration for 45Q Symbolic Logic Bridge.

All tunable parameters are defined here. Import this module from
any engine to ensure consistent behaviour across the system.
"""

# ── Truth Table / SAT ─────────────────────────────────────────────────────────
TRUTH_TABLE_MAX_VARS: int = 12        # Above this, switch to DPLL SAT solver
SAT_MAX_VARS: int = 200               # Hard cap for SAT solving

# ── Number Theory ─────────────────────────────────────────────────────────────
MILLER_RABIN_ROUNDS: int = 20         # Witness rounds; 20 gives error < 4^-20
SMALL_PRIME_BRUTE_LIMIT: int = 10**6  # Cross-check number theory below this

# ── Code Verification ─────────────────────────────────────────────────────────
CODE_TEST_COUNT: int = 10             # Auto-generated test cases per function
SANDBOX_TIMEOUT_SEC: int = 5          # Hard kill timeout for sandboxed runs
SANDBOX_MEMORY_MB: int = 256          # Memory limit for sandbox subprocess

COMPLEXITY_CYCLOMATIC_MAX: int = 10   # Flag if cyclomatic complexity exceeds
COMPLEXITY_COGNITIVE_MAX: int = 15    # Flag if cognitive complexity exceeds

# ── Dual-Pass Self-Check ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.95    # Below this → UNCERTAIN response
NUMERIC_VERIFY_POINTS: int = 5        # Random points for numeric cross-check
NUMERIC_DELTA_THRESHOLD: float = 1e-8 # Max allowed symbolic-vs-numeric delta

# ── Mode Detection ────────────────────────────────────────────────────────────
DETECTION_MAX_INPUT_CHARS: int = 10_000  # Detection must run in < 30ms
DETECTION_TIMEOUT_MS: int = 30

# ── Response / Output ─────────────────────────────────────────────────────────
MAX_TRUTH_TABLE_DISPLAY_ROWS: int = 256  # Cap display rows for large tables
LATEX_ENABLED: bool = True               # Include LaTeX in math output

# ── Versioning ────────────────────────────────────────────────────────────────
VERSION: str = "2.0.0"
MODULE_NAME: str = "45Q Symbolic Logic Bridge"
