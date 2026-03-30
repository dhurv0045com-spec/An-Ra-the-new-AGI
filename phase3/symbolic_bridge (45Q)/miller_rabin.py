"""
miller_rabin.py — Miller-Rabin Probabilistic Primality Test (from scratch).

Implements the Miller-Rabin primality test entirely without SymPy's isprime.
Uses deterministic witness sets for small N and random witnesses for large N.

Algorithm overview
──────────────────
Given n to test:
1. Write n-1 = 2^r · d  (factor out all 2s)
2. For each witness a:
   a. Compute x = a^d mod n
   b. If x == 1 or x == n-1: continue (probably prime for this witness)
   c. Repeat r-1 times:
        x = x² mod n
        if x == n-1: go to next witness (probably prime for this witness)
   d. If we exit without x == n-1: n is COMPOSITE
3. If all witnesses pass: n is PROBABLY PRIME

For deterministic behaviour below 3,215,031,751 we use fixed witness sets.
Above that threshold we use MILLER_RABIN_ROUNDS random witnesses.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from . import config


# ── Deterministic witness sets ─────────────────────────────────────────────────
# These are proven sufficient for the corresponding n bounds:
# (from Pomerance, Selfridge, Wagstaff; and Jaeschke 1993)

_DETERMINISTIC_SETS: list[tuple[int, list[int]]] = [
    (2_047,                   [2]),
    (1_373_653,               [2, 3]),
    (9_080_191,               [31, 73]),
    (25_326_001,              [2, 3, 5]),
    (3_215_031_751,           [2, 3, 5, 7]),
    (4_759_123_141,           [2, 7, 61]),
    (1_122_004_669_633,       [2, 13, 23, 1662803]),
    (2_152_302_898_747,       [2, 3, 5, 7, 11]),
    (3_474_749_660_383,       [2, 3, 5, 7, 11, 13]),
    (341_550_071_728_321,     [2, 3, 5, 7, 11, 13, 17]),
    (3_825_123_056_546_413_051, [2, 3, 5, 7, 11, 13, 17, 19, 23]),
    # Beyond this: use random witnesses (MILLER_RABIN_ROUNDS of them)
]


@dataclass
class PrimalityResult:
    """
    Result of a Miller-Rabin primality test.

    Attributes
    ----------
    n : int
        The number tested.
    is_prime : bool
        True if probably (or certainly) prime.
    is_deterministic : bool
        True if the result is mathematically certain (fixed witness set used).
    witnesses_checked : list[int]
        The witnesses used in this test.
    rounds : int
        Number of witness rounds performed.
    explanation : str
        Human-readable explanation of the result.
    """
    n: int
    is_prime: bool
    is_deterministic: bool
    witnesses_checked: list[int]
    rounds: int
    explanation: str


def _factor_out_twos(n: int) -> tuple[int, int]:
    """
    Write n = 2^r * d where d is odd.

    Parameters
    ----------
    n : int
        A positive even integer (typically n-1 in Miller-Rabin).

    Returns
    -------
    tuple[int, int]
        (r, d) such that 2^r * d == n and d is odd.

    Examples
    --------
    >>> _factor_out_twos(12)
    (2, 3)   # 12 = 4 * 3
    """
    r = 0
    d = n
    while d % 2 == 0:
        d //= 2
        r += 1
    return r, d


def _miller_rabin_witness(n: int, a: int, r: int, d: int) -> bool:
    """
    Test whether `a` is a Miller-Rabin witness for the compositeness of `n`.

    Parameters
    ----------
    n : int
        The number to test (must be odd and > 2).
    a : int
        The witness candidate (2 ≤ a ≤ n-2).
    r : int
        Exponent such that n-1 = 2^r * d.
    d : int
        Odd part such that n-1 = 2^r * d.

    Returns
    -------
    bool
        True  → a does NOT witness compositeness (n is probably prime).
        False → a witnesses that n is COMPOSITE.

    Algorithm
    ---------
    Compute x = a^d mod n.
    If x == 1 or x == n-1: return True (probably prime).
    Loop r-1 times:
        x = x^2 mod n
        if x == n-1: return True (probably prime)
    If we reach here: n is composite.
    """
    # Step 1: compute a^d mod n  (Python's built-in pow is fast)
    x = pow(a, d, n)

    # Step 2: trivial passing cases
    if x == 1 or x == n - 1:
        return True  # probably prime for this witness

    # Step 3: square up to r-1 times
    for _ in range(r - 1):
        x = pow(x, 2, n)
        if x == n - 1:
            return True  # probably prime for this witness

    # If none of the squarings produced n-1, a witnesses compositeness
    return False


def _get_witnesses(n: int) -> tuple[list[int], bool]:
    """
    Return the appropriate witness set for testing n.

    For n below a proven threshold, returns a deterministic witness set
    that guarantees a correct answer. For larger n, returns
    config.MILLER_RABIN_ROUNDS random witnesses.

    Parameters
    ----------
    n : int
        The number being tested.

    Returns
    -------
    tuple[list[int], bool]
        (witnesses, is_deterministic)
    """
    for limit, witnesses in _DETERMINISTIC_SETS:
        if n < limit:
            # Filter witnesses to valid range [2, n-2]
            valid = [a for a in witnesses if 2 <= a <= n - 2]
            return valid or [2], True

    # Large n: use random witnesses
    witnesses = []
    seen: set[int] = set()
    attempts = 0
    while len(witnesses) < config.MILLER_RABIN_ROUNDS and attempts < config.MILLER_RABIN_ROUNDS * 3:
        a = random.randint(2, n - 2)
        if a not in seen:
            seen.add(a)
            witnesses.append(a)
        attempts += 1
    return witnesses, False


def is_prime(n: int) -> PrimalityResult:
    """
    Test whether n is prime using the Miller-Rabin algorithm.

    Handles small cases directly:
      n < 2         → not prime
      n == 2 or 3   → prime
      n even or div by 3 → not prime

    For n ≥ 5, applies Miller-Rabin with the appropriate witness set.

    Parameters
    ----------
    n : int
        The integer to test. Must be a non-negative integer.

    Returns
    -------
    PrimalityResult
        Complete result including verdict, witnesses, and explanation.

    Raises
    ------
    ValueError
        If n < 0.

    Examples
    --------
    >>> is_prime(982451653)
    PrimalityResult(n=982451653, is_prime=True, ...)
    >>> is_prime(4)
    PrimalityResult(n=4, is_prime=False, ...)
    """
    if n < 0:
        raise ValueError(f"Miller-Rabin requires n ≥ 0, got {n}")

    # ── Trivial cases ──────────────────────────────────────────────────────────
    if n < 2:
        return PrimalityResult(
            n=n, is_prime=False, is_deterministic=True,
            witnesses_checked=[], rounds=0,
            explanation=f"{n} < 2, not prime by definition.",
        )
    if n == 2:
        return PrimalityResult(
            n=n, is_prime=True, is_deterministic=True,
            witnesses_checked=[], rounds=0,
            explanation="2 is the smallest prime.",
        )
    if n == 3:
        return PrimalityResult(
            n=n, is_prime=True, is_deterministic=True,
            witnesses_checked=[], rounds=0,
            explanation="3 is prime.",
        )
    if n % 2 == 0:
        return PrimalityResult(
            n=n, is_prime=False, is_deterministic=True,
            witnesses_checked=[], rounds=0,
            explanation=f"{n} is even (divisible by 2), not prime.",
        )
    if n % 3 == 0:
        return PrimalityResult(
            n=n, is_prime=False, is_deterministic=True,
            witnesses_checked=[], rounds=0,
            explanation=f"{n} is divisible by 3, not prime.",
        )

    # ── Small primes sieve (to catch composites quickly) ──────────────────────
    small_primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for p in small_primes:
        if n == p:
            return PrimalityResult(
                n=n, is_prime=True, is_deterministic=True,
                witnesses_checked=[], rounds=0,
                explanation=f"{n} is a known small prime.",
            )
        if n % p == 0:
            return PrimalityResult(
                n=n, is_prime=False, is_deterministic=True,
                witnesses_checked=[], rounds=0,
                explanation=f"{n} is divisible by {p}, not prime.",
            )

    # ── Miller-Rabin core ──────────────────────────────────────────────────────
    # Write n-1 = 2^r * d
    r, d = _factor_out_twos(n - 1)

    witnesses, is_deterministic = _get_witnesses(n)

    composite_witness: Optional[int] = None
    for a in witnesses:
        if not _miller_rabin_witness(n, a, r, d):
            # Found a witness to compositeness
            composite_witness = a
            break

    if composite_witness is not None:
        return PrimalityResult(
            n=n,
            is_prime=False,
            is_deterministic=True,  # compositeness is always certain
            witnesses_checked=witnesses,
            rounds=len(witnesses),
            explanation=(
                f"{n} is COMPOSITE. Witness a={composite_witness} "
                f"demonstrates that a^(n-1) ≢ 1 (mod n)."
            ),
        )

    cert = "certainly" if is_deterministic else f"probably (error < 4^-{len(witnesses)})"
    return PrimalityResult(
        n=n,
        is_prime=True,
        is_deterministic=is_deterministic,
        witnesses_checked=witnesses,
        rounds=len(witnesses),
        explanation=(
            f"{n} is {cert} PRIME. "
            f"All {len(witnesses)} Miller-Rabin witnesses passed "
            f"(n-1 = 2^{r} × {d})."
        ),
    )


def verify_with_sympy(n: int) -> bool:
    """
    Cross-check our Miller-Rabin result using SymPy's isprime.

    Used in the dual-pass self-check layer. This is the ONLY place
    in 45Q where sympy.isprime is called — as a verifier, not a primary.

    Parameters
    ----------
    n : int
        The integer to verify.

    Returns
    -------
    bool
        SymPy's verdict.
    """
    from sympy import isprime as sympy_isprime
    return bool(sympy_isprime(n))
