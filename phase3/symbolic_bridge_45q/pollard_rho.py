"""
pollard_rho.py — Pollard's Rho Factorisation Algorithm (from scratch).

Implements Brent's improvement of Pollard's rho algorithm for integer
factorisation. Handles numbers up to 10^15 in < 1 second for typical inputs.

Algorithm overview (Brent's variant)
──────────────────────────────────────
Given composite n:
1. Pick random starting value x, set y = x, c = random non-zero value.
2. Use Floyd-like cycle detection (Brent's: exponential-step tortoise).
3. Track the product of |x - y| values, compute gcd in batches.
4. If gcd gives a non-trivial factor, recurse on both parts.
5. If gcd == n (cycle detected without factor), retry with new c.

The iteration function is f(x) = (x² + c) mod n.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional
from .miller_rabin import is_prime


@dataclass
class FactorisationResult:
    """
    Result of Pollard's rho factorisation.

    Attributes
    ----------
    n : int
        The number factorised.
    factors : dict[int, int]
        Prime factors mapped to their multiplicity, e.g. {2: 3, 7: 1} for 56.
    is_prime : bool
        True if n itself is prime (trivial factorisation: {n: 1}).
    factorisation_str : str
        Human-readable factorisation, e.g. "2^3 × 7".
    steps : list[str]
        Description of the factorisation process.
    """
    n: int
    factors: dict[int, int] = field(default_factory=dict)
    is_prime: bool = False
    factorisation_str: str = ""
    steps: list[str] = field(default_factory=list)


def _gcd(a: int, b: int) -> int:
    """
    Compute gcd(a, b) iteratively.

    Parameters
    ----------
    a, b : int
        Non-negative integers.

    Returns
    -------
    int
        Greatest common divisor.
    """
    while b:
        a, b = b, a % b
    return a


def _brent_rho(n: int) -> Optional[int]:
    """
    Find a non-trivial factor of n using Brent's variant of Pollard's rho.

    Returns a factor strictly between 1 and n, or None if the attempt
    failed (e.g. after too many retries with different starting values).

    Parameters
    ----------
    n : int
        Composite integer to factor (must be > 1 and odd).

    Returns
    -------
    Optional[int]
        A non-trivial factor, or None if detection failed.

    Algorithm details
    -----------------
    Brent's improvement uses:
    - A "power of two" step schedule (r = 1, 2, 4, 8, ...) for the
      tortoise position, reducing the number of gcd calls.
    - Batch gcd: multiply up to 128 |x-y| values modulo n before
      computing a single gcd. This speeds up the bottleneck operation.
    - If the batch product produces gcd == n (we overshot), fall back
      to element-wise gcd to recover the exact factor.
    """
    MAX_RETRIES = 25

    for attempt in range(MAX_RETRIES):
        # Random starting parameters
        y = random.randint(1, n - 1)
        c = random.randint(1, n - 1)
        m = random.randint(1, n - 1)

        g = 1
        q = 1
        r = 1
        x = y

        while g == 1:
            x = y
            for _ in range(r):
                # f(y) = (y^2 + c) mod n
                y = (y * y + c) % n

            k = 0
            while k < r and g == 1:
                ys = y
                # Batch: compute product of min(m, r-k) differences
                batch = min(m, r - k)
                for _ in range(batch):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = _gcd(q, n)
                k += m

            r *= 2  # Double the step size (Brent's key idea)

        if g == n:
            # Overshot: recover by stepping one at a time from ys
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = _gcd(abs(x - ys), n)

        if g != n:
            return g  # Found a non-trivial factor

    # All retries exhausted — caller should try a different method
    return None


def _trial_division_small(n: int, steps: list[str]) -> tuple[int, dict[int, int]]:
    """
    Remove all small prime factors (≤ 1000) from n by trial division.

    Parameters
    ----------
    n : int
        The number to trial-divide.
    steps : list[str]
        Mutable list to append step descriptions to.

    Returns
    -------
    tuple[int, dict[int, int]]
        (remaining, small_factors) where small_factors maps prime→count.
    """
    small_factors: dict[int, int] = {}
    remaining = n

    # Check 2 separately for speed
    if remaining % 2 == 0:
        count = 0
        while remaining % 2 == 0:
            remaining //= 2
            count += 1
        small_factors[2] = count
        steps.append(f"Trial division: extracted 2^{count}")

    # Odd factors up to min(1000, sqrt(remaining))
    p = 3
    while p <= 1000 and p * p <= remaining:
        if remaining % p == 0:
            count = 0
            while remaining % p == 0:
                remaining //= p
                count += 1
            small_factors[p] = count
            steps.append(f"Trial division: extracted {p}^{count}")
        p += 2

    return remaining, small_factors


def _factorise_recursive(n: int, steps: list[str]) -> dict[int, int]:
    """
    Recursively factorise n into prime factors using Pollard's rho.

    Parameters
    ----------
    n : int
        The number to factorise (> 1).
    steps : list[str]
        Mutable list to append step descriptions to.

    Returns
    -------
    dict[int, int]
        Mapping of prime factor → multiplicity.
    """
    if n == 1:
        return {}

    # Test primality first (avoids unnecessary rho calls)
    primality = is_prime(n)
    if primality.is_prime:
        steps.append(f"Miller-Rabin: {n} is prime")
        return {n: 1}

    steps.append(f"Pollard's rho: factorising {n}")
    factor = _brent_rho(n)

    if factor is None:
        # Rho failed; try harder trial division up to sqrt(n)
        steps.append(f"Rho failed for {n}; falling back to extended trial division")
        p = 1009  # Resume after our initial trial division stopped at 1000
        while p * p <= n:
            if n % p == 0:
                factor = p
                break
            p += 2
        if factor is None:
            # n must be prime or we cannot factor it
            steps.append(f"Cannot factor {n} — treating as prime")
            return {n: 1}

    # Merge factors from both halves
    other = n // factor
    steps.append(f"Split {n} → {factor} × {other}")

    left = _factorise_recursive(factor, steps)
    right = _factorise_recursive(other, steps)

    merged: dict[int, int] = dict(left)
    for p, e in right.items():
        merged[p] = merged.get(p, 0) + e
    return merged


def factorise(n: int) -> FactorisationResult:
    """
    Completely factorise n into its prime factors.

    For n < 2: returns trivial result.
    For n prime: returns {n: 1}.
    Otherwise: uses trial division for small factors, then Pollard's rho.

    Parameters
    ----------
    n : int
        The integer to factorise. Must be ≥ 0.

    Returns
    -------
    FactorisationResult
        Complete factorisation with prime factors, multiplicities, and steps.

    Raises
    ------
    ValueError
        If n < 0.

    Examples
    --------
    >>> factorise(360)
    FactorisationResult(n=360, factors={2: 3, 3: 2, 5: 1}, ...)
    >>> factorise(982451653)
    FactorisationResult(n=982451653, factors={982451653: 1}, is_prime=True, ...)
    """
    if n < 0:
        raise ValueError(f"factorise requires n ≥ 0, got {n}")

    steps: list[str] = []

    if n < 2:
        result_str = str(n)
        return FactorisationResult(
            n=n, factors={}, is_prime=False,
            factorisation_str=result_str, steps=[f"{n} has no prime factors"],
        )

    if n == 2:
        return FactorisationResult(
            n=n, factors={2: 1}, is_prime=True,
            factorisation_str="2", steps=["2 is prime"],
        )

    steps.append(f"Factorising n = {n}")

    # Phase 1: trial division for small factors
    remaining, small_factors = _trial_division_small(n, steps)

    # Phase 2: Pollard's rho for remaining large factor
    if remaining > 1:
        large_factors = _factorise_recursive(remaining, steps)
        for p, e in large_factors.items():
            small_factors[p] = small_factors.get(p, 0) + e

    all_factors = dict(sorted(small_factors.items()))

    # Build human-readable string
    parts = []
    for p in sorted(all_factors):
        e = all_factors[p]
        parts.append(f"{p}^{e}" if e > 1 else str(p))
    factorisation_str = " × ".join(parts)

    # Check if n is itself prime
    n_is_prime = (len(all_factors) == 1 and list(all_factors.values())[0] == 1)

    steps.append(f"Final factorisation: {n} = {factorisation_str}")

    return FactorisationResult(
        n=n,
        factors=all_factors,
        is_prime=n_is_prime,
        factorisation_str=factorisation_str,
        steps=steps,
    )


def verify_factorisation(result: FactorisationResult) -> bool:
    """
    Verify a factorisation by reconstructing n from its factors.

    Parameters
    ----------
    result : FactorisationResult
        The factorisation to verify.

    Returns
    -------
    bool
        True if ∏ p^e == n.
    """
    product = 1
    for p, e in result.factors.items():
        product *= p ** e
    return product == result.n
