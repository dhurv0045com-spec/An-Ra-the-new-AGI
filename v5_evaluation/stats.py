"""Registered statistical analyses: small, frozen, executable.

Three rules, each with a stable analysis ID and implementation hash. Unknown
rules fail before any run. Bootstrap randomness derives from the bound seed
so analyses reproduce byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Callable, Mapping


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _impl_sha(rule_id: str, version: str) -> str:
    return hashlib.sha256(f"{rule_id}\0{version}".encode("utf-8")).hexdigest()


def _binomial_tail(n: int, k: int) -> float:
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n


def wilson_binomial(
    records: list[Mapping[str, Any]], *, seed: int = 0
) -> dict[str, object]:
    """Wilson 95% interval over record correctness."""

    _ = seed
    from .metrics import wilson_lcb

    trials = len(records)
    if trials == 0:
        raise ValueError("Wilson analysis needs task records")
    successes = sum(1 for record in records if record.get("correct"))
    return {
        "analysis_id": "anra-v5-stats/wilson-binomial-v1",
        "implementation_sha256": _impl_sha("anra-v5-stats/wilson-binomial-v1", "v1"),
        "trials": trials,
        "successes": successes,
        "lower_confidence_bound": wilson_lcb(successes, trials),
    }


def exact_mcnemar(
    pairs: list[tuple[bool, bool]], *, seed: int = 0
) -> dict[str, object]:
    """Exact two-sided McNemar test over paired binary outcomes."""

    _ = seed
    if not pairs:
        raise ValueError("McNemar analysis needs paired outcomes")
    discordant_a = sum(1 for a, b in pairs if a and not b)
    discordant_b = sum(1 for a, b in pairs if not a and b)
    discordant = discordant_a + discordant_b
    if discordant == 0:
        raise ValueError("McNemar analysis needs discordant pairs")
    extreme = _binomial_tail(discordant, max(discordant_a, discordant_b))
    return {
        "analysis_id": "anra-v5-stats/exact-mcnemar-v1",
        "implementation_sha256": _impl_sha("anra-v5-stats/exact-mcnemar-v1", "v1"),
        "discordant_pairs": discordant,
        "two_sided_p_value": min(1.0, 2 * extreme),
    }


def cluster_bootstrap_delta(
    records: list[Mapping[str, Any]], *, seed: int, resamples: int = 10000
) -> dict[str, object]:
    """Percentile bootstrap CI of the cluster-mean correctness delta vs chance."""

    if resamples <= 0:
        raise ValueError("resample count must be positive")
    clusters: dict[str, list[bool]] = {}
    for record in records:
        clusters.setdefault(str(record.get("cluster_id", "")), []).append(
            bool(record.get("correct"))
        )
    if len(clusters) < 2:
        raise ValueError("cluster bootstrap needs at least two clusters")
    base = [sum(values) / len(values) for values in clusters.values()]
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = [base[rng.randrange(len(base))] for _ in range(len(base))]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    lower = draws[int(0.025 * resamples)]
    upper = draws[int(0.975 * resamples) - 1]
    return {
        "analysis_id": "anra-v5-stats/cluster-bootstrap-delta-v1",
        "implementation_sha256": _impl_sha("anra-v5-stats/cluster-bootstrap-delta-v1", "v1"),
        "clusters": len(clusters),
        "resamples": resamples,
        "mean": sum(base) / len(base),
        "ci95": [lower, upper],
    }


STATISTICAL_RULES: dict[str, Callable[..., dict[str, object]]] = {
    "WILSON_BINOMIAL": wilson_binomial,
    "EXACT_MCNEMAR": exact_mcnemar,
    "CLUSTER_BOOTSTRAP_DELTA": cluster_bootstrap_delta,
}


__all__ = [
    "STATISTICAL_RULES",
    "cluster_bootstrap_delta",
    "exact_mcnemar",
    "wilson_binomial",
]
