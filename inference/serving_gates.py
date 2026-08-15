"""Distribution-parity and latency gates for optional serving accelerators."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from statistics import quantiles

from inference.speculative import SpeculativeBenchmark


@dataclass(frozen=True)
class LatencySample:
    ttft_ms: float
    decode_tokens: int
    decode_ms: float
    verified: bool = False

    @property
    def decode_tokens_per_second(self) -> float:
        return self.decode_tokens / max(self.decode_ms / 1000.0, 1e-9)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=100, method="inclusive")[max(0, int(fraction * 100) - 1)]


def evaluate_latency_budget(
    samples: Iterable[LatencySample],
    *,
    max_ttft_p95_ms: float = 300.0,
    min_decode_tokens_per_second: float = 25.0,
    max_verified_p95_multiplier: float = 1.6,
) -> dict[str, object]:
    """Evaluate measured samples; empty or malformed inputs fail closed."""
    rows = list(samples)
    invalid = any(
        row.ttft_ms < 0 or row.decode_ms <= 0 or row.decode_tokens < 0 for row in rows
    )
    if not rows or invalid:
        return {
            "passed": False,
            "reason": "missing_or_invalid_latency_samples",
            "samples": len(rows),
        }
    ttfts = sorted(row.ttft_ms for row in rows)
    speeds = [row.decode_tokens_per_second for row in rows]
    verified = [row.ttft_ms + row.decode_ms for row in rows if row.verified]
    unverified = [row.ttft_ms + row.decode_ms for row in rows if not row.verified]
    verified_multiplier = (
        _percentile(verified, 0.95) / max(_percentile(unverified, 0.95), 1e-9)
        if verified and unverified
        else None
    )
    gates = {
        "ttft_p95": _percentile(ttfts, 0.95) <= max_ttft_p95_ms,
        "decode_rate": min(speeds) >= min_decode_tokens_per_second,
        "verified_latency": (
            verified_multiplier is not None
            and verified_multiplier <= max_verified_p95_multiplier
        ),
    }
    return {
        "schema_version": 1,
        "samples": len(rows),
        "ttft_p50_ms": _percentile(ttfts, 0.50),
        "ttft_p95_ms": _percentile(ttfts, 0.95),
        "decode_tokens_per_second_min": min(speeds),
        "verified_end_to_end_p95_ms": _percentile(verified, 0.95) if verified else None,
        "unverified_end_to_end_p95_ms": _percentile(unverified, 0.95) if unverified else None,
        "verified_p95_multiplier": verified_multiplier,
        "budgets": {
            "max_ttft_p95_ms": max_ttft_p95_ms,
            "min_decode_tokens_per_second": min_decode_tokens_per_second,
            "max_verified_p95_multiplier": max_verified_p95_multiplier,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "sample_rows": [asdict(row) for row in rows],
    }


def evaluate_accelerator_gate(
    *,
    speculative: SpeculativeBenchmark,
    parity: Mapping[str, object],
    qat_max_relative_error: float,
    latency: Mapping[str, object],
) -> dict[str, object]:
    """Require parity, QAT quality, speculative benefit, and latency together."""
    gates = {
        "speculative_promotion": speculative.promotion_allowed,
        "token_parity": parity.get("token_parity") is True,
        "distribution_parity": parity.get("distribution_parity") is True,
        "qat_delta": float(qat_max_relative_error) <= 0.01,
        "latency_budget": latency.get("passed") is True,
    }
    return {
        "schema_version": 1,
        "gates": gates,
        "passed": all(gates.values()),
        "speculative": asdict(speculative),
        "parity": dict(parity),
        "qat_max_relative_error": float(qat_max_relative_error),
        "latency": dict(latency),
    }
