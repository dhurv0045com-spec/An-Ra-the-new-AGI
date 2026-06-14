"""Evidence-maturity-aware AGI benchmark registry and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Literal


EvidenceMaturity = Literal["automated", "human_reviewed", "longitudinal", "insufficient_data"]


@dataclass(frozen=True)
class AGIBenchmarkSpec:
    benchmark_id: str
    name: str
    sample_requirement: int
    target: str
    promotion_blocking: bool


@dataclass(frozen=True)
class AGIBenchmarkResult:
    benchmark_id: str
    value: float | None
    sample_count: int
    maturity: EvidenceMaturity
    passing: bool | None
    evidence_path: str | None = None


SPECS = (
    AGIBenchmarkSpec("A-01", "Causal accuracy", 200, ">0.80", True),
    AGIBenchmarkSpec("A-02", "Epistemic calibration Brier", 500, "<0.15", True),
    AGIBenchmarkSpec("A-03", "Human-model blind prediction", 10, ">0.70", False),
    AGIBenchmarkSpec("A-04", "SSIE experiment confirmation", 10, ">0.60", False),
    AGIBenchmarkSpec("A-05", "Expert-reviewed synthesis", 20, ">0.40", False),
    AGIBenchmarkSpec("A-06", "Quality trend", 50, "positive", False),
    AGIBenchmarkSpec("A-07", "Flourishing", 2, "improvement in 2/3", False),
)


def evaluate_result(benchmark_id: str, value: float, sample_count: int) -> bool | None:
    if benchmark_id == "A-01":
        return sample_count >= 200 and value > 0.80
    if benchmark_id == "A-02":
        return sample_count >= 500 and value < 0.15
    if benchmark_id in {"A-03", "A-04", "A-05"}:
        threshold = {"A-03": 0.70, "A-04": 0.60, "A-05": 0.40}[benchmark_id]
        required = {"A-03": 10, "A-04": 10, "A-05": 20}[benchmark_id]
        return sample_count >= required and value > threshold
    if benchmark_id == "A-06":
        return sample_count >= 50 and value > 0
    if benchmark_id == "A-07":
        return sample_count >= 2 and value >= 2
    raise ValueError(f"Unknown AGI benchmark: {benchmark_id}")


def build_report(measurements: dict[str, tuple[float, int, EvidenceMaturity, str | None]]) -> dict[str, object]:
    results = []
    for spec in SPECS:
        measurement = measurements.get(spec.benchmark_id)
        if measurement is None:
            result = AGIBenchmarkResult(spec.benchmark_id, None, 0, "insufficient_data", None)
        else:
            value, samples, maturity, path = measurement
            result = AGIBenchmarkResult(
                spec.benchmark_id,
                float(value),
                int(samples),
                maturity,
                evaluate_result(spec.benchmark_id, float(value), int(samples)),
                path,
            )
        results.append(asdict(result))
    return {
        "schema_version": 1,
        "generated_at": time.time(),
        "specs": [asdict(spec) for spec in SPECS],
        "results": results,
        "promotion_ready": all(
            result["passing"] is True
            for result in results
            if next(spec for spec in SPECS if spec.benchmark_id == result["benchmark_id"]).promotion_blocking
        ),
    }


def write_report(report: dict[str, object], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
