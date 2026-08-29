"""Validate and aggregate replicated E2 full-stack execution receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        raise ValueError("benchmark measurements must be positive")
    return numerator / denominator


def aggregate_receipts(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 3:
        raise ValueError("full-stack replication requires at least three receipts")
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for receipt in receipts:
        if receipt.get("schema") != "esoes-e2-full-stack-benchmark/v1":
            raise ValueError("unexpected full-stack receipt schema")
        if receipt.get("status") != "PASS":
            raise ValueError("only passing full-stack receipts may be aggregated")
    fixed_fields = (
        "scope",
        "implementation_sha256",
        "static_plan_sha256",
        "torch_version",
        "cuda_runtime",
        "device_name",
    )
    for field in fixed_fields:
        if len({json.dumps(receipt[field], sort_keys=True) for receipt in receipts}) != 1:
            raise ValueError(f"replicated receipts disagree on {field}")
    fixed_config = {
        key: value for key, value in receipts[0]["config"].items() if key != "seed"
    }
    for receipt in receipts[1:]:
        candidate = {key: value for key, value in receipt["config"].items() if key != "seed"}
        if candidate != fixed_config:
            raise ValueError("replicated receipts disagree on benchmark configuration")
    seeds = [int(receipt["config"]["seed"]) for receipt in receipts]
    if len(set(seeds)) != len(seeds):
        raise ValueError("replication seeds must be distinct")

    expected_keys = [(row["arm"], int(row["sequence_length"])) for row in receipts[0]["rows"]]
    for receipt in receipts[1:]:
        keys = [(row["arm"], int(row["sequence_length"])) for row in receipt["rows"]]
        if keys != expected_keys:
            raise ValueError("replicated receipts disagree on ordered arm/context rows")

    rows: list[dict[str, Any]] = []
    for arm, sequence_length in expected_keys:
        source_rows = [
            next(
                row
                for row in receipt["rows"]
                if row["arm"] == arm and int(row["sequence_length"]) == sequence_length
            )
            for receipt in receipts
        ]
        for row in source_rows:
            checks = row["correctness"]
            if not (
                checks["parameter_count_exact"]
                and checks["finite_loss"]
                and checks["all_gradients_finite"]
            ):
                raise ValueError("a replicated full-stack row failed correctness")
        forward = [float(row["forward"]["median_ms"]) for row in source_rows]
        backward = [float(row["forward_backward"]["median_ms"]) for row in source_rows]
        memory = [int(row["forward_backward_peak_allocated_bytes"]) for row in source_rows]
        rows.append(
            {
                "arm": arm,
                "sequence_length": sequence_length,
                "parameters": int(source_rows[0]["parameters"]),
                "forward_median_ms_across_runs": statistics.median(forward),
                "forward_median_ms_range": [min(forward), max(forward)],
                "forward_backward_median_ms_across_runs": statistics.median(backward),
                "forward_backward_median_ms_range": [min(backward), max(backward)],
                "forward_backward_peak_allocated_bytes": statistics.median(memory),
                "tokens_per_second": sequence_length
                / (statistics.median(backward) / 1_000.0),
            }
        )
    by_key = {(row["arm"], row["sequence_length"]): row for row in rows}
    comparisons: dict[str, dict[str, float]] = {}
    for sequence_length in sorted({row["sequence_length"] for row in rows}):
        latency = lambda arm: float(
            by_key[(arm, sequence_length)]["forward_backward_median_ms_across_runs"]
        )
        memory = lambda arm: float(
            by_key[(arm, sequence_length)]["forward_backward_peak_allocated_bytes"]
        )
        comparisons[str(sequence_length)] = {
            "deep_vs_middle_latency_ratio": _ratio(latency("deep-narrow"), latency("middle")),
            "deep_vs_wide_latency_ratio": _ratio(latency("deep-narrow"), latency("wide-shallow")),
            "middle_vs_wide_latency_ratio": _ratio(latency("middle"), latency("wide-shallow")),
            "deep_vs_wide_memory_ratio": _ratio(memory("deep-narrow"), memory("wide-shallow")),
            "middle_vs_wide_memory_ratio": _ratio(memory("middle"), memory("wide-shallow")),
        }
    context_scaling: dict[str, dict[str, float]] = {}
    lengths = sorted({row["sequence_length"] for row in rows})
    if len(lengths) >= 2:
        shortest, longest = lengths[0], lengths[-1]
        for arm in ("deep-narrow", "middle", "wide-shallow"):
            left = by_key[(arm, shortest)]
            right = by_key[(arm, longest)]
            context_scaling[arm] = {
                "long_vs_short_length_ratio": longest / shortest,
                "training_latency_ratio": _ratio(
                    float(right["forward_backward_median_ms_across_runs"]),
                    float(left["forward_backward_median_ms_across_runs"]),
                ),
                "peak_memory_ratio": _ratio(
                    float(right["forward_backward_peak_allocated_bytes"]),
                    float(left["forward_backward_peak_allocated_bytes"]),
                ),
            }
    return {
        "schema": "esoes-e2-full-stack-aggregate/v1",
        "status": "PASS_REPLICATED",
        "scope": receipts[0]["scope"],
        "implementation_sha256": receipts[0]["implementation_sha256"],
        "static_plan_sha256": receipts[0]["static_plan_sha256"],
        "device_name": receipts[0]["device_name"],
        "torch_version": receipts[0]["torch_version"],
        "cuda_runtime": receipts[0]["cuda_runtime"],
        "config": fixed_config,
        "seeds": seeds,
        "source_receipts": [
            {"path": path.name, "sha256": _sha256_file(path)} for path in paths
        ],
        "rows": rows,
        "shape_comparisons": comparisons,
        "context_scaling": context_scaling,
        "interpretation": (
            "This exact eager RTX stack strongly favors fewer wider blocks for execution. "
            "The result is a throughput/memory prior only; E2 cognition per measured FLOP "
            "must decide whether the deeper shapes earn their cost."
        ),
        "limitations": receipts[0]["limitations"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate_receipts(args.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
