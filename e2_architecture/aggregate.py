"""Validate and aggregate replicated E2 CUDA attention receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def _sha256_json(path: Path) -> str:
    """Hash JSON semantics, not platform-dependent newline bytes."""
    value = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        raise ValueError("benchmark latency must be positive")
    return numerator / denominator


def aggregate_receipts(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 3:
        raise ValueError("replicated device evidence requires at least three receipts")
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for receipt in receipts:
        if receipt.get("schema") != "esoes-e2-device-benchmark/v1" or receipt.get("status") != "PASS":
            raise ValueError("only passing E2 device receipts may be aggregated")
    fixed_fields = (
        "implementation_sha256",
        "device_name",
        "device_total_memory_bytes",
        "torch_version",
        "cuda_runtime",
        "bf16_supported",
        "warmup",
        "repeats",
    )
    for field in fixed_fields:
        if len({json.dumps(receipt[field], sort_keys=True) for receipt in receipts}) != 1:
            raise ValueError(f"replicated receipts disagree on {field}")
    seeds = [int(receipt["seed"]) for receipt in receipts]
    if len(set(seeds)) != len(seeds):
        raise ValueError("replicated receipts require distinct seeds")
    names = [row["name"] for row in receipts[0]["cases"]]
    if any([row["name"] for row in receipt["cases"]] != names for receipt in receipts[1:]):
        raise ValueError("replicated receipts disagree on ordered benchmark cases")

    rows: list[dict[str, Any]] = []
    for name in names:
        per_run = [next(row for row in receipt["cases"] if row["name"] == name) for receipt in receipts]
        training = [float(row["forward_backward"]["median_ms"]) for row in per_run]
        forward = [float(row["forward"]["median_ms"]) for row in per_run]
        memory = [int(row["forward_backward_peak_allocated_bytes"]) for row in per_run]
        rows.append(
            {
                "name": name,
                "forward_median_ms_across_runs": statistics.median(forward),
                "forward_median_ms_range": [min(forward), max(forward)],
                "forward_backward_median_ms_across_runs": statistics.median(training),
                "forward_backward_median_ms_range": [min(training), max(training)],
                "forward_backward_peak_allocated_bytes": statistics.median(memory),
            }
        )
    by_name = {row["name"]: row for row in rows}
    latency = lambda name: float(by_name[name]["forward_backward_median_ms_across_runs"])
    memory = lambda name: float(by_name[name]["forward_backward_peak_allocated_bytes"])
    comparisons = {
        "native_gqa_vs_mha_training_latency_ratio": _ratio(latency("gqa-qk-2k"), latency("mha-qk-2k")),
        "repeat_kv_gqa_vs_mha_training_latency_ratio": _ratio(
            latency("gqa-repeat-kv-qk-2k"), latency("mha-qk-2k")
        ),
        "native_qk_norm_training_latency_ratio": _ratio(
            latency("gqa-qk-2k"), latency("gqa-no-qk-2k")
        ),
        "native_gqa_4k_vs_2k_training_latency_ratio": _ratio(
            latency("gqa-qk-4k"), latency("gqa-qk-2k")
        ),
        "native_gqa_4k_vs_2k_training_memory_ratio": _ratio(
            memory("gqa-qk-4k"), memory("gqa-qk-2k")
        ),
        "native_gqa_vs_mha_training_memory_ratio": _ratio(
            memory("gqa-qk-2k"), memory("mha-qk-2k")
        ),
        "repeat_kv_gqa_vs_mha_training_memory_ratio": _ratio(
            memory("gqa-repeat-kv-qk-2k"), memory("mha-qk-2k")
        ),
    }
    backend_support = [receipt["native_gqa_backend_support"] for receipt in receipts]
    if any(value != backend_support[0] for value in backend_support[1:]):
        raise ValueError("native GQA backend support changed between replications")
    return {
        "schema": "esoes-e2-device-benchmark-aggregate/v1",
        "status": "PASS_REPLICATED",
        "scope": receipts[0]["scope"],
        "device_name": receipts[0]["device_name"],
        "torch_version": receipts[0]["torch_version"],
        "cuda_runtime": receipts[0]["cuda_runtime"],
        "implementation_sha256": receipts[0]["implementation_sha256"],
        "seeds": seeds,
        "source_receipts": [
            {"path": path.name, "sha256": _sha256_json(path)} for path in paths
        ],
        "native_gqa_backend_support": backend_support[0],
        "gqa_equivalence_maximum_absolute_error": max(
            float(receipt["gqa_equivalence"]["maximum_absolute_error"])
            for receipt in receipts
        ),
        "cases": rows,
        "comparisons": comparisons,
        "interpretation": (
            "On this exact Windows/PyTorch/CUDA stack, native GQA is confined to the math "
            "backend. Repeated-K/V GQA recovers a faster fused path at higher activation memory. "
            "This is an implementation selection result, not evidence against GQA on TPU."
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
