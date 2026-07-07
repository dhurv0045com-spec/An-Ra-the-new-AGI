from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from runtime.experience_ledger import ExperienceLedger, content_hash


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def run_write_benchmark(
    root: str | Path,
    *,
    events: int = 1_000,
    max_shard_bytes: int = 64 * 1024 * 1024,
    seal: bool = True,
) -> dict[str, Any]:
    ledger = ExperienceLedger(root, strict=True, max_shard_bytes=max_shard_bytes)
    latencies_ms: list[float] = []
    for index in range(events):
        start = time.perf_counter_ns()
        ledger.record(
            trace_id=f"bench-{index}",
            kind="benchmark",
            inputs={"index": index, "payload_hash": content_hash(index)},
            output={"ok": True},
            gate_record={"allowed": True, "gate": "benchmark"},
            latency={"synthetic_ms": 0.0},
        )
        latencies_ms.append((time.perf_counter_ns() - start) / 1_000_000)

    event_count = sum(1 for _ in ledger.iter_events())
    manifest = ledger.seal_shards(include_active=True) if seal else None
    verification = ledger.verify_sealed_manifest() if seal else None
    return {
        "events": events,
        "validated_events": event_count,
        "write_failures": ledger.write_failures,
        "p50_ms": statistics.median(latencies_ms),
        "p99_ms": _percentile(latencies_ms, 0.99),
        "max_ms": max(latencies_ms) if latencies_ms else 0.0,
        "sealed_shards": len(manifest["shards"]) if manifest else 0,
        "seal_verified": bool(verification["verified"]) if verification else False,
    }


def _stress_worker(root: str, worker_id: int, events: int) -> None:
    ledger = ExperienceLedger(root, strict=True, max_shard_bytes=4096)
    for index in range(events):
        ledger.record(
            trace_id=f"stress-{worker_id}-{index}",
            kind="stress",
            inputs={"worker": worker_id, "index": index},
            output={"worker": worker_id, "index": index},
            gate_record={"allowed": True, "gate": "stress"},
        )


def run_crash_flush_stress(
    root: str | Path,
    *,
    workers: int = 4,
    events_per_worker: int = 100,
    terminate_one: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    processes = [
        mp.Process(target=_stress_worker, args=(str(root), worker_id, events_per_worker))
        for worker_id in range(workers)
    ]
    for process in processes:
        process.start()
    if terminate_one and processes:
        time.sleep(0.05)
        processes[0].terminate()
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)

    ledger = ExperienceLedger(root, strict=True)
    validated = sum(1 for _ in ledger.iter_events())
    manifest = ledger.seal_shards(include_active=True)
    verification = ledger.verify_sealed_manifest()
    return {
        "workers": workers,
        "events_per_worker": events_per_worker,
        "terminate_one": terminate_one,
        "validated_events": validated,
        "sealed_shards": len(manifest["shards"]),
        "seal_verified": bool(verification["verified"]),
        "exit_codes": [process.exitcode for process in processes],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark and stress the Experience Ledger")
    parser.add_argument("--events", type=int, default=1_000)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--max-p50-ms", type=float, default=10.0)
    parser.add_argument("--max-p99-ms", type=float, default=50.0)
    parser.add_argument("--crash-stress", action="store_true")
    parser.add_argument("--stress-workers", type=int, default=4)
    parser.add_argument("--stress-events", type=int, default=100)
    args = parser.parse_args()

    temporary = args.root is None
    root = args.root or Path(tempfile.mkdtemp(prefix="anra-ledger-bench-"))
    try:
        benchmark = run_write_benchmark(root, events=args.events)
        result: dict[str, Any] = {"benchmark": benchmark}
        if args.crash_stress:
            result["crash_stress"] = run_crash_flush_stress(
                root / "crash",
                workers=args.stress_workers,
                events_per_worker=args.stress_events,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        if benchmark["validated_events"] != args.events or benchmark["write_failures"]:
            return 1
        if benchmark["p50_ms"] > args.max_p50_ms or benchmark["p99_ms"] > args.max_p99_ms:
            return 2
        if args.crash_stress and not result["crash_stress"]["seal_verified"]:
            return 3
        return 0
    finally:
        if temporary:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
