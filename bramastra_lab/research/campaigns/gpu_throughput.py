"""Paired dual-GPU microbatch throughput experiment for the K8 model.

Each physical GPU runs the same model and batch-size schedule concurrently.
This is an optimization pilot only: random byte-token batches measure training
throughput, not learning quality. It never changes the frozen K8 campaign.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def parse_batch_sizes(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as exc:
        raise ValueError("batch sizes must be comma-separated positive integers") from exc
    if not values or any(value <= 0 for value in values) or len(set(values)) != len(values):
        raise ValueError("batch sizes must be unique positive integers")
    return values


def summarize_samples(rows: list[dict[str, float]]) -> dict[str, float | int | None]:
    if not rows:
        return {"samples": 0, "mean_gpu_utilization_pct": None,
                "p50_gpu_utilization_pct": None, "peak_memory_mib": None}
    utils = [row["utilization_pct"] for row in rows]
    memories = [row["memory_mib"] for row in rows]
    return {
        "samples": len(rows),
        "mean_gpu_utilization_pct": round(statistics.mean(utils), 2),
        "p50_gpu_utilization_pct": round(statistics.median(utils), 2),
        "peak_memory_mib": round(max(memories), 1),
    }


def recommend_batch_size(results: list[dict[str, Any]], telemetry: dict[str, Any],
                         target_utilization_pct: float) -> dict[str, Any]:
    """Choose the smallest measured batch meeting the utilization target.

    If telemetry never reaches the target, choose the highest-throughput
    completed condition and label the unmet target explicitly.
    """
    if not 1 <= target_utilization_pct <= 100:
        raise ValueError("target utilization must be in [1, 100]")
    by_batch = telemetry.get("by_batch_size", {})
    eligible = [row for row in results if row.get("status") == "complete"
                and row.get("finite") is True
                and (by_batch.get(str(row.get("batch_size")), {}).get(
                    "mean_gpu_utilization_pct") or 0) >= target_utilization_pct]
    if eligible:
        chosen = min(eligible, key=lambda row: int(row["batch_size"]))
        return {"batch_size": int(chosen["batch_size"]),
                "reason": "smallest measured batch meeting utilization target",
                "target_met": True,
                "measured_utilization_pct": by_batch[str(chosen["batch_size"])]
                ["mean_gpu_utilization_pct"]}
    completed = [row for row in results if row.get("status") == "complete"
                 and row.get("finite") is True]
    if not completed:
        return {"batch_size": None, "reason": "no finite completed candidate",
                "target_met": False}
    chosen = max(completed, key=lambda row: float(row.get("tokens_per_second", 0)))
    return {"batch_size": int(chosen["batch_size"]),
            "reason": "best observed throughput; utilization target not reached",
            "target_met": False,
            "measured_utilization_pct": by_batch.get(str(chosen["batch_size"]), {}).get(
                "mean_gpu_utilization_pct")}


def _sample_nvidia_smi() -> list[dict[str, float]]:
    command = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
               "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, capture_output=True, text=True,
                               timeout=5, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi query failed")
    rows: list[dict[str, float]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        index, utilization, memory = map(float, fields)
        rows.append({"device_index": index, "utilization_pct": utilization,
                     "memory_mib": memory})
    return rows


def _worker(*, device_index: int, batch_sizes: tuple[int, ...], seconds: float,
            sequence_length: int, seed: int, initial_state: dict[str, Any],
            output: dict[str, Any], errors: list[str], barrier: threading.Barrier,
            active_batch: dict[int, int | None]) -> None:
    import torch
    from bramastra_lab.research.campaigns.phases.ops import k8_campaign_config
    from bramastra_lab.research.models.wrapper import IntegratedModel

    try:
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
        config = k8_campaign_config()
        generator = torch.Generator(device=device).manual_seed(seed + device_index)
        device_rows: list[dict[str, Any]] = []
        barrier.wait(timeout=60)
        for batch_size in batch_sizes:
            model = optimizer = step = None
            try:
                # Every condition starts from the same from-scratch weights and
                # fresh optimizer; only the per-update microbatch changes.
                model = IntegratedModel(config).to(device)
                model.load_state_dict(initial_state)
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                              weight_decay=0.01)
                tokens = torch.randint(0, config.model.vocab,
                                       (batch_size, sequence_length),
                                       device=device, generator=generator)
                targets = torch.roll(tokens, shifts=-1, dims=1)
                active_batch[device_index] = batch_size
                def step() -> Any:
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(tokens).logits
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    return loss.detach()

                step()
                step()
                torch.cuda.synchronize(device)
                started = time.monotonic()
                deadline = started + seconds
                steps = 0
                last_loss = None
                while time.monotonic() < deadline:
                    last_loss = step()
                    steps += 1
                torch.cuda.synchronize(device)
                elapsed = time.monotonic() - started
                final_loss = float(last_loss) if last_loss is not None else float("nan")
                device_rows.append({
                    "batch_size": batch_size, "status": "complete",
                    "seconds": round(elapsed, 3), "optimizer_steps": steps,
                    "tokens_per_second": round(steps * batch_size * sequence_length / elapsed, 1),
                    "final_loss": round(final_loss, 6),
                    "finite": math.isfinite(final_loss),
                    "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
                })
                torch.cuda.reset_peak_memory_stats(device)
            except torch.cuda.OutOfMemoryError:
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                device_rows.append({"batch_size": batch_size, "status": "oom"})
            except Exception as exc:
                device_rows.append({"batch_size": batch_size, "status": "error",
                                    "error": f"{type(exc).__name__}: {exc}"[:500]})
            finally:
                active_batch[device_index] = None
                model = optimizer = step = None
                torch.cuda.empty_cache()
            output[str(device_index)] = device_rows
        output[str(device_index)] = device_rows
        torch.cuda.empty_cache()
    except Exception as exc:  # thread failures must become explicit report rows
        errors.append(f"cuda:{device_index}: {type(exc).__name__}: {exc}")


def run(*, devices: tuple[int, int] = (0, 1), batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        seconds_per_case: float = 20.0, sequence_length: int = 512,
        seed: int = 1701, target_utilization_pct: float = 75.0) -> dict[str, Any]:
    import torch
    from bramastra_lab.research.campaigns.phases.ops import k8_campaign_config
    from bramastra_lab.research.models.wrapper import IntegratedModel

    if devices[0] == devices[1] or min(devices) < 0:
        raise ValueError("two distinct non-negative CUDA device indices are required")
    if seconds_per_case < 2 or sequence_length < 8:
        raise ValueError("each case needs >=2 seconds and sequence length >=8")
    if not 1 <= target_utilization_pct <= 100:
        raise ValueError("target utilization must be in [1, 100]")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= max(devices):
        raise RuntimeError("two requested CUDA devices are not available")
    if sequence_length > 512:
        raise ValueError("sequence length exceeds frozen K8 campaign context (512)")

    config = k8_campaign_config()
    torch.manual_seed(seed)
    reference = IntegratedModel(config)
    initial_state = {key: value.detach().cpu().clone()
                     for key, value in reference.state_dict().items()}
    del reference
    outputs: dict[str, Any] = {}
    errors: list[str] = []
    active_batch: dict[int, int | None] = {index: None for index in devices}
    barrier = threading.Barrier(2)
    threads = [threading.Thread(
        target=_worker,
        kwargs={"device_index": index, "batch_sizes": batch_sizes,
                "seconds": seconds_per_case, "sequence_length": sequence_length,
                "seed": seed, "initial_state": initial_state,
                "output": outputs, "errors": errors, "barrier": barrier,
                "active_batch": active_batch},
        daemon=True,
    ) for index in devices]
    sampler_rows: dict[int, list[dict[str, float]]] = {index: [] for index in devices}
    stop_sampling = threading.Event()

    def sample_loop() -> None:
        while not stop_sampling.is_set():
            try:
                for row in _sample_nvidia_smi():
                    index = int(row["device_index"])
                    if index in sampler_rows:
                        row["batch_size"] = active_batch[index] or 0
                        sampler_rows[index].append(row)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
                pass  # performance telemetry is optional; training report survives
            stop_sampling.wait(1.0)

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=len(batch_sizes) * seconds_per_case + 180)
    stop_sampling.set()
    sampler.join(timeout=6)
    alive = [thread.name for thread in threads if thread.is_alive()]
    if alive:
        errors.append(f"worker thread timeout: {alive}")
    report = {
        "schema": "bramastra-dual-gpu-throughput/v1",
        "purpose": "paired from-scratch training throughput pilot; no capability claim",
        "devices": {str(index): torch.cuda.get_device_name(index) for index in devices},
        "batch_sizes": list(batch_sizes), "sequence_length": sequence_length,
        "seconds_per_case": seconds_per_case,
        "same_initialization": True,
        "results": outputs,
        "gpu_telemetry": {
            str(index): {
                "all_cases": summarize_samples(sampler_rows[index]),
                "by_batch_size": {
                    str(batch): summarize_samples([
                        row for row in sampler_rows[index]
                        if row.get("batch_size") == batch])
                    for batch in batch_sizes
                },
            } for index in devices
        },
        "errors": errors,
        "utilization_target_pct": target_utilization_pct,
        "recommendations": {
            str(index): recommend_batch_size(
                outputs.get(str(index), []),
                {"by_batch_size": {
                    str(batch): summarize_samples([
                        row for row in sampler_rows[index]
                        if row.get("batch_size") == batch])
                    for batch in batch_sizes}},
                target_utilization_pct,
            ) for index in devices
        },
        "quality_warning": "Random-token throughput probe only; it does not test task quality, speech transcription, or the frozen K8 objectives.",
    }
    report["status"] = "complete" if not errors and all(
        len(outputs.get(str(index), [])) == len(batch_sizes) for index in devices) else "partial"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64")
    parser.add_argument("--seconds-per-case", type=float, default=20.0)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--target-utilization", type=float, default=75.0)
    parser.add_argument("--out", type=Path, default=Path("dual_gpu_throughput.json"))
    args = parser.parse_args()
    try:
        devices = tuple(int(value.strip()) for value in args.devices.split(","))
        if len(devices) != 2:
            raise ValueError("--devices must contain exactly two indices")
        report = run(devices=devices, batch_sizes=parse_batch_sizes(args.batch_sizes),
                     seconds_per_case=args.seconds_per_case,
                     sequence_length=args.sequence_length, seed=args.seed,
                     target_utilization_pct=args.target_utilization)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
