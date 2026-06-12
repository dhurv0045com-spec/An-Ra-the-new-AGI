"""Measured CPU/CUDA memory and throughput profiling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Callable

import torch


@dataclass(frozen=True)
class ProfileReport:
    device: str
    elapsed_seconds: float
    tokens: int
    tokens_per_second: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    parameter_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _optimizer_bytes(optimizer: torch.optim.Optimizer | None) -> int:
    if optimizer is None:
        return 0
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += _tensor_bytes(value)
    return total


def profile_step(
    model: torch.nn.Module,
    run_step: Callable[[], None],
    *,
    tokens: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    run_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    parameter_bytes = sum(_tensor_bytes(p) for p in model.parameters())
    gradient_bytes = sum(_tensor_bytes(p.grad) for p in model.parameters() if p.grad is not None)
    report = ProfileReport(
        device=str(device),
        elapsed_seconds=elapsed,
        tokens=int(tokens),
        tokens_per_second=float(tokens) / max(elapsed, 1e-9),
        peak_allocated_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        peak_reserved_bytes=(
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_state_bytes=_optimizer_bytes(optimizer),
    )
    return asdict(report)
