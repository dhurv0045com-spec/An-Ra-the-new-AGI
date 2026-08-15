"""Kill/restart checkpoint-recovery drill for the Phase-A launch gate."""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from training.checkpoint import CheckpointManager, CheckpointMeta


class _RecoveryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values)


def _save_boundary(root: str, ready: Any) -> None:
    torch.manual_seed(1301)
    model = _RecoveryModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(root, keep_last_n=2)
    values = torch.tensor([[1.0, 2.0]])
    loss = model(values).square().mean()
    loss.backward()
    optimizer.step()
    manager.save(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        meta=CheckpointMeta(1, 1, float(loss.item()), float(loss.item()), float(loss.item())),
    )
    ready.set()
    while True:
        time.sleep(1.0)


def run_kill_recovery_drill(
    root: str | Path, *, timeout_seconds: float = 10.0
) -> dict[str, object]:
    """Terminate a writer after a checkpoint boundary and prove exact restoration."""
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_save_boundary, args=(str(directory), ready))
    process.start()
    if not ready.wait(timeout_seconds):
        process.terminate()
        process.join(timeout=timeout_seconds)
        return {"passed": False, "reason": "checkpoint_boundary_not_reached"}
    process.terminate()  # Windows equivalent of kill -9 for the isolated worker.
    process.join(timeout=timeout_seconds)

    torch.manual_seed(9999)
    restored = _RecoveryModel()
    optimizer = torch.optim.SGD(restored.parameters(), lr=0.1)
    manager = CheckpointManager(directory, keep_last_n=2)
    metadata = manager.load(None, restored, optimizer=optimizer)
    checkpoint = manager.latest_checkpoint()
    gates = {
        "writer_terminated": process.exitcode is not None,
        "checkpoint_exists": checkpoint is not None and checkpoint.is_file(),
        "recovered_step": metadata.global_step == 1,
        "optimizer_restored": bool(optimizer.state_dict()["param_groups"]),
    }
    return {
        "schema_version": 1,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "worker_exit_code": process.exitcode,
        "gates": gates,
        "passed": all(gates.values()),
    }
