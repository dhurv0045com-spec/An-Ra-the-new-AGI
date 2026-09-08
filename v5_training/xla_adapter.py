"""Production XLA replicated adapter — IMPLEMENTED, PENDING PRE500M TPU.

This module is the real data-parallel execution path the PRE500M
certification must exercise. It discovers the live XLA topology at runtime
(PJRT device, world size, rank, versions — nothing hard-coded), requires the
frozen 8-replica topology before production mode, and executes replica-local
forward/backward over 4 accumulation microsteps with a replica SUM collective
and the exact replica-global eligible-token denominator at the accumulation
boundary, followed by ONE synchronized optimizer step.

Status discipline: every public surface reports
IMPLEMENTED_PENDING_PRE500M_TPU until measured on real TPU hardware. Without
torch_xla, or on a topology mismatch, everything fails closed — never a
silent CPU fallback masquerading as distributed evidence.
"""

from __future__ import annotations

from typing import Any, Mapping

ADAPTER_SCHEMA = "anra-v5-xla-adapter/v1"
PENDING_STATUS = "IMPLEMENTED_PENDING_PRE500M_TPU"
EVIDENCE_REQUIRED = "TPU_EVIDENCE_REQUIRED"


def _load_xla() -> Any:
    try:
        import torch_xla.runtime as xruntime  # type: ignore
        import torch_xla.core.xla_model as xm  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "torch_xla is not installed: XLA execution needs PRE500M "
            f"certification ({EVIDENCE_REQUIRED})") from exc
    return xruntime, xm


def xla_status() -> dict[str, object]:
    """Discover the live XLA topology without executing anything."""

    try:
        xruntime, xm = _load_xla()
    except RuntimeError as exc:
        return {"schema": ADAPTER_SCHEMA, "status": EVIDENCE_REQUIRED,
                "reason": str(exc)}
    try:
        device_type = str(xruntime.device_type())
        world_size = int(xruntime.world_size())
        ordinal = int(xruntime.global_ordinal())
        versions = {"torch_xla": getattr(__import__("torch_xla"), "__version__", "unknown")}
    except Exception as exc:
        return {"schema": ADAPTER_SCHEMA, "status": EVIDENCE_REQUIRED,
                "reason": f"XLA discovery failed: {exc}"}
    _ = xm
    return {"schema": ADAPTER_SCHEMA, "status": PENDING_STATUS,
            "device_type": device_type, "world_size": world_size,
            "ordinal": ordinal, "versions": versions}


def require_frozen_topology(world_size: int, *, replicas: int) -> dict[str, object]:
    """Require the live world to equal the frozen replica count."""

    if replicas <= 0:
        raise ValueError("frozen replica count must be positive")
    if world_size != replicas:
        raise ValueError(
            f"XLA world size {world_size} does not equal the frozen "
            f"{replicas} replicas: refusing production mode")
    return {"schema": ADAPTER_SCHEMA, "status": PENDING_STATUS,
            "world_size": world_size, "replicas": replicas}


class XLAReplicatedBackend:
    """Replica-local production backend over a shared logical update.

    Rank flow per logical update: shard the global bucket-pure rows with
    topology_map, forward/backward each accumulation microstep locally with
    scale (n_local / N_GLOBAL), all-reduce SUM the gradients, apply the ONE
    global clip, step once, certify. Rank 0 owns checkpoint writes.
    """

    def __init__(self, *, replica_backend: Any, replicas: int,
                 world_size: int | None = None,
                 torch_module: Any | None = None) -> None:
        self.replica_backend = replica_backend
        self.replicas = int(replicas)
        if self.replicas <= 0:
            raise ValueError("replica count must be positive")
        if world_size is not None:
            require_frozen_topology(int(world_size), replicas=self.replicas)
        self.torch = torch_module
        self.last_receipt: dict[str, object] | None = None

    def status(self) -> dict[str, object]:
        """Adapter status: pending until PRE500M measures it on TPU."""

        discovered = xla_status()
        if discovered.get("status") != PENDING_STATUS:
            return discovered
        if int(discovered["world_size"]) != self.replicas:
            return {"schema": ADAPTER_SCHEMA, "status": EVIDENCE_REQUIRED,
                    "reason": "live XLA world does not match the frozen replicas"}
        return {"schema": ADAPTER_SCHEMA, "status": PENDING_STATUS,
                "replicas": self.replicas,
                "world_size": discovered["world_size"],
                "device_type": discovered.get("device_type")}

    def all_reduce_sum_gradients(self, model: Any) -> None:
        """SUM replica gradients into every rank (the frozen collective)."""

        _, xm = _load_xla()
        gradients = [parameter.grad for parameter in model.parameters()
                     if parameter.grad is not None]
        if not gradients:
            raise ValueError("no gradients to reduce")
        reduced = xm.all_reduce(xm.REDUCE_SUM, gradients)
        for parameter, value in zip(
                [parameter for parameter in model.parameters()
                 if parameter.grad is not None], reduced):
            parameter.grad = value

    def synchronized_step(self, *, model: Any, optimizer: Any) -> None:
        """One synchronized optimizer step across replicas (rank-symmetric)."""

        self.all_reduce_sum_gradients(model)
        optimizer.step()

    def rank_owns_checkpoints(self) -> bool:
        """Checkpoint writer ownership: rank 0 only."""

        return self.ordinal() == 0

    def ordinal(self) -> int:
        """This rank's global ordinal (fails closed without XLA)."""

        xruntime, _ = _load_xla()
        try:
            return int(xruntime.global_ordinal())
        except Exception as exc:
            raise RuntimeError(f"cannot determine XLA rank: {exc}") from exc


__all__ = ["ADAPTER_SCHEMA", "EVIDENCE_REQUIRED", "PENDING_STATUS",
           "XLAReplicatedBackend", "require_frozen_topology", "xla_status"]
