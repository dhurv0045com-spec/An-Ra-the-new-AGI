"""Explicit PyTorch/XLA runtime seam for Kaggle TPU data parallelism.

Importing this module is safe on CPU-only hosts: torch_xla is imported only
inside TPU entry points. Reports distinguish a visible TPU runtime from a
validated model fit or a completed training run.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any, Mapping

from bramastra_lab.research.experience.supervision import OBJECTIVE_TERMS


TPU_RUNTIME_NOT_SELECTED = "TPU_RUNTIME_NOT_SELECTED"
TPU_RUNTIME_UNAVAILABLE = "TPU_RUNTIME_UNAVAILABLE"
TPU_TOPOLOGY_MISMATCH = "TPU_TOPOLOGY_MISMATCH"
TPU_RUNTIME_READY_FOR_PREFLIGHT = "TPU_RUNTIME_READY_FOR_PREFLIGHT"
_XLA_LAUNCH_MARKER = "BRAMASTRA_XLA_REPLICA_LAUNCH"


def inspect_tpu_runtime(*, expected_replicas: int = 8) -> dict[str, Any]:
    """Read-only check of PJRT selection, PyTorch/XLA, and replica topology.

    This is only a runtime gate. It does not allocate a model, run a forward
    pass, certify memory fit, or start training.
    """
    if (not isinstance(expected_replicas, int)
            or isinstance(expected_replicas, bool) or expected_replicas < 1):
        raise ValueError("expected_replicas must be a positive integer")
    selected = os.environ.get("PJRT_DEVICE", "").strip().upper()
    report: dict[str, Any] = {
        "status": TPU_RUNTIME_NOT_SELECTED,
        "pjrt_device": selected or None,
        "expected_replicas": int(expected_replicas),
        "training_started": False,
    }
    if selected != "TPU":
        report["detail"] = "select a Kaggle TPU runtime that exposes PJRT_DEVICE=TPU"
        return report
    try:
        import torch
        import torch_xla
        import torch_xla.runtime as xr
    except Exception as exc:
        report["status"] = TPU_RUNTIME_UNAVAILABLE
        report["detail"] = f"PyTorch/XLA import failed: {type(exc).__name__}: {exc}"
        return report
    try:
        device_type = str(xr.device_type()).upper()
        world_size = int(xr.world_size())
        device_count = int(xr.global_runtime_device_count())
    except Exception as exc:
        report["status"] = TPU_RUNTIME_UNAVAILABLE
        report["detail"] = f"PJRT topology query failed: {type(exc).__name__}: {exc}"
        return report
    report.update({
        "torch_version": str(torch.__version__),
        "torch_xla_version": str(getattr(torch_xla, "__version__", "unknown")),
        "runtime_device_type": device_type,
        "world_size": world_size,
        "global_device_count": device_count,
    })
    if device_type != "TPU" or world_size != expected_replicas \
            or device_count < expected_replicas:
        report["status"] = TPU_TOPOLOGY_MISMATCH
        report["detail"] = (
            "runtime must expose the requested TPU replica count before model "
            "preflight; no training was started")
        return report
    report["status"] = TPU_RUNTIME_READY_FOR_PREFLIGHT
    report["detail"] = (
        "PyTorch/XLA and expected TPU replicas are visible; run the no-update "
        "model preflight before any optimizer update")
    return report


@dataclass(frozen=True)
class XLAReplicaBackend:
    """All-reduce objective counts and gradients over an XLA replica group.

    Construct inside the function passed to ``torch_xla.launch``. Training
    data must be supplied through ``MpDeviceLoader`` so every replica receives
    its assigned shard and XLA step barriers are driven consistently.
    """

    device: Any
    world_size: int

    @classmethod
    def current(cls, device: Any, *, expected_replicas: int = 8) -> "XLAReplicaBackend":
        if os.environ.get("PJRT_DEVICE", "").strip().upper() != "TPU":
            raise RuntimeError("TPU backend requires PJRT_DEVICE=TPU")
        if os.environ.get(_XLA_LAUNCH_MARKER) != "1":
            raise RuntimeError(
                "replicated TPU trainer construction is only allowed inside "
                "launch_tpu_workers; a visible TPU alone is not a worker mesh")
        try:
            import torch_xla.core.xla_model as xm
            import torch_xla.runtime as xr
        except Exception as exc:
            raise RuntimeError(f"PyTorch/XLA is unavailable: {exc}") from exc
        device_type = str(xr.device_type()).upper()
        hardware = str(xm.xla_device_hw(device)).upper()
        world_size = int(xr.world_size())
        device_count = int(xr.global_runtime_device_count())
        if device_type != "TPU" or "TPU" not in hardware:
            raise RuntimeError(
                f"expected TPU hardware, got runtime={device_type!r}, hardware={hardware!r}")
        if world_size != expected_replicas or device_count < expected_replicas:
            raise RuntimeError(
                f"expected {expected_replicas} TPU replicas; "
                f"runtime reports world_size={world_size}, devices={device_count}")
        return cls(device=device, world_size=world_size)

    def reduce_counts(self, counts: Mapping[str, int]) -> dict[str, Any]:
        """Return per-objective counts summed across every training replica."""
        import torch
        import torch_xla.core.xla_model as xm

        unknown = set(counts) - set(OBJECTIVE_TERMS)
        if unknown:
            raise ValueError(f"unknown objective counts: {sorted(unknown)}")
        terms = tuple(sorted(counts))
        if not terms:
            return {}
        values = []
        for term in terms:
            count = counts[term]
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f"count for {term!r} must be a nonnegative integer")
            values.append(count)
        local = torch.tensor(values, dtype=torch.int32, device=self.device)
        global_counts = xm.all_reduce(xm.REDUCE_SUM, local, groups=None)
        return {term: global_counts[index] for index, term in enumerate(terms)}

    def materialize_counts(self, counts: Mapping[str, Any]) -> dict[str, int]:
        """Synchronize one small count vector for fail-closed eligibility checks."""
        import torch

        terms = tuple(sorted(counts))
        if not terms:
            return {}
        values = torch.stack([counts[term].reshape(()) for term in terms])
        host_values = values.detach().to(device="cpu", dtype=torch.int64).tolist()
        return {term: int(host_values[index]) for index, term in enumerate(terms)}

    def reduce_gradients(self, optimizer: Any) -> None:
        """Average gradients across replicas before global clipping."""
        import torch_xla.core.xla_model as xm

        xm.reduce_gradients(optimizer, groups=None)

    def reduce_metric_sum(self, local_sum: Any) -> Any:
        """Sum one detached scalar metric across replicas for honest reports."""
        import torch_xla.core.xla_model as xm

        return xm.all_reduce(xm.REDUCE_SUM, local_sum.detach(), groups=None)

    def mark_step(self) -> None:
        """Flush the current XLA graph before a memory-intensive second pass."""
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    def optimizer_step(self, optimizer: Any) -> None:
        """Update already-reduced/clipped gradients and enqueue the XLA step."""
        import torch_xla.core.xla_model as xm

        # Gradient reduction must precede clipping so the norm is the norm of
        # the true global mean gradient, rather than a mean of local clips.
        optimizer.step()
        xm.mark_step()


def as_xla_device_loader(loader: Any, device: Any) -> Any:
    """Wrap a host DataLoader for XLA prefetch and host-to-device transfer.

    This does *not* partition examples between replicated TPU workers. Use
    :func:`make_xla_data_loader` for training; it binds a rank-specific
    ``DistributedSampler`` before wrapping the loader.
    """
    try:
        import torch_xla.distributed.parallel_loader as pl
    except Exception as exc:
        raise RuntimeError(f"PyTorch/XLA parallel loader is unavailable: {exc}") from exc
    return pl.MpDeviceLoader(loader, device)


def replica_training_sampler(dataset: Any, *, rank: int, world_size: int,
                             seed: int = 0) -> Any:
    """Build a deterministic, non-padding training shard for one replica.

    ``DistributedSampler(drop_last=True)`` keeps replicas at equal length
    without silently duplicating training examples to pad a short final shard.
    The omitted tail rotates when callers invoke ``sampler.set_epoch(epoch)``.
    """
    import torch.utils.data

    if (not isinstance(world_size, int) or isinstance(world_size, bool)
            or world_size < 1):
        raise ValueError("world_size must be a positive integer")
    if (not isinstance(rank, int) or isinstance(rank, bool)
            or not 0 <= rank < world_size):
        raise ValueError("rank must be an integer in [0, world_size)")
    if (not isinstance(seed, int) or isinstance(seed, bool)
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    try:
        dataset_size = len(dataset)
    except Exception as exc:
        raise ValueError("replicated TPU training requires a sized dataset") from exc
    if dataset_size < world_size:
        raise ValueError(
            f"dataset has {dataset_size} rows for {world_size} replicas; "
            "every replica needs at least one unique training row")
    return torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank,
        shuffle=True, seed=seed, drop_last=True)


def make_xla_data_loader(dataset: Any, device: Any, *, batch_size: int,
                         seed: int = 0, num_workers: int = 0,
                         drop_last: bool = False, collate_fn: Any | None = None
                         ) -> tuple[Any, Any]:
    """Create a rank-sharded host loader and its XLA device-prefetch wrapper.

    Return ``(sampler, device_loader)``. Call ``sampler.set_epoch(epoch)``
    before each epoch so deterministic shuffling advances without overlap.
    The helper must be called inside a worker launched by
    :func:`launch_tpu_workers`.
    """
    if (not isinstance(batch_size, int) or isinstance(batch_size, bool)
            or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    if (not isinstance(num_workers, int) or isinstance(num_workers, bool)
            or num_workers < 0):
        raise ValueError("num_workers must be a nonnegative integer")
    if not isinstance(drop_last, bool):
        raise ValueError("drop_last must be boolean")
    if os.environ.get("PJRT_DEVICE", "").strip().upper() != "TPU":
        raise RuntimeError("XLA data loading requires PJRT_DEVICE=TPU")
    try:
        import torch
        import torch_xla.runtime as xr
    except Exception as exc:
        raise RuntimeError(f"PyTorch/XLA runtime is unavailable: {exc}") from exc
    world_size = int(xr.world_size())
    rank = int(xr.global_ordinal())
    sampler = replica_training_sampler(
        dataset, rank=rank, world_size=world_size, seed=seed)
    host_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, drop_last=drop_last, collate_fn=collate_fn)
    return sampler, as_xla_device_loader(host_loader, device)


def launch_tpu_workers(worker: Any, *, args: tuple[Any, ...] = ()) -> None:
    """Launch the worker on Kaggle's TPU replica set using PyTorch/XLA PJRT."""
    if os.environ.get("PJRT_DEVICE", "").strip().upper() != "TPU":
        raise RuntimeError("select Kaggle TPU and PJRT_DEVICE=TPU before launch")
    try:
        import torch_xla
    except Exception as exc:
        raise RuntimeError(f"PyTorch/XLA is unavailable: {exc}") from exc
    launch = getattr(torch_xla, "launch", None)
    if not callable(launch):
        raise RuntimeError("installed torch_xla does not expose torch_xla.launch")
    previous = os.environ.get(_XLA_LAUNCH_MARKER)
    os.environ[_XLA_LAUNCH_MARKER] = "1"
    try:
        launch(worker, args=args)
    finally:
        if previous is None:
            os.environ.pop(_XLA_LAUNCH_MARKER, None)
        else:
            os.environ[_XLA_LAUNCH_MARKER] = previous


def broadcast_replica_parameters(model: Any) -> None:
    """Make the master replica's initialized weights authoritative.

    TPU v2/v3 PJRT can run multiple threads inside one process; repeating
    ``torch.manual_seed`` per worker is not sufficient to guarantee identical
    initialization. Call this after moving the model to its worker-local XLA
    device and before constructing or stepping the optimizer.
    """
    if os.environ.get("PJRT_DEVICE", "").strip().upper() != "TPU":
        raise RuntimeError("parameter broadcast requires PJRT_DEVICE=TPU")
    if os.environ.get(_XLA_LAUNCH_MARKER) != "1":
        raise RuntimeError(
            "parameter broadcast is only allowed inside launch_tpu_workers")
    try:
        import torch_xla.core.xla_model as xm
    except Exception as exc:
        raise RuntimeError(f"PyTorch/XLA is unavailable: {exc}") from exc
    broadcast = getattr(xm, "broadcast_master_param", None)
    if not callable(broadcast):
        try:
            from torch_xla.experimental.pjrt import broadcast_master_param as broadcast
        except Exception as exc:
            raise RuntimeError(
                "installed PyTorch/XLA has no supported master-parameter "
                f"broadcast API: {exc}") from exc
    broadcast(model)
    xm.mark_step()


def state_dict_sha256(state: Mapping[str, Any]) -> str:
    """Hash exact tensor names, shapes, dtypes and bytes without a byte copy."""
    import torch

    if not isinstance(state, Mapping) or not state:
        raise ValueError("model state must be a nonempty mapping")
    if not all(isinstance(name, str) for name in state):
        raise ValueError("model state keys must be strings")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("model state must map string names to tensors")
        if tensor.layout != torch.strided:
            raise ValueError(f"unsupported non-strided state tensor {name!r}")
        host = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(host.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(host.dtype).encode("ascii"))
        digest.update(b"\0")
        # flatten() handles both scalar buffers and empty tensors before the
        # dtype reinterpretation; view(uint8) on a zero-dimensional float
        # raises even though its exact bytes are hashable.
        raw = host.flatten().view(torch.uint8).numpy()
        digest.update(memoryview(raw).cast("B"))
        digest.update(b"\0")
    return digest.hexdigest()


def write_replica_initialization_receipt(
    report_dir: str, *, rank: int, world_size: int,
    config_identity: str, model: Any,
) -> dict[str, Any]:
    """Persist one worker's exact initial model fingerprint atomically."""
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    if (not isinstance(world_size, int) or isinstance(world_size, bool)
            or world_size < 1 or rank >= world_size):
        raise ValueError("rank must be below a positive world_size")
    if not isinstance(config_identity, str) or not config_identity:
        raise ValueError("config_identity must be nonempty")
    state = model.state_dict()
    receipt = {
        "schema": "bramastra-tpu-replica-init/v1",
        "rank": rank,
        "world_size": world_size,
        "config_identity": config_identity,
        "parameter_count": sum(int(param.numel())
                                for param in model.parameters()),
        "state_sha256": state_dict_sha256(state),
    }
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, f"replica-{rank:03d}.json")
    temp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(temp, "x", encoding="utf-8") as handle:
            json.dump(receipt, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        # A hard-link publishes the fully written file atomically and fails
        # if another writer already published this rank. os.replace would
        # silently let a duplicate writer overwrite the original receipt.
        os.link(temp, path)
    finally:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
    return receipt


def verify_replica_initialization(report_dir: str, *,
                                  expected_replicas: int = 8
                                  ) -> dict[str, Any]:
    """Require one identical config/state fingerprint from every TPU rank."""
    if (not isinstance(expected_replicas, int)
            or isinstance(expected_replicas, bool) or expected_replicas < 1):
        raise ValueError("expected_replicas must be a positive integer")
    expected_files = {f"replica-{rank:03d}.json"
                      for rank in range(expected_replicas)}
    try:
        names = os.listdir(report_dir)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"TPU initialization report directory does not exist: {report_dir}") from exc
    observed_files = {name for name in names
                      if name.startswith("replica-") and name.endswith(".json")}
    if observed_files != expected_files:
        raise RuntimeError(
            "TPU initialization report directory must contain exactly one "
            f"receipt per rank; expected={sorted(expected_files)}, "
            f"observed={sorted(observed_files)}")
    receipts = []
    for rank in range(expected_replicas):
        path = os.path.join(report_dir, f"replica-{rank:03d}.json")
        try:
            with open(path, encoding="utf-8") as handle:
                receipt = json.load(handle)
        except Exception as exc:
            raise RuntimeError(
                f"missing or unreadable TPU initialization receipt for rank "
                f"{rank}: {exc}") from exc
        if (not isinstance(receipt, dict)
                or not isinstance(receipt.get("rank"), int)
                or isinstance(receipt.get("rank"), bool)
                or receipt.get("rank") != rank
                or not isinstance(receipt.get("world_size"), int)
                or isinstance(receipt.get("world_size"), bool)
                or receipt.get("world_size") != expected_replicas
                or receipt.get("schema") != "bramastra-tpu-replica-init/v1"):
            raise RuntimeError(f"TPU initialization receipt mismatch at rank {rank}")
        if (not isinstance(receipt.get("config_identity"), str)
                or not receipt["config_identity"]
                or not isinstance(receipt.get("state_sha256"), str)
                or len(receipt["state_sha256"]) != 64
                or any(char not in "0123456789abcdef"
                       for char in receipt["state_sha256"])
                or not isinstance(receipt.get("parameter_count"), int)
                or isinstance(receipt.get("parameter_count"), bool)
                or receipt["parameter_count"] <= 0):
            raise RuntimeError(f"TPU initialization receipt is incomplete at rank {rank}")
        receipts.append(receipt)
    identities = {(r.get("config_identity"), r.get("state_sha256"),
                   r.get("parameter_count")) for r in receipts}
    if len(identities) != 1:
        raise RuntimeError(
            "TPU replicas do not share exact initialized config and weights")
    config_identity, state_sha256, parameter_count = next(iter(identities))
    return {"status": "REPLICAS_IDENTICAL", "replicas": expected_replicas,
            "config_identity": config_identity,
            "state_sha256": state_sha256,
            "parameter_count": parameter_count,
            "training_started": False}
