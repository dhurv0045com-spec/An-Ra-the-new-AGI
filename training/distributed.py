"""Explicit same-host distributed runtime and campaign estimates.

Separate Colab/Kaggle machines are checkpoint-baton workers.  This module is
only for GPUs joined by one low-latency NCCL host/network and launched through
``torchrun``.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import TypeVar

import torch
import torch.distributed as dist

DDP_CONTRACT_SCHEMA = "anra-ddp-contract/v1"
DDP_SAMPLER_PARTITION = "rank_strided_global_position_v1"
DDP_GRADIENT_REDUCTION = "ddp_mean_v1"
CANONICAL_DDP_TRAINER = "anra-v4-canonical-raw-causal/v1"
T = TypeVar("T")


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    backend: str
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def contract(
        self,
        *,
        micro_batch_size_per_rank: int,
        gradient_accumulation: int,
    ) -> dict[str, object]:
        micro = int(micro_batch_size_per_rank)
        accumulation = int(gradient_accumulation)
        if micro < 1 or accumulation < 1:
            raise ValueError("distributed batch dimensions must be positive")
        return {
            "schema": DDP_CONTRACT_SCHEMA,
            "backend": self.backend,
            "world_size": self.world_size,
            "micro_batch_size_per_rank": micro,
            "gradient_accumulation": accumulation,
            "global_sequences_per_step": micro * accumulation * self.world_size,
            "sampler_partition": DDP_SAMPLER_PARTITION,
            "gradient_reduction": DDP_GRADIENT_REDUCTION,
            "same_host": True,
            "rank_to_local_rank": {str(rank): rank for rank in range(self.world_size)},
            "visible_device_order": os.environ.get(
                "CUDA_VISIBLE_DEVICES", ",".join(str(rank) for rank in range(self.world_size))
            ),
        }


def canonical_training_ddp_contract(
    *,
    backend: str,
    world_size: int,
    micro_batch_size_per_rank: int,
    gradient_accumulation: int,
    visible_device_order: str,
) -> dict[str, object]:
    """Build the signed logical topology used by owner and trainer contracts."""

    world = int(world_size)
    micro = int(micro_batch_size_per_rank)
    accumulation = int(gradient_accumulation)
    visible = [item.strip() for item in str(visible_device_order).split(",") if item.strip()]
    if world < 2 or micro < 1 or accumulation < 1:
        raise ValueError("canonical DDP topology and batch dimensions must be positive")
    if len(visible) != world or len(set(visible)) != world:
        raise ValueError("visible device order must name every DDP rank exactly once")
    return {
        "schema": DDP_CONTRACT_SCHEMA,
        "backend": str(backend),
        "world_size": world,
        "micro_batch_size_per_rank": micro,
        "gradient_accumulation": accumulation,
        "global_sequences_per_step": micro * accumulation * world,
        "sampler_partition": DDP_SAMPLER_PARTITION,
        "gradient_reduction": DDP_GRADIENT_REDUCTION,
        "same_host": True,
        "rank_to_local_rank": {str(rank): rank for rank in range(world)},
        "visible_device_order": ",".join(visible),
        "trainer": CANONICAL_DDP_TRAINER,
        "checkpoint_owner": "rank_zero_only",
        "rng_ownership": "every_rank",
        "find_unused_parameters": True,
    }


def distributed_context_from_environment(mode: str = "off") -> DistributedContext:
    requested = str(mode).strip().lower()
    if requested not in {"off", "ddp"}:
        raise ValueError("distributed mode must be off or ddp")
    if requested == "off":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistributedContext(False, "none", 0, 0, 1, device)
    required = {name: os.environ.get(name) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")}
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise RuntimeError(f"DDP requires torchrun environment variables: {missing}")
    rank = int(required["RANK"] or -1)
    local_rank = int(required["LOCAL_RANK"] or -1)
    world_size = int(required["WORLD_SIZE"] or -1)
    if world_size < 2 or rank < 0 or rank >= world_size or local_rank < 0:
        raise RuntimeError("invalid torchrun rank topology")
    if not torch.cuda.is_available():
        raise RuntimeError("An-Ra DDP requires CUDA/NCCL")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError("LOCAL_RANK does not address a visible CUDA device")
    return DistributedContext(
        True,
        "nccl",
        rank,
        local_rank,
        world_size,
        torch.device("cuda", local_rank),
    )


def initialize_distributed(mode: str = "off") -> DistributedContext:
    context = distributed_context_from_environment(mode)
    if not context.enabled:
        return context
    if dist.is_initialized():
        raise RuntimeError(
            "torch.distributed was initialized before An-Ra established its contract"
        )
    torch.cuda.set_device(context.local_rank)
    dist.init_process_group(backend=context.backend, init_method="env://")
    dist.barrier()
    return context


def validate_same_host_topology(context: DistributedContext) -> None:
    """Reject multi-node or duplicate-device launches in the initial DDP mode."""

    if not context.enabled:
        return
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "hostname": socket.gethostname(),
        "visible_cuda_devices": torch.cuda.device_count(),
    }
    gathered: list[dict[str, object] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local)
    if any(row is None for row in gathered):
        raise RuntimeError("DDP topology gather returned an incomplete rank set")
    rows = [row for row in gathered if row is not None]
    if len({str(row["hostname"]) for row in rows}) != 1:
        raise RuntimeError("Canonical An-Ra DDP currently supports one physical host only")
    if {int(row["rank"]) for row in rows} != set(range(context.world_size)):
        raise RuntimeError("DDP topology does not contain every global rank exactly once")
    if len({int(row["local_rank"]) for row in rows}) != context.world_size:
        raise RuntimeError("same-host DDP ranks must own unique CUDA devices")
    if any(int(row["rank"]) != int(row["local_rank"]) for row in rows):
        raise RuntimeError("canonical same-host DDP requires the stable rank == local-rank mapping")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def all_reduce_sum(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    result = value.clone()
    if context.enabled:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def all_reduce_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    result = all_reduce_sum(value, context)
    if context.enabled:
        result.div_(context.world_size)
    return result


def all_reduce_bool_or(value: bool, context: DistributedContext) -> bool:
    flag = torch.tensor(int(bool(value)), dtype=torch.int32, device=context.device)
    if context.enabled:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def all_reduce_bool_and(value: bool, context: DistributedContext) -> bool:
    flag = torch.tensor(int(bool(value)), dtype=torch.int32, device=context.device)
    if context.enabled:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def all_gather_objects(value: T, context: DistributedContext) -> list[T]:
    if not context.enabled:
        return [value]
    gathered: list[T | None] = [None for _ in range(context.world_size)]
    dist.all_gather_object(gathered, value)
    if any(item is None for item in gathered):
        raise RuntimeError("distributed object gather returned an incomplete rank set")
    return [item for item in gathered if item is not None]


def broadcast_primary_result(value: T, context: DistributedContext) -> T:
    if not context.enabled:
        return value
    payload: list[T | None] = [value if context.is_primary else None]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is None:
        raise RuntimeError("rank zero broadcast no result")
    return payload[0]


def barrier_or_raise(
    context: DistributedContext,
    *,
    primary_error: str | None = None,
) -> None:
    """Broadcast a rank-zero filesystem result before the next collective."""

    envelope = broadcast_primary_result(
        {
            "ok": primary_error is None,
            "error": primary_error,
        },
        context,
    )
    if not bool(envelope.get("ok")):
        raise RuntimeError(f"distributed rank-zero operation failed: {envelope.get('error')}")
    if context.enabled:
        dist.barrier()


def destroy_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


@dataclass(frozen=True)
class DistributedProfile:
    world_size: int
    precision: str
    sharding: str
    activation_checkpointing: bool
    tokens_per_second: float = 0.0
    peak_bytes_per_rank: int = 0


def recommended_profile(
    *,
    world_size: int,
    bf16_supported: bool,
    full_shard: bool = False,
) -> DistributedProfile:
    if world_size < 2:
        raise ValueError("Distributed profile requires at least two ranks.")
    return DistributedProfile(
        world_size=int(world_size),
        precision="bf16" if bf16_supported else "fp16",
        sharding="FULL_SHARD" if full_shard else "SHARD_GRAD_OP",
        activation_checkpointing=True,
    )


def estimate_campaign(
    *,
    token_target: int,
    measured_tokens_per_second: float,
    hourly_cost: float,
) -> dict[str, float]:
    if measured_tokens_per_second <= 0:
        raise ValueError("Use measured positive throughput.")
    seconds = float(token_target) / measured_tokens_per_second
    hours = seconds / 3600.0
    return {"hours": hours, "estimated_cost": hours * float(hourly_cost)}
