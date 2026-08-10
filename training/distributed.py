"""Explicit same-host distributed runtime and campaign estimates.

Separate Colab/Kaggle machines are checkpoint-baton workers.  This module is
only for GPUs joined by one low-latency NCCL host/network and launched through
``torchrun``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypeVar

import torch
import torch.distributed as dist

DDP_CONTRACT_SCHEMA = "anra-ddp-contract/v1"
DDP_SAMPLER_PARTITION = "rank_strided_global_position_v1"
DDP_GRADIENT_REDUCTION = "ddp_mean_v1"
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
