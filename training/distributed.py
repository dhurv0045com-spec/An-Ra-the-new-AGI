"""Measured FSDP configuration and distributed campaign estimates."""

from __future__ import annotations

from dataclasses import dataclass


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
