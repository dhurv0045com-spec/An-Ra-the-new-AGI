"""Live parameter/optimizer mutation evidence for real updates.

Every training receipt must prove the optimizer actually changed the model.
These helpers capture content hashes of parameters and Adam moments plus
optimizer step counters before and after an update, then verify change. This
makes the historical core-vnext failure mode (metadata claims training while
weights never move) mechanically impossible to miss: an unchanged model
raises instead of certifying.
"""

from __future__ import annotations

import hashlib
from typing import Any


def parameter_sha(model: Any, *, torch_module: Any) -> str:
    """Hash every trainable parameter's bytes in name order (chunked)."""

    torch = torch_module
    digest = hashlib.sha256()
    for name, parameter in sorted(
        model.named_parameters(), key=lambda item: item[0]
    ):
        if not parameter.requires_grad:
            raise ValueError(f"parameter {name} is frozen; V5 trains every parameter")
        data = parameter.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(data.tobytes())
    return digest.hexdigest()


def moment_fingerprint(optimizer: Any, *, torch_module: Any) -> str:
    """Hash Adam first/second moments and step counters in stable order."""

    torch = torch_module
    digest = hashlib.sha256()
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            step = int(state.get("step", 0))
            digest.update(f"{group_index}:{step}:".encode("utf-8"))
            for key in ("exp_avg", "exp_avg_sq"):
                moment = state.get(key)
                if moment is None:
                    digest.update(b"absent\0")
                    continue
                data = moment.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
                digest.update(data.tobytes())
    return digest.hexdigest()


def optimizer_step(optimizer: Any) -> int:
    """Return the shared Adam step counter, requiring unanimity."""

    steps = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            steps.add(int(optimizer.state[parameter].get("step", 0)))
    if len(steps) != 1:
        raise ValueError("optimizer states disagree on the step counter")
    return steps.pop()


def global_grad_norm(model: Any, *, torch_module: Any) -> float:
    """Replica-global L2 gradient norm in FP32; refuses all-None gradients."""

    torch = torch_module
    total = 0.0
    seen = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        seen += 1
        total += float(gradient.detach().float().pow(2).sum().item())
    if seen == 0:
        raise ValueError("no gradients materialized; refusing a silent no-op update")
    return total**0.5


def assert_mutation(
    *,
    before_sha: str,
    after_sha: str,
    before_moments: str,
    after_moments: str,
    before_step: int,
    after_step: int,
    expected_steps: int = 1,
    learning_rate: float = -1.0,
) -> None:
    """Require observable parameter, moment, and step advancement.

    At learning rate exactly zero the parameters must NOT move (the schedule
    opens at 0.0); moments and step counters still must advance, proving the
    optimizer executed rather than no-op'd.
    """

    if learning_rate < 0:
        raise ValueError("mutation evidence requires the effective learning rate")
    if learning_rate > 0 and before_sha == after_sha:
        raise ValueError("parameters did not change; refusing silent no-op certification")
    if learning_rate == 0 and before_sha != after_sha:
        raise ValueError("parameters moved at zero learning rate; refusing certification")
    if before_moments == after_moments:
        raise ValueError("optimizer moments did not change; refusing certification")
    if after_step - before_step != expected_steps:
        raise ValueError("optimizer step did not advance exactly once")


__all__ = [
    "assert_mutation",
    "global_grad_norm",
    "moment_fingerprint",
    "optimizer_step",
    "parameter_sha",
]
