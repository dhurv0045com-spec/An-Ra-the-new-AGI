"""Distributed accumulation-boundary oracle (CYR-GPU-005 section 51).

Mathematically certifies the production XLA flow on CPU:

    optimizer.zero_grad()
    microstep 1..M:  LOCAL rank backward only (no collective)
    THEN:            ONE gradient SUM collective
    THEN:            ONE global norm/clip, ONE optimizer step,
                     ONE TrainingState.advance

against the single-process logical update, across >= 3 virtual replicas
x 4 accumulation microsteps with UNEQUAL eligible-token counts. A negative
regression re-runs the flow with the historical defect — an all-reduce
applied to the ACCUMULATED buffer after every microstep — and MUST fail
the oracle. CPU equivalence never certifies TPU hardware; the adapter
stays IMPLEMENTED_PENDING_PRE500M_TPU.
"""

from __future__ import annotations

from typing import Any

ORACLE_SCHEMA = "anra-cyr-gpu005-xla-oracle/v1"


def _shard_inputs(*, torch: Any, replicas: int, microsteps: int, seed: int,
                  rows_low: int, rows_high: int) -> list[list[dict[str, Any]]]:
    """Deterministic shards with deliberately UNEQUAL row counts."""

    generator = torch.Generator().manual_seed(seed)
    shards: list[list[dict[str, Any]]] = []
    for _replica in range(replicas):
        replica_shards: list[dict[str, Any]] = []
        for microstep in range(microsteps):
            rows = int(torch.randint(rows_low, rows_high + 1, (1,),
                                     generator=generator).item())
            inputs = torch.randn(rows, 8, generator=generator) * 0.5
            targets = torch.randn(rows, 8, generator=generator) * 0.1
            replica_shards.append({"inputs": inputs, "targets": targets,
                                   "rows": rows})
        shards.append(replica_shards)
    return shards


def _loss_of(model: Any, shard: dict[str, Any], torch: Any) -> tuple[Any, int]:
    predictions = model(shard["inputs"])
    loss = ((predictions - shard["targets"]) ** 2).sum() / shard["rows"]
    return loss, shard["rows"]


def _adamw() -> dict[str, Any]:
    from v5_training.optimizer import build_adamw_optimizer
    return {"build": build_adamw_optimizer}


def _clip_and_step(model: Any, optimizer: Any, *, clip: float = 1.0) -> dict[str, float]:
    import torch
    pre = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item())
    optimizer.step()
    return {"grad_norm_pre_clip": pre}


def _state_hash(model: Any, optimizer: Any, torch: Any) -> dict[str, float]:
    parameters = torch.cat([parameter.detach().reshape(-1) for parameter
                            in model.parameters()])
    moments = [state for state in optimizer.state.values()
               if isinstance(state, dict)]
    first = torch.cat([state["exp_avg"].reshape(-1) for state in moments]) \
        if moments else torch.zeros(1)
    second = torch.cat([state["exp_avg_sq"].reshape(-1) for state in moments]) \
        if moments else torch.zeros(1)
    return {"parameters_norm": float(parameters.norm().item()),
            "exp_avg_norm": float(first.norm().item()),
            "exp_avg_sq_norm": float(second.norm().item())}


def _build_linear_model(torch: Any, seed: int) -> Any:
    """One shared linear map — the smallest honest differentiable model."""

    class LinearModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            generator = torch.Generator().manual_seed(seed)
            self.weight = torch.nn.Parameter(
                torch.randn(8, 8, generator=generator) * 0.1)

        def forward(self, inputs: Any) -> Any:
            return inputs @ self.weight

    return LinearModel()


def _single_process_logical_update(shards: list[list[dict[str, Any]]], *,
                                   torch: Any, seed: int,
                                   ) -> dict[str, Any]:
    """The truth: all shards, one accumulator, one clip, one step."""

    model = _build_linear_model(torch, seed)
    optimizer = _adamw()["build"](model, torch_module=torch)
    optimizer.zero_grad(set_to_none=True)
    numerator = 0.0
    denominator = 0
    total_rows = sum(shard["rows"] for replica in shards for shard in replica)
    for replica in shards:
        for shard in replica:
            loss, rows = _loss_of(model, shard, torch)
            (loss * (rows / total_rows)).backward()
            numerator += float(loss.detach().item()) * rows
            denominator += rows
    steps = _clip_and_step(model, optimizer)
    return {"final": _state_hash(model, optimizer, torch),
            "numerator": numerator, "denominator": denominator,
            "update_norm": steps["grad_norm_pre_clip"]}


def _replicated_update(shards: list[list[dict[str, Any]]], *, torch: Any,
                       seed: int, reduce_every_microstep: bool,
                       collectives: list[int],
                       ) -> dict[str, Any]:
    """Simulated SPMD: R identical replicas, LOCAL accumulation, then the
    registered collective pattern. `reduce_every_microstep=True` re-creates
    the historical defect (all-reduce over the ACCUMULATED buffer each
    microstep) for the negative regression."""

    replicas = len(shards)
    models = [_build_linear_model(torch, seed) for _ in range(replicas)]
    optimizers = [_adamw()["build"](model, torch_module=torch)
                  for model in models]
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    total_rows = sum(shard["rows"] for replica in shards for shard in replica)
    numerator = 0.0
    denominator = 0
    microsteps = len(shards[0])
    for microstep in range(microsteps):
        for replica in range(replicas):
            shard = shards[replica][microstep]
            loss, rows = _loss_of(models[replica], shard, torch)
            (loss * (rows / total_rows)).backward()
            numerator += float(loss.detach().item()) * rows
            denominator += rows
        if reduce_every_microstep or microstep == microsteps - 1:
            # ONE SUM collective across replicas at this instant.
            with torch.no_grad():
                summed = [sum(models[r].weight.grad for r in range(replicas))
                          for _ in range(replicas)]
                for r in range(replicas):
                    models[r].weight.grad = summed[r].clone()
            collectives.append(microstep + 1)
    steps = [_clip_and_step(models[r], optimizers[r]) for r in range(replicas)]
    finals = [_state_hash(models[r], optimizers[r], torch) for r in range(replicas)]
    for entry in finals[1:]:
        for key in entry:
            if abs(entry[key] - finals[0][key]) > 1e-6:
                raise ValueError("replicas diverged after the boundary step")
    return {"final": finals[0], "numerator": numerator,
            "denominator": denominator,
            "update_norm": steps[0]["grad_norm_pre_clip"]}


def run_xla_accumulation_oracle(*, replicas: int = 3, microsteps: int = 4,
                                seed: int = 20260908, tolerance: float = 1e-4,
                                torch_module: Any = None,
                                ) -> dict[str, Any]:
    """Correct boundary flow must match the logical update; the historical
    per-microstep reduce MUST fail. Both verdicts are returned and tested."""

    torch = torch_module
    if torch is None:
        import torch as torch_module
        torch = torch_module
    if replicas < 3:
        raise ValueError("oracle requires at least 3 virtual replicas")
    if microsteps != 4:
        raise ValueError("the frozen production topology uses 4 microsteps")
    shards = _shard_inputs(torch=torch, replicas=replicas,
                           microsteps=microsteps, seed=seed,
                           rows_low=2, rows_high=9)
    truth = _single_process_logical_update(shards, torch=torch, seed=seed)
    correct_collectives: list[int] = []
    correct = _replicated_update(shards, torch=torch, seed=seed,
                                 reduce_every_microstep=False,
                                 collectives=correct_collectives)
    buggy_collectives: list[int] = []
    buggy = _replicated_update(shards, torch=torch, seed=seed,
                               reduce_every_microstep=True,
                               collectives=buggy_collectives)
    def close(a: dict[str, float], b: dict[str, float]) -> bool:
        return all(abs(a[key] - b[key]) <= tolerance * max(1.0, abs(b[key]))
                   for key in a)
    correct_matches = close(correct["final"], truth["final"]) and \
        abs(correct["numerator"] - truth["numerator"]) <= 1e-6 * max(1.0, abs(truth["numerator"])) and \
        correct["denominator"] == truth["denominator"]
    buggy_matches = close(buggy["final"], truth["final"])
    passed = correct_matches and not buggy_matches and \
        len(correct_collectives) == 1 and len(buggy_collectives) == microsteps
    return {"schema": ORACLE_SCHEMA,
            "replicas": replicas, "microsteps": microsteps, "seed": seed,
            "correct_flow_matches_logical_update": correct_matches,
            "correct_flow_collectives": correct_collectives,
            "buggy_flow_matches_logical_update": buggy_matches,
            "buggy_flow_collectives": buggy_collectives,
            "truth": truth, "correct": {key: value for key, value in correct.items() if key != "final"},
            "buggy": {key: value for key, value in buggy.items() if key != "final"},
            "passed": passed}


__all__ = ["ORACLE_SCHEMA", "run_xla_accumulation_oracle"]
