"""Hitchhiker deterministic training engine for Stage 1-3 controlled arms."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import time
from typing import Any

import torch

from v5_contracts.model_spec import ModelSpec
from v5_training.optimizer import validate_parameter_ownership
from .codec import digest
from .foundation_training import Recipe
from .hitchhiker_data import build_training_stream, validate_stream_contract
from .hitchhiker_diagnostics import TOKEN_Q, TOKEN_R
from .hitchhiker_protocol import (
    ALL_ARMS,
    PROTOCOL_SHA,
    resolve_scale_profile,
    validate_job,
    validate_stage_selection,
)
from .representation import UTF8
from .training import batch_tensors, configure_device
from .v6_data import V6Dataset
from .v6_training import V6Trainer
from .v7_training import ordered_indices


def _public_availability_from_tokens(tokens: tuple[int, ...]) -> tuple[bool, bool]:
    """Read Query={0|1} and Memory={0|1} directly from public prompt bytes only."""
    # Strip BOS and trailing ANSWER/answer/EOS to inspect public header text.
    content = [t for t in tokens[1:] if t >= 4]
    text = UTF8.decode(content)
    if "Query=1" in text:
        qry_on = True
    elif "Query=0" in text:
        qry_on = False
    else:
        raise ValueError("public prompt missing Query availability header")
    if "Memory=1" in text:
        mem_on = True
    elif "Memory=0" in text:
        mem_on = False
    else:
        raise ValueError("public prompt missing Memory availability header")
    return qry_on, mem_on


class HitchhikerTrainer(V6Trainer):
    """Deterministic trainer supporting static, fresh, relabeled, coverage, and aux penalty arms."""

    def __init__(
        self,
        samples: tuple[Any, ...],
        *,
        spec: ModelSpec,
        recipe: Recipe,
        source_commit: str,
        root: Path,
        lineage: str,
        data_provenance: dict[str, Any],
        tokenizer_manifest: dict[str, Any],
        order_seed: int = 7,
        method: str = "blocked",
        use_static_indexer: bool = True,
        aux_illegal_weight: float = 0.0,
        device: str = "cpu",
        resume: bool = False,
    ) -> None:
        if type(aux_illegal_weight) not in (int, float) or not math.isfinite(aux_illegal_weight) or aux_illegal_weight < 0.0:
            raise ValueError("aux_illegal_weight must be a finite non-negative number")
        self.order_seed = order_seed
        self.method = method
        self.use_static_indexer = use_static_indexer
        self.aux_illegal_weight = float(aux_illegal_weight)
        self._sample_availability = {
            s.task_id: _public_availability_from_tokens(s.tokens)
            for s in samples
        } if self.aux_illegal_weight > 0.0 else {}
        provenance = {
            **data_provenance,
            "hitchhiker_use_static_indexer": use_static_indexer,
            "hitchhiker_aux_illegal_weight": self.aux_illegal_weight,
            "hitchhiker_order_seed": order_seed,
            "hitchhiker_sampling_method": method,
        }
        super().__init__(
            samples,
            spec=spec,
            recipe=recipe,
            source_commit=source_commit,
            root=root,
            lineage=lineage,
            data_provenance=provenance,
            tokenizer_manifest=tokenizer_manifest,
            device=device,
            resume=resume,
        )

    def indices(self, update: int) -> tuple[int, ...]:
        if not self.use_static_indexer:
            start = update * self.recipe.batch_size
            end = start + self.recipe.batch_size
            if end > len(self.samples):
                raise ValueError("update exceeds materialized Hitchhiker stream length")
            return tuple(range(start, end))
        per_epoch = math.ceil(len(self.samples) / self.recipe.batch_size)
        epoch, batch = divmod(update, per_epoch)
        if epoch not in self._epoch_orders:
            self._epoch_orders[epoch] = ordered_indices(self.samples, self.order_seed, epoch, self.method)
        start = batch * self.recipe.batch_size
        chosen = self._epoch_orders[epoch][start : start + self.recipe.batch_size]
        if len(chosen) < self.recipe.batch_size:
            chosen = (*chosen, *self._epoch_orders[epoch][: self.recipe.batch_size - len(chosen)])
        return chosen

    def sample_cursor(self, completed_updates: int) -> int:
        if not self.use_static_indexer:
            return completed_updates * self.recipe.batch_size
        return super().sample_cursor(completed_updates)

    def _compute_microbatch_loss(self, chunk: list[Any], chunk_weights: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, segments, _ = batch_tensors([(s.tokens, s.eligible) for s in chunk], self.device)
        loss_weights = torch.tensor(
            [row + [0.0] * (tokens.shape[1] - len(row)) for row in chunk_weights],
            device=self.device,
        )
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.recipe.precision == "fp16"):
            logits = self.backend.logits(
                tokens, segments, activation_checkpointing=self.recipe.activation_checkpointing
            )
            losses = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                tokens[:, 1:].reshape(-1),
                reduction="none",
            ).reshape_as(tokens[:, 1:])
            value = (losses * loss_weights[:, 1:]).sum()
            if self.aux_illegal_weight > 0.0:
                aux_terms = []
                for row_idx, sample in enumerate(chunk):
                    first = sample.eligible.index(True)
                    qry_on, mem_on = self._sample_availability[sample.task_id]
                    probs = logits[row_idx, first - 1].float().softmax(-1)
                    p_illegal = probs.new_tensor(0.0)
                    if not mem_on:
                        p_illegal = p_illegal + probs[TOKEN_R]
                    if not qry_on:
                        p_illegal = p_illegal + probs[TOKEN_Q]
                    if not mem_on or not qry_on:
                        p_clamped = torch.clamp(p_illegal, max=1.0 - 1e-7)
                        aux_terms.append(-torch.log1p(-p_clamped))
                if aux_terms:
                    value = value + self.aux_illegal_weight * torch.stack(aux_terms).sum()
        return logits, value

    def step(self) -> dict[str, Any]:
        if self.faulted or self.state.complete:
            raise ValueError("complete or faulted engine requires a fresh restore")
        self.faulted = True
        u = self.state.global_update
        chosen = [self.samples[i] for i in self.plan[u]]
        weights = [[float(x) for x in sample.eligible] for sample in chosen]
        denominator = sum(map(sum, weights))
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        lr = self.learning_rate(u)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        numerator = 0.0
        for start in range(0, len(chosen), self.recipe.micro_batch_size):
            chunk = chosen[start : start + self.recipe.micro_batch_size]
            chunk_weights = weights[start : start + len(chunk)]
            logits, value = self._compute_microbatch_loss(chunk, chunk_weights)
            if not torch.isfinite(logits).all() or not torch.isfinite(value):
                raise FloatingPointError("nonfinite forward; update rejected")
            self.scaler.scale(value / denominator).backward()
            numerator += float(value.item())
        self.scaler.unscale_(self.optimizer)
        validate_parameter_ownership(self.model, self.optimizer)
        if any(p.grad is None for p in self.model.parameters()):
            raise ValueError("silent missing gradient")
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe.clip, error_if_nonfinite=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if len(self.optimizer.state) != len(list(self.model.parameters())) or any(
            slot["step"].item() != u + 1 for slot in self.optimizer.state.values()
        ):
            raise ValueError("optimizer skipped or duplicated the proposed update")
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise FloatingPointError("nonfinite parameter; restore committed parent")
        from dataclasses import replace
        from hashlib import sha256

        self.state = replace(
            self.state,
            generation=u + 1,
            global_update=u + 1,
            cumulative_tokens=self.state.cumulative_tokens + self.counts[u],
            tokens_by_source={"action": self.state.cumulative_tokens + self.counts[u]},
            optimizer_step_max=u + 1,
            schedule_tokens=self.state.cumulative_tokens + self.counts[u],
            cursor=replace(self.state.cursor, sequence_ordinal=self.sample_cursor(u + 1)),
            rng_state_sha256=sha256(self.rng_bytes()).hexdigest(),
            parent_checkpoint_sha256=self.parent,
        )
        self.state.assert_valid()
        self.faulted = False
        return {
            "update": u + 1,
            "loss": numerator / denominator,
            "weighted_denominator": denominator,
            "supervised_tokens": self.counts[u],
            "input_positions": sum(len(s.tokens) for s in chosen),
            "examples": len(chosen),
            "microbatches": math.ceil(len(chosen) / self.recipe.micro_batch_size),
            "pre_clip": float(norm),
            "lr": lr,
            "indices": self.plan[u],
        }


def aux_penalty_gradient_probe(trainer: HitchhikerTrainer) -> dict[str, Any]:
    """Compare per-example vs microbatch gradient accumulation with optional aux penalty."""
    u = trainer.state.global_update
    chosen = [trainer.samples[i] for i in trainer.plan[u]]
    weights = [[float(x) for x in sample.eligible] for sample in chosen]
    denominator = sum(map(sum, weights))
    named = list(trainer.model.named_parameters())
    parameters = [p for _, p in named]
    modes = [(m, m.training) for m in trainer.model.modules()]
    trainer.model.eval()

    def accumulated(micro: int) -> tuple[float, list[torch.Tensor]]:
        total = [torch.zeros_like(p) for p in parameters]
        loss_sum = 0.0
        for start in range(0, len(chosen), micro):
            chunk = chosen[start : start + micro]
            chunk_weights = weights[start : start + len(chunk)]
            _, value = trainer._compute_microbatch_loss(chunk, chunk_weights)
            objective = value / denominator
            grads = torch.autograd.grad(objective, parameters)
            if not torch.isfinite(objective) or any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError("nonfinite gradient probe")
            for target, g in zip(total, grads):
                target.add_(g)
            loss_sum += float(objective.detach())
        return loss_sum, total

    try:
        ref_loss, ref_grads = accumulated(1)
        act_loss, act_grads = accumulated(trainer.recipe.micro_batch_size)
    finally:
        for m, mode in modes:
            m.training = mode

    max_diff = max(float((e - o).abs().max()) for e, o in zip(ref_grads, act_grads))
    passed = math.isclose(ref_loss, act_loss, rel_tol=1e-5, abs_tol=1e-6) and all(
        torch.allclose(e, o, rtol=1e-4, atol=1e-6) for e, o in zip(ref_grads, act_grads)
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "reference_loss": ref_loss,
        "microbatch_loss": act_loss,
        "max_gradient_diff": max_diff,
        "aux_illegal_weight": trainer.aux_illegal_weight,
    }


def resolve_exposure_schedule(
    job: dict[str, Any],
    *,
    stream: dict[str, Any] | None = None,
    base_data: V6Dataset | None = None,
    batch_size: int | None = None,
    updates_override: int | None = None,
    is_cpu_canary: bool = False,
) -> int:
    """Resolve exact total optimizer updates without double-multiplying materialized streams."""
    resolved_batch = job["batch_size"] if batch_size is None else batch_size
    if type(resolved_batch) is not int or resolved_batch <= 0:
        raise ValueError("resolved batch_size must be a positive integer")

    if not is_cpu_canary:
        if updates_override is not None:
            raise ValueError("updates_override is forbidden in production training")
        if base_data is None and stream is None:
            raise ValueError("production training requires explicit base_data or stream")
        if stream is not None:
            use_static = stream.get("use_static_indexer")
            if type(use_static) is not bool:
                raise ValueError("stream must declare boolean use_static_indexer")
            samples = stream.get("samples")
            if samples is None or len(samples) <= 0:
                raise ValueError("training stream must provide nonempty samples")
            if not use_static:
                total_updates = stream.get("total_updates")
                if type(total_updates) is not int or total_updates <= 0:
                    raise ValueError("materialized stream requires positive integer total_updates")
                if len(samples) != total_updates * resolved_batch:
                    raise ValueError(
                        f"materialized stream length {len(samples)} disagrees with "
                        f"total_updates * batch_size ({total_updates * resolved_batch})"
                    )
                return total_updates
            expected = math.ceil(len(samples) / resolved_batch) * job["epochs"]
            if "total_updates" in stream and stream["total_updates"] is not None:
                if stream["total_updates"] != expected:
                    raise ValueError(
                        f"static stream total_updates {stream['total_updates']} disagrees with expected {expected}"
                    )
            return expected
        if len(base_data.samples) <= 0:
            raise ValueError("base_data must contain nonempty samples")
        return math.ceil(len(base_data.samples) / resolved_batch) * job["epochs"]

    if type(updates_override) is not int or not 1 <= updates_override <= 16:
        raise ValueError("CPU mechanical check requires updates_override in 1..16")
    if stream is not None and not stream.get("use_static_indexer", False):
        samples = stream.get("samples")
        if samples is None or len(samples) != updates_override * resolved_batch:
            actual = 0 if samples is None else len(samples)
            raise ValueError(
                f"materialized CPU canary stream length {actual} disagrees with "
                f"updates_override * batch_size ({updates_override * resolved_batch})"
            )
    return updates_override


def create_hitchhiker_trainer(
    job: dict[str, Any],
    *,
    source_commit: str,
    root: Path,
    base_data: V6Dataset | None = None,
    stream: dict[str, Any] | None = None,
    heldout_suites: tuple[Any, ...] = (),
    data_seed: int = 606,
    relabel_seed: int = 1009,
    slot_assignment: str = "first-mention",
    resume: bool = False,
    device: str = "cuda:0",
    cpu_canary: bool = False,
    spec_override: ModelSpec | None = None,
    updates_override: int | None = None,
    batch_override: int | None = None,
    winning_arm: str = "AB",
    selection_receipt: dict[str, Any] | None = None,
) -> tuple[HitchhikerTrainer, dict[str, Any]]:
    """Construct a verified HitchhikerTrainer and stream/resource accounting metadata."""
    validate_job(job, job["stage"], winning_arm=winning_arm)
    if device != "cuda:0" and not (device == "cpu" and cpu_canary is True):
        raise ValueError("production requires isolated CUDA; pass cpu_canary=True for tiny CPU checks")
    if device != "cpu" and (cpu_canary or spec_override is not None or updates_override is not None or batch_override is not None):
        raise ValueError("CPU canary overrides are forbidden on CUDA")
    if job["stage"] in ("stage2_followups", "stage3_scale_confirmation") and device != "cpu":
        validate_stage_selection(job["stage"], selection_receipt, winning_arm=winning_arm)

    spec = spec_override or resolve_scale_profile(job["profile"])
    batch_size = batch_override or job["batch_size"]
    is_cpu = device == "cpu"
    if is_cpu and spec.parameter_receipt().total > 100_000:
        raise ValueError("CPU mechanical check model exceeds 100,000 parameter bound")

    total_updates = resolve_exposure_schedule(
        job,
        stream=stream,
        base_data=base_data,
        batch_size=batch_size,
        updates_override=updates_override,
        is_cpu_canary=is_cpu,
    )

    if stream is None:
        stream = build_training_stream(
            job["arm"],
            base_data=base_data,
            total_updates=total_updates,
            batch_size=batch_size,
            data_seed=data_seed,
            order_seed=job["order_seed"],
            relabel_seed=relabel_seed,
            slot_assignment=slot_assignment,
            heldout_suites=heldout_suites,
        )
    validate_stream_contract(
        stream,
        expected_arm=job["arm"],
        total_updates=total_updates,
        batch_size=batch_size,
    )
    if not is_cpu and stream.get("data_custody_verified") is not True:
        raise ValueError("CUDA training requires stream with data_custody_verified=True")

    effective_batch = min(batch_size, len(stream["samples"]))
    configure_device(device)
    recipe = Recipe(
        seed=job["seed"],
        total_updates=total_updates,
        batch_size=effective_batch,
        micro_batch_size=min(job["micro_batch_size"], effective_batch),
        lr=job["lr"],
        objective="ce",
        sampling="uniform",
        schedule="cosine",
        precision="fp32",
    )
    method = "blocked"
    deterministic_stream_counts = {
        "accepted_candidates": stream["stream_metrics"]["accepted_candidates"],
        "rejected_candidates": stream["stream_metrics"]["rejected_candidates"],
    }
    trainer = HitchhikerTrainer(
        stream["samples"],
        spec=spec,
        recipe=recipe,
        source_commit=source_commit,
        root=Path(root),
        lineage=job["job_id"],
        data_provenance={
            "split": "train",
            "license": "repository LICENSE; newly authored synthetic worlds",
            "hitchhiker_protocol_sha256": PROTOCOL_SHA,
            "hitchhiker_job": job,
            "stream_generator": stream["generator_version"],
            "stream_counts": deterministic_stream_counts,
        },
        tokenizer_manifest=UTF8.manifest.record(),
        order_seed=job["order_seed"],
        method=method,
        use_static_indexer=stream["use_static_indexer"],
        aux_illegal_weight=job["aux_illegal_weight"],
        device=device,
        resume=resume,
    )
    return trainer, stream


def run_hitchhiker_updates(trainer: HitchhikerTrainer, *, save_every: int | None = None) -> dict[str, Any]:
    """Run remaining updates on trainer and return exact compute/resource accounting."""
    t0 = time.perf_counter()
    logs: list[dict[str, Any]] = []
    start_u = trainer.state.global_update
    for _ in range(start_u, trainer.recipe.total_updates):
        step_log = trainer.step()
        logs.append(step_log)
        if save_every and (step_log["update"] % save_every == 0 or step_log["update"] == trainer.recipe.total_updates):
            trainer.save()
    elapsed = time.perf_counter() - t0
    return {
        "updates_executed": len(logs),
        "global_update": trainer.state.global_update,
        "wall_seconds": elapsed,
        "supervised_tokens": sum(r["supervised_tokens"] for r in logs),
        "input_positions": sum(r["input_positions"] for r in logs),
        "examples_processed": sum(r["examples"] for r in logs),
        "final_loss": logs[-1]["loss"] if logs else None,
        "logs": logs,
    }
