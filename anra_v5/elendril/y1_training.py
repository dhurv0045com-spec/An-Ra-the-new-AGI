"""AN-RA V8-Y1 deterministic trainer supporting T0, T1 factorial, T3 Group-CE continuation, and T3 S20."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import math
from pathlib import Path
import time
from typing import Any

import torch

from v5_contracts.model_spec import ModelSpec
from v5_training.checkpoint import CheckpointStore
from v5_training.optimizer import validate_parameter_ownership
from .codec import digest
from .foundation_training import Recipe, StageTransition
from .hitchhiker_diagnostics import (
    ANSWER_TOKEN_IDS,
    TOKEN_ABSTAIN,
    TOKEN_DELIBERATE,
    TOKEN_Q,
    TOKEN_R,
)
from .representation import UTF8
from .training import MODEL, batch_tensors, configure_device
from .v6_training import V6Trainer
from .y1_protocol import TRAIN_HYPERPARAMS, resolve_y1_scale


CPU_CANARY_SPEC = replace(MODEL, vocabulary_size=UTF8.vocabulary_size, context_length=512)


_ANSWER_SET = frozenset(ANSWER_TOKEN_IDS)
_NAMED_GROUP_SET = _ANSWER_SET | {TOKEN_Q, TOKEN_R, TOKEN_ABSTAIN, TOKEN_DELIBERATE}
_OTHER_IDS = tuple(i for i in range(UTF8.vocabulary_size) if i not in _NAMED_GROUP_SET)


def first_action_group_id(token_id: int) -> int:
    """Return group index in {0:ANSWER, 1:Q, 2:R, 3:!, 4:~, 5:OTHER} from public target token."""
    if token_id in _ANSWER_SET:
        return 0
    if token_id == TOKEN_Q:
        return 1
    if token_id == TOKEN_R:
        return 2
    if token_id == TOKEN_ABSTAIN:
        return 3
    if token_id == TOKEN_DELIBERATE:
        return 4
    return 5


class Y1Trainer(V6Trainer):
    """Deterministic trainer supporting ordinary CE and exact first-action Group-CE."""

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
        group_ce_weight: float = 0.0,
        device: str = "cpu",
        resume: bool = False,
        transition: StageTransition | None = None,
        parent_store: CheckpointStore | None = None,
    ) -> None:
        if type(group_ce_weight) not in (int, float) or not math.isfinite(group_ce_weight) or group_ce_weight < 0.0:
            raise ValueError("group_ce_weight must be finite and non-negative")
        self.group_ce_weight = float(group_ce_weight)
        provenance = {
            **data_provenance,
            "y1_group_ce_weight": self.group_ce_weight,
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
            transition=transition,
            parent_store=parent_store,
        )
        self._answer_ids_t = torch.tensor(ANSWER_TOKEN_IDS, device=self.device, dtype=torch.int64)
        self._other_ids_t = torch.tensor(_OTHER_IDS, device=self.device, dtype=torch.int64)

    def indices(self, update: int) -> tuple[int, ...]:
        start = update * self.recipe.batch_size
        end = start + self.recipe.batch_size
        if end > len(self.samples):
            raise ValueError(f"update {update} exceeds materialized Y1 sample stream ({len(self.samples)})")
        return tuple(range(start, end))

    def sample_cursor(self, completed_updates: int) -> int:
        return completed_updates * self.recipe.batch_size

    def _compute_microbatch_terms(
        self,
        chunk: list[Any],
        chunk_weights: list[list[float]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            ce_sum = (losses * loss_weights[:, 1:]).sum()
            group_sum = ce_sum.new_tensor(0.0)
            if self.group_ce_weight > 0.0:
                terms = []
                for row_idx, sample in enumerate(chunk):
                    first = sample.eligible.index(True)
                    probs = logits[row_idx, first - 1].float().softmax(-1)
                    gid = first_action_group_id(sample.tokens[first])
                    if gid == 0:
                        p_gold = probs[self._answer_ids_t].sum()
                    elif gid == 1:
                        p_gold = probs[TOKEN_Q]
                    elif gid == 2:
                        p_gold = probs[TOKEN_R]
                    elif gid == 3:
                        p_gold = probs[TOKEN_ABSTAIN]
                    elif gid == 4:
                        p_gold = probs[TOKEN_DELIBERATE]
                    else:
                        p_gold = probs[self._other_ids_t].sum()
                    terms.append(-torch.log(torch.clamp(p_gold, min=1e-12)))
                if terms:
                    group_sum = torch.stack(terms).sum()
        return logits, ce_sum, group_sum

    def step(self) -> dict[str, Any]:
        if self.faulted or self.state.complete:
            raise ValueError("complete or faulted engine requires a fresh restore")
        self.faulted = True
        u = self.state.global_update
        chosen = [self.samples[i] for i in self.plan[u]]
        batch_len = len(chosen)
        weights = [[float(x) for x in sample.eligible] for sample in chosen]
        denominator = sum(map(sum, weights))
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        lr = self.learning_rate(u)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        ce_num = 0.0
        group_num = 0.0
        for start in range(0, batch_len, self.recipe.micro_batch_size):
            chunk = chosen[start : start + self.recipe.micro_batch_size]
            chunk_weights = weights[start : start + len(chunk)]
            logits, ce_sum, group_sum = self._compute_microbatch_terms(chunk, chunk_weights)
            objective = (ce_sum / denominator) + (self.group_ce_weight * (group_sum / batch_len) if self.group_ce_weight > 0.0 else 0.0)
            if not torch.isfinite(logits).all() or not torch.isfinite(objective):
                raise FloatingPointError("nonfinite forward; update rejected")
            self.scaler.scale(objective).backward()
            ce_num += float(ce_sum.item())
            group_num += float(group_sum.item())

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
        ce_loss = ce_num / denominator
        group_loss = group_num / batch_len
        return {
            "update": u + 1,
            "loss": ce_loss + self.group_ce_weight * group_loss,
            "ce_loss": ce_loss,
            "group_ce_loss": group_loss,
            "group_ce_weight": self.group_ce_weight,
            "weighted_denominator": denominator,
            "supervised_tokens": self.counts[u],
            "input_positions": sum(len(s.tokens) for s in chosen),
            "examples": batch_len,
            "microbatches": math.ceil(batch_len / self.recipe.micro_batch_size),
            "pre_clip": float(norm),
            "lr": lr,
        }


def verify_group_ce_gradient_equivalence(trainer: Y1Trainer) -> dict[str, Any]:
    """Compare microbatch-accumulated gradients vs single-batch gradients including L_group."""
    u = trainer.state.global_update
    chosen = [trainer.samples[i] for i in trainer.plan[u]]
    batch_len = len(chosen)
    weights = [[float(x) for x in sample.eligible] for sample in chosen]
    denominator = sum(map(sum, weights))
    parameters = [p for _, p in trainer.model.named_parameters()]
    modes = [(m, m.training) for m in trainer.model.modules()]
    trainer.model.eval()

    def accum(micro: int) -> tuple[float, list[torch.Tensor]]:
        total = [torch.zeros_like(p) for p in parameters]
        obj_sum = 0.0
        for start in range(0, batch_len, micro):
            chunk = chosen[start : start + micro]
            cw = weights[start : start + len(chunk)]
            _, ce_sum, grp_sum = trainer._compute_microbatch_terms(chunk, cw)
            obj = (ce_sum / denominator) + (trainer.group_ce_weight * (grp_sum / batch_len) if trainer.group_ce_weight > 0.0 else 0.0)
            grads = torch.autograd.grad(obj, parameters)
            if not torch.isfinite(obj) or any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError("nonfinite gradient in equivalence check")
            for t, g in zip(total, grads):
                t.add_(g)
            obj_sum += float(obj.detach())
        return obj_sum, total

    try:
        ref_loss, ref_grads = accum(batch_len)
        act_loss, act_grads = accum(trainer.recipe.micro_batch_size)
    finally:
        for m, md in modes:
            m.training = md

    max_diff = max(float((a - b).abs().max()) for a, b in zip(ref_grads, act_grads))
    ok = math.isclose(ref_loss, act_loss, rel_tol=1e-5, abs_tol=1e-6) and all(
        torch.allclose(a, b, rtol=1e-4, atol=1e-6) for a, b in zip(ref_grads, act_grads)
    )
    return {"status": "PASS" if ok else "FAIL", "reference_loss": ref_loss, "microbatch_loss": act_loss, "max_grad_diff": max_diff}


def create_y1_trainer(
    *,
    job: dict[str, Any],
    samples: tuple[Any, ...],
    data_provenance: dict[str, Any],
    source_commit: str,
    root: Path,
    device: str = "cuda:0",
    cpu_canary: bool = False,
    resume: bool = False,
    spec_override: ModelSpec | None = None,
    updates_override: int | None = None,
    batch_override: int | None = None,
    parent_store: CheckpointStore | None = None,
    parent_checkpoint_sha: str | None = None,
) -> tuple[Y1Trainer, Recipe]:
    """Instantiate a verified Y1Trainer with exact optimizer/schedule/transition bindings."""
    if device != "cuda:0" and not (device == "cpu" and cpu_canary):
        raise ValueError("production training requires isolated CUDA device cuda:0")
    if device != "cpu" and (spec_override is not None or updates_override is not None or batch_override is not None):
        raise ValueError("CPU canary overrides forbidden on CUDA")

    spec = spec_override or (CPU_CANARY_SPEC if cpu_canary and device == "cpu" else resolve_y1_scale(job["profile"]))
    batch_size = batch_override or int(job.get("batch_size", TRAIN_HYPERPARAMS["batch_size"]))
    micro_batch = min(int(job.get("micro_batch_size", TRAIN_HYPERPARAMS["micro_batch_size"])), batch_size)
    total_updates = updates_override or int(job["updates"])
    if len(samples) < total_updates * batch_size:
        raise ValueError(f"samples length {len(samples)} < required {total_updates * batch_size}")

    configure_device(device)
    recipe = Recipe(
        seed=int(job["seed"]),
        total_updates=total_updates,
        batch_size=batch_size,
        micro_batch_size=micro_batch,
        lr=float(job.get("lr", TRAIN_HYPERPARAMS["lr_peak"])),
        clip=float(TRAIN_HYPERPARAMS["grad_clip"]),
        objective="ce",
        sampling="uniform",
        schedule="cosine",
        precision="fp32",
    )

    transition: StageTransition | None = None
    if parent_store is not None and not resume:
        if not parent_checkpoint_sha:
            raise ValueError("parent_checkpoint_sha required when migrating from parent_store")
        p_state, p_payloads = parent_store.restore(parent_checkpoint_sha)
        p_sched = __import__("json").loads(p_payloads["scheduler.json"])
        parent_recipe_sha = digest(p_sched["metadata"]["recipe"])
        data_sha = digest({"samples": [asdict(s) for s in samples], "provenance": {**data_provenance, "y1_group_ce_weight": float(job.get("group_ce_weight", 0.0))}})
        transition = StageTransition(
            parent_checkpoint=parent_checkpoint_sha,
            parent_recipe=parent_recipe_sha,
            child_recipe=digest(asdict(recipe)),
            child_data=data_sha,
            optimizer_policy="reset",
        )

    trainer = Y1Trainer(
        samples,
        spec=spec,
        recipe=recipe,
        source_commit=source_commit,
        root=Path(root),
        lineage=job["job_id"],
        data_provenance=data_provenance,
        tokenizer_manifest=UTF8.manifest.record(),
        group_ce_weight=float(job.get("group_ce_weight", 0.0)),
        device=device,
        resume=resume,
        transition=transition,
        parent_store=parent_store,
    )
    return trainer, recipe
