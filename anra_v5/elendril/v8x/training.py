"""Transactional V8-X1 S20/canary trainer supporting D1-H interleaved states, D2 paired groups, and milestones."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.publication import quarantine_interrupted_publication, writer_lock
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.training import _dump, _load, batch_tensors, configure_device, runtime_identity
from v5_training.checkpoint import CheckpointStore
from v5_training.optimizer import build_adamw_optimizer, optimizer_group_receipt, validate_parameter_ownership
from v5_training.state import CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState, IdentityBindings, TrainingState

from .contracts import X1ModelSpec, X1_ARCHITECTURES, X1_MAX_SEQUENCE, X1_OBJECTIVES, build_x1_model_spec
from .data import X1D1HDataset, X1D2Dataset, X1D2PairedGroup, X1SupervisedState
from .models import build_x1_s20_model, expected_x1_active_parameter_names
from .objectives import compute_x1_objective_batch

_MAX_CPU_PARAMS = 100_000
_MAX_CPU_SAMPLES = 32
_MAX_CPU_UPDATES = 8


@dataclass(frozen=True, slots=True)
class X1TrainRecipe:
    """Versioned V8-X1 optimizer and training schedule recipe."""

    seed: int = 7
    total_updates: int = 2368
    batch_size: int = 16
    micro_batch_size: int = 2
    lr: float = 1e-4
    clip: float = 1.0
    objective: str = "standard"
    eq_weight: float = 0.1
    cf_weight: float = 0.1
    cf_margin: float = 0.5
    sampling: str = "uniform"
    schedule: str = "cosine"
    precision: str = "fp32"
    activation_checkpointing: bool = False
    version: str = "anra-x1-s20-recipe-v1"

    def validate(self) -> None:
        if self.version != "anra-x1-s20-recipe-v1":
            raise ValueError("unsupported X1TrainRecipe version")
        for f in ("seed", "total_updates", "batch_size", "micro_batch_size"):
            if type(getattr(self, f)) is not int:
                raise ValueError(f"X1TrainRecipe.{f} must be int")
        if not (0 <= self.seed < 2**32 and 1 <= self.total_updates <= 100_000 and 1 <= self.micro_batch_size <= self.batch_size <= 256):
            raise ValueError("X1TrainRecipe counters out of bounds")
        if not math.isfinite(self.lr) or self.lr <= 0 or not math.isfinite(self.clip) or self.clip <= 0:
            raise ValueError("X1TrainRecipe lr/clip must be finite and positive")
        if any(not math.isfinite(x) or x < 0 for x in (self.eq_weight, self.cf_weight, self.cf_margin)):
            raise ValueError("X1TrainRecipe loss coefficients must be nonnegative and finite")
        if self.objective not in X1_OBJECTIVES or self.sampling not in ("uniform", "interleaved") or self.schedule not in ("constant", "cosine") or self.precision != "fp32":
            raise ValueError(f"unsupported X1TrainRecipe config: obj={self.objective!r}, samp={self.sampling!r}, prec={self.precision!r}")


def _state_epoch_order(n: int, order_seed: int, epoch: int, base_order: Sequence[int] | None) -> tuple[int, ...]:
    pool = list(base_order) if base_order is not None else list(range(n))
    rng = random.Random(int(digest(["x1-state-order-v1", order_seed, epoch]), 16))
    if base_order is None or epoch > 0:
        rng.shuffle(pool)
    return tuple(pool)


class X1S20Trainer:
    """Transactional trainer for S20 / tiny X1 models on D1-H interleaved states or D2 paired groups."""

    def __init__(
        self,
        *,
        x1_spec: X1ModelSpec,
        recipe: X1TrainRecipe,
        order_seed: int,
        source_commit: str,
        root: Path,
        lineage: str,
        data_provenance: Mapping[str, Any],
        states: Sequence[X1SupervisedState] = (),
        interleaved_schedule: Sequence[int] | None = None,
        groups: Sequence[X1D2PairedGroup] = (),
        device: str = "cpu",
        cpu_canary: bool = False,
        resume: bool = False,
    ) -> None:
        x1_spec.assert_valid()
        recipe.validate()
        is_paired = recipe.objective != "standard"
        if is_paired:
            if not groups:
                raise ValueError(f"{recipe.objective} requires D2 paired groups")
            if x1_spec.architecture in ("B0",):
                raise ValueError("B0 does not support grouped D2 paired objectives")
        else:
            if not states:
                raise ValueError("standard objective requires non-empty supervised states")

        self.x1_spec = x1_spec
        self.recipe = recipe
        self.order_seed = int(order_seed)
        self.states = tuple(states or ())
        self.groups = tuple(groups or ())
        self.is_paired = is_paired
        self.device = torch.device(device)

        sample_count = sum(len(g.samples) for g in self.groups) if is_paired else len(self.states)
        if self.device.type == "cpu":
            if (
                not cpu_canary
                or sample_count > _MAX_CPU_SAMPLES
                or x1_spec.total_parameters() > _MAX_CPU_PARAMS
                or recipe.total_updates > _MAX_CPU_UPDATES
            ):
                raise ValueError("CPU canary limits exceeded for X1S20Trainer (<=100k params, <=32 samples, <=8 updates)")

        configure_device(str(self.device))
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(recipe.seed)

        self.model = build_x1_s20_model(x1_spec, recipe.seed, torch_module=torch).to(self.device)
        self.optimizer = build_adamw_optimizer(self.model, lr=recipe.lr, torch_module=torch)
        self.eq_weight = float(recipe.eq_weight) if recipe.objective in ("PAIR-EQ", "PAIR-EQ-CF") else 0.0
        self.cf_weight = float(recipe.cf_weight) if recipe.objective == "PAIR-EQ-CF" else 0.0
        self.cf_margin = float(recipe.cf_margin)

        # Deterministic epoch-indexed batch plan matching horizon
        self.plan: list[tuple[int, ...]] = []
        if is_paired:
            groups_per_step = max(1, recipe.batch_size // 4)
            n_g = len(self.groups)
            g_orders: dict[int, list[int]] = {}
            for u in range(recipe.total_updates):
                chosen: list[int] = []
                for slot in range(groups_per_step):
                    pos = u * groups_per_step + slot
                    ep, off = divmod(pos, n_g)
                    if ep not in g_orders:
                        order = list(range(n_g))
                        random.Random(int(digest(["x1-d2-group-order-v1", order_seed, ep]), 16)).shuffle(order)
                        g_orders[ep] = order
                    chosen.append(g_orders[ep][off])
                self.plan.append(tuple(chosen))
            self.counts = tuple(
                sum(len(self.groups[gi].samples[j].tokens) for gi in idxs for j in range(len(self.groups[gi].samples)))
                for idxs in self.plan
            )
            data_blob = {"groups": [g.group_id for g in self.groups], "provenance": dict(data_provenance)}
        else:
            n_s = len(self.states)
            s_orders: dict[int, tuple[int, ...]] = {}
            base_ord = tuple(interleaved_schedule) if interleaved_schedule is not None else None
            for u in range(recipe.total_updates):
                chosen = []
                for slot in range(recipe.batch_size):
                    pos = u * recipe.batch_size + slot
                    ep, off = divmod(pos, n_s)
                    if ep not in s_orders:
                        s_orders[ep] = _state_epoch_order(n_s, order_seed, ep, base_ord if recipe.sampling == "interleaved" else None)
                    chosen.append(s_orders[ep][off])
                self.plan.append(tuple(chosen))
            self.counts = tuple(sum(len(self.states[i].tokens) for i in idxs) for idxs in self.plan)
            data_blob = {"states": [s.task_id for s in self.states], "provenance": dict(data_provenance)}

        self.data_sha = digest(data_blob)
        self.runtime = runtime_identity(self.device)
        self.metadata = json.loads(
            canonical(
                {
                    "schema": "elendril-x1-s20-training-v1",
                    "backend_id": "x1_s20",
                    "x1_model_spec": x1_spec.canonical(),
                    "model_spec_sha256": x1_spec.sha256(),
                    "recipe": asdict(recipe),
                    "order_seed": self.order_seed,
                    "plan_sha256": digest(self.plan),
                    "tokenizer": UTF8.manifest.record(),
                    "provenance": dict(data_provenance),
                    "runtime": self.runtime,
                }
            )
        )
        identities = IdentityBindings(
            IDENTITY_SCHEMA,
            source_commit,
            x1_spec.sha256(),
            UTF8.identity,
            self.data_sha,
            self.data_sha,
            digest(self.metadata),
            optimizer_group_receipt(self.model, self.optimizer)["sha256"],
            digest([recipe.schedule, recipe.lr, recipe.total_updates]),
            digest([recipe.sampling, self.order_seed]),
            sampler_spec_sha256=digest(self.plan),
        )
        self.store = CheckpointStore(Path(root), lineage, payload_validator=self.validate_payloads)
        self.parent = self.store.latest_sha256()
        self.state = TrainingState.initial(
            lineage_id=lineage,
            token_budget=sum(self.counts),
            tokens_per_update=max(self.counts),
            cursor=CursorState(CURSOR_SCHEMA, self.data_sha, 0, 0, 0),
            rng_state_sha256=sha256(self.rng_bytes()).hexdigest(),
            curriculum_phase=f"x1-{x1_spec.architecture}-{recipe.objective}",
            identities=identities,
        )
        self.faulted = False
        self.published_update = 0
        self._first_update_norm_ok = False
        self.initial_state_snapshot = {
            "init_weights_sha256": digest({k: sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k, v in sorted(self.model.state_dict().items())}),
            "parameter_shapes": {k: list(v.shape) for k, v in sorted(self.model.state_dict().items())},
            "optimizer_group_sha256": identities.optimizer_spec_sha256,
        }

        if resume:
            if self.parent is None:
                raise ValueError("no committed X1 S20 checkpoint to resume")
            state, payloads = self.store.restore(self.parent)
            self.model.load_state_dict(_load(payloads["model.bin"]), strict=True)
            self.optimizer.load_state_dict(_load(payloads["optimizer.bin"]))
            self.rng.set_state(torch.tensor(list(payloads["rng.bin"]), dtype=torch.uint8))
            self.state = replace(state)
            self.published_update = state.global_update
        elif self.parent is not None:
            raise ValueError("existing X1 S20 lineage requires explicit resume=True")

    def rng_bytes(self) -> bytes:
        return bytes(self.rng.get_state().tolist())

    def next_plan_indices(self) -> tuple[int, ...]:
        if self.state.global_update >= len(self.plan):
            return ()
        return self.plan[self.state.global_update]

    def learning_rate(self, update: int) -> float:
        if self.recipe.schedule == "constant":
            return float(self.recipe.lr)
        return float(
            self.recipe.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * update / self.recipe.total_updates)))
        )

    def validate_payloads(self, state: TrainingState, payloads: dict[str, bytes]) -> None:
        if state.identities != self.state.identities:
            raise ValueError("X1 checkpoint source/model/tokenizer/data/recipe mismatch")
        sched = json.loads(payloads["scheduler.json"])
        if sched.get("metadata") != self.metadata:
            raise ValueError("X1 checkpoint metadata mismatch")
        if sha256(payloads["rng.bin"]).hexdigest() != state.rng_state_sha256:
            raise ValueError("X1 checkpoint RNG digest mismatch")
        weights = _load(payloads["model.bin"])
        expected = self.model.state_dict()
        if set(weights) != set(expected):
            raise ValueError("X1 checkpoint tensor names mismatch")
        for name, tensor in weights.items():
            ref = expected[name]
            if not isinstance(tensor, torch.Tensor) or tensor.shape != ref.shape or tensor.dtype != ref.dtype or not torch.isfinite(tensor).all():
                raise ValueError(f"X1 checkpoint tensor invalid: {name}")

    def _step_b0_states(self, chosen: Sequence[X1SupervisedState], lr: float) -> dict[str, Any]:
        from v5_model.core import packed_layout

        total_loss = 0.0
        denom = float(sum(sum(1 for e in s.eligible[1:] if e) for s in chosen))
        if denom <= 0:
            raise ValueError("B0 batch has zero eligible tokens")
        for start in range(0, len(chosen), self.recipe.micro_batch_size):
            chunk = chosen[start : start + self.recipe.micro_batch_size]
            tokens, segments, eligible = batch_tensors([(s.tokens, s.eligible) for s in chunk], self.device)
            positions, mask_attn = packed_layout(segments, torch_module=torch)
            logits = self.model(tokens, positions, mask_attn, use_activation_checkpointing=self.recipe.activation_checkpointing)
            losses = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                tokens[:, 1:].reshape(-1),
                reduction="none",
            ).reshape_as(tokens[:, 1:])
            mask = eligible[:, 1:].to(losses.dtype)
            val = (losses * mask).sum()
            if not torch.isfinite(val):
                raise FloatingPointError("nonfinite B0 forward loss")
            (val / denom).backward()
            total_loss += float(val.detach().item())
        return {
            "loss": total_loss / denom,
            "ce_loss": total_loss / denom,
            "eq_loss": 0.0,
            "cf_loss": 0.0,
            "has_answer_targets": True,
        }

    def step(self) -> dict[str, Any]:
        if self.faulted or self.state.complete:
            raise ValueError("complete or faulted X1S20Trainer requires a fresh restore")
        self.faulted = True
        u = self.state.global_update
        lr = self.learning_rate(u)
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        if self.is_paired:
            chosen_groups = [self.groups[gi] for gi in self.plan[u]]
            total_sm = sum(1 for g in chosen_groups for s in g.samples if s.state_mask)
            total_eq = sum(1 for g in chosen_groups for ep in g.eq_pairs if ep.eligible)
            total_cf = sum(1 for g in chosen_groups for cp in g.cf_pairs if cp.eligible)
            g_chunk = max(1, min(len(chosen_groups), self.recipe.micro_batch_size))
            acc_loss = 0.0
            acc_ce = 0.0
            acc_eq = 0.0
            acc_cf = 0.0
            has_ans = False
            for start in range(0, len(chosen_groups), g_chunk):
                sub_groups = chosen_groups[start : start + g_chunk]
                m = compute_x1_objective_batch(
                    self.model,
                    objective=self.recipe.objective,
                    groups=sub_groups,
                    device=self.device,
                    eq_weight=self.eq_weight,
                    cf_weight=self.cf_weight,
                    cf_margin=self.cf_margin,
                    ce_denom_override=float(max(total_sm, 1)),
                    eq_denom_override=float(max(total_eq, 1)),
                    cf_denom_override=float(max(total_cf, 1)),
                    backward=True,
                )
                acc_loss += float(m["loss_value"])
                acc_ce += float(m["ce_weighted"])
                acc_eq += float(m["eq_weighted"])
                acc_cf += float(m["cf_weighted"])
                has_ans = has_ans or bool(m["has_answer_targets"])
            step_stats = {"loss": acc_loss, "ce_loss": acc_ce, "eq_loss": acc_eq, "cf_loss": acc_cf, "has_answer_targets": has_ans}
        else:
            chosen_states = [self.states[i] for i in self.plan[u]]
            if self.x1_spec.architecture == "B0":
                step_stats = self._step_b0_states(chosen_states, lr)
            else:
                denom = float(len(chosen_states))
                acc_loss = 0.0
                has_ans = any(s.gold_op == 0 for s in chosen_states)
                for start in range(0, len(chosen_states), self.recipe.micro_batch_size):
                    chunk = chosen_states[start : start + self.recipe.micro_batch_size]
                    m = compute_x1_objective_batch(
                        self.model,
                        objective="standard",
                        states=chunk,
                        device=self.device,
                        ce_denom_override=denom,
                        backward=True,
                    )
                    acc_loss += float(m["loss_value"])
                step_stats = {"loss": acc_loss, "ce_loss": acc_loss, "eq_loss": 0.0, "cf_loss": 0.0, "has_answer_targets": has_ans}

        validate_parameter_ownership(self.model, self.optimizer)
        expected_active = expected_x1_active_parameter_names(
            self.model, self.x1_spec, has_answer_targets=bool(step_stats["has_answer_targets"])
        )
        for name, param in self.model.named_parameters():
            if name in expected_active:
                if param.grad is None:
                    raise ValueError(f"missing gradient for active parameter {name}")
                if not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"nonfinite gradient for {name}")

        before_sample = None
        if u == 0:
            before_sample = {k: v.detach().clone() for k, v in self.model.named_parameters() if k in expected_active}

        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe.clip, error_if_nonfinite=True)
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise FloatingPointError("nonfinite parameter after optimizer step")
        if before_sample is not None:
            changed = sum(1 for k, prev in before_sample.items() if not torch.equal(prev, dict(self.model.named_parameters())[k]))
            if changed == 0:
                raise RuntimeError("first optimizer update made zero parameter change across all active tensors")
            self._first_update_norm_ok = True

        step_tokens = self.counts[u]
        self.state = replace(
            self.state,
            generation=u + 1,
            global_update=u + 1,
            cumulative_tokens=self.state.cumulative_tokens + step_tokens,
            tokens_by_source={"x1": self.state.cumulative_tokens + step_tokens},
            optimizer_step_max=u + 1,
            schedule_tokens=self.state.cumulative_tokens + step_tokens,
            cursor=replace(self.state.cursor, sequence_ordinal=u + 1),
            rng_state_sha256=sha256(self.rng_bytes()).hexdigest(),
            parent_checkpoint_sha256=self.parent,
        )
        self.state.assert_valid()
        self.faulted = False
        return {
            "update": u + 1,
            "loss": float(step_stats["loss"]),
            "ce_loss": float(step_stats["ce_loss"]),
            "eq_loss": float(step_stats["eq_loss"]),
            "cf_loss": float(step_stats["cf_loss"]),
            "pre_clip": float(norm),
            "lr": float(lr),
            "supervised_tokens": int(step_tokens),
            "states": int(len(self.plan[u]) * (4 if self.is_paired else 1)),
        }

    def save(self, *, inject_crash_at: str | None = None, prune_old: bool = True) -> str:
        if self.faulted or self.state.global_update <= self.published_update:
            raise ValueError("cannot publish faulted or repeated X1 update")
        state = replace(self.state, parent_checkpoint_sha256=self.parent)
        payloads = {
            "model.bin": _dump({k: v.detach().cpu() for k, v in self.model.state_dict().items()}),
            "optimizer.bin": _dump(self.optimizer.state_dict()),
            "rng.bin": self.rng_bytes(),
            "scheduler.json": canonical({"metadata": self.metadata, "scaler": {}}),
            "cursor.json": canonical(asdict(state.cursor)),
            "ledger.json": canonical(dict(state.tokens_by_source)),
            "training_state.json": canonical(state.canonical()),
        }
        self.faulted = True
        with writer_lock(self.store.lineage_root):
            quarantine_interrupted_publication(self.store.lineage_root, state.generation)
            self.parent = self.store.publish(
                state=state,
                payloads=payloads,
                expected_parent_sha256=self.parent,
                inject_crash_at=inject_crash_at,
            )
            if inject_crash_at is None and prune_old:
                self.store.prune(keep=set())
        self.published_update = state.global_update
        self.state = state
        self.faulted = False
        return self.parent

    def save_inference_milestone_weights(self, milestones_dir: Path, *, update: int, shared_backbone_dir: Path | None = None) -> Path:
        """Persist inference-only model weights (never falsely labeled resume-capable)."""

        milestones_dir.mkdir(parents=True, exist_ok=True)
        sd = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        out_dir = milestones_dir / f"u{int(update):04d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Deduplicate common initial core weights once per init_seed when saving update == 0
        if int(update) == 0 and shared_backbone_dir is not None and self.x1_spec.architecture != "B0":
            shared_backbone_dir.mkdir(parents=True, exist_ok=True)
            bb_path = shared_backbone_dir / f"init_backbone_s{self.recipe.seed}_{self.x1_spec.base_spec.sha256()[:12]}.bin"
            core_keys = {k: v for k, v in sd.items() if k.startswith("core.")}
            head_keys = {k: v for k, v in sd.items() if not k.startswith("core.")}
            bb_bytes = _dump(core_keys)
            bb_sha = sha256(bb_bytes).hexdigest()
            if not bb_path.exists():
                tmp_bb = bb_path.with_suffix(f".{os.getpid()}.tmp")
                tmp_bb.write_bytes(bb_bytes)
                os.replace(tmp_bb, bb_path)
            head_path = out_dir / "head_weights.bin"
            head_bytes = _dump(head_keys)
            head_path.write_bytes(head_bytes)
            meta = {
                "schema": "elendril-x1-inference-milestone-v1",
                "resume_capable": False,
                "global_update": 0,
                "architecture": self.x1_spec.architecture,
                "model_spec_sha256": self.x1_spec.sha256(),
                "shared_backbone_file": bb_path.name,
                "shared_backbone_sha256": bb_sha,
                "head_weights_file": "head_weights.bin",
                "head_weights_sha256": sha256(head_bytes).hexdigest(),
            }
            (out_dir / "milestone_weights_manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return out_dir

        w_path = out_dir / "model_inference.bin"
        w_bytes = _dump(sd)
        w_path.write_bytes(w_bytes)
        meta = {
            "schema": "elendril-x1-inference-milestone-v1",
            "resume_capable": False,
            "global_update": int(update),
            "architecture": self.x1_spec.architecture,
            "model_spec_sha256": self.x1_spec.sha256(),
            "weights_file": "model_inference.bin",
            "weights_sha256": sha256(w_bytes).hexdigest(),
        }
        (out_dir / "milestone_weights_manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out_dir


def create_x1_s20_trainer(
    dataset: X1D1HDataset | X1D2Dataset,
    job: Mapping[str, Any],
    *,
    source_commit: str,
    root: Path,
    device: str = "cpu",
    cpu_canary: bool = False,
    resume: bool = False,
    x1_spec_override: X1ModelSpec | None = None,
    updates_override: int | None = None,
    batch_size_override: int | None = None,
    micro_batch_size_override: int | None = None,
) -> X1S20Trainer:
    """Construct an X1S20Trainer directly from a registered job dict and X1 dataset."""

    arch = str(job["architecture"])
    if arch not in X1_ARCHITECTURES:
        raise ValueError(f"unknown X1 architecture: {arch!r}")
    spec = x1_spec_override or build_x1_model_spec(arch)
    objective = str(job["objective"])
    is_paired = objective != "standard"
    updates = int(updates_override) if updates_override is not None else int(job["updates"])
    batch = int(batch_size_override) if batch_size_override is not None else int(job.get("effective_state_batch", 16))
    micro = int(micro_batch_size_override) if micro_batch_size_override is not None else (1 if is_paired else 2)
    recipe = X1TrainRecipe(
        seed=int(job["init_seed"]),
        total_updates=updates,
        batch_size=batch,
        micro_batch_size=micro,
        lr=float(job.get("lr", 1e-4)),
        objective=objective,
        sampling="uniform" if is_paired else "interleaved",
    )
    if isinstance(dataset, X1D2Dataset):
        return X1S20Trainer(
            x1_spec=spec,
            recipe=recipe,
            order_seed=int(job["order_seed"]),
            source_commit=source_commit,
            root=root,
            lineage=str(job["job_id"]),
            data_provenance=dataset.manifest,
            groups=dataset.groups,
            device=device,
            cpu_canary=cpu_canary,
            resume=resume,
        )
    return X1S20Trainer(
        x1_spec=spec,
        recipe=recipe,
        order_seed=int(job["order_seed"]),
        source_commit=source_commit,
        root=root,
        lineage=str(job["job_id"]),
        data_provenance=dataset.manifest,
        states=dataset.states,
        interleaved_schedule=dataset.interleaved_schedule,
        device=device,
        cpu_canary=cpu_canary,
        resume=resume,
    )


__all__ = [
    "X1S20Trainer",
    "X1TrainRecipe",
    "create_x1_s20_trainer",
]
