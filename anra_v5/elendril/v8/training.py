"""V8 training engines for byte-autoregressive control (B0) and typed heads (J/F/P/F-cap/U).

Constructs versioned models explicitly before AdamW parameter binding, enforces
per-objective active parameter gradient checks, preserves exact macrobatch
accumulation parity, and uses transactional CheckpointStore persistence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Any

import torch

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.foundation_training import (
    Recipe,
    StageTransition,
    TokenSample,
    tensor_inventory,
)
from anra_v5.elendril.publication import quarantine_interrupted_publication, writer_lock
from anra_v5.elendril.representation import UTF8, SpecialToken
from anra_v5.elendril.training import _dump, _load, batch_tensors, configure_device, runtime_identity
from anra_v5.elendril.v7_training import METHODS as V7_METHODS, ordered_indices
from anra_v5.elendril.v717_training import IndependentOrderTrainer
from v5_training.checkpoint import CheckpointStore
from v5_training.optimizer import build_adamw_optimizer, optimizer_group_receipt, validate_parameter_ownership
from v5_training.state import CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState, IdentityBindings, TrainingState

from .contracts import SCREEN_MAX_SEQUENCE, SYMBOLS, V8ModelSpec, build_v8_model_spec
from .data import build_d1_training_dataset, load_d0_dataset
from .models import (
    OP_TO_INDEX,
    build_v8_model,
    compute_typed_nll,
    expected_active_parameter_names,
)
from .protocol import PHASES, PROTOCOL_SHA, jobs as protocol_jobs
from .teacher import TeacherStep

_MAX_CPU_PARAMS = 100_000
_MAX_CPU_SAMPLES = 32
_MAX_CPU_UPDATES = 8


@dataclass(frozen=True, slots=True)
class V8TypedSample:
    """One supervised action state framed for V8 typed heads."""

    task_id: str
    tokens: tuple[int, ...]
    eligible: tuple[bool, ...]
    action_position: int
    gold_op: int
    gold_arg: int
    world_id: str
    category: str = "ordinary"
    stage: str = "formation"


@dataclass(frozen=True)
class V8TrainingData:
    """Prepared training corpus + provenance bound into trainer identity."""

    dataset_name: str
    data_identity: str
    typed_samples: tuple[V8TypedSample, ...]
    byte_samples: tuple[TokenSample, ...]
    manifest: dict[str, object]


def _step_op_arg_indices(step: TeacherStep) -> tuple[int, int]:
    if step.gold_op not in OP_TO_INDEX:
        raise ValueError(f"unsupported gold_op: {step.gold_op!r}")
    op_idx = OP_TO_INDEX[step.gold_op]
    if op_idx == 0:
        if step.gold_arg not in SYMBOLS:
            raise ValueError(f"ANSWER step missing symbol arg: {step.gold_arg!r}")
        return op_idx, SYMBOLS.index(step.gold_arg)
    return op_idx, 0


def teacher_steps_to_typed_samples(
    steps: tuple[TeacherStep, ...],
    *,
    max_length: int = SCREEN_MAX_SEQUENCE,
) -> tuple[V8TypedSample, ...]:
    """Frame public TeacherSteps into typed samples anchored at the ANSWER marker."""

    samples: list[V8TypedSample] = []
    answer_marker_id = UTF8.special(SpecialToken.ANSWER)
    for step in steps:
        framed = UTF8.frame(step.public_input.text_utf8, step.target_text, max_length=max_length)
        action_pos = framed.answer_span[0] - 1
        if action_pos < 0 or framed.ids[action_pos] != answer_marker_id:
            raise ValueError("framed TeacherStep missing ANSWER marker immediately before target")
        op_idx, arg_idx = _step_op_arg_indices(step)
        samples.append(
            V8TypedSample(
                task_id=step.task_id,
                tokens=framed.ids,
                eligible=framed.loss_mask,
                action_position=action_pos,
                gold_op=op_idx,
                gold_arg=arg_idx,
                world_id=step.world_id,
                category=step.family,
                stage=step.stage,
            )
        )
    return tuple(samples)


def teacher_steps_to_byte_samples(
    steps: tuple[TeacherStep, ...],
    *,
    max_length: int = SCREEN_MAX_SEQUENCE,
) -> tuple[TokenSample, ...]:
    """Frame public TeacherSteps into TokenSamples for the B0 byte-autoregressive arm."""

    samples: list[TokenSample] = []
    for step in steps:
        framed = UTF8.frame(step.public_input.text_utf8, step.target_text, max_length=max_length)
        samples.append(
            TokenSample(
                task_id=step.task_id,
                tokens=framed.ids,
                eligible=framed.loss_mask,
                world_id=step.world_id,
                category=step.family,
            )
        )
    return tuple(samples)


def prepare_v8_training_data(
    dataset_name: str,
    *,
    d1_worlds_per_family: int = 512,
    sample_limit_for_cpu: int | None = None,
) -> V8TrainingData:
    """Build or load the exact D0 (blocked) or D1 (uniform) training data bundle."""

    if dataset_name == "D0":
        d0 = load_d0_dataset()
        samples = d0.samples[:sample_limit_for_cpu] if sample_limit_for_cpu is not None else d0.samples
        manifest = {
            **d0.manifest,
            "dataset": "D0",
            "d0_identity": d0.identity,
            "v7_sampling_method": "blocked",
            "sample_count": len(samples),
            "slice_limit": sample_limit_for_cpu,
        }
        return V8TrainingData(
            dataset_name="D0",
            data_identity=digest(manifest) if sample_limit_for_cpu is not None else d0.identity,
            typed_samples=(),
            byte_samples=tuple(samples),
            manifest=manifest,
        )
    if dataset_name == "D1":
        d1 = build_d1_training_dataset(quota_per_family=d1_worlds_per_family)
        steps = d1.steps[:sample_limit_for_cpu] if sample_limit_for_cpu is not None else d1.steps
        typed = teacher_steps_to_typed_samples(steps)
        byte_s = teacher_steps_to_byte_samples(steps)
        manifest = {
            "split": "train",
            "license": "internal-v8-procedural-corpus",
            "dataset": "D1",
            "d1_identity": d1.identity,
            "d1_audit": d1.audit,
            "v7_sampling_method": "uniform",
            "world_count": len(d1.worlds),
            "sample_count": len(steps),
            "slice_limit": sample_limit_for_cpu,
        }
        return V8TrainingData(
            dataset_name="D1",
            data_identity=digest(manifest) if sample_limit_for_cpu is not None else d1.identity,
            typed_samples=typed,
            byte_samples=byte_s,
            manifest=manifest,
        )
    raise ValueError(f"unsupported V8 dataset: {dataset_name!r}")


class V8TypedTrainer:
    """Transactional trainer for V8 typed heads (J, F, P, F-cap, U)."""

    def __init__(
        self,
        samples: tuple[V8TypedSample, ...] | list[V8TypedSample],
        *,
        v8_spec: V8ModelSpec,
        recipe: Recipe,
        order_seed: int,
        method: str = "uniform",
        source_commit: str,
        root: Path,
        lineage: str,
        data_provenance: dict[str, object],
        tokenizer_manifest: dict[str, object] | None = None,
        device: str = "cpu",
        cpu_canary: bool = False,
        resume: bool = False,
        transition: StageTransition | None = None,
        parent_store: CheckpointStore | None = None,
    ) -> None:
        v8_spec.assert_valid()
        if v8_spec.architecture == "B0":
            raise ValueError("B0 uses IndependentOrderTrainer, not V8TypedTrainer")
        recipe.validate()
        if method not in V7_METHODS:
            raise ValueError(f"unknown sampling method: {method!r}")
        if type(order_seed) is not int or not 0 <= order_seed < 2**32:
            raise ValueError("order_seed must be uint32")
        if not samples:
            raise ValueError("training requires at least one V8TypedSample")
        if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
            raise ValueError("source_commit must be 40-char lowercase hex SHA")

        tok_manifest = tokenizer_manifest or UTF8.manifest.record()
        UTF8.require_manifest(tok_manifest)

        self.samples = tuple(samples)
        self.v8_spec = v8_spec
        self.recipe = recipe
        self.order_seed = order_seed
        self.method = method
        self.device = torch.device(device)
        if self.device.type == "cpu":
            if not cpu_canary:
                raise ValueError("CPU training requires explicit cpu_canary=True")
            if (
                len(self.samples) > _MAX_CPU_SAMPLES
                or v8_spec.total_parameters() > _MAX_CPU_PARAMS
                or recipe.total_updates > _MAX_CPU_UPDATES
            ):
                raise ValueError("CPU mechanical canary exceeds tiny fixture bounds")

        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(recipe.seed)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(recipe.precision == "fp16" and self.device.type == "cuda"))

        # Build model strictly BEFORE optimizer creation so head + backbone share one AdamW binding.
        self.model = build_v8_model(v8_spec, recipe.seed, torch_module=torch).to(self.device)
        self.optimizer = build_adamw_optimizer(self.model, lr=recipe.lr, torch_module=torch)

        self.codec_sha = UTF8.identity
        self.tokenizer = tok_manifest
        self.data_sha = digest({"samples": [asdict(s) for s in self.samples], "provenance": data_provenance})

        self._epoch_orders: dict[int, tuple[int, ...]] = {}
        self.plan = tuple(self.indices(u) for u in range(recipe.total_updates))
        # Count states/tokens deterministically for cursor and ledger monotonic accounting.
        self.counts = tuple(sum(len(self.samples[i].tokens) for i in batch) for batch in self.plan)
        self.runtime = runtime_identity(self.device)
        self.metadata: dict[str, Any] = {
            "schema": "elendril-v8-typed-training-v1",
            "recipe": asdict(recipe),
            "order_seed": order_seed,
            "method": method,
            "v8_model_spec": v8_spec.canonical(),
            "model_spec_sha256": v8_spec.sha256(),
            "tensors": tensor_inventory(self.model),
            "tokenizer": self.tokenizer,
            "provenance": data_provenance,
            "runtime": self.runtime,
            "plan_sha256": digest(self.plan),
        }
        if transition is not None:
            transition.validate()
            if transition.child_recipe != digest(asdict(recipe)) or transition.child_data != self.data_sha:
                raise ValueError("child stage recipe or data mismatch")
            self.metadata["stage_transition"] = asdict(transition)
        self.metadata = json.loads(canonical(self.metadata))

        identities = IdentityBindings(
            IDENTITY_SCHEMA,
            source_commit,
            v8_spec.sha256(),
            self.codec_sha,
            self.data_sha,
            self.data_sha,
            digest(self.metadata),
            optimizer_group_receipt(self.model, self.optimizer)["sha256"],
            digest([recipe.schedule, recipe.lr, recipe.total_updates]),
            digest([method, order_seed]),
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
            curriculum_phase=f"v8-{v8_spec.architecture}",
            identities=identities,
        )
        self.faulted = False
        self.published_update = 0

        if resume:
            if self.parent is None:
                raise ValueError("no committed generation to resume")
            state, payloads = self.store.restore(self.parent)
            self.model.load_state_dict(_load(payloads["model.bin"]), strict=True)
            self.optimizer.load_state_dict(_load(payloads["optimizer.bin"]))
            scaler_state = json.loads(payloads["scheduler.json"])["scaler"]
            if scaler_state:
                self.scaler.load_state_dict(scaler_state)
            self.rng.set_state(torch.tensor(list(payloads["rng.bin"]), dtype=torch.uint8))
            self.state = state
            self.published_update = state.global_update
        elif self.parent is not None:
            raise ValueError("existing lineage requires explicit resume=True")
        elif transition is not None:
            if parent_store is None:
                raise ValueError("stage migration requires a verified parent_store")
            self.migrate_stage(parent_store, transition)

    def indices(self, update: int) -> tuple[int, ...]:
        per_epoch = math.ceil(len(self.samples) / self.recipe.batch_size)
        epoch, batch = divmod(update, per_epoch)
        if epoch not in self._epoch_orders:
            self._epoch_orders[epoch] = ordered_indices(self.samples, self.order_seed, epoch, self.method)
        start = batch * self.recipe.batch_size
        return self._epoch_orders[epoch][start : start + self.recipe.batch_size]

    def sample_cursor(self, completed_updates: int) -> int:
        return completed_updates * self.recipe.batch_size

    def rng_bytes(self) -> bytes:
        return bytes(self.rng.get_state().tolist())

    def learning_rate(self, update: int) -> float:
        if self.recipe.schedule == "constant":
            return float(self.recipe.lr)
        return float(
            self.recipe.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * update / self.recipe.total_updates)))
        )

    def validate_payloads(self, state: TrainingState, payloads: dict[str, bytes]) -> None:
        if state.identities != self.state.identities:
            raise ValueError("checkpoint source/model/tokenizer/data/recipe/runtime mismatch")
        u = state.global_update
        expected_cursor = replace(self.state.cursor, sequence_ordinal=self.sample_cursor(u))
        if (
            not 1 <= u <= self.recipe.total_updates
            or state.cursor != expected_cursor
            or state.cumulative_tokens != sum(self.counts[:u])
            or state.token_budget != sum(self.counts)
            or state.tokens_per_update != max(self.counts)
            or sha256(payloads["rng.bin"]).hexdigest() != state.rng_state_sha256
        ):
            raise ValueError("checkpoint RNG, token ledger, or next-update cursor mismatch")
        torch.Generator().set_state(torch.tensor(list(payloads["rng.bin"]), dtype=torch.uint8))
        scheduler = json.loads(payloads["scheduler.json"])
        if set(scheduler) != {"metadata", "scaler"} or scheduler["metadata"] != self.metadata:
            raise ValueError("checkpoint metadata/tensor manifest mismatch")
        scale_state = scheduler["scaler"]
        if not self.scaler.is_enabled() and scale_state != {}:
            raise ValueError("FP32 trainer cannot restore nonempty gradient scaler")
        if self.scaler.is_enabled():
            if (
                set(scale_state) != {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in scale_state.values())
                or scale_state["scale"] <= 0
                or scale_state["growth_factor"] <= 1
                or not 0 < scale_state["backoff_factor"] < 1
            ):
                raise ValueError("invalid FP16 scaler state")

        weights = _load(payloads["model.bin"])
        expected_state = self.model.state_dict()
        if set(weights) != set(expected_state):
            raise ValueError("checkpoint model tensor names mismatch")
        for name, tensor in weights.items():
            ref = expected_state[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != ref.shape
                or tensor.dtype != ref.dtype
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"checkpoint tensor invalid: {name}")

        opt = _load(payloads["optimizer.bin"])
        groups = opt.get("param_groups", [])
        expected_groups = self.optimizer.state_dict()["param_groups"]
        if len(groups) != len(expected_groups):
            raise ValueError("optimizer group count mismatch")
        ids: list[int] = []
        live_params: list[torch.nn.Parameter] = []
        for group, expected, live in zip(groups, expected_groups, self.optimizer.param_groups):
            if (
                group["params"] != expected["params"]
                or group["lr"] != self.learning_rate(u - 1)
                or {k: v for k, v in group.items() if k not in ("params", "lr")}
                != {k: v for k, v in expected.items() if k not in ("params", "lr")}
            ):
                raise ValueError("optimizer ownership or settings mismatch")
            ids.extend(group["params"])
            live_params.extend(live["params"])
        if set(opt["state"]) != set(ids):
            raise ValueError("optimizer state ownership mismatch")
        for identity, parameter in zip(ids, live_params):
            slot = opt["state"][identity]
            if (
                set(slot) != {"step", "exp_avg", "exp_avg_sq"}
                or not isinstance(slot["step"], torch.Tensor)
                or slot["step"].numel() != 1
                or slot["step"].item() != u
            ):
                raise ValueError("optimizer update counter mismatch")
            for key in ("exp_avg", "exp_avg_sq"):
                val = slot[key]
                if (
                    not isinstance(val, torch.Tensor)
                    or val.shape != parameter.shape
                    or val.dtype != parameter.dtype
                    or not torch.isfinite(val).all()
                    or (key == "exp_avg_sq" and (val < 0).any())
                ):
                    raise ValueError(f"optimizer moment invalid for {key}")

    def step(self, *, custom_batch: tuple[V8TypedSample, ...] | None = None) -> dict[str, object]:
        """Run one atomic macro-batch update with exact microbatch gradient accumulation."""

        if self.faulted or self.state.complete:
            raise ValueError("complete or faulted trainer requires a fresh restore")
        self.faulted = True
        u = self.state.global_update
        chosen = list(custom_batch) if custom_batch is not None else [self.samples[i] for i in self.plan[u]]
        if not chosen:
            raise ValueError("training update cannot receive empty batch")

        denominator = float(len(chosen))
        has_answer = any(s.gold_op == 0 for s in chosen)
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        lr = self.learning_rate(u)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        total_nll = 0.0
        total_op_nll = 0.0
        total_val_nll = 0.0
        gate_sum = 0.0
        gate_count = 0
        entropy_sum = 0.0
        entropy_count = 0

        use_amp = self.recipe.precision == "fp16" and self.device.type == "cuda"
        for start in range(0, len(chosen), self.recipe.micro_batch_size):
            chunk = chosen[start : start + self.recipe.micro_batch_size]
            tokens, segments, _ = batch_tensors([(s.tokens, s.eligible) for s in chunk], self.device)
            action_pos = torch.tensor([s.action_position for s in chunk], device=self.device, dtype=torch.int64)
            gold_ops = torch.tensor([s.gold_op for s in chunk], device=self.device, dtype=torch.int64)
            gold_args = torch.tensor([s.gold_arg for s in chunk], device=self.device, dtype=torch.int64)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                out = self.model.forward_action(
                    tokens,
                    segments,
                    action_pos,
                    use_activation_checkpointing=self.recipe.activation_checkpointing,
                )
                micro_nll, _ = compute_typed_nll(out, gold_ops, gold_args)

            if not torch.isfinite(out.log_probs_19).all() or not torch.isfinite(micro_nll):
                raise FloatingPointError("nonfinite forward output; update rejected")

            self.scaler.scale(micro_nll / denominator).backward()
            total_nll += float(micro_nll.item())
            with torch.no_grad():
                op_nll = -out.log_p_op.gather(1, gold_ops.unsqueeze(1)).squeeze(1)
                is_ans = (gold_ops == 0).to(op_nll.dtype)
                val_nll = -out.log_p_val.gather(1, gold_args.clamp(0, 15).unsqueeze(1)).squeeze(1) * is_ans
                total_op_nll += float(op_nll.sum().item())
                total_val_nll += float(val_nll.sum().item())
                if out.gate is not None:
                    gate_sum += float(out.gate.sum().item())
                    gate_count += int(out.gate.numel())
                if out.attention_entropy is not None and out.has_pointer_candidates is not None:
                    mask = out.has_pointer_candidates
                    if bool(mask.any().item()):
                        entropy_sum += float(out.attention_entropy[mask].sum().item())
                        entropy_count += int(mask.sum().item())

        self.scaler.unscale_(self.optimizer)
        validate_parameter_ownership(self.model, self.optimizer)
        expected_active = expected_active_parameter_names(self.model, self.v8_spec, has_answer_targets=has_answer)
        for name, param in self.model.named_parameters():
            if param.grad is None:
                raise ValueError(f"silent missing gradient tensor for {name}")
            if not torch.isfinite(param.grad).all():
                raise FloatingPointError(f"nonfinite gradient for {name}")
            if name in expected_active and param.grad.numel() == 0:
                raise ValueError(f"empty active parameter {name}")

        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe.clip, error_if_nonfinite=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if len(self.optimizer.state) != len(list(self.model.parameters())) or any(
            slot["step"].item() != u + 1 for slot in self.optimizer.state.values()
        ):
            raise ValueError("optimizer skipped or duplicated the proposed update")
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise FloatingPointError("nonfinite parameter after update")

        step_tokens = self.counts[u] if custom_batch is None else sum(len(s.tokens) for s in chosen)
        self.state = replace(
            self.state,
            generation=u + 1,
            global_update=u + 1,
            cumulative_tokens=self.state.cumulative_tokens + step_tokens,
            tokens_by_source={"action": self.state.cumulative_tokens + step_tokens},
            optimizer_step_max=u + 1,
            schedule_tokens=self.state.cumulative_tokens + step_tokens,
            cursor=replace(self.state.cursor, sequence_ordinal=self.sample_cursor(u + 1)),
            rng_state_sha256=sha256(self.rng_bytes()).hexdigest(),
            parent_checkpoint_sha256=self.parent,
        )
        self.state.assert_valid()
        self.faulted = False
        return {
            "update": u + 1,
            "loss": total_nll / denominator,
            "op_nll": total_op_nll / denominator,
            "val_nll": total_val_nll / max(sum(1 for s in chosen if s.gold_op == 0), 1),
            "answer_states": sum(1 for s in chosen if s.gold_op == 0),
            "states": len(chosen),
            "supervised_tokens": step_tokens,
            "pre_clip": float(norm),
            "lr": lr,
            "mean_gate": (gate_sum / gate_count) if gate_count else None,
            "mean_pointer_entropy": (entropy_sum / entropy_count) if entropy_count else None,
            "indices": self.plan[u] if custom_batch is None else tuple(range(len(chosen))),
        }

    def save(self, *, inject_crash_at: str | None = None) -> str:
        if self.faulted or self.state.global_update <= self.published_update:
            raise ValueError("cannot publish a failed or repeated update")
        state = replace(self.state, parent_checkpoint_sha256=self.parent)
        payloads = {
            "model.bin": _dump(self.model.state_dict()),
            "optimizer.bin": _dump(self.optimizer.state_dict()),
            "rng.bin": self.rng_bytes(),
            "scheduler.json": canonical({"metadata": self.metadata, "scaler": self.scaler.state_dict()}),
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
        self.published_update = state.global_update
        self.state = state
        self.faulted = False
        return self.parent

    def migrate_stage(self, parent_store: CheckpointStore, transition: StageTransition) -> None:
        """Explicit fresh-child migration with strict weight match and reset optimizer."""

        transition.validate()
        if self.state.global_update or self.parent is not None:
            raise ValueError("stage migration requires a fresh child")
        parent_state, payloads = parent_store.restore(transition.parent_checkpoint)
        sched = json.loads(payloads["scheduler.json"])
        parent_recipe = sched["metadata"].get("recipe")
        if (
            transition.child_recipe != digest(asdict(self.recipe))
            or transition.child_data != self.data_sha
            or parent_recipe is None
            or digest(parent_recipe) != transition.parent_recipe
            or parent_state.identities.model_spec_sha256 != self.v8_spec.sha256()
            or parent_state.identities.tokenizer_sha256 != self.codec_sha
        ):
            raise ValueError("stage transition identities or compatible architecture mismatch")
        weights = _load(payloads["model.bin"])
        if set(weights) != set(self.model.state_dict()):
            raise ValueError("migration tensor names differ")
        for name, tensor in weights.items():
            expected = self.model.state_dict()[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != expected.shape
                or tensor.dtype != expected.dtype
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"migration tensor mismatch for {name}")
        self.model.load_state_dict(weights, strict=True)
        self.metadata["stage_transition"] = asdict(transition)
        self.metadata = json.loads(canonical(self.metadata))
        self.state = replace(
            self.state,
            identities=replace(self.state.identities, run_spec_sha256=digest(self.metadata)),
        )


def create_v8_trainer(
    data: V8TrainingData,
    job: dict[str, object],
    *,
    source: str,
    root: Path,
    resume: bool = False,
    device: str = "cuda:0",
    cpu_canary: bool = False,
    v8_spec_override: V8ModelSpec | None = None,
    updates_override: int | None = None,
    transition: StageTransition | None = None,
    parent_store: CheckpointStore | None = None,
):
    """Instantiate the exact trainer for a registered V8 job (B0 or typed head)."""

    if not any(job == j for p in PHASES for j in protocol_jobs(p)):
        raise ValueError("unregistered V8 job")
    if data.dataset_name != job["dataset"]:
        raise ValueError(f"dataset mismatch: got {data.dataset_name!r}, expected {job['dataset']!r}")
    if device != "cuda:0" and not (device == "cpu" and cpu_canary is True):
        raise ValueError("production V8 runs require isolated CUDA device")
    if device != "cpu" and (cpu_canary or v8_spec_override is not None or updates_override is not None):
        raise ValueError("CPU canary overrides are disallowed on GPU")

    arm = str(job["arm"])
    v8_spec = v8_spec_override or build_v8_model_spec(arm)
    if arm == "B0":
        samples = data.byte_samples
    else:
        samples = data.typed_samples
    batch = min(int(job["batch_size"]), len(samples))
    micro = min(int(job["micro_batch_size"]), batch)
    epochs = int(job["epochs"])
    updates = max(1, math.ceil(len(samples) / batch) * max(epochs, 1)) if epochs > 0 else 1
    if updates_override is not None:
        updates = updates_override
    if device == "cpu":
        if (
            len(samples) > _MAX_CPU_SAMPLES
            or v8_spec.total_parameters() > _MAX_CPU_PARAMS
            or type(updates) is not int
            or not 1 <= updates <= _MAX_CPU_UPDATES
        ):
            raise ValueError("CPU mechanical canary exceeds tiny fixture bounds")

    recipe = Recipe(
        seed=int(job["init_seed"]),
        total_updates=updates,
        batch_size=batch,
        micro_batch_size=micro,
        lr=float(job["lr"]),
        clip=float(job["grad_clip"]),
        objective="ce",
        sampling="uniform",
        schedule=str(job["schedule"]),
        precision=str(job["precision"]),
    )
    configure_device(device)
    provenance = {
        **data.manifest,
        "v8_job": job,
        "v8_protocol_sha256": PROTOCOL_SHA,
        "v8_data_identity": data.data_identity,
        "v7_sampling_method": str(job["sampling"]),
    }

    if arm == "B0":
        trainer = IndependentOrderTrainer(
            data.byte_samples,
            spec=v8_spec.base_spec,
            recipe=recipe,
            source_commit=source,
            root=Path(root),
            lineage=str(job["job_id"]),
            data_provenance=provenance,
            tokenizer_manifest=UTF8.manifest.record(),
            device=device,
            method=str(job["sampling"]),
            order_seed=int(job["order_seed"]),
            resume=resume,
            transition=transition,
            parent_store=parent_store,
        )
        trainer.v8_spec = v8_spec
        return trainer

    return V8TypedTrainer(
        data.typed_samples,
        v8_spec=v8_spec,
        recipe=recipe,
        order_seed=int(job["order_seed"]),
        method=str(job["sampling"]),
        source_commit=source,
        root=Path(root),
        lineage=str(job["job_id"]),
        data_provenance=provenance,
        tokenizer_manifest=UTF8.manifest.record(),
        device=device,
        cpu_canary=cpu_canary,
        resume=resume,
        transition=transition,
        parent_store=parent_store,
    )


__all__ = [
    "V8TrainingData",
    "V8TypedSample",
    "V8TypedTrainer",
    "create_v8_trainer",
    "prepare_v8_training_data",
    "teacher_steps_to_byte_samples",
    "teacher_steps_to_typed_samples",
]
