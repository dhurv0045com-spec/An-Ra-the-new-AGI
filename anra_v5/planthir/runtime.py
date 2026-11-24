"""Generic self-contained ExperimentModule, Trainer, Batched Rollout, CampaignOrchestrator & Safe Exporter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.foundation_training import TokenSample
from anra_v5.elendril.publication import quarantine_interrupted_publication, writer_lock
from anra_v5.elendril.representation import SpecialToken, UTF8
from anra_v5.elendril.training import _dump, _load, batch_tensors, configure_device, runtime_identity
from v5_contracts.model_spec import ModelSpec
from v5_model.core import initialize, packed_layout
from v5_training.checkpoint import CheckpointStore
from v5_training.optimizer import build_adamw_optimizer, optimizer_group_receipt, validate_parameter_ownership
from v5_training.state import CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState, IdentityBindings, TrainingState

from .archives import (
    _stream_file_sha256,
    collect_minimal_checkpoint_files,
    export_session_archives,
    mirror_to_working_dir,
    prune_completed_job_artifacts,
    safe_extract_planthir_archive,
    verify_session_archive,
    win_long_path,
)
from .backend import BACKEND_CONTRACT_VERSION, BackendCapabilities, register_backend
from .contracts import EnvironmentKernel, ObservablePolicy, UniversalAction, UniversalObservation
from .custody import (
    CampaignLedgerVerifier,
    ConfirmLock,
    canonical_structural_hash,
    verify_campaign_ledger,
    verify_split_disjointness,
)
from .execution import dump_json_safe, sanitize_for_json
from .instrument import validate_byte_decoder_resilience
from .notebooks import build_pinned_kaggle_notebook, serialize_pinned_notebook, verify_pinned_notebook_bytes
from .reporting import (
    build_analyst_first_page_report,
    build_verification_matrix,
    render_campaign_dashboard,
    render_campaign_results_markdown,
    rescore_compact_analysis_archive,
)
from .resources import measure_free_disk_bytes
from .scheduler import IndependentWorkerScheduler, WorkerJobSpec
from .scoring import (
    TypedByteSequenceKernel,
    decode_generated_bytes,
    infer_public_byte_bound,
    normalize_typed_value,
    parse_typed_output_envelope,
    score_typed_output,
)


@dataclass(frozen=True, slots=True)
class GenericTrainRecipe:
    """Standardized optimizer/schedule recipe usable by any ExperimentModule backend."""

    seed: int = 7
    total_updates: int = 4
    batch_size: int = 8
    micro_batch_size: int = 4
    lr: float = 1e-3
    clip: float = 1.0
    objective: str = "ce"
    sampling: str = "uniform"
    schedule: str = "cosine"
    precision: str = "fp32"
    eq_weight: float = 0.0
    cf_weight: float = 0.0
    cf_margin: float = 0.5
    extra_config: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for fname in ("seed", "total_updates", "batch_size", "micro_batch_size"):
            if type(getattr(self, fname)) is not int:
                raise ValueError(f"GenericTrainRecipe.{fname} must be int")
        if not (0 <= self.seed < 2**32 and 1 <= self.total_updates <= 1_000_000):
            raise ValueError("GenericTrainRecipe seed/total_updates out of bounds")
        if not (1 <= self.micro_batch_size <= self.batch_size <= 4096):
            raise ValueError("GenericTrainRecipe micro_batch_size/batch_size out of bounds")
        if not math.isfinite(self.lr) or self.lr <= 0 or not math.isfinite(self.clip) or self.clip <= 0:
            raise ValueError("GenericTrainRecipe lr/clip must be finite and positive")
        if any(not math.isfinite(v) or v < 0 for v in (self.eq_weight, self.cf_weight, self.cf_margin)):
            raise ValueError("GenericTrainRecipe weights/margin must be finite and nonnegative")
        if self.schedule not in ("cosine", "constant"):
            raise ValueError(f"unsupported schedule: {self.schedule!r}")
        if str(self.precision).lower() not in ("fp32", "fp16", "amp", "float16", "bf16", "bfloat16"):
            raise ValueError(f"unsupported precision: {self.precision!r}")


@dataclass(frozen=True, slots=True)
class GenericModelSpec:
    """Content-addressed model specification container wrapping arbitrary architecture configs."""

    architecture: str
    model_config: Mapping[str, Any]
    parameter_count: int
    backend_id: str
    schema: str = "anra-planthir-generic-model-spec-v1"

    def canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "backend_id": self.backend_id,
            "architecture": self.architecture,
            "parameter_count": int(self.parameter_count),
            **{str(k): v for k, v in sorted(self.model_config.items())},
        }

    def sha256(self) -> str:
        return digest(["planthir-generic-model-spec-v1", self.canonical()])

    def total_parameters(self) -> int:
        return int(self.parameter_count)


@dataclass(frozen=True, slots=True)
class GenericDatasetBundle:
    """Dataset container accepted by GenericExperimentTrainer and ExperimentModule."""

    dataset_name: str
    items: tuple[Any, ...]
    data_identity: str
    manifest: dict[str, Any]
    split: str = "train"


def _compute_loss_from_fn(
    loss_fn: Callable[[nn.Module, Sequence[Any], GenericTrainRecipe], Any],
    model: nn.Module,
    chunk: Sequence[Any],
    recipe: GenericTrainRecipe,
    scale: float,
) -> tuple[torch.Tensor, float]:
    out = loss_fn(model, chunk, recipe)
    if isinstance(out, tuple):
        raw_loss = out[0]
    else:
        raw_loss = out
    if not isinstance(raw_loss, torch.Tensor):
        raise TypeError("loss_fn must return a torch.Tensor or (torch.Tensor, stats)")
    if not torch.isfinite(raw_loss):
        raise FloatingPointError("nonfinite loss tensor")
    scaled = raw_loss * scale
    return scaled, float(raw_loss.detach().item() * scale)


class GenericExperimentTrainer:
    """Reusable PyTorch trainer with CheckpointStore custody, microbatch normalization, and u0000 deduplication."""

    def __init__(
        self,
        *,
        backend_id: str,
        model: nn.Module,
        model_spec: GenericModelSpec | Any,
        recipe: GenericTrainRecipe | Any,
        loss_fn: Callable[[nn.Module, Sequence[Any], Any], Any],
        items: Sequence[Any],
        data_provenance: Mapping[str, Any],
        order_seed: int,
        source_commit: str,
        root: Path | str,
        lineage: str,
        device: str = "cpu",
        cpu_canary: bool = True,
        resume: bool = False,
        max_cpu_params: int = 100_000,
        max_cpu_samples: int = 32,
        max_cpu_updates: int = 8,
        post_step_hook: Callable[["GenericExperimentTrainer", dict[str, Any]], None] | None = None,
    ) -> None:
        if not items:
            raise ValueError("GenericExperimentTrainer requires non-empty training items")
        if hasattr(recipe, "validate"):
            recipe.validate()

        self.backend_id = str(backend_id)
        self.model_spec = model_spec
        self.recipe = recipe
        self.loss_fn = loss_fn
        self.items = tuple(items)
        self.order_seed = int(order_seed)
        self.post_step_hook = post_step_hook
        self.device = torch.device(device)
        param_count = sum(int(p.numel()) for p in model.parameters())

        if self.device.type == "cpu":
            if (
                not cpu_canary
                or len(self.items) > max_cpu_samples
                or param_count > max_cpu_params
                or int(recipe.total_updates) > max_cpu_updates
            ):
                raise ValueError(
                    f"CPU canary limits exceeded (params={param_count}, samples={len(self.items)}, updates={recipe.total_updates})"
                )

        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(int(recipe.seed))
        self.model = model.to(self.device)
        prec = str(getattr(recipe, "precision", "fp32")).lower()
        self.precision = prec
        self.amp_enabled = prec in ("fp16", "amp", "float16", "bf16", "bfloat16")
        self.amp_dtype = torch.bfloat16 if prec in ("bf16", "bfloat16") else torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(self.device.type == "cuda" and prec in ("fp16", "amp", "float16")),
        )
        self.optimizer = build_adamw_optimizer(self.model, lr=float(recipe.lr), torch_module=torch)

        # Build deterministic epoch-shuffled update plan
        self.plan: list[tuple[int, ...]] = []
        n_items = len(self.items)
        orders: dict[int, list[int]] = {}
        for u in range(int(recipe.total_updates)):
            chosen: list[int] = []
            for slot in range(int(recipe.batch_size)):
                pos = u * int(recipe.batch_size) + slot
                ep, off = divmod(pos, n_items)
                if ep not in orders:
                    perm = list(range(n_items))
                    random.Random(int(digest(["planthir-generic-order-v1", order_seed, ep]), 16)).shuffle(perm)
                    orders[ep] = perm
                chosen.append(orders[ep][off])
            self.plan.append(tuple(chosen))

        spec_canon = model_spec.canonical() if hasattr(model_spec, "canonical") else {"architecture": getattr(model_spec, "architecture", "custom"), "parameter_count": param_count}
        spec_sha = model_spec.sha256() if hasattr(model_spec, "sha256") else digest(spec_canon)
        self.data_sha = digest({"count": len(self.items), "provenance": dict(data_provenance)})
        self.runtime = runtime_identity(self.device)
        self.metadata = json.loads(
            canonical(
                {
                    "schema": "anra-planthir-generic-training-v1",
                    "backend_id": self.backend_id,
                    "architecture": str(spec_canon.get("architecture", "custom")),
                    "parameter_count": param_count,
                    "model_spec": spec_canon,
                    "model_spec_sha256": spec_sha,
                    "recipe": asdict(recipe) if hasattr(recipe, "__dataclass_fields__") else {},
                    "order_seed": self.order_seed,
                    "plan_sha256": digest(self.plan),
                    "provenance": dict(data_provenance),
                    "runtime": self.runtime,
                }
            )
        )
        identities = IdentityBindings(
            IDENTITY_SCHEMA,
            source_commit,
            spec_sha,
            UTF8.identity,
            self.data_sha,
            self.data_sha,
            digest(self.metadata),
            optimizer_group_receipt(self.model, self.optimizer)["sha256"],
            digest([recipe.schedule, float(recipe.lr), int(recipe.total_updates)]),
            digest([recipe.sampling, self.order_seed]),
            sampler_spec_sha256=digest(self.plan),
        )
        self.store = CheckpointStore(Path(root), lineage, payload_validator=self.validate_payloads)
        self.parent = self.store.latest_sha256()
        self.state = TrainingState.initial(
            lineage_id=lineage,
            token_budget=int(recipe.total_updates) * int(recipe.batch_size),
            tokens_per_update=int(recipe.batch_size),
            cursor=CursorState(CURSOR_SCHEMA, self.data_sha, 0, 0, 0),
            rng_state_sha256=sha256(self.rng_bytes()).hexdigest(),
            curriculum_phase=f"{self.backend_id}-{spec_canon.get('architecture', 'custom')}",
            identities=identities,
        )
        self.faulted = False
        self.published_update = 0

        if resume:
            if self.parent is None:
                raise ValueError("no committed checkpoint to resume")
            state, payloads = self.store.restore(self.parent)
            self.model.load_state_dict(_load(payloads["model.bin"]), strict=True)
            self.optimizer.load_state_dict(_load(payloads["optimizer.bin"]))
            if self.scaler.is_enabled():
                sched_doc = json.loads(payloads["scheduler.json"])
                if sched_doc.get("scaler"):
                    self.scaler.load_state_dict(sched_doc["scaler"])
            self.rng.set_state(torch.tensor(list(payloads["rng.bin"]), dtype=torch.uint8))
            self.state = replace(state)
            self.published_update = state.global_update
        elif self.parent is not None:
            raise ValueError("existing checkpoint lineage requires explicit resume=True")

    def rng_bytes(self) -> bytes:
        return bytes(self.rng.get_state().tolist())

    def learning_rate(self, update: int) -> float:
        if self.recipe.schedule == "constant":
            return float(self.recipe.lr)
        return float(
            self.recipe.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * update / max(1, int(self.recipe.total_updates)))))
        )

    def validate_payloads(self, state: TrainingState, payloads: dict[str, bytes]) -> None:
        if state.identities != self.state.identities:
            raise ValueError("checkpoint source/model/data/recipe identity mismatch")
        sched = json.loads(payloads["scheduler.json"])
        if sched.get("metadata") != self.metadata:
            raise ValueError("checkpoint metadata mismatch")
        if sha256(payloads["rng.bin"]).hexdigest() != state.rng_state_sha256:
            raise ValueError("checkpoint RNG digest mismatch")
        weights = _load(payloads["model.bin"])
        expected = self.model.state_dict()
        if set(weights) != set(expected):
            raise ValueError("checkpoint tensor names mismatch")
        for name, tensor in weights.items():
            ref = expected[name]
            if not isinstance(tensor, torch.Tensor) or tensor.shape != ref.shape or not torch.isfinite(tensor).all():
                raise ValueError(f"checkpoint tensor invalid: {name}")

    def step(
        self,
        *,
        post_step_hook: Callable[["GenericExperimentTrainer", dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if self.faulted or self.state.complete:
            raise ValueError("complete or faulted trainer requires fresh restore")
        self.faulted = True
        u = self.state.global_update
        lr = self.learning_rate(u)
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        chosen = [self.items[i] for i in self.plan[u]]
        micro = max(1, min(len(chosen), int(self.recipe.micro_batch_size)))
        acc_loss = 0.0
        amp_dev_type = "cuda" if self.device.type == "cuda" else "cpu"
        for start in range(0, len(chosen), micro):
            chunk = chosen[start : start + micro]
            scale = float(len(chunk)) / float(max(1, len(chosen)))
            with torch.amp.autocast(device_type=amp_dev_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                scaled_loss, scalar = _compute_loss_from_fn(self.loss_fn, self.model, chunk, self.recipe, scale)
            if self.scaler.is_enabled():
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            acc_loss += scalar

        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        validate_parameter_ownership(self.model, self.optimizer)
        amp_overflow_skipped = False
        if self.scaler.is_enabled():
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float(self.recipe.clip), error_if_nonfinite=False
            )
            norm_val = float(norm)
            if not math.isfinite(norm_val):
                amp_overflow_skipped = True
                norm_val = 0.0
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float(self.recipe.clip), error_if_nonfinite=True
            )
            norm_val = float(norm)
            self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise FloatingPointError("nonfinite parameter after optimizer step")

        step_meta: dict[str, Any] = {
            "update": u + 1,
            "loss": float(acc_loss),
            "pre_clip": float(norm_val),
            "lr": float(lr),
        }
        if self.amp_enabled:
            step_meta["amp_overflow_skipped"] = bool(amp_overflow_skipped)
        active_hook = post_step_hook if post_step_hook is not None else self.post_step_hook
        if active_hook is not None:
            active_hook(self, dict(step_meta))
            if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                raise FloatingPointError("nonfinite parameter after post_step_hook")

        step_tokens = len(chosen)
        self.state = replace(
            self.state,
            generation=u + 1,
            global_update=u + 1,
            cumulative_tokens=self.state.cumulative_tokens + step_tokens,
            tokens_by_source={self.backend_id: self.state.cumulative_tokens + step_tokens},
            optimizer_step_max=u + 1,
            schedule_tokens=self.state.cumulative_tokens + step_tokens,
            cursor=replace(self.state.cursor, sequence_ordinal=u + 1),
            rng_state_sha256=sha256(self.rng_bytes()).hexdigest(),
            parent_checkpoint_sha256=self.parent,
        )
        self.state.assert_valid()
        self.faulted = False
        return step_meta

    def save(self, *, inject_crash_at: str | None = None, prune_old: bool = True) -> str:
        if self.faulted or self.state.global_update <= self.published_update:
            raise ValueError("cannot publish faulted or repeated update")
        state = replace(self.state, parent_checkpoint_sha256=self.parent)
        payloads = {
            "model.bin": _dump({k: v.detach().cpu() for k, v in self.model.state_dict().items()}),
            "optimizer.bin": _dump(self.optimizer.state_dict()),
            "rng.bin": self.rng_bytes(),
            "scheduler.json": canonical(
                {"metadata": self.metadata, "scaler": (self.scaler.state_dict() if self.scaler.is_enabled() else {})}
            ),
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

    def train_loop(
        self,
        *,
        total_updates: int | None = None,
        checkpoint_every: int = 0,
        save_final: bool = True,
        post_step_hook: Callable[["GenericExperimentTrainer", dict[str, Any]], None] | None = None,
        post_train_hook: Callable[["GenericExperimentTrainer"], None] | None = None,
        prune_old: bool = True,
    ) -> dict[str, Any]:
        """Run bounded updates with optional periodic checkpoints, post-step/post-train hooks, and single final save."""

        target_u = int(self.recipe.total_updates if total_updates is None else total_updates)
        if target_u < 0 or target_u > int(self.recipe.total_updates):
            raise ValueError(
                f"total_updates={target_u} must lie in [0, recipe.total_updates={int(self.recipe.total_updates)}]"
            )
        ckpt_interval = max(0, int(checkpoint_every))
        step_logs: list[dict[str, Any]] = []
        while self.state.global_update < target_u:
            rec = self.step(post_step_hook=post_step_hook)
            step_logs.append(rec)
            cur_u = int(self.state.global_update)
            if ckpt_interval > 0 and cur_u % ckpt_interval == 0:
                if cur_u == target_u and (post_train_hook is not None or not save_final):
                    continue
                self.save(prune_old=prune_old)

        if post_train_hook is not None:
            post_train_hook(self)
            if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                raise FloatingPointError("nonfinite parameter after post_train_hook")

        if save_final and self.state.global_update > self.published_update:
            self.save(prune_old=prune_old)

        return {
            "updates_completed": int(self.state.global_update),
            "saved_final": bool(save_final and self.published_update == self.state.global_update),
            "latest_checkpoint_sha256": self.parent,
            "step_logs": step_logs,
        }

    def save_inference_milestone_weights(
        self,
        milestones_dir: Path | str,
        *,
        update: int,
        shared_backbone_dir: Path | str | None = None,
        backbone_prefix: str = "core.",
    ) -> Path:
        """Persist inference milestone weights with optional u0000 shared-backbone deduplication."""

        mdir = Path(milestones_dir)
        mdir.mkdir(parents=True, exist_ok=True)
        sd = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        out_dir = mdir / f"u{int(update):04d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if int(update) == 0 and shared_backbone_dir is not None and any(k.startswith(backbone_prefix) for k in sd):
            sb_dir = Path(shared_backbone_dir)
            sb_dir.mkdir(parents=True, exist_ok=True)
            core_keys = {k: v for k, v in sd.items() if k.startswith(backbone_prefix)}
            head_keys = {k: v for k, v in sd.items() if not k.startswith(backbone_prefix)}
            bb_bytes = _dump(core_keys)
            bb_sha = sha256(bb_bytes).hexdigest()
            bb_path = sb_dir / f"init_backbone_s{self.recipe.seed}_{bb_sha[:12]}.bin"
            if not bb_path.exists():
                tmp_bb = bb_path.with_suffix(f".{os.getpid()}.tmp")
                tmp_bb.write_bytes(bb_bytes)
                os.replace(tmp_bb, bb_path)
            head_bytes = _dump(head_keys)
            (out_dir / "head_weights.bin").write_bytes(head_bytes)
            meta = {
                "schema": "anra-planthir-inference-milestone-v1",
                "resume_capable": False,
                "global_update": 0,
                "shared_backbone_file": bb_path.name,
                "shared_backbone_sha256": bb_sha,
                "head_weights_file": "head_weights.bin",
                "head_weights_sha256": sha256(head_bytes).hexdigest(),
            }
            (out_dir / "milestone_weights_manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return out_dir

        w_bytes = _dump(sd)
        (out_dir / "model_inference.bin").write_bytes(w_bytes)
        meta = {
            "schema": "anra-planthir-inference-milestone-v1",
            "resume_capable": False,
            "global_update": int(update),
            "weights_file": "model_inference.bin",
            "weights_sha256": sha256(w_bytes).hexdigest(),
        }
        (out_dir / "milestone_weights_manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out_dir


def _state_is_terminal(state: Any) -> bool:
    if isinstance(state, Mapping):
        return bool(state.get("terminal", False))
    return bool(getattr(state, "terminal", False))


def batched_rollout(
    policy: ObservablePolicy,
    kernel: EnvironmentKernel,
    worlds: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 64,
    max_actions: int = 4,
    verify_serial_parity_n: int = 0,
    parity_atol: float = 1e-4,
) -> list[dict[str, Any]]:
    """Execute episodes in lockstep batches (`policy.propose_batch`), with optional serial parity verification."""

    if callable(getattr(policy, "reset_episode", None)):
        policy.reset_episode()  # type: ignore[attr-defined]
    states = [kernel.initial_state(w) for w in worlds]
    actions_per_world: list[list[tuple[str, str | None]]] = [[] for _ in worlds]
    batched_trace: list[list[tuple[str, str | None, float]]] = [[] for _ in worlds]
    done = [_state_is_terminal(st) for st in states]
    batch_fn = getattr(policy, "propose_batch", None)

    for _step in range(max_actions):
        active_indices = [i for i, d in enumerate(done) if not d]
        if not active_indices:
            break
        obs_list = [kernel.observe(states[i]) for i in active_indices]
        if callable(batch_fn):
            props = []
            for start in range(0, len(obs_list), max(1, int(batch_size))):
                props.extend(batch_fn(obs_list[start : start + max(1, int(batch_size))]))
        else:
            props = [policy.propose_with_telemetry(obs) for obs in obs_list]
        for idx, act in zip(active_indices, props):
            op_s = str(act.operation)
            actions_per_world[idx].append((op_s, act.argument))
            batched_trace[idx].append((op_s, act.argument, float(act.log_probability)))
            states[idx] = kernel.step(states[idx], act.operation, act.argument)
            done[idx] = _state_is_terminal(states[idx])

    if int(verify_serial_parity_n) > 0 and callable(batch_fn) and worlds:
        n_check = min(int(verify_serial_parity_n), len(worlds))
        for idx in range(n_check):
            w = worlds[idx]
            if callable(getattr(policy, "reset_episode", None)):
                policy.reset_episode()  # type: ignore[attr-defined]
            s_state = kernel.initial_state(w)
            s_trace: list[tuple[str, str | None, float]] = []
            for _step in range(max_actions):
                if _state_is_terminal(s_state):
                    break
                s_obs = kernel.observe(s_state)
                s_act = policy.propose_with_telemetry(s_obs)
                s_trace.append((str(s_act.operation), s_act.argument, float(s_act.log_probability)))
                s_state = kernel.step(s_state, s_act.operation, s_act.argument)
            b_trace = batched_trace[idx]
            if [(op, arg) for op, arg, _ in b_trace] != [(op, arg) for op, arg, _ in s_trace]:
                raise ValueError(
                    f"BATCHED_SERIAL_PARITY_MISMATCH:world={w.get('world_id', idx)}:actions_differ"
                )
            for step_i, ((_, _, b_lp), (_, _, s_lp)) in enumerate(zip(b_trace, s_trace)):
                if math.isfinite(b_lp) and math.isfinite(s_lp) and abs(b_lp - s_lp) > float(parity_atol):
                    raise ValueError(
                        f"BATCHED_SERIAL_PARITY_MISMATCH:world={w.get('world_id', idx)}:step={step_i}:delta={abs(b_lp - s_lp):.6g}>{parity_atol}"
                    )

    results: list[dict[str, Any]] = []
    for world, acts in zip(worlds, actions_per_world):
        ver = kernel.verify(world, acts)
        results.append(
            {
                "world_id": str(world.get("world_id", "")),
                "actions": acts,
                "correct": bool(ver.get("correct", False)),
                "verification": ver,
            }
        )
    return results


_AUTO_SPLIT_ID_KEYS = (
    "world_id",
    "task_id",
    "id",
    "base_world_id",
    "base_id",
    "parent_world_id",
    "parent_id",
    "counterfactual_mate_id",
    "mate_world_id",
    "eq_mate_id",
    "group_id",
    "cohort_id",
    "cluster_id",
    "lineage_group",
    "base_signature",
    "signature",
    "canonical_family_key",
    "family_id",
    "sig",
)
_DEFAULT_ACTION_LABELS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P")


def _auto_namespace_fallback_split(worlds: Sequence[Mapping[str, Any]], tag: str) -> tuple[dict[str, Any], ...]:
    """Rewrite cohort/world identifiers in synthesized fallback splits to prevent train/probe overlap."""
    out: list[dict[str, Any]] = []
    suffix = f"::__auto_{tag}"
    for idx, w in enumerate(worlds):
        if not isinstance(w, Mapping):
            out.append({"world_id": f"{tag}_{idx}{suffix}", "value": w})
            continue
        cloned = dict(w)
        has_any_id = False
        for key in _AUTO_SPLIT_ID_KEYS:
            val = cloned.get(key)
            if val not in (None, ""):
                cloned[key] = f"{val}{suffix}"
                has_any_id = True
        links = cloned.get("related_world_ids")
        if isinstance(links, Sequence) and not isinstance(links, (str, bytes)):
            cloned["related_world_ids"] = [f"{lv}{suffix}" if lv not in (None, "") else lv for lv in links]
        if not has_any_id:
            cloned["world_id"] = f"{tag}_{idx}{suffix}"
        out.append(cloned)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class _StaticPromptState:
    world_id: str
    public_prompt: str
    rendered_text: str
    available_tools: dict[str, bool]
    consumed_tools: frozenset[str]
    step: int = 0
    terminal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class StaticPromptKernel:
    """Default truth-firewalled EnvironmentKernel for static prompt->action or feature->label worlds."""

    kernel_id = "planthir-static-prompt-kernel-v1"

    def __init__(self, *, action_labels: Sequence[str] | None = None) -> None:
        self.action_labels = tuple(str(x) for x in action_labels) if action_labels is not None else _DEFAULT_ACTION_LABELS
        self._expected_by_wid: dict[str, tuple[str, str | None]] = {}

    @staticmethod
    def _world_id(world: Mapping[str, Any]) -> str:
        return str(world.get("world_id") or world.get("case_id") or world.get("task_id") or world.get("id") or "w0")

    def _extract_expected(self, world: Mapping[str, Any]) -> tuple[str, str | None]:
        hidden = world.get("hidden") if isinstance(world.get("hidden"), Mapping) else {}
        raw_op = hidden.get("expected_operation", world.get("expected_operation"))
        raw_arg = None
        for key in ("expected", "expected_answer", "target", "label"):
            if key in hidden and hidden[key] is not None:
                raw_arg = hidden[key]
                break
            if key in world and world[key] is not None:
                raw_arg = world[key]
                break
        if isinstance(raw_arg, bool):
            lbl = self.action_labels[1 if raw_arg else 0]
            return ("ANSWER", lbl)
        if isinstance(raw_arg, int) and 0 <= raw_arg < len(self.action_labels) and raw_op is None:
            raw_arg = self.action_labels[raw_arg]
        if raw_op is not None:
            op_s = str(raw_op)
            if op_s in ("QUERY", "RETRIEVE", "ABSTAIN"):
                return (op_s, None if op_s != "ANSWER" else (str(raw_arg) if raw_arg is not None else None))
            return (op_s, str(raw_arg) if raw_arg is not None else None)
        if raw_arg is None or str(raw_arg) in ("!", "ABSTAIN"):
            return ("ABSTAIN", None)
        if str(raw_arg) in ("Q", "QUERY"):
            return ("QUERY", None)
        if str(raw_arg) in ("R", "RETRIEVE"):
            return ("RETRIEVE", None)
        return ("ANSWER", str(raw_arg))

    def structural_signature(self, world: Mapping[str, Any]) -> str:
        return str(
            world.get("canonical_family_key")
            or world.get("sig")
            or world.get("family_id")
            or f"hops:{world.get('hops', 1)}"
        )

    def inject_distractor(self, world: Mapping[str, Any]) -> dict[str, Any]:
        w = dict(world)
        wid = self._world_id(world)
        w["world_id"] = f"{wid}::dist"
        if "sig" in w:
            w["sig"] = f"{w['sig']}::dist"
        if "public_prompt" in w and w["public_prompt"] is not None:
            w["public_prompt"] = f"{w['public_prompt']}\n# distractor=Z0"
        if "feat" in w and isinstance(w["feat"], list):
            w["feat"] = list(w["feat"])
        return w

    def permute_world(self, world: Mapping[str, Any], shift: int = 1) -> dict[str, Any]:
        w = dict(world)
        wid = self._world_id(world)
        w["world_id"] = f"{wid}::perm{shift}"
        if "sig" in w:
            w["sig"] = f"{w['sig']}::perm{shift}"
        return w

    def reverse_world(self, world: Mapping[str, Any]) -> dict[str, Any]:
        w = dict(world)
        wid = self._world_id(world)
        w["world_id"] = f"{wid}::rev"
        if "sig" in w:
            w["sig"] = f"{w['sig']}::rev"
        return w

    def initial_state(self, world: Mapping[str, Any]) -> _StaticPromptState:
        wid = self._world_id(world)
        self._expected_by_wid[wid] = self._extract_expected(world)
        if isinstance(world.get("available_tools"), Mapping):
            avail = {str(k): bool(v) for k, v in world["available_tools"].items()}
        else:
            avail = {
                "query": bool(world.get("query_available", False)),
                "memory": bool(world.get("memory_available", False)),
            }
        if world.get("public_prompt") is not None:
            public_prompt = str(world["public_prompt"])
            rendered = public_prompt
        elif world.get("rendered_text") is not None:
            public_prompt = str(world["rendered_text"])
            rendered = public_prompt
        elif world.get("prompt") is not None:
            public_prompt = str(world["prompt"])
            rendered = public_prompt
        else:
            sig = world.get("sig", wid)
            feat = world.get("feat")
            public_prompt = f"PROMPT sig={sig}" + (f" feat={json.dumps(list(feat))}" if feat is not None else "")
            rendered = public_prompt
        meta: dict[str, Any] = {"world_id": wid}
        if world.get("feat") is not None:
            meta["feat"] = list(world["feat"])
        if world.get("prompt_tokens") is not None:
            meta["prompt_tokens"] = [int(t) for t in world["prompt_tokens"]]
        return _StaticPromptState(
            world_id=wid,
            public_prompt=public_prompt,
            rendered_text=rendered,
            available_tools=avail,
            consumed_tools=frozenset(),
            step=0,
            terminal=False,
            metadata=meta,
        )

    def observe(self, state: _StaticPromptState) -> UniversalObservation:
        return UniversalObservation(
            obs_id=f"{state.world_id}:s{state.step}",
            rendered_text=state.rendered_text,
            available_tools=dict(state.available_tools),
            consumed_tools=state.consumed_tools,
            decision_index=int(state.step),
            budget_remaining=max(0, 1 - int(state.step)),
            metadata=dict(state.metadata),
        )

    def ideal_action(self, state: _StaticPromptState | Mapping[str, Any]) -> tuple[str, str | None]:
        if isinstance(state, _StaticPromptState):
            if state.world_id in self._expected_by_wid:
                return self._expected_by_wid[state.world_id]
            return ("ANSWER", self.action_labels[0])
        return self._extract_expected(state)

    def step(self, state: _StaticPromptState, operation: str, argument: str | None) -> _StaticPromptState:
        consumed = set(state.consumed_tools)
        avail = dict(state.available_tools)
        if operation == "QUERY":
            consumed.add("query")
            avail["query"] = False
        elif operation == "RETRIEVE":
            consumed.add("retrieve")
            avail["retrieve"] = False
        return _StaticPromptState(
            world_id=state.world_id,
            public_prompt=state.public_prompt,
            rendered_text=state.rendered_text,
            available_tools=avail,
            consumed_tools=frozenset(consumed),
            step=state.step + 1,
            terminal=True,
            metadata=dict(state.metadata),
        )

    def verify(self, world: Mapping[str, Any], actions: Sequence[tuple[str, str | None]]) -> dict[str, Any]:
        expected = self._extract_expected(world)
        act = actions[0] if actions else ("INVALID", None)
        return {"correct": bool(act == expected), "terminal_status": "answered", "expected": expected, "observed": act}

    def decompose_subgoals(self, world: Mapping[str, Any]) -> tuple[Any, ...]:
        return ()


class LogitsPolicy:
    """Default ObservablePolicy wrapping any nn.Module returning logits or `(logits, hidden)`."""

    policy_id = "planthir-logits-policy-v1"

    def __init__(
        self,
        model: nn.Module,
        model_spec: Any = None,
        *,
        model_identity: str = "planthir-logits-policy",
        mode: str = "raw",
        copy_enabled_override: bool | None = None,
        action_labels: Sequence[str] | None = None,
        input_encoder: Callable[[UniversalObservation], Any] | None = None,
    ) -> None:
        self.model = model
        self.model_spec = model_spec
        self.model_identity = str(model_identity)
        self.mode = str(mode)
        self.copy_enabled_override = copy_enabled_override
        self.action_labels = tuple(str(x) for x in action_labels) if action_labels is not None else None
        self.input_encoder = input_encoder

    def _encode_obs(self, obs: UniversalObservation) -> torch.Tensor:
        if self.input_encoder is not None:
            val = self.input_encoder(obs)
            return val if isinstance(val, torch.Tensor) else torch.tensor(val, dtype=torch.float32)
        if "feat" in obs.metadata and obs.metadata["feat"] is not None:
            return torch.tensor(obs.metadata["feat"], dtype=torch.float32)
        if "feat=" in obs.rendered_text:
            raw = obs.rendered_text.split("feat=", 1)[1].splitlines()[0].strip()
            if "#" in raw:
                raw = raw.split("#", 1)[0].strip()
            return torch.tensor(json.loads(raw), dtype=torch.float32)
        if "prompt_tokens" in obs.metadata and obs.metadata["prompt_tokens"] is not None:
            return torch.tensor(obs.metadata["prompt_tokens"], dtype=torch.int64)
        return torch.tensor((UTF8.bos_id, *UTF8.encode(obs.rendered_text)), dtype=torch.int64)

    def _forward_logits_and_hidden(self, obs_batch: Sequence[UniversalObservation]) -> tuple[torch.Tensor, torch.Tensor]:
        first_param = next(self.model.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        encoded = [self._encode_obs(o) for o in obs_batch]
        if encoded and encoded[0].ndim == 1 and encoded[0].dtype in (torch.int32, torch.int64):
            max_len = max(int(t.shape[0]) for t in encoded)
            padded = [
                F.pad(t, (0, max_len - int(t.shape[0])), value=UTF8.pad_id) if int(t.shape[0]) < max_len else t
                for t in encoded
            ]
            x = torch.stack(padded, dim=0).to(device)
        else:
            x = torch.stack(encoded, dim=0).to(device)

        was_training = bool(self.model.training)
        self.model.eval()
        try:
            with torch.no_grad():
                out = self.model(x)
        finally:
            self.model.train(was_training)

        if isinstance(out, tuple) and len(out) >= 2:
            logits, hidden = out[0], out[1]
        elif isinstance(out, Mapping):
            logits = out["logits"]
            hidden = out.get("hidden", logits)
        else:
            logits = out
            hidden = out
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        if hidden.ndim == 3:
            hidden = hidden[:, -1, :]
        return logits, hidden

    def _labels_for_dim(self, dim: int) -> tuple[str, ...]:
        if self.action_labels is not None and len(self.action_labels) >= dim:
            return tuple(self.action_labels[:dim])
        if dim <= len(_DEFAULT_ACTION_LABELS):
            return _DEFAULT_ACTION_LABELS[:dim]
        return tuple(f"C{i}" for i in range(dim))

    @staticmethod
    def _decode_label(label: str) -> tuple[str, str | None]:
        if label in ("Q", "QUERY"):
            return ("QUERY", None)
        if label in ("R", "RETRIEVE"):
            return ("RETRIEVE", None)
        if label in ("!", "ABSTAIN"):
            return ("ABSTAIN", None)
        return ("ANSWER", label)

    def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction:
        return self.propose_batch([obs])[0]

    def propose_batch(self, obs_batch: Sequence[UniversalObservation]) -> list[UniversalAction]:
        logits, _ = self._forward_logits_and_hidden(obs_batch)
        probs = F.softmax(logits.float(), dim=-1)
        labels = self._labels_for_dim(int(logits.shape[-1]))
        actions: list[UniversalAction] = []
        for row in probs:
            idx = int(torch.argmax(row).item())
            lbl = labels[idx]
            op, arg = self._decode_label(lbl)
            lp = float(torch.log(row[idx].clamp_min(1e-8)).item())
            dist = {
                f"{self._decode_label(l)[0]}:{self._decode_label(l)[1] or l}": float(row[j].item())
                for j, l in enumerate(labels)
            }
            actions.append(
                UniversalAction(
                    operation=op,
                    argument=arg,
                    log_probability=lp,
                    action_distribution=dist,
                    mode=self.mode,
                    raw_operation=op,
                    raw_argument=arg,
                )
            )
        return actions

    def reset_episode(self) -> None:
        """Reset per-episode model/policy state if the underlying module tracks episodic buffers."""
        if callable(getattr(self.model, "reset_episode", None)):
            self.model.reset_episode()  # type: ignore[attr-defined]

    def extract_representations(self, obs_batch: Sequence[UniversalObservation]) -> dict[str, torch.Tensor]:
        _, hidden = self._forward_logits_and_hidden(obs_batch)
        return {"decision_hidden": hidden.detach().cpu()}


@dataclass(frozen=True)
class BytePromptSample(TokenSample):
    """UTF-8 framed byte-token sample with optional counterfactual endpoint IDs."""

    family_id: str = ""
    cell_key: str = ""
    gold_token_id: int = 0
    rival_token_id: int = 0


def frame_byte_prompt_sample(
    prompt: str,
    answer: str,
    *,
    task_id: str = "sample0",
    world_id: str = "w0",
    category: str = "route",
    family_id: str = "",
    cell_key: str = "",
    rival_answer: str | None = None,
    max_length: int | None = None,
    pad_to: int | None = None,
    truncate_prompt: bool = False,
) -> BytePromptSample:
    """Frame `(prompt, answer)` via `UTF8.frame` and return a `TokenSample`-compatible `BytePromptSample`."""
    framed = UTF8.frame(
        prompt,
        answer,
        max_length=max_length,
        pad_to=pad_to,
        truncate_prompt=truncate_prompt,
    )
    gold_id = int(UTF8.encode(answer)[0]) if answer else int(UTF8.eos_id)
    rival_id = int(UTF8.encode(rival_answer)[0]) if rival_answer else int(UTF8.eos_id)
    return BytePromptSample(
        task_id=str(task_id),
        tokens=framed.ids,
        eligible=framed.loss_mask,
        world_id=str(world_id),
        category=str(category),
        family_id=str(family_id),
        cell_key=str(cell_key),
        gold_token_id=gold_id,
        rival_token_id=rival_id,
    )


def compute_ce_and_pair_margin_terms(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    loss_weights: torch.Tensor,
    chunk: Sequence[Any],
    *,
    margin: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Compute eligible-token cross-entropy sum and counterfactual pair-margin relu sum."""
    ce_losses = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        tokens[:, 1:].reshape(-1),
        reduction="none",
    ).reshape_as(tokens[:, 1:])
    ce_sum = (ce_losses * loss_weights[:, 1:]).sum()
    eligible_tokens = int(loss_weights[:, 1:].sum().item())

    margin_terms: list[torch.Tensor] = []
    for row_idx, sample in enumerate(chunk):
        gold_id = int(getattr(sample, "gold_token_id", 0) or 0)
        rival_id = int(getattr(sample, "rival_token_id", 0) or 0)
        elig = getattr(sample, "eligible", ())
        if gold_id <= 0 or rival_id <= 0 or gold_id == rival_id or not any(elig):
            continue
        first_pos = elig.index(True)
        if first_pos < 1:
            continue
        step_logits = logits[row_idx, first_pos - 1].float()
        s_i = step_logits[gold_id] - step_logits[rival_id]
        margin_terms.append(F.relu(float(margin) - s_i))
    margin_sum = torch.stack(margin_terms).sum() if margin_terms else ce_sum.new_tensor(0.0)
    return ce_sum, margin_sum, eligible_tokens, len(margin_terms)


def audit_pair_margin_gradient_sign_reversal() -> dict[str, Any]:
    """Negative control: swapping gold and rival pair labels reverses the auxiliary margin gradient."""
    logits = torch.zeros((1, 4, UTF8.vocabulary_size), dtype=torch.float32, requires_grad=True)
    tokens = torch.tensor([[UTF8.bos_id, 69, 70, UTF8.eos_id]], dtype=torch.int64)
    weights = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)

    s_orig = BytePromptSample(
        task_id="neg_orig",
        tokens=(UTF8.bos_id, 69, 70, UTF8.eos_id),
        eligible=(False, False, True, True),
        world_id="fam0",
        gold_token_id=69,
        rival_token_id=70,
    )
    s_swap = replace(s_orig, task_id="neg_swap", gold_token_id=70, rival_token_id=69)

    _, m_orig, _, n1 = compute_ce_and_pair_margin_terms(logits, tokens, weights, [s_orig], margin=1.0)
    (g_orig,) = torch.autograd.grad(m_orig, logits)
    v_orig = g_orig[0, 1].detach().clone()

    _, m_swap, _, n2 = compute_ce_and_pair_margin_terms(logits, tokens, weights, [s_swap], margin=1.0)
    (g_swap,) = torch.autograd.grad(m_swap, logits)
    v_swap = g_swap[0, 1].detach().clone()

    cos = float(F.cosine_similarity(v_orig.unsqueeze(0), v_swap.unsqueeze(0)).item())
    passed = n1 == 1 and n2 == 1 and float(v_orig[69]) < 0.0 and float(v_swap[69]) > 0.0 and cos <= -0.999
    return {
        "schema": "anra-planthir-pair-margin-grad-control-v1",
        "passed": bool(passed),
        "orig_gold_grad": float(v_orig[69]),
        "swap_gold_grad": float(v_swap[69]),
        "gradient_cosine": cos,
    }


class BytePromptTrainer(GenericExperimentTrainer):
    """Built-in byte-token trainer for `v5_contracts.model_spec.ModelSpec` / `Core` with CheckpointCatalog custody."""

    def __init__(
        self,
        samples: Sequence[TokenSample],
        *,
        spec: ModelSpec,
        recipe: GenericTrainRecipe | None = None,
        source_commit: str,
        root: Path | str,
        lineage: str,
        data_provenance: Mapping[str, Any],
        order_seed: int | None = None,
        device: str = "cpu",
        cpu_canary: bool = True,
        resume: bool = False,
        max_cpu_params: int = 100_000,
        max_cpu_samples: int = 32,
        max_cpu_updates: int = 8,
        post_step_hook: Callable[["GenericExperimentTrainer", dict[str, Any]], None] | None = None,
    ) -> None:
        spec.assert_valid()
        rcp = recipe if recipe is not None else GenericTrainRecipe()
        configure_device(device)
        model = initialize(spec, int(rcp.seed))

        def _byte_loss(m: nn.Module, chunk: Sequence[TokenSample], cur_recipe: GenericTrainRecipe) -> torch.Tensor:
            dev = next(m.parameters()).device
            toks, segs, elig = batch_tensors([(s.tokens, s.eligible) for s in chunk], device=dev)
            pos, mask = packed_layout(segs, torch_module=torch)
            logits = m(toks, pos, mask)
            ce_sum, margin_sum, elig_cnt, m_cnt = compute_ce_and_pair_margin_terms(
                logits,
                toks,
                elig.float(),
                chunk,
                margin=float(getattr(cur_recipe, "cf_margin", 1.0)),
            )
            ce_loss = ce_sum / float(max(1, elig_cnt))
            cf_w = float(getattr(cur_recipe, "cf_weight", 0.0))
            if (getattr(cur_recipe, "objective", "ce") in ("ce_pair_margin", "pair_margin") or cf_w > 0.0) and m_cnt > 0:
                return ce_loss + cf_w * (margin_sum / float(m_cnt))
            return ce_loss

        super().__init__(
            backend_id="byte_prompt",
            model=model,
            model_spec=spec,
            recipe=rcp,
            loss_fn=_byte_loss,
            items=samples,
            data_provenance=data_provenance,
            order_seed=(int(rcp.seed) if order_seed is None else int(order_seed)),
            source_commit=source_commit,
            root=root,
            lineage=lineage,
            device=device,
            cpu_canary=cpu_canary,
            resume=resume,
            max_cpu_params=max_cpu_params,
            max_cpu_samples=max_cpu_samples,
            max_cpu_updates=max_cpu_updates,
            post_step_hook=post_step_hook,
        )


class ExperimentModule:
    """Single-file PlanthirBackend plugin turning core scientific hooks into a full CLI & Gate backend."""

    contract_version = BACKEND_CONTRACT_VERSION

    def __init__(
        self,
        *,
        backend_id: str,
        backend_version: str = "1.0",
        program: str | None = None,
        architectures: Sequence[str] = ("baseline", "treatment"),
        objectives: Sequence[str] = ("ce",),
        model_factory: Callable[[str, Mapping[str, Any], int], nn.Module],
        dataset_factory: Callable[[Mapping[str, Any], bool], Mapping[str, Any]],
        loss_fn: Callable[[nn.Module, Sequence[Any], Any], Any],
        kernel_factory: Callable[[], EnvironmentKernel] | None = None,
        policy_factory: Callable[..., ObservablePolicy] | None = None,
        action_labels: Sequence[str] | None = None,
        input_encoder: Callable[[UniversalObservation], Any] | None = None,
        default_dataset_config: Mapping[str, Any] | None = None,
        default_model_config: Mapping[str, Any] | None = None,
        treatment_architecture: str | None = None,
        treatment_model_config: Mapping[str, Any] | None = None,
        extra_model_keys: Sequence[str] = (),
        extra_dataset_keys: Sequence[str] = (),
        extra_recipe_keys: Sequence[str] = (),
        extra_arch_head_keys: Sequence[str] = (),
        supports_grouped_loss: bool = False,
        supports_fp16: bool = False,
    ) -> None:
        self.backend_id = str(backend_id)
        self.backend_version = str(backend_version)
        self.program = str(program or backend_id.upper())
        self.architectures = tuple(str(a) for a in architectures)
        self.objectives = tuple(str(o) for o in objectives)
        self.model_factory = model_factory
        self.dataset_factory = dataset_factory
        self.loss_fn = loss_fn
        self.action_labels = tuple(str(x) for x in action_labels) if action_labels is not None else None
        self.input_encoder = input_encoder
        if kernel_factory is not None:
            self.kernel_factory = kernel_factory
        else:
            lbls = self.action_labels
            self.kernel_factory = lambda: StaticPromptKernel(action_labels=lbls)
        if policy_factory is not None:
            self.policy_factory = policy_factory
        else:
            lbls = self.action_labels
            enc = self.input_encoder
            self.policy_factory = (
                lambda model, model_spec, *, model_identity="planthir-logits-policy", mode="raw", copy_enabled_override=None: LogitsPolicy(
                    model,
                    model_spec,
                    model_identity=model_identity,
                    mode=mode,
                    copy_enabled_override=copy_enabled_override,
                    action_labels=lbls,
                    input_encoder=enc,
                )
            )
        self.default_dataset_config = dict(default_dataset_config or {"dataset_name": "default", "num_samples": 16})
        self.default_model_config = dict(default_model_config or {"width": 16})
        self.treatment_architecture = str(treatment_architecture or (self.architectures[1] if len(self.architectures) > 1 else self.architectures[0]))
        self.treatment_model_config = dict(treatment_model_config or self.default_model_config)
        self.extra_model_keys = tuple(sorted({str(k) for k in extra_model_keys} | set(self.default_model_config) | set(self.treatment_model_config)))
        self.extra_dataset_keys = tuple(sorted({str(k) for k in extra_dataset_keys} | set(self.default_dataset_config)))
        self.extra_recipe_keys = tuple(str(k) for k in extra_recipe_keys)
        self.extra_arch_head_keys = tuple(str(k) for k in extra_arch_head_keys)
        self.supports_grouped_loss = bool(supports_grouped_loss)
        self.supports_fp16 = bool(supports_fp16)

    def register(self) -> "ExperimentModule":
        register_backend(self)
        return self

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            backend_version=self.backend_version,
            contract_version=self.contract_version,
            supported_architectures=self.architectures,
            supported_objectives=self.objectives,
            supported_samplings=("uniform",),
            supported_devices=("cpu", "cuda"),
            supports_grouped_loss=self.supports_grouped_loss,
            supports_resume=True,
            supports_fp16=self.supports_fp16,
        )

    def json_schema_extensions(self) -> dict[str, tuple[str, ...]]:
        return {
            "dataset": self.extra_dataset_keys,
            "model": self.extra_model_keys,
            "recipe": self.extra_recipe_keys,
            "execution": (),
        }

    def capacity_match_keys(self) -> tuple[str, ...]:
        return self.extra_arch_head_keys or tuple(
            k for k in self.extra_model_keys if k not in ("width", "hidden_dim", "layers", "num_layers", "num_heads", "query_heads")
        )

    def create_kernel(self) -> EnvironmentKernel:
        kernel = self.kernel_factory()
        if not callable(getattr(kernel, "ideal_action", None)):
            try:
                setattr(kernel, "ideal_action", lambda _state: ("ABSTAIN", None))
            except Exception:
                pass
        if not callable(getattr(kernel, "decompose_subgoals", None)):
            try:
                setattr(kernel, "decompose_subgoals", lambda _world: ())
            except Exception:
                pass
        return kernel

    def default_template(self, *, canary: bool = True) -> dict[str, Any]:
        ctrl_arch = self.architectures[0]
        trt_arch = self.treatment_architecture
        obj = self.objectives[0]
        same_arch = ctrl_arch == trt_arch
        return {
            "schema": "anra-planthir-spec-v1",
            "campaign_id": f"{self.backend_id.upper()}-{'CANARY' if canary else 'CAMPAIGN'}",
            "program": self.program,
            "source_commit": "HEAD",
            "backend_id": self.backend_id,
            "dataset": dict(self.default_dataset_config),
            "control_arm": {
                "arm_id": "CTRL",
                "architecture": ctrl_arch,
                "model": dict(self.default_model_config),
                "recipe": {
                    "seed": 7,
                    "total_updates": 2,
                    "batch_size": 4,
                    "micro_batch_size": 2,
                    "lr": 1e-3,
                    "objective": obj,
                    "sampling": "uniform",
                    "schedule": "cosine",
                    "precision": "fp32",
                },
                "milestone_updates": [1, 2],
                "order_seed": 7,
            },
            "treatment_arms": [
                {
                    "arm_id": "TRT",
                    "architecture": trt_arch,
                    "model": dict(self.treatment_model_config),
                    "recipe": {
                        "seed": 7,
                        "total_updates": 2,
                        "batch_size": 4,
                        "micro_batch_size": 2,
                        "lr": 2e-3 if same_arch else 1e-3,
                        "objective": obj,
                        "sampling": "uniform",
                        "schedule": "cosine",
                        "precision": "fp32",
                    },
                    "milestone_updates": [1, 2],
                    "order_seed": 7,
                }
            ],
            "execution": {
                "device": "cpu",
                "cpu_canary": bool(canary),
                "checkpoint_interval": 1,
                "parameter_tolerance_ratio": 2.0,
            },
        }

    def build_arm_and_data(self, doc: Mapping[str, Any], *, cpu_canary: bool) -> dict[str, Any]:
        from .cartridge import CartridgeArmSpec
        from .contracts import CausalContrastSpec, FalsifierSpec, HypothesisSpec

        ds_bundle = self.dataset_factory(dict(doc.get("dataset", {})), bool(cpu_canary))
        train_data = ds_bundle["train_data"]
        train_worlds = tuple(ds_bundle["train_worlds"])
        eval_worlds = tuple(ds_bundle["eval_worlds"])
        mh_worlds = (
            tuple(ds_bundle["multihop_worlds"])
            if "multihop_worlds" in ds_bundle
            else _auto_namespace_fallback_split(eval_worlds[:2], "multihop")
        )
        probe_worlds = (
            tuple(ds_bundle["probe_train_worlds"])
            if "probe_train_worlds" in ds_bundle
            else _auto_namespace_fallback_split(train_worlds[:4], "probe_train")
        )
        dataset_identity = str(getattr(train_data, "data_identity", digest(str(len(train_worlds)))))

        def _compile_arm(arm_doc: Mapping[str, Any], is_ctrl: bool) -> CartridgeArmSpec:
            arch = str(arm_doc["architecture"])
            m_cfg = dict(arm_doc.get("model", {}))
            r_doc = dict(arm_doc.get("recipe", {}))
            seed = int(r_doc.get("seed", 7))
            probe_mod = self.model_factory(arch, m_cfg, seed)
            pcount = sum(int(p.numel()) for p in probe_mod.parameters())
            del probe_mod
            mspec = GenericModelSpec(
                architecture=arch,
                model_config=m_cfg,
                parameter_count=pcount,
                backend_id=self.backend_id,
            )
            recipe = GenericTrainRecipe(
                seed=seed,
                total_updates=int(r_doc.get("total_updates", 2)),
                batch_size=int(r_doc.get("batch_size", 4)),
                micro_batch_size=int(r_doc.get("micro_batch_size", 2)),
                lr=float(r_doc.get("lr", 1e-3)),
                clip=float(r_doc.get("clip", 1.0)),
                objective=str(r_doc.get("objective", self.objectives[0])),
                sampling=str(r_doc.get("sampling", "uniform")),
                schedule=str(r_doc.get("schedule", "cosine")),
                precision=str(r_doc.get("precision", "fp32")),
                eq_weight=float(r_doc.get("eq_weight", 0.0)),
                cf_weight=float(r_doc.get("cf_weight", 0.0)),
                cf_margin=float(r_doc.get("cf_margin", 0.5)),
                extra_config=dict(r_doc.get("extra_config", {})),
            )
            milestones = tuple(int(u) for u in arm_doc.get("milestone_updates", []) or (recipe.total_updates,))
            cons = tuple(
                CausalContrastSpec(
                    contrast_id=str(c["contrast_id"]),
                    factor_name=str(c.get("factor_name", c.get("factor", ""))),
                    control_arm=str(c.get("control_arm", c.get("control_arm_id", ""))),
                    treatment_arm=str(c.get("treatment_arm", c.get("treatment_arm_id", ""))),
                    held_constant=tuple(str(x) for x in c.get("held_constant", ())),
                    expected_mechanism_signature=str(c.get("expected_mechanism_signature", c.get("target_mechanism", ""))),
                )
                for c in arm_doc.get("contrasts", ())
            )
            hyps = tuple(
                HypothesisSpec(
                    hypothesis_id=str(h["hypothesis_id"]),
                    statement=str(h["statement"]),
                    mechanism_target=str(h["mechanism_target"]),
                    falsifiers=tuple(
                        FalsifierSpec(
                            falsifier_id=str(f["falsifier_id"]),
                            description=str(f["description"]),
                            metric_name=str(f["metric_name"]),
                            comparator=str(f["comparator"]),
                            threshold=float(f["threshold"]),
                        )
                        for f in h.get("falsifiers", ())
                    ),
                )
                for h in arm_doc.get("hypotheses", ())
            )
            return CartridgeArmSpec(
                arm_id=str(arm_doc["arm_id"]),
                architecture_family=arch,
                is_control=is_ctrl,
                v8_spec=None,
                recipe=recipe,
                dataset_identity=dataset_identity,
                milestone_updates=milestones,
                policy_mode=str(arm_doc.get("policy_mode", "raw")),
                order_seed=int(arm_doc.get("order_seed", 7)),
                hypotheses=hyps,
                contrasts=cons,
                backend_id=self.backend_id,
                model_spec=mspec,
            )

        ctrl = _compile_arm(doc["control_arm"], True)
        trts = [_compile_arm(t, False) for t in doc.get("treatment_arms", ())]
        return {
            "train_data": train_data,
            "train_worlds": train_worlds,
            "eval_worlds": eval_worlds,
            "multihop_worlds": mh_worlds,
            "probe_train_worlds": probe_worlds,
            "kernel": self.create_kernel(),
            "control_arm": ctrl,
            "treatment_arms": trts,
        }

    def resolve_model_spec(self, arm: Any) -> GenericModelSpec:
        spec = getattr(arm, "model_spec", None)
        if not isinstance(spec, GenericModelSpec):
            raise ValueError(f"backend {self.backend_id!r} requires GenericModelSpec")
        return spec

    def canonical_model_spec(self, model_spec: Any) -> dict[str, Any]:
        return self.resolve_model_spec_obj(model_spec).canonical()

    def resolve_model_spec_obj(self, model_spec: Any) -> GenericModelSpec:
        if not isinstance(model_spec, GenericModelSpec):
            raise ValueError("expected GenericModelSpec")
        return model_spec

    def model_parameter_count(self, model_spec: Any) -> int:
        return self.resolve_model_spec_obj(model_spec).total_parameters()

    def canonical_factor_config(self, arm: Any) -> dict[str, Any]:
        return self.resolve_model_spec(arm).canonical()

    def validate_kernel(self, kernel: EnvironmentKernel) -> None:
        if not isinstance(kernel, EnvironmentKernel):
            raise ValueError("kernel does not satisfy EnvironmentKernel")

    def validate_arm(self, arm: Any, train_data: Any, *, device: str, cpu_canary: bool) -> dict[str, Any]:
        caps = self.capabilities()
        if getattr(arm, "trainer_factory", None) is not None:
            raise ValueError("custom trainer_factory unsupported")
        spec = self.resolve_model_spec(arm)
        if spec.architecture not in caps.supported_architectures:
            raise ValueError(f"unsupported architecture {spec.architecture!r}")
        arm.recipe.validate()
        items = train_data.items if hasattr(train_data, "items") else tuple(train_data)
        if torch.device(device).type == "cpu" and (
            not cpu_canary
            or len(items) > caps.max_cpu_canary_samples
            or spec.total_parameters() > caps.max_cpu_canary_params
            or int(arm.recipe.total_updates) > caps.max_cpu_canary_updates
        ):
            raise ValueError("CPU canary limits exceeded")
        return {"backend_id": self.backend_id, "architecture": spec.architecture, "parameter_count": spec.total_parameters()}

    def validate_dataset_and_worlds(self, train_data: Any, train_worlds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        ident = str(getattr(train_data, "data_identity", "custom"))
        return {
            "dataset_name": str(getattr(train_data, "dataset_name", "custom")),
            "data_identity": ident,
            "content_sha256": ident,
            "world_count": len(train_worlds),
            "resolved_train_worlds": tuple(train_worlds),
        }

    def create_trainer(
        self,
        arm: Any,
        train_data: Any,
        *,
        source_commit: str,
        root: Path,
        lineage: str,
        device: str,
        cpu_canary: bool,
        resume: bool,
    ) -> GenericExperimentTrainer:
        spec = self.resolve_model_spec(arm)
        model = self.model_factory(spec.architecture, spec.model_config, int(arm.recipe.seed))
        items = train_data.items if hasattr(train_data, "items") else tuple(train_data)
        prov = dict(getattr(train_data, "manifest", {"data_identity": getattr(train_data, "data_identity", "custom")}))
        arm_resume = resume and (Path(root) / lineage / "LATEST").is_file()
        return GenericExperimentTrainer(
            backend_id=self.backend_id,
            model=model,
            model_spec=spec,
            recipe=arm.recipe,
            loss_fn=self.loss_fn,
            items=items,
            data_provenance=prov,
            order_seed=int(arm.order_seed),
            source_commit=source_commit,
            root=root,
            lineage=lineage,
            device=device,
            cpu_canary=cpu_canary,
            resume=arm_resume,
        )

    def sample_gradient_loss_fn(
        self,
        trainer: GenericExperimentTrainer,
        arm: Any,
        train_data: Any,
        device: torch.device,
    ) -> Callable[[nn.Module], torch.Tensor]:
        items = train_data.items if hasattr(train_data, "items") else tuple(train_data)
        sample_chunk = items[: min(4, len(items))]

        def _loss(m: nn.Module) -> torch.Tensor:
            out = self.loss_fn(m, sample_chunk, arm.recipe)
            return out[0] if isinstance(out, tuple) else out

        return _loss

    def create_policy(
        self,
        model: Any,
        model_spec: Any,
        *,
        model_identity: str,
        mode: str = "raw",
        copy_enabled_override: bool | None = None,
    ) -> ObservablePolicy:
        pol = self.policy_factory(
            model,
            model_spec,
            model_identity=model_identity,
            mode=mode,
            copy_enabled_override=copy_enabled_override,
        )
        if not hasattr(pol, "policy_id"):
            try:
                setattr(pol, "policy_id", f"{self.backend_id}:{model_identity}:{mode}")
            except Exception:
                pass
        return pol

    def extract_checkpoint_metadata(self, meta: Mapping[str, Any]) -> tuple[str, str, int | None, Any, None]:
        raw_ms = meta.get("model_spec", {})
        arch = str(meta.get("architecture", raw_ms.get("architecture", "custom")))
        pcount = int(meta.get("parameter_count", raw_ms.get("parameter_count", 0)))
        cfg = {k: v for k, v in raw_ms.items() if k not in ("schema", "backend_id", "architecture", "parameter_count")}
        return self.program, arch, pcount, GenericModelSpec(arch, cfg, pcount, self.backend_id), None

    def restore_checkpoint_bundle(
        self,
        *,
        state: TrainingState,
        sched: Mapping[str, Any],
        weights: Mapping[str, Any],
        sha: str,
        mode: str,
        device: str,
        copy_enabled_override: bool | None,
    ) -> dict[str, Any]:
        meta = sched.get("metadata", sched)
        prog, arch, pcount, mspec, _ = self.extract_checkpoint_metadata(meta)
        seed = int(meta.get("recipe", {}).get("seed", 0))
        model = self.model_factory(mspec.architecture, mspec.model_config, seed).to(torch.device(device))
        model.load_state_dict(weights, strict=True)
        model.eval()
        policy = self.create_policy(
            model,
            mspec,
            model_identity=f"{state.lineage_id}:{sha[:12]}",
            mode=mode,
            copy_enabled_override=copy_enabled_override,
        )
        return {
            "checkpoint_sha256": sha,
            "lineage_id": state.lineage_id,
            "global_update": state.global_update,
            "cumulative_tokens": state.cumulative_tokens,
            "curriculum_phase": state.curriculum_phase,
            "backend_id": self.backend_id,
            "program_family": prog,
            "architecture": arch,
            "parameter_count": pcount,
            "state": state,
            "model": model,
            "policy": policy,
            "v8_spec": None,
        }


@dataclass(frozen=True, slots=True)
class SessionJob:
    """One job entry inside a multi-job Planthir session campaign."""

    job_id: str
    architecture: str
    init_seed: int
    order_seed: int
    updates: int
    session: str = "S0"
    device: str = "cuda:0"
    extra: dict[str, Any] = field(default_factory=dict)


class CampaignOrchestrator:
    """Multi-job prepare / validate / admit / 2-GPU wave run & resume engine built on IndependentWorkerScheduler."""

    def __init__(
        self,
        *,
        campaign_id: str,
        source_commit: str,
        work_dir: Path | str,
        max_minutes: float = 330.0,
        reserve_minutes: float = 45.0,
        min_disk_bytes: int = 2 * 1024**3,
        in_process_runner: Callable[..., Any] | None = None,
    ) -> None:
        self.campaign_id = str(campaign_id)
        self.source_commit = str(source_commit)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.max_minutes = float(max_minutes)
        self.reserve_minutes = float(reserve_minutes)
        self.min_disk_bytes = int(min_disk_bytes)
        self.in_process_runner = in_process_runner

    def prepare(self, jobs: Sequence[SessionJob]) -> dict[str, Any]:
        doc = {
            "schema": "anra-planthir-campaign-manifest-v1",
            "campaign_id": self.campaign_id,
            "source_commit": self.source_commit,
            "job_count": len(jobs),
            "jobs": [asdict(j) for j in jobs],
        }
        (self.work_dir / "PREPARED_MANIFEST.json").write_text(dump_json_safe(doc), encoding="utf-8")
        return doc

    def admit(
        self,
        jobs: Sequence[SessionJob],
        *,
        est_seconds_per_update: float = 0.05,
        per_job_disk_bytes: int = 128 * 1024 * 1024,
        parallel_workers: int = 2,
        available_disk_bytes: int | None = None,
        outer_remaining_minutes: float | None = None,
    ) -> dict[str, Any]:
        total_updates = sum(int(j.updates) for j in jobs)
        est_minutes = ((total_updates * float(est_seconds_per_update)) / max(1, int(parallel_workers))) / 60.0
        eff_max = (
            min(self.max_minutes, float(outer_remaining_minutes))
            if outer_remaining_minutes is not None
            else self.max_minutes
        )
        usable_minutes = max(0.0, eff_max - self.reserve_minutes)
        free_disk = int(available_disk_bytes if available_disk_bytes is not None else measure_free_disk_bytes(self.work_dir))
        est_disk = len(jobs) * int(per_job_disk_bytes)
        time_ok = est_minutes <= usable_minutes
        disk_ok = (free_disk - est_disk) >= self.min_disk_bytes
        admitted = time_ok and disk_ok
        receipt = {
            "schema": "anra-planthir-admission-v1",
            "admitted": admitted,
            "status": "ADMITTED" if admitted else "DENIED",
            "estimated_wall_minutes": round(est_minutes, 3),
            "usable_wall_minutes": round(usable_minutes, 3),
            "outer_remaining_minutes": (float(outer_remaining_minutes) if outer_remaining_minutes is not None else None),
            "free_disk_bytes": free_disk,
            "estimated_job_disk_bytes": est_disk,
            "min_disk_reserve_bytes": self.min_disk_bytes,
            "time_ok": time_ok,
            "disk_ok": disk_ok,
        }
        (self.work_dir / "ADMISSION_RECEIPT.json").write_text(dump_json_safe(receipt), encoding="utf-8")
        if not admitted:
            raise RuntimeError(f"campaign admission denied: time_ok={time_ok}, disk_ok={disk_ok}")
        return receipt

    def admit_slice(
        self,
        jobs: Sequence[SessionJob],
        *,
        measured_seconds_per_update: float,
        per_job_disk_bytes: int,
        parallel_workers: int = 2,
        available_disk_bytes: int | None = None,
        outer_remaining_minutes: float | None = None,
        startup_overhead_seconds: float = 0.0,
        export_reserve_bytes: int = 0,
        per_job_seconds_per_update: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Greedily pack the largest whole-job prefix fitting dual wall clocks and disk reserves."""
        if float(measured_seconds_per_update) <= 0.0:
            raise ValueError("measured_seconds_per_update must be positive (supply warmed profile timing)")
        if int(per_job_disk_bytes) <= 0:
            raise ValueError("per_job_disk_bytes must be positive")
        n_workers = max(1, int(parallel_workers))
        eff_max = (
            min(self.max_minutes, float(outer_remaining_minutes))
            if outer_remaining_minutes is not None
            else self.max_minutes
        )
        usable_minutes = max(0.0, eff_max - self.reserve_minutes)
        free_disk = int(available_disk_bytes if available_disk_bytes is not None else measure_free_disk_bytes(self.work_dir))

        lane_secs = [0.0 for _ in range(n_workers)]
        admitted_ids: list[str] = []
        deferred_ids: list[str] = []
        packing_stopped = False
        for job in jobs:
            if packing_stopped:
                deferred_ids.append(job.job_id)
                continue
            job_spu = (
                float(per_job_seconds_per_update[job.job_id])
                if per_job_seconds_per_update is not None and job.job_id in per_job_seconds_per_update
                else float(measured_seconds_per_update)
            )
            if not math.isfinite(job_spu) or job_spu <= 0.0:
                raise ValueError(f"per_job_seconds_per_update[{job.job_id!r}] must be positive, got {job_spu!r}")
            cand_lanes = list(lane_secs)
            min_lane = min(range(n_workers), key=lambda idx: cand_lanes[idx])
            cand_lanes[min_lane] += float(startup_overhead_seconds) + int(job.updates) * job_spu
            cand_minutes = max(cand_lanes) / 60.0
            cand_disk = (len(admitted_ids) + 1) * int(per_job_disk_bytes) + int(export_reserve_bytes)
            if cand_minutes <= usable_minutes and (free_disk - cand_disk) >= self.min_disk_bytes:
                lane_secs = cand_lanes
                admitted_ids.append(job.job_id)
            else:
                packing_stopped = True
                deferred_ids.append(job.job_id)

        est_minutes = (max(lane_secs) / 60.0) if admitted_ids else 0.0
        est_disk = len(admitted_ids) * int(per_job_disk_bytes) + int(export_reserve_bytes)
        plan = {
            "schema": "anra-planthir-session-admission-plan-v1",
            "campaign_id": self.campaign_id,
            "source_commit": self.source_commit,
            "admitted_job_ids": admitted_ids,
            "deferred_pending_job_ids": deferred_ids,
            "next_resumable_job_id": deferred_ids[0] if deferred_ids else None,
            "admitted_count": len(admitted_ids),
            "deferred_count": len(deferred_ids),
            "measured_seconds_per_update": float(measured_seconds_per_update),
            "estimated_wall_minutes": round(est_minutes, 3),
            "usable_wall_minutes": round(usable_minutes, 3),
            "session_max_minutes": self.max_minutes,
            "outer_remaining_minutes": (float(outer_remaining_minutes) if outer_remaining_minutes is not None else None),
            "reserve_minutes": self.reserve_minutes,
            "free_disk_bytes": free_disk,
            "estimated_disk_bytes": est_disk,
            "min_disk_reserve_bytes": self.min_disk_bytes,
        }
        if per_job_seconds_per_update is not None:
            plan["per_job_seconds_per_update"] = {
                str(k): float(v) for k, v in sorted(per_job_seconds_per_update.items())
            }
        queue_doc = {
            "schema": "anra-planthir-pending-queue-v1",
            "campaign_id": self.campaign_id,
            "pending_job_ids": deferred_ids,
            "next_resumable_job_id": plan["next_resumable_job_id"],
        }
        (self.work_dir / "SESSION_ADMISSION_PLAN.json").write_text(dump_json_safe(plan), encoding="utf-8")
        (self.work_dir / "PENDING_QUEUE.json").write_text(dump_json_safe(queue_doc), encoding="utf-8")
        return plan

    def reconcile_verified_resume(
        self,
        imported_receipts: Sequence[Mapping[str, Any]],
        registry_rows: Sequence[Mapping[str, Any]],
        *,
        required_archive_members: Sequence[str] = (),
        fail_on_rejection: bool = False,
    ) -> dict[str, Any]:
        """Verify imported job receipts (commit, spec, steps, eval rows, archive SHA/members) before skip-list rebuild."""
        reg_by_id = {str(r["job_id"]): dict(r) for r in registry_rows}
        verified_ids: list[str] = []
        rejected: list[dict[str, Any]] = []

        for raw in imported_receipts:
            jid = str(raw.get("job_id", ""))
            reg = reg_by_id.get(jid)
            reasons: list[str] = []
            if reg is None:
                reasons.append("UNKNOWN_JOB_ID")
            else:
                expected_commit = str(reg.get("source_commit") or self.source_commit)
                if str(raw.get("source_commit", "")) != expected_commit or expected_commit != self.source_commit:
                    reasons.append("SOURCE_COMMIT_MISMATCH")
                if reg.get("spec_sha256") and str(raw.get("spec_sha256", "")) != str(reg["spec_sha256"]):
                    reasons.append("SWAPPED_SPEC_SHA256")
                target_updates = int(reg.get("updates", 0))
                if int(raw.get("completed_updates", 0)) < target_updates:
                    reasons.append("INCOMPLETE_UPDATES")
                if int(raw.get("eval_rows", 0)) <= 0:
                    reasons.append("MISSING_EVAL_ROWS")
                arch_raw = raw.get("archive_path")
                if arch_raw:
                    ap = Path(str(arch_raw))
                    lp_ap = win_long_path(ap)
                    if not lp_ap.is_file():
                        reasons.append("MISSING_ARCHIVE_FILE")
                    else:
                        if raw.get("archive_sha256"):
                            actual_sha = _stream_file_sha256(lp_ap)
                            if actual_sha != str(raw["archive_sha256"]):
                                reasons.append("ARCHIVE_SHA256_MISMATCH")
                        req_members = tuple(raw.get("required_archive_members", ()) or required_archive_members)
                        if req_members:
                            with zipfile.ZipFile(lp_ap, "r") as zf:
                                z_names = set(zf.namelist())
                            for mem in req_members:
                                if str(mem) not in z_names:
                                    reasons.append(f"MISSING_ARCHIVE_MEMBER:{mem}")
                elif required_archive_members:
                    reasons.append("MISSING_ARCHIVE_PATH")

            if reasons:
                rejected.append({"job_id": jid, "reasons": reasons})
            elif jid not in verified_ids:
                verified_ids.append(jid)

        verified_sorted = sorted(verified_ids)
        pending_ids = [str(r["job_id"]) for r in registry_rows if str(r["job_id"]) not in set(verified_sorted)]
        (self.work_dir / "COMPLETED_JOBS.json").write_text(
            dump_json_safe({"completed_jobs": verified_sorted}),
            encoding="utf-8",
        )
        receipt = {
            "schema": "anra-planthir-verified-resume-v1",
            "campaign_id": self.campaign_id,
            "source_commit": self.source_commit,
            "verified_completed_jobs": verified_sorted,
            "rejected_jobs": rejected,
            "pending_job_ids": pending_ids,
            "next_resumable_job_id": pending_ids[0] if pending_ids else None,
        }
        (self.work_dir / "VERIFIED_RESUME_RECEIPT.json").write_text(
            dump_json_safe(receipt),
            encoding="utf-8",
        )
        if fail_on_rejection and rejected:
            raise ValueError(f"verified resume rejected jobs: {rejected}")
        return receipt

    def run_waves(
        self,
        jobs: Sequence[SessionJob],
        *,
        command_builder: Callable[[SessionJob, Path], tuple[str, ...]],
        devices: Sequence[str] = ("cuda:0", "cuda:1"),
        resume: bool = False,
    ) -> dict[str, Any]:
        done_file = self.work_dir / "COMPLETED_JOBS.json"
        completed: set[str] = set()
        if resume and done_file.is_file():
            completed = set(json.loads(done_file.read_text(encoding="utf-8")).get("completed_jobs", []))

        pending = [j for j in jobs if j.job_id not in completed]
        wave_receipts: list[dict[str, Any]] = []
        n_devs = max(1, len(devices))
        for w_idx, start in enumerate(range(0, len(pending), n_devs)):
            chunk = pending[start : start + n_devs]
            worker_specs = []
            for lane, job in enumerate(chunk):
                dev = str(devices[lane])
                wroot = self.work_dir / "jobs" / job.job_id
                worker_specs.append(
                    WorkerJobSpec(
                        worker_id=job.job_id,
                        work_root=wroot,
                        command=tuple(command_builder(job, wroot)),
                        device=dev,
                    )
                )
            sched = IndependentWorkerScheduler(
                coordinator_root=self.work_dir / f"wave_{w_idx:03d}",
                max_parallel=len(worker_specs),
                in_process_runner=self.in_process_runner,
            )
            rec = sched.run(worker_specs)
            wave_receipts.append(rec)
            completed.update(rec["succeeded_workers"])
            done_file.write_text(dump_json_safe({"completed_jobs": sorted(completed)}), encoding="utf-8")

        all_done = len(completed) >= len(jobs)
        summary = {
            "schema": "anra-planthir-campaign-status-v1",
            "campaign_id": self.campaign_id,
            "status": "COMPLETE" if all_done else "PARTIAL",
            "total_jobs": len(jobs),
            "completed_count": len(completed),
            "completed_jobs": sorted(completed),
            "waves": wave_receipts,
        }
        (self.work_dir / "CAMPAIGN_STATUS.json").write_text(dump_json_safe(summary), encoding="utf-8")
        return summary

    def run_stage_dag(
        self,
        stages: Sequence["CampaignStageSpec"],
        *,
        selected_phase: str = "all",
        initial_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a multi-stage conditional admission pipeline with phase selection and gating receipts."""
        ctx: dict[str, Any] = dict(initial_context or {})
        stage_records: list[dict[str, Any]] = []
        any_failed = False
        try:
            for stage in stages:
                if selected_phase not in ("all", stage.phase, stage.stage_id):
                    stage_records.append(
                        {"stage_id": stage.stage_id, "phase": stage.phase, "status": "SKIPPED_BY_PHASE", "reason": f"selected_phase={selected_phase!r}"}
                    )
                    continue
                if stage.predicate_fn is not None:
                    eligible, reason = stage.predicate_fn(ctx)
                    if not eligible:
                        rec = {"stage_id": stage.stage_id, "phase": stage.phase, "status": "NOT_APPLICABLE", "reason": str(reason)}
                        stage_records.append(rec)
                        ctx.setdefault("stages", {})[stage.stage_id] = rec
                        continue
                try:
                    res = dict(stage.runner_fn(ctx))
                except Exception as exc:
                    any_failed = True
                    rec = {
                        "stage_id": stage.stage_id,
                        "phase": stage.phase,
                        "status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    stage_records.append(rec)
                    ctx.setdefault("stages", {})[stage.stage_id] = rec
                    raise
                status = str(res.get("status", "COMPLETE"))
                if status in ("FAILED", "INVALID", "INVALID_RESULTS", "ERROR"):
                    any_failed = True
                rec = {"stage_id": stage.stage_id, "phase": stage.phase, "status": status, "result": res}
                stage_records.append(rec)
                ctx.setdefault("stages", {})[stage.stage_id] = rec
                ctx.update({k: v for k, v in res.items() if k not in ("status", "stages")})
        finally:
            receipt = {
                "schema": "anra-planthir-stage-dag-v1",
                "campaign_id": self.campaign_id,
                "selected_phase": selected_phase,
                "status": "FAILED" if any_failed else "COMPLETE",
                "stages": stage_records,
            }
            (self.work_dir / "STAGE_DAG_RECEIPT.json").write_text(
                dump_json_safe(receipt),
                encoding="utf-8",
            )
        return receipt

    def evaluate_frozen_checkpoints(
        self,
        checkpoint_root_or_zip: Path | str | None,
        *,
        expected_lineages: Sequence[str],
        eval_fn: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
        retraining_recipe_builder: Callable[[Sequence[str]], Mapping[str, Any]] | None = None,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Locate & evaluate frozen checkpoints or emit missing-artifact quarantine + retraining recipe."""
        from .checkpoint_intelligence import (
            CheckpointCatalog,
            build_deterministic_retraining_recipe,
            locate_checkpoint_store_root,
        )

        expected = [str(x) for x in expected_lineages]
        missing: list[str] = []
        evaluated: dict[str, Any] = {}
        store_root: Path | None = None

        if checkpoint_root_or_zip is None or not Path(checkpoint_root_or_zip).exists():
            missing = list(expected)
        else:
            import tempfile

            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                store_root = locate_checkpoint_store_root(
                    checkpoint_root_or_zip,
                    temp_extract_dir=Path(tmp),
                    expected_lineages=expected,
                )
                if store_root is None or not store_root.is_dir():
                    missing = list(expected)
                else:
                    for lin in expected:
                        lin_dir = store_root / lin
                        if not ((lin_dir / "LATEST").is_file() or (lin_dir / "milestones.json").is_file()):
                            missing.append(lin)
                            continue
                        bundle = CheckpointCatalog.load_checkpoint_for_causal_probing(
                            store_root,
                            lineage_id=lin,
                            mode="raw",
                            device=device,
                        )
                        try:
                            eval_out = dict(eval_fn(bundle)) if eval_fn is not None else {
                                "lineage_id": lin,
                                "checkpoint_sha256": bundle["checkpoint_sha256"],
                                "global_update": bundle["global_update"],
                                "policy_loaded": bundle.get("policy") is not None,
                            }
                            evaluated[lin] = eval_out
                        finally:
                            del bundle

        if missing:
            recipe = (
                dict(retraining_recipe_builder(missing))
                if retraining_recipe_builder is not None
                else build_deterministic_retraining_recipe(
                    missing,
                    original_source_commit=self.source_commit,
                )
            )
            receipt = {
                "schema": "anra-planthir-frozen-eval-receipt-v1",
                "campaign_id": self.campaign_id,
                "status": "MISSING_CHECKPOINT_ARTIFACT",
                "expected_lineages": expected,
                "missing_lineages": missing,
                "evaluated_lineages": evaluated,
                "retraining_recipe": recipe,
            }
        else:
            receipt = {
                "schema": "anra-planthir-frozen-eval-receipt-v1",
                "campaign_id": self.campaign_id,
                "status": "COMPLETE",
                "expected_lineages": expected,
                "missing_lineages": [],
                "evaluated_lineages": evaluated,
                "retraining_recipe": None,
            }
        (self.work_dir / "FROZEN_EVAL_RECEIPT.json").write_text(
            dump_json_safe(receipt),
            encoding="utf-8",
        )
        return receipt

    def run_registered_plan(
        self,
        jobs: Sequence[SessionJob],
        *,
        command_builder: Callable[[SessionJob, Path], Sequence[str]] | None = None,
        job_runner: Callable[[SessionJob, Path, str], Mapping[str, Any] | int] | None = None,
        devices: Sequence[str] = ("cuda:0", "cuda:1"),
        resume_input: Path | str | Sequence[Path | str] | None = None,
        measured_seconds_per_update: float | None = None,
        measured_seconds_per_update_by_group: Mapping[str, float] | None = None,
        profile_fn: Callable[[SessionJob, Path], float] | None = None,
        per_job_disk_bytes: int = 128 * 1024 * 1024,
        available_disk_bytes: int | None = None,
        outer_remaining_minutes: float | None = None,
        startup_overhead_seconds: float = 0.0,
        export_reserve_bytes: int = 0,
        inventory_only: bool = False,
        required_archive_members: Sequence[str] = (),
        fail_on_resume_rejection: bool = True,
        stage_gates: Mapping[str, Callable[[Mapping[str, Any]], tuple[bool, str]]] | None = None,
        parent_checker: Callable[[SessionJob, set[str]], tuple[bool, str]] | None = None,
        live_wall_clock_preemption: bool = True,
        clock_fn: Callable[[], float] | None = None,
        u0000_smoke_fn: Callable[[str, SessionJob, Path], Mapping[str, Any] | bool] | None = None,
        u0000_fp16_autocast_fn: Callable[[str, SessionJob, Path], Mapping[str, Any] | bool] | None = None,
        require_u0000_fp16_autocast: bool = False,
        byte_decode_fn: Callable[[Sequence[int]], Any] | None = None,
        stage5_preflight_fn: Callable[[Path], Mapping[str, Any] | bool] | None = None,
        export_prefix: str | None = None,
        export_dir: Path | str | None = None,
        extra_summary_files: Mapping[str, Path | str] | None = None,
        extra_evidence_files: Mapping[str, Path | str] | None = None,
        extra_analysis_files: Mapping[str, Path | str] | None = None,
        extra_full_files: Mapping[str, Path | str] | None = None,
        summary_max_bytes: int = 5 * 1024 * 1024,
        write_full_zip: bool = True,
        run_rescore: bool = False,
        confirm_lock_path: Path | str | None = None,
        confirm_required_job_ids: Sequence[str] = (),
        prune_completed_checkpoints: bool = False,
        strip_leaf_optimizers: bool = False,
    ) -> dict[str, Any]:
        """Execute a full registered multi-job campaign with resume, profiling, gating, 2-GPU preemption, and export."""

        clock = clock_fn or time.monotonic
        t_start = float(clock())
        eff_max_minutes = (
            min(self.max_minutes, float(outer_remaining_minutes))
            if outer_remaining_minutes is not None
            else self.max_minutes
        )

        job_list = list(jobs)
        if not job_list:
            raise ValueError("run_registered_plan requires at least one SessionJob")
        job_by_id: dict[str, SessionJob] = {}
        for job in job_list:
            if job.job_id in job_by_id:
                raise ValueError(f"duplicate job_id in run_registered_plan: {job.job_id!r}")
            job_by_id[job.job_id] = job
        device_list = tuple(str(d) for d in devices)
        if not device_list:
            raise ValueError("run_registered_plan requires at least one device")
        if not inventory_only and command_builder is None and job_runner is None and self.in_process_runner is None:
            raise ValueError("run_registered_plan requires command_builder, job_runner, or in_process_runner")

        prepared = self.prepare(job_list)
        registry_rows: list[dict[str, Any]] = []
        for job in job_list:
            spec_sha = job.extra.get("spec_sha256")
            if not spec_sha:
                spec_sha = digest(
                    {
                        "job_id": job.job_id,
                        "architecture": job.architecture,
                        "init_seed": int(job.init_seed),
                        "order_seed": int(job.order_seed),
                        "updates": int(job.updates),
                        "session": job.session,
                        "extra": {k: v for k, v in job.extra.items() if k != "spec_sha256"},
                    }
                )
            registry_rows.append(
                {
                    "job_id": job.job_id,
                    "source_commit": str(job.extra.get("source_commit") or self.source_commit),
                    "spec_sha256": str(spec_sha),
                    "updates": int(job.updates),
                    "architecture": job.architecture,
                    "init_seed": int(job.init_seed),
                    "order_seed": int(job.order_seed),
                }
            )
        reg_by_id = {r["job_id"]: r for r in registry_rows}

        raw_inputs: tuple[Path | str, ...]
        if resume_input is None:
            raw_inputs = ()
        elif isinstance(resume_input, (str, Path)):
            raw_inputs = (resume_input,) if str(resume_input).strip() else ()
        else:
            raw_inputs = tuple(x for x in resume_input if x is not None and str(x).strip())

        imported_receipts: list[dict[str, Any]] = []
        seen_receipt_keys: set[tuple[Any, ...]] = set()

        def _append_receipt(rec: Mapping[str, Any], *, base_dir: Path | None = None) -> None:
            row = dict(rec)
            arch_raw = row.get("archive_path")
            if arch_raw and base_dir is not None and not Path(str(arch_raw)).is_file():
                rel_candidate = base_dir / str(arch_raw)
                if rel_candidate.is_file():
                    row["archive_path"] = str(rel_candidate)
            key = (
                str(row.get("job_id", "")),
                str(row.get("source_commit", "")),
                str(row.get("spec_sha256", "")),
                row.get("completed_updates"),
                row.get("eval_rows"),
                str(row.get("archive_path", "")),
                str(row.get("archive_sha256", "")),
            )
            if key not in seen_receipt_keys:
                seen_receipt_keys.add(key)
                imported_receipts.append(row)

        def _collect_dir_receipts(root_dir: Path) -> None:
            jr_file = win_long_path(root_dir / "JOB_RECEIPTS.json")
            if jr_file.is_file():
                doc = json.loads(jr_file.read_text(encoding="utf-8"))
                rows = doc if isinstance(doc, list) else doc.get("receipts", [])
                for item in rows:
                    if isinstance(item, Mapping):
                        _append_receipt(item, base_dir=root_dir)
            jobs_root = root_dir / "jobs"
            if jobs_root.is_dir():
                for per_job in sorted(jobs_root.glob("*/JOB_RECEIPT.json")):
                    item = json.loads(win_long_path(per_job).read_text(encoding="utf-8"))
                    if isinstance(item, Mapping):
                        _append_receipt(item, base_dir=root_dir)

        _collect_dir_receipts(self.work_dir)
        staging_root = self.work_dir / "_rs"
        for idx, inp in enumerate(raw_inputs):
            ip = Path(inp)
            if ip.is_file() and ip.suffix.lower() == ".zip":
                verify_session_archive(ip, require_sidecar_sha256=False, require_v6_manifest=False)
                ext_dir = staging_root / f"z{idx:03d}"
                shutil.rmtree(ext_dir, ignore_errors=True)
                safe_extract_planthir_archive(ip, ext_dir)
                _collect_dir_receipts(ext_dir)
            elif ip.is_dir():
                _collect_dir_receipts(ip)
                zip_cands: list[Path] = []
                for scan_dir in (ip, ip / "exports"):
                    if scan_dir.is_dir():
                        zip_cands.extend(sorted(scan_dir.glob("*.zip")))
                zip_cands.sort(
                    key=lambda z: (
                        0 if "_FULL" in z.name else (1 if "_EVIDENCE" in z.name else (2 if "_ANALYSIS" in z.name else 3)),
                        z.name,
                    )
                )
                for zidx, zp in enumerate(zip_cands):
                    verify_session_archive(zp, require_sidecar_sha256=False, require_v6_manifest=False)
                    ext_dir = staging_root / f"d{idx:03d}_{zidx:03d}"
                    shutil.rmtree(ext_dir, ignore_errors=True)
                    safe_extract_planthir_archive(zp, ext_dir)
                    _collect_dir_receipts(ext_dir)
            else:
                raise FileNotFoundError(f"resume_input not found: {ip}")

        resume_receipt = self.reconcile_verified_resume(
            imported_receipts,
            registry_rows,
            required_archive_members=required_archive_members,
            fail_on_rejection=fail_on_resume_rejection,
        )
        verified_completed = set(resume_receipt["verified_completed_jobs"])
        remaining_jobs = [j for j in job_list if j.job_id not in verified_completed]

        per_job_spu: dict[str, float] = {}
        use_cuda_sync = torch.cuda.is_available() and any(str(d).startswith("cuda") for d in device_list)
        if measured_seconds_per_update_by_group is not None:
            for j in remaining_jobs:
                grp = str(j.extra.get("profile_group") or j.architecture)
                if grp in measured_seconds_per_update_by_group:
                    spu = float(measured_seconds_per_update_by_group[grp])
                elif measured_seconds_per_update is not None:
                    spu = float(measured_seconds_per_update)
                else:
                    raise ValueError(f"missing measured_seconds_per_update for profile_group {grp!r}")
                if not math.isfinite(spu) or spu <= 0.0:
                    raise ValueError(f"measured seconds per update for {grp!r} must be positive, got {spu!r}")
                per_job_spu[j.job_id] = spu
            sec_per_upd = (
                float(measured_seconds_per_update)
                if measured_seconds_per_update is not None
                else max(per_job_spu.values(), default=1e-6)
            )
        elif measured_seconds_per_update is None:
            if remaining_jobs and profile_fn is not None:
                explicit_groups = [j for j in remaining_jobs if j.extra.get("profile_group") not in (None, "")]
                with tempfile.TemporaryDirectory(prefix="prof_", ignore_cleanup_errors=True) as prof_tmp:
                    if explicit_groups:
                        reps_by_group: dict[str, SessionJob] = {}
                        for j in remaining_jobs:
                            grp = str(j.extra.get("profile_group") or j.architecture)
                            reps_by_group.setdefault(grp, j)
                        group_spu: dict[str, float] = {}
                        for g_idx, (grp, rep_job) in enumerate(reps_by_group.items()):
                            g_dir = Path(prof_tmp) / f"g{g_idx:02d}"
                            g_dir.mkdir(parents=True, exist_ok=True)
                            if use_cuda_sync:
                                torch.cuda.synchronize()
                            spu = float(profile_fn(rep_job, g_dir))
                            if use_cuda_sync:
                                torch.cuda.synchronize()
                            if not math.isfinite(spu) or spu <= 0.0:
                                raise ValueError(f"profile_fn must return positive seconds per update for {grp!r}, got {spu!r}")
                            group_spu[grp] = spu
                        for j in remaining_jobs:
                            grp = str(j.extra.get("profile_group") or j.architecture)
                            per_job_spu[j.job_id] = group_spu[grp]
                        sec_per_upd = max(group_spu.values())
                    else:
                        prof_dir = Path(prof_tmp) / "profile_clone"
                        prof_dir.mkdir(parents=True, exist_ok=True)
                        if use_cuda_sync:
                            torch.cuda.synchronize()
                        sec_per_upd = float(profile_fn(remaining_jobs[0], prof_dir))
                        if use_cuda_sync:
                            torch.cuda.synchronize()
                        if not math.isfinite(sec_per_upd) or sec_per_upd <= 0.0:
                            raise ValueError(f"profile_fn must return positive seconds per update, got {sec_per_upd!r}")
            elif not remaining_jobs:
                sec_per_upd = 1e-6
            else:
                raise ValueError("run_registered_plan requires positive measured_seconds_per_update or profile_fn")
        else:
            sec_per_upd = float(measured_seconds_per_update)

        admission_plan = self.admit_slice(
            remaining_jobs,
            measured_seconds_per_update=sec_per_upd,
            per_job_disk_bytes=per_job_disk_bytes,
            parallel_workers=len(device_list),
            available_disk_bytes=available_disk_bytes,
            outer_remaining_minutes=outer_remaining_minutes,
            startup_overhead_seconds=startup_overhead_seconds,
            export_reserve_bytes=export_reserve_bytes,
            per_job_seconds_per_update=(per_job_spu or None),
        )
        admitted_ids = list(admission_plan["admitted_job_ids"])
        admitted_set = set(admitted_ids)
        admitted_jobs = [j for j in remaining_jobs if j.job_id in admitted_set]
        deferred_ids = list(admission_plan["deferred_pending_job_ids"])

        if inventory_only:
            inv_doc = {
                "schema": "anra-planthir-registered-plan-status-v1",
                "status": "INVENTORY_ONLY",
                "campaign_id": self.campaign_id,
                "source_commit": self.source_commit,
                "work_dir": str(self.work_dir),
                "prepared": prepared,
                "resume_receipt": resume_receipt,
                "admission_plan": admission_plan,
                "admitted_job_ids": admitted_ids,
                "deferred_pending_job_ids": deferred_ids,
                "dispatched_job_ids": [],
                "newly_executed_job_ids": [],
                "failed_job_ids": [],
                "blocked_jobs": [],
                "blocked_job_ids": [],
                "stage_gate_results": {},
                "gated_locked_job_ids": [],
                "wall_clock_preempted": False,
                "preemption_reason": None,
                "u0000_smoke_receipt": None,
                "stage5_preflight_receipt": None,
                "byte_decoder_shield": None,
                "completed_jobs": sorted(verified_completed),
                "dispatched_devices": [],
                "observed_seeds": sorted({int(j.init_seed) for j in job_list}),
                "dataset_counts": {},
                "pending_queue_path": str(self.work_dir / "PENDING_QUEUE.json"),
                "waves": [],
                "confirm_lock": (
                    ConfirmLock(confirm_lock_path).inspect(sorted(verified_completed))
                    if confirm_lock_path is not None
                    else None
                ),
                "pruned_artifacts": None,
                "export_receipt": None,
                "archive_verifications": {},
                "rescore_receipt": None,
            }
            (self.work_dir / "INVENTORY_PLAN.json").write_text(
                dump_json_safe(inv_doc),
                encoding="utf-8",
            )
            return inv_doc

        confirm_lock_obj: ConfirmLock | None = None
        if confirm_lock_path is not None:
            confirm_lock_obj = ConfirmLock(confirm_lock_path)
            req_lock_ids = [str(x) for x in confirm_required_job_ids] if confirm_required_job_ids else [j.job_id for j in job_list]
            confirm_lock_obj.seal(
                req_lock_ids,
                source_commit=self.source_commit,
                metadata={"campaign_id": self.campaign_id},
            )

        def _job_requests_amp(j: SessionJob) -> bool:
            return bool(
                str(j.extra.get("precision", "")).lower() in ("fp16", "amp", "float16")
                or bool(j.extra.get("amp", False))
            )

        byte_shield: dict[str, Any] | None = None
        stage5_preflight_receipt: dict[str, Any] | None = None
        u0000_smoke_receipt: dict[str, Any] | None = None
        if admitted_jobs:
            byte_shield = validate_byte_decoder_resilience(decode_fn=byte_decode_fn)
            if not bool(byte_shield.get("passed", False)):
                u0000_smoke_receipt = {
                    "schema": "anra-planthir-u0000-smoke-v1",
                    "campaign_id": self.campaign_id,
                    "passed": False,
                    "byte_decoder_shield": byte_shield,
                    "fp16_autocast_checked": False,
                    "architectures": {},
                }
                (self.work_dir / "U0000_SMOKE_RECEIPT.json").write_text(
                    dump_json_safe(u0000_smoke_receipt),
                    encoding="utf-8",
                )
                raise RuntimeError(f"U0000_BYTE_DECODER_SHIELD_FAILED:{byte_shield}")

        any_amp_job = any(_job_requests_amp(j) for j in admitted_jobs)
        if (
            u0000_smoke_fn is not None
            or u0000_fp16_autocast_fn is not None
            or require_u0000_fp16_autocast
            or any_amp_job
            or byte_decode_fn is not None
        ) and admitted_jobs:
            arch_reps: dict[str, SessionJob] = {}
            arch_amp_rep: dict[str, SessionJob] = {}
            for j in admitted_jobs:
                arch_reps.setdefault(j.architecture, j)
                if _job_requests_amp(j):
                    arch_amp_rep.setdefault(j.architecture, j)
            arch_results: dict[str, Any] = {}
            with tempfile.TemporaryDirectory(prefix="u0_", ignore_cleanup_errors=True) as u0_tmp:
                for a_idx, (arch, rep_job) in enumerate(arch_reps.items()):
                    rep_eff = arch_amp_rep.get(arch, rep_job)
                    s_dir = Path(u0_tmp) / f"a{a_idx:02d}"
                    s_dir.mkdir(parents=True, exist_ok=True)
                    smoke_passed = True
                    detail: dict[str, Any] = {}
                    if u0000_smoke_fn is not None:
                        raw_smoke = u0000_smoke_fn(arch, rep_eff, s_dir)
                        if isinstance(raw_smoke, Mapping):
                            smoke_passed = bool(raw_smoke.get("passed", True))
                            detail = dict(raw_smoke)
                        else:
                            smoke_passed = bool(raw_smoke)
                            detail = {"passed": smoke_passed}

                    arch_needs_fp16 = bool(
                        require_u0000_fp16_autocast
                        or u0000_fp16_autocast_fn is not None
                        or _job_requests_amp(rep_eff)
                    )
                    fp16_passed: bool | None = None
                    fp16_detail: dict[str, Any] | None = None
                    if u0000_fp16_autocast_fn is not None:
                        fp16_dir = s_dir / "fp16"
                        fp16_dir.mkdir(parents=True, exist_ok=True)
                        raw_fp16 = u0000_fp16_autocast_fn(arch, rep_eff, fp16_dir)
                        if isinstance(raw_fp16, Mapping):
                            fp16_passed = bool(raw_fp16.get("passed", True))
                            fp16_detail = dict(raw_fp16)
                        else:
                            fp16_passed = bool(raw_fp16)
                            fp16_detail = {"passed": fp16_passed}
                    elif arch_needs_fp16:
                        fp16_passed = bool(detail.get("fp16_autocast_passed", False))

                    passed = bool(smoke_passed and (True if fp16_passed is None else fp16_passed))
                    entry: dict[str, Any] = {
                        "architecture": arch,
                        "sample_job_id": rep_eff.job_id,
                        **detail,
                        "passed": passed,
                    }
                    if fp16_detail is not None:
                        entry["fp16_autocast"] = fp16_detail
                    if fp16_passed is not None:
                        entry["fp16_autocast_passed"] = fp16_passed
                    arch_results[arch] = entry
                    u0000_smoke_receipt = {
                        "schema": "anra-planthir-u0000-smoke-v1",
                        "campaign_id": self.campaign_id,
                        "passed": bool(all(r["passed"] for r in arch_results.values()) and bool(byte_shield is None or byte_shield.get("passed"))),
                        "byte_decoder_shield": byte_shield,
                        "fp16_autocast_checked": any(bool(r.get("fp16_autocast_passed")) for r in arch_results.values()),
                        "architectures": arch_results,
                    }
                    (self.work_dir / "U0000_SMOKE_RECEIPT.json").write_text(
                        dump_json_safe(u0000_smoke_receipt),
                        encoding="utf-8",
                    )
                    if u0000_smoke_fn is not None and not smoke_passed:
                        raise RuntimeError(f"U0000_SMOKE_FAILED:{arch}:{detail}")
                    if u0000_fp16_autocast_fn is not None and not bool(fp16_passed):
                        raise RuntimeError(f"U0000_FP16_AUTOCAST_FAILED:{arch}:{fp16_detail}")
                    if arch_needs_fp16 and u0000_fp16_autocast_fn is None and not bool(fp16_passed):
                        raise RuntimeError(f"U0000_FP16_AUTOCAST_REQUIRED_FOR_AMP_JOB:{arch}:{rep_eff.job_id}")

        if stage5_preflight_fn is not None and admitted_jobs:
            with tempfile.TemporaryDirectory(prefix="s5p_", ignore_cleanup_errors=True) as s5_tmp:
                s5_dir = Path(s5_tmp) / "stage5"
                s5_dir.mkdir(parents=True, exist_ok=True)
                s5_err: BaseException | None = None
                try:
                    raw_s5 = stage5_preflight_fn(s5_dir)
                    if isinstance(raw_s5, Mapping):
                        s5_passed = bool(raw_s5.get("passed", True))
                        s5_detail = dict(raw_s5)
                        s5_detail.setdefault("passed", s5_passed)
                    else:
                        s5_passed = bool(raw_s5)
                        s5_detail = {"passed": s5_passed}
                except BaseException as exc:
                    s5_passed = False
                    s5_err = exc
                    s5_detail = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
                stage5_preflight_receipt = {
                    "schema": "anra-planthir-stage5-preflight-v1",
                    "campaign_id": self.campaign_id,
                    **s5_detail,
                }
                if u0000_smoke_receipt is None:
                    u0000_smoke_receipt = {
                        "schema": "anra-planthir-u0000-smoke-v1",
                        "campaign_id": self.campaign_id,
                        "passed": bool(s5_passed and (byte_shield is None or byte_shield.get("passed"))),
                        "byte_decoder_shield": byte_shield,
                        "fp16_autocast_checked": False,
                        "architectures": {},
                    }
                else:
                    u0000_smoke_receipt["passed"] = bool(u0000_smoke_receipt.get("passed", True) and s5_passed)
                u0000_smoke_receipt["stage5_preflight"] = stage5_preflight_receipt
                (self.work_dir / "U0000_SMOKE_RECEIPT.json").write_text(
                    dump_json_safe(u0000_smoke_receipt),
                    encoding="utf-8",
                )
                (self.work_dir / "STAGE5_PREFLIGHT_RECEIPT.json").write_text(
                    dump_json_safe(stage5_preflight_receipt),
                    encoding="utf-8",
                )
                if not s5_passed:
                    if s5_err is not None:
                        raise RuntimeError(f"STAGE5_PREFLIGHT_FAILED:{s5_detail}") from s5_err
                    raise RuntimeError(f"STAGE5_PREFLIGHT_FAILED:{s5_detail}")

        def _job_deps(j: SessionJob) -> tuple[str, ...]:
            raw: list[str] = []
            for p_key in ("parent_job_id", "adaptation_parent_id"):
                p_val = j.extra.get(p_key)
                if p_val is not None and str(p_val).strip():
                    raw.append(str(p_val).strip())
            d_val = j.extra.get("depends_on")
            if isinstance(d_val, str) and d_val.strip():
                raw.append(d_val.strip())
            elif isinstance(d_val, Sequence) and not isinstance(d_val, (str, bytes)):
                for item in d_val:
                    if item is not None and str(item).strip():
                        raw.append(str(item).strip())
            return tuple(dict.fromkeys(raw))

        children_by_parent: dict[str, list[str]] = {}
        for j in job_list:
            for dep_id in _job_deps(j):
                children_by_parent.setdefault(dep_id, []).append(j.job_id)

        def _job_req_paths(j: SessionJob) -> tuple[str, ...]:
            rp_val = j.extra.get("required_parent_paths")
            if isinstance(rp_val, (str, Path)) and str(rp_val).strip():
                return (str(rp_val),)
            if isinstance(rp_val, Sequence) and not isinstance(rp_val, (str, bytes)):
                return tuple(str(x) for x in rp_val if x is not None and str(x).strip())
            return ()

        cmd_builder = command_builder or (
            lambda j, wroot: ("planthir-registered-job", j.job_id, "--work-root", str(wroot), "--device", j.device)
        )

        receipts_by_id: dict[str, dict[str, Any]] = {}
        imported_by_id = {str(r.get("job_id", "")): dict(r) for r in imported_receipts}
        for jid in sorted(verified_completed):
            if jid in imported_by_id:
                receipts_by_id[jid] = imported_by_id[jid]

        done_file = self.work_dir / "COMPLETED_JOBS.json"
        completed_set: set[str] = set(verified_completed)
        dispatched_job_ids: list[str] = []
        dispatched_devices_set: set[str] = set()
        succeeded_this_session: list[str] = []
        failed_job_ids: list[str] = []
        worker_by_id: dict[str, dict[str, Any]] = {}
        waves: list[dict[str, Any]] = []
        blocked_jobs: list[dict[str, Any]] = []
        blocked_ids: set[str] = set()
        stage_gate_results: dict[str, dict[str, Any]] = {}
        gated_locked_job_ids: list[str] = []
        locked_ids: set[str] = set()
        unlocked_stages: set[str] = set()
        wall_clock_preempted = False
        preemption_reason: str | None = None

        pending_queue_path = self.work_dir / "PENDING_QUEUE.json"
        reg_status_path = self.work_dir / "REGISTERED_PLAN_STATUS.json"

        def _write_queue_and_receipts() -> tuple[list[str], list[str]]:
            jobs_dir = self.work_dir / "jobs"
            if jobs_dir.is_dir():
                for j in job_list:
                    if j.job_id in receipts_by_id:
                        continue
                    jr_path = win_long_path(jobs_dir / j.job_id / "JOB_RECEIPT.json")
                    if jr_path.is_file():
                        try:
                            jr_data = json.loads(jr_path.read_text(encoding="utf-8"))
                            if isinstance(jr_data, Mapping):
                                receipts_by_id[j.job_id] = dict(jr_data)
                                completed_set.add(j.job_id)
                                if j.job_id not in succeeded_this_session:
                                    succeeded_this_session.append(j.job_id)
                        except (OSError, ValueError):
                            pass

            comp_jobs = sorted(completed_set)
            pend_jobs = [
                j.job_id
                for j in job_list
                if j.job_id not in completed_set and j.job_id not in blocked_ids and j.job_id not in locked_ids
            ]
            (self.work_dir / "JOB_RECEIPTS.json").write_text(
                dump_json_safe(
                    {
                        "schema": "anra-planthir-job-receipts-v1",
                        "campaign_id": self.campaign_id,
                        "source_commit": self.source_commit,
                        "receipts": [receipts_by_id[k] for k in sorted(receipts_by_id)],
                    }
                ),
                encoding="utf-8",
            )
            done_file.write_text(
                dump_json_safe({"completed_jobs": comp_jobs}),
                encoding="utf-8",
            )
            pending_queue_path.write_text(
                dump_json_safe(
                    {
                        "schema": "anra-planthir-pending-queue-v1",
                        "campaign_id": self.campaign_id,
                        "pending_job_ids": pend_jobs,
                        "next_resumable_job_id": pend_jobs[0] if pend_jobs else None,
                    }
                ),
                encoding="utf-8",
            )
            (self.work_dir / "BLOCKED_JOBS.json").write_text(
                dump_json_safe(
                    {
                        "schema": "anra-planthir-blocked-jobs-v1",
                        "campaign_id": self.campaign_id,
                        "blocked_job_ids": sorted(blocked_ids),
                        "blocked_jobs": blocked_jobs,
                    }
                ),
                encoding="utf-8",
            )
            (self.work_dir / "STAGE_GATES.json").write_text(
                dump_json_safe(
                    {
                        "schema": "anra-planthir-stage-gates-v1",
                        "campaign_id": self.campaign_id,
                        "stage_gate_results": stage_gate_results,
                        "gated_locked_job_ids": sorted(locked_ids),
                    }
                ),
                encoding="utf-8",
            )
            return comp_jobs, pend_jobs

        def _write_status_artifacts(
            comp_jobs: list[str],
            pend_jobs: list[str],
            *,
            status_override: str | None = None,
            error_msg: str | None = None,
            confirm_lock_report: dict[str, Any] | None = None,
            pruned_summary: dict[str, Any] | None = None,
        ) -> tuple[str, dict[str, Any]]:
            obs_seeds = sorted({int(job_by_id[jid].init_seed) for jid in comp_jobs if jid in job_by_id})
            ds_counts: dict[str, int] = {}
            for jid in comp_jobs:
                r_counts = (
                    receipts_by_id[jid].get("dataset_counts")
                    if jid in receipts_by_id and isinstance(receipts_by_id[jid].get("dataset_counts"), Mapping)
                    else None
                )
                e_counts = (
                    job_by_id[jid].extra.get("dataset_counts")
                    if jid in job_by_id and isinstance(job_by_id[jid].extra.get("dataset_counts"), Mapping)
                    else None
                )
                src = r_counts if r_counts else e_counts
                if isinstance(src, Mapping):
                    for dk, dv in src.items():
                        iv = int(dv)
                        prev_v = ds_counts.get(str(dk))
                        ds_counts[str(dk)] = iv if prev_v is None else max(prev_v, iv)

            camp_status = status_override or (
                "COMPLETE"
                if (not pend_jobs and not failed_job_ids and not blocked_jobs and not wall_clock_preempted)
                else "PARTIAL"
            )
            c_doc: dict[str, Any] = {
                "schema": "anra-planthir-campaign-status-v1",
                "campaign_id": self.campaign_id,
                "status": camp_status,
                "total_jobs": len(job_list),
                "completed_count": len(comp_jobs),
                "completed_jobs": comp_jobs,
                "waves": waves,
                "confirm_lock": confirm_lock_report,
                "pruned_artifacts": pruned_summary,
            }
            if error_msg is not None:
                c_doc["error"] = error_msg
            (self.work_dir / "CAMPAIGN_STATUS.json").write_text(dump_json_safe(c_doc), encoding="utf-8")

            s_doc: dict[str, Any] = {
                "schema": "anra-planthir-registered-plan-status-v1",
                "status": camp_status,
                "campaign_id": self.campaign_id,
                "source_commit": self.source_commit,
                "work_dir": str(self.work_dir),
                "total_jobs": len(job_list),
                "prepared": prepared,
                "resume_receipt": resume_receipt,
                "admission_plan": admission_plan,
                "admitted_job_ids": admitted_ids,
                "deferred_pending_job_ids": pend_jobs,
                "dispatched_job_ids": dispatched_job_ids,
                "newly_executed_job_ids": succeeded_this_session,
                "failed_job_ids": failed_job_ids,
                "blocked_jobs": blocked_jobs,
                "blocked_job_ids": sorted(blocked_ids),
                "stage_gate_results": stage_gate_results,
                "gated_locked_job_ids": sorted(locked_ids),
                "wall_clock_preempted": wall_clock_preempted,
                "preemption_reason": preemption_reason,
                "u0000_smoke_receipt": u0000_smoke_receipt,
                "stage5_preflight_receipt": stage5_preflight_receipt,
                "byte_decoder_shield": byte_shield,
                "completed_jobs": comp_jobs,
                "dispatched_devices": sorted(d for d in dispatched_devices_set if d),
                "observed_seeds": obs_seeds,
                "dataset_counts": ds_counts,
                "pending_queue_path": str(pending_queue_path),
                "waves": waves,
                "confirm_lock": confirm_lock_report,
                "pruned_artifacts": pruned_summary,
                "export_receipt": None,
                "archive_verifications": {},
                "rescore_receipt": None,
            }
            if error_msg is not None:
                s_doc["error"] = error_msg
            reg_status_path.write_text(dump_json_safe(s_doc), encoding="utf-8")
            return camp_status, s_doc

        prev_runner = self.in_process_runner
        try:
            if job_runner is not None:
                def _inproc_adapter(wspec: WorkerJobSpec, env: Mapping[str, str], _cmd: Sequence[str]) -> dict[str, Any]:
                    job = job_by_id[wspec.worker_id]
                    assigned_dev = str(env.get("PLANTHIR_ASSIGNED_DEVICE", wspec.device))
                    res = job_runner(job, Path(wspec.work_root), assigned_dev)
                    if isinstance(res, Mapping):
                        exit_code = int(res.get("exit_code", 0))
                        payload = dict(res)
                    else:
                        exit_code = int(res)
                        payload = {}
                    receipt_file = Path(wspec.work_root) / "JOB_RECEIPT.json"
                    if exit_code == 0 and not receipt_file.is_file():
                        reg_row = reg_by_id[job.job_id]
                        j_rec = {
                            **payload,
                            "job_id": job.job_id,
                            "source_commit": reg_row["source_commit"],
                            "spec_sha256": reg_row["spec_sha256"],
                            "architecture": job.architecture,
                            "init_seed": int(job.init_seed),
                            "order_seed": int(job.order_seed),
                            "device": assigned_dev,
                            "completed_updates": int(payload.get("completed_updates", job.updates)),
                            "eval_rows": int(payload.get("eval_rows", max(1, int(job.extra.get("eval_rows", 1))))),
                            "dataset_counts": dict(payload.get("dataset_counts") or job.extra.get("dataset_counts", {})),
                        }
                        j_rec.pop("exit_code", None)
                        receipt_file.write_text(
                            dump_json_safe(j_rec),
                            encoding="utf-8",
                        )
                    return {"exit_code": exit_code}

                self.in_process_runner = _inproc_adapter

            pending_admitted = list(admitted_jobs)
            wave_idx = 0
            n_devs = max(1, len(device_list))
            while pending_admitted:
                # 1. Fixed-point check: quarantine any job whose upstream parent is missing, failed, or blocked.
                while True:
                    newly_blocked = False
                    still_pending: list[SessionJob] = []
                    pending_id_set = {pj.job_id for pj in pending_admitted}
                    for pj in pending_admitted:
                        deps = _job_deps(pj)
                        req_paths = _job_req_paths(pj)
                        missing_deps = [d for d in deps if d not in completed_set]
                        missing_paths = [rp for rp in req_paths if not win_long_path(rp).exists()]
                        unresolvable = [d for d in missing_deps if d not in pending_id_set]
                        block_reason: str | None = None
                        if unresolvable:
                            block_reason = f"MISSING_PARENT_DEPENDENCY:{unresolvable}"
                        elif missing_paths and not missing_deps:
                            block_reason = f"MISSING_PARENT_ARTIFACT:{missing_paths}"
                        elif parent_checker is not None and not missing_deps:
                            p_ok, p_reason = parent_checker(pj, completed_set)
                            if not p_ok:
                                block_reason = str(p_reason or "PARENT_CHECK_FAILED")
                        if block_reason is not None:
                            blocked_jobs.append(
                                {
                                    "job_id": pj.job_id,
                                    "architecture": pj.architecture,
                                    "status": "BLOCKED_MISSING_PARENT",
                                    "reason": block_reason,
                                }
                            )
                            blocked_ids.add(pj.job_id)
                            newly_blocked = True
                        else:
                            still_pending.append(pj)
                    pending_admitted = still_pending
                    if not newly_blocked:
                        break

                if not pending_admitted:
                    break

                # 2. Separate ready ungated vs stage-gated jobs.
                ungated_ready: list[SessionJob] = []
                gated_ready: list[SessionJob] = []
                for pj in pending_admitted:
                    deps = _job_deps(pj)
                    if not all(d in completed_set for d in deps):
                        continue
                    s_key = str(pj.extra.get("stage") or pj.session)
                    if stage_gates is not None and s_key in stage_gates and s_key not in unlocked_stages:
                        gated_ready.append(pj)
                    else:
                        ungated_ready.append(pj)

                selected_jobs: list[SessionJob] = []
                if ungated_ready:
                    selected_jobs = ungated_ready[:n_devs]
                elif gated_ready and stage_gates is not None:
                    target_stage = str(gated_ready[0].extra.get("stage") or gated_ready[0].session)
                    live_ctx = {
                        "campaign_id": self.campaign_id,
                        "source_commit": self.source_commit,
                        "work_dir": str(self.work_dir),
                        "completed_jobs": sorted(completed_set),
                        "receipts_by_id": dict(receipts_by_id),
                        "failed_job_ids": list(failed_job_ids),
                        "blocked_jobs": list(blocked_jobs),
                        "stage": target_stage,
                    }
                    g_ok, g_reason = stage_gates[target_stage](live_ctx)
                    if bool(g_ok):
                        unlocked_stages.add(target_stage)
                        stage_gate_results[target_stage] = {"status": "UNLOCKED", "reason": str(g_reason)}
                        stage_jobs = [pj for pj in gated_ready if str(pj.extra.get("stage") or pj.session) == target_stage]
                        selected_jobs = stage_jobs[:n_devs]
                    else:
                        stage_gate_results[target_stage] = {"status": "GATED_LOCKED", "reason": str(g_reason)}
                        next_pending: list[SessionJob] = []
                        for pj in pending_admitted:
                            if str(pj.extra.get("stage") or pj.session) == target_stage:
                                gated_locked_job_ids.append(pj.job_id)
                                locked_ids.add(pj.job_id)
                            else:
                                next_pending.append(pj)
                        pending_admitted = next_pending
                        continue
                else:
                    for pj in pending_admitted:
                        blocked_jobs.append(
                            {
                                "job_id": pj.job_id,
                                "architecture": pj.architecture,
                                "status": "BLOCKED_MISSING_PARENT",
                                "reason": "UNSATISFIED_DEPENDENCY_AFTER_UPSTREAM_COMPLETION",
                            }
                        )
                        blocked_ids.add(pj.job_id)
                    pending_admitted = []
                    break

                # 3. Live inter-wave wall-clock reserve preemption check before dispatching the next wave.
                if live_wall_clock_preemption:
                    elapsed_min = max(0.0, (float(clock()) - t_start) / 60.0)
                    rem_min = eff_max_minutes - elapsed_min
                    if rem_min <= self.reserve_minutes:
                        wall_clock_preempted = True
                        preemption_reason = (
                            f"WALL_CLOCK_RESERVE_REACHED:rem={rem_min:.2f}m<=reserve={self.reserve_minutes:.2f}m"
                        )
                        break

                selected_ids = {sj.job_id for sj in selected_jobs}
                worker_specs = []
                for lane, job in enumerate(selected_jobs):
                    dev = str(device_list[lane])
                    wroot = self.work_dir / "jobs" / job.job_id
                    worker_specs.append(
                        WorkerJobSpec(
                            worker_id=job.job_id,
                            work_root=wroot,
                            command=tuple(cmd_builder(job, wroot)),
                            device=dev,
                        )
                    )
                sched = IndependentWorkerScheduler(
                    coordinator_root=self.work_dir / f"wave_{wave_idx:03d}",
                    max_parallel=len(worker_specs),
                    in_process_runner=self.in_process_runner,
                )
                rec = sched.run(worker_specs)
                waves.append(rec)
                wave_idx += 1

                raw_workers = rec.get("workers", [])
                w_iter = raw_workers.values() if isinstance(raw_workers, Mapping) else raw_workers
                for w in w_iter:
                    wid = str(w["worker_id"])
                    dispatched_job_ids.append(wid)
                    dispatched_devices_set.add(str(w.get("device", "")))
                    worker_by_id[wid] = dict(w)
                    if w.get("status") == "SUCCEEDED":
                        completed_set.add(wid)
                        if wid not in succeeded_this_session:
                            succeeded_this_session.append(wid)
                        job = job_by_id[wid]
                        reg_row = reg_by_id[wid]
                        per_job_path = self.work_dir / "jobs" / wid / "JOB_RECEIPT.json"
                        if per_job_path.is_file():
                            j_rec = dict(json.loads(per_job_path.read_text(encoding="utf-8")))
                        else:
                            j_rec = {
                                "job_id": wid,
                                "source_commit": reg_row["source_commit"],
                                "spec_sha256": reg_row["spec_sha256"],
                                "architecture": job.architecture,
                                "init_seed": int(job.init_seed),
                                "order_seed": int(job.order_seed),
                                "device": str(w.get("device", job.device)),
                                "completed_updates": int(job.updates),
                                "eval_rows": max(1, int(job.extra.get("eval_rows", 1))),
                                "dataset_counts": dict(job.extra.get("dataset_counts", {})),
                            }
                            per_job_path.parent.mkdir(parents=True, exist_ok=True)
                            per_job_path.write_text(dump_json_safe(j_rec), encoding="utf-8")
                        receipts_by_id[wid] = j_rec
                    else:
                        if wid not in failed_job_ids:
                            failed_job_ids.append(wid)

                done_file.write_text(
                    dump_json_safe({"completed_jobs": sorted(completed_set)}),
                    encoding="utf-8",
                )
                pending_admitted = [pj for pj in pending_admitted if pj.job_id not in selected_ids]
        except BaseException as exc:
            interrupt_label = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
            p_comp, p_pend = _write_queue_and_receipts()
            _write_status_artifacts(
                p_comp,
                p_pend,
                status_override=interrupt_label,
                error_msg=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            self.in_process_runner = prev_runner

        completed_jobs, all_pending = _write_queue_and_receipts()

        pruned_summary: dict[str, Any] | None = None
        if prune_completed_checkpoints:
            total_removed = 0
            total_freed = 0
            pruned_jobs = 0
            preserved_optimizer_jobs: list[str] = []
            compacted_leaf_jobs: list[str] = []
            for jid in completed_jobs:
                jdir = self.work_dir / "jobs" / jid
                if jdir.is_dir():
                    job_obj = job_by_id.get(jid)
                    child_ids = children_by_parent.get(jid, [])
                    rec_u = receipts_by_id.get(jid, {}).get("completed_updates")
                    completed_u = int(rec_u) if rec_u is not None else None
                    raw_full_u = job_obj.extra.get("full_job_updates", job_obj.updates) if job_obj is not None else None
                    target_full_u = int(raw_full_u) if raw_full_u is not None else None
                    must_preserve = (
                        bool(job_obj.extra.get("preserve_final_optimizer", False)) if job_obj is not None else False
                    ) or bool(child_ids) or (
                        completed_u is not None and target_full_u is not None and completed_u < target_full_u
                    )
                    pr = prune_completed_job_artifacts(
                        jdir,
                        keep_latest=True,
                        keep_milestones=True,
                        strip_final_optimizer=bool(strip_leaf_optimizers),
                        preserve_final_optimizer=must_preserve,
                        child_job_ids=child_ids,
                        completed_updates=completed_u,
                        full_job_updates=target_full_u,
                    )
                    total_removed += int(pr["removed_files"])
                    total_freed += int(pr["freed_bytes"])
                    pruned_jobs += 1
                    if pr.get("optimizer_preserved"):
                        preserved_optimizer_jobs.append(jid)
                    if pr.get("optimizer_stripped"):
                        compacted_leaf_jobs.append(jid)
            pruned_summary = {
                "pruned_jobs": pruned_jobs,
                "removed_files": total_removed,
                "freed_bytes": total_freed,
                "preserved_optimizer_jobs": preserved_optimizer_jobs,
                "compacted_leaf_jobs": compacted_leaf_jobs,
            }

        confirm_lock_report: dict[str, Any] | None = None
        if confirm_lock_obj is not None:
            confirm_lock_report = confirm_lock_obj.unlock_if_ready(completed_jobs)

        campaign_status, status_doc = _write_status_artifacts(
            completed_jobs,
            all_pending,
            confirm_lock_report=confirm_lock_report,
            pruned_summary=pruned_summary,
        )

        if export_prefix is not None:
            out_dir = Path(export_dir) if export_dir is not None else (self.work_dir / "exports")
            core_names = (
                "PREPARED_MANIFEST.json",
                "SESSION_ADMISSION_PLAN.json",
                "PENDING_QUEUE.json",
                "VERIFIED_RESUME_RECEIPT.json",
                "COMPLETED_JOBS.json",
                "JOB_RECEIPTS.json",
                "CAMPAIGN_STATUS.json",
                "BLOCKED_JOBS.json",
                "STAGE_GATES.json",
                "U0000_SMOKE_RECEIPT.json",
                "STAGE5_PREFLIGHT_RECEIPT.json",
                "CONFIRM_LOCK.json",
                "REGISTERED_PLAN_STATUS.json",
            )
            core_files = {name: self.work_dir / name for name in core_names if (self.work_dir / name).is_file()}
            if confirm_lock_obj is not None and confirm_lock_obj.lock_path.is_file():
                core_files.setdefault(confirm_lock_obj.lock_path.name, confirm_lock_obj.lock_path)
            job_receipt_files: dict[str, Path] = {}
            all_job_files: dict[str, Path] = {}
            jobs_dir = self.work_dir / "jobs"
            if jobs_dir.is_dir():
                for p in sorted(jobs_dir.rglob("*")):
                    if not p.is_file() or p.name in (".planthir_worker.lock", ".planthir-run-writer"):
                        continue
                    rel_arc = p.relative_to(self.work_dir).as_posix()
                    all_job_files[rel_arc] = p
                    if p.name == "JOB_RECEIPT.json":
                        job_receipt_files[rel_arc] = p
            summary_files = {**core_files, **{str(k): Path(v) for k, v in (extra_summary_files or {}).items()}}
            evidence_files = {
                **core_files,
                **job_receipt_files,
                **{str(k): Path(v) for k, v in (extra_evidence_files or {}).items()},
            }
            analysis_files = {**core_files, **{str(k): Path(v) for k, v in (extra_analysis_files or {}).items()}}
            full_files = {
                **core_files,
                **all_job_files,
                **{str(k): Path(v) for k, v in (extra_full_files or {}).items()},
            }
            export_receipt = export_session_archives(
                self.work_dir,
                out_dir,
                prefix=str(export_prefix),
                summary_files=summary_files,
                evidence_files=evidence_files,
                analysis_files=analysis_files,
                full_files=full_files,
                status=campaign_status,
                source_commit=self.source_commit,
                summary_max_bytes=summary_max_bytes,
                include_v6_manifest=True,
                write_full_zip=write_full_zip,
            )
            expected_export_commit = export_receipt["final_status"]["source_commit"]
            archive_verifications = {
                kind: verify_session_archive(
                    arch_p,
                    max_bytes=(summary_max_bytes if kind == "summary" else None),
                    expected_source_commit=expected_export_commit,
                    require_sidecar_sha256=True,
                    require_v6_manifest=True,
                )
                for kind, arch_p in sorted(export_receipt["archives"].items())
            }
            rescore_receipt = None
            if run_rescore and "METRICS.json" in analysis_files and "analysis" in export_receipt["archives"]:
                rescore_receipt = rescore_compact_analysis_archive(
                    export_receipt["archives"]["analysis"],
                    out_dir / "rescored_report",
                )
            status_doc["export_receipt"] = export_receipt
            status_doc["archive_verifications"] = archive_verifications
            status_doc["rescore_receipt"] = rescore_receipt
            reg_status_path.write_text(dump_json_safe(status_doc), encoding="utf-8")

        return status_doc


def _extract_dispatch_telemetry(res: Mapping[str, Any], work_dir: Path) -> dict[str, Any]:
    """Normalize dispatch, GPU lane, seed, dataset denominator, and queue telemetry from a dispatch call."""

    waves = list(res.get("waves", []))
    if not waves and (work_dir / "CAMPAIGN_STATUS.json").is_file():
        waves = list(json.loads((work_dir / "CAMPAIGN_STATUS.json").read_text(encoding="utf-8")).get("waves", []))

    wave_workers: list[dict[str, Any]] = []
    for wave in waves:
        raw_workers = wave.get("workers", [])
        w_iter = raw_workers.values() if isinstance(raw_workers, Mapping) else raw_workers
        for w in w_iter:
            if isinstance(w, Mapping):
                wave_workers.append(dict(w))

    if "dispatched_job_ids" in res and res["dispatched_job_ids"] is not None:
        dispatched_job_ids = [str(x) for x in res["dispatched_job_ids"]]
    elif "newly_executed_job_ids" in res and res["newly_executed_job_ids"] is not None:
        dispatched_job_ids = [str(x) for x in res["newly_executed_job_ids"]]
    else:
        dispatched_job_ids = [str(w["worker_id"]) for w in wave_workers if "worker_id" in w]

    if "dispatched_devices" in res and res["dispatched_devices"] is not None:
        dispatched_devices = {str(x) for x in res["dispatched_devices"] if str(x)}
    else:
        dispatched_devices = {str(w["device"]) for w in wave_workers if w.get("device")}

    receipts: list[dict[str, Any]] = []
    jr_file = work_dir / "JOB_RECEIPTS.json"
    if jr_file.is_file():
        jr_doc = json.loads(jr_file.read_text(encoding="utf-8"))
        raw_rows = jr_doc if isinstance(jr_doc, list) else jr_doc.get("receipts", [])
        receipts = [dict(r) for r in raw_rows if isinstance(r, Mapping)]

    if "completed_jobs" in res and res["completed_jobs"] is not None:
        completed_jobs = [str(x) for x in res["completed_jobs"]]
    else:
        completed_jobs = [str(r["job_id"]) for r in receipts if "job_id" in r]

    if "observed_seeds" in res and res["observed_seeds"] is not None:
        observed_seeds = {int(s) for s in res["observed_seeds"]}
    else:
        observed_seeds = {int(r["init_seed"]) for r in receipts if r.get("init_seed") is not None}

    dataset_counts: dict[str, int] = {}
    if isinstance(res.get("dataset_counts"), Mapping):
        for k, v in res["dataset_counts"].items():
            dataset_counts[str(k)] = int(v)
    else:
        for r in receipts:
            d_map = r.get("dataset_counts")
            if isinstance(d_map, Mapping):
                for k, v in d_map.items():
                    iv = int(v)
                    prev = dataset_counts.get(str(k))
                    dataset_counts[str(k)] = iv if prev is None else max(prev, iv)

    if "deferred_pending_job_ids" in res and res["deferred_pending_job_ids"] is not None:
        deferred_ids = [str(x) for x in res["deferred_pending_job_ids"]]
    elif (work_dir / "PENDING_QUEUE.json").is_file():
        pq = json.loads((work_dir / "PENDING_QUEUE.json").read_text(encoding="utf-8"))
        deferred_ids = [str(x) for x in pq.get("pending_job_ids", [])]
    else:
        deferred_ids = []

    return {
        "dispatched_job_ids": dispatched_job_ids,
        "dispatched_devices": dispatched_devices,
        "completed_jobs": completed_jobs,
        "observed_seeds": observed_seeds,
        "dataset_counts": dataset_counts,
        "deferred_pending_job_ids": deferred_ids,
        "waves": waves,
    }


def verify_full_mode_dispatch(
    dispatch_fn: Callable[..., Mapping[str, Any]],
    *,
    expected_job_ids: Sequence[str],
    expected_seeds: Sequence[int] = (7, 19, 41),
    required_devices: Sequence[str] = ("cuda:0", "cuda:1"),
    min_dataset_counts: Mapping[str, int] | None = None,
    tight_session_minutes: float | None = None,
    full_session_minutes: float = 300.0,
    work_root: Path | str | None = None,
    require_multi_session_resume: bool = True,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Verify FULL-mode multi-GPU job dispatch, uncompressed dataset denominators, and tight-session overflow/resume."""

    exp_jobs = [str(x) for x in expected_job_ids]
    exp_seeds = [int(s) for s in expected_seeds]
    req_devs = [str(d) for d in required_devices]
    min_counts = {str(k): int(v) for k, v in dict(min_dataset_counts or {}).items()}
    defects: list[str] = []

    def _run_suite(root_path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
        full_dir = root_path / "full_pass"
        full_dir.mkdir(parents=True, exist_ok=True)
        res_full = dispatch_fn(full_dir, session_minutes=float(full_session_minutes), resume_input=None)
        fp = _extract_dispatch_telemetry(res_full, full_dir)

        missing_jobs = [jid for jid in exp_jobs if jid not in set(fp["dispatched_job_ids"])]
        if missing_jobs:
            defects.append(f"MISSING_REGISTERED_JOBS:{missing_jobs}")

        missing_devs = [d for d in req_devs if d not in fp["dispatched_devices"]]
        if missing_devs:
            defects.append(f"MISSING_GPU_LANES:{missing_devs}")

        max_wave_workers = max(
            (
                len(w.get("workers", []))
                for w in fp["waves"]
                if isinstance(w, Mapping)
            ),
            default=0,
        )
        if len(req_devs) >= 2 and len(exp_jobs) >= 2 and fp["waves"] and max_wave_workers < 2:
            defects.append("SERIAL_DISPATCH_DETECTED")

        missing_seeds = [s for s in exp_seeds if s not in fp["observed_seeds"]]
        if missing_seeds:
            defects.append(f"MISSING_REQUIRED_SEEDS:{missing_seeds}")

        for k, min_v in min_counts.items():
            obs_v = int(fp["dataset_counts"].get(k, -1))
            if obs_v < min_v:
                defects.append(f"COMPRESSED_DATASET_DENOMINATOR:{k}:{obs_v}<{min_v}")

        full_summary = {
            "dispatched_count": len(fp["dispatched_job_ids"]),
            "dispatched_devices": sorted(fp["dispatched_devices"]),
            "observed_seeds": sorted(fp["observed_seeds"]),
            "dataset_counts": fp["dataset_counts"],
            "max_wave_workers": max_wave_workers,
        }

        resume_summary: dict[str, Any] | None = None
        if require_multi_session_resume and tight_session_minutes is not None:
            s1_dir = root_path / "session_1"
            s1_dir.mkdir(parents=True, exist_ok=True)
            res_s1 = dispatch_fn(s1_dir, session_minutes=float(tight_session_minutes), resume_input=None)
            s1 = _extract_dispatch_telemetry(res_s1, s1_dir)
            if not s1["dispatched_job_ids"]:
                defects.append("TIGHT_SESSION_ADMITTED_ZERO_JOBS")
            if not s1["deferred_pending_job_ids"]:
                defects.append("TIGHT_SESSION_DID_NOT_DEFER_OVERFLOW")

            pq_path = s1_dir / "PENDING_QUEUE.json"
            if not pq_path.is_file():
                defects.append("MISSING_OR_MISMATCHED_PENDING_QUEUE")
            else:
                pq_ids = set(json.loads(pq_path.read_text(encoding="utf-8")).get("pending_job_ids", []))
                if not pq_ids or not set(s1["deferred_pending_job_ids"]).issubset(pq_ids):
                    defects.append("MISSING_OR_MISMATCHED_PENDING_QUEUE")

            exp_archives = (res_s1.get("export_receipt") or {}).get("archives") or {}
            resume_src: Path | str = s1_dir
            for key in ("full", "evidence", "analysis", "summary"):
                cand = exp_archives.get(key)
                if cand and Path(cand).is_file():
                    resume_src = cand
                    break

            s2_dir = root_path / "session_2"
            s2_dir.mkdir(parents=True, exist_ok=True)
            res_s2 = dispatch_fn(s2_dir, session_minutes=float(full_session_minutes), resume_input=resume_src)
            s2 = _extract_dispatch_telemetry(res_s2, s2_dir)

            overlap = sorted(set(s1["dispatched_job_ids"]) & set(s2["dispatched_job_ids"]))
            if overlap:
                defects.append(f"RESUME_REEXECUTED_COMPLETED_JOBS:{overlap}")
            missing_after = [jid for jid in exp_jobs if jid not in set(s2["completed_jobs"])]
            if missing_after:
                defects.append(f"RESUME_INCOMPLETE_CAMPAIGN:{missing_after}")
            if s2["deferred_pending_job_ids"]:
                defects.append(f"RESUME_LEFT_PENDING_JOBS:{s2['deferred_pending_job_ids']}")

            resume_summary = {
                "session_1_dispatched": s1["dispatched_job_ids"],
                "session_1_deferred": s1["deferred_pending_job_ids"],
                "session_2_dispatched": s2["dispatched_job_ids"],
                "session_2_completed": s2["completed_jobs"],
                "resume_source": str(resume_src),
            }

        return full_summary, resume_summary

    if work_root is not None:
        rp = Path(work_root)
        rp.mkdir(parents=True, exist_ok=True)
        full_summary, resume_summary = _run_suite(rp)
    else:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            full_summary, resume_summary = _run_suite(Path(tmp))

    passed = len(defects) == 0
    if not passed and raise_on_failure:
        raise ValueError(f"FULL-mode dispatch verification failed: {defects}")
    return {
        "schema": "anra-planthir-full-mode-dispatch-verify-v1",
        "passed": passed,
        "defects": defects,
        "full_pass": full_summary,
        "multi_session_resume": resume_summary,
    }


@dataclass(frozen=True, slots=True)
class CampaignStageSpec:
    """Declarative stage node inside `CampaignOrchestrator.run_stage_dag`."""

    stage_id: str
    phase: str
    runner_fn: Callable[[dict[str, Any]], Mapping[str, Any]]
    predicate_fn: Callable[[dict[str, Any]], tuple[bool, str]] | None = None
    ceiling_minutes: float | None = None
    description: str = ""


def verify_shared_backbone_and_head_overhead(
    control_model: nn.Module,
    treatment_model: nn.Module,
    *,
    backbone_prefix: str = "core.",
    max_head_overhead_ratio: float | None = 0.05,
    max_control_parity_ratio: float | None = None,
) -> dict[str, Any]:
    """Verify byte-identical shared backbone tensor digests plus head-overhead and control parity budgets."""

    def _summarize(mod: nn.Module) -> tuple[str, int, int, int]:
        h = sha256()
        bb_params = 0
        head_params = 0
        total_params = 0
        for name, param in sorted(mod.named_parameters()):
            n = int(param.numel())
            total_params += n
            if name.startswith(backbone_prefix):
                bb_params += n
                t = param.detach().cpu().contiguous()
                raw = t.view(torch.int16).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().tobytes()
                h.update(f"{name}:{tuple(t.shape)}:{t.dtype}:".encode("utf-8"))
                h.update(raw)
            else:
                head_params += n
        return h.hexdigest(), bb_params, head_params, total_params

    c_sha, c_bb, c_head, c_tot = _summarize(control_model)
    t_sha, t_bb, t_head, t_tot = _summarize(treatment_model)
    defects: list[str] = []
    if c_bb == 0 or t_bb == 0:
        defects.append(f"MISSING_BACKBONE_PREFIX:{backbone_prefix}")
    if c_sha != t_sha:
        defects.append("BACKBONE_DIGEST_MISMATCH")
    overhead_ratio = (float(t_head) / float(t_bb)) if t_bb > 0 else float("inf")
    if max_head_overhead_ratio is not None and overhead_ratio > float(max_head_overhead_ratio):
        defects.append(f"HEAD_OVERHEAD_EXCEEDED:{overhead_ratio:.4f}>{float(max_head_overhead_ratio):.4f}")
    parity_diff_ratio = (abs(float(t_tot) - float(c_tot)) / float(max(c_tot, 1)))
    if max_control_parity_ratio is not None and parity_diff_ratio > float(max_control_parity_ratio):
        defects.append(f"CONTROL_PARITY_EXCEEDED:{parity_diff_ratio:.4f}>{float(max_control_parity_ratio):.4f}")
    return {
        "schema": "anra-planthir-backbone-overhead-v1",
        "passed": len(defects) == 0,
        "backbone_prefix": str(backbone_prefix),
        "backbone_digest_match": c_sha == t_sha and c_bb > 0 and t_bb > 0,
        "control_backbone_sha256": c_sha,
        "treatment_backbone_sha256": t_sha,
        "backbone_parameters": int(t_bb),
        "control_head_parameters": int(c_head),
        "treatment_head_parameters": int(t_head),
        "treatment_head_overhead_ratio": round(overhead_ratio, 6) if math.isfinite(overhead_ratio) else None,
        "total_parameters_control": int(c_tot),
        "total_parameters_treatment": int(t_tot),
        "control_parity_diff_ratio": round(parity_diff_ratio, 6),
        "defects": defects,
    }


def build_x1_spec(
    architecture: str,
    *,
    cpu_canary: bool = False,
    base_spec: Any | None = None,
    width: int | None = None,
    layers: int | None = None,
    query_heads: int | None = None,
    kv_heads: int | None = None,
    ffn_width: int | None = None,
    context_length: int | None = None,
    pointer_dim: int | None = None,
    slot_dim: int | None = None,
    num_slots: int | None = None,
    slot_edge_hidden: int | None = None,
    agg_iterations: int | None = None,
    msg_iterations: int | None = None,
    recurrent_dim: int | None = None,
    max_sequence_length: int | None = None,
) -> Any:
    """Build a validated X1ModelSpec from canonical or `X1-*` names with CPU canary defaults."""

    from anra_v5.elendril.v8x.contracts import (
        DEFAULT_X1_AGG_ITERS,
        DEFAULT_X1_EDGE_HIDDEN,
        DEFAULT_X1_MSG_ITERS,
        DEFAULT_X1_NUM_SLOTS,
        DEFAULT_X1_POINTER_DIM,
        DEFAULT_X1_RECURRENT_DIM,
        DEFAULT_X1_SLOT_DIM,
        X1_ARCHITECTURES,
        build_x1_model_spec,
        build_x1_s20_base,
    )

    alias_map = {
        "X1-P": "P",
        "X1-R": "R",
        "X1-R-noedge": "R-noedge",
        "X1-P-capR": "P-capR",
    }
    arch_raw = str(architecture).strip()
    arch = alias_map.get(arch_raw, arch_raw)
    if arch not in X1_ARCHITECTURES:
        raise ValueError(f"unsupported X1 architecture: {architecture!r}")

    resolved_base = base_spec if base_spec is not None else build_x1_s20_base()
    has_base_overrides = any(
        x is not None for x in (width, layers, query_heads, kv_heads, ffn_width, context_length)
    )
    if (base_spec is None and cpu_canary) or has_base_overrides:
        use_canary_defaults = bool(base_spec is None and cpu_canary)
        w = (32 if use_canary_defaults else int(resolved_base.width)) if width is None else int(width)
        n_layers = (2 if use_canary_defaults else int(resolved_base.layers)) if layers is None else int(layers)
        qh = (2 if use_canary_defaults else int(resolved_base.query_heads)) if query_heads is None else int(query_heads)
        kh = (2 if use_canary_defaults else int(resolved_base.kv_heads)) if kv_heads is None else int(kv_heads)
        ffn = (64 if use_canary_defaults else int(resolved_base.ffn_width)) if ffn_width is None else int(ffn_width)
        ctx = (128 if use_canary_defaults else int(resolved_base.context_length)) if context_length is None else int(context_length)
        if qh <= 0 or w % qh != 0:
            raise ValueError(f"width ({w}) must be cleanly divisible by query_heads ({qh})")
        resolved_base = replace(
            resolved_base,
            width=w,
            layers=n_layers,
            query_heads=qh,
            kv_heads=kh,
            head_dimension=w // qh,
            ffn_width=ffn,
            context_length=ctx,
        )

    return build_x1_model_spec(
        arch,
        base_spec=resolved_base,
        pointer_dim=(16 if cpu_canary else DEFAULT_X1_POINTER_DIM) if pointer_dim is None else int(pointer_dim),
        slot_dim=(16 if cpu_canary else DEFAULT_X1_SLOT_DIM) if slot_dim is None else int(slot_dim),
        num_slots=(4 if cpu_canary else DEFAULT_X1_NUM_SLOTS) if num_slots is None else int(num_slots),
        slot_edge_hidden=(16 if cpu_canary else DEFAULT_X1_EDGE_HIDDEN) if slot_edge_hidden is None else int(slot_edge_hidden),
        agg_iterations=DEFAULT_X1_AGG_ITERS if agg_iterations is None else int(agg_iterations),
        msg_iterations=DEFAULT_X1_MSG_ITERS if msg_iterations is None else int(msg_iterations),
        recurrent_dim=(16 if cpu_canary else DEFAULT_X1_RECURRENT_DIM) if recurrent_dim is None else int(recurrent_dim),
        max_sequence_length=max_sequence_length,
    )


__all__ = [
    "BytePromptSample",
    "BytePromptTrainer",
    "CampaignLedgerVerifier",
    "CampaignOrchestrator",
    "CampaignStageSpec",
    "ConfirmLock",
    "ExperimentModule",
    "GenericDatasetBundle",
    "GenericExperimentTrainer",
    "GenericModelSpec",
    "GenericTrainRecipe",
    "LogitsPolicy",
    "SessionJob",
    "StaticPromptKernel",
    "TypedByteSequenceKernel",
    "audit_pair_margin_gradient_sign_reversal",
    "batched_rollout",
    "build_analyst_first_page_report",
    "build_pinned_kaggle_notebook",
    "build_verification_matrix",
    "build_x1_spec",
    "canonical_structural_hash",
    "collect_minimal_checkpoint_files",
    "compute_ce_and_pair_margin_terms",
    "decode_generated_bytes",
    "dump_json_safe",
    "export_session_archives",
    "frame_byte_prompt_sample",
    "infer_public_byte_bound",
    "mirror_to_working_dir",
    "normalize_typed_value",
    "parse_typed_output_envelope",
    "prune_completed_job_artifacts",
    "render_campaign_dashboard",
    "render_campaign_results_markdown",
    "rescore_compact_analysis_archive",
    "safe_extract_planthir_archive",
    "sanitize_for_json",
    "score_typed_output",
    "serialize_pinned_notebook",
    "validate_byte_decoder_resilience",
    "verify_campaign_ledger",
    "verify_full_mode_dispatch",
    "verify_pinned_notebook_bytes",
    "verify_session_archive",
    "verify_shared_backbone_and_head_overhead",
    "verify_split_disjointness",
    "win_long_path",
]
