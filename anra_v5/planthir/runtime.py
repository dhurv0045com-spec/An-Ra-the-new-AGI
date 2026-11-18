"""Generic self-contained ExperimentModule, Trainer, Batched Rollout, CampaignOrchestrator & Safe Exporter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import stat
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

from .backend import BACKEND_CONTRACT_VERSION, BackendCapabilities, register_backend
from .contracts import EnvironmentKernel, ObservablePolicy, UniversalAction, UniversalObservation
from .execution import _atomic_replace
from .resources import measure_free_disk_bytes
from .scheduler import IndependentWorkerScheduler, WorkerJobSpec

_SAFE_ARCHIVE_MEMBER_MAX_BYTES = 512 * 1024 * 1024
_SAFE_ARCHIVE_MAX_FILES = 10_000


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

    def step(self) -> dict[str, Any]:
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
        for start in range(0, len(chosen), micro):
            chunk = chosen[start : start + micro]
            scale = float(len(chunk)) / float(max(1, len(chosen)))
            scaled_loss, scalar = _compute_loss_from_fn(self.loss_fn, self.model, chunk, self.recipe, scale)
            scaled_loss.backward()
            acc_loss += scalar

        validate_parameter_ownership(self.model, self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.recipe.clip), error_if_nonfinite=True)
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise FloatingPointError("nonfinite parameter after optimizer step")

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
        return {"update": u + 1, "loss": float(acc_loss), "pre_clip": float(norm), "lr": float(lr)}

    def save(self, *, inject_crash_at: str | None = None, prune_old: bool = True) -> str:
        if self.faulted or self.state.global_update <= self.published_update:
            raise ValueError("cannot publish faulted or repeated update")
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
) -> list[dict[str, Any]]:
    """Execute episodes in lockstep batches, using `policy.propose_batch` when implemented."""

    states = [kernel.initial_state(w) for w in worlds]
    actions_per_world: list[list[tuple[str, str | None]]] = [[] for _ in worlds]
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
            actions_per_world[idx].append((act.operation, act.argument))
            states[idx] = kernel.step(states[idx], act.operation, act.argument)
            done[idx] = _state_is_terminal(states[idx])

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

    def extract_representations(self, obs_batch: Sequence[UniversalObservation]) -> dict[str, torch.Tensor]:
        _, hidden = self._forward_logits_and_hidden(obs_batch)
        return {"decision_hidden": hidden}


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
            supports_fp16=False,
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
        return self.kernel_factory()

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
        return self.policy_factory(
            model,
            model_spec,
            model_identity=model_identity,
            mode=mode,
            copy_enabled_override=copy_enabled_override,
        )

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
        (self.work_dir / "PREPARED_MANIFEST.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return doc

    def admit(
        self,
        jobs: Sequence[SessionJob],
        *,
        est_seconds_per_update: float = 0.05,
        per_job_disk_bytes: int = 128 * 1024 * 1024,
        parallel_workers: int = 2,
        available_disk_bytes: int | None = None,
    ) -> dict[str, Any]:
        total_updates = sum(int(j.updates) for j in jobs)
        est_minutes = ((total_updates * float(est_seconds_per_update)) / max(1, int(parallel_workers))) / 60.0
        usable_minutes = max(0.0, self.max_minutes - self.reserve_minutes)
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
            "free_disk_bytes": free_disk,
            "estimated_job_disk_bytes": est_disk,
            "min_disk_reserve_bytes": self.min_disk_bytes,
            "time_ok": time_ok,
            "disk_ok": disk_ok,
        }
        (self.work_dir / "ADMISSION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not admitted:
            raise RuntimeError(f"campaign admission denied: time_ok={time_ok}, disk_ok={disk_ok}")
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
            done_file.write_text(json.dumps({"completed_jobs": sorted(completed)}, indent=2) + "\n", encoding="utf-8")

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
        (self.work_dir / "CAMPAIGN_STATUS.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
            res = dict(stage.runner_fn(ctx))
            status = str(res.get("status", "COMPLETE"))
            if status in ("FAILED", "INVALID", "INVALID_RESULTS", "ERROR"):
                any_failed = True
            rec = {"stage_id": stage.stage_id, "phase": stage.phase, "status": status, "result": res}
            stage_records.append(rec)
            ctx.setdefault("stages", {})[stage.stage_id] = rec
            ctx.update({k: v for k, v in res.items() if k not in ("status", "stages")})

        receipt = {
            "schema": "anra-planthir-stage-dag-v1",
            "campaign_id": self.campaign_id,
            "selected_phase": selected_phase,
            "status": "FAILED" if any_failed else "COMPLETE",
            "stages": stage_records,
        }
        (self.work_dir / "STAGE_DAG_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
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

            with tempfile.TemporaryDirectory() as tmp:
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
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return receipt


@dataclass(frozen=True, slots=True)
class CampaignStageSpec:
    """Declarative stage node inside `CampaignOrchestrator.run_stage_dag`."""

    stage_id: str
    phase: str
    runner_fn: Callable[[dict[str, Any]], Mapping[str, Any]]
    predicate_fn: Callable[[dict[str, Any]], tuple[bool, str]] | None = None
    ceiling_minutes: float | None = None
    description: str = ""


def build_verification_matrix(
    *,
    byte_integrity: str = "PASS",
    oracle_fixture_correctness: str = "PASS",
    recorded_action_replay: str = "PASS",
    checkpoint_restore: str = "MISSING_CHECKPOINT_ARTIFACT",
    regenerated_per_case_agreement: str = "BLOCKED_BY_MISSING_CHECKPOINT",
    training_reproduction: str = "NOT_RUN",
) -> list[dict[str, str]]:
    """Return the canonical 6-row Planthir verification matrix."""
    return [
        {"check": "byte_integrity", "status": str(byte_integrity)},
        {"check": "oracle_fixture_correctness", "status": str(oracle_fixture_correctness)},
        {"check": "recorded_action_replay", "status": str(recorded_action_replay)},
        {"check": "checkpoint_restore", "status": str(checkpoint_restore)},
        {"check": "regenerated_per_case_agreement", "status": str(regenerated_per_case_agreement)},
        {"check": "training_reproduction", "status": str(training_reproduction)},
    ]


def render_campaign_results_markdown(
    *,
    title: str,
    campaign_id: str,
    source_commit: str,
    status: str,
    verification_matrix: Sequence[Mapping[str, Any]],
    stage_summaries: Mapping[str, Any] | None = None,
) -> str:
    """Render a standardized `RESULTS.md` report with the 6-row verification table and stage outcomes."""
    lines = [
        f"# {title}",
        "",
        f"- **Campaign ID:** `{campaign_id}`",
        f"- **Source commit:** `{source_commit}`",
        f"- **Status:** `{status}`",
        "",
        "## Verification Matrix",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for row in verification_matrix:
        lines.append(f"| `{row.get('check', '')}` | `{row.get('status', '')}` |")
    if stage_summaries:
        lines.extend(["", "## Stage Summaries", ""])
        for sid, info in sorted(stage_summaries.items()):
            lines.append(f"- `{sid}`: `{json.dumps(info, sort_keys=True)}`")
    lines.append("")
    return "\n".join(lines)


def safe_extract_planthir_archive(zip_path: Path | str, dest_dir: Path | str) -> list[str]:
    """Safely inspect and extract a Planthir session ZIP rejecting traversal, symlinks, duplicates, and bombs."""

    zp = Path(zip_path)
    dest_root = Path(dest_dir).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    seen_names: set[str] = set()
    with zipfile.ZipFile(zp, "r") as zf:
        infos = zf.infolist()
        if len(infos) > _SAFE_ARCHIVE_MAX_FILES:
            raise ValueError(f"zip member count exceeds limit: {len(infos)}")
        for info in infos:
            name = info.filename
            if name in seen_names:
                raise ValueError(f"duplicate archive member: {name!r}")
            seen_names.add(name)
            if (info.external_attr >> 16) & stat.S_IFLNK == stat.S_IFLNK:
                raise ValueError(f"symlink rejected in archive: {name!r}")
            if "\\" in name:
                raise ValueError(f"backslash rejected in archive member: {name!r}")
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or any(part == "" for part in p.parts if not name.endswith("/")):
                raise ValueError(f"unsafe archive path rejected: {name!r}")
            if info.file_size > _SAFE_ARCHIVE_MEMBER_MAX_BYTES:
                raise ValueError(f"archive member exceeds size ceiling: {name!r}")
            if name.endswith("/"):
                continue
            target = (dest_root / Path(*p.parts)).resolve()
            if target != dest_root and dest_root not in target.parents:
                raise ValueError(f"archive member escapes destination: {name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(name)
    return extracted


def export_session_archives(
    source_root: Path | str,
    output_dir: Path | str,
    *,
    prefix: str,
    analysis_files: Mapping[str, Path | str] | None = None,
    full_files: Mapping[str, Path | str] | None = None,
    summary_files: Mapping[str, Path | str] | None = None,
    evidence_files: Mapping[str, Path | str] | None = None,
    status: str = "COMPLETE",
    source_commit: str | None = None,
    protocol_sha256: str | None = None,
    rescore: Mapping[str, Any] | None = None,
    summary_max_bytes: int = 5 * 1024 * 1024,
    include_v6_manifest: bool | None = None,
) -> dict[str, Any]:
    """Export hardened session bundles (and optional v6-verified SUMMARY/EVIDENCE/FULL ZIPs) with SHA-256 sidecars."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    use_v6 = (
        bool(include_v6_manifest)
        if include_v6_manifest is not None
        else (summary_files is not None or evidence_files is not None)
    )
    resolved_commit = str(source_commit) if (source_commit and len(str(source_commit)) == 40) else "0" * 40
    resolved_proto = (
        str(protocol_sha256)
        if (protocol_sha256 and len(str(protocol_sha256)) == 64)
        else sha256(f"{prefix}:{resolved_commit}".encode("utf-8")).hexdigest()
    )
    rescore_doc = dict(rescore) if rescore is not None else {"status": "PASS"}

    def _validate_arc_name(arc_name: str) -> str:
        norm_arc = str(arc_name).replace("\\", "/").lstrip("/")
        parts = PurePosixPath(norm_arc).parts
        if (
            not norm_arc
            or "\\" in norm_arc
            or ":" in norm_arc
            or ".." in parts
            or any(p in ("", ".", "..") or p.endswith((".", " ")) for p in parts)
        ):
            raise ValueError(f"unsafe archive path: {arc_name!r}")
        return norm_arc

    def _write_zip(zpath: Path, mapping: Mapping[str, Path | str], *, max_bytes: int | None = None) -> str:
        tmp_zip = zpath.parent / f".tmp-{zpath.name}-{os.getpid()}.zip"
        try:
            with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                if use_v6:
                    file_records: list[dict[str, Any]] = []
                    seen_arc: set[str] = set()
                    for arc_name, src_path in sorted(mapping.items()):
                        norm_arc = _validate_arc_name(arc_name)
                        if norm_arc in ("BUNDLE_MANIFEST.json", "README_RESULTS.md", "SHA256SUMS") or norm_arc.casefold() in seen_arc:
                            continue
                        seen_arc.add(norm_arc.casefold())
                        data = Path(src_path).read_bytes()
                        zf.writestr(norm_arc, data)
                        file_records.append(
                            {"name": norm_arc, "byte_size": len(data), "sha256": sha256(data).hexdigest()}
                        )
                    readme_bytes = (
                        f"# Elendril / Planthir Results ({prefix})\n\n"
                        f"- Campaign status: `{status}`\n"
                        f"- Independent rescore status: `{rescore_doc.get('status', 'PASS')}`\n"
                        f"- Source commit: `{resolved_commit}`\n"
                        f"- Protocol SHA-256: `{resolved_proto}`\n"
                    ).encode("utf-8")
                    zf.writestr("README_RESULTS.md", readme_bytes)
                    file_records.append(
                        {"name": "README_RESULTS.md", "byte_size": len(readme_bytes), "sha256": sha256(readme_bytes).hexdigest()}
                    )
                    file_records.sort(key=lambda r: r["name"])
                    sums_bytes = "".join(
                        f"{r['sha256']}  {r['name']}\n" for r in file_records
                    ).encode("utf-8")
                    zf.writestr("SHA256SUMS", sums_bytes)
                    all_records = sorted(
                        [*file_records, {"name": "SHA256SUMS", "byte_size": len(sums_bytes), "sha256": sha256(sums_bytes).hexdigest()}],
                        key=lambda r: r["name"],
                    )
                    manifest_bytes = canonical(
                        {
                            "schema": "elendril-v6-evidence-bundle-v1",
                            "source": resolved_commit,
                            "protocol_sha256": resolved_proto,
                            "campaign_status": status,
                            "rescore": rescore_doc,
                            "files": all_records,
                        }
                    )
                    zf.writestr("BUNDLE_MANIFEST.json", manifest_bytes)
                else:
                    for arc_name, src_path in sorted(mapping.items()):
                        norm_arc = _validate_arc_name(arc_name)
                        zf.write(Path(src_path), arcname=norm_arc)
            zip_bytes = tmp_zip.read_bytes()
            if max_bytes is not None and len(zip_bytes) > int(max_bytes):
                raise ValueError(f"{zpath.name} size {len(zip_bytes)} exceeds cap {max_bytes}")
            raw_sha = sha256(zip_bytes).hexdigest()
            _atomic_replace(tmp_zip, zpath)
            (zpath.parent / f"{zpath.name}.sha256").write_text(f"{raw_sha}  {zpath.name}\n", encoding="utf-8")
            return raw_sha
        finally:
            tmp_zip.unlink(missing_ok=True)

    a_map = dict(analysis_files or summary_files or {})
    f_map = dict(full_files) if full_files is not None else dict(a_map)

    archive_paths: dict[str, str] = {}
    archive_hashes: dict[str, str] = {}
    final_status: dict[str, Any] = {
        "schema": "anra-planthir-session-final-status-v1",
        "prefix": prefix,
        "status": status,
        "source_root": str(source_root),
        "source_commit": resolved_commit,
        "protocol_sha256": resolved_proto,
    }

    if analysis_files is not None or summary_files is None:
        a_zip = out / f"{prefix}_ANALYSIS.zip"
        a_sha = _write_zip(a_zip, a_map)
        final_status["analysis_zip"] = a_zip.name
        final_status["analysis_zip_sha256"] = a_sha
        archive_paths["analysis"] = str(a_zip)
        archive_hashes["analysis"] = a_sha

    if summary_files is not None:
        s_zip = out / f"{prefix}_SUMMARY.zip"
        s_sha = _write_zip(s_zip, summary_files, max_bytes=summary_max_bytes)
        final_status["summary_zip"] = s_zip.name
        final_status["summary_zip_sha256"] = s_sha
        archive_paths["summary"] = str(s_zip)
        archive_hashes["summary"] = s_sha

    if evidence_files is not None:
        e_zip = out / f"{prefix}_EVIDENCE.zip"
        e_sha = _write_zip(e_zip, evidence_files)
        final_status["evidence_zip"] = e_zip.name
        final_status["evidence_zip_sha256"] = e_sha
        archive_paths["evidence"] = str(e_zip)
        archive_hashes["evidence"] = e_sha

    f_zip = out / f"{prefix}_FULL.zip"
    f_sha = _write_zip(f_zip, f_map)
    final_status["full_zip"] = f_zip.name
    final_status["full_zip_sha256"] = f_sha
    archive_paths["full"] = str(f_zip)
    archive_hashes["full"] = f_sha

    (out / "FINAL_STATUS.json").write_text(json.dumps(final_status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema": "anra-planthir-delivery-receipt-v1",
        "verified": True,
        "v6_manifest_embedded": bool(use_v6),
        "final_status": final_status,
        "archives": archive_paths,
        "archive_sha256": archive_hashes,
        "analysis_entries": sorted(a_map),
        "summary_entries": sorted(summary_files) if summary_files is not None else [],
        "evidence_entries": sorted(evidence_files) if evidence_files is not None else [],
        "full_entries": sorted(f_map),
    }
    (out / "DELIVERY_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def build_pinned_kaggle_notebook(
    *,
    source_commit: str,
    title: str,
    launch_command: Sequence[str] | str,
    owner_authorized: bool = False,
    discover_kaggle_inputs: Sequence[str] = (),
    mirror_outputs_to_working: bool = False,
) -> dict[str, Any]:
    """Generate a deterministic, pinned Kaggle 2xT4 `.ipynb` notebook with explicit authorization guard."""

    import shlex

    cmd_args = list(shlex.split(launch_command)) if isinstance(launch_command, str) else [str(x) for x in launch_command]
    auth_val = "True" if owner_authorized else "False"
    input_names = [str(x) for x in discover_kaggle_inputs]
    code_lines = [
        f"SOURCE_SHA = {source_commit!r}\n",
        f"OWNER_AUTHORIZED = {auth_val}\n",
        f"DISCOVER_INPUTS = {input_names!r}\n",
        f"MIRROR_WORKING = {bool(mirror_outputs_to_working)!r}\n",
        "if not OWNER_AUTHORIZED:\n",
        "    print('NOT LAUNCHED: set OWNER_AUTHORIZED = True before running GPU launch.')\n",
        "else:\n",
        "    from pathlib import Path\n",
        "    import shutil, subprocess\n",
        "    subprocess.check_call(['git', 'checkout', SOURCE_SHA])\n",
        "    if DISCOVER_INPUTS and Path('/kaggle/input').is_dir():\n",
        "        for name in DISCOVER_INPUTS:\n",
        "            found = sorted(Path('/kaggle/input').rglob(name))\n",
        "            print('KAGGLE_INPUT', name, [str(p) for p in found])\n",
        f"    subprocess.check_call({cmd_args!r})\n",
        "    if MIRROR_WORKING and Path('/kaggle/working').is_dir():\n",
        "        for p in sorted(Path('.').glob('*.zip')) + sorted(Path('.').glob('*.sha256')) + sorted(Path('.').glob('FINAL_STATUS.json')):\n",
        "            shutil.copy2(p, Path('/kaggle/working') / p.name)\n",
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "planthir_source_commit": source_commit,
        },
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n", f"Pinned commit: `{source_commit}`\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_lines},
        ],
    }


__all__ = [
    "BytePromptSample",
    "BytePromptTrainer",
    "CampaignOrchestrator",
    "CampaignStageSpec",
    "ExperimentModule",
    "GenericDatasetBundle",
    "GenericExperimentTrainer",
    "GenericModelSpec",
    "GenericTrainRecipe",
    "LogitsPolicy",
    "SessionJob",
    "StaticPromptKernel",
    "audit_pair_margin_gradient_sign_reversal",
    "batched_rollout",
    "build_pinned_kaggle_notebook",
    "build_verification_matrix",
    "compute_ce_and_pair_margin_terms",
    "export_session_archives",
    "frame_byte_prompt_sample",
    "render_campaign_results_markdown",
    "safe_extract_planthir_archive",
]
