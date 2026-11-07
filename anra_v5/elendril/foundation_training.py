"""Versioned variable-length training beside the immutable experiment control.

One update is atomic in memory: any exception faults the instance. Recovery
always constructs a fresh engine from a committed generation. Publication
decodes the actual readback before LATEST can change. No import launches work.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import random

import torch

from v5_model.core import initialize, packed_layout
from v5_training.checkpoint import CheckpointStore
from v5_training.optimizer import build_adamw_optimizer, optimizer_group_receipt, validate_parameter_ownership
from v5_training.state import CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState, IdentityBindings, TrainingState
from .codec import BOS, EOS, PAD, CODEC_SHA, canonical, digest
from .publication import writer_lock, quarantine_interrupted_publication
from .training import _dump, _load, batch_tensors, runtime_identity
from .representation import LEGACY, tokenizer_from_manifest
from .backend import BackendIdentity, NeuralBackend


@dataclass(frozen=True)
class TokenSample:
    task_id: str
    tokens: tuple[int, ...]
    eligible: tuple[bool, ...]
    world_id: str
    category: str = "ordinary"


@dataclass(frozen=True)
class Recipe:
    seed: int = 7
    total_updates: int = 8
    batch_size: int = 8
    micro_batch_size: int = 2
    lr: float = .003
    clip: float = 1.
    objective: str = "ce"
    action_weight: float = 3.
    sampling: str = "uniform"
    schedule: str = "constant"
    precision: str = "fp32"
    activation_checkpointing: bool = False
    version: str = "elendril-foundation-recipe-v1"

    def validate(self):
        if self.version != "elendril-foundation-recipe-v1":
            raise ValueError("unknown recipe version")
        for name in ("seed", "total_updates", "batch_size", "micro_batch_size"):
            if type(getattr(self, name)) is not int:
                raise ValueError("recipe counters must be integers")
        if not (0 <= self.seed < 2**32 and 1 <= self.total_updates <= 9216
                and 1 <= self.micro_batch_size <= self.batch_size <= 256):
            raise ValueError("recipe exceeds bounded engine limits")
        if any(type(x) not in (float, int) or not math.isfinite(x) or x <= 0
               for x in (self.lr, self.clip, self.action_weight)):
            raise ValueError("recipe scalars must be positive and finite")
        if (self.objective not in ("ce", "action-weighted") or self.sampling not in ("uniform", "uncertainty-first")
                or self.schedule not in ("constant", "cosine") or self.precision not in ("fp32", "fp16")
                or type(self.activation_checkpointing) is not bool):
            raise ValueError("unsupported training policy")


@dataclass(frozen=True)
class StageTransition:
    parent_checkpoint: str
    parent_recipe: str
    child_recipe: str
    child_data: str
    optimizer_policy: str = "reset"
    version: str = "elendril-stage-transition-v1"

    def validate(self):
        if self.version != "elendril-stage-transition-v1" or self.optimizer_policy != "reset":
            raise ValueError("only explicit optimizer reset is qualified for stage migration")
        for value in (self.parent_checkpoint, self.parent_recipe, self.child_recipe, self.child_data):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("transition must bind immutable parent/child identities")


def tensor_inventory(model):
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(model.state_dict().items())}


class FoundationTrainer:
    def __init__(self, samples, *, spec, recipe: Recipe, source_commit: str,
                 root: Path, lineage: str, data_provenance: dict, tokenizer_manifest: dict,
                 device="cpu", resume=False, transition: StageTransition | None = None, parent_store=None):
        recipe.validate()
        spec.assert_valid()
        self.samples = tuple(samples)
        self.recipe, self.spec = recipe, spec
        self.device = torch.device(device)
        if (not self.samples or len(self.samples) > 100_000 or len({s.task_id for s in self.samples}) != len(self.samples)
                or recipe.batch_size > len(self.samples)):
            raise ValueError("empty, duplicate or undersized training dataset")
        if data_provenance.get("split") != "train" or not data_provenance.get("license"):
            raise ValueError("training requires an explicit training split and license provenance")
        self.tokenizer = json.loads(canonical(tokenizer_manifest))
        tokenizer = tokenizer_from_manifest(self.tokenizer)
        tokenizer.require_binding(tokenizer.identity, spec.vocabulary_size)
        self.codec_sha = tokenizer.identity
        self.pad, self.bos, self.eos = tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id
        for sample in self.samples:
            if (not isinstance(sample, TokenSample) or type(sample.tokens) is not tuple or type(sample.eligible) is not tuple or not sample.world_id
                    or not 3 <= len(sample.tokens) <= spec.context_length
                    or len(sample.tokens) != len(sample.eligible)
                    or any(type(x) is not int or not 0 <= x < spec.vocabulary_size for x in sample.tokens)
                    or any(type(x) is not bool for x in sample.eligible)
                    or sample.tokens[0] != self.bos or sample.tokens[-1] != self.eos
                    or self.pad in sample.tokens or self.bos in sample.tokens[1:]
                    or any(x == self.eos for x in sample.tokens[:-1])
                    or sample.eligible[0] or not sample.eligible[-1] or not any(sample.eligible[1:])):
                raise ValueError("invalid training framing, tokens or loss mask")
            start = sample.eligible.index(True)
            if start == len(sample.tokens) - 1 or not all(sample.eligible[start:]):
                raise ValueError("answer span must be contiguous through EOS")
        if self.device.type not in ("cpu", "cuda") or (recipe.precision == "fp16" and self.device.type != "cuda"):
            raise ValueError("FP16 requires a qualified CUDA device")
        if not torch.are_deterministic_algorithms_enabled() or torch.get_num_threads() != 1:
            raise ValueError("configure_device/configure_cpu before constructing trainer")
        self.model = initialize(spec, recipe.seed).to(self.device)
        self.backend = NeuralBackend(self.model, BackendIdentity(spec.sha256(), self.codec_sha,
            spec.vocabulary_size, spec.context_length), pad_id=self.pad, bos_id=self.bos, eos_id=self.eos)
        self.optimizer = build_adamw_optimizer(self.model, lr=recipe.lr)
        self.scaler = torch.amp.GradScaler("cuda", enabled=recipe.precision == "fp16")
        self.rng = torch.Generator().manual_seed(recipe.seed)
        self.data_sha = digest({"samples": [asdict(s) for s in self.samples], "provenance": data_provenance})
        # Explicit finite sampler plan makes variable eligible-token accounting
        # reproducible and prevents an interrupted update consuming an example.
        self.plan = tuple(self.indices(u) for u in range(recipe.total_updates))
        self.counts = tuple(sum(sum(self.samples[i].eligible) for i in batch) for batch in self.plan)
        self.runtime = runtime_identity(self.device)
        self.metadata = {"schema": "elendril-foundation-training-v1", "recipe": asdict(recipe),
                         "model": spec.canonical(), "tensors": tensor_inventory(self.model),
                         "tokenizer": self.tokenizer, "provenance": data_provenance,
                         "runtime": self.runtime, "plan_sha256": digest(self.plan)}
        if transition is not None:
            transition.validate()
            if transition.child_recipe != digest(asdict(recipe)) or transition.child_data != self.data_sha:
                raise ValueError("child stage recipe or data mismatch")
            self.metadata["stage_transition"] = asdict(transition)
        self.metadata = json.loads(canonical(self.metadata))
        identities = IdentityBindings(IDENTITY_SCHEMA, source_commit, spec.sha256(), self.codec_sha,
            self.data_sha, self.data_sha, digest(self.metadata),
            optimizer_group_receipt(self.model, self.optimizer)["sha256"],
            digest([recipe.schedule, recipe.lr, recipe.total_updates]), digest(recipe.sampling),
            sampler_spec_sha256=digest(self.plan))
        self.store = CheckpointStore(root, lineage, payload_validator=self.validate_payloads)
        self.parent = self.store.latest_sha256()
        self.state = TrainingState.initial(lineage_id=lineage, token_budget=sum(self.counts),
            tokens_per_update=max(self.counts), cursor=CursorState(CURSOR_SCHEMA, self.data_sha, 0, 0, 0),
            rng_state_sha256=sha256(self.rng_bytes()).hexdigest(), curriculum_phase="action-formation",
            identities=identities)
        self.faulted = False
        self.published_update = 0
        if resume:
            if self.parent is None:
                raise ValueError("no committed generation to resume")
            state, payloads = self.store.restore(self.parent)
            self.model.load_state_dict(_load(payloads["model.bin"]), strict=True)
            self.optimizer.load_state_dict(_load(payloads["optimizer.bin"]))
            self.scaler.load_state_dict(json.loads(payloads["scheduler.json"])["scaler"])
            self.rng.set_state(torch.tensor(list(payloads["rng.bin"]), dtype=torch.uint8))
            self.state, self.published_update = state, state.global_update
        elif self.parent is not None:
            raise ValueError("existing lineage requires explicit resume")
        elif transition is not None:
            if parent_store is None:
                raise ValueError("stage migration requires a verified parent store")
            self.migrate_stage(parent_store, transition)

    def indices(self, update):
        result = []
        offset = update * self.recipe.batch_size
        while len(result) < self.recipe.batch_size:
            epoch, position = divmod(offset, len(self.samples))
            pool = list(range(len(self.samples)))
            random.Random(int(digest([self.recipe.seed, epoch, "foundation-order-v1"]), 16)).shuffle(pool)
            if self.recipe.sampling == "uncertainty-first" and epoch == 0:
                pool.sort(key=lambda i: self.samples[i].category not in ("missing", "conflict"))
            take = min(self.recipe.batch_size - len(result), len(pool) - position)
            result.extend(pool[position:position + take])
            offset += take
        return tuple(result)

    def rng_bytes(self):
        return bytes(self.rng.get_state().tolist())

    def learning_rate(self, update):
        if self.recipe.schedule == "constant":
            return self.recipe.lr
        return self.recipe.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * update / self.recipe.total_updates)))

    def validate_payloads(self, state, payloads):
        if state.identities != self.state.identities:
            raise ValueError("checkpoint source/model/tokenizer/data/recipe/runtime mismatch")
        u = state.global_update
        expected_cursor = replace(self.state.cursor, sequence_ordinal=u * self.recipe.batch_size)
        if (not 1 <= u <= self.recipe.total_updates or state.cursor != expected_cursor
                or state.cumulative_tokens != sum(self.counts[:u]) or state.token_budget != sum(self.counts)
                or state.tokens_per_update != max(self.counts)
                or sha256(payloads["rng.bin"]).hexdigest() != state.rng_state_sha256):
            raise ValueError("checkpoint RNG, token ledger or next-update cursor mismatch")
        torch.Generator().set_state(torch.tensor(list(payloads["rng.bin"]), dtype=torch.uint8))
        scheduler = json.loads(payloads["scheduler.json"])
        if set(scheduler) != {"metadata", "scaler"} or scheduler["metadata"] != self.metadata:
            raise ValueError("checkpoint metadata/tensor manifest mismatch")
        scaler = torch.amp.GradScaler("cuda", enabled=self.recipe.precision == "fp16")
        scale_state = scheduler["scaler"]
        if self.recipe.precision == "fp32" and scale_state != {}:
            raise ValueError("FP32 cannot restore a nonempty gradient scaler")
        if self.recipe.precision == "fp16":
            if (set(scale_state) != {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in scale_state.values())
                    or scale_state["scale"] <= 0 or scale_state["growth_factor"] <= 1
                    or not 0 < scale_state["backoff_factor"] < 1
                    or type(scale_state["growth_interval"]) is not int or scale_state["growth_interval"] <= 0
                    or type(scale_state["_growth_tracker"]) is not int
                    or not 0 <= scale_state["_growth_tracker"] < scale_state["growth_interval"]):
                raise ValueError("invalid gradient scaler state")
        scaler.load_state_dict(scheduler["scaler"])
        weights, opt = _load(payloads["model.bin"]), _load(payloads["optimizer.bin"])
        if set(weights) != set(self.model.state_dict()):
            raise ValueError("model tensor names mismatch")
        for name, tensor in weights.items():
            expected = self.model.state_dict()[name]
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected.shape or tensor.dtype != expected.dtype
                    or not torch.isfinite(tensor).all()):
                raise ValueError("model tensor shape/dtype/finite validation failed")
        groups = opt.get("param_groups", [])
        expected_groups = self.optimizer.state_dict()["param_groups"]
        if len(groups) != len(expected_groups):
            raise ValueError("optimizer group count mismatch")
        ids = []
        named_params = []
        for group, expected, live in zip(groups, expected_groups, self.optimizer.param_groups):
            if (group["params"] != expected["params"] or group["lr"] != self.learning_rate(u - 1)
                    or {k: v for k, v in group.items() if k not in ("params", "lr")}
                    != {k: v for k, v in expected.items() if k not in ("params", "lr")}):
                raise ValueError("optimizer ownership or settings mismatch")
            ids.extend(group["params"])
            named_params.extend(live["params"])
        if set(opt["state"]) != set(ids):
            raise ValueError("optimizer state ownership mismatch")
        for identity, parameter in zip(ids, named_params):
            slot = opt["state"][identity]
            if (set(slot) != {"step", "exp_avg", "exp_avg_sq"} or not isinstance(slot["step"], torch.Tensor)
                    or slot["step"].numel() != 1 or slot["step"].item() != u):
                raise ValueError("optimizer update counter mismatch")
            for key in ("exp_avg", "exp_avg_sq"):
                value = slot[key]
                if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                        or value.dtype != parameter.dtype or not torch.isfinite(value).all()
                        or (key == "exp_avg_sq" and (value < 0).any())):
                    raise ValueError("optimizer moment shape/dtype/finite mismatch")

    def step(self):
        if self.faulted or self.state.complete:
            raise ValueError("complete or faulted engine requires a fresh restore")
        self.faulted = True
        u = self.state.global_update
        chosen = [self.samples[i] for i in self.plan[u]]
        weights = []
        for sample in chosen:
            row = [float(x) for x in sample.eligible]
            if self.recipe.objective == "action-weighted":
                row[sample.eligible.index(True)] = self.recipe.action_weight
            weights.append(row)
        denominator = sum(map(sum, weights))
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        lr = self.learning_rate(u)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        numerator = 0.
        for start in range(0, len(chosen), self.recipe.micro_batch_size):
            chunk = chosen[start:start + self.recipe.micro_batch_size]
            tokens, segments, eligible = batch_tensors([(s.tokens, s.eligible) for s in chunk], self.device)
            loss_weights = torch.tensor([row + [0.] * (tokens.shape[1] - len(row))
                                        for row in weights[start:start + len(chunk)]], device=self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.recipe.precision == "fp16"):
                logits = self.backend.logits(tokens, segments,
                    activation_checkpointing=self.recipe.activation_checkpointing)
                losses = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    tokens[:, 1:].reshape(-1), reduction="none").reshape_as(tokens[:, 1:])
                value = (losses * loss_weights[:, 1:]).sum()
            if not torch.isfinite(logits).all() or not torch.isfinite(value):
                raise FloatingPointError("nonfinite forward; update rejected")
            self.scaler.scale(value / denominator).backward()
            numerator += value.item()
        self.scaler.unscale_(self.optimizer)
        validate_parameter_ownership(self.model, self.optimizer)
        if any(p.grad is None for p in self.model.parameters()):
            raise ValueError("silent missing gradient")
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe.clip, error_if_nonfinite=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if (len(self.optimizer.state) != len(list(self.model.parameters())) or any(
                slot["step"].item() != u + 1 for slot in self.optimizer.state.values())):
            raise ValueError("optimizer skipped or duplicated the proposed update")
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise FloatingPointError("nonfinite parameter; restore committed parent")
        # No dropout or stochastic forward exists in this decoder. Sampler is
        # index-addressed; own CPU RNG is bound for future stochastic consumers.
        self.state = replace(self.state, generation=u + 1, global_update=u + 1,
            cumulative_tokens=self.state.cumulative_tokens + self.counts[u],
            tokens_by_source={"action": self.state.cumulative_tokens + self.counts[u]},
            optimizer_step_max=u + 1, schedule_tokens=self.state.cumulative_tokens + self.counts[u],
            cursor=replace(self.state.cursor, sequence_ordinal=(u + 1) * self.recipe.batch_size),
            rng_state_sha256=sha256(self.rng_bytes()).hexdigest(), parent_checkpoint_sha256=self.parent)
        self.state.assert_valid()
        self.faulted = False
        return {"update": u + 1, "loss": numerator / denominator, "weighted_denominator": denominator,
                "supervised_tokens": self.counts[u], "input_positions": sum(len(s.tokens) for s in chosen),
                "examples": len(chosen), "microbatches": math.ceil(len(chosen) / self.recipe.micro_batch_size),
                "pre_clip": float(norm), "lr": lr, "indices": self.plan[u]}

    def save(self, *, inject_crash_at=None):
        if self.faulted or self.state.global_update <= self.published_update:
            raise ValueError("cannot publish a failed or repeated update")
        state = replace(self.state, parent_checkpoint_sha256=self.parent)
        payloads = {"model.bin": _dump(self.model.state_dict()), "optimizer.bin": _dump(self.optimizer.state_dict()),
            "rng.bin": self.rng_bytes(), "scheduler.json": canonical({"metadata": self.metadata, "scaler": self.scaler.state_dict()}),
            "cursor.json": canonical(asdict(state.cursor)), "ledger.json": canonical(dict(state.tokens_by_source)),
            "training_state.json": canonical(state.canonical())}
        # Any ambiguous publish requires a fresh session, including after-pointer
        # crashes. Never retry mutation based on an in-memory counter.
        self.faulted = True
        with writer_lock(self.store.lineage_root):
            quarantine_interrupted_publication(self.store.lineage_root, state.generation)
            self.parent = self.store.publish(state=state, payloads=payloads,
                expected_parent_sha256=self.parent, inject_crash_at=inject_crash_at)
        self.published_update, self.state, self.faulted = state.global_update, state, False
        return self.parent

    def migrate_stage(self, parent_store, transition: StageTransition):
        """Explicit fresh lineage, compatible weights, reset AdamW. Never exact resume."""
        transition.validate()
        if self.state.global_update or self.parent is not None:
            raise ValueError("stage migration requires a fresh child")
        parent, payloads = parent_store.restore(transition.parent_checkpoint)
        if (transition.child_recipe != digest(asdict(self.recipe)) or transition.child_data != self.data_sha
                or json.loads(payloads["scheduler.json"])["metadata"]["recipe"] is None
                or digest(json.loads(payloads["scheduler.json"])["metadata"]["recipe"]) != transition.parent_recipe
                or parent.identities.model_spec_sha256 != self.spec.sha256()
                or parent.identities.tokenizer_sha256 != self.codec_sha):
            raise ValueError("stage transition identities or compatible architecture mismatch")
        weights = _load(payloads["model.bin"])
        if set(weights) != set(self.model.state_dict()):
            raise ValueError("migration tensor names differ")
        for name, tensor in weights.items():
            expected = self.model.state_dict()[name]
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected.shape
                    or tensor.dtype != expected.dtype or not torch.isfinite(tensor).all()):
                raise ValueError("migration tensor shape/dtype/finite mismatch")
        self.model.load_state_dict(weights, strict=True)
        # Bind migration provenance into new run identity; lifetime counters reset.
        self.metadata["stage_transition"] = asdict(transition)
        self.state = replace(self.state, identities=replace(self.state.identities, run_spec_sha256=digest(self.metadata)))


def legacy_tokenizer_manifest():
    return LEGACY.manifest.record()
