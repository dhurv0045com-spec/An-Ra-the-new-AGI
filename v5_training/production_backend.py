"""Canonical production training backend: the one real V5 update.

This module closes the gap between the orchestration contracts and real
mathematics.  ``ProductionTrainingBackend.step`` executes the full frozen
chain on a supplied packed batch:

    packed batch -> V5 core -> causal CE -> backward -> global clip 1.0
    -> token-indexed LR -> AdamW step -> mutation certification.

Certification is mechanical, not self-reported: the backend captures live
parameter/optimizer evidence before the update and verifies after it that the
optimizer owns the live model parameters, gradients exist and are finite, the
global pre-clip norm was measured and the post-clip norm is within 1.0,
parameters actually changed, Adam moments actually changed for every
parameter that received gradient, the Adam step counter incremented, the
learning rate equals the token-indexed schedule expectation, and the single
tied embedding table kept its storage identity.  Any failure raises before a
``BackendReport`` can be produced, so a stale optimizer or a disconnected
update cannot advance training state.

The historical core-vnext failure mode -- training metadata advancing while
parameters do not change -- is caught here by parameter/moment hashing, and
by the stale-ownership boundary that rejects an optimizer bound to parameters
that are no longer the live model's (the TPU/XLA device-movement lesson).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .optimizer import validate_parameter_ownership
from .distributed import UPDATE_LOSS_SCHEMA
from .schedule import lr_at, schedule_receipt
from .state import CURSOR_SCHEMA, CursorState
from .step import CLIP_NORM_TOLERANCE, GRAD_CLIP_GLOBAL_L2
from .trainer import BackendReport
from .mutation import update_fp32_tensor_bytes
from v5_model.core import packed_layout, _packed_layout_from_validated_segments
from v5_objectives.causal_lm import causal_lm_loss_from_hidden


BACKEND_SCHEMA = "anra-v5-production-backend-receipt/v3"

EXECUTABLE_RUNTIMES = ("cpu", "cuda")


def _runtime_of(device: Any) -> str:
    if device is None:
        return "cpu"
    return str(getattr(device, "type", device))


def precision_receipt(*, runtime: str, torch_module: Any = None) -> dict[str, object]:
    """Frozen precision contract for a runtime, without executing anything.

    Persistent/master parameters are FP32, compute is BF16 autocast on CUDA
    only, and loss/reductions/moments are FP32. Anything outside {"cpu",
    "cuda"} is NOT certified here: XLA/TPU dtype behavior requires PRE500M
    evidence and fails closed as TPU_EVIDENCE_REQUIRED.
    """

    if runtime not in EXECUTABLE_RUNTIMES:
        return {
            "schema": "anra-v5-precision-contract/v1",
            "runtime": runtime,
            "status": "TPU_EVIDENCE_REQUIRED",
            "reason": "XLA collectives, memory fit, and bf16 execution need PRE500M certification",
        }
    compute = "bfloat16-autocast" if runtime == "cuda" else "float32"
    return {
        "schema": "anra-v5-precision-contract/v1",
        "runtime": runtime,
        "persistent_parameters": "float32",
        "compute": compute,
        "logits_loss_reductions": "float32",
        "global_gradient_norm": "float32 replica-global",
        "optimizer_moments": "float32",
        "persistent_bfloat16_shadow": False,
        "loss_scaler": None,
        "status": "CERTIFIED_LOCAL",
    }



class StaleOptimizerOwnership(ValueError):
    """Optimizer no longer owns the live model parameters (core-vnext lesson)."""


def bounded_warmup_schedule(*, peak_learning_rate: float, warmup_tokens: int = 0):
    """Pure canary/testing schedule: linear warmup to a constant peak.

    The canonical 5B-token WSD schedule (``v5_training.schedule.lr_at``) yields
    underflow-level learning rates at canary token counts, so bounded canaries
    bind this schedule instead and record its identity through
    ``IdentityBindings.schedule_spec_sha256``.  The production default remains
    the canonical schedule.
    """

    if peak_learning_rate <= 0:
        raise ValueError("peak learning rate must be positive")
    if warmup_tokens < 0:
        raise ValueError("warmup tokens cannot be negative")

    def schedule(cumulative_tokens: int) -> float:
        if warmup_tokens == 0:
            return float(peak_learning_rate)
        if cumulative_tokens < 0:
            raise ValueError("cumulative tokens cannot be negative")
        return float(
            peak_learning_rate * min(1.0, cumulative_tokens / warmup_tokens)
        )

    return schedule


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _validate_loss_aggregate(
    aggregate: Mapping[str, object],
    *,
    state: Any,
    planned_total: int,
    expected_local_tokens: int,
    expected_local_numerator: float,
    expected_rank: int,
) -> dict[str, object]:
    required = {
        "schema", "status", "world_size", "global_update", "global_tokens",
        "loss_numerator", "global_loss", "rank_contributions", "receipt_sha256",
    }
    if set(aggregate) != required:
        raise ValueError("distributed loss aggregate fields do not match schema")
    if aggregate.get("schema") != UPDATE_LOSS_SCHEMA or aggregate.get("status") != "READY":
        raise ValueError("distributed loss aggregate is not a ready update-loss receipt")
    world_size = aggregate.get("world_size")
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("distributed loss aggregate world size is invalid")
    if (aggregate.get("global_update") != state.global_update + 1
            or aggregate.get("global_tokens") != planned_total):
        raise ValueError("distributed loss aggregate update or denominator disagrees")
    numerator = aggregate.get("loss_numerator")
    mean = aggregate.get("global_loss")
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, (int, float))
        or not math.isfinite(float(numerator))
        or float(numerator) < 0.0
        or isinstance(mean, bool)
        or not isinstance(mean, (int, float))
        or not math.isfinite(float(mean))
        or not math.isclose(
            float(mean), float(numerator) / planned_total,
            rel_tol=1e-12, abs_tol=1e-12,
        )
    ):
        raise ValueError("distributed loss aggregate mean or numerator is invalid")
    contributions = aggregate.get("rank_contributions")
    if not isinstance(contributions, list) or len(contributions) != world_size:
        raise ValueError("distributed loss aggregate has an incomplete rank list")
    ordered = sorted(contributions, key=lambda item: item.get("rank", -1)
                     if isinstance(item, Mapping) else -1)
    if [item.get("rank") if isinstance(item, Mapping) else None for item in ordered] != list(
        range(world_size)
    ):
        raise ValueError("distributed loss aggregate must contain each rank exactly once")
    if any(not isinstance(item, Mapping)
           or set(item) != {"rank", "eligible_tokens", "loss_numerator"}
           or type(item.get("eligible_tokens")) is not int
           or item["eligible_tokens"] < 0
           or isinstance(item.get("loss_numerator"), bool)
           or not isinstance(item.get("loss_numerator"), (int, float))
           or not math.isfinite(float(item["loss_numerator"]))
           or float(item["loss_numerator"]) < 0.0 for item in ordered):
        raise ValueError("distributed loss aggregate contains an invalid rank contribution")
    if any(item["eligible_tokens"] == 0
           and float(item["loss_numerator"]) != 0.0 for item in ordered):
        raise ValueError("zero-token rank contribution must have a zero loss numerator")
    if sum(item["eligible_tokens"] for item in ordered) != planned_total:
        raise ValueError("distributed loss aggregate rank counts do not equal global total")
    numerator_sum = sum(float(item["loss_numerator"]) for item in ordered)
    if not math.isclose(numerator_sum, float(numerator), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("distributed loss aggregate numerator disagrees with rank values")
    local = ordered[expected_rank] if 0 <= expected_rank < world_size else None
    if local is None or local["eligible_tokens"] != expected_local_tokens:
        raise ValueError("distributed loss aggregate local count disagrees with this rank")
    if float(local["loss_numerator"]) != expected_local_numerator:
        raise ValueError("distributed loss aggregate local numerator disagrees with this rank")
    receipt_sha256 = aggregate.get("receipt_sha256")
    unsigned = {key: value for key, value in aggregate.items() if key != "receipt_sha256"}
    if (
        not isinstance(receipt_sha256, str)
        or len(receipt_sha256) != 64
        or any(char not in "0123456789abcdef" for char in receipt_sha256)
        or hashlib.sha256(_canonical_json(unsigned)).hexdigest() != receipt_sha256
    ):
        raise ValueError("distributed loss aggregate receipt hash is invalid")
    return dict(aggregate)


def _tensor_sha256(value: Any, torch: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    update_fp32_tensor_bytes(digest, value, torch_module=torch)
    return digest.hexdigest()


def _rng_state_sha256(torch: Any) -> str:
    digest = hashlib.sha256()
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PackedBatch:
    """One deterministic packed batch plus its exact ledger bookkeeping."""

    tokens: Any
    segment_ids: Any
    tokens_by_source: Mapping[str, int]
    cursor: CursorState
    rng_state_sha256: str


@dataclass(frozen=True, slots=True)
class UpdateEvidence:
    """Hash-bound snapshot of live model/optimizer state."""

    parameter_sha256: str
    moment_sha256: str
    optimizer_steps: dict[str, int]
    embedding_identity: str
    owned_parameter_ids: frozenset[int]

    def as_receipt(self) -> dict[str, object]:
        return {
            "parameter_sha256": self.parameter_sha256,
            "moment_sha256": self.moment_sha256,
            "optimizer_steps": dict(sorted(self.optimizer_steps.items())),
            "embedding_identity": self.embedding_identity,
        }


def _embedding_weight(model: Any) -> Any:
    names = [name for name, _ in model.named_parameters() if name.endswith("embedding.weight")]
    if len(names) != 1:
        raise ValueError("backend requires exactly one tied embedding table")
    return dict(model.named_parameters())[names[0]]


def _embedding_identity(parameter: Any) -> str:
    """Return a run-local storage/object identity without XLA data_ptr calls."""

    if _runtime_of(getattr(parameter, "device", None)) == "xla":
        # Lazy XLA tensors do not expose a stable host storage pointer. The
        # model contract has one shared embedding Parameter object, so object
        # identity is the meaningful in-process invariant on this runtime.
        return f"xla-parameter-object:{id(parameter)}"
    return f"storage-pointer:{parameter.data_ptr()}"


def capture_evidence(model: Any, optimizer: Any, *, torch: Any) -> UpdateEvidence:
    """Capture live parameter/optimizer identities and hashes."""

    steps: dict[str, int] = {}
    tensor_step_names: list[str] = []
    tensor_steps: list[Any] = []
    moment_digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter, {})
        step = state.get("step")
        if torch.is_tensor(step):
            if step.numel() != 1:
                raise ValueError(f"optimizer step for {name} must be scalar")
            tensor_step_names.append(name)
            tensor_steps.append(step.reshape(()))
        else:
            steps[name] = int(step or 0)
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if torch.is_tensor(value):
                moment_digest.update(name.encode("utf-8"))
                moment_digest.update(key.encode("utf-8"))
                moment_digest.update(_tensor_sha256(value, torch).encode("ascii"))
    for name, value in zip(tensor_step_names, _batched_scalar_values(tensor_steps, torch)):
        steps[name] = int(value)
    embedding = _embedding_weight(model)
    return UpdateEvidence(
        parameter_sha256=_model_sha256(model, torch=torch),
        moment_sha256=moment_digest.hexdigest(),
        optimizer_steps=steps,
        embedding_identity=_embedding_identity(embedding),
        owned_parameter_ids=frozenset(
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        ),
    )


def _model_sha256(model: Any, *, torch: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_sha256(parameter, torch).encode("ascii"))
    return digest.hexdigest()


def assert_live_ownership(model: Any, optimizer: Any) -> None:
    """Require optimizer params to be the live model parameter objects."""

    try:
        validate_parameter_ownership(model, optimizer)
    except ValueError as exc:
        raise StaleOptimizerOwnership(str(exc)) from exc
    parameters = list(model.parameters())
    runtimes = {_runtime_of(getattr(parameter, "device", None)) for parameter in parameters}
    if len(runtimes) > 1:
        raise StaleOptimizerOwnership("model parameters span multiple device runtimes")
    runtime = next(iter(runtimes), "cpu")
    live_ids = {id(parameter) for parameter in parameters}
    live_ptrs = ({id(parameter): parameter.data_ptr() for parameter in parameters}
                 if runtime != "xla" else {})
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in live_ids:
                raise StaleOptimizerOwnership(
                    "stale optimizer ownership: an optimizer parameter is not a live model "
                    "parameter (historical core-vnext device-movement failure)"
                )
            if runtime != "xla" and parameter.data_ptr() != live_ptrs[id(parameter)]:
                raise StaleOptimizerOwnership(
                    "stale optimizer ownership: optimizer parameter storage diverged from the model"
                )
            if not parameter.requires_grad:
                raise StaleOptimizerOwnership("optimizer owns a frozen parameter")


def _global_norm(graduates: list[Any], torch: Any) -> float:
    total = torch.zeros((), dtype=torch.float32, device=graduates[0].device)
    for gradient in graduates:
        total = total + torch.linalg.vector_norm(gradient.detach().float()) ** 2
    return float(torch.sqrt(total).item())


def _batched_scalar_values(values: list[Any], torch: Any) -> list[Any]:
    """Resolve scalar tensors with one host transfer per device/dtype group."""

    results: list[Any | None] = [None] * len(values)
    by_device_dtype: dict[tuple[str, str], list[tuple[int, Any]]] = {}
    for index, value in enumerate(values):
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError("batched update checks must be scalar values")
            key = (str(value.device), str(value.dtype))
            by_device_dtype.setdefault(key, []).append((index, value.reshape(())))
        else:
            results[index] = value
    for checks in by_device_dtype.values():
        indices, tensors = zip(*checks)
        host_values = torch.stack(list(tensors)).detach().cpu().tolist()
        for index, value in zip(indices, host_values):
            results[index] = value
    if any(value is None for value in results):
        raise ValueError("batched update checks did not resolve every result")
    return results


def _batched_boolean_values(values: list[Any], torch: Any) -> list[bool]:
    """Resolve scalar checks with one host transfer per device/dtype group."""

    return [bool(value) for value in _batched_scalar_values(values, torch)]


def certify_real_update(
    *,
    model: Any,
    optimizer: Any,
    before: UpdateEvidence,
    after: UpdateEvidence,
    expected_learning_rate: float,
    supervised_tokens: int,
    loss: float,
    loss_scope: str = "GLOBAL_BATCH_MEAN",
    loss_numerator: float | None = None,
    loss_denominator: int | None = None,
    loss_aggregation: Mapping[str, object] | None = None,
    grad_norm_pre_clip: float,
    grad_norm_post_clip: float,
    torch: Any,
) -> dict[str, object]:
    """Mechanically prove one real parameter/optimizer mutation happened."""

    assert_live_ownership(model, optimizer)
    if supervised_tokens <= 0:
        raise ValueError("abort NO_SUPERVISED_TOKENS: update carried no eligible targets")
    if not math.isfinite(loss):
        raise ValueError("abort NONFINITE_LOSS")
    if loss_scope not in {"GLOBAL_BATCH_MEAN", "RANK_LOCAL_CONTRIBUTION"}:
        raise ValueError("abort INVALID_LOSS_SCOPE")
    if loss_denominator is None:
        loss_denominator = supervised_tokens
    if loss_numerator is None:
        loss_numerator = loss * loss_denominator
    if (
        type(loss_denominator) is not int
        or loss_denominator != supervised_tokens
        or not math.isfinite(loss_numerator)
        or loss_numerator < 0.0
        or not math.isclose(
            loss,
            loss_numerator / loss_denominator,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("abort INVALID_LOSS_ACCOUNTING")
    if not math.isfinite(grad_norm_pre_clip) or not math.isfinite(grad_norm_post_clip):
        raise ValueError("abort NONFINITE_GRADIENT")
    if grad_norm_post_clip > GRAD_CLIP_GLOBAL_L2 + CLIP_NORM_TOLERANCE:
        raise ValueError(
            f"abort CLIP_BREACH: post-clip global norm {grad_norm_post_clip} exceeds 1.0"
        )
    if (
        expected_learning_rate > 0.0
        and before.parameter_sha256 == after.parameter_sha256
    ):
        raise ValueError(
            "abort PARAMETERS_UNCHANGED: metadata advanced but parameter bytes did not "
            "(historical core-vnext failure mode)"
        )
    validation_events: list[tuple[str, str, Any]] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            validation_events.append(("missing_gradient", name, False))
            continue
        validation_events.append(("gradient", name, torch.isfinite(gradient).all()))
        expected_step = before.optimizer_steps[name] + 1
        state = optimizer.state.get(parameter, {})
        step = state.get("step")
        if torch.is_tensor(step):
            step_check = step.eq(expected_step) if step.numel() == 1 else False
        else:
            step_check = int(step or 0) == expected_step
        validation_events.append(("step", name, step_check))
        observed_step = after.optimizer_steps.get(name)
        validation_events.append((
            "step", name,
            type(observed_step) is int and observed_step == expected_step,
        ))
        moments = (state.get("exp_avg"), state.get("exp_avg_sq"))
        if not all(torch.is_tensor(moment) for moment in moments):
            validation_events.append(("missing_moments", name, False))
            continue
        for moment_name, moment in zip(("exp_avg", "exp_avg_sq"), moments):
            validation_events.append((
                "moment", f"{name} {moment_name}", torch.isfinite(moment).all(),
            ))
    validation_results = _batched_boolean_values(
        [event[2] for event in validation_events], torch,
    )
    for (kind, name, _), valid in zip(validation_events, validation_results):
        if valid:
            continue
        if kind == "missing_gradient":
            raise ValueError(f"abort NO_GRADIENT: {name} received no gradient")
        if kind == "gradient":
            raise ValueError(f"abort NONFINITE_GRADIENT: {name} gradient is not finite")
        if kind == "step":
            raise ValueError(
                f"abort OPTIMIZER_NOT_STEPPED: {name} Adam step did not advance exactly once"
            )
        if kind == "missing_moments":
            raise ValueError(f"abort MOMENTS_MISSING: {name} lacks Adam moments after step")
        if kind == "moment":
            tensor_name = name.rsplit(" ", 1)[0]
            raise ValueError(f"abort NONFINITE_MOMENT: {tensor_name} moment is not finite")
    if after.moment_sha256 == before.moment_sha256:
        raise ValueError(
            "abort MOMENTS_UNCHANGED: Adam first/second moments did not mutate this update"
        )
    learning_rates = {float(group["lr"]) for group in optimizer.param_groups}
    if learning_rates != {float(expected_learning_rate)}:
        raise ValueError(
            "abort SCHEDULE_MISMATCH: optimizer LR does not equal the token-indexed schedule"
        )
    embedding = _embedding_weight(model)
    if _embedding_identity(embedding) != before.embedding_identity:
        raise ValueError("abort TIED_WEIGHT_BROKEN: embedding storage identity changed")
    receipt: dict[str, object] = {
        "schema": BACKEND_SCHEMA,
        "loss": loss,
        "loss_scope": loss_scope,
        "loss_numerator": loss_numerator,
        "loss_denominator": loss_denominator,
        "loss_aggregation": (
            None if loss_aggregation is None else dict(loss_aggregation)
        ),
        "supervised_tokens": supervised_tokens,
        "grad_norm_pre_clip": grad_norm_pre_clip,
        "grad_norm_post_clip": grad_norm_post_clip,
        "learning_rate": float(expected_learning_rate),
        "before": before.as_receipt(),
        "after": after.as_receipt(),
        "parameter_sha256_changed": True,
        "moments_changed": True,
        "tied_preserved": True,
        "schedule_index": "pre-update cumulative real non-padding tokens",
    }
    receipt["sha256"] = hashlib.sha256(_canonical_json(receipt)).hexdigest()
    return receipt


class ProductionTrainingBackend:
    """Owns the live model/optimizer and executes one certified update per call."""

    def __init__(
        self,
        *,
        model: Any,
        optimizer: Any,
        bos_id: int,
        pad_id: int,
        device: Any | None = None,
        bfloat16_autocast: bool = False,
        schedule: Any = None,
        torch_module: Any = None,
        activation_checkpointing: bool = True,
        allow_unqualified_xla: bool = False,
    ) -> None:
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.model = model
        self.optimizer = optimizer
        self.bos_id = int(bos_id)
        self.pad_id = int(pad_id)
        self.device = device
        self.bfloat16_autocast = bool(bfloat16_autocast)
        self.schedule = schedule if schedule is not None else lr_at
        self.activation_checkpointing = bool(activation_checkpointing)
        self.allow_unqualified_xla = bool(allow_unqualified_xla)
        if not callable(self.schedule):
            raise ValueError("schedule must map cumulative tokens to a learning rate")
        if self.bfloat16_autocast and not hasattr(self.torch, "autocast"):
            raise ValueError("bfloat16 autocast requires a framework with autocast support")
        self.runtime = _runtime_of(device)
        if self.runtime not in EXECUTABLE_RUNTIMES and not (
                self.runtime == "xla" and self.allow_unqualified_xla):
            raise ValueError(
                f"runtime {self.runtime!r} is not locally executable: XLA/TPU dtype "
                "behavior needs PRE500M certification (TPU_EVIDENCE_REQUIRED)"
            )
        if self.allow_unqualified_xla and self.runtime != "xla":
            raise ValueError("unqualified XLA mode is valid only on the XLA runtime")
        if self.bfloat16_autocast and self.runtime not in ("cuda", "xla"):
            raise ValueError("bfloat16 autocast is valid only on CUDA or XLA")
        assert_live_ownership(self.model, self.optimizer)
        self.last_receipt: dict[str, object] | None = None
        self._uncertified_optimizer_step = False

    def _assert_update_ready(self) -> None:
        if self._uncertified_optimizer_step:
            raise RuntimeError(
                "backend has an optimizer step without a successful update certificate; "
                "restore the checkpointed model and optimizer before training"
            )

    def _begin_uncertified_optimizer_step(self) -> None:
        # From this point the optimizer may mutate only part of its state before
        # failing. Do not permit a later update or checkpoint to hide that gap.
        self._uncertified_optimizer_step = True
        self.last_receipt = None

    # -- batch validation --------------------------------------------------
    def _validate_batch(self, batch: PackedBatch) -> tuple[int, int]:
        torch = self.torch
        if batch.tokens.ndim != 2 or batch.segment_ids.shape != batch.tokens.shape:
            raise ValueError("batch tensors must be rank-two and identically shaped")
        if batch.tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("batch tokens must be integer token ids")
        if int(batch.tokens.min().item()) < 0 or int(batch.tokens.max().item()) >= int(
            self.model.spec.vocabulary_size
        ):
            raise ValueError("batch token ids fall outside the model vocabulary")
        if sum(batch.tokens_by_source.values()) <= 0:
            raise ValueError("batch ledger must consume a positive token count")
        batch_cursor = batch.cursor
        if batch_cursor.schema != CURSOR_SCHEMA:
            raise ValueError("batch cursor schema is not the frozen pack-cursor schema")
        return batch.tokens.shape[0], batch.tokens.shape[1]

    # -- the one real update ----------------------------------------------
    def step(self, state: Any, batch: PackedBatch) -> BackendReport:
        """Execute and certify one real optimizer update for ``state``."""

        self._assert_update_ready()
        torch = self.torch
        assert_live_ownership(self.model, self.optimizer)
        self._validate_batch(batch)
        expected_lr = float(self.schedule(cumulative_tokens=int(state.cumulative_tokens)))
        for group in self.optimizer.param_groups:
            group["lr"] = float(expected_lr)
        self.optimizer.zero_grad(set_to_none=True)

        before = capture_evidence(self.model, self.optimizer, torch=torch)
        tokens = batch.tokens if self.device is None else batch.tokens.to(self.device)
        segment_ids = (
            batch.segment_ids if self.device is None else batch.segment_ids.to(self.device)
        )
        length = tokens.shape[1]
        positions, mask = packed_layout(segment_ids, torch_module=torch)
        mask = mask.to(tokens.device)
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        autocast = (
            torch.autocast(device_type=self.runtime, dtype=torch.bfloat16)
            if self.bfloat16_autocast else nullcontext()
        )
        with autocast:
            hidden = self.model.forward_hidden(
                tokens, positions, mask,
                use_activation_checkpointing=self.activation_checkpointing,
            )
            loss, supervised_tokens = causal_lm_loss_from_hidden(
                hidden, self.model.embedding.weight, tokens, segment_ids,
                bos_id=self.bos_id, pad_id=self.pad_id, torch_module=torch,
            )
        loss.backward()
        trainable = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise ValueError("model has no trainable parameters")
        grad_norm_pre_clip = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_GLOBAL_L2)
        )
        grad_norm_post_clip = _global_norm(
            [parameter.grad for parameter in trainable], torch
        )
        self._begin_uncertified_optimizer_step()
        self.optimizer.step()
        self._mark_xla_step()

        loss_value = float(loss.detach().item())
        after = capture_evidence(self.model, self.optimizer, torch=torch)
        receipt = certify_real_update(
            model=self.model,
            optimizer=self.optimizer,
            before=before,
            after=after,
            expected_learning_rate=expected_lr,
            supervised_tokens=supervised_tokens,
            loss=loss_value,
            grad_norm_pre_clip=grad_norm_pre_clip,
            grad_norm_post_clip=grad_norm_post_clip,
            torch=torch,
        )
        receipt["rng_state_sha256"] = _rng_state_sha256(torch)
        receipt["consumed_real_tokens"] = self._real_tokens(batch)
        # Certification is fail-closed on nonfinite loss, gradients, and Adam
        # moments. Rechecking here would add one device scalar readback per grad.
        report = BackendReport(
            tokens_by_source=dict(batch.tokens_by_source),
            cursor=batch.cursor,
            rng_state_sha256=receipt["rng_state_sha256"],
            loss_finite=True,
            grad_finite=True,
            grad_norm_post_clip=float(grad_norm_post_clip),
            tied_preserved=True,
        )
        self.last_receipt = receipt
        self._uncertified_optimizer_step = False
        return report

    # -- accumulated optimizer transaction ------------------------------
    # One logical optimizer update = N accumulation microsteps sharing one
    # global eligible-token denominator. Gradients accumulate as exact
    # token-weighted sums: microstep i contributes (n_i / N) * mean_i, which
    # equals sum_i / N, the frozen replica-global mean. A single clip, a
    # single optimizer.step(), and a single TrainingState.advance() happen
    # at the accumulation boundary only.
    def begin_update(self, state: Any) -> dict[str, Any]:
        """Open one logical update: set token-indexed LR, zero grads, capture before."""

        self._assert_update_ready()
        torch = self.torch
        assert_live_ownership(self.model, self.optimizer)
        expected_lr = float(self.schedule(cumulative_tokens=int(state.cumulative_tokens)))
        for group in self.optimizer.param_groups:
            group["lr"] = float(expected_lr)
        self.optimizer.zero_grad(set_to_none=True)
        before = capture_evidence(self.model, self.optimizer, torch=torch)
        return {
            "learning_rate": expected_lr,
            "before": before,
            "loss_numerators": [],
            "eligible_counts": [],
            "tokens_by_source": {},
            "microsteps": 0,
        }

    def _forward_microstep(
        self, tokens: Any, segment_ids: Any, *, segment_layout_prevalidated: bool = False,
    ) -> Any:
        torch = self.torch
        if segment_layout_prevalidated:
            positions, mask = _packed_layout_from_validated_segments(
                segment_ids, torch_module=torch,
            )
        else:
            positions, mask = packed_layout(segment_ids, torch_module=torch)
        mask = mask.to(tokens.device)
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        return self.model.forward_hidden(
            tokens, positions, mask,
            use_activation_checkpointing=self.activation_checkpointing,
        )

    def accumulate_microstep(
        self,
        ctx: dict[str, Any],
        *,
        tokens: Any,
        segment_ids: Any,
        eligible: Any,
        tokens_by_source: Mapping[str, int],
        planned_total: int,
        segment_layout_prevalidated: bool = False,
    ) -> dict[str, Any]:
        """Forward one microstep and backward its exact share of the global mean."""

        self._assert_update_ready()
        torch = self.torch
        if tokens.ndim != 2 or segment_ids.shape != tokens.shape:
            raise ValueError("microstep tensors must be rank-two and identically shaped")
        if eligible.shape != tokens.shape:
            raise ValueError("eligibility mask must match microstep token shape")
        if planned_total <= 0:
            raise ValueError("planned update total must be positive")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("microstep tokens must be integer token ids")
        if type(segment_layout_prevalidated) is not bool:
            raise ValueError("segment layout validation marker must be boolean")
        autocast = (
            torch.autocast(device_type=self.runtime, dtype=torch.bfloat16)
            if self.bfloat16_autocast else nullcontext()
        )
        with autocast:
            hidden = self._forward_microstep(
                tokens, segment_ids,
                segment_layout_prevalidated=segment_layout_prevalidated,
            )
            loss, supervised, loss_numerator = causal_lm_loss_from_hidden(
                hidden, self.model.embedding.weight, tokens, segment_ids,
                bos_id=self.bos_id, pad_id=self.pad_id, eligible=eligible,
                return_numerator=True, allow_empty=True, torch_module=torch,
            )
        loss_numerator_value = float(loss_numerator.detach().item())
        if not math.isfinite(loss_numerator_value) or loss_numerator_value < 0.0:
            raise ValueError("abort NONFINITE_LOSS: microstep loss is not finite")
        scale = supervised / planned_total
        (loss * scale).backward()
        merged = dict(ctx["tokens_by_source"])
        for source, count in tokens_by_source.items():
            if not source or count < 0:
                raise ValueError("microstep ledger requires names and nonnegative counts")
            merged[source] = merged.get(source, 0) + int(count)
        return {
            "learning_rate": ctx["learning_rate"],
            "before": ctx["before"],
            "loss_numerators": [*ctx["loss_numerators"], loss_numerator_value],
            "eligible_counts": [*ctx["eligible_counts"], supervised],
            "tokens_by_source": merged,
            "microsteps": ctx["microsteps"] + 1,
        }

    @staticmethod
    def validate_local_supervision(
        ctx: Mapping[str, Any], expected_local_tokens: int, *, allow_zero: bool = False,
    ) -> None:
        """Check rank-local loss counts before replicas enter gradient reduction."""

        if type(allow_zero) is not bool:
            raise ValueError("allow-zero supervision marker must be boolean")
        minimum = 0 if allow_zero else 1
        if type(expected_local_tokens) is not int or expected_local_tokens < minimum:
            requirement = "nonnegative" if allow_zero else "positive"
            raise ValueError(
                f"expected local supervised-token count must be {requirement}"
            )
        eligible_total = sum(ctx["eligible_counts"])
        if eligible_total != expected_local_tokens:
            raise ValueError(
                f"accumulated {eligible_total} rank-local eligible tokens but expected "
                f"{expected_local_tokens}"
            )

    def finish_update(
        self,
        state: Any,
        ctx: dict[str, Any],
        *,
        planned_total: int,
        expected_local_tokens: int | None = None,
        allow_zero_local_tokens: bool = False,
        loss_aggregate: Mapping[str, object] | None = None,
        loss_rank: int | None = None,
        cursor: Any,
    ) -> BackendReport:
        """Close one logical update: global clip, single step, certified report.

        The post-step RNG digest is captured inside this boundary from the
        live framework state, never supplied by the caller: a caller-provided
        digest could disagree with the actual resume bytes.
        """

        self._assert_update_ready()
        torch = self.torch
        if ctx["microsteps"] <= 0:
            raise ValueError("cannot finish an update with no microsteps")
        if expected_local_tokens is None:
            expected_local_tokens = planned_total
        if type(allow_zero_local_tokens) is not bool:
            raise ValueError("allow-zero local-token marker must be boolean")
        self.validate_local_supervision(
            ctx,
            expected_local_tokens,
            allow_zero=(allow_zero_local_tokens or loss_aggregate is not None),
        )
        local_loss_numerator = sum(ctx["loss_numerators"])
        if loss_aggregate is None:
            if loss_rank is not None:
                raise ValueError("loss rank requires a distributed loss aggregate")
            loss_numerator = local_loss_numerator
            loss_value = loss_numerator / planned_total
            loss_scope = (
                "GLOBAL_BATCH_MEAN"
                if expected_local_tokens == planned_total
                else "RANK_LOCAL_CONTRIBUTION"
            )
            validated_loss_aggregate = None
        else:
            if type(loss_rank) is not int:
                raise ValueError("distributed loss aggregate requires this rank identity")
            validated_loss_aggregate = _validate_loss_aggregate(
                loss_aggregate,
                state=state,
                planned_total=planned_total,
                expected_local_tokens=expected_local_tokens,
                expected_local_numerator=local_loss_numerator,
                expected_rank=loss_rank,
            )
            loss_numerator = float(validated_loss_aggregate["loss_numerator"])
            loss_value = float(validated_loss_aggregate["global_loss"])
            loss_scope = "GLOBAL_BATCH_MEAN"
        trainable = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise ValueError("model has no trainable parameters")
        grad_norm_pre_clip = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_GLOBAL_L2)
        )
        grad_norm_post_clip = _global_norm(
            [parameter.grad for parameter in trainable], torch
        )
        self._begin_uncertified_optimizer_step()
        self.optimizer.step()
        self._mark_xla_step()
        after = capture_evidence(self.model, self.optimizer, torch=torch)
        receipt = certify_real_update(
            model=self.model,
            optimizer=self.optimizer,
            before=ctx["before"],
            after=after,
            expected_learning_rate=float(ctx["learning_rate"]),
            supervised_tokens=planned_total,
            loss=loss_value,
            loss_scope=loss_scope,
            loss_numerator=loss_numerator,
            loss_denominator=planned_total,
            loss_aggregation=validated_loss_aggregate,
            grad_norm_pre_clip=grad_norm_pre_clip,
            grad_norm_post_clip=grad_norm_post_clip,
            torch=torch,
        )
        receipt["rng_state_sha256"] = _rng_state_sha256(torch)
        receipt["consumed_real_tokens"] = planned_total
        receipt["microsteps"] = ctx["microsteps"]
        # A report exists only after certification checked every gradient.
        report = BackendReport(
            tokens_by_source=dict(ctx["tokens_by_source"]),
            cursor=cursor,
            rng_state_sha256=receipt["rng_state_sha256"],
            loss_finite=True,
            grad_finite=True,
            grad_norm_post_clip=float(grad_norm_post_clip),
            tied_preserved=True,
        )
        self.last_receipt = receipt
        self._uncertified_optimizer_step = False
        return report

    def _real_tokens(self, batch: PackedBatch) -> int:
        torch = self.torch
        if batch.segment_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("segment ids must be integer")
        nonpad = int((batch.tokens != self.pad_id).sum().item())
        valid_segments = int((batch.segment_ids >= 0).sum().item())
        if nonpad != valid_segments:
            raise ValueError("batch padding and segment validity disagree")
        ledger_total = sum(batch.tokens_by_source.values())
        if ledger_total != nonpad:
            raise ValueError(
                f"batch ledger claims {ledger_total} real tokens but batch carries {nonpad}"
            )
        return nonpad

    def _mark_xla_step(self) -> None:
        """Commit the lazy XLA graph after each logical optimizer update."""

        if self.runtime != "xla":
            return
        try:
            from torch_xla.core import xla_model as xm
        except ImportError as exc:
            raise RuntimeError("XLA development execution requires torch_xla") from exc
        mark_step = getattr(xm, "mark_step", None)
        if not callable(mark_step):
            raise RuntimeError("XLA runtime lacks mark_step for update boundaries")
        mark_step()


def production_payloads(
    backend: ProductionTrainingBackend, *, state: Any
) -> dict[str, bytes]:
    """Serialize the exact resume inventory from the live production objects."""

    import io

    torch = backend.torch
    payloads = production_shared_payloads(backend)
    buffer = io.BytesIO()
    rng_state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng_state, buffer)
    rng_bin = buffer.getvalue()
    payloads.update({
        "rng.bin": rng_bin,
        "cursor.json": _canonical_json(asdict(state.cursor)),
        "ledger.json": _canonical_json(dict(state.tokens_by_source)),
        "training_state.json": _canonical_json(state.canonical()),
    })
    return payloads


def _cpu_checkpoint_tree(value: Any, *, torch: Any, memo: dict[int, Any] | None = None) -> Any:
    """Stage a state tree as detached CPU values before checkpoint serialization.

    XLA tensors must cross an explicit device-to-host boundary before
    ``torch.save``.  Memoizing tensors keeps repeated references in state dicts
    shared and avoids an unnecessary second transfer for tied entries.
    """

    if memo is None:
        memo = {}
    identity = id(value)
    if identity in memo:
        return memo[identity]

    if isinstance(value, torch.Tensor):
        staged = value.detach().to(device="cpu").contiguous()
        memo[identity] = staged
        return staged
    if isinstance(value, dict):
        # Copy first to preserve OrderedDict/defaultdict types and attributes
        # such as a module state_dict's version metadata.
        staged = copy.copy(value)
        memo[identity] = staged
        staged.clear()
        for key, item in value.items():
            staged[key] = _cpu_checkpoint_tree(item, torch=torch, memo=memo)
        if hasattr(value, "_metadata"):
            staged._metadata = _cpu_checkpoint_tree(
                value._metadata, torch=torch, memo=memo,
            )
        return staged
    if isinstance(value, list):
        staged_list: list[Any] = []
        memo[identity] = staged_list
        staged_list.extend(_cpu_checkpoint_tree(item, torch=torch, memo=memo) for item in value)
        return staged_list
    if isinstance(value, tuple):
        staged_tuple = tuple(
            _cpu_checkpoint_tree(item, torch=torch, memo=memo) for item in value
        )
        memo[identity] = staged_tuple
        return staged_tuple
    return value


def production_shared_payloads(
    backend: ProductionTrainingBackend,
) -> dict[str, bytes]:
    """Serialize model, optimizer, and schedule state shared by all replicas."""

    backend._assert_update_ready()
    import io

    buffer = io.BytesIO()
    model_state = _cpu_checkpoint_tree(
        backend.model.state_dict(), torch=backend.torch,
    )
    backend.torch.save(model_state, buffer)
    model_bin = buffer.getvalue()
    buffer.seek(0)
    buffer.truncate()
    optimizer_state = _cpu_checkpoint_tree(
        backend.optimizer.state_dict(), torch=backend.torch,
    )
    backend.torch.save(optimizer_state, buffer)
    optimizer_bin = buffer.getvalue()
    scheduler = {
        "schema": "anra-v5-scheduler-state/v1",
        "index": "pre-update cumulative real non-padding tokens",
        "applied_learning_rate": float(backend.optimizer.param_groups[0]["lr"]),
        "schedule": schedule_receipt(),
    }
    return {
        "model.bin": model_bin,
        "optimizer.bin": optimizer_bin,
        "scheduler.json": _canonical_json(scheduler),
    }


def _validate_scheduler_payload(
    payload: bytes, *, expected_learning_rate: float | None,
) -> float:
    try:
        scheduler = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("scheduler checkpoint is not valid JSON") from exc
    expected_fields = {
        "schema", "index", "applied_learning_rate", "schedule",
    }
    if not isinstance(scheduler, dict) or set(scheduler) != expected_fields:
        raise ValueError("scheduler checkpoint fields do not match schema")
    if (scheduler["schema"] != "anra-v5-scheduler-state/v1"
            or scheduler["index"] != "pre-update cumulative real non-padding tokens"
            or scheduler["schedule"] != schedule_receipt()):
        raise ValueError("scheduler checkpoint identity differs from this training schedule")
    learning_rate = scheduler["applied_learning_rate"]
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float))
            or not math.isfinite(float(learning_rate)) or float(learning_rate) < 0):
        raise ValueError("scheduler checkpoint learning rate is invalid")
    learning_rate = float(learning_rate)
    if expected_learning_rate is not None and learning_rate != float(expected_learning_rate):
        raise ValueError("scheduler checkpoint learning rate differs from expected update")
    if payload != _canonical_json(scheduler):
        raise ValueError("scheduler checkpoint must use canonical JSON")
    return learning_rate


def restore_production_shared(
    backend: ProductionTrainingBackend,
    *,
    payloads: Mapping[str, bytes],
    expected_learning_rate: float | None = None,
) -> None:
    """Load model/optimizer/schedule state without touching process RNG."""

    import io

    for name in ("model.bin", "optimizer.bin", "scheduler.json"):
        if name not in payloads or not isinstance(payloads[name], bytes):
            raise ValueError(f"shared production checkpoint lacks {name}")
    expected_group_lr = _validate_scheduler_payload(
        payloads["scheduler.json"],
        expected_learning_rate=expected_learning_rate,
    )
    torch = backend.torch
    device = next(backend.model.parameters()).device
    # A failed load can leave a partial state. Keep the backend closed until
    # every shared payload has loaded and live optimizer ownership is verified.
    backend._uncertified_optimizer_step = True
    backend.last_receipt = None
    backend.model.load_state_dict(
        torch.load(io.BytesIO(payloads["model.bin"]), map_location=device, weights_only=True)
    )
    backend.optimizer.load_state_dict(
        torch.load(io.BytesIO(payloads["optimizer.bin"]), map_location="cpu", weights_only=True)
    )
    observed_lrs = [float(group["lr"]) for group in backend.optimizer.param_groups]
    if not observed_lrs or any(value != expected_group_lr for value in observed_lrs):
        raise ValueError("optimizer checkpoint learning rates disagree with scheduler receipt")
    assert_live_ownership(backend.model, backend.optimizer)
    backend._uncertified_optimizer_step = False


def restore_production(
    backend: ProductionTrainingBackend,
    *,
    payloads: Mapping[str, bytes],
) -> None:
    """Restore model/optimizer/RNG payloads into the live production objects."""

    torch = backend.torch
    restore_production_shared(backend, payloads=payloads)
    import io
    rng = torch.load(io.BytesIO(payloads["rng.bin"]), map_location="cpu", weights_only=True)
    torch.set_rng_state(rng["cpu"].detach().cpu())
    if torch.cuda.is_available() and "cuda" in rng:
        torch.cuda.set_rng_state_all(
            [state.detach().cpu() for state in rng["cuda"]]
        )
    assert_live_ownership(backend.model, backend.optimizer)


__all__ = [
    "BACKEND_SCHEMA",
    "CLIP_NORM_TOLERANCE",
    "GRAD_CLIP_GLOBAL_L2",
    "PackedBatch",
    "ProductionTrainingBackend",
    "StaleOptimizerOwnership",
    "UpdateEvidence",
    "assert_live_ownership",
    "bounded_warmup_schedule",
    "capture_evidence",
    "certify_real_update",
    "production_payloads",
    "production_shared_payloads",
    "restore_production_shared",
    "restore_production",
]
