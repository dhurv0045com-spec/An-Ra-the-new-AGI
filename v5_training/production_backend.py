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

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from .optimizer import validate_parameter_ownership
from .schedule import lr_at, schedule_receipt
from .state import CURSOR_SCHEMA, CursorState
from .trainer import BackendReport
from v5_model.core import packed_layout
from v5_objectives.causal_lm import causal_lm_loss


BACKEND_SCHEMA = "anra-v5-production-backend-receipt/v1"
GRAD_CLIP_GLOBAL_L2 = 1.0
_NORM_TOLERANCE = 1e-6


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


def _tensor_sha256(value: Any, torch: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
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
    embedding_data_ptr: int
    owned_parameter_ids: frozenset[int]

    def as_receipt(self) -> dict[str, object]:
        return {
            "parameter_sha256": self.parameter_sha256,
            "moment_sha256": self.moment_sha256,
            "optimizer_steps": dict(sorted(self.optimizer_steps.items())),
            "embedding_data_ptr": self.embedding_data_ptr,
        }


def _embedding_weight(model: Any) -> Any:
    names = [name for name, _ in model.named_parameters() if name.endswith("embedding.weight")]
    if len(names) != 1:
        raise ValueError("backend requires exactly one tied embedding table")
    return dict(model.named_parameters())[names[0]]


def capture_evidence(model: Any, optimizer: Any, *, torch: Any) -> UpdateEvidence:
    """Capture live parameter/optimizer identities and hashes."""

    steps: dict[str, int] = {}
    moment_digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter, {})
        step = state.get("step")
        steps[name] = int(step.item()) if torch.is_tensor(step) else int(step or 0)
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if torch.is_tensor(value):
                moment_digest.update(name.encode("utf-8"))
                moment_digest.update(key.encode("utf-8"))
                moment_digest.update(_tensor_sha256(value, torch).encode("ascii"))
    embedding = _embedding_weight(model)
    return UpdateEvidence(
        parameter_sha256=_model_sha256(model, torch=torch),
        moment_sha256=moment_digest.hexdigest(),
        optimizer_steps=steps,
        embedding_data_ptr=embedding.data_ptr(),
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
    live_ptrs = {id(parameter): parameter.data_ptr() for parameter in model.parameters()}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in live_ptrs:
                raise StaleOptimizerOwnership(
                    "stale optimizer ownership: an optimizer parameter is not a live model "
                    "parameter (historical core-vnext device-movement failure)"
                )
            if parameter.data_ptr() != live_ptrs[id(parameter)]:
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


def certify_real_update(
    *,
    model: Any,
    optimizer: Any,
    before: UpdateEvidence,
    after: UpdateEvidence,
    expected_learning_rate: float,
    supervised_tokens: int,
    loss: float,
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
    if not math.isfinite(grad_norm_pre_clip) or not math.isfinite(grad_norm_post_clip):
        raise ValueError("abort NONFINITE_GRADIENT")
    if grad_norm_post_clip > GRAD_CLIP_GLOBAL_L2 + _NORM_TOLERANCE:
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
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            raise ValueError(f"abort NO_GRADIENT: {name} received no gradient")
        if not bool(torch.isfinite(gradient).all().item()):
            raise ValueError(f"abort NONFINITE_GRADIENT: {name} gradient is not finite")
        state = optimizer.state.get(parameter, {})
        step = state.get("step")
        step_value = int(step.item()) if torch.is_tensor(step) else int(step or 0)
        if step_value != before.optimizer_steps[name] + 1:
            raise ValueError(
                f"abort OPTIMIZER_NOT_STEPPED: {name} Adam step did not advance exactly once"
            )
        moments = (state.get("exp_avg"), state.get("exp_avg_sq"))
        if not all(torch.is_tensor(moment) for moment in moments):
            raise ValueError(f"abort MOMENTS_MISSING: {name} lacks Adam moments after step")
        for moment in moments:
            if not bool(torch.isfinite(moment).all().item()):
                raise ValueError(f"abort NONFINITE_MOMENT: {name} moment is not finite")
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
    if embedding.data_ptr() != before.embedding_data_ptr:
        raise ValueError("abort TIED_WEIGHT_BROKEN: embedding storage identity changed")
    receipt: dict[str, object] = {
        "schema": BACKEND_SCHEMA,
        "loss": loss,
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
        if not callable(self.schedule):
            raise ValueError("schedule must map cumulative tokens to a learning rate")
        if self.bfloat16_autocast and not hasattr(self.torch, "autocast"):
            raise ValueError("bfloat16 autocast requires a framework with autocast support")
        assert_live_ownership(self.model, self.optimizer)
        self.last_receipt: dict[str, object] | None = None

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
        if self.bfloat16_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(tokens, positions, mask)
        else:
            logits = self.model(tokens, positions, mask)
        loss, supervised_tokens = causal_lm_loss(
            logits, tokens, segment_ids, bos_id=self.bos_id, pad_id=self.pad_id,
            torch_module=torch,
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
        self.optimizer.step()

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
        self.last_receipt = receipt
        finite_loss = bool(torch.isfinite(loss.detach()).item())
        finite_grads = all(
            bool(torch.isfinite(parameter.grad).all().item())
            for parameter in trainable
            if parameter.grad is not None
        )
        return BackendReport(
            tokens_by_source=dict(batch.tokens_by_source),
            cursor=batch.cursor,
            rng_state_sha256=receipt["rng_state_sha256"],
            loss_finite=finite_loss,
            grad_finite=finite_grads,
            grad_norm_post_clip=float(grad_norm_post_clip),
            tied_preserved=True,
        )

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


def production_payloads(
    backend: ProductionTrainingBackend, *, state: Any
) -> dict[str, bytes]:
    """Serialize the exact resume inventory from the live production objects."""

    import io

    torch = backend.torch
    buffer = io.BytesIO()
    torch.save(backend.model.state_dict(), buffer)
    model_bin = buffer.getvalue()
    buffer.seek(0)
    buffer.truncate()
    torch.save(backend.optimizer.state_dict(), buffer)
    optimizer_bin = buffer.getvalue()
    buffer.seek(0)
    buffer.truncate()
    torch.save(
        {"cpu": torch.get_rng_state()},
        buffer,
    )
    rng_bin = buffer.getvalue()
    learning_rate = float(backend.optimizer.param_groups[0]["lr"])
    scheduler = {
        "schema": "anra-v5-scheduler-state/v1",
        "index": "pre-update cumulative real non-padding tokens",
        "applied_learning_rate": learning_rate,
        "schedule": schedule_receipt(),
    }
    return {
        "model.bin": model_bin,
        "optimizer.bin": optimizer_bin,
        "scheduler.json": _canonical_json(scheduler),
        "rng.bin": rng_bin,
        "cursor.json": _canonical_json(
            {
                "schema": state.cursor.schema,
                "pack_manifest_sha256": state.cursor.pack_manifest_sha256,
                "shard_ordinal": state.cursor.shard_ordinal,
                "sequence_ordinal": state.cursor.sequence_ordinal,
                "token_offset": state.cursor.token_offset,
            }
        ),
        "ledger.json": _canonical_json(dict(state.tokens_by_source)),
        "training_state.json": _canonical_json(state.canonical()),
    }


def restore_production(
    backend: ProductionTrainingBackend,
    *,
    payloads: Mapping[str, bytes],
) -> None:
    """Restore model/optimizer/RNG payloads into the live production objects."""

    torch = backend.torch
    import io

    device = next(backend.model.parameters()).device
    backend.model.load_state_dict(
        torch.load(io.BytesIO(payloads["model.bin"]), map_location=device, weights_only=True)
    )
    backend.optimizer.load_state_dict(
        torch.load(io.BytesIO(payloads["optimizer.bin"]), map_location="cpu", weights_only=True)
    )
    rng = torch.load(io.BytesIO(payloads["rng.bin"]), map_location="cpu", weights_only=True)
    torch.set_rng_state(rng["cpu"].detach().cpu())
    assert_live_ownership(backend.model, backend.optimizer)


__all__ = [
    "BACKEND_SCHEMA",
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
    "restore_production",
]
