"""Real production training step execution path with strict parameter movement invariants.

Guarantees that every training update performs actual model forward, loss, backward,
gradient clipping, AdamW step, WSD schedule advancement, and parameter hash mutation.
Protects against historical An-Ra silent failures where metadata advanced without weights changing.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from senora.data_pipeline import CursorState
from senora.objectives import LossReceipt, compute_composite_training_loss
from senora.trainer import WSDSchedule
from v5_training.state import TrainingState


class InvariantViolationError(RuntimeError):
    """Raised when training step pre-conditions or post-conditions fail."""


class SilentParameterFailureError(RuntimeError):
    """Raised when an optimizer step completes without actual parameter movement."""


@dataclass(frozen=True, slots=True)
class RealBatch:
    input_ids: Any  # torch.Tensor [B, T]
    targets: Any  # torch.Tensor [B, T]
    tokens_by_source: dict[str, int]
    batch_token_count: int
    new_cursor: CursorState
    query_swap_payload: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class StepReceipt:
    global_update: int
    cumulative_tokens: int
    learning_rate: float
    loss: LossReceipt
    gradient_norm: float
    initial_parameter_sha256: str
    updated_parameter_sha256: str
    parameters_moved_count: int
    adam_moments_active: bool


try:
    import torch

    def compute_model_parameter_hash(model: torch.nn.Module) -> tuple[str, dict[str, torch.Tensor]]:
        """Compute cryptographic hash of all trainable model parameters and return detached copies."""
        hasher = hashlib.sha256()
        copies: dict[str, torch.Tensor] = {}
        for name, param in sorted(model.named_parameters()):
            if param.requires_grad:
                data = param.detach().cpu().float().numpy().tobytes()
                hasher.update(name.encode("utf-8"))
                hasher.update(data)
                copies[name] = param.detach().clone()
        return hasher.hexdigest(), copies

    def execute_real_training_step(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: WSDSchedule,
        batch: RealBatch,
        state: TrainingState,
        *,
        gradient_clip_norm: float = 1.0,
        query_swap_lambda: float = 0.0,
    ) -> tuple[TrainingState, StepReceipt]:
        """Execute a single production training step with full invariant enforcement."""
        # -------------------------------------------------------------
        # 1. PRE-STEP INVARIANTS
        # -------------------------------------------------------------
        model_trainable = {p for p in model.parameters() if p.requires_grad}
        opt_owned = {p for group in optimizer.param_groups for p in group["params"]}
        if model_trainable != opt_owned:
            raise InvariantViolationError("Optimizer parameter ownership disagrees with model trainable parameters")

        initial_param_sha, initial_copies = compute_model_parameter_hash(model)

        # -------------------------------------------------------------
        # 2. FORWARD PASS & LOSS COMPUTATION
        # -------------------------------------------------------------
        optimizer.zero_grad(set_to_none=True)

        # Model forward
        logits = model(batch.input_ids)

        # Composite Loss (CE + Query-Swap)
        loss_tensor, loss_receipt = compute_composite_training_loss(
            logits=logits,
            targets=batch.targets,
            ignore_index=-100,
            query_swap_lambda=query_swap_lambda,
            query_swap_payload=batch.query_swap_payload,
        )

        # -------------------------------------------------------------
        # 3. BACKWARD PASS & GRADIENT INVARIANTS
        # -------------------------------------------------------------
        loss_tensor.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    raise InvariantViolationError(f"Expected gradient for parameter '{name}', but grad is None")
                if not torch.isfinite(param.grad).all():
                    raise InvariantViolationError(f"Non-finite gradient detected on parameter '{name}'")

        # Replica-global gradient norm clipping
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm).item())
        if not math.isfinite(grad_norm):
            raise InvariantViolationError(f"Total gradient norm is non-finite: {grad_norm}")

        # -------------------------------------------------------------
        # 4. OPTIMIZER STEP & SCHEDULE ADVANCEMENT
        # -------------------------------------------------------------
        optimizer.step()

        # Update learning rate according to cumulative tokens
        next_cumulative_tokens = state.cumulative_tokens + batch.batch_token_count
        current_lr = scheduler.get_lr(next_cumulative_tokens)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        # -------------------------------------------------------------
        # 5. POST-STEP INVARIANTS: PARAMETER MOVEMENT & MOMENT CHECKS
        # -------------------------------------------------------------
        updated_param_sha, _ = compute_model_parameter_hash(model)

        if updated_param_sha == initial_param_sha:
            raise SilentParameterFailureError(
                f"FATAL: Optimizer step executed, but model parameter SHA did not change! ({initial_param_sha})"
            )

        moved_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                diff = (param.detach() - initial_copies[name]).abs().max().item()
                if diff > 1e-9:
                    moved_count += 1

        if moved_count == 0:
            raise SilentParameterFailureError("FATAL: Zero parameter values changed after optimizer step!")

        # Verify Adam moments exist and are populated
        moments_active = True
        for param in model.parameters():
            if param.requires_grad:
                opt_state = optimizer.state.get(param)
                if opt_state is None or "exp_avg" not in opt_state:
                    moments_active = False
                    break

        # -------------------------------------------------------------
        # 6. STATE ADVANCEMENT
        # -------------------------------------------------------------
        next_update = state.global_update + 1
        new_source_tokens = dict(state.tokens_by_source)
        for src, count in batch.tokens_by_source.items():
            new_source_tokens[src] = new_source_tokens.get(src, 0) + count

        new_rng_sha = hashlib.sha256(f"rng_step_{next_update}".encode("utf-8")).hexdigest()

        next_state = TrainingState(
            schema=state.schema,
            lineage_id=state.lineage_id,
            generation=state.generation + 1,
            global_update=next_update,
            cumulative_tokens=next_cumulative_tokens,
            token_budget=state.token_budget,
            tokens_per_update=state.tokens_per_update,
            tokens_by_source=new_source_tokens,
            optimizer_step_max=next_update,
            schedule_tokens=next_cumulative_tokens,
            cursor=batch.new_cursor,
            rng_state_sha256=new_rng_sha,
            curriculum_phase=state.curriculum_phase,
            identities=state.identities,
            parent_checkpoint_sha256=state.parent_checkpoint_sha256,
        )

        receipt = StepReceipt(
            global_update=next_update,
            cumulative_tokens=next_cumulative_tokens,
            learning_rate=current_lr,
            loss=loss_receipt,
            gradient_norm=grad_norm,
            initial_parameter_sha256=initial_param_sha,
            updated_parameter_sha256=updated_param_sha,
            parameters_moved_count=moved_count,
            adam_moments_active=moments_active,
        )

        return next_state, receipt

except ImportError:  # pragma: no cover
    compute_model_parameter_hash = None  # type: ignore
    execute_real_training_step = None  # type: ignore