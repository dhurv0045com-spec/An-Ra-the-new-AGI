"""Pure step-accounting and durability decisions for the TPU trainer.

Kept free of torch_xla so the resume mathematics is deterministically
testable on CPU. The trainer consumes these helpers; this module owns the
semantics:

- ``global_step``: monotonically increasing across packs (checkpoint identity).
- ``pack_step``   : position within the CURRENT pack's schedule horizon.
- ``schedule_step``: what the LR scheduler observes == pack_step, because WSD
  decay must land at THIS pack's boundary, not at a position inherited from
  a previous pack's history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class PackHorizon:
    """Resolved step accounting for one training session over one pack."""

    pack_total_steps: int      # optimizer updates this pack should receive
    start_pack_step: int       # restored pack-relative progress (fresh pack: 0)
    updates_remaining: int     # exactly how many updates this session runs


def resolve_pack_horizon(
    *,
    global_step: int,
    restored_pack_step: int,
    token_budget: int,
    tokens_per_step: int,
    max_steps_override: int = 0,
) -> PackHorizon:
    """Derive pack-local accounting. Global step NEVER bounds the pack loop.

    Regression being fixed: the previous trainer computed total_steps from the
    token budget (~2,500 for a 330M pack) but looped ``while global_step <
    total_steps`` with global_step resumed at 20,000 - executing ZERO updates.
    """
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be positive")
    budget_steps = max(1, token_budget // tokens_per_step)
    pack_total = (
        min(budget_steps, max_steps_override)
        if max_steps_override > 0
        else budget_steps
    )
    start_pack = max(0, int(restored_pack_step))
    remaining = max(0, pack_total - start_pack)
    return PackHorizon(
        pack_total_steps=pack_total,
        start_pack_step=start_pack,
        updates_remaining=remaining,
    )


def should_periodic_save(pack_step: int, save_interval: int) -> bool:
    """True when pack_step hits a save boundary (interval >= 1)."""
    if save_interval <= 0:
        return False
    return pack_step > 0 and pack_step % save_interval == 0


def update_best(
    best_loss: float | None, current_loss: float
) -> tuple[float, bool]:
    """Return (best_loss_after, improved). Degradation never lowers the bar."""
    if best_loss is None or current_loss < best_loss:
        return current_loss, True
    return best_loss, False


def degradation_ratio(best_loss: float, current_loss: float) -> float:
    """How much worse the current loss is than best (1.10 == 10% worse)."""
    if best_loss <= 0:
        raise ValueError("best_loss must be positive")
    return current_loss / best_loss


# --------------------------------------------------------------------------
# Canonical parameter hashing - ONE function used by loader, trainer, receipt,
# candidate writer, and evaluation. Tied embeddings handled once, here.
# --------------------------------------------------------------------------


def canonical_parameter_sha256(state: dict[str, "torch.Tensor"]) -> str:
    """Hash the normalized dense contract. Aliases (lm_head) excluded so the
    hash is identical before and after a save/reload round trip."""
    import hashlib

    import torch

    hasher = hashlib.sha256()
    for name in sorted(state):
        if name == "lm_head.weight":  # tied alias of token_embedding_table
            continue
        tensor = state[name].detach().cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        header = f"{name}\0{tuple(tensor.shape)}\0{tensor.dtype}\0".encode()
        hasher.update(header)
        raw = tensor.view(torch.uint8).reshape(-1)
        for start in range(0, raw.numel(), 4 * 1024 * 1024):
            hasher.update(raw[start : start + 4 * 1024 * 1024].numpy().tobytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


@dataclass(slots=True)
class PreparedTrainingState:
    """Everything a worker needs after resume - explicit, no loose payload.

    One model. One optimizer. One restore. The worker consumes this struct;
    it never re-reads raw checkpoint payloads or re-restores state.
    """

    model: object  # AnRaCore
    optimizer: object
    global_step: int
    optimizer_updates: int
    resume_mode: str  # "new_pack_parent" | "same_pack" | "fresh"
    source_checkpoint: str | None
    checkpoint_parameter_sha256: str
    optimizer_restored: bool
    checkpoint_schema_version: int
    pack_step: int = 0
    trainer_state: dict | None = None
    lr_schedule: dict | None = None

    def require_step_at_least(self, expected_resume_step: int) -> None:
        """P0-4: explicit minimum-step check on REAL restored metadata.
        No synthetic checkpoint dicts passed to validators."""
        if self.global_step < expected_resume_step:
            raise RuntimeError(
                f"parent global_step {self.global_step:,} is below the expected "
                f"resume step {expected_resume_step:,} - wrong parent artifact?"
            )


def prepare_training_state(
    *,
    parent_checkpoint: str | None,
    model_config,  # CoreConfig
    learning_rate: float,
    weight_decay: float,
    expected_resume_step: int,
    resume_mode: str = "new_pack_parent",
    current_pack_manifest_sha256: str | None = None,
    allow_legacy_checkpoint: bool = False,
) -> PreparedTrainingState:
    """CPU-testable pre-XLA orchestration: construct → attach → restore.

    Lifecycle (the P0 fix):
      1. construct model (deterministic seed responsibility of caller)
      2. construct optimizer ATTACHED to that model
      3. restore checkpoint INTO both via restore_training_state (once)
      4. validate restored metadata explicitly
    The worker then moves model+optimizer to XLA and trains. No competing
    temporary state, no second restore path, no discarded payload.
    """
    import torch

    from anra_core.model import AnRaCore

    model = AnRaCore(model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate,
        betas=(0.9, 0.95), weight_decay=weight_decay,
    )

    if parent_checkpoint:
        restored = restore_training_state(
            str(parent_checkpoint), model, optimizer, mode=resume_mode,
            current_pack_manifest_sha256=current_pack_manifest_sha256,
            allow_legacy_checkpoint=allow_legacy_checkpoint,
        )
        # P0-4: explicit minimum-step check on REAL restored metadata.
        if restored.global_step < expected_resume_step:
            raise RuntimeError(
                f"parent global_step {restored.global_step:,} is below the expected "
                f"resume step {expected_resume_step:,} - wrong parent artifact?"
            )
        # Optimizer update count from the RESTORED moments (authoritative).
        counts = [
            int(state["step"].detach().cpu().item())
            if isinstance(state.get("step"), torch.Tensor)
            else int(state.get("step", 0))
            for state in restored_optimizer_snapshot(optimizer).values()
        ] or [0]
        return PreparedTrainingState(
            model=model,
            optimizer=optimizer,
            global_step=restored.global_step,
            optimizer_updates=max(counts),
            resume_mode=restored.mode,
            source_checkpoint=str(parent_checkpoint),
            checkpoint_parameter_sha256=restored.checkpoint_parameter_sha256,
            optimizer_restored=restored.optimizer_restored,
            checkpoint_schema_version=restored.checkpoint_schema_version,
            pack_step=restored.pack_step,
            trainer_state=(
                restored.trainer_state if restored.mode == RESUME_SAME_PACK else None
            ),
            lr_schedule=(
                restored.lr_schedule if restored.mode == RESUME_SAME_PACK else None
            ),
        )

    return PreparedTrainingState(
        model=model,
        optimizer=optimizer,
        global_step=0,
        optimizer_updates=0,
        resume_mode="fresh",
        source_checkpoint=None,
        checkpoint_parameter_sha256=canonical_parameter_sha256(model.state_dict()),
        optimizer_restored=False,
        checkpoint_schema_version=3,
    )


def restored_optimizer_snapshot(optimizer) -> dict:
    """The optimizer's CURRENT state dict (post-restore). Used to read the
    authoritative moment counts without touching raw checkpoint payloads."""
    return optimizer.state_dict()["state"]


# --------------------------------------------------------------------------
# Checkpoint restore into the CALLER's model (the P0 fix).
#
# History: a trainer resume helper called load_core_checkpoint(), which builds
# its own model internally, and returned only metadata - silently discarding
# the loaded weights while logs looked healthy. The canonical restore lives
# here so both the trainer and the integration tests exercise one path.
# --------------------------------------------------------------------------

RESUME_SAME_PACK = "same_pack"
RESUME_NEW_PACK_PARENT = "new_pack_parent"


@dataclass(slots=True)
class RestoredTrainingState:
    """What a resume actually restored - recorded, never implied."""

    global_step: int
    pack_step: int
    pack_manifest_sha256: str | None
    optimizer_restored: bool
    mode: str
    checkpoint_parameter_sha256: str
    checkpoint_schema_version: int
    trainer_state: dict | None
    lr_schedule: dict | None


def _load_payload(
    resume_from: str, *, allow_legacy_checkpoint: bool = False
) -> tuple[dict[str, "torch.Tensor"], dict]:
    """Load once through the strict, weights-only checkpoint boundary."""
    from anra_core.checkpoint import load_core_checkpoint

    loaded_model, payload, identity = load_core_checkpoint(
        resume_from, legacy_unverified=allow_legacy_checkpoint
    )
    if identity.artifact_class != "full_resume":
        raise RuntimeError("resume checkpoint must be a full_resume artifact")
    schema = int(identity.artifact_schema_version or 0)
    if schema not in {1, 2, 3, 9}:
        raise RuntimeError(f"unsupported full-resume schema: {schema}")
    optimizer_state = payload.get("optimizer_state_dict") or payload.get("optimizer")
    if not isinstance(optimizer_state, dict) or not optimizer_state.get("state"):
        raise RuntimeError("resume checkpoint has no populated optimizer state")
    state = loaded_model.state_dict()
    return state, payload


def restore_training_state(
    resume_from: str,
    model,  # AnRaCore (typed loosely to keep this module import-light)
    optimizer,
    *,
    mode: str = RESUME_SAME_PACK,
    current_pack_manifest_sha256: str | None = None,
    allow_legacy_checkpoint: bool = False,
) -> RestoredTrainingState:
    """Restore checkpoint INTO the caller's model and prove it took effect.

    Mode semantics:
      same_pack: pack_step restored only when the checkpoint's
                 pack_manifest_sha256 matches the attached pack; mismatch or
                 unverifiable progress fails closed.
      new_pack_parent: pack_step forced to 0; checkpoint pack identity is
                 irrelevant (legacy full-resume -> schema-v3 boundary).
    """
    import torch

    state, payload = _load_payload(
        resume_from, allow_legacy_checkpoint=allow_legacy_checkpoint
    )

    # Install into the CALLER's model and prove installation per tensor.
    model.load_state_dict(state, strict=True)
    installed = 0
    live = model.state_dict()
    for key, tensor in state.items():
        if torch.is_floating_point(tensor):
            if not torch.equal(live[key], tensor):
                raise RuntimeError(f"resume verification failed: {key} did not install")
            installed += 1
    if installed == 0:
        raise RuntimeError("resume verification failed: no float tensors compared")

    optimizer_state = payload.get("optimizer") or payload.get("optimizer_state_dict")
    optimizer_restored = False
    if isinstance(optimizer_state, dict) and optimizer_state.get("state"):
        try:
            saved_groups = optimizer_state.get("param_groups") or []
            live_lengths = [len(group["params"]) for group in optimizer.param_groups]
            saved_lengths = [len(group.get("params", [])) for group in saved_groups]
            if live_lengths != saved_lengths:
                trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                decay = [parameter for parameter in trainable if parameter.ndim >= 2]
                no_decay = [parameter for parameter in trainable if parameter.ndim < 2]
                if saved_lengths == [len(trainable)]:
                    rebuilt_groups = [trainable]
                elif saved_lengths == [len(decay), len(no_decay)]:
                    rebuilt_groups = [decay, no_decay]
                else:
                    raise RuntimeError(
                        "optimizer parameter-group layout is incompatible: "
                        f"checkpoint={saved_lengths}, model="
                        f"single={[len(trainable)]}, decay_split="
                        f"{[len(decay), len(no_decay)]}"
                    )
                optimizer.state.clear()
                optimizer.param_groups.clear()
                for parameters in rebuilt_groups:
                    optimizer.add_param_group({"params": parameters})
            optimizer.load_state_dict(optimizer_state)
            optimizer_restored = True
        except Exception as exc:
            print(f"[resume] optimizer state present but failed to load: {exc}", flush=True)
            raise

    schema_version = int(payload.get("checkpoint_schema_version", 0))
    trainer_state = payload.get("trainer_state")
    if schema_version in {2, 3} and not isinstance(trainer_state, dict):
        raise RuntimeError("resumable checkpoint is missing trainer_state")
    saved_pack_step_value = (trainer_state or {}).get("pack_step", payload.get("pack_step"))
    if saved_pack_step_value is None and schema_version == 2:
        saved_schedule = payload.get("lr_schedule") or {}
        saved_pack_step_value = int(payload.get("global_step", 0)) - int(
            saved_schedule.get("origin_step", payload.get("global_step", 0))
        )
    saved_pack_step = int(saved_pack_step_value or 0)
    checkpoint_pack_sha = payload.get("pack_manifest_sha256") or (
        trainer_state or {}
    ).get("pack_manifest_sha256")
    if mode == RESUME_SAME_PACK:
        restored_pack_step = saved_pack_step
        if restored_pack_step > 0 and not checkpoint_pack_sha:
            raise RuntimeError(
                "SAME_PACK resume refused: checkpoint carries pack_step without "
                "pack_manifest_sha256 - identity cannot be verified (fail closed)"
            )
        if (
            current_pack_manifest_sha256
            and checkpoint_pack_sha
            and current_pack_manifest_sha256 != checkpoint_pack_sha
        ):
            raise RuntimeError(
                "SAME_PACK resume refused: checkpoint was trained on a different "
                f"pack ({str(checkpoint_pack_sha)[:16]}... != "
                f"{str(current_pack_manifest_sha256)[:16]}...). Use NEW_PACK_PARENT."
            )
    else:  # NEW_PACK_PARENT
        restored_pack_step = 0

    return RestoredTrainingState(
        global_step=int(payload.get("global_step", 0) or 0),
        pack_step=restored_pack_step,
        pack_manifest_sha256=checkpoint_pack_sha,
        optimizer_restored=optimizer_restored,
        mode=mode,
        checkpoint_parameter_sha256=canonical_parameter_sha256(state),
        checkpoint_schema_version=schema_version,
        trainer_state=trainer_state if isinstance(trainer_state, dict) else None,
        lr_schedule=(
            payload.get("lr_schedule")
            if isinstance(payload.get("lr_schedule"), dict)
            else None
        ),
    )
