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
    pack_total = max_steps_override if max_steps_override > 0 else budget_steps
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


def _load_payload(resume_from: str) -> tuple[dict[str, "torch.Tensor"], dict]:
    """Load raw state + payload. Strict-first; legacy fallback ONLY for the
    specific known condition (missing tokenizer contract on older writers).
    Corruption, architecture mismatch, and loader bugs surface."""
    import torch

    from anra_core.checkpoint import load_core_checkpoint
    from anra_core.errors import RepresentationIncompatibleError

    try:
        _model, _payload, _identity = load_core_checkpoint(resume_from)
    except RepresentationIncompatibleError as exc:
        if "tokenizer contract" not in str(exc):
            raise  # corruption/mismatch/bug: never disguise as legacy
        print(f"[resume] strict load refused ({exc}); explicit legacy fallback "
              "for older-writer contract absence", flush=True)
    payload = torch.load(resume_from, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict") or payload.get("model")
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint has no model tensors: {resume_from}")
    return state, payload


def restore_training_state(
    resume_from: str,
    model,  # AnRaCore (typed loosely to keep this module import-light)
    optimizer,
    *,
    mode: str = RESUME_SAME_PACK,
    current_pack_manifest_sha256: str | None = None,
) -> RestoredTrainingState:
    """Restore checkpoint INTO the caller's model and prove it took effect.

    Mode semantics:
      same_pack: pack_step restored only when the checkpoint's
                 pack_manifest_sha256 matches the attached pack; mismatch or
                 unverifiable progress fails closed.
      new_pack_parent: pack_step forced to 0; checkpoint pack identity is
                 irrelevant (LEGACY FULL-RESUME -> schema-v2 boundary).
    """
    import hashlib

    import torch

    state, payload = _load_payload(resume_from)

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
            optimizer.load_state_dict(optimizer_state)
            optimizer_restored = True
        except Exception as exc:
            print(f"[resume] optimizer state present but failed to load: {exc}", flush=True)
            raise

    checkpoint_pack_sha = payload.get("pack_manifest_sha256")
    if mode == RESUME_SAME_PACK:
        restored_pack_step = int(payload.get("pack_step", 0) or 0)
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

    param_digest = hashlib.sha256()
    for key in sorted(state):
        param_digest.update(key.encode())
        param_digest.update(state[key].numpy().tobytes())

    return RestoredTrainingState(
        global_step=int(payload.get("global_step", 0) or 0),
        pack_step=restored_pack_step,
        pack_manifest_sha256=checkpoint_pack_sha,
        optimizer_restored=optimizer_restored,
        mode=mode,
        checkpoint_parameter_sha256=param_digest.hexdigest(),
    )
