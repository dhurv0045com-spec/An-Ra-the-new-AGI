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
