"""Evidence-driven plasticity controller (B06 + B2.1 refinements).

A pure state machine connecting capability formation, preservation and
reacquisition. ``transition`` is a pure function of (state, metrics, config):
it may set a schedule multiplier, replay mixture or checkpoint request; it
never mutates tensors and never reads a sealed split. Metrics come only from
the declared controller pool — development, measurement and sealed inputs
reject. Missing or stale measurements can never count as formation passes,
and a collapse never automatically lowers the learning rate (the ARK-010
finding). The controller is off by default; the disabled and fixed-schedule
controls remain available for causal comparison.

Evidence-aligned refinements (Arkenstone ARK-007R/010/011/012/013 and
CONTROLLER_SYNTHESIS):

- **Protection is earned at qualification.** A capability family joins the
  protected set only when its formation window is confirmed (STABILIZE
  entry); an introduced-but-unqualified family is merely tracked as
  qualifying. Formation failure is never misattributed to preservation
  machinery.
- **Collapse confirmation.** A protected-family breach must persist across
  ``collapse_confirmation_evaluations`` consecutive fresh evaluations before
  REACQUIRE entry, mirroring the preregistered onset→confirmation pattern
  (collapse90 onset/confirm). A single dipping evaluation does not hand over
  the state machine.
- **Sustained recovery.** Leaving REACQUIRE requires the exit threshold to
  hold across ``recovery_confirmation_evaluations`` consecutive evaluations
  (recovery90 semantics; 8/9 HIGH arms recovered, so instability episodes are
  treated as potentially self-recovering, never terminal).
- **Relative displacement telemetry** (final_relative_displacement, ~0.008
  LOW vs ~0.379 HIGH in ARK-007R) is accepted alongside absolute
  displacement as plasticity evidence; near-freezing must not masquerade as
  consolidation.
- **Proposed transitions are recorded** in history with their status, so a
  no-event or budget-blocked regime is visible rather than silently absent
  (ARK-013: ``adaptive_switches: 0`` must be observable).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from bramastra_lab.research.config import ControllerSection

STATES = frozenset({"FORM", "STABILIZE", "EXPAND", "REACQUIRE", "HOLD",
                    "FIXED_SCHEDULE"})
FORBIDDEN_POOLS = frozenset({
    "development", "sealed", "measurement", "strategy_validation", "confirmation",
    "dev", "test", "eval",
})
TRANSITION_STATUSES = frozenset({"confirmed", "proposed"})

REPLAY_MIXTURES: Mapping[str, Mapping[str, float]] = {
    "FORM": {"acquisition": 1.0},
    "EXPAND": {"acquisition": 0.75, "protected_replay": 0.25},
    "STABILIZE": {"acquisition": 0.5, "protected_replay": 0.5},
    "REACQUIRE": {"acquisition": 0.25, "protected_replay": 0.75},
    "HOLD": {"acquisition": 0.0, "protected_replay": 0.0},
    "FIXED_SCHEDULE": {"acquisition": 1.0},
}


class ControllerInputError(ValueError):
    """Controller metrics violate the pool firewall or metric contract."""


class PoolViolationError(ControllerInputError):
    """Metrics from a development/measurement/sealed pool reached the controller."""


@dataclass(frozen=True)
class ControllerMetrics:
    """One controller-pool evaluation.

    ``pool_id`` must be the declared controller pool. ``family_scores`` are
    per-capability-family scores in [0, 1]; every protected family must be
    present and fresh. Family scores should be robustness-aware (for example
    the robust minimum over probe variants), because canonical accuracy alone
    can hide invariance erosion. ``parameter_displacement`` and
    ``relative_parameter_displacement`` (or ``score_improvements``) are the
    learning/plasticity evidence required before stable scores may be treated
    as formed capability.
    """

    pool_id: str
    at_update: int
    family_scores: Mapping[str, float]
    parameter_displacement: float | None = None
    relative_parameter_displacement: float | None = None
    score_improvements: Mapping[str, bool] = field(default_factory=dict)
    nonfinite_observed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.pool_id, str) or not self.pool_id.strip():
            raise ControllerInputError("pool_id must be a nonempty string")
        if self.pool_id in FORBIDDEN_POOLS:
            raise PoolViolationError(
                f"pool {self.pool_id!r} is a measurement/sealed pool and can never "
                "feed the controller")
        if not isinstance(self.at_update, int) or isinstance(self.at_update, bool) \
                or self.at_update < 0:
            raise ControllerInputError("at_update must be a nonnegative integer")
        if not isinstance(self.family_scores, Mapping) or not self.family_scores:
            raise ControllerInputError("family_scores must be a nonempty mapping")
        for family, score in self.family_scores.items():
            if not isinstance(family, str) or not family:
                raise ControllerInputError("family names must be nonempty strings")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise ControllerInputError(f"score for {family!r} must be a number")
            if not 0.0 <= float(score) <= 1.0:
                raise ControllerInputError(
                    f"score for {family!r} must be within [0, 1]")
        for name in ("parameter_displacement", "relative_parameter_displacement"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                    raise ControllerInputError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class ControllerDecision:
    """What the controller wants applied at this optimizer boundary."""

    state: str
    lr_multiplier: float
    replay_mixture: Mapping[str, float]
    checkpoint_requested: bool
    reason: str
    pause_updates: bool = False
    proposed_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state, "lr_multiplier": self.lr_multiplier,
            "replay_mixture": dict(self.replay_mixture),
            "checkpoint_requested": self.checkpoint_requested, "reason": self.reason,
            "pause_updates": self.pause_updates, "proposed_only": self.proposed_only,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ControllerDecision":
        return cls(
            state=raw["state"], lr_multiplier=float(raw["lr_multiplier"]),
            replay_mixture=dict(raw["replay_mixture"]),
            checkpoint_requested=bool(raw["checkpoint_requested"]),
            reason=raw["reason"], pause_updates=bool(raw["pause_updates"]),
            proposed_only=bool(raw["proposed_only"]),
        )


@dataclass(frozen=True)
class TransitionRecord:
    at_update: int
    from_state: str
    to_state: str
    reason: str
    status: str = "confirmed"

    def __post_init__(self) -> None:
        if self.status not in TRANSITION_STATUSES:
            raise ControllerInputError(f"transition status must be one of {sorted(TRANSITION_STATUSES)}")

    def to_dict(self) -> dict[str, Any]:
        return {"at_update": self.at_update, "from_state": self.from_state,
                "to_state": self.to_state, "reason": self.reason, "status": self.status}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TransitionRecord":
        return cls(**raw)


HISTORY_LIMIT = 64


@dataclass(frozen=True)
class ControllerState:
    """Persisted controller state; stored in every checkpoint."""

    state: str = "FORM"
    consecutive_passes: int = 0
    at_update: int = 0
    transitions: int = 0
    protected_families: tuple[str, ...] = ()
    qualifying_families: tuple[str, ...] = ()
    acquiring_family: str | None = None
    window_start_score: float | None = None
    hold_return_state: str | None = None
    hold_reason: str | None = None
    last_metrics_update: int | None = None
    pending_collapse: Mapping[str, int] = field(default_factory=dict)
    consecutive_recovery_passes: int = 0
    last_controller_eval_update: int | None = None
    history: tuple[TransitionRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ControllerInputError(f"unknown controller state {self.state!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state, "consecutive_passes": self.consecutive_passes,
            "at_update": self.at_update, "transitions": self.transitions,
            "protected_families": list(self.protected_families),
            "qualifying_families": list(self.qualifying_families),
            "acquiring_family": self.acquiring_family,
            "window_start_score": self.window_start_score,
            "hold_return_state": self.hold_return_state,
            "hold_reason": self.hold_reason,
            "last_metrics_update": self.last_metrics_update,
            "pending_collapse": dict(self.pending_collapse),
            "consecutive_recovery_passes": self.consecutive_recovery_passes,
            "last_controller_eval_update": self.last_controller_eval_update,
            "history": [record.to_dict() for record in self.history],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ControllerState":
        known = {"state", "consecutive_passes", "at_update", "transitions",
                 "protected_families", "qualifying_families", "acquiring_family",
                 "window_start_score", "hold_return_state", "hold_reason",
                 "last_metrics_update", "pending_collapse", "consecutive_recovery_passes",
                 "last_controller_eval_update", "history"}
        unknown = set(raw) - known
        if unknown:
            raise ControllerInputError(f"controller state has unknown fields: {sorted(unknown)}")
        return cls(
            state=raw["state"], consecutive_passes=raw["consecutive_passes"],
            at_update=raw["at_update"], transitions=raw["transitions"],
            protected_families=tuple(raw["protected_families"]),
            qualifying_families=tuple(raw.get("qualifying_families", ())),
            acquiring_family=raw["acquiring_family"],
            window_start_score=raw["window_start_score"],
            hold_return_state=raw["hold_return_state"],
            hold_reason=raw["hold_reason"],
            last_metrics_update=raw["last_metrics_update"],
            pending_collapse=dict(raw.get("pending_collapse", {})),
            consecutive_recovery_passes=raw.get("consecutive_recovery_passes", 0),
            last_controller_eval_update=raw.get("last_controller_eval_update"),
            history=tuple(TransitionRecord.from_dict(record) for record in raw["history"]),
        )


def _decide(state: str, *, lr_multiplier: float, reason: str,
            checkpoint: bool = False, pause: bool = False,
            proposed_only: bool = False) -> ControllerDecision:
    return ControllerDecision(
        state=state, lr_multiplier=lr_multiplier,
        replay_mixture=dict(REPLAY_MIXTURES[state]), checkpoint_requested=checkpoint,
        reason=reason, pause_updates=pause, proposed_only=proposed_only)


def _validate_freshness(metrics: ControllerMetrics, config: ControllerSection,
                        current_update: int) -> bool:
    """Return True when metrics are fresh enough to count as a pass.

    Thresholds only resolve if measurement cadence is dense relative to
    recovery speed (the ARK-012 cadence-aliasing lesson); operators should
    evaluate frequently enough that freshness holds.
    """
    return (current_update - metrics.at_update) <= config.max_metrics_age_updates


def _record(state: ControllerState, current_update: int, from_state: str,
            to_state: str, reason: str, status: str = "confirmed") -> ControllerState:
    record = TransitionRecord(at_update=current_update, from_state=from_state,
                              to_state=to_state, reason=reason, status=status)
    return replace(state, history=(state.history + (record,))[-HISTORY_LIMIT:])


def _confirmed_transition(state: ControllerState, to_state: str, current_update: int,
                          reason: str) -> ControllerState:
    next_state = _record(state, current_update, state.state, to_state, reason)
    return replace(next_state, state=to_state, at_update=current_update,
                   transitions=state.transitions + 1,
                   consecutive_passes=0, hold_reason=None)


def _transition_reason(state: ControllerState, to_state: str, current_update: int,
                       reason: str) -> ControllerState:
    """Backwards-compatible alias for a confirmed transition."""
    return _confirmed_transition(state, to_state, current_update, reason)


def _cooldown_elapsed(state: ControllerState, config: ControllerSection,
                      current_update: int) -> bool:
    if state.transitions == 0:
        return True  # nothing has transitioned yet; no cooldown applies
    return (current_update - state.at_update) >= config.cooldown_updates


def _plasticity_evidence(state: ControllerState, metrics: ControllerMetrics,
                         score: float) -> bool:
    """Stable scores alone are not acquired improvement.

    A pass requires nonzero learning/plasticity evidence: measured parameter
    displacement (absolute or relative), or a score that actually improved
    during the window. Near-freezing must not masquerade as consolidation.
    """
    if metrics.parameter_displacement is not None and metrics.parameter_displacement > 0.0:
        return True
    if metrics.relative_parameter_displacement is not None \
            and metrics.relative_parameter_displacement > 0.0:
        return True
    if state.window_start_score is not None and score > state.window_start_score:
        return True
    return bool(metrics.score_improvements.get(state.acquiring_family or "", False))


def _multiplier_for(state: str, config: ControllerSection) -> float:
    return {
        "FORM": 1.0,
        "EXPAND": config.expand_lr_multiplier,
        "STABILIZE": config.stabilize_lr_multiplier,
        "REACQUIRE": config.reacquire_lr_multiplier,
        "HOLD": 1.0,
        "FIXED_SCHEDULE": 1.0,
    }[state]


def _metrics_are_invalid(metrics: ControllerMetrics, config: ControllerSection) -> bool:
    return bool(metrics.nonfinite_observed)


def _enter_hold(state: ControllerState, current_update: int, reason: str) -> ControllerState:
    next_state = _record(state, current_update, state.state, "HOLD", reason)
    return replace(next_state, state="HOLD", hold_return_state=state.state,
                   hold_reason=reason, at_update=current_update,
                   transitions=state.transitions + 1)


def transition(state: ControllerState, metrics: ControllerMetrics | None,
               config: ControllerSection, *, current_update: int,
               family_request: str | None = None) -> tuple[ControllerState, ControllerDecision]:
    """Pure controller transition, evaluated only at optimizer boundaries.

    Returns the next persisted state and the decision to apply. When a budget
    boundary (transition exhaustion) prevents application, the decision is
    returned with ``proposed_only=True`` and the proposal is recorded in
    history with status ``proposed``.
    """
    if config.mode == "disabled":
        return state, _decide("FORM", lr_multiplier=1.0, reason="controller_disabled")
    if config.mode == "fixed_schedule":
        return state, _decide("FIXED_SCHEDULE", lr_multiplier=1.0,
                              reason="fixed_schedule_control")

    if current_update < state.at_update:
        raise ControllerInputError("controller update counter moved backwards")
    if metrics is not None and config.controller_pool_id \
            and metrics.pool_id != config.controller_pool_id:
        raise PoolViolationError(
            f"metrics from pool {metrics.pool_id!r} do not match the declared "
            f"controller pool {config.controller_pool_id!r}")

    # HOLD exit: only a fresh, valid evaluation releases the pause.
    if state.state == "HOLD":
        if metrics is None:
            return state, _decide("HOLD", lr_multiplier=1.0,
                                  reason=f"hold_persists:{state.hold_reason}", pause=True)
        if _metrics_are_invalid(metrics, config):
            return state, _decide("HOLD", lr_multiplier=1.0,
                                  reason="hold_persists:invalid_metrics", pause=True)
        returned = replace(state, state=state.hold_return_state or "FORM",
                           hold_reason=None, hold_return_state=None,
                           last_metrics_update=metrics.at_update)
        return returned, _decide(returned.state,
                                 lr_multiplier=_multiplier_for(returned.state, config),
                                 reason="hold_released_with_valid_metrics")

    if metrics is not None:
        if _metrics_are_invalid(metrics, config):
            held = _enter_hold(state, current_update, "invalid_metrics")
            return held, _decide("HOLD", lr_multiplier=1.0, reason="invalid_metrics",
                                 pause=True, checkpoint=True)
        if not _validate_freshness(metrics, config, current_update):
            # Stale measurements can never count as passes; remain in state.
            return state, _decide(state.state, lr_multiplier=_multiplier_for(state.state, config),
                                  reason="stale_metrics_ignored")
        state = replace(state, last_metrics_update=metrics.at_update)

    if metrics is None:
        return state, _decide(state.state, lr_multiplier=_multiplier_for(state.state, config),
                              reason="no_metrics_this_boundary")

    # --- preservation first: a confirmed protected-family breach overrides
    #     everything. A single dipping evaluation only arms the pending
    #     collapse; confirmation (onset→confirm) hands over the state. ---
    missing_protected = [family for family in state.protected_families
                         if family not in metrics.family_scores]
    if missing_protected:
        held = _enter_hold(state, current_update,
                           f"missing_protected_metric:{sorted(missing_protected)[0]}")
        return held, _decide("HOLD", lr_multiplier=1.0,
                             reason=f"missing_protected_metric:{sorted(missing_protected)[0]}",
                             pause=True, checkpoint=True)
    pending = dict(state.pending_collapse)
    for family in state.protected_families:
        if metrics.family_scores[family] < config.reacquire_threshold:
            pending[family] = pending.get(family, 0) + 1
        elif metrics.family_scores[family] >= config.reacquire_threshold + config.exit_hysteresis:
            pending.pop(family, None)
        # between enter and exit thresholds: keep the current count (hysteresis band)
    confirmed = sorted(
        family for family, count in pending.items()
        if count >= config.collapse_confirmation_evaluations)
    pending = {family: count for family, count in pending.items() if count > 0}
    state = replace(state, pending_collapse=pending)
    if confirmed and state.state != "REACQUIRE":
        if state.transitions >= config.max_transitions:
            state = _record(state, current_update, state.state, state.state,
                            f"proposed_reacquire:{confirmed[0]}", status="proposed")
            return state, _decide(state.state, lr_multiplier=_multiplier_for(state.state, config),
                                  reason=f"proposed_reacquire:{confirmed[0]}",
                                  proposed_only=True)
        # A confirmed preservation collapse bypasses cooldown: the
        # confirmation window is itself the anti-chatter mechanism, and
        # protecting a qualified capability outranks transition discipline.
        next_state = replace(
            _confirmed_transition(state, "REACQUIRE", current_update,
                                  f"protected_family_below_threshold:{confirmed[0]}"),
            pending_collapse={})
        return next_state, _decide("REACQUIRE", lr_multiplier=config.reacquire_lr_multiplier,
                                   reason=f"protected_family_below_threshold:{confirmed[0]}",
                                   checkpoint=True)
    if confirmed:
        return state, _decide("REACQUIRE", lr_multiplier=config.reacquire_lr_multiplier,
                              reason=f"still_unstable:{confirmed[0]}")

    # --- family introduction request (EXPAND) ---
    if family_request is not None:
        if family_request == state.acquiring_family:
            pass  # redundant request; ignore
        elif state.state in ("FORM", "STABILIZE"):
            if state.transitions >= config.max_transitions:
                state = _record(state, current_update, state.state, state.state,
                                f"proposed_expand:{family_request}", status="proposed")
                return state, _decide(state.state,
                                      lr_multiplier=_multiplier_for(state.state, config),
                                      reason=f"proposed_expand:{family_request}",
                                      proposed_only=True)
            if not _cooldown_elapsed(state, config, current_update):
                return state, _decide(state.state,
                                      lr_multiplier=_multiplier_for(state.state, config),
                                      reason="cooldown_expand")
            # Protection is earned at qualification: the previously acquiring
            # family moves to the qualifying set (unless already protected);
            # it is promoted to protected only when its formation window is
            # later confirmed (STABILIZE entry).
            previously = state.acquiring_family
            qualifying = list(state.qualifying_families)
            if previously and previously not in state.protected_families \
                    and previously not in qualifying:
                qualifying.append(previously)
            next_state = replace(
                _confirmed_transition(state, "EXPAND", current_update,
                                      f"new_family_introduced:{family_request}"),
                acquiring_family=family_request, qualifying_families=tuple(qualifying),
                window_start_score=metrics.family_scores.get(family_request))
            return next_state, _decide("EXPAND", lr_multiplier=config.expand_lr_multiplier,
                                       reason=f"new_family_introduced:{family_request}",
                                       checkpoint=True)
        else:
            return state, _decide(state.state, lr_multiplier=_multiplier_for(state.state, config),
                                  reason="expand_request_rejected_in_state")

    if state.state == "EXPAND":
        # Formation counting under the acquisition family.
        acquiring = state.acquiring_family
        if acquiring is None or acquiring not in metrics.family_scores:
            # A missing acquiring metric delays formation counting; it is not
            # a preservation failure (qualification has not happened yet).
            return state, _decide("EXPAND", lr_multiplier=config.expand_lr_multiplier,
                                  reason="missing_acquiring_metric_no_pass")
        score = metrics.family_scores[acquiring]
        plasticity = _plasticity_evidence(state, metrics, score)
        if score >= config.formation_threshold and plasticity:
            state = replace(state, consecutive_passes=state.consecutive_passes + 1)
        elif score < config.formation_threshold - config.exit_hysteresis:
            state = replace(state, consecutive_passes=0)
        if state.consecutive_passes >= config.stabilize_window:
            if state.transitions >= config.max_transitions:
                state = _record(state, current_update, "EXPAND", "EXPAND",
                                "proposed_stabilize", status="proposed")
                return state, _decide("EXPAND", lr_multiplier=config.expand_lr_multiplier,
                                      reason="proposed_stabilize", proposed_only=True)
            if not _cooldown_elapsed(state, config, current_update):
                return state, _decide("EXPAND", lr_multiplier=config.expand_lr_multiplier,
                                      reason="cooldown_stabilize")
            # Qualification: the acquiring family now joins the protected set.
            protected = tuple(dict.fromkeys(
                (*state.protected_families, acquiring)))
            qualifying = tuple(f for f in state.qualifying_families if f != acquiring)
            next_state = replace(
                _confirmed_transition(state, "STABILIZE", current_update,
                                      "formation_window_met"),
                protected_families=protected, qualifying_families=qualifying)
            return next_state, _decide("STABILIZE", lr_multiplier=config.stabilize_lr_multiplier,
                                       reason="formation_window_met", checkpoint=True)
        return state, _decide("EXPAND", lr_multiplier=config.expand_lr_multiplier,
                              reason="acquiring_formation_in_progress")

    if state.state == "REACQUIRE":
        recovered = all(metrics.family_scores[family]
                        >= config.reacquire_threshold + config.exit_hysteresis
                        for family in state.protected_families)
        if recovered:
            state = replace(state, consecutive_recovery_passes=state.consecutive_recovery_passes + 1)
        else:
            state = replace(state, consecutive_recovery_passes=0)
        if state.consecutive_recovery_passes >= config.recovery_confirmation_evaluations:
            target = "STABILIZE" if state.acquiring_family else "FORM"
            next_state = replace(
                _confirmed_transition(state, target, current_update,
                                      "protected_families_recovered"),
                consecutive_recovery_passes=0)
            return next_state, _decide(target, lr_multiplier=_multiplier_for(target, config),
                                       reason="protected_families_recovered", checkpoint=True)
        return state, _decide("REACQUIRE", lr_multiplier=config.reacquire_lr_multiplier,
                              reason="reacquisition_in_progress")

    if state.state == "STABILIZE":
        acquiring = state.acquiring_family
        if acquiring is not None and acquiring in metrics.family_scores:
            score = metrics.family_scores[acquiring]
            if score < config.formation_threshold - config.exit_hysteresis:
                next_state = _confirmed_transition(state, "EXPAND", current_update,
                                                   "acquired_family_destabilized")
                # Destabilization after qualification suspends, but does not
                # erase, protection: the family stays protected while it
                # re-forms under EXPAND.
                return next_state, _decide("EXPAND", lr_multiplier=config.expand_lr_multiplier,
                                           reason="acquired_family_destabilized",
                                           checkpoint=True)
        return state, _decide("STABILIZE", lr_multiplier=config.stabilize_lr_multiplier,
                              reason="stabilizing")

    if state.state == "FORM":
        return state, _decide("FORM", lr_multiplier=1.0,
                              reason="awaiting_first_family_introduction")

    return state, _decide(state.state, lr_multiplier=_multiplier_for(state.state, config),
                          reason="no_applicable_transition")
