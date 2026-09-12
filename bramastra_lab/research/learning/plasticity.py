"""Evidence-driven plasticity controller (B06).

A pure state machine connecting capability formation, preservation and
reacquisition. ``transition`` is a pure function of (state, metrics, config):
it may set a schedule multiplier, replay mixture or checkpoint request; it
never mutates tensors and never reads a sealed split. Metrics come only from
the declared controller pool — development, measurement and sealed inputs
reject. Missing or stale measurements can never count as formation passes,
and a collapse never automatically lowers the learning rate (the ARK-010
finding). The controller is off by default; the disabled and fixed-schedule
controls remain available for causal comparison.

Evidence provenance: separation of preservation from acquisition and the
explicit REACQUIRE state follow ARK-007R/ARK-010; the adaptive policy itself
is an unvalidated integration choice and is scientifically unqualified.
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
    present and fresh. ``parameter_displacement`` and ``score_improvements``
    are the learning/plasticity evidence required before stable scores may be
    treated as formed capability.
    """

    pool_id: str
    at_update: int
    family_scores: Mapping[str, float]
    parameter_displacement: float | None = None
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
        if self.parameter_displacement is not None:
            displacement = self.parameter_displacement
            if not isinstance(displacement, (int, float)) or isinstance(displacement, bool) \
                    or displacement < 0:
                raise ControllerInputError("parameter_displacement must be nonnegative")


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

    def to_dict(self) -> dict[str, Any]:
        return {"at_update": self.at_update, "from_state": self.from_state,
                "to_state": self.to_state, "reason": self.reason}

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
    acquiring_family: str | None = None
    window_start_score: float | None = None
    hold_return_state: str | None = None
    hold_reason: str | None = None
    last_metrics_update: int | None = None
    history: tuple[TransitionRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ControllerInputError(f"unknown controller state {self.state!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state, "consecutive_passes": self.consecutive_passes,
            "at_update": self.at_update, "transitions": self.transitions,
            "protected_families": list(self.protected_families),
            "acquiring_family": self.acquiring_family,
            "window_start_score": self.window_start_score,
            "hold_return_state": self.hold_return_state,
            "hold_reason": self.hold_reason,
            "last_metrics_update": self.last_metrics_update,
            "history": [record.to_dict() for record in self.history],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ControllerState":
        known = {"state", "consecutive_passes", "at_update", "transitions",
                 "protected_families", "acquiring_family", "window_start_score",
                 "hold_return_state", "hold_reason", "last_metrics_update", "history"}
        unknown = set(raw) - known
        if unknown:
            raise ControllerInputError(f"controller state has unknown fields: {sorted(unknown)}")
        return cls(
            state=raw["state"], consecutive_passes=raw["consecutive_passes"],
            at_update=raw["at_update"], transitions=raw["transitions"],
            protected_families=tuple(raw["protected_families"]),
            acquiring_family=raw["acquiring_family"],
            window_start_score=raw["window_start_score"],
            hold_return_state=raw["hold_return_state"],
            hold_reason=raw["hold_reason"],
            last_metrics_update=raw["last_metrics_update"],
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
    """Return True when metrics are fresh enough to count as a pass."""
    return (current_update - metrics.at_update) <= config.max_metrics_age_updates


def _transition_reason(state: ControllerState, to_state: str, current_update: int,
                       reason: str) -> ControllerState:
    record = TransitionRecord(at_update=current_update, from_state=state.state,
                              to_state=to_state, reason=reason)
    history = (state.history + (record,))[-HISTORY_LIMIT:]
    return replace(state, state=to_state, at_update=current_update,
                   transitions=state.transitions + 1,
                   consecutive_passes=0, hold_reason=None, history=history)


def transition(state: ControllerState, metrics: ControllerMetrics | None,
               config: ControllerSection, *, current_update: int,
               family_request: str | None = None) -> tuple[ControllerState, ControllerDecision]:
    """Pure controller transition, evaluated only at optimizer boundaries.

    Returns the next persisted state and the decision to apply. When a budget
    boundary (transition exhaustion) prevents application, the decision is
    returned with ``proposed_only=True`` and the intended change is recorded
    in the returned state's history as a proposal.
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
                                  reason=f"hold_persists:invalid_metrics", pause=True)
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

    # --- preservation first: a protected-family failure overrides everything ---
    missing_protected = [family for family in state.protected_families
                         if family not in metrics.family_scores]
    if missing_protected:
        # Without a protected measurement we cannot verify preservation.
        held = _enter_hold(state, current_update,
                           f"missing_protected_metric:{sorted(missing_protected)[0]}")
        return held, _decide("HOLD", lr_multiplier=1.0,
                             reason=f"missing_protected_metric:{sorted(missing_protected)[0]}",
                             pause=True, checkpoint=True)
    collapsed = sorted(
        family for family in state.protected_families
        if metrics.family_scores[family] < config.reacquire_threshold)
    if collapsed and state.state != "REACQUIRE":
        if state.transitions >= config.max_transitions:
            return state, _decide(state.state, lr_multiplier=_multiplier_for(state.state, config),
                                  reason=f"proposed_reacquire:{collapsed[0]}",
                                  proposed_only=True)
        if not _cooldown_elapsed(state, config, current_update):
            return state, _decide(state.state, lr_multiplier=_multiplier_for(state.state, config),
                                  reason=f"cooldown_reacquire:{collapsed[0]}")
        next_state = _transition_reason(state, "REACQUIRE", current_update,
                                        f"protected_family_below_threshold:{collapsed[0]}")
        return next_state, _decide("REACQUIRE", lr_multiplier=config.reacquire_lr_multiplier,
                                   reason=f"protected_family_below_threshold:{collapsed[0]}",
                                   checkpoint=True)
    if collapsed:
        return state, _decide("REACQUIRE", lr_multiplier=config.reacquire_lr_multiplier,
                              reason=f"still_collapsed:{collapsed[0]}")

    # --- family introduction request (EXPAND) ---
    if family_request is not None:
        if family_request == state.acquiring_family:
            pass  # redundant request; ignore
        elif state.state in ("FORM", "STABILIZE"):
            if state.transitions >= config.max_transitions:
                return state, _decide(state.state,
                                      lr_multiplier=_multiplier_for(state.state, config),
                                      reason=f"proposed_expand:{family_request}",
                                      proposed_only=True)
            if not _cooldown_elapsed(state, config, current_update):
                return state, _decide(state.state,
                                      lr_multiplier=_multiplier_for(state.state, config),
                                      reason="cooldown_expand")
            protected = tuple(dict.fromkeys(
                (*state.protected_families,
                 *([state.acquiring_family] if state.acquiring_family else []),
                 *([family for family in metrics.family_scores
                    if family not in (state.acquiring_family, family_request)]))))
            next_state = replace(
                _transition_reason(state, "EXPAND", current_update,
                                   f"new_family_introduced:{family_request}"),
                acquiring_family=family_request, protected_families=protected,
                window_start_score=metrics.family_scores.get(family_request))
            return next_state, _decide("EXPAND", lr_multiplier=config.expand_lr_multiplier,
                                       reason=f"new_family_introduced:{family_request}",
                                       checkpoint=True)
        else:
            return state, _decide(state.state, lr_multiplier=_multiplier_for(state.state, config),
                                  reason="expand_request_rejected_in_state")

    if state.state == "EXPAND":
        # Move to formation counting under the acquisition family.
        acquiring = state.acquiring_family
        if acquiring is None or acquiring not in metrics.family_scores:
            held = _enter_hold(state, current_update, "missing_acquiring_metric")
            return held, _decide("HOLD", lr_multiplier=1.0, reason="missing_acquiring_metric",
                                 pause=True, checkpoint=True)
        score = metrics.family_scores[acquiring]
        plasticity = _plasticity_evidence(state, metrics, score)
        if score >= config.formation_threshold and plasticity:
            state = replace(state, consecutive_passes=state.consecutive_passes + 1)
        elif score < config.formation_threshold - config.exit_hysteresis:
            state = replace(state, consecutive_passes=0)
        if state.consecutive_passes >= config.stabilize_window:
            if state.transitions >= config.max_transitions:
                return state, _decide("EXPAND", lr_multiplier=config.expand_lr_multiplier,
                                      reason="proposed_stabilize", proposed_only=True)
            if not _cooldown_elapsed(state, config, current_update):
                return state, _decide("EXPAND", lr_multiplier=config.expand_lr_multiplier,
                                      reason="cooldown_stabilize")
            next_state = _transition_reason(state, "STABILIZE", current_update,
                                            "formation_window_met")
            return next_state, _decide("STABILIZE", lr_multiplier=config.stabilize_lr_multiplier,
                                       reason="formation_window_met", checkpoint=True)
        return state, _decide("EXPAND", lr_multiplier=config.expand_lr_multiplier,
                              reason="acquiring_formation_in_progress")

    if state.state == "REACQUIRE":
        recovered = all(metrics.family_scores[family]
                        >= config.reacquire_threshold + config.exit_hysteresis
                        for family in state.protected_families)
        if recovered:
            target = "STABILIZE" if state.acquiring_family else "FORM"
            next_state = _transition_reason(state, target, current_update,
                                            "protected_families_recovered")
            return next_state, _decide(target, lr_multiplier=_multiplier_for(target, config),
                                       reason="protected_families_recovered", checkpoint=True)
        return state, _decide("REACQUIRE", lr_multiplier=config.reacquire_lr_multiplier,
                              reason="reacquisition_in_progress")

    if state.state == "STABILIZE":
        acquiring = state.acquiring_family
        if acquiring is not None and acquiring in metrics.family_scores:
            score = metrics.family_scores[acquiring]
            if score < config.formation_threshold - config.exit_hysteresis:
                next_state = _transition_reason(state, "EXPAND", current_update,
                                                "acquired_family_destabilized")
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


def _cooldown_elapsed(state: ControllerState, config: ControllerSection,
                      current_update: int) -> bool:
    if state.transitions == 0:
        return True  # nothing has transitioned yet; no cooldown applies
    return (current_update - state.at_update) >= config.cooldown_updates


def _plasticity_evidence(state: ControllerState, metrics: ControllerMetrics,
                         score: float) -> bool:
    """Stable scores alone are not acquired improvement.

    A pass requires nonzero learning/plasticity evidence: measured parameter
    displacement, or a score that actually improved during the window.
    """
    if metrics.parameter_displacement is not None and metrics.parameter_displacement > 0.0:
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
    record = TransitionRecord(at_update=current_update, from_state=state.state,
                              to_state="HOLD", reason=reason)
    return replace(state, state="HOLD", hold_return_state=state.state,
                   hold_reason=reason, at_update=current_update,
                   transitions=state.transitions + 1,
                   history=(state.history + (record,))[-HISTORY_LIMIT:])
