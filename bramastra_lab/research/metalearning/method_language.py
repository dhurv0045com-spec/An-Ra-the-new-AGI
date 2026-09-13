"""Typed learning-method language, validator and compiler (M23).

The first method language is intentionally small: schedule expressions over
committed counters, replay weights, admitted objective coefficients and one
bounded gradient transformation. The interpreter exposes no network, shell,
arbitrary imports or evaluator mutation. Compilation of the same method and
config yields the same identity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from bramastra_lab.research.contracts.core import content_identity


class MethodError(ValueError):
    """A method proposal violates the typed language contract."""


SCHEDULE_COUNTERS = frozenset({"optimizer_updates", "presentations", "episodes"})
TRANSFORM_KINDS = frozenset({"clip_norm", "grad_scale"})


@dataclass(frozen=True)
class SchedulePoint:
    at_update: int
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.at_update, int) or isinstance(self.at_update, bool) \
                or self.at_update < 0:
            raise MethodError("schedule point at_update must be a nonnegative integer")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) \
                or not math.isfinite(float(self.value)) or float(self.value) <= 0:
            raise MethodError("schedule point value must be finite and positive")


@dataclass(frozen=True)
class ScheduleExpression:
    counter: str
    points: tuple[SchedulePoint, ...]

    def __post_init__(self) -> None:
        if self.counter not in SCHEDULE_COUNTERS:
            raise MethodError(
                f"schedule counter must be one of {sorted(SCHEDULE_COUNTERS)}; "
                "sealed accuracy and evaluator metrics are prohibited inputs")
        if not self.points:
            raise MethodError("schedule requires at least one point")
        updates = [point.at_update for point in self.points]
        if updates != sorted(set(updates)):
            raise MethodError("schedule points must be strictly increasing in update")

    def value_at(self, counter_value: int) -> float:
        current = self.points[0]
        for point in self.points:
            if point.at_update <= counter_value:
                current = point
        return current.value


@dataclass(frozen=True)
class ReplayWeights:
    family_weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.family_weights, Mapping) or not self.family_weights:
            raise MethodError("replay weights must be a nonempty mapping")
        for family, weight in self.family_weights.items():
            if not isinstance(family, str) or not family:
                raise MethodError("family names must be nonempty")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) \
                    or not math.isfinite(float(weight)) or float(weight) <= 0:
                raise MethodError(f"family weight for {family!r} must be finite and positive")


@dataclass(frozen=True)
class ObjectiveCoefficients:
    token: float = 1.0
    world: float = 0.0
    action: float = 0.0
    value: float = 0.0
    pair: float = 0.0
    pg: float = 0.0

    def __post_init__(self) -> None:
        for name in ("token", "world", "action", "value", "pair", "pg"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)) or float(value) < 0:
                raise MethodError(f"objective coefficient {name} must be finite and >= 0")


@dataclass(frozen=True)
class GradientTransform:
    kind: str
    bound: float
    state_tensors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in TRANSFORM_KINDS:
            raise MethodError(f"transform kind must be one of {sorted(TRANSFORM_KINDS)}")
        if isinstance(self.bound, bool) or not isinstance(self.bound, (int, float)) \
                or not math.isfinite(float(self.bound)) or float(self.bound) <= 0:
            raise MethodError("transform bound must be finite and positive")
        if self.kind == "grad_scale" and not self.state_tensors:
            # A per-parameter scale transformation must declare its state
            # tensors and initialization (RSI §2).
            raise MethodError(
                "grad_scale transformations must declare their state tensors")


@dataclass(frozen=True)
class MethodProgram:
    """One typed method proposal: exactly which components change."""

    schedule: ScheduleExpression | None = None
    replay_weights: ReplayWeights | None = None
    objective_coefficients: ObjectiveCoefficients | None = None
    gradient_transform: GradientTransform | None = None
    expected_gain: float = 0.0
    predicted_cost: float = 0.0
    failure_conditions: tuple[str, ...] = ()
    comparison_protocol: str = ""
    eligibility_scope: tuple[str, ...] = ("training",)

    def __post_init__(self) -> None:
        if self.schedule is None and self.replay_weights is None \
                and self.objective_coefficients is None and self.gradient_transform is None:
            raise MethodError(
                "a proposal must declare at least one changed component; the "
                "unchanged parent/no-change proposal is represented separately")
        if isinstance(self.expected_gain, bool) or not isinstance(self.expected_gain, (int, float)) \
                or not math.isfinite(float(self.expected_gain)):
            raise MethodError("expected_gain must be finite")
        if isinstance(self.predicted_cost, bool) or not isinstance(self.predicted_cost, (int, float)) \
                or not math.isfinite(float(self.predicted_cost)) or self.predicted_cost < 0:
            raise MethodError("predicted_cost must be finite and nonnegative")
        if not isinstance(self.comparison_protocol, str) or not self.comparison_protocol:
            raise MethodError("comparison_protocol must reference a declared protocol")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule": None if self.schedule is None else {
                "counter": self.schedule.counter,
                "points": [{"at_update": p.at_update, "value": p.value}
                           for p in self.schedule.points]},
            "replay_weights": None if self.replay_weights is None
            else dict(self.replay_weights.family_weights),
            "objective_coefficients": None if self.objective_coefficients is None
            else dict(self.objective_coefficients.__dict__),
            "gradient_transform": None if self.gradient_transform is None
            else {"kind": self.gradient_transform.kind,
                  "bound": self.gradient_transform.bound,
                  "state_tensors": list(self.gradient_transform.state_tensors)},
            "expected_gain": self.expected_gain,
            "predicted_cost": self.predicted_cost,
            "failure_conditions": list(self.failure_conditions),
            "comparison_protocol": self.comparison_protocol,
            "eligibility_scope": list(self.eligibility_scope),
        }

    def identity(self) -> str:
        return content_identity(self.to_dict())


NO_CHANGE = "no_change"


def parse_no_change() -> str:
    """The unchanged parent/no-change proposal is always representable."""
    return NO_CHANGE


def compile_method(program: MethodProgram, *, runtime_config: Mapping[str, Any]) -> dict[str, Any]:
    """Compile a program into a bound method instance with exact identity.

    Pure transformation: no network, no imports, no evaluator access. The
    same program and runtime config always compile to the same identity.
    """
    if not isinstance(runtime_config, Mapping) or not runtime_config:
        raise MethodError("runtime config must be a nonempty mapping")
    compiled = {
        "schema": "bramastra-compiled-method/v1",
        "program_identity": program.identity(),
        "runtime_config_identity": content_identity(dict(runtime_config)),
        "schedule": None if program.schedule is None else {
            "counter": program.schedule.counter,
            "kind": "piecewise_constant",
            "points": [(p.at_update, p.value) for p in program.schedule.points]},
        "replay_weights": None if program.replay_weights is None
        else dict(program.replay_weights.family_weights),
        "objective_coefficients": None if program.objective_coefficients is None
        else dict(program.objective_coefficients.__dict__),
        "gradient_transform": None if program.gradient_transform is None
        else {"kind": program.gradient_transform.kind,
              "bound": program.gradient_transform.bound,
              "state": {name: 1.0 for name in program.gradient_transform.state_tensors}},
        "state_migration": {},
    }
    compiled["identity"] = content_identity(
        {key: value for key, value in compiled.items() if key != "identity"})
    return compiled
