"""Architecture-agnostic protocols and immutable schemas for Planthir."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch

from anra_v5.elendril.codec import digest
from v5_evaluation.firewall import LEAK_MARKERS as V5_LEAK_MARKERS

PLANTHIR_LEAK_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(V5_LEAK_MARKERS)
        | {
            "expected=",
            "expected:",
            "hidden=",
            "hidden:",
            "ground_truth=",
            "ground_truth:",
        }
    )
)


def detect_truth_leaks(rendered_text: str) -> list[str]:
    """Return any truth-channel leak markers detected inside an actor-facing string."""

    violations: list[str] = []
    lowered = rendered_text.lower()
    stripped = lowered.rstrip()
    for marker in PLANTHIR_LEAK_MARKERS:
        if marker in lowered:
            if marker == "answer:" and stripped.endswith("answer:") and lowered.count("answer:") == 1:
                continue
            violations.append(f"truth_leak_marker:{marker}")
    return violations


@dataclass(frozen=True, slots=True)
class UniversalObservation:
    """Architecture-agnostic, truth-firewalled actor observation."""

    obs_id: str
    rendered_text: str
    available_tools: dict[str, bool]
    consumed_tools: frozenset[str]
    decision_index: int
    budget_remaining: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_invariants(self) -> list[str]:
        """Verify structural invariants on actor observations before model execution."""

        violations: list[str] = []
        if not isinstance(self.obs_id, str) or not self.obs_id:
            violations.append("empty_obs_id")
        if not isinstance(self.rendered_text, str) or not self.rendered_text.strip():
            violations.append("empty_rendered_text")
        else:
            violations.extend(detect_truth_leaks(self.rendered_text))

        if type(self.decision_index) is not int or self.decision_index < 0:
            violations.append(f"invalid_decision_index:{self.decision_index!r}")
        if type(self.budget_remaining) is not int or self.budget_remaining < 1:
            violations.append(f"non_positive_budget_remaining:{self.budget_remaining!r}")

        for tool_name in sorted(self.consumed_tools):
            if self.available_tools.get(tool_name, False):
                violations.append(f"consumed_tool_still_available:{tool_name}")

        for key in ("expected", "gold", "gold_answer", "hidden", "truth"):
            if key in self.metadata:
                violations.append(f"forbidden_metadata_key:{key}")

        return violations


@dataclass(frozen=True, slots=True)
class UniversalAction:
    """Action proposal with full distribution and internal telemetry."""

    operation: str
    argument: str | None
    log_probability: float
    action_distribution: dict[str, float]
    mode: str
    raw_operation: str
    raw_argument: str | None
    telemetry: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("operation must be nonempty")
        if not isinstance(self.log_probability, (int, float)) or (
            not math.isfinite(self.log_probability) and self.log_probability != float("-inf")
        ):
            raise ValueError("log_probability must be finite or -inf")


@dataclass(frozen=True, slots=True)
class SubgoalProbeSpec:
    """Intermediate causal subgoal derived from a multi-step world."""

    subgoal_id: str
    stage_name: str
    hop_index: int
    modified_world: dict[str, Any]
    expected_operation: str
    expected_argument: str | None
    description: str = ""


@runtime_checkable
class EnvironmentKernel(Protocol):
    """Single source-of-truth state machine shared across teacher, rollout, and verifier."""

    kernel_id: str

    def initial_state(self, world: Mapping[str, Any]) -> Any: ...

    def observe(self, state: Any) -> UniversalObservation: ...

    def ideal_action(self, state: Any) -> tuple[str, str | None]: ...

    def step(self, state: Any, operation: str, argument: str | None) -> Any: ...

    def verify(
        self, world: Mapping[str, Any], actions: Sequence[tuple[str, str | None]]
    ) -> dict[str, Any]: ...

    def decompose_subgoals(self, world: Mapping[str, Any]) -> Sequence[SubgoalProbeSpec]: ...


@runtime_checkable
class ObservablePolicy(Protocol):
    """Policy interface exposing distributions and hidden representations for diagnostics."""

    policy_id: str

    def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction: ...

    def extract_representations(
        self, obs_batch: Sequence[UniversalObservation]
    ) -> dict[str, torch.Tensor]: ...


@dataclass(frozen=True, slots=True)
class FalsifierSpec:
    """Explicit falsification criterion that can refute a hypothesis."""

    falsifier_id: str
    description: str
    metric_name: str
    comparator: str
    threshold: float

    def evaluate(self, observed_value: float | None) -> dict[str, Any]:
        if observed_value is None or not math.isfinite(observed_value):
            return {
                "falsifier_id": self.falsifier_id,
                "status": "INDETERMINATE",
                "observed": None,
                "threshold": self.threshold,
                "comparator": self.comparator,
            }
        ops = {
            ">=": observed_value >= self.threshold,
            ">": observed_value > self.threshold,
            "<=": observed_value <= self.threshold,
            "<": observed_value < self.threshold,
        }
        if self.comparator not in ops:
            raise ValueError(f"unsupported comparator: {self.comparator!r}")
        passed = ops[self.comparator]
        return {
            "falsifier_id": self.falsifier_id,
            "status": "NOT_FALSIFIED" if passed else "FALSIFIED",
            "observed": float(observed_value),
            "threshold": self.threshold,
            "comparator": self.comparator,
        }


@dataclass(frozen=True, slots=True)
class CausalContrastSpec:
    """Controlled contrast isolating one architectural, data, or algorithmic factor."""

    contrast_id: str
    factor_name: str
    control_arm: str
    treatment_arm: str
    held_constant: tuple[str, ...]
    expected_mechanism_signature: str


@dataclass(frozen=True, slots=True)
class HypothesisSpec:
    """Preregistered scientific hypothesis with explicit falsifiers."""

    hypothesis_id: str
    statement: str
    mechanism_target: str
    falsifiers: tuple[FalsifierSpec, ...]
    expected_signatures: tuple[str, ...] = ()

    def sha256(self) -> str:
        return digest(["planthir-hypothesis-v1", asdict(self)])


@dataclass(frozen=True, slots=True)
class ExperimentRegistration:
    """Universal registration record across architectures, scales, and research programs."""

    experiment_id: str
    program: str
    architecture_family: str
    parameter_count: int
    dataset_identity: str
    source_commit: str
    hypotheses: tuple[HypothesisSpec, ...]
    contrasts: tuple[CausalContrastSpec, ...]
    budget: dict[str, Any] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    status: str = "REGISTERED"
    schema: str = "anra-planthir-registration-v1"

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        return digest(["planthir-registration-v1", self.canonical_dict()])


__all__ = [
    "CausalContrastSpec",
    "EnvironmentKernel",
    "ExperimentRegistration",
    "FalsifierSpec",
    "HypothesisSpec",
    "ObservablePolicy",
    "PLANTHIR_LEAK_MARKERS",
    "SubgoalProbeSpec",
    "UniversalAction",
    "UniversalObservation",
    "detect_truth_leaks",
]
