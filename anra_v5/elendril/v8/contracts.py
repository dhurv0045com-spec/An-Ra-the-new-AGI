"""Immutable actor, policy, outcome, and versioned model contracts for V8."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from v5_contracts.model_spec import ModelSpec

SYMBOLS = "ABCDEFGHIJKLMNOP"
OP_VOCABULARY = ("ANSWER", "QUERY", "RETRIEVE", "ABSTAIN")
ACTION_19_LABELS = tuple(SYMBOLS) + ("QUERY", "RETRIEVE", "ABSTAIN")
POLICY_MODES = ("raw", "grammar", "availability-mask", "direct-answer-only")
ARCHITECTURES = ("B0", "J", "F", "P", "F-cap", "U")
OBSERVATION_VERSION = "elendril-v8-public-obs-v1"
VERIFIER_IDENTITY = "elendril-v8-verifier-v1"
SCREEN_MAX_SEQUENCE = 512
DEFAULT_POINTER_DIM = 64


@dataclass(frozen=True, slots=True)
class PublicInput:
    """Actor-facing public input; structurally excludes truth, IDs, and hidden state."""

    text_utf8: str
    query_available: bool
    memory_available: bool
    action_budget: int
    observation_version: str = OBSERVATION_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.text_utf8, str) or not self.text_utf8:
            raise ValueError("PublicInput.text_utf8 must be a nonempty string")
        if type(self.query_available) is not bool or type(self.memory_available) is not bool:
            raise ValueError("PublicInput availability flags must be booleans")
        if type(self.action_budget) is not int or self.action_budget < 1:
            raise ValueError("PublicInput.action_budget must be a positive integer")
        if self.observation_version != OBSERVATION_VERSION:
            raise ValueError("unsupported PublicInput observation_version")


@dataclass(frozen=True, slots=True)
class PolicyProposal:
    """Immutable actor proposal logged separately for unconstrained and executed steps."""

    operation: str
    argument: str | None
    joint_log_probability: float
    raw_output: str
    model_calls: int
    evaluated_positions: int
    mode: str
    model_identity: str

    def __post_init__(self) -> None:
        if self.operation not in (*OP_VOCABULARY, "INVALID"):
            raise ValueError(f"unsupported proposal operation: {self.operation!r}")
        if self.operation == "ANSWER":
            if self.argument not in SYMBOLS or len(str(self.argument)) != 1:
                raise ValueError("ANSWER proposal requires single argument in A-P")
        elif self.operation in ("RETRIEVE", "ABSTAIN", "INVALID"):
            if self.argument is not None:
                raise ValueError(f"{self.operation} proposal must not carry an answer argument")
        if not isinstance(self.joint_log_probability, (int, float)) or (
            not math.isfinite(self.joint_log_probability) and self.joint_log_probability != float("-inf")
        ):
            raise ValueError("joint_log_probability must be finite or -inf")
        if not isinstance(self.raw_output, str):
            raise ValueError("raw_output must be a string")
        if type(self.model_calls) is not int or self.model_calls < 0:
            raise ValueError("model_calls must be a non-negative integer")
        if type(self.evaluated_positions) is not int or self.evaluated_positions < 0:
            raise ValueError("evaluated_positions must be a non-negative integer")
        if self.mode not in POLICY_MODES:
            raise ValueError(f"unsupported policy mode: {self.mode!r}")
        if not isinstance(self.model_identity, str) or not self.model_identity:
            raise ValueError("model_identity must be a nonempty string")


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """Immutable verifier outcome separate from the actor."""

    correct: bool
    terminal_status: str
    charged_cost: float
    verifier_identity: str = VERIFIER_IDENTITY

    def __post_init__(self) -> None:
        if type(self.correct) is not bool:
            raise ValueError("EvaluationOutcome.correct must be a bool")
        if not isinstance(self.terminal_status, str) or not self.terminal_status:
            raise ValueError("terminal_status must be a nonempty string")
        if not isinstance(self.charged_cost, (int, float)) or not math.isfinite(self.charged_cost):
            raise ValueError("charged_cost must be finite")
        if self.verifier_identity != VERIFIER_IDENTITY:
            raise ValueError("unsupported verifier_identity")


def _fcap_extra_params(width: int, bottleneck_width: int) -> int:
    # h' = h + W2 SiLU(W1 h + b1) + b2
    return (width * bottleneck_width + bottleneck_width) + (bottleneck_width * width + width)


def _pointer_extra_params(width: int, pointer_dim: int) -> int:
    # q = hWq, K = HWk (no bias), g = sigmoid(w_g . h + b_g)
    return 2 * (width * pointer_dim) + (width + 1)


def choose_fcap_bottleneck(base_spec: ModelSpec, pointer_dim: int = DEFAULT_POINTER_DIM) -> tuple[int, int, float]:
    """Enumerate positive integer bottleneck widths matching P within <= 1% added params."""

    base_spec.assert_valid()
    if type(pointer_dim) is not int or pointer_dim <= 0:
        raise ValueError("pointer_dim must be positive")
    f_added = (base_spec.width * 4 + 4) + (base_spec.width * 16 + 16)
    p_added = f_added + _pointer_extra_params(base_spec.width, pointer_dim)
    best: tuple[int, int, float] | None = None
    for r in range(1, 4 * base_spec.width + 1):
        fcap_added = f_added + _fcap_extra_params(base_spec.width, r)
        rel_diff = abs(fcap_added - p_added) / p_added
        if best is None or abs(fcap_added - p_added) < abs(best[1] - p_added):
            best = (r, fcap_added, rel_diff)
    assert best is not None
    if best[2] > 0.01:
        raise ValueError(
            f"no positive bottleneck width matches P within 1% added parameters "
            f"(closest r={best[0]}, fcap_added={best[1]}, p_added={p_added}, rel_diff={best[2]:.4f})"
        )
    return best


@dataclass(frozen=True, slots=True)
class V8ModelSpec:
    """Versioned V8 model specification including base backbone and head parameters."""

    architecture: str
    base_spec: ModelSpec
    pointer_dim: int = 0
    bottleneck_width: int = 0
    recurrent_dim: int = 0
    recurrent_steps: int = 0
    copy_enabled: bool = True
    policy_interface: str = "byte-autoregressive-v1"
    vocabulary_size: int = 260
    output_mapping: tuple[str, ...] = ("utf8-byte-plus-eos",)
    max_sequence_length: int = SCREEN_MAX_SEQUENCE
    schema: str = "elendril-v8-model-spec-v1"

    def __post_init__(self) -> None:
        self.assert_valid()

    def assert_valid(self) -> None:
        if self.schema != "elendril-v8-model-spec-v1":
            raise ValueError("unsupported V8ModelSpec schema")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"unknown V8 architecture: {self.architecture!r}")
        self.base_spec.assert_valid()
        if self.vocabulary_size != self.base_spec.vocabulary_size:
            raise ValueError("vocabulary_size must match base_spec")
        if type(self.max_sequence_length) is not int or not 16 <= self.max_sequence_length <= self.base_spec.context_length:
            raise ValueError("max_sequence_length must be within base_spec.context_length")
        if self.architecture == "B0":
            if self.policy_interface != "byte-autoregressive-v1" or self.output_mapping != ("utf8-byte-plus-eos",):
                raise ValueError("B0 requires byte-autoregressive-v1 interface")
            if self.pointer_dim != 0 or self.bottleneck_width != 0 or self.recurrent_dim != 0 or self.recurrent_steps != 0:
                raise ValueError("B0 cannot carry extra head dimensions")
        else:
            if self.policy_interface != "typed-19-v1" or self.output_mapping != ACTION_19_LABELS:
                raise ValueError("typed heads require typed-19-v1 interface and 19 action labels")
        if self.architecture in ("J", "F"):
            if self.pointer_dim != 0 or self.bottleneck_width != 0 or self.recurrent_dim != 0 or self.recurrent_steps != 0:
                raise ValueError(f"{self.architecture} cannot set pointer/bottleneck/recurrent dimensions")
        elif self.architecture == "P":
            if type(self.copy_enabled) is not bool:
                raise ValueError("copy_enabled must be bool")
            if self.copy_enabled and (type(self.pointer_dim) is not int or self.pointer_dim <= 0):
                raise ValueError("P with copy_enabled=True requires positive pointer_dim")
            if self.bottleneck_width != 0 or self.recurrent_dim != 0 or self.recurrent_steps != 0:
                raise ValueError("P cannot set bottleneck or recurrent dimensions")
        elif self.architecture == "F-cap":
            if type(self.bottleneck_width) is not int or self.bottleneck_width <= 0:
                raise ValueError("F-cap requires positive bottleneck_width")
            if self.pointer_dim != 0 or self.recurrent_dim != 0 or self.recurrent_steps != 0:
                raise ValueError("F-cap cannot set pointer or recurrent dimensions")
        elif self.architecture == "U":
            if type(self.recurrent_dim) is not int or self.recurrent_dim <= 0:
                raise ValueError("U requires positive recurrent_dim")
            if type(self.recurrent_steps) is not int or self.recurrent_steps not in (1, 2, 3, 4):
                raise ValueError("U requires recurrent_steps in {1, 2, 3, 4}")
            if self.bottleneck_width != 0:
                raise ValueError("U cannot set bottleneck_width")

    def added_parameters(self) -> int:
        w = self.base_spec.width
        if self.architecture == "B0":
            return 0
        if self.architecture == "J":
            return w * 19 + 19
        f_added = (w * 4 + 4) + (w * 16 + 16)
        if self.architecture == "F":
            return f_added
        if self.architecture == "P":
            if not self.copy_enabled:
                return f_added
            return f_added + _pointer_extra_params(w, self.pointer_dim)
        if self.architecture == "F-cap":
            return f_added + _fcap_extra_params(w, self.bottleneck_width)
        if self.architecture == "U":
            # q/k/v projections of width recurrent_dim + out proj to w + GRUCell(w, w)
            attn_read = 3 * (w * self.recurrent_dim) + (self.recurrent_dim * w + w)
            gru = 6 * w * w + 6 * w
            return f_added + attn_read + gru
        raise AssertionError("unreachable architecture")

    def total_parameters(self) -> int:
        return self.base_spec.parameter_receipt().total + self.added_parameters()

    def extra_head_seed(self, init_seed: int) -> int:
        if type(init_seed) is not int or not 0 <= init_seed < 2**32:
            raise ValueError("init_seed must be uint32")
        return int(digest(["elendril-v8-head-seed-v1", self.architecture, init_seed, self.sha256()]), 16) % (2**32)

    def canonical(self) -> dict[str, object]:
        self.assert_valid()
        data = asdict(self)
        data["base_spec"] = self.base_spec.canonical()
        data["base_spec_sha256"] = self.base_spec.sha256()
        data["output_mapping"] = list(self.output_mapping)
        data["added_parameters"] = self.added_parameters()
        data["total_parameters"] = self.total_parameters()
        return data

    def sha256(self) -> str:
        # Compute without recursive sha256 call
        self.assert_valid()
        data = asdict(self)
        data["base_spec"] = self.base_spec.canonical()
        data["base_spec_sha256"] = self.base_spec.sha256()
        data["output_mapping"] = list(self.output_mapping)
        return digest(data)

    def json(self) -> str:
        return canonical(self.canonical())


def build_v8_model_spec(
    architecture: str,
    *,
    base_spec: ModelSpec | None = None,
    pointer_dim: int = DEFAULT_POINTER_DIM,
    copy_enabled: bool = True,
    recurrent_steps: int = 3,
    recurrent_dim: int = 64,
    max_sequence_length: int = SCREEN_MAX_SEQUENCE,
) -> V8ModelSpec:
    """Build a validated V8ModelSpec for a named architecture."""

    spec = base_spec or MODEL_PROFILES["S20"]
    if architecture == "B0":
        return V8ModelSpec(
            architecture="B0",
            base_spec=spec,
            policy_interface="byte-autoregressive-v1",
            vocabulary_size=spec.vocabulary_size,
            output_mapping=("utf8-byte-plus-eos",),
            max_sequence_length=max_sequence_length,
        )
    if architecture in ("J", "F"):
        return V8ModelSpec(
            architecture=architecture,
            base_spec=spec,
            policy_interface="typed-19-v1",
            vocabulary_size=spec.vocabulary_size,
            output_mapping=ACTION_19_LABELS,
            max_sequence_length=max_sequence_length,
        )
    if architecture == "P":
        return V8ModelSpec(
            architecture="P",
            base_spec=spec,
            pointer_dim=pointer_dim if copy_enabled else 0,
            copy_enabled=copy_enabled,
            policy_interface="typed-19-v1",
            vocabulary_size=spec.vocabulary_size,
            output_mapping=ACTION_19_LABELS,
            max_sequence_length=max_sequence_length,
        )
    if architecture == "F-cap":
        width_r, _, _ = choose_fcap_bottleneck(spec, pointer_dim=pointer_dim)
        return V8ModelSpec(
            architecture="F-cap",
            base_spec=spec,
            bottleneck_width=width_r,
            policy_interface="typed-19-v1",
            vocabulary_size=spec.vocabulary_size,
            output_mapping=ACTION_19_LABELS,
            max_sequence_length=max_sequence_length,
        )
    if architecture == "U":
        return V8ModelSpec(
            architecture="U",
            base_spec=spec,
            recurrent_dim=recurrent_dim,
            recurrent_steps=recurrent_steps,
            policy_interface="typed-19-v1",
            vocabulary_size=spec.vocabulary_size,
            output_mapping=ACTION_19_LABELS,
            max_sequence_length=max_sequence_length,
        )
    raise ValueError(f"unsupported architecture: {architecture!r}")


__all__ = [
    "ACTION_19_LABELS",
    "ARCHITECTURES",
    "DEFAULT_POINTER_DIM",
    "EvaluationOutcome",
    "OBSERVATION_VERSION",
    "OP_VOCABULARY",
    "POLICY_MODES",
    "PolicyProposal",
    "PublicInput",
    "SCREEN_MAX_SEQUENCE",
    "SYMBOLS",
    "V8ModelSpec",
    "VERIFIER_IDENTITY",
    "build_v8_model_spec",
    "choose_fcap_bottleneck",
]
