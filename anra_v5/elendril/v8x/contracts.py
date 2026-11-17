"""Versioned model, observation, outcome, and capacity-control contracts for ELENDRIL V8-X1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8.contracts import (
    ACTION_19_LABELS,
    OP_VOCABULARY,
    POLICY_MODES,
    SYMBOLS,
    _fcap_extra_params,
    _pointer_extra_params,
)
from v5_contracts.model_spec import ModelSpec

X1_MAX_SEQUENCE = 640
X1_OBSERVATION_VERSION = "elendril-v8x-public-obs-v1"
X1_VERIFIER_IDENTITY = "elendril-v8x-verifier-v1"
X1_DATASET_SCHEMA = "x1-d1h-v1"
X1_D2_DATASET_SCHEMA = "x1-d2-pairs-v1"

X1_ARCHITECTURES = (
    "B0",
    "J",
    "F",
    "F-cap",
    "P",
    "P-capR",
    "R",
    "R-noedge",
    "U-1",
    "U-2",
    "U-4",
    "U2-untied",
)
X1_CORE_ARCHITECTURES = ("B0", "J", "F", "F-cap", "P", "P-capR", "R", "R-noedge")
X1_U_ARCHITECTURES = ("U-1", "U-2", "U-4", "U2-untied")
X1_OBJECTIVES = ("standard", "PAIR-CE", "PAIR-EQ", "PAIR-EQ-CF")
X1_FAMILIES = ("X01", "X02", "X03", "X04", "X05", "X06", "X07", "X08", "X09", "X10", "X11", "X12")
X1_RENDERER_FAMILIES = ("compact_tuples", "named_statements", "keyed_records", "heldout_narrative")

DEFAULT_X1_POINTER_DIM = 64
DEFAULT_X1_SLOT_DIM = 128
DEFAULT_X1_NUM_SLOTS = 8
DEFAULT_X1_EDGE_HIDDEN = 64
DEFAULT_X1_AGG_ITERS = 2
DEFAULT_X1_MSG_ITERS = 2
DEFAULT_X1_RECURRENT_DIM = 64
DEFAULT_X1_SOURCE_RANKS = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}


@dataclass(frozen=True, slots=True)
class X1EvaluationOutcome:
    """Immutable X1 epistemic episode verifier outcome."""

    correct: bool
    terminal_status: str
    charged_cost: float
    epistemic_supported: bool
    lucky_unsupported_guess: bool
    verifier_identity: str = X1_VERIFIER_IDENTITY

    def __post_init__(self) -> None:
        if type(self.correct) is not bool or type(self.epistemic_supported) is not bool or type(self.lucky_unsupported_guess) is not bool:
            raise ValueError("X1EvaluationOutcome boolean fields must be strict bools")
        if not isinstance(self.terminal_status, str) or not self.terminal_status:
            raise ValueError("terminal_status must be nonempty")
        if not isinstance(self.charged_cost, (int, float)) or not math.isfinite(self.charged_cost):
            raise ValueError("charged_cost must be finite")
        if self.verifier_identity != X1_VERIFIER_IDENTITY:
            raise ValueError("unsupported verifier_identity")


def _r_common_params(width: int, pointer_dim: int, slot_dim: int, num_slots: int) -> int:
    # tok_ln(2w) + tok_proj(w*s) + slots_init(k*s) + slot_q/k/v(3s^2) + slot_gru(6s^2 + 6s)
    # + agg_ln(2s) + agg_mlp(4s^2 + 3s) + goal_q_proj(w*s) + read_ln(2s) + r_to_ptr(s*d)
    return (
        2 * width
        + 2 * width * slot_dim
        + num_slots * slot_dim
        + slot_dim * pointer_dim
        + 13 * slot_dim * slot_dim
        + 13 * slot_dim
    )


def _r_edge_params(slot_dim: int, edge_hidden: int) -> int:
    # rel_ln(2s) + edge_mlp1(3s*eh + eh) + edge_mlp2(eh + 1) + msg_proj(s^2) + rel_gru(6s^2 + 6s)
    return 7 * slot_dim * slot_dim + 8 * slot_dim + (3 * slot_dim + 2) * edge_hidden + 1


def _r_noedge_msg_params(slot_dim: int, noedge_mlp_dim: int) -> int:
    # noedge_ln(2s) + noedge_mlp1(s*m + m) + noedge_mlp2(m*s + s) + noedge_gru(6s^2 + 6s)
    return 6 * slot_dim * slot_dim + 9 * slot_dim + (2 * slot_dim + 1) * noedge_mlp_dim


def _p_cap_r_params(width: int, pointer_dim: int, cap_r_hidden: int) -> int:
    # cap_r_in(w*c + c) + cap_r_out(c*d + d)
    return cap_r_hidden * (width + pointer_dim + 1) + pointer_dim


def _u_recurrent_block_params(width: int, recurrent_dim: int) -> int:
    # u_q/k/v(3*w*du) + u_out(du*w + w) + u_gru(6w^2 + 6w)
    return 3 * (width * recurrent_dim) + (recurrent_dim * width + width) + (6 * width * width + 6 * width)


def solve_x1_capacity_controls(
    base_spec: ModelSpec,
    *,
    pointer_dim: int = DEFAULT_X1_POINTER_DIM,
    slot_dim: int = DEFAULT_X1_SLOT_DIM,
    num_slots: int = DEFAULT_X1_NUM_SLOTS,
    slot_edge_hidden: int = DEFAULT_X1_EDGE_HIDDEN,
) -> dict[str, float | int]:
    """Deterministically choose integer R-noedge and P-capR hidden widths matching R within <= 1%."""

    base_spec.assert_valid()
    w = base_spec.width
    f_added = (w * 4 + 4) + (w * 16 + 16)
    p_ptr_added = _pointer_extra_params(w, pointer_dim)
    r_extra = _r_common_params(w, pointer_dim, slot_dim, num_slots) + _r_edge_params(slot_dim, slot_edge_hidden)
    r_total_added = f_added + p_ptr_added + r_extra

    best_noedge: tuple[int, int, float] | None = None
    common = _r_common_params(w, pointer_dim, slot_dim, num_slots)
    for m in range(1, max(32, 8 * slot_dim) + 1):
        ne_extra = common + _r_noedge_msg_params(slot_dim, m)
        ne_added = f_added + p_ptr_added + ne_extra
        rel = abs(ne_added - r_total_added) / r_total_added
        if best_noedge is None or abs(ne_added - r_total_added) < abs(best_noedge[1] - r_total_added):
            best_noedge = (m, ne_added, rel)

    best_capr: tuple[int, int, float] | None = None
    for c in range(1, max(32, 8 * w) + 1):
        cr_added = f_added + p_ptr_added + _p_cap_r_params(w, pointer_dim, c)
        rel = abs(cr_added - r_total_added) / r_total_added
        if best_capr is None or abs(cr_added - r_total_added) < abs(best_capr[1] - r_total_added):
            best_capr = (c, cr_added, rel)

    assert best_noedge is not None and best_capr is not None
    if best_noedge[2] > 0.01:
        raise ValueError(
            f"cannot match R-noedge added parameters within 1% (best m={best_noedge[0]}, rel={best_noedge[2]:.4f})"
        )
    if best_capr[2] > 0.01:
        raise ValueError(
            f"cannot match P-capR added parameters within 1% (best c={best_capr[0]}, rel={best_capr[2]:.4f})"
        )
    return {
        "r_added_parameters": r_total_added,
        "r_extra_above_p": r_extra,
        "noedge_mlp_dim": best_noedge[0],
        "r_noedge_added_parameters": best_noedge[1],
        "r_noedge_rel_diff": best_noedge[2],
        "cap_r_hidden": best_capr[0],
        "p_cap_r_added_parameters": best_capr[1],
        "p_cap_r_rel_diff": best_capr[2],
    }


def choose_fcap_bottleneck_x1(base_spec: ModelSpec, pointer_dim: int) -> int:
    w = base_spec.width
    target = _pointer_extra_params(w, pointer_dim)
    best_r = 1
    best_diff = float("inf")
    for r in range(1, 4 * w + 1):
        diff = abs(_fcap_extra_params(w, r) - target)
        if diff < best_diff:
            best_diff = diff
            best_r = r
    return best_r


@dataclass(frozen=True, slots=True)
class X1ModelSpec:
    """Versioned X1 architecture specification anchored cleanly onto a V5 ModelSpec backbone."""

    architecture: str
    base_spec: ModelSpec
    pointer_dim: int = 0
    bottleneck_width: int = 0
    cap_r_hidden: int = 0
    slot_dim: int = 0
    num_slots: int = 0
    slot_edge_hidden: int = 0
    agg_iterations: int = 0
    msg_iterations: int = 0
    noedge_mlp_dim: int = 0
    recurrent_dim: int = 0
    recurrent_steps: int = 0
    copy_enabled: bool = True
    policy_interface: str = "typed-19-v1"
    vocabulary_size: int = 260
    output_mapping: tuple[str, ...] = ACTION_19_LABELS
    max_sequence_length: int = X1_MAX_SEQUENCE
    schema: str = "elendril-x1-model-spec-v1"

    def __post_init__(self) -> None:
        self.assert_valid()

    def assert_valid(self) -> None:
        if self.schema != "elendril-x1-model-spec-v1":
            raise ValueError("unsupported X1ModelSpec schema")
        if self.architecture not in X1_ARCHITECTURES:
            raise ValueError(f"unsupported X1 architecture: {self.architecture!r}")
        self.base_spec.assert_valid()
        if self.vocabulary_size != self.base_spec.vocabulary_size:
            raise ValueError("vocabulary_size must match base_spec")
        if type(self.max_sequence_length) is not int or not 16 <= self.max_sequence_length <= self.base_spec.context_length:
            raise ValueError(
                f"max_sequence_length {self.max_sequence_length} must lie within [16, {self.base_spec.context_length}]"
            )
        if self.architecture == "B0":
            if self.policy_interface != "byte-autoregressive-v1" or self.output_mapping != ("utf8-byte-plus-eos",):
                raise ValueError("B0 requires byte-autoregressive-v1 interface")
            if any(
                getattr(self, f) != 0
                for f in (
                    "pointer_dim",
                    "bottleneck_width",
                    "cap_r_hidden",
                    "slot_dim",
                    "num_slots",
                    "slot_edge_hidden",
                    "agg_iterations",
                    "msg_iterations",
                    "noedge_mlp_dim",
                    "recurrent_dim",
                    "recurrent_steps",
                )
            ):
                raise ValueError("B0 cannot carry extra head dimensions")
        else:
            if self.policy_interface != "typed-19-v1" or self.output_mapping != ACTION_19_LABELS:
                raise ValueError("typed X1 heads require typed-19-v1 interface and 19 labels")

        if self.architecture == "F-cap" and self.bottleneck_width <= 0:
            raise ValueError("F-cap requires positive bottleneck_width")
        if self.architecture in ("P", "P-capR", "R", "R-noedge"):
            if self.pointer_dim <= 0:
                raise ValueError(f"{self.architecture} requires positive pointer_dim")
        if self.architecture == "P-capR" and self.cap_r_hidden <= 0:
            raise ValueError("P-capR requires positive cap_r_hidden")
        if self.architecture in ("R", "R-noedge"):
            if min(self.slot_dim, self.num_slots, self.agg_iterations, self.msg_iterations) <= 0:
                raise ValueError(f"{self.architecture} requires positive slot/iteration dimensions")
            if self.architecture == "R" and self.slot_edge_hidden <= 0:
                raise ValueError("R requires positive slot_edge_hidden")
            if self.architecture == "R-noedge" and self.noedge_mlp_dim <= 0:
                raise ValueError("R-noedge requires positive noedge_mlp_dim")
        if self.architecture in X1_U_ARCHITECTURES:
            if self.recurrent_dim <= 0 or self.recurrent_steps <= 0:
                raise ValueError(f"{self.architecture} requires positive recurrent_dim and recurrent_steps")

    def added_parameters(self) -> int:
        w = self.base_spec.width
        if self.architecture == "B0":
            return 0
        if self.architecture == "J":
            return w * 19 + 19
        f_added = (w * 4 + 4) + (w * 16 + 16)
        if self.architecture == "F":
            return f_added
        if self.architecture == "F-cap":
            return f_added + _fcap_extra_params(w, self.bottleneck_width)
        if self.architecture in ("P", "P-capR", "R", "R-noedge"):
            total = f_added + _pointer_extra_params(w, self.pointer_dim)
            if self.architecture == "P-capR":
                total += _p_cap_r_params(w, self.pointer_dim, self.cap_r_hidden)
            elif self.architecture == "R":
                total += _r_common_params(w, self.pointer_dim, self.slot_dim, self.num_slots)
                total += _r_edge_params(self.slot_dim, self.slot_edge_hidden)
            elif self.architecture == "R-noedge":
                total += _r_common_params(w, self.pointer_dim, self.slot_dim, self.num_slots)
                total += _r_noedge_msg_params(self.slot_dim, self.noedge_mlp_dim)
            return total
        if self.architecture in ("U-1", "U-2", "U-4"):
            return f_added + _u_recurrent_block_params(w, self.recurrent_dim)
        if self.architecture == "U2-untied":
            return f_added + 2 * _u_recurrent_block_params(w, self.recurrent_dim)
        raise AssertionError(f"unreachable architecture: {self.architecture}")

    def total_parameters(self) -> int:
        return self.base_spec.parameter_receipt().total + self.added_parameters()

    def f_head_seed(self, init_seed: int) -> int:
        return int(digest(["elendril-x1-f-seed-v1", init_seed, self.base_spec.sha256()]), 16) % (2**32)

    def pointer_seed(self, init_seed: int) -> int:
        return int(digest(["elendril-x1-ptr-seed-v1", init_seed, self.base_spec.sha256(), self.pointer_dim]), 16) % (2**32)

    def module_seed(self, init_seed: int) -> int:
        return int(digest(["elendril-x1-mod-seed-v1", self.architecture, init_seed, self.sha256()]), 16) % (2**32)

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
        self.assert_valid()
        data = asdict(self)
        data["base_spec"] = self.base_spec.canonical()
        data["base_spec_sha256"] = self.base_spec.sha256()
        data["output_mapping"] = list(self.output_mapping)
        return digest(data)

    def json(self) -> str:
        return canonical(self.canonical())


def build_x1_s20_base(*, context_length: int = X1_MAX_SEQUENCE) -> ModelSpec:
    """Return the qualified V5 S20 backbone at context_length=640 (19,871,616 parameters)."""

    from dataclasses import replace

    s20 = replace(MODEL_PROFILES["S20"], context_length=int(context_length))
    if s20.parameter_receipt().total != 19_871_616:
        raise ValueError(f"unexpected S20 backbone parameter count: {s20.parameter_receipt().total}")
    return s20


def build_x1_model_spec(
    architecture: str,
    *,
    base_spec: ModelSpec | None = None,
    pointer_dim: int = DEFAULT_X1_POINTER_DIM,
    slot_dim: int = DEFAULT_X1_SLOT_DIM,
    num_slots: int = DEFAULT_X1_NUM_SLOTS,
    slot_edge_hidden: int = DEFAULT_X1_EDGE_HIDDEN,
    agg_iterations: int = DEFAULT_X1_AGG_ITERS,
    msg_iterations: int = DEFAULT_X1_MSG_ITERS,
    recurrent_dim: int = DEFAULT_X1_RECURRENT_DIM,
    max_sequence_length: int | None = None,
) -> X1ModelSpec:
    """Build a canonical X1ModelSpec with capacity-matched R-noedge and P-capR controls."""

    base = base_spec or build_x1_s20_base()
    max_seq = base.context_length if max_sequence_length is None else int(max_sequence_length)

    if architecture == "B0":
        return X1ModelSpec(
            architecture="B0",
            base_spec=base,
            copy_enabled=False,
            policy_interface="byte-autoregressive-v1",
            output_mapping=("utf8-byte-plus-eos",),
            max_sequence_length=max_seq,
        )
    if architecture in ("J", "F"):
        return X1ModelSpec(architecture=architecture, base_spec=base, max_sequence_length=max_seq)
    if architecture == "F-cap":
        bw = choose_fcap_bottleneck_x1(base, pointer_dim)
        return X1ModelSpec(architecture="F-cap", base_spec=base, bottleneck_width=bw, max_sequence_length=max_seq)
    if architecture == "P":
        return X1ModelSpec(architecture="P", base_spec=base, pointer_dim=pointer_dim, max_sequence_length=max_seq)
    if architecture in ("R", "R-noedge", "P-capR"):
        matched = solve_x1_capacity_controls(
            base,
            pointer_dim=pointer_dim,
            slot_dim=slot_dim,
            num_slots=num_slots,
            slot_edge_hidden=slot_edge_hidden,
        )
        if architecture == "P-capR":
            return X1ModelSpec(
                architecture="P-capR",
                base_spec=base,
                pointer_dim=pointer_dim,
                cap_r_hidden=int(matched["cap_r_hidden"]),
                max_sequence_length=max_seq,
            )
        if architecture == "R":
            return X1ModelSpec(
                architecture="R",
                base_spec=base,
                pointer_dim=pointer_dim,
                slot_dim=slot_dim,
                num_slots=num_slots,
                slot_edge_hidden=slot_edge_hidden,
                agg_iterations=agg_iterations,
                msg_iterations=msg_iterations,
                max_sequence_length=max_seq,
            )
        return X1ModelSpec(
            architecture="R-noedge",
            base_spec=base,
            pointer_dim=pointer_dim,
            slot_dim=slot_dim,
            num_slots=num_slots,
            agg_iterations=agg_iterations,
            msg_iterations=msg_iterations,
            noedge_mlp_dim=int(matched["noedge_mlp_dim"]),
            max_sequence_length=max_seq,
        )
    if architecture in X1_U_ARCHITECTURES:
        steps = {"U-1": 1, "U-2": 2, "U-4": 4, "U2-untied": 2}[architecture]
        return X1ModelSpec(
            architecture=architecture,
            base_spec=base,
            recurrent_dim=recurrent_dim,
            recurrent_steps=steps,
            max_sequence_length=max_seq,
        )
    raise ValueError(f"unsupported X1 architecture: {architecture!r}")


__all__ = [
    "ACTION_19_LABELS",
    "DEFAULT_X1_AGG_ITERS",
    "DEFAULT_X1_EDGE_HIDDEN",
    "DEFAULT_X1_MSG_ITERS",
    "DEFAULT_X1_NUM_SLOTS",
    "DEFAULT_X1_POINTER_DIM",
    "DEFAULT_X1_RECURRENT_DIM",
    "DEFAULT_X1_SLOT_DIM",
    "DEFAULT_X1_SOURCE_RANKS",
    "OP_VOCABULARY",
    "POLICY_MODES",
    "SYMBOLS",
    "X1EvaluationOutcome",
    "X1ModelSpec",
    "X1_ARCHITECTURES",
    "X1_CORE_ARCHITECTURES",
    "X1_D2_DATASET_SCHEMA",
    "X1_DATASET_SCHEMA",
    "X1_FAMILIES",
    "X1_MAX_SEQUENCE",
    "X1_OBJECTIVES",
    "X1_OBSERVATION_VERSION",
    "X1_RENDERER_FAMILIES",
    "X1_U_ARCHITECTURES",
    "X1_VERIFIER_IDENTITY",
    "build_x1_model_spec",
    "build_x1_s20_base",
    "choose_fcap_bottleneck_x1",
    "solve_x1_capacity_controls",
]
