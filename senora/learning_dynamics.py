"""Causal Learning Dynamics (CLD) & Cognitive Acquisition Dynamics (CAD) Engine.

Measures how cognitive capabilities emerge, regress, interact, or phase-transition
across training time under controlled pretraining treatments:

    C(T, t, c)  = raw-Core capability under treatment T at token time t for family c
    ΔC(t, c)    = C(TREATMENT, t, c) - C(CONTROL, t, c)

Provides analytical tools for:
- Tokens-to-Threshold (TTT) and Acquisition Ordering
- Phase Transition detection (max slope, curvature, transition width)
- Sample Efficiency (AULC, TE-AUC)
- Synthetic-to-Natural Transfer Lag
- Loss-Matched Cognition Gap (decoupling loss reduction from cognitive computation)
- Cognitive Forgetting Index (CFI)
- Early Run Triage Predictor
- Training Intervention Record (Triquetra bridge)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


CHECKPOINT_SCHEDULE_TOKENS: tuple[int, ...] = (
    0,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    2_0000_000,
    35_000_000,
    50_000_000,
)


@dataclass(frozen=True, slots=True)
class CognitiveAcquisitionReceipt:
    """Snapshot of neural cognition, substrate performance, and provenance at a training milestone."""
    schema: str
    checkpoint_sha256: str
    arm_name: str
    seed: int
    tokens_seen: int
    global_update: int
    training_flops_6nd: float
    substrate_validation_loss: float
    raw_core_macro_accuracy: float
    natural_analogue_macro_accuracy: float
    structural_ood_macro_accuracy: float
    query_sensitivity_flip_rate: float
    pair_invariance_stable_rate: float
    family_accuracies: Mapping[str, float]
    triquetra_trace_pointer: str | None = None

    def canonical(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PhaseTransitionMetrics:
    family: str
    max_instantaneous_slope: float
    inflection_token_milestone: int
    transition_width_tokens: int
    is_sharp_transition: bool


@dataclass(frozen=True, slots=True)
class TrajectoryMetricSummary:
    """Comprehensive causal learning dynamics comparison across control and treatment."""
    schema: str
    treatment_arm: str
    control_arm: str
    seed: int
    tokens_evaluated: tuple[int, ...]
    area_under_learning_curve_treatment: Mapping[str, float]
    area_under_learning_curve_control: Mapping[str, float]
    treatment_effect_auc: Mapping[str, float]
    tokens_to_threshold_treatment: Mapping[str, int | None]
    tokens_to_threshold_control: Mapping[str, int | None]
    acquisition_order_treatment: list[str]
    acquisition_order_control: list[str]
    transfer_lag_tokens: int | None
    cognitive_forgetting_index: Mapping[str, float]
    phase_transitions: Mapping[str, PhaseTransitionMetrics]
    loss_matched_gap_at_threshold: float | None
    early_triage_decision: str
    early_triage_rationale: str

    def canonical(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingInterventionRecord:
    """Connects training experience with cognitive trajectory and Triquetra failure geometry."""
    schema: str
    intervention_id: str
    cognition_fraction: float
    objective_type: str
    model_family: str
    parameter_count: int
    seed: int
    milestone_tokens: int
    capability_vector: Mapping[str, float]
    substrate_loss: float
    triquetra_bridge_trace_path: str


class TrajectoryAnalysisEngine:
    """Analytical engine for longitudinal cognitive acquisition trajectories."""

    @staticmethod
    def compute_aulc(tokens: Sequence[int], values: Sequence[float]) -> float:
        """Compute Area Under Learning Curve via trapezoidal integration normalized by max tokens."""
        if len(tokens) < 2 or len(tokens) != len(values):
            return 0.0
        max_t = tokens[-1]
        if max_t <= 0:
            return 0.0
        area = 0.0
        for i in range(len(tokens) - 1):
            dt = tokens[i + 1] - tokens[i]
            avg_v = (values[i] + values[i + 1]) / 2.0
            area += avg_v * dt
        return round(area / max_t, 4)

    @staticmethod
    def compute_tokens_to_threshold(
        tokens: Sequence[int],
        values: Sequence[float],
        threshold: float = 0.50,
    ) -> int | None:
        """Return the earliest token milestone where performance crosses and maintains threshold."""
        for t, v in zip(tokens, values):
            if v >= threshold:
                return t
        return None

    @staticmethod
    def detect_phase_transition(
        tokens: Sequence[int],
        values: Sequence[float],
        family: str,
        threshold_sharpness: float = 2.0e-7,
    ) -> PhaseTransitionMetrics:
        """Detect sharp changes in capability acquisition slope and curvature."""
        if len(tokens) < 3:
            return PhaseTransitionMetrics(
                family=family,
                max_instantaneous_slope=0.0,
                inflection_token_milestone=0,
                transition_width_tokens=0,
                is_sharp_transition=False,
            )

        slopes = []
        for i in range(len(tokens) - 1):
            dt = max(1, tokens[i + 1] - tokens[i])
            dv = values[i + 1] - values[i]
            slopes.append((dv / dt, tokens[i]))

        max_slope, inflection_t = max(slopes, key=lambda s: s[0])

        # Transition width from 10% to 90% of max value
        max_val = max(values)
        t10 = None
        t90 = None
        for t, v in zip(tokens, values):
            if t10 is None and v >= 0.10 * max_val:
                t10 = t
            if t90 is None and v >= 0.90 * max_val:
                t90 = t
                break

        width = (t90 - t10) if (t10 is not None and t90 is not None) else 0
        is_sharp = max_slope >= threshold_sharpness

        return PhaseTransitionMetrics(
            family=family,
            max_instantaneous_slope=round(max_slope, 8),
            inflection_token_milestone=inflection_t,
            transition_width_tokens=width,
            is_sharp_transition=is_sharp,
        )

    @staticmethod
    def compute_forgetting_index(values: Sequence[float]) -> float:
        """Measure peak capability minus final capability: max(0, peak - final)."""
        if not values:
            return 0.0
        peak = max(values)
        final = values[-1]
        return round(max(0.0, peak - final), 4)

    @classmethod
    def analyze_trajectories(
        cls,
        treatment_receipts: Sequence[CognitiveAcquisitionReceipt],
        control_receipts: Sequence[CognitiveAcquisitionReceipt],
        *,
        capability_threshold: float = 0.50,
        loss_matched_target: float = 2.50,
    ) -> TrajectoryMetricSummary:
        """Compute full comparative learning dynamics between control and treatment."""
        if len(treatment_receipts) != len(control_receipts):
            raise ValueError("Treatment and control longitudinal receipt counts must match")

        tokens = [r.tokens_seen for r in treatment_receipts]
        families = list(treatment_receipts[0].family_accuracies.keys())

        aulc_treat: dict[str, float] = {}
        aulc_ctrl: dict[str, float] = {}
        te_auc: dict[str, float] = {}
        ttt_treat: dict[str, int | None] = {}
        ttt_ctrl: dict[str, int | None] = {}
        cfi: dict[str, float] = {}
        pts: dict[str, PhaseTransitionMetrics] = {}

        # 1. Macro & Family Trajectories
        treat_macros = [r.raw_core_macro_accuracy for r in treatment_receipts]
        ctrl_macros = [r.raw_core_macro_accuracy for r in control_receipts]
        aulc_treat["macro"] = cls.compute_aulc(tokens, treat_macros)
        aulc_ctrl["macro"] = cls.compute_aulc(tokens, ctrl_macros)
        te_auc["macro"] = round(aulc_treat["macro"] - aulc_ctrl["macro"], 4)
        ttt_treat["macro"] = cls.compute_tokens_to_threshold(tokens, treat_macros, capability_threshold)
        ttt_ctrl["macro"] = cls.compute_tokens_to_threshold(tokens, ctrl_macros, capability_threshold)
        cfi["macro"] = cls.compute_forgetting_index(treat_macros)
        pts["macro"] = cls.detect_phase_transition(tokens, treat_macros, "macro")

        for f in families:
            t_vals = [r.family_accuracies.get(f, 0.0) for r in treatment_receipts]
            c_vals = [r.family_accuracies.get(f, 0.0) for r in control_receipts]
            aulc_treat[f] = cls.compute_aulc(tokens, t_vals)
            aulc_ctrl[f] = cls.compute_aulc(tokens, c_vals)
            te_auc[f] = round(aulc_treat[f] - aulc_ctrl[f], 4)
            ttt_treat[f] = cls.compute_tokens_to_threshold(tokens, t_vals, capability_threshold)
            ttt_ctrl[f] = cls.compute_tokens_to_threshold(tokens, c_vals, capability_threshold)
            cfi[f] = cls.compute_forgetting_index(t_vals)
            pts[f] = cls.detect_phase_transition(tokens, t_vals, f)

        # 2. Acquisition Ordering
        order_treat = sorted(
            [f for f in families if ttt_treat[f] is not None],
            key=lambda f: (ttt_treat[f], -aulc_treat[f]),
        )
        order_ctrl = sorted(
            [f for f in families if ttt_ctrl[f] is not None],
            key=lambda f: (ttt_ctrl[f], -aulc_ctrl[f]),
        )

        # 3. Transfer Lag (Synthetic Dev Macro vs Natural Analogue Macro)
        t_nat_vals = [r.natural_analogue_macro_accuracy for r in treatment_receipts]
        ttt_synth = ttt_treat["macro"]
        ttt_nat = cls.compute_tokens_to_threshold(tokens, t_nat_vals, capability_threshold)
        transfer_lag = (ttt_nat - ttt_synth) if (ttt_synth is not None and ttt_nat is not None) else None

        # 4. Loss-Matched Cognition Gap
        # Find checkpoint where treatment and control have loss closest to loss_matched_target
        best_t_idx = min(range(len(treatment_receipts)), key=lambda i: abs(treatment_receipts[i].substrate_validation_loss - loss_matched_target))
        best_c_idx = min(range(len(control_receipts)), key=lambda i: abs(control_receipts[i].substrate_validation_loss - loss_matched_target))
        loss_matched_gap = round(
            treatment_receipts[best_t_idx].raw_core_macro_accuracy - control_receipts[best_c_idx].raw_core_macro_accuracy,
            4,
        )

        # 5. Early Triage Predictor (Evaluation at 10M tokens)
        m10_idx = None
        for i, t in enumerate(tokens):
            if t == 10_000_000:
                m10_idx = i
                break

        if m10_idx is not None:
            m10_delta = treatment_receipts[m10_idx].raw_core_macro_accuracy - control_receipts[m10_idx].raw_core_macro_accuracy
            if m10_delta >= 0.10:
                triage = "CONTINUE_HIGH_CONFIDENCE"
                rationale = f"At 10M tokens, causal treatment effect is robust ({m10_delta:+.3f} >= +0.10); proceed to full 50M."
            elif m10_delta <= 0.02:
                triage = "EARLY_STOPPING_CANDIDATE"
                rationale = f"At 10M tokens, causal treatment effect is negligible ({m10_delta:+.3f} <= +0.02); arm is an early-termination candidate."
            else:
                triage = "CONTINUE_INTERMEDIATE"
                rationale = f"At 10M tokens, treatment effect is moderate ({m10_delta:+.3f}); continue observation."
        else:
            triage = "INSUFFICIENT_MILESTONES"
            rationale = "10M token milestone receipt missing."

        return TrajectoryMetricSummary(
            schema="senora-trajectory-metrics/v1",
            treatment_arm=treatment_receipts[0].arm_name,
            control_arm=control_receipts[0].arm_name,
            seed=treatment_receipts[0].seed,
            tokens_evaluated=tuple(tokens),
            area_under_learning_curve_treatment=aulc_treat,
            area_under_learning_curve_control=aulc_ctrl,
            treatment_effect_auc=te_auc,
            tokens_to_threshold_treatment=ttt_treat,
            tokens_to_threshold_control=ttt_ctrl,
            acquisition_order_treatment=order_treat,
            acquisition_order_control=order_ctrl,
            transfer_lag_tokens=transfer_lag,
            cognitive_forgetting_index=cfi,
            phase_transitions=pts,
            loss_matched_gap_at_threshold=loss_matched_gap,
            early_triage_decision=triage,
            early_triage_rationale=rationale,
        )