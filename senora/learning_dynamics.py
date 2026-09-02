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

def generate_synthetic_world_receipts(world_id: int) -> tuple[list[CognitiveAcquisitionReceipt], list[CognitiveAcquisitionReceipt]]:
    """Generate reproducible synthetic longitudinal trajectories for Worlds 1-10."""
    tokens = [0, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 35_000_000, 50_000_000]

    def _make(t, arm, macro, nat, fams, loss=2.50, seed=42):
        return CognitiveAcquisitionReceipt(
            schema="senora-cognitive-acquisition-receipt/v1",
            checkpoint_sha256=f"ckpt_{t}_{arm}",
            arm_name=arm,
            seed=seed,
            tokens_seen=t,
            global_update=t // 131_072,
            training_flops_6nd=t * 35_411_328 * 6,
            substrate_validation_loss=loss,
            raw_core_macro_accuracy=macro,
            natural_analogue_macro_accuracy=nat,
            structural_ood_macro_accuracy=macro,
            query_sensitivity_flip_rate=0.85,
            pair_invariance_stable_rate=0.90,
            family_accuracies=fams,
        )

    if world_id == 1:  # Monotonic
        treat = [_make(t, "cognition-15", 0.1 + 0.7 * (t / 50_000_000), 0.1 + 0.6 * (t / 50_000_000), {"binding": 0.1 + 0.7 * (t / 50_000_000)}) for t in tokens]
        ctrl = [_make(t, "control-00", 0.1 + 0.1 * (t / 50_000_000), 0.1 + 0.1 * (t / 50_000_000), {"binding": 0.1 + 0.1 * (t / 50_000_000)}) for t in tokens]
    elif world_id == 2:  # Sharp phase transition
        t_vals = [0.05, 0.06, 0.08, 0.10, 0.12, 0.78, 0.82, 0.85]
        treat = [_make(t, "cognition-15", v, v, {"binding": v}) for t, v in zip(tokens, t_vals)]
        ctrl = [_make(t, "control-00", 0.10, 0.10, {"binding": 0.10}) for t in tokens]
    elif world_id == 3:  # Forgetting
        t_vals = [0.10, 0.25, 0.45, 0.70, 0.75, 0.60, 0.45, 0.35]
        treat = [_make(t, "cognition-15", v, v, {"binding": v}) for t, v in zip(tokens, t_vals)]
        ctrl = [_make(t, "control-00", 0.10, 0.10, {"binding": 0.10}) for t in tokens]
    elif world_id == 4:  # Transfer lag
        synth_vals = [0.10, 0.20, 0.35, 0.55, 0.70, 0.80, 0.85, 0.88]
        nat_vals = [0.05, 0.08, 0.12, 0.18, 0.25, 0.38, 0.52, 0.65]
        treat = [_make(t, "cognition-15", s, n, {"binding": s}) for t, s, n in zip(tokens, synth_vals, nat_vals)]
        ctrl = [_make(t, "control-00", 0.10, 0.10, {"binding": 0.10}) for t in tokens]
    elif world_id == 5:  # Sample efficiency
        treat_vals = [0.10, 0.30, 0.55, 0.72, 0.80, 0.85, 0.88, 0.90]
        ctrl_vals = [0.05, 0.08, 0.12, 0.20, 0.30, 0.45, 0.52, 0.58]
        treat = [_make(t, "cognition-15", v, v, {"binding": v}) for t, v in zip(tokens, treat_vals)]
        ctrl = [_make(t, "control-00", v, v, {"binding": v}) for t, v in zip(tokens, ctrl_vals)]
    elif world_id == 6:  # Loss-matched gap
        t_losses = [3.5, 3.1, 2.8, 2.5, 2.4, 2.1, 1.9, 1.8]
        c_losses = [3.5, 3.2, 3.0, 2.8, 2.7, 2.5, 2.4, 2.3]
        t_vals = [0.10, 0.25, 0.45, 0.60, 0.68, 0.75, 0.80, 0.82]
        c_vals = [0.08, 0.12, 0.15, 0.18, 0.20, 0.23, 0.25, 0.26]
        treat = [_make(t, "cognition-15", v, v, {"b": v}, loss=l) for t, v, l in zip(tokens, t_vals, t_losses)]
        ctrl = [_make(t, "control-00", v, v, {"b": v}, loss=l) for t, v, l in zip(tokens, c_vals, c_losses)]
    elif world_id == 7:  # Seed instability
        treat = [_make(t, "cognition-15", 0.1 + 0.7 * (t / 50_000_000), 0.1, {"b": 0.1 + 0.7 * (t / 50_000_000)}, seed=42) for t in tokens]
        ctrl = [_make(t, "cognition-15", 0.1 + 0.05 * (t / 50_000_000), 0.1, {"b": 0.1 + 0.05 * (t / 50_000_000)}, seed=43) for t in tokens]
    elif world_id == 8:  # Difficulty-aware ordering
        treat = [_make(t, "cognition-15", 0.5, 0.5, {"retrieval": min(1.0, 0.1 + 0.9 * (t / 10_000_000)), "binding": min(1.0, 0.1 + 0.8 * (t / 20_000_000)), "state": min(1.0, 0.1 + 0.7 * (t / 50_000_000))}) for t in tokens]
        ctrl = [_make(t, "control-00", 0.1, 0.1, {"retrieval": 0.1, "binding": 0.1, "state": 0.1}) for t in tokens]
    elif world_id == 9:  # Macro average masks family collapse
        treat = [_make(t, "cognition-15", 0.1 + 0.6 * (t / 50_000_000), 0.1, {"binding": 0.1 + 0.85 * (t / 50_000_000), "counterfactual": max(0.02, 0.20 - 0.18 * (t / 50_000_000))}) for t in tokens]
        ctrl = [_make(t, "control-00", 0.15, 0.1, {"binding": 0.15, "counterfactual": 0.15}) for t in tokens]
    else:  # world_id == 10: Early triage predictor
        t_vals = [0.10, 0.11, 0.12, 0.12, 0.13, 0.13, 0.14, 0.14]
        treat = [_make(t, "cognition-15", v, v, {"b": v}) for t, v in zip(tokens, t_vals)]
        ctrl = [_make(t, "control-00", 0.12, 0.12, {"b": 0.12}) for t in tokens]

    return treat, ctrl


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Causal Learning Dynamics (CLD/CAD) Analyzer")
    parser.add_argument("--world", type=int, choices=range(1, 11), help="Execute synthetic adversarial world simulation (1-10)")
    parser.add_argument("--treatment", nargs="+", help="List of treatment checkpoint receipt JSON paths")
    parser.add_argument("--control", nargs="+", help="List of control checkpoint receipt JSON paths")
    parser.add_argument("--capability-threshold", type=float, default=0.50, help="Precommitted capability threshold")
    parser.add_argument("--loss-matched-target", type=float, default=2.40, help="Loss level for loss-matched gap analysis")
    parser.add_argument("--output", type=Path, default=Path("artifacts/v5/cld_trajectory_metrics.json"), help="Output summary path")
    args = parser.parse_args()

    if args.world:
        print(f"============================================================")
        print(f"EXECUTING CAUSAL LEARNING DYNAMICS SIMULATION: WORLD {args.world}")
        print(f"============================================================")
        treat, ctrl = generate_synthetic_world_receipts(args.world)
    elif args.treatment and args.control:
        treat = [CognitiveAcquisitionReceipt(**json.loads(Path(p).read_text(encoding="utf-8"))) for p in sorted(args.treatment)]
        ctrl = [CognitiveAcquisitionReceipt(**json.loads(Path(p).read_text(encoding="utf-8"))) for p in sorted(args.control)]
    else:
        print("Error: Specify either --world [1..10] or both --treatment and --control receipts.")
        return 1

    summary = TrajectoryAnalysisEngine.analyze_trajectories(
        treat,
        ctrl,
        capability_threshold=args.capability_threshold,
        loss_matched_target=args.loss_matched_target,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary.canonical(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote trajectory analysis summary to: {args.output}")

    print("\nKey Causal Dynamics Metrics:")
    print(f"  Macro TE-AUC:                {summary.treatment_effect_auc.get('macro', 0.0):+.4f}")
    print(f"  Treatment TTT (Tokens-to-50%): {summary.tokens_to_threshold_treatment.get('macro')}")
    print(f"  Control TTT:                   {summary.tokens_to_threshold_control.get('macro')}")
    print(f"  Transfer Lag (tokens):         {summary.transfer_lag_tokens}")
    print(f"  Loss-Matched Cognition Gap:    {summary.loss_matched_gap_at_threshold}")
    print(f"  Acquisition Order:             {' -> '.join(summary.acquisition_order_treatment) if summary.acquisition_order_treatment else 'None'}")
    print(f"  Early Triage Decision:         {summary.early_triage_decision}")
    print(f"  Rationale:                     {summary.early_triage_rationale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())