"""Epistemic scientific interpreter separating confirmed causes from suspicions and open questions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from anra_v5.elendril.codec import digest

from .contracts import HypothesisSpec


@dataclass(frozen=True, slots=True)
class ScientificClaim:
    claim_id: str
    category: str  # confirmed | suspected | falsified | unresolved
    statement: str
    evidence_basis: dict[str, Any]
    confidence_rule: str


@dataclass(frozen=True, slots=True)
class RecommendedExperiment:
    rank: int
    action_id: str
    title: str
    rationale: str
    cost_tier: str
    expected_info_gain_per_gpu_hour: float
    discriminates_between: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScientificInterpretationReport:
    schema: str
    experiment_id: str
    status: str
    instrument_passed: bool
    confirmed_causes: tuple[ScientificClaim, ...] = ()
    suspected_mechanisms: tuple[ScientificClaim, ...] = ()
    falsified_hypotheses: tuple[ScientificClaim, ...] = ()
    unresolved_questions: tuple[ScientificClaim, ...] = ()
    recommended_next_experiments: tuple[RecommendedExperiment, ...] = ()
    instrument_defects: tuple[dict[str, Any], ...] = ()
    digest_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ci_excludes_zero(ci: Sequence[float] | None) -> bool:
    if not ci or len(ci) != 2:
        return False
    lo, hi = float(ci[0]), float(ci[1])
    return (lo > 0.0 and hi > 0.0) or (lo < 0.0 and hi < 0.0)


def synthesize_scientific_interpretation(
    *,
    experiment_id: str,
    instrument_validation: Mapping[str, Any],
    process_funnel: Mapping[str, Any] | None = None,
    oracle_intervention: Mapping[str, Any] | None = None,
    subgoal_decomposition: Mapping[str, Any] | None = None,
    invariance_counterfactuals: Mapping[str, Any] | None = None,
    linear_probe: Mapping[str, Any] | None = None,
    surgical_ablation: Mapping[str, Any] | None = None,
    compute_depth_scaling: Mapping[str, Any] | None = None,
    learning_trajectory: Mapping[str, Any] | None = None,
    internal_telemetry: Mapping[str, Any] | None = None,
    hypotheses: Sequence[HypothesisSpec] = (),
    observed_metrics_for_falsifiers: Mapping[str, float] | None = None,
) -> ScientificInterpretationReport:
    """Synthesize evidence strictly under the epistemic firewall."""

    inst_ok = bool(instrument_validation.get("passed", False))
    defects = tuple(instrument_validation.get("defects", ()))

    if not inst_ok:
        rec = RecommendedExperiment(
            rank=1,
            action_id="REPAIR_INSTRUMENT_DEFECT",
            title="Repair and re-verify failing instrument checks before interpreting model metrics",
            rationale=(
                f"Instrument validation flagged {len(defects)} defect(s) in "
                f"{instrument_validation.get('kernel_id', 'instrument')}; observed scores cannot be "
                "attributed to the model until experimental parity and tool-consumption invariants hold."
            ),
            cost_tier="CPU_SECONDS",
            expected_info_gain_per_gpu_hour=100.0,
            discriminates_between=("instrument_bug", "model_failure"),
        )
        unresolved = (
            ScientificClaim(
                claim_id="U-INSTRUMENT-BLOCK",
                category="unresolved",
                statement=(
                    "Model capability cannot be interpreted because instrument validation failed."
                ),
                evidence_basis={"defects_count": len(defects), "status": instrument_validation.get("status")},
                confidence_rule="epistemic_firewall_instrument_halt",
            ),
        )
        payload = {
            "schema": "anra-planthir-interpretation-v1",
            "experiment_id": experiment_id,
            "status": "INVALIDATED_BY_INSTRUMENT_DEFECT",
            "instrument_passed": False,
            "confirmed_causes": (),
            "suspected_mechanisms": (),
            "falsified_hypotheses": (),
            "unresolved_questions": unresolved,
            "recommended_next_experiments": (rec,),
            "instrument_defects": defects,
            "metadata": {"instrument_status": instrument_validation.get("status")},
        }
        return ScientificInterpretationReport(**payload, digest_sha256=digest([asdict(u) for u in unresolved]))

    confirmed: list[ScientificClaim] = []
    suspected: list[ScientificClaim] = []
    falsified: list[ScientificClaim] = []
    unresolved: list[ScientificClaim] = []
    recs: list[RecommendedExperiment] = []

    # 1. Process funnel decoupling (e.g., Stage 1 information seeking vs Stage 4 end-to-end)
    if process_funnel:
        funnel = process_funnel.get("funnel", {})
        s1 = funnel.get("stage1_info_seeking")
        s2 = funnel.get("stage2_tool_discipline")
        s3 = funnel.get("stage3_single_hop_grounding")
        s4 = funnel.get("stage4_multi_hop_composition")
        if s1 is not None and s4 is not None and (s1 - s4) >= 0.25:
            confirmed.append(
                ScientificClaim(
                    claim_id="C-FUNNEL-DROP",
                    category="confirmed",
                    statement=(
                        f"Observable decoupling between initial tool selection (Stage 1={s1:.3f}) "
                        f"and end-to-end multi-hop execution (Stage 4={s4:.3f})."
                    ),
                    evidence_basis={"stage1": s1, "stage2": s2, "stage3": s3, "stage4": s4},
                    confidence_rule="process_funnel_stage_separation>=0.25",
                )
            )
        if s2 is not None and s2 < 0.80:
            confirmed.append(
                ScientificClaim(
                    claim_id="C-POST-TOOL-DISCIPLINE",
                    category="confirmed",
                    statement=(
                        f"Post-tool state transition breakdown: Stage 2 discipline is {s2:.3f} "
                        "(repeated consumed tool or invalid post-tool step)."
                    ),
                    evidence_basis={"stage2_tool_discipline": s2},
                    confidence_rule="direct_rollout_action_count",
                )
            )

    # 2. Oracle information injection intervention
    if oracle_intervention:
        delta = oracle_intervention.get("tool_delta_oracle")
        ci = oracle_intervention.get("tool_delta_oracle_95ci")
        if delta is None:
            delta = oracle_intervention.get("delta_oracle")
            ci = oracle_intervention.get("delta_oracle_95ci")
        orac_acc = oracle_intervention.get("tool_oracle_accuracy")
        if orac_acc is None:
            orac_acc = oracle_intervention.get("oracle_injected_accuracy")

        if delta is not None and abs(float(delta)) >= 0.15 and _ci_excludes_zero(ci):
            confirmed.append(
                ScientificClaim(
                    claim_id="C-TOOL-ACQUISITION-BOTTLENECK",
                    category="confirmed",
                    statement=(
                        f"Injecting hidden tool facts directly into initial context changes accuracy by "
                        f"{float(delta):+.3f} (95% CI {ci}), confirming a multi-step tool-use/transition bottleneck."
                    ),
                    evidence_basis=dict(oracle_intervention),
                    confidence_rule="paired_intervention_ci_excludes_zero",
                )
            )
        elif orac_acc is not None and float(orac_acc) < 0.50:
            confirmed.append(
                ScientificClaim(
                    claim_id="C-INCONTEXT-BOTTLENECK",
                    category="confirmed",
                    statement=(
                        f"Even with oracle tool information pre-merged in context (accuracy={float(orac_acc):.3f}), "
                        "the model fails to resolve answers; failure is not solely due to tool acquisition."
                    ),
                    evidence_basis=dict(oracle_intervention),
                    confidence_rule="oracle_ceiling_failure<0.50",
                )
            )
        elif delta is not None and abs(float(delta)) >= 0.05:
            suspected.append(
                ScientificClaim(
                    claim_id="S-TOOL-TRANSITION-GAP",
                    category="suspected",
                    statement=f"Directional oracle-information gain of {float(delta):+.3f}, but CI {ci} does not establish certainty.",
                    evidence_basis=dict(oracle_intervention),
                    confidence_rule="directional_effect_ci_includes_zero",
                )
            )

    # 3. Subgoal hop decomposition
    if subgoal_decomposition and subgoal_decomposition.get("worlds_decomposed", 0) > 0:
        comp_gap = subgoal_decomposition.get("composition_gap")
        ci = subgoal_decomposition.get("composition_gap_95ci")
        joint = subgoal_decomposition.get("joint_individual_hops_solvable")
        full = subgoal_decomposition.get("full_chain_accuracy")
        if comp_gap is not None and float(comp_gap) >= 0.15 and _ci_excludes_zero(ci):
            confirmed.append(
                ScientificClaim(
                    claim_id="C-COMPOSITION-GAP",
                    category="confirmed",
                    statement=(
                        f"Genuine multi-hop composition bottleneck: joint individual subgoals are solvable at "
                        f"{float(joint):.3f} vs full chain {float(full):.3f} (gap {float(comp_gap):+.3f}, CI {ci})."
                    ),
                    evidence_basis=dict(subgoal_decomposition),
                    confidence_rule="subgoal_paired_gap_ci_excludes_zero",
                )
            )
        elif comp_gap is not None and float(comp_gap) > 0.0:
            suspected.append(
                ScientificClaim(
                    claim_id="S-COMPOSITION-GAP",
                    category="suspected",
                    statement=(
                        f"Observed directional composition gap of {float(comp_gap):+.3f} "
                        f"(joint={float(joint):.3f}, full={float(full):.3f}), insufficient statistical separation."
                    ),
                    evidence_basis=dict(subgoal_decomposition),
                    confidence_rule="directional_subgoal_gap",
                )
            )
        elif joint is not None and float(joint) < 0.40:
            suspected.append(
                ScientificClaim(
                    claim_id="S-SINGLE-HOP-GROUNDING-DEFICIT",
                    category="suspected",
                    statement=(
                        f"Individual single-hop subgoals themselves succeed at only {float(joint):.3f}; "
                        "grounding failure precedes multi-hop chaining."
                    ),
                    evidence_basis=dict(subgoal_decomposition),
                    confidence_rule="subgoal_grounding_floor<0.40",
                )
            )

    # 4. Invariance counterfactuals
    if invariance_counterfactuals:
        sym = invariance_counterfactuals.get("symbol_permutation", {})
        eq_rate = sym.get("equivariance_rate")
        if eq_rate is not None and float(eq_rate) < 0.70:
            suspected.append(
                ScientificClaim(
                    claim_id="S-SYMBOL-EQUIVARIANCE-DEFICIT",
                    category="suspected",
                    statement=f"Low symbol-permutation equivariance ({float(eq_rate):.3f} < 0.70) suggests token-identity bias rather than structural binding.",
                    evidence_basis=dict(sym),
                    confidence_rule="counterfactual_equivariance_threshold",
                )
            )
        rev = invariance_counterfactuals.get("row_order_reversal", {})
        rev_inv = rev.get("prediction_invariance_rate")
        if rev_inv is not None and float(rev_inv) < 0.75:
            suspected.append(
                ScientificClaim(
                    claim_id="S-POSITIONAL-ORDER-SENSITIVITY",
                    category="suspected",
                    statement=f"Row-order reversal stability is {float(rev_inv):.3f}, indicating positional sensitivity.",
                    evidence_basis=dict(rev),
                    confidence_rule="counterfactual_row_reversal_threshold",
                )
            )

    # 5. Frozen-backbone linear probe vs head readout
    if linear_probe:
        probe_final = linear_probe.get("probe_final_symbol_accuracy")
        head_final = linear_probe.get("model_head_final_accuracy_on_visible")
        gap = linear_probe.get("head_minus_probe_gap")
        if gap is not None and float(gap) >= 0.15:
            suspected.append(
                ScientificClaim(
                    claim_id="S-READOUT-HEAD-BOTTLENECK",
                    category="suspected",
                    statement=(
                        f"Frozen-backbone linear probe ({float(probe_final):.3f}) outperforms model head "
                        f"({float(head_final):.3f}) by {float(gap):+.3f}, pointing toward readout/gate under-extraction."
                    ),
                    evidence_basis=dict(linear_probe),
                    confidence_rule="probe_vs_head_gap>=0.15",
                )
            )
        mh_h1 = linear_probe.get("multihop_probe_hop1_accuracy")
        mh_fin = linear_probe.get("multihop_probe_final_accuracy")
        if mh_h1 is not None and mh_fin is not None and (float(mh_h1) - float(mh_fin)) >= 0.20:
            suspected.append(
                ScientificClaim(
                    claim_id="S-BACKBONE-HOP2-REPRESENTATION-DEFICIT",
                    category="suspected",
                    statement=(
                        f"Backbone linearly encodes Hop-1 ({float(mh_h1):.3f}) substantially better than "
                        f"multi-hop target ({float(mh_fin):.3f})."
                    ),
                    evidence_basis={"mh_probe_hop1": mh_h1, "mh_probe_final": mh_fin},
                    confidence_rule="probe_hop1_minus_final>=0.20",
                )
            )

    # 6. Internal telemetry (gate openness, representation rank, gradient health)
    if internal_telemetry:
        rep = internal_telemetry.get("representation", {})
        mech = internal_telemetry.get("mechanism_telemetry", {})
        eff_rank = rep.get("effective_rank")
        if eff_rank is not None and float(eff_rank) < 3.0:
            suspected.append(
                ScientificClaim(
                    claim_id="S-REPRESENTATION-COLLAPSE-RISK",
                    category="suspected",
                    statement=f"Low representation effective rank ({float(eff_rank):.2f}) suggests narrow feature collapse.",
                    evidence_basis=dict(rep),
                    confidence_rule="effective_rank<3.0",
                )
            )
        gate_mean = mech.get("pointer_gate_mean")
        if gate_mean is not None and float(gate_mean) < 0.15:
            suspected.append(
                ScientificClaim(
                    claim_id="S-POINTER-GATE-CLOSED",
                    category="suspected",
                    statement=f"Pointer copy gate remains largely closed (mean={float(gate_mean):.3f} < 0.15), suppressing copy-path contribution.",
                    evidence_basis=dict(mech),
                    confidence_rule="pointer_gate_mean<0.15",
                )
            )

    # 6b. Surgical mechanism ablation & compute-depth scaling counterfactuals
    if surgical_ablation and surgical_ablation.get("ablation_supported"):
        delta = float(surgical_ablation.get("accuracy_delta", 0.0))
        mtype = str(surgical_ablation.get("mechanism_type", "mechanism"))
        if surgical_ablation.get("ornamental_mechanism_bypass"):
            suspected.append(
                ScientificClaim(
                    claim_id="S-ORNAMENTAL-MECHANISM-BYPASS",
                    category="suspected",
                    statement=(
                        f"Surgical clamping of {mtype} produced zero accuracy delta and "
                        f"{float(surgical_ablation.get('action_agreement_rate', 1.0)):.3f} action agreement; "
                        "the specialized mechanism is currently bypassed by optimization."
                    ),
                    evidence_basis=dict(surgical_ablation),
                    confidence_rule="surgical_ablation_agreement>=0.98_and_delta==0",
                )
            )
        elif delta >= 0.05:
            confirmed.append(
                ScientificClaim(
                    claim_id="C-ACTIVE-MECHANISM-CONTRIBUTION",
                    category="confirmed",
                    statement=(
                        f"Surgical ablation of {mtype} reduced episode accuracy by {delta:+.3f}, "
                        "confirming active causal contribution without retraining."
                    ),
                    evidence_basis=dict(surgical_ablation),
                    confidence_rule="surgical_ablation_accuracy_delta>=0.05",
                )
            )

    if compute_depth_scaling and compute_depth_scaling.get("supports_variable_depth"):
        rgain = float(compute_depth_scaling.get("recurrence_gain_over_zero", 0.0))
        if rgain >= 0.05:
            confirmed.append(
                ScientificClaim(
                    claim_id="C-RECURRENT-COMPUTE-GAIN",
                    category="confirmed",
                    statement=(
                        f"Test-time recurrent compute increases accuracy by {rgain:+.3f} over 0-step bypass "
                        f"(trained steps={compute_depth_scaling.get('trained_steps')})."
                    ),
                    evidence_basis=dict(compute_depth_scaling),
                    confidence_rule="compute_depth_gain>=0.05",
                )
            )

    # 6c. Multi-checkpoint trajectory analysis
    if learning_trajectory and learning_trajectory.get("representation_leads_readout_detected"):
        max_lead = float(learning_trajectory.get("max_probe_lead_over_head", 0.0))
        suspected.append(
            ScientificClaim(
                claim_id="S-TRAJECTORY-PROBE-LEADS-HEAD",
                category="suspected",
                statement=(
                    f"Across training milestones, linear probe accuracy leads output head accuracy by "
                    f"{max_lead:+.3f}, indicating internal representation precedes readout alignment."
                ),
                evidence_basis={
                    "checkpoints_evaluated": learning_trajectory.get("checkpoints_evaluated"),
                    "max_probe_lead_over_head": max_lead,
                },
                confidence_rule="trajectory_max_probe_lead>=0.10",
            )
        )

    # 7. Evaluate preregistered hypothesis falsifiers
    obs_map = observed_metrics_for_falsifiers or {}
    for hyp in hypotheses:
        evals = [f.evaluate(obs_map.get(f.metric_name)) for f in hyp.falsifiers]
        if any(e["status"] == "FALSIFIED" for e in evals):
            falsified.append(
                ScientificClaim(
                    claim_id=f"F-{hyp.hypothesis_id}",
                    category="falsified",
                    statement=f"Hypothesis {hyp.hypothesis_id!r} falsified: {hyp.statement}",
                    evidence_basis={"falsifier_results": evals},
                    confidence_rule="preregistered_falsifier_failed",
                )
            )
        elif evals and all(e["status"] == "NOT_FALSIFIED" for e in evals):
            confirmed.append(
                ScientificClaim(
                    claim_id=f"C-{hyp.hypothesis_id}",
                    category="confirmed",
                    statement=f"Hypothesis {hyp.hypothesis_id!r} survived preregistered falsifiers: {hyp.statement}",
                    evidence_basis={"falsifier_results": evals},
                    confidence_rule="all_preregistered_falsifiers_passed",
                )
            )
        else:
            unresolved.append(
                ScientificClaim(
                    claim_id=f"U-{hyp.hypothesis_id}",
                    category="unresolved",
                    statement=f"Hypothesis {hyp.hypothesis_id!r} remains indeterminate under current metrics: {hyp.statement}",
                    evidence_basis={"falsifier_results": evals},
                    confidence_rule="insufficient_falsifier_metrics",
                )
            )

    # Always preserve explicit unresolved competing hypotheses so certainty is never manufactured
    unresolved.append(
        ScientificClaim(
            claim_id="U-SCALE-AND-HORIZON",
            category="unresolved",
            statement=(
                "Diagnostic probes on a single checkpoint or tiny scale do not resolve whether "
                "larger parameter scale, longer horizon, or alternative optimization dynamics overcome the bottleneck."
            ),
            evidence_basis={"note": "single_regime_observability_bound"},
            confidence_rule="epistemic_scope_guard",
        )
    )

    # Ranked next experiments by expected info gain / GPU hour
    if any(c.claim_id == "S-POINTER-GATE-CLOSED" for c in suspected):
        recs.append(
            RecommendedExperiment(
                rank=1,
                action_id="INTERVENE_POINTER_GATE_AND_COPY",
                title="Run frozen pointer no-copy vs cyclic-permutation vs forced-gate sweep",
                rationale="Gate openness is low; testing frozen interventions immediately distinguishes closed-gate optimization inertia from attention misalignment.",
                cost_tier="CPU_SECONDS",
                expected_info_gain_per_gpu_hour=90.0,
                discriminates_between=("gate_bias_inertia", "attention_key_failure"),
            )
        )
    if any(c.claim_id in ("C-COMPOSITION-GAP", "S-COMPOSITION-GAP") for c in confirmed + suspected):
        recs.append(
            RecommendedExperiment(
                rank=len(recs) + 1,
                action_id="TEST_RECURRENT_REFINEMENT_T",
                title="Evaluate recurrent refinement T=1..3 vs non-recurrent F-cap on decomposed multi-hop worlds",
                rationale="Single-hop lookup succeeds better than multi-hop chaining; iterative state refinement directly targets intermediate variable passing.",
                cost_tier="SINGLE_T4_MINUTES",
                expected_info_gain_per_gpu_hour=35.0,
                discriminates_between=("depth_limitation", "training_objective_supervision"),
            )
        )
    if not recs:
        recs.append(
            RecommendedExperiment(
                rank=1,
                action_id="EXPAND_SEED_AND_SCALE_CONTRAST",
                title="Run matched-seed causal contrast across scales (S3 vs S20) using registered Protocol",
                rationale="Current diagnostics found no single defect or dominant bottleneck; multi-seed scaling resolves capacity vs optimization hypotheses.",
                cost_tier="DUAL_T4_HOURS",
                expected_info_gain_per_gpu_hour=15.0,
                discriminates_between=("capacity_limit", "optimization_schedule"),
            )
        )

    # Re-index ranks cleanly
    ranked_recs = tuple(
        RecommendedExperiment(
            rank=idx + 1,
            action_id=r.action_id,
            title=r.title,
            rationale=r.rationale,
            cost_tier=r.cost_tier,
            expected_info_gain_per_gpu_hour=r.expected_info_gain_per_gpu_hour,
            discriminates_between=r.discriminates_between,
        )
        for idx, r in enumerate(sorted(recs, key=lambda x: x.expected_info_gain_per_gpu_hour, reverse=True))
    )

    body = {
        "schema": "anra-planthir-interpretation-v1",
        "experiment_id": experiment_id,
        "status": "INTERPRETED_WITH_INSTRUMENT_VERIFIED",
        "instrument_passed": True,
        "confirmed_causes": tuple(confirmed),
        "suspected_mechanisms": tuple(suspected),
        "falsified_hypotheses": tuple(falsified),
        "unresolved_questions": tuple(unresolved),
        "recommended_next_experiments": ranked_recs,
        "instrument_defects": (),
        "metadata": {
            "counts": {
                "confirmed": len(confirmed),
                "suspected": len(suspected),
                "falsified": len(falsified),
                "unresolved": len(unresolved),
            }
        },
    }
    digest_val = digest([asdict(c) for c in confirmed + suspected + falsified + unresolved])
    return ScientificInterpretationReport(**body, digest_sha256=digest_val)


__all__ = [
    "RecommendedExperiment",
    "ScientificClaim",
    "ScientificInterpretationReport",
    "synthesize_scientific_interpretation",
]
