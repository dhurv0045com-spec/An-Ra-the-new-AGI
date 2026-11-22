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
    slice_comparisons: Mapping[str, Any] | None = None,
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
            suspected.append(
                ScientificClaim(
                    claim_id="S-FUNNEL-DROP",
                    category="suspected",
                    statement=(
                        f"Observable decoupling between initial tool selection (Stage 1={s1:.3f}) "
                        f"and end-to-end multi-hop execution (Stage 4={s4:.3f})."
                    ),
                    evidence_basis={"stage1": s1, "stage2": s2, "stage3": s3, "stage4": s4},
                    confidence_rule="process_funnel_stage_separation>=0.25",
                )
            )
        if s2 is not None and s2 < 0.80:
            suspected.append(
                ScientificClaim(
                    claim_id="S-POST-TOOL-DISCIPLINE",
                    category="suspected",
                    statement=(
                        f"Post-tool discipline on the sampled valid callers is {s2:.3f} "
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
            suspected.append(
                ScientificClaim(
                    claim_id="S-ORACLE-INFORMATION-EFFECT",
                    category="suspected",
                    statement=(
                        f"Injecting hidden tool facts directly into initial context changes accuracy by "
                        f"{float(delta):+.3f} (conditional cluster-bootstrap 95% CI {ci}); this intervention also changes context and does not isolate a tool-use mechanism."
                    ),
                    evidence_basis=dict(oracle_intervention),
                    confidence_rule="paired_intervention_ci_excludes_zero",
                )
            )
        elif orac_acc is not None and float(orac_acc) < 0.50:
            suspected.append(
                ScientificClaim(
                    claim_id="S-LOW-ORACLE-ACCURACY",
                    category="suspected",
                    statement=(
                        f"Even with oracle tool information pre-merged in context (accuracy={float(orac_acc):.3f}), "
                        "accuracy is low on this suite; training progress and altered context remain competing explanations."
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
            suspected.append(
                ScientificClaim(
                    claim_id="S-COMPOSITION-GAP",
                    category="suspected",
                    statement=(
                        f"Joint individual subgoals succeed at "
                        f"{float(joint):.3f} vs full chain {float(full):.3f} (gap {float(comp_gap):+.3f}, CI {ci}); the altered tasks do not isolate composition as the cause."
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
    if surgical_ablation and (surgical_ablation.get("ablation_supported") or (surgical_ablation.get("coverage") or {}).get("target_families")):
        delta = float(surgical_ablation.get("accuracy_delta", 0.0))
        mtype = str(surgical_ablation.get("mechanism_type", "mechanism"))
        cov = surgical_ablation.get("coverage") or {}
        if cov.get("target_families") and cov.get("coverage_complete") is False:
            unresolved.append(
                ScientificClaim(
                    claim_id="U-INCOMPLETE-DIAGNOSTIC-COVERAGE",
                    category="unresolved",
                    statement=(
                        f"Intervention {mtype!r} recorded {delta:+.3f} net accuracy delta on "
                        f"{cov.get('distinct_base_world_count', cov.get('sample_episodes'))} base worlds "
                        f"({cov.get('distinct_families')}), with target-family coverage "
                        f"{cov.get('target_family_coverage_ratio')} (missing {cov.get('missing_target_families')}). "
                        "Mechanism unresolved."
                    ),
                    evidence_basis={"coverage": dict(cov), "accuracy_delta": delta},
                    confidence_rule="incomplete_target_family_coverage",
                )
            )
            recs.append(
                RecommendedExperiment(
                    rank=len(recs) + 1,
                    action_id="EVALUATE_FROZEN_LESIONS_ON_TARGET_FAMILIES",
                    title="Evaluate surgical lesions on frozen checkpoints across all missing target families",
                    rationale=(
                        f"Target-family coverage was {cov.get('target_family_coverage_ratio')}; "
                        "re-run frozen lesions stratified across all mechanistic families before claiming bypass."
                    ),
                    cost_tier="CPU_OR_SINGLE_GPU_MINUTES",
                    expected_info_gain_per_gpu_hour=95.0,
                    discriminates_between=("ornamental_mechanism", "untested_family_engagement"),
                )
            )
        elif surgical_ablation.get("ablation_supported") and surgical_ablation.get("ornamental_mechanism_bypass"):
            suspected.append(
                ScientificClaim(
                    claim_id="S-ORNAMENTAL-MECHANISM-BYPASS",
                    category="suspected",
                    statement=(
                        f"Surgical clamping of {mtype} produced zero accuracy delta and "
                        f"{float(surgical_ablation.get('action_agreement_rate', 1.0)):.3f} action agreement; "
                        "this finite suite does not establish that the mechanism is bypassed in general."
                    ),
                    evidence_basis=dict(surgical_ablation),
                    confidence_rule="surgical_ablation_agreement>=0.98_and_delta==0",
                )
            )
        elif surgical_ablation.get("ablation_supported") and delta >= 0.05:
            suspected.append(
                ScientificClaim(
                    claim_id="S-ABLATION-ACCURACY-DROP",
                    category="suspected",
                    statement=(
                        f"Surgical ablation of {mtype} reduced episode accuracy by {delta:+.3f}, "
                        "on these worlds; replication and uncertainty estimates are needed for a general claim."
                    ),
                    evidence_basis=dict(surgical_ablation),
                    confidence_rule="surgical_ablation_accuracy_delta>=0.05",
                )
            )

    # 6d. Predeclared slice / renderer sensitivity
    if slice_comparisons:
        largest_drop = 0.0
        largest_info: dict[str, Any] = {}
        slices_iter: Sequence[Mapping[str, Any]]
        if isinstance(slice_comparisons.get("slices"), Sequence):
            slices_iter = slice_comparisons["slices"]
        else:
            slices_iter = [
                {"slice_name": str(k), **(dict(v) if isinstance(v, Mapping) else {"value": v})}
                for k, v in slice_comparisons.items()
            ]
        for srow in slices_iter:
            drop_val: float | None = None
            if srow.get("drop") is not None:
                drop_val = float(srow["drop"])
            elif srow.get("baseline_accuracy") is not None and srow.get("slice_accuracy") is not None:
                drop_val = float(srow["baseline_accuracy"]) - float(srow["slice_accuracy"])
            elif srow.get("canonical_accuracy") is not None and srow.get("alt_accuracy") is not None:
                drop_val = float(srow["canonical_accuracy"]) - float(srow["alt_accuracy"])
            if drop_val is not None and drop_val > largest_drop:
                largest_drop = drop_val
                largest_info = dict(srow)
        if largest_drop >= 0.20:
            suspected.append(
                ScientificClaim(
                    claim_id="S-RENDERER-OR-SLICE-TRANSFER-GAP",
                    category="suspected",
                    statement=(
                        f"Largest predeclared slice drop was {largest_drop:.3f} on "
                        f"{largest_info.get('slice_name', 'alt_slice')}; wording, token layout, and "
                        "tool-selection effects remain competing explanations."
                    ),
                    evidence_basis=largest_info,
                    confidence_rule="predeclared_slice_drop>=0.20",
                )
            )
            recs.append(
                RecommendedExperiment(
                    rank=len(recs) + 1,
                    action_id="CROSS_FROZEN_LESIONS_WITH_RENDERER_VIEWS",
                    title="Cross frozen surgical lesions with canonical vs narrative/held-out renderer views of identical base worlds",
                    rationale="Discriminates representation/copy/relational reliance from surface presentation fragility using existing checkpoints.",
                    cost_tier="SINGLE_T4_MINUTES",
                    expected_info_gain_per_gpu_hour=92.0,
                    discriminates_between=("presentation_sensitivity", "core_mechanism_failure"),
                )
            )

    if compute_depth_scaling and compute_depth_scaling.get("supports_variable_depth"):
        rgain = float(compute_depth_scaling.get("recurrence_gain_over_zero", 0.0))
        if rgain >= 0.05:
            suspected.append(
                ScientificClaim(
                    claim_id="S-RECURRENT-COMPUTE-GAIN",
                    category="suspected",
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
            unresolved.append(
                ScientificClaim(
                    claim_id=f"U-{hyp.hypothesis_id}",
                    category="unresolved",
                    statement=f"Hypothesis {hyp.hypothesis_id!r} was not falsified by the available checks; it remains unconfirmed: {hyp.statement}",
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

    obs_accs: list[float] = []
    if process_funnel:
        ov = process_funnel.get("overall_accuracy")
        if isinstance(ov, (int, float)):
            obs_accs.append(float(ov))
        funnel = process_funnel.get("funnel")
        if isinstance(funnel, Mapping):
            for k in ("stage3_single_hop_grounding", "stage4_multi_hop_composition"):
                fv = funnel.get(k)
                if isinstance(fv, (int, float)):
                    obs_accs.append(float(fv))
    if oracle_intervention:
        for k in ("baseline_accuracy", "observed_baseline_accuracy"):
            ov = oracle_intervention.get(k)
            if isinstance(ov, (int, float)):
                obs_accs.append(float(ov))
    if linear_probe:
        hv = linear_probe.get("model_head_final_accuracy_on_visible")
        if isinstance(hv, (int, float)):
            obs_accs.append(float(hv))
    max_obs_acc = max(obs_accs) if obs_accs else None
    low_unconverged = bool(max_obs_acc is not None and max_obs_acc < 0.05)

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
            },
            "convergence_guard": {
                "low_accuracy_unconverged": low_unconverged,
                "max_observed_baseline_accuracy": max_obs_acc,
                "note": (
                    "Observed baseline accuracy is near zero (<0.05); capability conclusions may reflect unconverged training."
                    if low_unconverged
                    else "Observed baseline accuracy meets convergence guard floor or was not supplied."
                ),
            },
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
