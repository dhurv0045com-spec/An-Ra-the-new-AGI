"""Prospective ROLE-TRANSFER-001 confirmatory protocol."""

from __future__ import annotations

import hashlib
import json
from typing import Any

CAMPAIGN = "ROLE-TRANSFER-001"
VERSION = "1"
CANONICAL_PROTOCOL_SHA256 = "ac2c330794aab69750ff25866f7e0e2b977d8308adc3a0004b7d41c647cceefe"
UPSTREAM_SCIENCE_COMMIT = "c15ad8beb409537db42d075684ea54847a074ebd"
UPSTREAM_RECOVERY_COMMIT = "8609ba95f4e978cf3cdf8d20bd8a907eea8f6728"
UPSTREAM_EVIDENCE_SHA256 = "3e3ad68cd7f80bd242733b61153d4bb8f3fedbb1e4fba8fbc0f3ebc5d904423f"
UPSTREAM_SEEDS = (73_011, 73_012, 73_013, 73_014)
PRIMARY_ARMS = ("T0_CANONICAL", "T3_NORM")
MECHANISM_ARMS = ("T0_CANONICAL", "T3_RAW", "T3_NORM", "P_NORM")
FACTORIAL_ARMS = ("T0_CANONICAL", "T1_INPUT_X4", "T2_OUTPUT_X025", "T3_RAW")
OPTIONAL_DIAGNOSTIC_ARMS = ("U_UNTIED_OUTPUT",)
CONFIRMATORY_SEEDS = tuple(range(74_001, 74_013))
REPLICATION_A_SEEDS = tuple(range(74_001, 74_007))
REPLICATION_B_SEEDS = tuple(range(74_007, 74_013))
CALIBRATION_SEEDS = tuple(range(74_101, 74_105))
MECHANISM_ENGINEERING_SEEDS = (74_201, 74_202)
OPTIONAL_SCALE_SHIFT_SEEDS = tuple(range(74_301, 74_305))
TASK_DOMAINS = (
    "unseen_composition",
    "state_workflow_reasoning",
    "tool_execution",
    "long_context_binding",
    "missing_information_handling",
    "structured_termination",
)
TRAIN_ROWS_PER_REPLICATION = 120_000
DEVELOPMENT_ROWS_PER_REPLICATION = 6_000
SEALED_TASKS_PER_REPLICATION = 12_000
ROWS_PER_FAMILY = {
    "training": 20_000,
    "development": 1_000,
    "sealed": 2_000,
}
PHYSICAL_VOCABULARY = 24_576
MODEL_SPEC = {
    "layers": 8,
    "width": 256,
    "query_heads": 4,
    "key_value_heads": 2,
    "head_dimension": 64,
    "ffn_width": 1_024,
    "context_length": 1_024,
    "dropout": 0.0,
    "tied_embeddings": True,
}
OPTIONAL_SCALE_SHIFT_SPEC = {
    "layers": 12,
    "width": 384,
    "query_heads": 6,
    "key_value_heads": 3,
    "head_dimension": 64,
    "ffn_width": 1_536,
    "context_length": 1_024,
    "dropout": 0.0,
    "tied_embeddings": True,
}
PROCESSED_TOKEN_BUDGET = 2_000_000
BATCH_ROWS = 32
MAX_UPDATES = 3_000
MAX_ROW_PRESENTATIONS = 96_000
CHECKPOINT_EVERY_TOKENS = 100_000
EXPOSURE_MISMATCH_TOLERANCE = 0.001
GLOBAL_CLIP_NORM = 1.0
NORM_MATCH_RELATIVE_TOLERANCE = 1e-6
MAX_GENERATION_TOKENS = 64
MAX_TOOL_TURNS = 4
PRIMARY_ENDPOINT = "production_bpe_within_ontology_downstream_macro_success"
PRIMARY_CONTRAST = ("T3_NORM", "T0_CANONICAL")
MECHANISM_CONTRAST = ("T3_RAW", "P_NORM")
PRACTICAL_THRESHOLDS = {
    "minimum_macro_success_gain": 0.08,
    "maximum_exact_permutation_p_value": 0.05,
    "confidence_interval_excludes_zero": True,
    "minimum_positive_seed_blocks": 9,
    "minimum_positive_seed_blocks_per_replication": 4,
    "minimum_replication_mean_gain": 0.0,
    "minimum_composition_gain": 0.05,
    "minimum_tool_execution_gain": 0.05,
    "maximum_guard_regression": 0.03,
    "maximum_unauthorized_actions": 0,
    "minimum_role_specificity_gain": 0.05,
    "role_specificity_interval_lower_bound_above_zero": True,
    "maximum_magnitude_placebo_gain": 0.05,
    "magnitude_placebo_interval_upper_bound_below": 0.05,
    "minimum_magnitude_placebo_gain_for_magnitude_only": 0.05,
    "minimum_latent_diagnostic_gain": 0.05,
    "minimum_output_to_input_norm_ratio_reduction": 0.25,
    "minimum_control_success": 0.30,
    "maximum_control_success": 0.70,
    "maximum_official_clip_fraction_per_seed_arm": 0.05,
}
STATISTICAL_ANALYSIS = {
    "sampling_population": "12 frozen matched seed blocks split into two independent replications",
    "inferential_unit": "matched_seed_block",
    "primary_test": "exact_stratified_two_sided_paired_sign_flip_mean_effect",
    "exact_sign_vector_count": 4096,
    "interval": "stratified_paired_percentile_bootstrap_100000_resamples_seed_20260924",
    "sensitivity_model": "mixed_effects_logistic_with_seed_block_random_intercept",
    "secondary_contrasts_use_same_exact_stratified_test_and_interval": True,
    "replication_means_must_be_positive": True,
    "missing_seed_policy": "INCONCLUSIVE_PROTOCOL_no_imputation",
}
MAGNITUDE_MATCHING = {
    "scope": "full_trainable_gradient_l2_norm_before_one_global_clip",
    "frequency": "every_optimizer_update",
    "reference_trajectories": {
        "T0_CANONICAL": "canonical reference trajectory for the T0_CANONICAL/T3_NORM pair",
        "T3_RAW": "raw role-routed reference trajectory for the P_NORM/T3_RAW pair",
    },
    "lockstep_requirement": (
        "all four arms advance in the same seed block and data order; each norm-matched arm "
        "uses the current pre-clip norm from its pair-specific reference trajectory"
    ),
    "global_clip_norm": GLOBAL_CLIP_NORM,
    "relative_norm_match_tolerance": NORM_MATCH_RELATIVE_TOLERANCE,
    "T3_NORM": "scale its raw role-routed gradient to the current T0_CANONICAL trajectory pre-clip norm",
    "P_NORM": "scale its canonical gradient to the current T3_RAW trajectory pre-clip norm",
    "paired_clip_invariant": "T0/T3_NORM and P_NORM/T3_RAW must have identical per-update clip decisions",
    "record": [
        "T0_reference_full_preclip_norm",
        "T3_RAW_reference_full_preclip_norm",
        "matched_full_preclip_norm",
        "canonical_full_postclip_norm",
        "raw_full_postclip_norm",
        "clip_decision",
        "parameter_displacement_l2",
    ],
    "interpretation_limit": "pre-clip norm matching does not guarantee identical AdamW displacement",
}
OUTCOME_FLAGS = (
    "protocol_valid",
    "substrate_valid",
    "clipping_valid",
    "manipulation_valid",
    "primary_gain_valid",
    "primary_p_valid",
    "primary_ci_valid",
    "replication_valid",
    "domain_transfer_valid",
    "guards_valid",
    "role_specificity_valid",
    "placebo_equivalent_valid",
    "placebo_gain_valid",
    "latent_gain_valid",
)
LATENT_DIAGNOSTIC = {
    "endpoint": "latent_id_identity_formation_auc",
    "split": "frozen_development_only_manifest",
    "manifest_receipt": "LATENT_DEVELOPMENT_MANIFEST_V1.json",
    "manifest_sha256_must_freeze_before_stage_1": True,
    "evaluator_sha256_must_freeze_before_stage_1": True,
    "minimum_gain": 0.05,
    "interval_lower_bound_above_zero": True,
    "may_support_primary_claim": False,
}
DECISION_ORDER = (
    "INCONCLUSIVE_PROTOCOL",
    "INCONCLUSIVE_SUBSTRATE",
    "CLIPPING_CONFOUNDED",
    "MECHANISM_NOT_ENGAGED",
    "SUPPORTED_TRANSFER",
    "PRIMARY_NONTRANSFER",
    "PRIMARY_NONREPLICATED",
    "MAGNITUDE_EFFECT",
    "MECHANISM_UNRESOLVED",
    "MAGNITUDE_ONLY",
    "LATENT_ONLY",
    "NULL_OR_REVERSE",
)
DECISION_RULES = {
    "INCONCLUSIVE_PROTOCOL": "protocol_valid is false",
    "INCONCLUSIVE_SUBSTRATE": "protocol valid but substrate_valid is false",
    "CLIPPING_CONFOUNDED": "protocol and substrate valid but clipping_valid is false",
    "MECHANISM_NOT_ENGAGED": "custody substrate and clipping valid but manipulation_valid is false",
    "SUPPORTED_TRANSFER": "primary core replication transfer guards role-specificity and placebo-equivalence flags all true",
    "PRIMARY_NONTRANSFER": "primary core and replication true but domain_transfer_valid or guards_valid false",
    "PRIMARY_NONREPLICATED": "primary core true but replication_valid false",
    "MAGNITUDE_EFFECT": "primary core replication transfer and guards true with placebo_gain_valid true",
    "MECHANISM_UNRESOLVED": "primary core replication transfer and guards true but neither supported nor magnitude criteria pass",
    "MAGNITUDE_ONLY": "primary core false with placebo_gain_valid true",
    "LATENT_ONLY": "primary core false placebo_gain_valid false and latent_gain_valid true",
    "NULL_OR_REVERSE": "all validity gates pass and no positive primary placebo or latent branch applies",
}
FALSIFICATION_GATES = (
    "custody_no_upstream_rows_checkpoints_seeds_or_sealed_commitments",
    "matched_initialization_initial_logits_data_and_exposure",
    "control_only_capability_between_floor_and_ceiling",
    "official_clip_fraction_at_most_five_percent_for_every_seed_arm",
    "paired_clip_decisions_identical_within_norm_matched_pairs",
    "output_to_input_norm_ratio_reduces_at_least_twenty_five_percent_every_seed",
    "magnitude_placebo_interval_upper_bound_below_five_points",
    "production_bpe_composition_and_tool_transfer",
    "query_blind_and_shuffled_label_evaluator_controls",
    "zero_unauthorized_actions_and_no_material_guard_regression",
)
RESULT_LABELS = (
    "SUPPORTED_TRANSFER",
    "MAGNITUDE_EFFECT",
    "MECHANISM_UNRESOLVED",
    "MAGNITUDE_ONLY",
    "PRIMARY_NONTRANSFER",
    "LATENT_ONLY",
    "NULL_OR_REVERSE",
    "INCONCLUSIVE_SUBSTRATE",
    "CLIPPING_CONFOUNDED",
    "MECHANISM_NOT_ENGAGED",
    "PRIMARY_NONREPLICATED",
    "INCONCLUSIVE_PROTOCOL",
)
STAGES = {
    "stage_0": "static_custody_forward_equivalence_norm_matching_and_sealed_firewall",
    "stage_1": "four_control_only_calibration_blocks_at_fixed_2m_tokens",
    "stage_2": "two_block_mechanism_engineering_check_at_fixed_2m_tokens",
    "stage_3": "twelve_block_four_arm_confirmatory_replication_at_fixed_2m_tokens",
    "stage_4": "single_sealed_clean_room_evaluation_after_development_freeze",
    "stage_5_optional": "four_block_valid_12l_384w_model_geometry_shift",
}
STAGE_TRANSITION_RULES = {
    "stage_1_to_2": "fixed control-only substrate and clipping gates pass",
    "stage_2_to_3": "gradient manipulation and norm-matching receipts pass; efficacy outcomes cannot change protocol",
    "stage_3_to_4": "all planned development blocks complete with no missing confirmatory seed",
    "stage_4": "single evaluator invocation after evaluator and development hashes freeze",
}
MANDATORY_PROCESSED_TOKEN_EXPOSURES = {
    "stage_1": 8_000_000,
    "stage_2": 16_000_000,
    "stage_3": 96_000_000,
    "total": 120_000_000,
}
REMOTE_COMPUTE_ESTIMATE = "PENDING_PRE_OUTCOME_T4X2_ENGINEERING_CALIBRATION"
CLAIM_CEILING = (
    "A positive result supports only a controlled within-synthetic-ontology claim that a forward-equivalent, "
    "full-preclip-norm-matched role-reweighted tied-gradient treatment improved independently generated "
    "held-out production-BPE downstream macro-success by at least eight percentage points at the specified "
    "model and exposure budget. It does not establish external benchmark validity, AGI, cognition, "
    "human-level reasoning, real-world tool competence, production architecture superiority, tokenizer "
    "optimality, large-scale authorization, or that the S5 null was wrong."
)


def protocol_payload() -> dict[str, Any]:
    return {
        "campaign": CAMPAIGN,
        "version": VERSION,
        "written_before_any_role_transfer_official_outcomes": True,
        "relationship_to_upstream": {
            "formation_mux_science_commit": UPSTREAM_SCIENCE_COMMIT,
            "formation_mux_recovery_commit": UPSTREAM_RECOVERY_COMMIT,
            "evidence_archive_sha256": UPSTREAM_EVIDENCE_SHA256,
            "upstream_confirmatory_seeds": list(UPSTREAM_SEEDS),
            "changes_upstream_science": False,
            "reuses_upstream_rows": False,
            "reuses_upstream_checkpoints": False,
            "reuses_upstream_seeds": False,
            "reuses_upstream_sealed_commitments": False,
            "upstream_recovery_and_frontier_completion_required_before_execution": True,
        },
        "causal_question": (
            "Does reweighting input-embedding and output-projection gradient roles, while matching pair-specific full "
            "pre-clip trainable-gradient norm and forward function, causally improve independently generated "
            "held-out production-BPE downstream task success within the new synthetic ontology?"
        ),
        "primary_estimand": (
            "Mean paired difference over the 12 frozen seed blocks: E_seed["
            "within-ontology production-BPE downstream macro-success(T3_NORM) - "
            "within-ontology production-BPE downstream macro-success(T0_CANONICAL)]"
        ),
        "scope_limit": "within_synthetic_ontology_not_external_benchmark_validity",
        "arms": {
            "primary": list(PRIMARY_ARMS),
            "mechanism_screen": list(MECHANISM_ARMS),
            "factorial_screen": list(FACTORIAL_ARMS),
            "optional_diagnostics": list(OPTIONAL_DIAGNOSTIC_ARMS),
            "gradient_scales": {
                "T0_CANONICAL": [1.0, 1.0],
                "T1_INPUT_X4": [4.0, 1.0],
                "T2_OUTPUT_X025": [1.0, 0.25],
                "T3_RAW": [4.0, 0.25],
                "T3_NORM": [4.0, 0.25],
                "P_NORM": [1.0, 1.0],
            },
        },
        "magnitude_matching": dict(MAGNITUDE_MATCHING),
        "mechanism_manipulation": {
            "decomposition": (
                "on each T0 reference state and batch, compute input-only and output-only tied-gradient "
                "contributions before canonical or raw role routing"
            ),
            "baseline_ratio": "L2_norm(output_only_tied_gradient) / L2_norm(input_only_tied_gradient)",
            "ratio_epsilon": 1e-12,
            "per_seed_statistic": (
                "median_across_updates of 1 - raw_reference_ratio / canonical_reference_ratio"
            ),
            "minimum_output_to_input_norm_ratio_reduction": 0.25,
            "required_for_every_confirmatory_seed": True,
            "stratified_interval_lower_bound_above_zero": True,
            "descriptive_label": "role_reweighted_not_assumed_balanced",
        },
        "norm_receipt_contract": {
            "schema": "anra.role-transfer-norm-receipt/v1",
            "frequency": "every_optimizer_update",
            "fields": [
                "campaign",
                "seed_block",
                "update_index",
                "processed_tokens",
                "reference_arm",
                "reference_full_preclip_norm",
                "matched_full_preclip_norm",
                "relative_norm_error",
                "clip_decision",
                "parameter_displacement_l2",
                "previous_receipt_sha256",
            ],
            "relative_norm_error_maximum": NORM_MATCH_RELATIVE_TOLERANCE,
            "hash_chain_required": True,
            "checkpoint_receipt_must_cover_every_update": True,
        },
        "contrasts": {
            "primary_efficacy": list(PRIMARY_CONTRAST),
            "role_specificity": list(MECHANISM_CONTRAST),
            "magnitude_placebo": ["P_NORM", "T0_CANONICAL"],
            "latent_diagnostic": ["T3_NORM", "T0_CANONICAL"],
        },
        "seeds": {
            "upstream_excluded": list(UPSTREAM_SEEDS),
            "confirmatory": list(CONFIRMATORY_SEEDS),
            "replication_a": list(REPLICATION_A_SEEDS),
            "replication_b": list(REPLICATION_B_SEEDS),
            "calibration": list(CALIBRATION_SEEDS),
            "mechanism_engineering": list(MECHANISM_ENGINEERING_SEEDS),
            "optional_scale_shift": list(OPTIONAL_SCALE_SHIFT_SEEDS),
        },
        "data": {
            "new_rule_graph_and_entity_namespace": True,
            "train_rows_per_replication": TRAIN_ROWS_PER_REPLICATION,
            "development_rows_per_replication": DEVELOPMENT_ROWS_PER_REPLICATION,
            "sealed_tasks_per_replication": SEALED_TASKS_PER_REPLICATION,
            "rows_per_family": dict(ROWS_PER_FAMILY),
            "held_out_relation_pairs_compositions_entities_and_tool_schemas": True,
            "production_bpe_primary_rendering": True,
            "latent_id_rendering_mechanism_diagnostic_only": True,
            "external_benchmark_claim": False,
        },
        "model": {
            "physical_vocabulary": PHYSICAL_VOCABULARY,
            "primary": dict(MODEL_SPEC),
            "optional_scale_shift": dict(OPTIONAL_SCALE_SHIFT_SPEC),
        },
        "exposure": {
            "processed_nonpadding_token_budget_per_arm": PROCESSED_TOKEN_BUDGET,
            "batch_rows": BATCH_ROWS,
            "maximum_updates": MAX_UPDATES,
            "maximum_row_presentations": MAX_ROW_PRESENTATIONS,
            "checkpoint_every_processed_tokens": CHECKPOINT_EVERY_TOKENS,
            "exposure_mismatch_tolerance": EXPOSURE_MISMATCH_TOLERANCE,
            "early_stopping": False,
            "dose_selection": False,
            "max_generation_tokens": MAX_GENERATION_TOKENS,
            "max_tool_turns": MAX_TOOL_TURNS,
        },
        "evaluator": {
            "domains": list(TASK_DOMAINS),
            "primary_endpoint": PRIMARY_ENDPOINT,
            "candidate_free": True,
            "model_generated_judge_forbidden": True,
            "tool_tasks_scored_by_environment_state": True,
            "guards": [
                "valid_schema",
                "valid_eos",
                "correct_abstention",
                "zero_unauthorized_actions",
                "identity_retention",
            ],
        },
        "thresholds": dict(PRACTICAL_THRESHOLDS),
        "statistical_analysis": dict(STATISTICAL_ANALYSIS),
        "decision_order": list(DECISION_ORDER),
        "decision_rules": dict(DECISION_RULES),
        "outcome_flags": list(OUTCOME_FLAGS),
        "latent_diagnostic": dict(LATENT_DIAGNOSTIC),
        "falsification_gates": list(FALSIFICATION_GATES),
        "result_labels": list(RESULT_LABELS),
        "stages": dict(STAGES),
        "stage_transition_rules": dict(STAGE_TRANSITION_RULES),
        "mandatory_processed_token_exposures": dict(MANDATORY_PROCESSED_TOKEN_EXPOSURES),
        "remote_compute_estimate": REMOTE_COMPUTE_ESTIMATE,
        "execution_authorized": False,
        "local_execution_authorized": False,
        "claim_ceiling": CLAIM_CEILING,
    }


def classify_outcome(flags: dict[str, bool]) -> str:
    if set(flags) != set(OUTCOME_FLAGS):
        raise RuntimeError("outcome flags do not match the frozen decision contract")
    if any(not isinstance(value, bool) for value in flags.values()):
        raise RuntimeError("outcome flags must be booleans")
    if not flags["protocol_valid"]:
        return "INCONCLUSIVE_PROTOCOL"
    if not flags["substrate_valid"]:
        return "INCONCLUSIVE_SUBSTRATE"
    if not flags["clipping_valid"]:
        return "CLIPPING_CONFOUNDED"
    if not flags["manipulation_valid"]:
        return "MECHANISM_NOT_ENGAGED"
    primary_core = (
        flags["primary_gain_valid"]
        and flags["primary_p_valid"]
        and flags["primary_ci_valid"]
    )
    full_primary = (
        primary_core
        and flags["replication_valid"]
        and flags["domain_transfer_valid"]
        and flags["guards_valid"]
    )
    if full_primary and flags["role_specificity_valid"] and flags["placebo_equivalent_valid"]:
        return "SUPPORTED_TRANSFER"
    if primary_core and flags["replication_valid"] and not (
        flags["domain_transfer_valid"] and flags["guards_valid"]
    ):
        return "PRIMARY_NONTRANSFER"
    if primary_core and not flags["replication_valid"]:
        return "PRIMARY_NONREPLICATED"
    if full_primary and flags["placebo_gain_valid"]:
        return "MAGNITUDE_EFFECT"
    if full_primary:
        return "MECHANISM_UNRESOLVED"
    if not primary_core and flags["placebo_gain_valid"]:
        return "MAGNITUDE_ONLY"
    if not primary_core and flags["latent_gain_valid"]:
        return "LATENT_ONLY"
    return "NULL_OR_REVERSE"


def protocol_sha256() -> str:
    encoded = json.dumps(
        protocol_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_protocol() -> None:
    payload = protocol_payload()
    if tuple(payload["arms"]["primary"]) != PRIMARY_ARMS:
        raise RuntimeError("primary arms drifted")
    seed_sets = {
        "confirmatory": set(CONFIRMATORY_SEEDS),
        "calibration": set(CALIBRATION_SEEDS),
        "mechanism_engineering": set(MECHANISM_ENGINEERING_SEEDS),
        "optional_scale_shift": set(OPTIONAL_SCALE_SHIFT_SEEDS),
    }
    expected_seed_counts = {
        "confirmatory": 12,
        "calibration": 4,
        "mechanism_engineering": 2,
        "optional_scale_shift": 4,
    }
    for role, expected_count in expected_seed_counts.items():
        values = {
            "confirmatory": CONFIRMATORY_SEEDS,
            "calibration": CALIBRATION_SEEDS,
            "mechanism_engineering": MECHANISM_ENGINEERING_SEEDS,
            "optional_scale_shift": OPTIONAL_SCALE_SHIFT_SEEDS,
        }[role]
        if len(values) != expected_count or len(set(values)) != expected_count:
            raise RuntimeError(f"{role} seed role is not {expected_count} unique blocks")
    if len(CONFIRMATORY_SEEDS) != 12 or len(seed_sets["confirmatory"]) != 12:
        raise RuntimeError("confirmatory seed inventory is not 12 unique blocks")
    if len(REPLICATION_A_SEEDS) != 6 or len(REPLICATION_B_SEEDS) != 6:
        raise RuntimeError("each replication must contain six seed blocks")
    if set(REPLICATION_A_SEEDS) & set(REPLICATION_B_SEEDS):
        raise RuntimeError("replication seed blocks overlap")
    if set(REPLICATION_A_SEEDS) | set(REPLICATION_B_SEEDS) != seed_sets["confirmatory"]:
        raise RuntimeError("replication seed partition is incomplete")
    all_role_transfer_seeds = set().union(*seed_sets.values())
    if len(all_role_transfer_seeds) != sum(len(value) for value in seed_sets.values()):
        raise RuntimeError("role-transfer seed roles overlap")
    if all_role_transfer_seeds & set(UPSTREAM_SEEDS):
        raise RuntimeError("role-transfer seeds reuse upstream seeds")
    if tuple(payload["contrasts"]["primary_efficacy"]) != PRIMARY_CONTRAST:
        raise RuntimeError("primary contrast drifted")
    if payload["evaluator"]["primary_endpoint"] != PRIMARY_ENDPOINT:
        raise RuntimeError("primary endpoint drifted")
    if payload["exposure"]["dose_selection"] is not False:
        raise RuntimeError("confirmatory dose selection must remain disabled")
    if payload["data"]["external_benchmark_claim"] is not False:
        raise RuntimeError("within-ontology design cannot claim external benchmark validity")
    if payload["execution_authorized"] is not False or payload["local_execution_authorized"] is not False:
        raise RuntimeError("design-only protocol must not authorize execution")
    if set(DECISION_ORDER) != set(RESULT_LABELS) or set(DECISION_ORDER) != set(DECISION_RULES):
        raise RuntimeError("decision order labels and rules are not exhaustive")
    if set(OUTCOME_FLAGS) != set(payload["outcome_flags"]):
        raise RuntimeError("outcome flag contract drifted")
    if classify_outcome({key: False for key in OUTCOME_FLAGS}) != "INCONCLUSIVE_PROTOCOL":
        raise RuntimeError("classifier does not fail closed")
    if classify_outcome({key: True for key in OUTCOME_FLAGS}) != "SUPPORTED_TRANSFER":
        raise RuntimeError("classifier all-pass branch drifted")
    if payload["latent_diagnostic"]["may_support_primary_claim"] is not False:
        raise RuntimeError("latent diagnostic cannot support the primary claim")
    if payload["latent_diagnostic"]["manifest_sha256_must_freeze_before_stage_1"] is not True:
        raise RuntimeError("latent manifest hash gate drifted")
    if payload["norm_receipt_contract"]["hash_chain_required"] is not True:
        raise RuntimeError("norm receipt hash chain is required")
    if payload["mechanism_manipulation"]["stratified_interval_lower_bound_above_zero"] is not True:
        raise RuntimeError("mechanism manipulation interval gate drifted")
    if payload["magnitude_matching"]["global_clip_norm"] != GLOBAL_CLIP_NORM:
        raise RuntimeError("global clip norm drifted")
    if payload["magnitude_matching"]["relative_norm_match_tolerance"] != NORM_MATCH_RELATIVE_TOLERANCE:
        raise RuntimeError("norm-match tolerance drifted")
    if protocol_sha256() != CANONICAL_PROTOCOL_SHA256:
        raise RuntimeError("canonical ROLE-TRANSFER protocol hash mismatch")
