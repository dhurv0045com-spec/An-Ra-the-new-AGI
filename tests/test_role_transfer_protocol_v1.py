from __future__ import annotations

import json
from pathlib import Path

from v5_experiments import role_transfer_protocol_v1 as protocol


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "docs/cymek/experiments/ROLE-TRANSFER-001/PREREGISTRATION_V1.json"
READINESS = ROOT / "docs/cymek/experiments/ROLE-TRANSFER-001/RUN_READINESS_V1.json"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _flags(**overrides: bool) -> dict[str, bool]:
    flags = {
        "protocol_valid": True,
        "substrate_valid": True,
        "clipping_valid": True,
        "manipulation_valid": True,
        "primary_gain_valid": False,
        "primary_p_valid": False,
        "primary_ci_valid": False,
        "replication_valid": False,
        "domain_transfer_valid": False,
        "guards_valid": False,
        "role_specificity_valid": False,
        "placebo_equivalent_valid": False,
        "placebo_gain_valid": False,
        "latent_gain_valid": False,
    }
    flags.update(overrides)
    return flags


def test_protocol_invariants() -> None:
    protocol.assert_protocol()
    role_seed_sets = [
        set(protocol.CONFIRMATORY_SEEDS),
        set(protocol.CALIBRATION_SEEDS),
        set(protocol.MECHANISM_ENGINEERING_SEEDS),
        set(protocol.OPTIONAL_SCALE_SHIFT_SEEDS),
    ]
    assert len(protocol.CONFIRMATORY_SEEDS) == 12
    assert len(set(protocol.CONFIRMATORY_SEEDS)) == 12
    assert len(protocol.REPLICATION_A_SEEDS) == 6
    assert len(protocol.REPLICATION_B_SEEDS) == 6
    assert set(protocol.REPLICATION_A_SEEDS).isdisjoint(protocol.REPLICATION_B_SEEDS)
    assert set(protocol.REPLICATION_A_SEEDS) | set(protocol.REPLICATION_B_SEEDS) == set(
        protocol.CONFIRMATORY_SEEDS
    )
    flattened = [value for values in role_seed_sets for value in values]
    assert len(protocol.CALIBRATION_SEEDS) == 4
    assert len(set(protocol.CALIBRATION_SEEDS)) == 4
    assert len(protocol.MECHANISM_ENGINEERING_SEEDS) == 2
    assert len(set(protocol.MECHANISM_ENGINEERING_SEEDS)) == 2
    assert len(protocol.OPTIONAL_SCALE_SHIFT_SEEDS) == 4
    assert len(set(protocol.OPTIONAL_SCALE_SHIFT_SEEDS)) == 4
    assert len(flattened) == len(set(flattened))
    assert set(flattened).isdisjoint(protocol.UPSTREAM_SEEDS)
    assert protocol.PROCESSED_TOKEN_BUDGET == 2_000_000
    assert protocol.PRACTICAL_THRESHOLDS["minimum_macro_success_gain"] == 0.08
    assert protocol.PRACTICAL_THRESHOLDS["confidence_interval_excludes_zero"] is True
    assert protocol.PRACTICAL_THRESHOLDS["maximum_unauthorized_actions"] == 0
    assert protocol.PRACTICAL_THRESHOLDS[
        "maximum_official_clip_fraction_per_seed_arm"
    ] == 0.05
    assert protocol.MAGNITUDE_MATCHING["scope"] == (
        "full_trainable_gradient_l2_norm_before_one_global_clip"
    )
    assert set(protocol.MAGNITUDE_MATCHING["reference_trajectories"]) == {
        "T0_CANONICAL",
        "T3_RAW",
    }
    assert "all four arms advance in the same seed block" in (
        protocol.MAGNITUDE_MATCHING["lockstep_requirement"]
    )
    assert "current T0_CANONICAL trajectory" in protocol.MAGNITUDE_MATCHING["T3_NORM"]
    assert "current T3_RAW trajectory" in protocol.MAGNITUDE_MATCHING["P_NORM"]
    assert protocol.MAGNITUDE_MATCHING["global_clip_norm"] == 1.0
    assert protocol.MAGNITUDE_MATCHING["relative_norm_match_tolerance"] == 1e-6
    assert "identical per-update clip decisions" in (
        protocol.MAGNITUDE_MATCHING["paired_clip_invariant"]
    )
    assert protocol.DECISION_ORDER == (
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
    assert set(protocol.DECISION_ORDER) == set(protocol.RESULT_LABELS)
    assert set(protocol.DECISION_ORDER) == set(protocol.DECISION_RULES)
    assert protocol.LATENT_DIAGNOSTIC["endpoint"] == "latent_id_identity_formation_auc"
    assert protocol.LATENT_DIAGNOSTIC["split"] == "frozen_development_only_manifest"
    assert protocol.LATENT_DIAGNOSTIC["minimum_gain"] == 0.05
    assert protocol.LATENT_DIAGNOSTIC["may_support_primary_claim"] is False
    assert protocol.LATENT_DIAGNOSTIC[
        "manifest_sha256_must_freeze_before_stage_1"
    ] is True
    assert protocol.LATENT_DIAGNOSTIC[
        "evaluator_sha256_must_freeze_before_stage_1"
    ] is True
    payload = protocol.protocol_payload()
    assert payload["norm_receipt_contract"]["schema"] == (
        "anra.role-transfer-norm-receipt/v1"
    )
    assert payload["norm_receipt_contract"]["hash_chain_required"] is True
    assert payload["norm_receipt_contract"][
        "checkpoint_receipt_must_cover_every_update"
    ] is True
    assert payload["mechanism_manipulation"][
        "stratified_interval_lower_bound_above_zero"
    ] is True
    assert protocol.PRACTICAL_THRESHOLDS[
        "minimum_output_to_input_norm_ratio_reduction"
    ] == 0.25
    assert protocol.PRACTICAL_THRESHOLDS[
        "role_specificity_interval_lower_bound_above_zero"
    ] is True
    assert protocol.PRACTICAL_THRESHOLDS[
        "magnitude_placebo_interval_upper_bound_below"
    ] == 0.05
    optional = protocol.OPTIONAL_SCALE_SHIFT_SPEC
    assert optional["width"] == optional["query_heads"] * optional["head_dimension"]
    assert optional["width"] % (optional["key_value_heads"] * optional["head_dimension"]) == 0


def test_outcome_classifier_truth_table() -> None:
    primary = {
        "primary_gain_valid": True,
        "primary_p_valid": True,
        "primary_ci_valid": True,
    }
    full_primary = {
        **primary,
        "replication_valid": True,
        "domain_transfer_valid": True,
        "guards_valid": True,
    }
    assert protocol.classify_outcome(_flags()) == "NULL_OR_REVERSE"
    assert protocol.classify_outcome(_flags(latent_gain_valid=True)) == "LATENT_ONLY"
    assert protocol.classify_outcome(_flags(placebo_gain_valid=True)) == "MAGNITUDE_ONLY"
    assert protocol.classify_outcome(_flags(**primary)) == "PRIMARY_NONREPLICATED"
    assert protocol.classify_outcome(
        _flags(**primary, replication_valid=True)
    ) == "PRIMARY_NONTRANSFER"
    assert protocol.classify_outcome(_flags(**full_primary)) == "MECHANISM_UNRESOLVED"
    assert protocol.classify_outcome(
        _flags(**full_primary, placebo_gain_valid=True)
    ) == "MAGNITUDE_EFFECT"
    assert protocol.classify_outcome(
        _flags(
            **full_primary,
            role_specificity_valid=True,
            placebo_equivalent_valid=True,
        )
    ) == "SUPPORTED_TRANSFER"
    assert protocol.classify_outcome(_flags(protocol_valid=False)) == "INCONCLUSIVE_PROTOCOL"
    assert protocol.classify_outcome(_flags(substrate_valid=False)) == "INCONCLUSIVE_SUBSTRATE"
    assert protocol.classify_outcome(_flags(clipping_valid=False)) == "CLIPPING_CONFOUNDED"
    assert protocol.classify_outcome(_flags(manipulation_valid=False)) == "MECHANISM_NOT_ENGAGED"


def test_preregistration_matches_protocol() -> None:
    preregistration = _json(PREREGISTRATION)
    assert preregistration["campaign"] == protocol.CAMPAIGN
    assert preregistration["protocol_source"] == "v5_experiments/role_transfer_protocol_v1.py"
    assert preregistration["protocol"] == protocol.protocol_payload()
    assert preregistration["protocol_sha256"] in {
        "PENDING_REMOTE_CANONICALIZATION",
        protocol.protocol_sha256(),
    }
    authorization = preregistration["authorization"]
    assert isinstance(authorization, dict)
    for key in (
        "official_experiments",
        "sealed_evaluation",
        "local_execution",
        "remote_execution",
    ):
        assert authorization[key] is False
    assert authorization["blocking_stage"] == (
        "implementation_and_independent_remote_qualification"
    )


def test_readiness_blocks_execution_and_rejects_bruteforce_scale() -> None:
    readiness = _json(READINESS)
    assert readiness["campaign"] == protocol.CAMPAIGN
    assert readiness["protocol_source"] == "v5_experiments/role_transfer_protocol_v1.py"
    assert readiness["preregistration"] == (
        "docs/cymek/experiments/ROLE-TRANSFER-001/PREREGISTRATION_V1.json"
    )
    assert readiness["protocol_sha256"] in {
        "PENDING_REMOTE_CANONICALIZATION",
        protocol.protocol_sha256(),
    }
    assert readiness["status"] == "DESIGN_PREREGISTERED_EXECUTION_BLOCKED"
    assert readiness["upstream_evidence"]["current_campaign_must_remain_frozen"] is True
    assert readiness["scale_request"]["literal_10000x_multiplier_rejected"] is True
    assert readiness["scale_request"][
        "planned_multiplier_relative_to_current_token_budget_per_arm"
    ] == 4
    assert readiness["scale_request"]["mandatory_processed_token_exposures"] == 120_000_000
    assert readiness["scale_request"]["compute_estimate"] == (
        "PENDING_PRE_OUTCOME_T4X2_ENGINEERING_CALIBRATION"
    )
    remote = readiness["remote_execution_policy"]
    assert isinstance(remote, dict)
    assert remote["local_training_authorized"] is False
    assert remote["local_tests_authorized"] is False
    assert remote["official_kaggle_t4x2_execution_authorized"] is False
    assert remote["exact_resume_required"] is True
    assert remote["dose_selection_forbidden"] is True
    assert remote["must_not_import_or_mutate_formation_mux_output"] is True
    stage_1 = readiness["stage_1_control_only_calibration"]
    assert isinstance(stage_1, dict)
    assert stage_1["fixed_token_dose"] == 2_000_000
    assert stage_1["maximum_clip_fraction_each_seed"] == 0.05
    confirmatory = readiness["confirmatory_experiment"]
    assert isinstance(confirmatory, dict)
    assert confirmatory["arms"] == list(protocol.MECHANISM_ARMS)
    assert confirmatory["minimum_positive_blocks_per_replication"] == 4
    assert confirmatory["confidence_interval_excludes_zero"] is True
    assert confirmatory["maximum_clip_fraction_each_seed_arm"] == 0.05
    assert confirmatory["paired_clip_decisions_must_match"] is True
    assert confirmatory["minimum_output_to_input_norm_ratio_reduction"] == 0.25
    assert confirmatory["role_specificity_interval_lower_bound_above_zero"] is True
    assert confirmatory["magnitude_placebo_interval_upper_bound_below"] == 0.05
    assert confirmatory["maximum_unauthorized_actions"] == 0
    assert readiness["scope_limit"] == "within_synthetic_ontology_not_external_benchmark_validity"
    blockers = readiness["blocking_requirements"]
    assert isinstance(blockers, list)
    assert len(blockers) == 10
    assert any("protocol hash" in item for item in blockers)
