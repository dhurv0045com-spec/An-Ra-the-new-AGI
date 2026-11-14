"""Hitchhiker mechanism-identification protocol and experiment registry.

Import performs no device, filesystem, or model work. Historical V6/V7/V7.17
protocol files remain untouched; this module establishes an explicit Hitchhiker
protocol namespace bound to the audited V7.17 source commit.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from v5_contracts.model_spec import ModelSpec
from .codec import digest
from .representation import UTF8
from .v6_registry import MODEL_PROFILES


SCHEMA = "elendril-hitchhiker-protocol-v1"
BRANCH_BASE_COMMIT = "2affffb806b7f73b00bc21aa90538019f37ce904"
AUDITED_V717_COMMIT = "0c9cb284d2c8c949e17ac342959c6f2bf351bd30"
V717_OPTIMIZATION_SOURCE = "724af4cb508173e0eb3f4397d5dcb4a42a66712f"
V717_OPTIMIZATION_RUN_ID = "f443d3f4141b4163bbc130d46ebd7395"
HISTORICAL_V717_PROTOCOL_SHA256 = "b3030032f5758a1eaf81b6677db9655b475fd18955feda7b28c627cb6eee0088"

SEEDS = (7, 19, 41)
REGISTERED_SCALES = ("S3", "S8", "S20")

HISTORICAL_S20_LOWLR_MANIFESTS: dict[int, dict[str, Any]] = {
    seed: {
        "lineage": f"S20-low-lr-i{seed}-o{seed}",
        "seed": seed,
        "order_seed": seed,
        "source_commit": V717_OPTIMIZATION_SOURCE,
        "source_run_id": V717_OPTIMIZATION_RUN_ID,
        "protocol_sha256": HISTORICAL_V717_PROTOCOL_SHA256,
        "profile": "S20",
        "model_spec_sha256": MODEL_PROFILES["S20"].sha256(),
        "tokenizer_sha256": UTF8.identity,
        "lr": 1e-4,
        "epochs": 4,
        "batch_size": 16,
        "micro_batch_size": 2,
        "method": "blocked",
        "mechanism": "ce",
        "expected_update": 2368,
        "train_seed": 606,
        "train_bytes": 2200000,
        "checkpoint_sha256": ckpt_sha,
    }
    for seed, ckpt_sha in (
        (7, "686c5f146c62d546bd3c7c536d9789b07cce83fc88becaa3ab3c7c3c799f465b"),
        (19, "c367bc768f392b6cd6735ed177cba6e1c54452e928625645486d43b275df3db3"),
        (41, "d98aec62db2c06f1c657c44c8390e9b303abf4d413fd9875afdbb75007c7bb7f"),
    )
}
DECODER_CONDITIONS = ("raw", "availability-only", "grammar")
STAGES = (
    "stage0_checkpoint_audit",
    "stage1_s3_screen",
    "stage2_followups",
    "stage3_scale_confirmation",
)

STAGE1_ARMS = {
    "B0": {
        "data_mode": "static",
        "relabel": False,
        "coverage": False,
        "aux_illegal_weight": 0.0,
        "external_slots": False,
        "description": "Static original-distribution CE baseline at 1e-4.",
    },
    "A": {
        "data_mode": "fresh",
        "relabel": False,
        "coverage": False,
        "aux_illegal_weight": 0.0,
        "external_slots": False,
        "description": "Fresh procedural worlds matching B0 task distribution and token exposure.",
    },
    "B": {
        "data_mode": "static",
        "relabel": True,
        "coverage": False,
        "aux_illegal_weight": 0.0,
        "external_slots": False,
        "description": "Static worlds with episode-consistent semantic A-P entity relabeling.",
    },
    "AB": {
        "data_mode": "fresh",
        "relabel": True,
        "coverage": False,
        "aux_illegal_weight": 0.0,
        "external_slots": False,
        "description": "Fresh procedural worlds plus episode-consistent entity relabeling.",
    },
    "C": {
        "data_mode": "static",
        "relabel": False,
        "coverage": True,
        "aux_illegal_weight": 0.0,
        "external_slots": False,
        "description": "Matched coverage crossing context completeness and availability, pure CE.",
    },
}

STAGE2_ARMS = {
    "AB+C": {
        "data_mode": "fresh",
        "relabel": True,
        "coverage": True,
        "aux_illegal_weight": 0.0,
        "external_slots": False,
        "description": "Conditional combination of fresh worlds, relabeling, and coverage.",
    },
    "C+penalty": {
        "data_mode": "static",
        "relabel": False,
        "coverage": True,
        "aux_illegal_weight": 0.1,
        "external_slots": False,
        "description": "Coverage supervision plus auxiliary illegal-action penalty lambda=0.1.",
    },
    "D-external": {
        "data_mode": "static",
        "relabel": False,
        "coverage": False,
        "aux_illegal_weight": 0.0,
        "external_slots": True,
        "origin": "engineered_canonicalizer",
        "description": "Public-only external slot canonicalization control (not learned binding).",
    },
    "S20-contrast": {
        "data_mode": "fresh",
        "relabel": True,
        "coverage": False,
        "aux_illegal_weight": 0.0,
        "external_slots": False,
        "profile": "S20",
        "description": "Preregistered S20 three-seed contrast when capacity interaction is plausible.",
    },
}

ALL_ARMS = {**STAGE1_ARMS, **STAGE2_ARMS}

SECOND_WAVE_MECHANISMS = {
    "D-learned": {
        "name": "Learned slot formation from raw bytes",
        "status": "DEFERRED_PENDING_STAGE1_GATES",
        "clean_formulation": "Learned mention-to-slot assignment and pointer output under a versioned contract.",
        "required_controls": ["external D-external canonicalizer", "matched raw baseline"],
        "falsifier": "External slots work while raw-input learned binder fails.",
    },
    "E": {
        "name": "Recurrent reasoning / pondering loop",
        "status": "DEFERRED_PENDING_STAGE1_GATES",
        "clean_formulation": "Shared transition block over working state for fixed 1/2/3/4 steps before learned stopping.",
        "required_controls": ["equal-work feed-forward control", "repeated no-op state perturbations"],
        "falsifier": "No benefit over equal-work feed-forward or intermediate-state corruption fails to alter behavior.",
    },
    "F": {
        "name": "Contrastive falsification / forward-support verifier",
        "status": "DEFERRED_PENDING_STAGE1_GATES",
        "clean_formulation": "Candidate-conditioned forward-support verifier trained with wrong-edge, stale, and missing-bridge negatives.",
        "required_controls": ["equal-cost candidate scoring", "random negatives"],
        "falsifier": "Verifier shares the base shortcut or agrees while wrong on causal edge edits.",
    },
    "G": {
        "name": "Calibrated entropy / uncertainty-gated modulation",
        "status": "DEFERRED_PENDING_STAGE1_GATES",
        "clean_formulation": "Routing based on calibrated output distribution entropy and public availability flags.",
        "required_controls": ["fixed routing", "availability-only routing at equal cost"],
        "falsifier": "High-confidence errors (e.g., 0.9994 invalid RETRIEVE) render uncertainty uninformative.",
    },
    "H": {
        "name": "Orthogonal associative working-memory scratchpad",
        "status": "DEFERRED_PENDING_STAGE1_GATES",
        "clean_formulation": "Episode-local learned key/value fast-weight matrix with explicit episode reset and collision metrics.",
        "required_controls": ["matched-work ordinary attention", "slot baseline", "write/read ablation"],
        "falsifier": "Key collisions/stale overwrites break chains or write/read ablations leave outputs unchanged.",
    },
}

SCREENING_GATE = {
    "min_mean_primary_gain": 0.10,
    "require_all_seeds_positive": True,
    "max_original_kind_regression": 0.05,
    "require_world_cluster_ci_positive": True,
    "min_dev_worlds_per_hop": 256,
    "status_labels": ("PASS_SCREENING", "INCONCLUSIVE", "FALSIFIED"),
}

FORMATION_GATE = {
    "min_raw_accuracy_per_hop": 0.90,
    "min_stress_subgroup_accuracy": 0.80,
    "min_answer_changing_pair_accuracy": 0.90,
    "min_renaming_consistency": 0.95,
    "max_invalid_action_rate": 0.01,
    "min_confirmation_worlds_per_cohort": 512,
    "status_labels": ("FORMATION_QUALIFIED", "MECHANICAL_ONLY_NOT_FORMATION_QUALIFIED", "FALSIFIED"),
}


def resolve_scale_profile(scale: str, custom_spec_receipt: dict[str, Any] | None = None) -> ModelSpec:
    """Resolve registered scale S3/S8/S20 or validate an explicit receipt for any new scale."""
    if scale in REGISTERED_SCALES:
        return MODEL_PROFILES[scale]
    if not isinstance(custom_spec_receipt, dict):
        raise ValueError(
            f"unregistered scale '{scale}' (registered: {REGISTERED_SCALES}); "
            "provide an explicit ModelSpec and verified parameter receipt"
        )
    spec = custom_spec_receipt.get("spec")
    if isinstance(spec, dict):
        spec = ModelSpec(**spec)
    if not isinstance(spec, ModelSpec):
        raise ValueError("custom scale receipt requires a ModelSpec")
    spec.assert_valid()
    receipt = spec.parameter_receipt()
    if custom_spec_receipt.get("parameters") != receipt.total:
        raise ValueError("custom scale parameter count disagrees with ModelSpec receipt")
    if custom_spec_receipt.get("model_sha256") != spec.sha256():
        raise ValueError("custom scale model_sha256 disagrees with ModelSpec")
    return spec


def _scale_receipts() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "model": MODEL_PROFILES[name].canonical(),
            "parameters": MODEL_PROFILES[name].parameter_receipt().total,
            "model_sha256": MODEL_PROFILES[name].sha256(),
        }
        for name in REGISTERED_SCALES
    }


def _training_job(stage: str, arm: str, seed: int, *, profile: str) -> dict[str, Any]:
    spec = dict(ALL_ARMS[arm])
    resolved_profile = spec.pop("profile", profile)
    resolve_scale_profile(resolved_profile)
    return {
        "job_id": f"HH-{stage}-{resolved_profile}-{arm}-s{seed}",
        "stage": stage,
        "arm": arm,
        "profile": resolved_profile,
        "seed": seed,
        "order_seed": seed,
        "lr": 1e-4,
        "epochs": 4,
        "batch_size": 16,
        "micro_batch_size": 2,
        "data_mode": spec["data_mode"],
        "relabel": spec["relabel"],
        "coverage": spec["coverage"],
        "aux_illegal_weight": spec["aux_illegal_weight"],
        "external_slots": spec["external_slots"],
        "attribution_origin": spec.get("origin", "model"),
    }


def stage_jobs(stage: str, *, winning_arm: str = "AB") -> list[dict[str, Any]]:
    """Return the preregistered jobs for a declared Hitchhiker stage."""
    if stage not in STAGES:
        raise ValueError(f"unknown Hitchhiker stage: {stage}")
    if stage == "stage0_checkpoint_audit":
        return [
            {
                "job_id": f"HH-stage0-S20-lowlr-s{seed}-{decoder}",
                "stage": stage,
                "profile": "S20",
                "seed": seed,
                "source_run_id": V717_OPTIMIZATION_RUN_ID,
                "source_commit": V717_OPTIMIZATION_SOURCE,
                "parent_lineage": f"S20-low-lr-i{seed}-o{seed}",
                "decoder": decoder,
                "include_answer_only_diagnostic": True,
                "training_updates": 0,
            }
            for seed in SEEDS
            for decoder in DECODER_CONDITIONS
        ]
    if stage == "stage1_s3_screen":
        return [
            _training_job(stage, arm, seed, profile="S3")
            for arm in ("B0", "A", "B", "AB", "C")
            for seed in SEEDS
        ]
    if stage == "stage2_followups":
        return [
            _training_job(stage, arm, seed, profile="S3")
            for arm in ("AB+C", "C+penalty", "D-external", "S20-contrast")
            for seed in SEEDS
        ]
    if winning_arm not in ALL_ARMS or winning_arm == "B0":
        raise ValueError("scale confirmation requires a valid non-baseline winning arm")
    rows = []
    for scale in ("S8", "S20"):
        for arm in ("B0", winning_arm):
            for seed in SEEDS:
                job = _training_job(stage, arm, seed, profile=scale)
                job["comparison_role"] = "baseline" if arm == "B0" else "treatment"
                rows.append(job)
    return rows


def validate_job(job: dict[str, Any], stage: str, *, winning_arm: str = "AB") -> None:
    """Validate that a job matches the registered Hitchhiker stage specification."""
    if job not in stage_jobs(stage, winning_arm=winning_arm):
        raise ValueError(f"job outside preregistered Hitchhiker stage '{stage}'")


def validate_stage_selection(
    stage: str,
    selection_receipt: dict[str, Any] | None,
    *,
    winning_arm: str | None = None,
) -> dict[str, Any]:
    """Enforce conditional admission gates before Stage 2 or Stage 3 execution."""
    if stage not in STAGES:
        raise ValueError(f"unknown Hitchhiker stage: {stage}")
    if stage in ("stage0_checkpoint_audit", "stage1_s3_screen"):
        jobs = stage_jobs(stage)
        return {
            "status": "ADMITTED",
            "stage": stage,
            "gate_required": False,
            "jobs": len(jobs),
        }
    if not isinstance(selection_receipt, dict):
        raise ValueError(f"stage '{stage}' requires an explicit selection_receipt")

    if stage == "stage2_followups":
        if selection_receipt.get("stage1_completed") is not True:
            raise ValueError("stage2_followups requires Stage 1 completion (stage1_completed=True)")
        raw_admitted = selection_receipt.get("admitted_followups")
        if isinstance(raw_admitted, dict):
            if not raw_admitted or any(not isinstance(v, str) or not v.strip() for v in raw_admitted.values()):
                raise ValueError("stage2_followups dict admitted_followups requires non-empty rationales")
            admitted_arms = list(raw_admitted.keys())
        elif isinstance(raw_admitted, (list, tuple, set)):
            admitted_arms = list(raw_admitted)
        else:
            raise ValueError("stage2_followups requires admitted_followups arms list")
        if not admitted_arms:
            raise ValueError("stage2_followups requires at least one admitted followup arm")
        for arm in admitted_arms:
            if arm not in STAGE2_ARMS:
                raise ValueError(f"unknown Stage 2 followup arm: {arm!r}")
        for rkey in ("rationale", "conditional_rationale"):
            if rkey in selection_receipt and (
                not isinstance(selection_receipt[rkey], str) or not selection_receipt[rkey].strip()
            ):
                raise ValueError(f"stage2_followups {rkey} must be a non-empty string")
        if winning_arm is not None:
            if winning_arm not in STAGE2_ARMS:
                raise ValueError(f"unknown Stage 2 arm: {winning_arm!r}")
            if winning_arm not in admitted_arms:
                raise ValueError(f"requested Stage 2 arm {winning_arm!r} is not in admitted_followups")
        all_stage2_jobs = stage_jobs("stage2_followups")
        if len({j["job_id"] for j in all_stage2_jobs}) != len(all_stage2_jobs):
            raise ValueError("duplicate job_id detected in stage2_followups")
        for job in all_stage2_jobs:
            if job["profile"] not in REGISTERED_SCALES:
                raise ValueError(f"unregistered profile {job['profile']!r} in stage2_followups")
        selected_arms = [winning_arm] if winning_arm is not None else admitted_arms
        admitted_jobs = [j for j in all_stage2_jobs if j["arm"] in selected_arms]
        return {
            "status": "ADMITTED",
            "stage": stage,
            "gate_required": True,
            "admitted_followups": admitted_arms,
            "requested_arm": winning_arm,
            "jobs": len(admitted_jobs),
            "job_ids": [j["job_id"] for j in admitted_jobs],
        }

    if selection_receipt.get("screening_status") != "PASS_SCREENING":
        raise ValueError("stage3_scale_confirmation requires screening_status == 'PASS_SCREENING'")
    eval_split = selection_receipt.get("split", selection_receipt.get("evaluation_split"))
    if eval_split != "validation":
        raise ValueError("stage3_scale_confirmation screening gate must be evaluated on split='validation'")
    locked_arm = selection_receipt.get("winning_arm")
    if not locked_arm or locked_arm not in ALL_ARMS or locked_arm == "B0":
        raise ValueError("stage3_scale_confirmation requires a non-baseline registered winning_arm")
    if winning_arm is not None and winning_arm != locked_arm:
        raise ValueError(f"requested winning_arm {winning_arm!r} disagrees with locked receipt {locked_arm!r}")
    model_profile = selection_receipt.get("model_profile")
    if model_profile not in ("S8", "S20"):
        raise ValueError("stage3_scale_confirmation requires model_profile in ('S8', 'S20')")
    resolve_scale_profile(model_profile)
    jobs = stage_jobs("stage3_scale_confirmation", winning_arm=locked_arm)
    if len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate job_id detected in stage3_scale_confirmation")
    return {
        "status": "ADMITTED",
        "stage": stage,
        "gate_required": True,
        "winning_arm": locked_arm,
        "model_profile": model_profile,
        "split": eval_split,
        "jobs": len(jobs),
        "job_ids": [j["job_id"] for j in jobs],
    }


def protocol() -> dict[str, Any]:
    """Return the canonical Hitchhiker scientific protocol manifest."""
    return {
        "schema": SCHEMA,
        "branch_base_commit": BRANCH_BASE_COMMIT,
        "audited_v717_commit": AUDITED_V717_COMMIT,
        "v717_optimization_source": V717_OPTIMIZATION_SOURCE,
        "v717_optimization_run_id": V717_OPTIMIZATION_RUN_ID,
        "historical_s20_lowlr_manifests": HISTORICAL_S20_LOWLR_MANIFESTS,
        "seeds": list(SEEDS),
        "scales": _scale_receipts(),
        "decoder_conditions": list(DECODER_CONDITIONS),
        "arms": ALL_ARMS,
        "stages": {
            stage: stage_jobs(stage)
            for stage in STAGES
        },
        "second_wave_mechanisms": SECOND_WAVE_MECHANISMS,
        "screening_gate": SCREENING_GATE,
        "formation_gate": FORMATION_GATE,
        "attribution_rule": (
            "External canonicalization, availability masks, and grammar decoding "
            "must remain separately attributed from learned representations."
        ),
        "scientific_status": "PROTOCOL_AND_DATA_REPAIRED_PENDING_EXECUTION",
    }


PROTOCOL_SHA = digest(protocol())
