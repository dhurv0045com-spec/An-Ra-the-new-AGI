"""Immutable ELENDRIL V8-X1 runtime protocol registry, core 42 jobs, optional extensions, and hypotheses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anra_v5.elendril.codec import digest
from anra_v5.planthir.contracts import CausalContrastSpec, FalsifierSpec, HypothesisSpec
from anra_v5.planthir.design import TypedContrastSpec, TypedFactorSpec

from .contracts import (
    ACTION_19_LABELS,
    POLICY_MODES,
    X1_D2_DATASET_SCHEMA,
    X1_DATASET_SCHEMA,
    X1_FAMILIES,
    X1_MAX_SEQUENCE,
    build_x1_model_spec,
)

X1_PROTOCOL_SCHEMA = "elendril-x1-runtime-protocol-v1"
X1_SEEDS = ((7, 7), (19, 19), (41, 41))
X1_REPLICATION_SEEDS = ((101, 101), (211, 211), (307, 307))
X1_MILESTONES = (0, 592, 1184, 1472, 2368)
X1_INTERMEDIATE_WEIGHT_MILESTONES = (0, 592, 1184, 1472)
X1_TOTAL_UPDATES = 2368

X1_CORE_SESSIONS = ("A0", "A1", "A2", "A3", "D0", "D1", "D2")
X1_SESSION_LABELS = {
    "A0": "X1-A0 (P vs R on D1-H)",
    "A1": "X1-A1 (F vs F-cap on D1-H)",
    "A2": "X1-A2 (R-noedge vs P-capR on D1-H)",
    "A3": "X1-A3 (B0 vs J on D1-H)",
    "D0": "X1-PAIR-CE (P/R on D2 paired views)",
    "D1": "X1-PAIR-EQ (P/R with JS equivariance)",
    "D2": "X1-PAIR-EQ-CF (P/R with JS equivariance + CF margin)",
    "U0": "EXT-U0 (U-1 vs U-2 recurrence extension)",
    "U1": "EXT-U1 (U-4 vs U2-untied recurrence extension)",
    "RT": "EXT-RETENTION-TRANSFER (conditional A->B->C extension)",
    "H03": "EXT-D1-EXPOSURE (B0 historical D0/D1 at 2368 updates)",
}

X1_OPTIMIZER_DEFAULTS = {
    "optimizer": "AdamW",
    "peak_lr": 1e-4,
    "betas": (0.9, 0.95),
    "epsilon": 1e-8,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "precision": "fp32",
    "schedule": "cosine",
    "cosine_terminal_ratio": 0.1,
    "effective_state_batch": 16,
    "normal_micro_batch_states": 2,
    "paired_groups_per_step": 4,
    "paired_states_per_group": 4,
    "paired_micro_batch_groups": 1,
    "total_updates": X1_TOTAL_UPDATES,
}

X1_D2_LOSS_DEFAULTS = {
    "view_count": 4,
    "groups_per_effective_batch": 4,
    "eq_weight": 0.1,
    "cf_weight": 0.1,
    "cf_margin": 0.5,
    "auxiliary_denominator": "eligible pairs across effective batch",
}

X1_SESSION_CONFIG = {
    "session_max_minutes": 330,
    "minimum_export_reserve_minutes": 45,
    "checkpoint_interval_updates": 64,
    "active_gpus_required": 2,
    "default_first_session": "A0",
}

X1_HYPOTHESIS_REGISTER = (
    {"id": "H01", "group": "instrument", "name": "Consumed-tool transition repair", "comparator": "legacy_v8 vs corrected"},
    {"id": "H02", "group": "surface", "name": "Surface representation sensitivity", "comparator": "compact vs unfamiliar renderer"},
    {"id": "H03", "group": "exposure", "name": "Recipe x exposure horizon", "comparator": "1472 vs 2368 on D0/D1"},
    {"id": "H04", "group": "core", "name": "Typed output interface", "comparator": "J/F vs B0; J algebraic null"},
    {"id": "H05", "group": "core", "name": "Evidence pointer reading", "comparator": "P vs F and F-cap"},
    {"id": "H06", "group": "core", "name": "Relational slot computation", "comparator": "R vs P, R-noedge, P-capR"},
    {"id": "H07", "group": "core", "name": "Functional equivariance", "comparator": "PAIR-EQ vs PAIR-CE within P and R"},
    {"id": "H08", "group": "core", "name": "Counterfactual sensitivity", "comparator": "PAIR-EQ-CF vs PAIR-EQ"},
    {"id": "H09", "group": "extension", "name": "Recurrent refinement", "comparator": "U-1 vs U-2 vs U-4 vs U2-untied"},
    {"id": "H10", "group": "conditional", "name": "Skill preservation under replay", "comparator": "A+B replay vs no-replay & equal-work B"},
    {"id": "H11", "group": "conditional", "name": "Forward transfer to composition C", "comparator": "A->B vs unrelated-history & fresh start"},
)

X1_ADVANCEMENT_THRESHOLDS = {
    "development_mean_gain_min": 0.05,
    "development_positive_seeds_min": 2,
    "development_max_single_seed_drop": -0.05,
    "confirmation_positive_seeds_min": 3,
    "confirmation_mean_gain_min": 0.05,
    "confirmation_raw_one_hop_min": 0.80,
    "confirmation_relevant_cf_both_correct_min": 0.70,
    "confirmation_irrel_perm_preservation_min": 0.90,
    "bootstrap_replicates": 10_000,
    "retention_a_qualification_raw_min": 0.80,
    "retention_a_qualification_cf_min": 0.70,
}


def _make_job(
    *,
    session: str,
    architecture: str,
    dataset: str,
    objective: str,
    init_seed: int,
    order_seed: int,
    updates: int = X1_TOTAL_UPDATES,
    is_core: bool = True,
) -> dict[str, Any]:
    spec = build_x1_model_spec(architecture)
    job_id = f"X1-{session}-S20-{architecture}-{objective}-s{init_seed}"
    return {
        "job_id": job_id,
        "session": session,
        "architecture": architecture,
        "profile": "S20",
        "init_seed": int(init_seed),
        "order_seed": int(order_seed),
        "dataset": dataset,
        "objective": objective,
        "updates": int(updates),
        "effective_state_batch": 16,
        "is_core": bool(is_core),
        "lr": float(X1_OPTIMIZER_DEFAULTS["peak_lr"]),
        "max_sequence_length": X1_MAX_SEQUENCE,
        "model_spec_sha256": spec.sha256(),
        "base_parameters": int(spec.base_spec.parameter_receipt().total),
        "added_parameters": int(spec.added_parameters()),
        "total_parameters": int(spec.total_parameters()),
        "action_labels": list(ACTION_19_LABELS) if architecture != "B0" else ["utf8-byte-plus-eos"],
        "milestones": list(X1_MILESTONES),
    }


def core_x1_jobs(session: str | None = None) -> tuple[dict[str, Any], ...]:
    """Return the exact 42 registered core S20 jobs matching X1_DESIGN_REGISTRY.json."""

    specs = (
        ("A0", "P", X1_DATASET_SCHEMA, "standard"),
        ("A0", "R", X1_DATASET_SCHEMA, "standard"),
        ("A1", "F", X1_DATASET_SCHEMA, "standard"),
        ("A1", "F-cap", X1_DATASET_SCHEMA, "standard"),
        ("A2", "P-capR", X1_DATASET_SCHEMA, "standard"),
        ("A2", "R-noedge", X1_DATASET_SCHEMA, "standard"),
        ("A3", "B0", X1_DATASET_SCHEMA, "standard"),
        ("A3", "J", X1_DATASET_SCHEMA, "standard"),
        ("D0", "P", X1_D2_DATASET_SCHEMA, "PAIR-CE"),
        ("D0", "R", X1_D2_DATASET_SCHEMA, "PAIR-CE"),
        ("D1", "P", X1_D2_DATASET_SCHEMA, "PAIR-EQ"),
        ("D1", "R", X1_D2_DATASET_SCHEMA, "PAIR-EQ"),
        ("D2", "P", X1_D2_DATASET_SCHEMA, "PAIR-EQ-CF"),
        ("D2", "R", X1_D2_DATASET_SCHEMA, "PAIR-EQ-CF"),
    )
    all_jobs: list[dict[str, Any]] = []
    for sess, arch, ds, obj in specs:
        for i_seed, o_seed in X1_SEEDS:
            all_jobs.append(
                _make_job(
                    session=sess,
                    architecture=arch,
                    dataset=ds,
                    objective=obj,
                    init_seed=i_seed,
                    order_seed=o_seed,
                    is_core=True,
                )
            )
    if session is None:
        return tuple(all_jobs)
    if session not in X1_CORE_SESSIONS:
        raise ValueError(f"unknown core session {session!r}; expected one of {X1_CORE_SESSIONS}")
    return tuple(j for j in all_jobs if j["session"] == session)


def extension_x1_jobs(module: str | None = None) -> tuple[dict[str, Any], ...]:
    """Return non-core optional X1 extension jobs (U-1/U-2/U-4/U2-untied)."""

    u_specs = (
        ("U0", "U-1", X1_DATASET_SCHEMA, "standard"),
        ("U0", "U-2", X1_DATASET_SCHEMA, "standard"),
        ("U1", "U-4", X1_DATASET_SCHEMA, "standard"),
        ("U1", "U2-untied", X1_DATASET_SCHEMA, "standard"),
    )
    jobs_list: list[dict[str, Any]] = []
    for sess, arch, ds, obj in u_specs:
        for i_seed, o_seed in X1_SEEDS:
            jobs_list.append(
                _make_job(
                    session=sess,
                    architecture=arch,
                    dataset=ds,
                    objective=obj,
                    init_seed=i_seed,
                    order_seed=o_seed,
                    is_core=False,
                )
            )
    if module is None:
        return tuple(jobs_list)
    return tuple(j for j in jobs_list if j["session"] == module)


def session_jobs(session: str) -> tuple[dict[str, Any], ...]:
    if session in X1_CORE_SESSIONS:
        return core_x1_jobs(session)
    ext = extension_x1_jobs(session)
    if ext:
        return ext
    raise ValueError(f"unknown X1 session {session!r}")


def replication_jobs(architecture: str, *, objective: str = "standard", dataset: str = X1_DATASET_SCHEMA) -> tuple[dict[str, Any], ...]:
    return tuple(
        _make_job(
            session="CONF",
            architecture=architecture,
            dataset=dataset,
            objective=objective,
            init_seed=i_seed,
            order_seed=o_seed,
            is_core=False,
        )
        for i_seed, o_seed in X1_REPLICATION_SEEDS
    )


def verify_against_design_registry(registry_path: Path | None = None) -> dict[str, Any]:
    """Verify 1-for-1 correspondence with research/v8-x/X1_DESIGN_REGISTRY.json."""

    path = registry_path or (Path(__file__).resolve().parents[3] / "research" / "v8-x" / "X1_DESIGN_REGISTRY.json")
    design = json.loads(path.read_text(encoding="utf-8"))
    reg_jobs = design["jobs"]
    runtime_jobs = core_x1_jobs()
    if len(reg_jobs) != 42 or len(runtime_jobs) != 42:
        raise ValueError(f"expected 42 core jobs, got registry={len(reg_jobs)}, runtime={len(runtime_jobs)}")

    fields_to_match = (
        "job_id",
        "session",
        "architecture",
        "profile",
        "init_seed",
        "order_seed",
        "dataset",
        "objective",
        "updates",
        "effective_state_batch",
    )
    for r_job, rt_job in zip(reg_jobs, runtime_jobs):
        for k in fields_to_match:
            if r_job[k] != rt_job[k]:
                raise ValueError(f"X1 registry mismatch on {rt_job['job_id']}.{k}: {r_job[k]!r} != {rt_job[k]!r}")
    return {
        "verified": True,
        "core_jobs": len(runtime_jobs),
        "sessions": list(X1_CORE_SESSIONS),
        "design_registry_sha256": digest(design),
        "runtime_protocol_sha256": X1_PROTOCOL_SHA,
    }


def x1_session_planthir_design(session: str) -> dict[str, Any]:
    """Return typed Planthir HypothesisSpec, FactorSpec, and ContrastSpec metadata for one X1 session."""

    if session == "A0":
        return {
            "factors": (TypedFactorSpec("architecture", "architecture", "P", ("P", "R")),),
            "typed_contrasts": (
                TypedContrastSpec(
                    "C_X1_R_VS_P",
                    "architecture",
                    "P",
                    "R",
                    "x1_relational_slot_readout",
                    held_constant=("lr", "objective", "precision", "seed_group"),
                    expected_treatment_value="R",
                    primary_endpoint="macro_12_family_accuracy",
                ),
            ),
            "hypotheses": (
                HypothesisSpec(
                    "H06_RELATIONAL_SLOTS",
                    "Query-conditioned relational slot message passing (R) improves composition & changed-edge binding over pointer (P).",
                    "x1_relational_slot_readout",
                    falsifiers=(
                        FalsifierSpec("F_H06_MEAN", "R mean gain >= 5pp over P", "macro_12_gain", ">=", 0.05),
                        FalsifierSpec("F_H06_CF", "R improves relevant-CF both-correct on answer-determinate cases", "r_cf_both_correct_gain", ">=", 0.05),
                    ),
                ),
            ),
        }
    if session == "D1":
        return {
            "factors": (TypedFactorSpec("objective", "objective", "PAIR-CE", ("PAIR-CE", "PAIR-EQ")),),
            "hypotheses": (
                HypothesisSpec(
                    "H07_EQUIVARIANCE",
                    "JS permutation/irrelevance consistency (PAIR-EQ) improves invariance robustness vs PAIR-CE.",
                    "d2_equivariance_js",
                    falsifiers=(FalsifierSpec("F_H07_PERM", "View conjunction gain over PAIR-CE", "view_conjunction_gain", ">=", 0.05),),
                ),
            ),
        }
    return {
        "factors": (),
        "typed_contrasts": (),
        "hypotheses": (
            HypothesisSpec(
                f"H_X1_{session}",
                f"Controlled X1 session {session} scientific measurement.",
                "controlled_measurement",
                falsifiers=(FalsifierSpec(f"F_X1_{session}_FINITE", "All accuracies finite >= 0", "overall_accuracy", ">=", 0.0),),
            ),
        ),
    }


X1_PROTOCOL_SHA = digest(
    {
        "schema": X1_PROTOCOL_SCHEMA,
        "core_jobs": [
            {k: j[k] for k in ("job_id", "session", "architecture", "init_seed", "order_seed", "dataset", "objective", "updates", "model_spec_sha256")}
            for j in core_x1_jobs()
        ],
        "milestones": list(X1_MILESTONES),
        "optimizer": X1_OPTIMIZER_DEFAULTS,
        "d2_loss": X1_D2_LOSS_DEFAULTS,
    }
)


__all__ = [
    "X1_ADVANCEMENT_THRESHOLDS",
    "X1_CORE_SESSIONS",
    "X1_D2_LOSS_DEFAULTS",
    "X1_HYPOTHESIS_REGISTER",
    "X1_INTERMEDIATE_WEIGHT_MILESTONES",
    "X1_MILESTONES",
    "X1_OPTIMIZER_DEFAULTS",
    "X1_PROTOCOL_SCHEMA",
    "X1_PROTOCOL_SHA",
    "X1_REPLICATION_SEEDS",
    "X1_SEEDS",
    "X1_SESSION_CONFIG",
    "X1_SESSION_LABELS",
    "X1_TOTAL_UPDATES",
    "core_x1_jobs",
    "extension_x1_jobs",
    "replication_jobs",
    "session_jobs",
    "verify_against_design_registry",
    "x1_session_planthir_design",
]
