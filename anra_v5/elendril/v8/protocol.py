"""Prospective ELENDRIL V8 run registry, scenario ledger, and advancement rules.

Importing this module performs no CUDA, dataset, or training work.
"""

from __future__ import annotations

from anra_v5.elendril.codec import digest
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from v5_training import optimizer as v5_opt

from .contracts import ACTION_19_LABELS, POLICY_MODES, build_v8_model_spec

SEEDS = ((7, 7), (19, 19), (41, 41))
REPLICATION_SEEDS = ((101, 101), (211, 211), (307, 307))
PHASES = (
    "archive-diagnostic",
    "data-control",
    "head-screen",
    "reader-screen",
    "replication",
)
SCENARIO_STATUSES = (
    "NOT_IMPLEMENTED",
    "IMPLEMENTED_NOT_RUN",
    "PREREQUISITE_NOT_QUALIFIED",
    "EXECUTED",
)

OPTIMIZER_DEFAULTS = {
    "schema": v5_opt.OPTIMIZER_SCHEMA,
    "optimizer": v5_opt.OPTIMIZER_NAME,
    "beta1": v5_opt.BETA1,
    "beta2": v5_opt.BETA2,
    "epsilon": v5_opt.EPSILON,
    "weight_decay": v5_opt.WEIGHT_DECAY,
    "peak_lr": 1e-4,
    "schedule": "cosine",
    "cosine_terminal_ratio": 0.1,
    "grad_clip": 1.0,
    "precision": "fp32",
    "batch_size": 16,
    "micro_batch_size": 2,
    "epochs": 4,
}

ADVANCEMENT_RULES = {
    "schema": "elendril-v8-advancement-rules-v1",
    "development_candidate": {
        "raw_one_hop_exact_episode_accuracy_min": 0.80,
        "paired_improvement_over_comparator_min": 0.10,
        "positive_seed_differences_min": 2,
        "seed_count": 3,
        "single_seed_regression_floor": -0.05,
        "canonical_raw_regression_max": 0.05,
        "relevant_counterfactual_both_correct_min": 0.70,
        "irrelevant_counterfactual_stability_min": 0.90,
    },
    "confirmation_candidate": {
        "positive_seed_differences_min": 3,
        "seed_count": 3,
        "mean_gain_min": 0.10,
        "raw_one_hop_exact_episode_accuracy_min": 0.80,
        "relevant_counterfactual_both_correct_min": 0.70,
        "irrelevant_counterfactual_stability_min": 0.90,
    },
    "retention": {
        "a_b_qualification_raw_min": 0.80,
        "a_b_qualification_cf_both_correct_min": 0.70,
        "old_a_improvement_vs_no_replay_min": 0.10,
        "new_b_within_no_replay_max_drop": 0.05,
        "replay_quota_per_16_b": 4,
        "guardian_monitor_interval_updates": 64,
        "guardian_monitor_worlds": 64,
        "guardian_ema_alpha": 0.5,
        "guardian_floor": 0.05,
    },
    "transfer": {
        "c_qualification_min": 0.80,
        "acquisition_example_or_work_reduction_min": 0.10,
    },
}

EVALUATION_ALLOCATION = {
    "primary_final_cohorts": ("R09", "R10", "E13", "E14", "E15", "A19"),
    "development_base_worlds_per_cohort": 128,
    "linked_forms_per_world": ("canonical", "row_shuffle", "fresh_alias", "relevant_cf", "irrelevant_cf"),
    "development_episodes_per_cohort_final": 640,
    "intermediate_milestone_base_worlds": 32,
    "intermediate_milestone_episodes_per_cohort": 160,
    "confirmation_base_worlds_per_cohort": 256,
    "confirmation_episodes_per_cohort": 1280,
    "policy_modes": POLICY_MODES,
    "max_actions_per_episode": 4,
    "milestones": ("initial", "epoch-1", "epoch-2", "epoch-4"),
    "recovery_checkpoint_interval_updates": 64,
    "annotations_on_primary_worlds": ("F01", "F02", "F04", "G05", "G06", "G07"),
    "diagnostic_extensions": ("F03", "A18", "A20"),
    "historical_regression_separate": True,
}

SCENARIO_LEDGER = (
    {"id": "F01", "stage": "formation", "status": "IMPLEMENTED_NOT_RUN", "description": "Public entity aliases with fixed answer-code legend"},
    {"id": "F02", "stage": "formation", "status": "IMPLEMENTED_NOT_RUN", "description": "Multiple entities, changed relevant attribute, row/distractor controls"},
    {"id": "F03", "stage": "diagnostic-extension", "status": "IMPLEMENTED_NOT_RUN", "description": "Reverse relation question with unique inverse vs ambiguous inverse; no training on held-out inverse labels"},
    {"id": "F04", "stage": "formation", "status": "IMPLEMENTED_NOT_RUN", "description": "Longer aliases and different notation within 512-token bound"},
    {"id": "G05", "stage": "formation-robustness", "status": "IMPLEMENTED_NOT_RUN", "description": "Novel input aliases and consistent symbol permutations"},
    {"id": "G06", "stage": "formation-robustness", "status": "IMPLEMENTED_NOT_RUN", "description": "Row shuffle and independent timestamps"},
    {"id": "G07", "stage": "formation", "status": "IMPLEMENTED_NOT_RUN", "description": "Matched-length distractors, same-source revision, cross-source conflict"},
    {"id": "G08", "stage": "confirmation", "status": "PREREQUISITE_NOT_QUALIFIED", "description": "Separate generator/renderer, explicit author provenance"},
    {"id": "R09", "stage": "primary-endpoint", "status": "IMPLEMENTED_NOT_RUN", "description": "Direct one-hop, raw/J/F/P comparison"},
    {"id": "R10", "stage": "composition-prerequisite", "status": "IMPLEMENTED_NOT_RUN", "description": "Two-hop routes, counterfactual edge replacement"},
    {"id": "R11", "stage": "conditional-composition", "status": "PREREQUISITE_NOT_QUALIFIED", "description": "Held-out three-hop, unrelated/sham clamp controls"},
    {"id": "R12", "stage": "conditional-transfer", "status": "PREREQUISITE_NOT_QUALIFIED", "description": "Source revision plus route task C"},
    {"id": "E13", "stage": "formation", "status": "IMPLEMENTED_NOT_RUN", "description": "Fully sufficient facts with tools available/unavailable independently"},
    {"id": "E14", "stage": "formation", "status": "IMPLEMENTED_NOT_RUN", "description": "Missing fact, legitimate query, changed returned payload"},
    {"id": "E15", "stage": "formation", "status": "IMPLEMENTED_NOT_RUN", "description": "Conflicting sources and subsequent authoritative observation"},
    {"id": "E16", "stage": "conditional-memory", "status": "PREREQUISITE_NOT_QUALIFIED", "description": "Stale memory followed by public correction"},
    {"id": "A17", "stage": "diagnostic", "status": "IMPLEMENTED_NOT_RUN", "description": "Raw vs grammar vs public-availability-only vs direct-answer-only modes"},
    {"id": "A18", "stage": "diagnostic-extension", "status": "IMPLEMENTED_NOT_RUN", "description": "Available queries at fixed-policy repricing costs 0/0.1/0.5/1"},
    {"id": "A19", "stage": "formation", "status": "IMPLEMENTED_NOT_RUN", "description": "Null/unavailable vs obtainable unknown"},
    {"id": "A20", "stage": "diagnostic-extension", "status": "IMPLEMENTED_NOT_RUN", "description": "Wrong proposed answer followed by public corrective evidence"},
    {"id": "C21", "stage": "conditional-retention", "status": "PREREQUISITE_NOT_QUALIFIED", "description": "Qualified A->B, no replay"},
    {"id": "C22", "stage": "conditional-retention", "status": "PREREQUISITE_NOT_QUALIFIED", "description": "Same B with fixed random/stratified sparse replay"},
    {"id": "C23", "stage": "conditional-retention", "status": "PREREQUISITE_NOT_QUALIFIED", "description": "Same replay quota, adaptive allocation and charged monitoring"},
    {"id": "C24", "stage": "conditional-transfer", "status": "PREREQUISITE_NOT_QUALIFIED", "description": "A->B->C vs unrelated-history/fresh controls"},
)

_S20_SPEC = MODEL_PROFILES["S20"]
_SPECS = {
    arm: build_v8_model_spec(arm, base_spec=_S20_SPEC)
    for arm in ("B0", "J", "F", "P", "F-cap")
}


def _job(
    *,
    phase: str,
    arm: str,
    dataset: str,
    sampling: str,
    init_seed: int,
    order_seed: int,
    epochs: int = 4,
    lr: float = 1e-4,
    tag: str | None = None,
) -> dict[str, object]:
    spec = _SPECS[arm]
    label = tag or f"{arm}-{dataset}"
    return {
        "job_id": f"V8-{label}-i{init_seed}-o{order_seed}",
        "phase": phase,
        "arm": arm,
        "dataset": dataset,
        "sampling": sampling,
        "profile": "S20",
        "init_seed": init_seed,
        "order_seed": order_seed,
        "epochs": epochs,
        "batch_size": OPTIMIZER_DEFAULTS["batch_size"],
        "micro_batch_size": OPTIMIZER_DEFAULTS["micro_batch_size"],
        "lr": lr,
        "schedule": OPTIMIZER_DEFAULTS["schedule"],
        "cosine_terminal_ratio": OPTIMIZER_DEFAULTS["cosine_terminal_ratio"],
        "grad_clip": OPTIMIZER_DEFAULTS["grad_clip"],
        "precision": OPTIMIZER_DEFAULTS["precision"],
        "max_sequence_length": spec.max_sequence_length,
        "architecture": spec.architecture,
        "policy_interface": spec.policy_interface,
        "model_spec_sha256": spec.sha256(),
        "base_parameters": spec.base_spec.parameter_receipt().total,
        "added_parameters": spec.added_parameters(),
        "total_parameters": spec.total_parameters(),
        "extra_head_seed": spec.extra_head_seed(init_seed) if arm != "B0" else None,
        "action_labels": list(ACTION_19_LABELS) if arm != "B0" else ["utf8-byte-plus-eos"],
    }


def jobs(
    phase: str,
    *,
    winner_arm: str | None = None,
    comparator_arm: str | None = None,
) -> tuple[dict[str, object], ...]:
    """Return the exact ordered immutable job matrix for a declared V8 phase."""

    if phase not in PHASES:
        raise ValueError(f"unknown V8 phase: {phase!r}")
    if phase == "archive-diagnostic":
        return tuple(
            _job(
                phase=phase,
                arm="B0",
                dataset="D0",
                sampling="blocked",
                init_seed=init_seed,
                order_seed=order_seed,
                epochs=0,
                tag="parent-v717-lowlr",
            )
            for init_seed, order_seed in SEEDS
        )
    if phase == "data-control":
        return tuple(
            [
                _job(phase=phase, arm="B0", dataset="D0", sampling="blocked", init_seed=i, order_seed=o)
                for i, o in SEEDS
            ]
            + [
                _job(phase=phase, arm="B0", dataset="D1", sampling="uniform", init_seed=i, order_seed=o)
                for i, o in SEEDS
            ]
        )
    if phase == "head-screen":
        return tuple(
            [
                _job(phase=phase, arm="J", dataset="D1", sampling="uniform", init_seed=i, order_seed=o)
                for i, o in SEEDS
            ]
            + [
                _job(phase=phase, arm="F", dataset="D1", sampling="uniform", init_seed=i, order_seed=o)
                for i, o in SEEDS
            ]
        )
    if phase == "reader-screen":
        return tuple(
            [
                _job(phase=phase, arm="P", dataset="D1", sampling="uniform", init_seed=i, order_seed=o)
                for i, o in SEEDS
            ]
            + [
                _job(phase=phase, arm="F-cap", dataset="D1", sampling="uniform", init_seed=i, order_seed=o)
                for i, o in SEEDS
            ]
        )
    # replication phase
    win = winner_arm or "F"
    comp = comparator_arm or "J"
    if win not in _SPECS or comp not in _SPECS or win == comp:
        raise ValueError("replication requires two distinct registered V8 arms")
    return tuple(
        [
            _job(phase=phase, arm=win, dataset="D1", sampling="uniform", init_seed=i, order_seed=o, tag=f"rep-win-{win}")
            for i, o in REPLICATION_SEEDS
        ]
        + [
            _job(phase=phase, arm=comp, dataset="D1", sampling="uniform", init_seed=i, order_seed=o, tag=f"rep-comp-{comp}")
            for i, o in REPLICATION_SEEDS
        ]
    )


def validate_job(
    job: dict[str, object],
    phase: str,
    *,
    winner_arm: str | None = None,
    comparator_arm: str | None = None,
) -> dict[str, object]:
    expected = jobs(phase, winner_arm=winner_arm, comparator_arm=comparator_arm)
    if job not in expected:
        raise ValueError(f"job {job.get('job_id')!r} is outside registered V8 phase {phase!r}")
    return job


def protocol() -> dict[str, object]:
    """Return the complete canonical V8 protocol manifest."""

    return {
        "schema": "elendril-v8-protocol-v1",
        "backbone": "S20",
        "base_spec": _S20_SPEC.canonical(),
        "base_spec_sha256": _S20_SPEC.sha256(),
        "architectures": {arm: spec.canonical() for arm, spec in _SPECS.items()},
        "optimizer": OPTIMIZER_DEFAULTS,
        "phases": {
            phase: list(jobs(phase))
            for phase in ("archive-diagnostic", "data-control", "head-screen", "reader-screen")
        },
        "replication_template": {
            "seeds": [list(pair) for pair in REPLICATION_SEEDS],
            "requires_preregistered_winner_and_comparator": True,
            "requires_owner_confirmation_seed": True,
        },
        "evaluation_allocation": EVALUATION_ALLOCATION,
        "advancement_rules": ADVANCEMENT_RULES,
        "scenario_ledger": list(SCENARIO_LEDGER),
        "scientific_status": "PROSPECTIVE_NOT_RUN",
        "parent_weights_status": "PARENT_WEIGHTS_UNAVAILABLE",
    }


PROTOCOL_SHA = digest(protocol())

__all__ = [
    "ADVANCEMENT_RULES",
    "EVALUATION_ALLOCATION",
    "OPTIMIZER_DEFAULTS",
    "PHASES",
    "PROTOCOL_SHA",
    "REPLICATION_SEEDS",
    "SCENARIO_LEDGER",
    "SCENARIO_STATUSES",
    "SEEDS",
    "jobs",
    "protocol",
    "validate_job",
]
