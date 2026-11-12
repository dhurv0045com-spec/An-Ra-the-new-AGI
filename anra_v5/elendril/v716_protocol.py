"""V7.16 causal falsification and mechanism suite registration.

Six falsifiable mechanism arms (H1, H2+H2B, H3+H3B, H4, H5, H6) are
preregistered against the blocked cross-entropy baseline of the Elendril
lineage. Importing this module runs no training and emits no files.

Claim ceiling: CONTROLLED_DEVELOPMENT_MECHANISM_ONLY. Every authorization
flag stays false. All numbers in ``ACCOUNTING`` are copied from recorded
owner-run evidence (V7.1 core 20c80f66..., V7.15 S8 run 3981e34a...) and are
separated deliberately so no metric can be silently conflated with another.
"""
from .codec import digest
from .v71_protocol import SEEDS, jobs as v71_jobs, PROTOCOL_SHA as V71_PROTOCOL_SHA

VERSION = "V7.16"
SCHEMA = "elendril-v716-protocol-v1"

# Custodial pins locked at branch creation (opposing-star forked from Elendril).
PARENT_BRANCH = "Elendril"
PARENT_BRANCH_HEAD = "29b1da646bd06c7f2deeb5e5f3c9f08fe24cabbb"
ESSAY_AUDIT_PIN = "9092acb1"
V71_CORE_SOURCE_SHA = "6104ffd7134cb4cf4b3f340bd060d24a1e2dd200"
PARENT_FULL_SHA256 = "1826c85b5a858d2a600661f7e5916f0fa812c55a8d53c37013c695df28a15d19"
S8_SOURCE_PIN = "e5779922a0072a070f64356f38ea47090e543d6a"
S8_RUN_ID = "3981e34a2ffe4b4b9528b4b4d588883b"
S20_SOURCE_PIN = "4cc1e993c09d1475a3bc1e250fda58909d539407"

CLAIM_CEILING = "CONTROLLED_DEVELOPMENT_MECHANISM_ONLY"
AUTHORIZATION_FLAGS = {
    "agi_claim": False,
    "broad_reasoning_claim": False,
    "pre500m": False,
    "training_500m": False,
    "production_tokenizer_change": False,
}

# Recorded parameter inventories; tests re-derive them from MODEL_PROFILES.
PARAMETER_COUNTS = {"S3": 3_084_800, "S8": 7_970_944, "S20": 19_871_616}

# Blocked cross-entropy baseline exposure. Every training arm except explicit
# exposure-control comparators must reproduce these totals exactly.
EXPOSURE = {
    "epochs": 4,
    "optimizer_updates": 2_368,
    "supervised_tokens": 75_776,
    "unpadded_input_positions": 8_914_404,
}

INTERMEDIATE_EXPOSURE = {
    "epochs": 4,
    "optimizer_updates": 3_068,
    "supervised_tokens": 109_280,
}

# Preregistered reference outcomes from the recorded runs (never re-counted
# as new evidence; used only as comparison baselines).
BASELINE = {
    "v71_s3_blocked_ce_confirmation": "270/336",
    "v71_s3_uniform_ce_confirmation": "207/336",
    "v71_s3_blocked_intermediate_confirmation": "269/336",
    "v715_s8_blocked_ce_confirmation": "258/336",
    "v715_s8_uniform_ce_confirmation": "165/336",
    "stale_memory_correction_blocked_ce": "28/48",
    "stale_memory_correction_intermediate": "2/48",
    "row_reversal_blocked_s3": "270/336 -> 139/336",
    "row_reversal_blocked_s8": "251/336 -> 137/336",
    "invalid_actions_raw_diagnostic_records": "861 of 6720",
    "two_hop_anti_shortcut_v7": "10/48",
}

ARMS = (
    "H1-COADAPT-MASK",
    "H2-ARK014-INVARIANCE",
    "H2B-NORM",
    "H3-CONTRAST-MEM",
    "H3B-CONTRASTIVE",
    "H4-OUTCOME-DELIB",
    "H5-POINTER-READ",
    "H6-ARK017-CONTINUAL",
)
CONTINUAL_ARMS = ("b-only", "replay", "fresh-b", "ark017-sparse")
PHASES = ("mechanism", "continual", "promotion")

ARM_MECHANISMS = {
    "H1-COADAPT-MASK": "coadapt-mask",
    "H2-ARK014-INVARIANCE": "ark014-invariance",
    "H2B-NORM": "canonical-normalizer",
    "H3-CONTRAST-MEM": "counterfactual-memory",
    "H3B-CONTRASTIVE": "order-contrastive",
    "H4-OUTCOME-DELIB": "outcome-deliberation",
    "H5-POINTER-READ": "pointer-two-read",
}
MECHANISM_ARMS = {value: key for key, value in ARM_MECHANISMS.items()}

# Evaluation family tree. Every trained checkpoint is evaluated on all of it.
E_FAMILY = {
    "E01": "112-world confirmation accuracy per seed and aggregate (X/112, X/336); seed inversions never hidden by averages",
    "E02": "raw invalid action count across the 6,720 diagnostic records versus grammar-constrained count",
    "E03": "stale-memory correction score X/48 across the six D3 memory conditions",
    "E04": "row-reversal invariance delta on canonical versus reversed public rows plus consistency rate on identical valid answers",
    "E05": "anti-shortcut multi-hop accuracy (D2) at 1/2/3/4 hops under scrambled timestamps and replaced distractors",
    "E06": "heuristic baseline comparison: copy-last-visible, copy-max-timestamp, always-query, query-blind, random",
    "E07": "net episode utility U(tau) versus the fixed controller reference 0.9714 (V6 P07)",
    "E08": "query precision/recall, needless queries, successful answer updates and cost-adjusted utility",
    "E09": "hidden-state linear probes: proxy mutual information I(H_L; resolved value) versus I(H_L; action legality)",
    "E10": "trace diff auditor: first divergent token between paired conditions (for example t=361/t=365/t=366 rows)",
    "E11": "balanced payload pair both-correct rate on counterfactual evidence swaps",
    "E12": "calibration (Brier), answer entropy and per-epoch training curves with exact exposure accounting",
}

# Preregistered promotion and falsification thresholds. A bad score is a
# research result: failures are reported, never erased.
DECISION_RULES = {
    "primary_metric": E_FAMILY["E01"],
    "promotion_requires": {
        "aggregate_confirmation": ">= 279/336 across seeds 7/19/41 (three points above the 270/336 blocked-CE mean)",
        "per_seed_floor": ">= 79/112 for every seed (no seed below 0.70)",
        "stale_memory_no_regression": ">= 28/48 on D3 stale-corrected condition",
        "row_reversal_consistency": ">= 0.5 identical valid answers across reversed row-order pairs (E04)",
        "invalid_actions": "raw <= 861 across 6,720 records; grammar == 0",
    },
    "falsified_when": {
        "aggregate_confirmation": "< 270/336 blocked-CE mean at matched exposure",
        "stale_memory_regression": "< 28/48 stale-corrected (the intermediate-supervision failure signature)",
        "row_reversal_collapse": "reversed accuracy < 0.5x canonical accuracy with consistency < 0.25",
    },
    "promotion_path": "winning S3 mechanism is rerun at S8 (7,970,944) with identical exposure locks; S20 only after S8 promotion",
    "seed_rule": "seeds 7/19/41 are predeclared; no seed is added, dropped or re-run after seeing scores",
    "cause_status": "CAUSE_UNRESOLVED unless a controlled replicated intervention isolates the mechanism",
}


def jobs(phase):
    """Predeclared job matrix; membership is exact and outcome-independent."""
    if phase == "mechanism":
        return [{"job_id": f"{arm}-s{seed}", "arm": arm, "profile": "S3", "seed": seed,
                 "epochs": EXPOSURE["epochs"], "mechanism": ARM_MECHANISMS[arm],
                 "exposure": "blocked-ce-baseline"}
                for seed in SEEDS for arm in ARM_MECHANISMS]
    if phase == "continual":
        return [{"job_id": f"S3-{stage}-ce-s{seed}", "arm": "H6-ARK017-CONTINUAL",
                 "profile": "S3", "seed": seed, "epochs": 2, "mechanism": "continual",
                 "stage": stage, "exposure": "stage-transition"}
                for seed in SEEDS for stage in CONTINUAL_ARMS]
    if phase == "promotion":
        return [{"job_id": f"S8-{arm}-s{seed}", "arm": arm, "profile": "S8", "seed": seed,
                 "epochs": EXPOSURE["epochs"], "mechanism": ARM_MECHANISMS[arm],
                 "exposure": "blocked-ce-baseline"}
                for seed in SEEDS for arm in ARM_MECHANISMS]
    raise ValueError("unknown V7.16 phase")


def validate(job, phase):
    if job not in jobs(phase):
        raise ValueError("job outside preregistered V7.16 phase")
    return True


def protocol():
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "parent_branch": PARENT_BRANCH,
        "parent_branch_head": PARENT_BRANCH_HEAD,
        "essay_audit_pin": ESSAY_AUDIT_PIN,
        "v71_core_source_sha": V71_CORE_SOURCE_SHA,
        "v71_protocol_sha256": V71_PROTOCOL_SHA,
        "parent_full_sha256": PARENT_FULL_SHA256,
        "v715_s8_source_pin": S8_SOURCE_PIN,
        "v715_s8_run_id": S8_RUN_ID,
        "v715_s20_source_pin": S20_SOURCE_PIN,
        "claim_ceiling": CLAIM_CEILING,
        "authorization_flags": AUTHORIZATION_FLAGS,
        "seeds": list(SEEDS),
        "phases": {phase: jobs(phase) for phase in PHASES},
        "parameter_counts": PARAMETER_COUNTS,
        "exposure_lock": EXPOSURE,
        "exposure_semantics": "optimizer updates and supervised tokens are locked exactly to the blocked-CE baseline per arm; unpadded input positions are recorded per epoch and stay identical for H1/H2A/H2B/H4/H5, while H3 counterfactual evidence may change rendered prompt length by construction",
        "intermediate_reference_exposure": INTERMEDIATE_EXPOSURE,
        "recorded_baseline": BASELINE,
        "arm_definitions": {
            "H1-COADAPT-MASK": "training-time controller logit masking from public state only; masked logits set to -1e9 before cross entropy",
            "H2-ARK014-INVARIANCE": "per-epoch deterministic surface curriculum: canonical, reversed rows, uniform timestamp offset, symbol permutation",
            "H2B-NORM": "controller canonicalization layer sorting observation rows by (entity, tick ascending) before rendering; un-augmented data",
            "H3-CONTRAST-MEM": "25% shape-matched replacement with stale-overwrite / consistent-control / unresolved-contradiction pairs",
            "H3B-CONTRASTIVE": "cosine-margin contrastive penalty between decision-token hidden states of reversed and chronological renderings",
            "H4-OUTCOME-DELIB": "outcome-verified bounded rollouts (K in 1/2/4) scored by U(tau); updates only on traces above the RLOO baseline",
            "H5-POINTER-READ": "minimal pointer-copy gate plus two-read relational pooling; parameter ceiling 3,090,000 at S3",
            "H6-ARK017-CONTINUAL": "b-only / replay / fresh-b / ark017-sparse replay with strictly fixed update count and logged A exposure",
        },
        "evaluation_family": E_FAMILY,
        "decision_rules": DECISION_RULES,
        "diagnostic_registration": {
            "conditions": 64,
            "case_records": 1_120,
            "synthetic_worlds": 288,
            "decoders": ["raw-neural", "engineered-legal-grammar"],
            "new_optimizer_updates_during_diagnostics": 0,
        },
        "confirmation_definition": "112 worlds x 3 seeds = 336 observations; development-procedural, not externally sealed",
        "costs": "U(tau) = I(correct and publicly knowable) - 0.1*verified_queries - deliberate_cost*deliberations - illegal_cost*illegal",
        "cost_parameters": {"query_cost": 0.1, "deliberate_cost": 0.0, "illegal_cost": 0.05,
                            "contrastive_weight": 0.1, "contrastive_margin": 0.5,
                            "replay_fraction": 0.1},
        "custody": "PREREGISTRATION.json committed before any run; EXECUTION_BINDING.json bound at execution start",
        "completion": "status per phase limited to COMPLETED / NOT_RUN / BLOCKED / NOT_QUALIFIED / ENGINEERING_ERROR",
        "scientific_status": "NOT_RUN: engineering preparation only; no GPU or research training has executed",
    }


PROTOCOL_SHA = digest(protocol())


def preregistration():
    """Content committed verbatim as research/v7.16/PREREGISTRATION.json."""
    registered = protocol()
    return {
        "schema": "elendril-v716-preregistration-v1",
        "protocol_sha256": PROTOCOL_SHA,
        "registered": registered,
        "primary_metrics_locked": [
            "E01 confirmation accuracy (aggregate and per-seed)",
            "E03 stale-memory correction X/48",
            "E04 row-reversal accuracy delta and consistency rate",
            "E05 anti-shortcut hop accuracy 1/2/3/4",
            "E02 raw and grammar invalid action counts",
        ],
        "falsification_thresholds": DECISION_RULES,
        "seed_list": list(SEEDS),
        "claim_ceiling": CLAIM_CEILING,
        "authorization_flags": AUTHORIZATION_FLAGS,
        "no_silent_mutation": [
            "notebooks/ELENDRIL_V715_KAGGLE_T4X2.ipynb",
            "notebooks/ELENDRIL_V715_S20_KAGGLE_T4X2.ipynb",
        ],
        "safe_cloud_defaults": {"RUN_STUDY": False, "requires_explicit_owner_flip": True},
        "accounting_separation": {
            "S3": {"parameters": 3_084_800, "optimizer_updates": 2_368, "supervised_tokens": 75_776},
            "S8": {"parameters": 7_970_944, "optimizer_updates": 2_368, "supervised_tokens": 75_776},
            "S20": {"parameters": 19_871_616, "optimizer_updates": 2_368, "supervised_tokens": 75_776},
            "intermediate_supervision": {"optimizer_updates": 3_068, "supervised_tokens": 109_280},
            "unpadded_input_positions_per_run": 8_914_404,
            "diagnostic_records": {"conditions": 64, "records": 1_120, "worlds": 288},
            "confirmation_observations": {"worlds": 112, "seeds": 3, "total": 336},
        },
    }


def binding_template():
    """Content committed as research/v7.16/EXECUTION_BINDING.json.

    The campaign binds these fields at execution start; an unbound template
    can never be mistaken for a run record.
    """
    return {
        "schema": "elendril-v716-execution-binding-v1",
        "status": "UNBOUND",
        "protocol_sha256": PROTOCOL_SHA,
        "preregistration_sha256": digest(preregistration()),
        "science_commit": None,
        "execution_commit": None,
        "public_surface_sha256": None,
        "data_split_sha256": None,
        "matched_seed_initialization": {str(seed): None for seed in SEEDS},
        "custody": {
            "parent_full_sha256": PARENT_FULL_SHA256,
            "parent_full_verified": None,
            "s8_analysis_sha256": "08cca3edcec070b85fac752faf89f156b85fb7b7fc2fbd9b02c908461175c02b",
            "s8_analysis_verified": None,
        },
        "binding_rule": "write-once at run start by the V7.16 campaign; refusing to overwrite; missing parent archive yields NO_GO without any training update",
    }
