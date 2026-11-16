"""AN-RA V8-Y1 prospective protocol and immutable experiment registry.

Import performs zero device, filesystem, or data-generation work. All seeds,
arm definitions, checkpoint manifests, gates, and phase ceilings are bound by
``REGISTRY_SHA`` before any execution begins.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .codec import canonical, digest
from .representation import UTF8
from .v6_registry import MODEL_PROFILES


SCHEMA = "anra-v8-y1-protocol-v1"
PROTOCOL_VERSION = "V8-Y1"

HITCHHIKER_REVIEWED_BASELINE_COMMIT = "64ed23a433ebb1ce2f620e0db8dcc24f4d8c23be"
HISTORICAL_HH_WEIGHTS_SOURCE = "6859904a02c8083c4a0ab320c9c4a511899f73e4"
HISTORICAL_HH_PROTOCOL_SHA256 = "11be27cf15a268501bb0c552438b2e12c0d02a3756e771160017e9d37043687f"
HISTORICAL_HH_RUN_ID = "hh-stage1-4bfe3a806e3b"
HISTORICAL_HH_FULL_ARCHIVE_BYTES = 1167950007
HISTORICAL_HH_FULL_ARCHIVE_SHA256 = "43a0c074cdcc3e12d8d00e0509663a2d47702586414317818ca0876579e487e2"
HISTORICAL_S20_RESTORATION_STATUS = "NOT_AVAILABLE"

SEEDS = (7, 19, 41)
REGISTERED_SCALES = ("S3", "S20")
PROMPT_TOKEN_BOUND = 512  # Prompt + reserved decision target budget fits S3 native 512 for both S3 and S20.
RESERVED_TARGET_TOKENS = 6

PHASES = ("prepare", "canary", "diagnose", "core", "followup", "finalize")
PHASE_CEILING_MINUTES = {
    "prepare": 30,
    "canary": 30,
    "diagnose": 90,
    "core": 360,
    "followup": 240,
    "finalize": 30,
}

GENERATOR_SEEDS = {
    "train": 606,
    "development": 707,
    "confirmation": 909,
    "capacity_followup": 1109,
    "probe_d2": 1307,
    "positive_t0": 1409,
    "t4_train": 1201,
    "t4_validation": 1202,
}

HOLDOUT_SERIALIZATION_COMBINATION = {
    "hop": 2,
    "terminal_edge_row_1based": 1,
    "distractor_count": 2,
    "queried_start_edge_row_1based": 3,
}

NON_ROUTE_REHEARSAL_COUNTS = {
    "conflict": 395,
    "known": 395,
    "memory": 395,
    "missing": 395,
    "null": 394,
    "unavailable": 394,
}  # Lexically sorted 6 non-route kinds: 4*395 + 2*394 = 2368

ROUTE_COVERAGE_COUNTS_PER_HOP = {
    "imbalanced": {"answerable": 1332, "query_needed_total": 222, "query_needed_families": 111, "unavailable": 222},
    "balanced": {"answerable": 592, "query_needed_total": 592, "query_needed_families": 296, "unavailable": 592},
}  # 1776 decision states/hop * 4 hops = 7104 route states/epoch; total 9472/epoch

TRAIN_HYPERPARAMS = {
    "epochs": 4,
    "examples_per_epoch": 9472,
    "total_examples": 37888,
    "total_updates": 2368,
    "batch_size": 16,
    "micro_batch_size": 2,
    "optimizer": "adamw",
    "precision": "fp32",
    "lr_peak": 1e-4,
    "lr_min_ratio": 0.1,
    "schedule": "cosine",
    "warmup_updates": 0,
    "betas": [0.9, 0.95],
    "eps": 1e-8,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "milestones": [0, 592, 1184, 2368],
    "checkpoint_interval_updates": 64,
    "checkpoint_interval_seconds": 300,
}

T0_SPEC = {
    "primary_seed": 7,
    "fallback_seed": 19,
    "profile": "S3",
    "families_total": 128,
    "families_per_hop": 32,
    "known_per_hop": 16,
    "query_needed_per_hop": 8,
    "unavailable_per_hop": 8,
    "supervised_states_per_hop": 40,
    "supervised_states_total": 160,
    "batch_size": 16,
    "micro_batch_size": 2,
    "updates": 256,
    "exposures": 4096,
    "min_raw_accuracy": 0.95,
}

T1_ARMS = {
    "F00": {"fresh": False, "coverage": "imbalanced", "description": "Static worlds + imbalanced coverage (new controlled baseline)."},
    "F10": {"fresh": True, "coverage": "imbalanced", "description": "Fresh procedural worlds per epoch + imbalanced coverage."},
    "F01": {"fresh": False, "coverage": "balanced", "description": "Static worlds + balanced C-style coverage."},
    "F11": {"fresh": True, "coverage": "balanced", "description": "Fresh procedural worlds per epoch + balanced C-style coverage."},
}

TIE_BREAK_ORDER = ("F01", "F10", "F11")

SPLIT_QUOTAS = {
    "hops": [1, 2, 3, 4],
    "base_families_total": 1024,
    "per_hop": {"answerable": 128, "query_needed": 64, "unavailable": 64},
    "variants_per_family": 8,  # cases 0..7
    "main_cases": 8192,
    "edge_change_panel_families": 256,  # 64/hop; base_ref + fixed_pos + moved_pos = 768 cases
    "edge_change_panel_cases_with_ref": 768,
    "missing_bridge_panel_families": 256,  # 64/hop; available vs unavailable = 512 cases
    "missing_bridge_panel_cases": 512,
    "dual_chain_query_families": 256,  # 64/hop; chain A vs chain B = 512 cases
    "dual_chain_query_cases": 512,
    "serialization_holdout_cases": 64,
    "preservation_kinds": 7,
    "preservation_families_per_kind": 256,
    "preservation_cases": 1792,
    "total_budgeted_cases_per_split_model": 11840,
    "historical_continuity_cases_separate": 80,
}

SCREENING_GATE = {
    "min_mean_primary_gain": 0.10,
    "require_all_seeds_positive": True,
    "min_advantage_over_best_shortcut": 0.10,
    "min_edge_panel_both_correct_per_seed": 0.50,
    "min_dual_query_both_correct_per_seed": 0.50,
    "min_query_needed_raw_terminal_per_seed": 0.50,
    "min_valid_action_rate_per_seed": 0.95,
    "min_unavailable_abstain_rate_per_seed": 0.90,
    "max_original_kind_regression_vs_f00_per_seed": 0.05,
    "bootstrap_resamples": 2000,
    "tie_break_order": list(TIE_BREAK_ORDER),
}

CONFIRMATION_GATE = {
    **SCREENING_GATE,
    "require_family_cluster_ci_lower_gt_zero": True,
}

BOTTLENECK_CRITERIA = {
    "source_arm": "F11",
    "min_answer_only_rank1_base_known_per_seed": 0.70,
    "min_rank1_minus_raw_gap_per_seed": 0.20,
    "min_constrained_edge_both_correct_per_seed": 0.50,
    "min_constrained_dual_query_both_correct_per_seed": 0.50,
}

T3_GROUP_CE_SPEC = {
    "branch": "targeted_group_ce",
    "profile": "S3",
    "parent_arm": "F11",
    "arms": ("F11_cont_ce", "F11_cont_group_ce"),
    "group_ce_weight": 0.5,
    "groups": ["ANSWER", "QUERY", "RETRIEVE", "ABSTAIN", "DELIBERATE", "OTHER"],
    "updates": 592,
    "lr_peak": 1e-4,
    "lr_min_ratio": 0.1,
    "schedule": "cosine",
    "optimizer_policy": "reset",
    "eval_split": "capacity_followup",
    "min_mean_primary_gain": 0.10,
    "require_all_seeds_positive": True,
}

T3_CAPACITY_SPEC = {
    "branch": "capacity_s20",
    "profile": "S20",
    "updates": 2368,
    "eval_split": "capacity_followup",
    "requires_t0_and_all_s3_complete": True,
}

T4_FUTURE_CANDIDATE = {
    "status": "FUTURE_CANDIDATE_NOT_ADMITTED",
    "admitted_in_y1_core_or_followup": False,
    "train_seed": 1201,
    "validation_seed": 1202,
    "decision_states_per_task": 4736,
    "standalone_updates": 1184,
    "sequence_updates": 2368,
    "min_standalone_val_accuracy": 0.90,
    "heldout_families_per_task": 256,
}

HEURISTIC_CONTROLS = (
    "last-row-object",
    "nearest-to-query-object",
    "highest-timestamp-object",
    "global-most-frequent-answer",
    "always-abstain",
    "random-legal",
    "oracle-traversal",
)
SHORTCUT_HEURISTICS = tuple(h for h in HEURISTIC_CONTROLS if h != "oracle-traversal")

_SHARED_HH_IDENTITIES = {
    "curriculum_spec_sha256": "88bb7a83001f785724793392953873fbbcdbf519a286888a649b8edb306402ef",
    "model_spec_sha256": "e492fb8d511d3ba76afb7104d567aaaf6dbb863fdf59cc59c41f74063a364b78",
    "optimizer_spec_sha256": "b564080cd02cb5730927656e8a28e3f3790a16852d93a7d805294a3b4b89d02e",
    "schedule_spec_sha256": "69f042a9f4b6ba3ce65b6be0ce2dac886e76cec28b4c4a28f57e1a041d76c0f0",
    "source_commit": HISTORICAL_HH_WEIGHTS_SOURCE,
    "tokenizer_sha256": "8343d8d2429e28ec91545a321ad4be4553ca87a89c30d86521b9d9e73acce270",
    "schema": "anra-v5-identity-bindings/v1",
}

_HH_JOB_ROWS = (
    ("A", 7, "9559a3e7d64abacdc6bcc185ec913fbd16c96a8c3131b42dda90c5dcdcfa4b0a", "9b7667423ee714dcacd2690bf9411ff436c4d0e1daa433d3afe436143e57d474", "8724c370b40bd74f56c481adffca067de071c00196626c1058243b6cb5fe3778", "2ab6aaba7000dc455b15829f4bdec97bd1d229c0b3b2906689ead6a93f7e7f76", "19bf25af36444eacdc29ea68b3b189983ac84fce49139b894df211f7c9c49ede"),
    ("A", 19, "a6f7861e2846016685bf857873b02f96a2ad201086c7effc1c20d1e89117d5e3", "64fc9078f53f6a5b11dd9ae92c1fbf28e8ade48936a58e2833f73b096f1ec49e", "9f2e59805577c374417f7b80e7706f265f3d63898d104adf765a7347f34c7335", "faae7994822e3c6d655a0e4e82ca6fcc5aca43765953f48a25db12a00cf7ab6c", "19bf25af36444eacdc29ea68b3b189983ac84fce49139b894df211f7c9c49ede"),
    ("A", 41, "0dffe0a81db0a234050c11a50a6cc583f466790821432d349f07bf81b528e02c", "963bd0f9e781aaadc5a03c921ae127dc72b031a0aee2199147323503ade2b804", "9a462cb8545ad8e4e62b1c76966be54b64ad8b3cd435809b3c72aa989ca79507", "152b4b5df5ebc341460193101b38e2c8434566f3f13a2f3cbfc2ee3996e9ba2e", "19bf25af36444eacdc29ea68b3b189983ac84fce49139b894df211f7c9c49ede"),
    ("AB", 7, "7cea8d042d1233f51985bb733b76ba9cff4eec88c2f1098f7a12ecc6fe47bee6", "e3b782b1a92928bce4dca857fafb26b6205268626069dc853cbbee1549600147", "aa31e487eecdf72eb3cd1ea0649da65ef638ff41833f0a8823bd485abaab7c59", "3e21b9588dfcd2e199c7aef69da836e15e388eea0975b12407b329023c2826ae", "19bf25af36444eacdc29ea68b3b189983ac84fce49139b894df211f7c9c49ede"),
    ("AB", 19, "a82ba48bff8dc4b34bd9b14e4fad887721095fef7a89aea62504ff1d19a47533", "96722789f87c667fb9b5560f90278c916690468c03be9e4ac7558c7500bd7d93", "f800e2e7b78a5fb7292b3e60f68795c460391bdf4e1d2e4aaf74b63b1f6839b1", "0e4e62c45d41bf99116c22953f5060a9b2e05038836a5e7be225ab254038fbc4", "19bf25af36444eacdc29ea68b3b189983ac84fce49139b894df211f7c9c49ede"),
    ("AB", 41, "d4d3fc4e4b27929e7d4ab0a925b41a2aa87f5e753eebe87f5db961ea74135fc8", "8eeee2ea60701b60d16e7bf7c44de5163a1639fa3235fbfdd5b8eac4412547d8", "e6dde7b0cd7b1021d46ef9952ddacf2d6f434a286c3e5d6ef44514079048b615", "39b6a70b84d16fa996470bf3eea40594f6b1756fb8d7ebb216fabd42d51b7b91", "19bf25af36444eacdc29ea68b3b189983ac84fce49139b894df211f7c9c49ede"),
    ("B", 7, "fbd8d2441ee3d1f08b7acb0f0ddf446108c1ed00407be01f21b171fdb670d30a", "e69ca350ec3e7de871a2888469bf93a5f7e2bf53e3bec6e4b7ff8404f9112512", "c132c334c0c87eaf86b6bd5b31f3b39b23a23c918f7cb5b1f4f516e6c5d40011", "b941c76bc376269480ca682bbfeedc06d8a157943a86c5c84f77b6f9b37cfc28", "19bf25af36444eacdc29ea68b3b189983ac84fce49139b894df211f7c9c49ede"),
    ("B", 19, "94f8bcc0b959362ba020e980b53947476a5273989ab1acace2ba43a6831db850", "ea281fc83243d2dfa7da5833ab796b667a8ef0893c71abc4249fddab3db82327", "453e65271d9d65191a69e9ec215571a1610e3578e125c88a87d212e2641d1012", "8fa448ccaded2a82c629981c0c23f1d4d4f271b00f28e6da494dea92fb34fcf9", "19bf25af36444eacdc29ea68b3b189983ac84fce49139b894df211f7c9c49ede"),
    ("B", 41, "098965a488d4d2c7ff1d9919f848ad0250dff21f28858b463059a56d09ffaed0", "814d7839527d83e3854483e16b1c740b0e8a1b098a79718055a5452e3cb917a2", "160930faedfb233c80736537f07466d8b45981a70b8e923f30c373ace913d9db", "c49669a48d963384d218dc546d71f622ce3b93ba26ac25a6ba72c9db6b5c25b0", "19bf25af36444eacdc29ea68b3b189983ac84fce49139b894df211f7c9c49ede"),
    ("B0", 7, "9b18bffef8cd320226f4d9576a14db8d1b997e3e29bc3a37e08ad65eae80e081", "0a1c5abecee17c4180c66feb0f7a9fae5fb17504e9deae5d7ed72fc8f8489d78", "2b63a56c10bbd178e6ff0065910e93d23ffe0d17ecc0be42e039be632688ca8a", "59c59e6f12ee8c18c45902c33476a6bacc152a2f863b6500c8785419c506c92b", "3b675595e324e4135fb3ba12dc7df83fe5ae0c1a5645df30c509b735aa0765f5"),
    ("B0", 19, "8d540a4ef6ef240de7f945af712b845068dad1ac9cc97be299632e3c54176df8", "ab6826bcddb753fe601a27cc64b6dde65a9c0a5b0108802f63e53ffd5d92a0fb", "fb3dc27b809364ca96afa8ce63fc2795096d5a8b4f2f22c55b29d39be9eb9ab7", "f0e4e61adf0652ab205066161cc81fb71838ff8f37ae70e78890d6b9220dba41", "d568aa90202999e3a85c82a691e9904054bde98c42119833057c8097a589b4a8"),
    ("B0", 41, "b4936525f8c0e410eb1d7bb07a1779375b2468651424e137776bb5c72fb8a7f1", "f9314b8ba02a7cedce351c869a89c0c289b8296c0af1f4e389e655fdd56af39a", "098528df7054480ebec5577ea0fec49f0aaa120b94155aa68718c068ff0f37f9", "24ee4d813c3799a5a2eb6b6ff0d98044b8bb6944b5968d7ed12ac36a129cce9a", "6e06d877353e927d46aefcc3cd1336731ff949a49efcb41f8f37e1545a14e031"),
    ("C", 7, "4465f6b4c420a2d5bd90a55b893cd0110c94a6e663800ed62262d2b08dd01cd4", "47cfb09501c0e25b4d2401d34b37ff903d1e92a0bea5ab41b348ad0453518424", "c7f2a857994e4503a68191ef1c07fb5711e796dbb685c58183ff401d09b3b4dd", "8e5154db188d1864bdf42a26fb34d9da99fd858bd15ca14381dc001e61d5bf1d", "f18393be30eae2e6f1ab65a888d8e5e8f995a6a53a61517c0e6b0b5af53667f3"),
    ("C", 19, "5107b741a93bd7d00bed2c38161292f9e5b4659cb00519e296f491d3fccc7780", "11330c9aa5aee5d50b201fc0590dc523b6602feab64856f0baaf541832551cd7", "02897dfe9bceccd87ba7cd2931c4ff84fe6ef3be2994f2a1dc6e141de62d0b68", "5556d3b28c5eccb273670e93326c9e582c6b9eb54c03b698c4de518a867eaf9d", "f514a1af689b18b2f2b6290b31a3ab04ef8e5ba789b0f9c6bd2c6cf884d46667"),
    ("C", 41, "6d56b79dc5045d6a660d285121c08bd46c7932606fa73b89ac6aebb24d4fda5e", "0eefbe1f442cbbafe5b5074fd85516786f7c6613c813ea962efd664ca6689fc9", "a882fae226987f2c5f540858ffd2b01a1b034a072d7f6852ac41e5a2968acb88", "4b632b7415d760ad1e896e897f223529cf98200c7066f83ad04185dae2b002ee", "ca7193bcf9da741ef8925f58cdb5aade5b809417769d2f0e0314ba03770028a8"),
)

HISTORICAL_HH_S3_MANIFESTS: dict[str, dict[str, Any]] = {
    f"HH-stage1_s3_screen-S3-{arm}-s{seed}": {
        "job_id": f"HH-stage1_s3_screen-S3-{arm}-s{seed}",
        "arm": arm,
        "seed": seed,
        "profile": "S3",
        "global_update": 2368,
        "generation": 2368,
        "checkpoint_manifest_sha256": ckpt_sha,
        "model_sha256": model_sha,
        "identities": {
            **_SHARED_HH_IDENTITIES,
            "data_manifest_sha256": data_sha,
            "pack_manifest_sha256": data_sha,
            "run_spec_sha256": run_sha,
            "sampler_spec_sha256": sampler_sha,
        },
    }
    for arm, seed, ckpt_sha, model_sha, data_sha, run_sha, sampler_sha in _HH_JOB_ROWS
}


def derive_seed_int(*parts: Any) -> int:
    """Derive a deterministic 32-bit unsigned integer seed via SHA-256 (never hash())."""
    payload = canonical(list(parts))
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _scale_receipts() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "model": MODEL_PROFILES[name].canonical(),
            "parameters": MODEL_PROFILES[name].parameter_receipt().total,
            "model_sha256": MODEL_PROFILES[name].sha256(),
            "effective_y1_prompt_bound": PROMPT_TOKEN_BOUND,
        }
        for name in REGISTERED_SCALES
    }


def resolve_y1_scale(scale: str) -> Any:
    """Resolve strictly registered Y1 scales (S3 or S20 only); refuse 60M/90M/unregistered scales."""
    if scale not in REGISTERED_SCALES:
        raise ValueError(f"unregistered Y1 model scale {scale!r}; allowed: {REGISTERED_SCALES}")
    return MODEL_PROFILES[scale]


def build_core_jobs(*, include_t0: bool = True) -> list[dict[str, Any]]:
    """Build deterministic T0 + seed-balanced interleaved T1 job specs."""
    jobs: list[dict[str, Any]] = []
    if include_t0:
        jobs.append(
            {
                "job_id": "Y1-t0_learnability-S3-T0-s7",
                "stage": "t0_learnability",
                "arm": "T0",
                "profile": "S3",
                "seed": 7,
                "order_seed": 7,
                "updates": T0_SPEC["updates"],
                "batch_size": T0_SPEC["batch_size"],
                "micro_batch_size": T0_SPEC["micro_batch_size"],
                "lr": TRAIN_HYPERPARAMS["lr_peak"],
                "lr_min_ratio": TRAIN_HYPERPARAMS["lr_min_ratio"],
                "objective": "ce",
                "group_ce_weight": 0.0,
                "fresh": False,
                "coverage": "balanced_t0",
                "fallback_seed": 19,
            }
        )
    # Interleave across seeds so interruption never retains only a favorable arm.
    for seed in SEEDS:
        for arm in ("F00", "F10", "F01", "F11"):
            spec = T1_ARMS[arm]
            jobs.append(
                {
                    "job_id": f"Y1-t1_factorial-S3-{arm}-s{seed}",
                    "stage": "t1_factorial",
                    "arm": arm,
                    "profile": "S3",
                    "seed": seed,
                    "order_seed": seed,
                    "updates": TRAIN_HYPERPARAMS["total_updates"],
                    "epochs": TRAIN_HYPERPARAMS["epochs"],
                    "batch_size": TRAIN_HYPERPARAMS["batch_size"],
                    "micro_batch_size": TRAIN_HYPERPARAMS["micro_batch_size"],
                    "lr": TRAIN_HYPERPARAMS["lr_peak"],
                    "lr_min_ratio": TRAIN_HYPERPARAMS["lr_min_ratio"],
                    "objective": "ce",
                    "group_ce_weight": 0.0,
                    "fresh": spec["fresh"],
                    "coverage": spec["coverage"],
                }
            )
    return jobs


def build_followup_jobs(branch: str, *, comparison_arm: str | None = None) -> list[dict[str, Any]]:
    """Build exact admitted T3 followup jobs for either targeted_group_ce or capacity_s20."""
    if branch == "targeted_group_ce":
        jobs = []
        for seed in SEEDS:
            parent_job = f"Y1-t1_factorial-S3-F11-s{seed}"
            for arm, obj, w in (("F11_cont_ce", "ce", 0.0), ("F11_cont_group_ce", "ce_plus_group_ce", 0.5)):
                jobs.append(
                    {
                        "job_id": f"Y1-t3_group_ce-S3-{arm}-s{seed}",
                        "stage": "t3_group_ce",
                        "arm": arm,
                        "profile": "S3",
                        "seed": seed,
                        "order_seed": seed,
                        "parent_job_id": parent_job,
                        "updates": T3_GROUP_CE_SPEC["updates"],
                        "batch_size": TRAIN_HYPERPARAMS["batch_size"],
                        "micro_batch_size": TRAIN_HYPERPARAMS["micro_batch_size"],
                        "lr": T3_GROUP_CE_SPEC["lr_peak"],
                        "lr_min_ratio": T3_GROUP_CE_SPEC["lr_min_ratio"],
                        "objective": obj,
                        "group_ce_weight": w,
                        "fresh": True,
                        "coverage": "balanced",
                        "optimizer_policy": "reset",
                    }
                )
        return jobs
    if branch == "capacity_s20":
        treat = comparison_arm or "F11"
        if treat not in ("F10", "F01", "F11"):
            raise ValueError(f"capacity_s20 comparison_arm must be F10/F01/F11, got {treat!r}")
        jobs = []
        for seed in SEEDS:
            for arm in ("F00", treat):
                spec = T1_ARMS[arm]
                jobs.append(
                    {
                        "job_id": f"Y1-t3_capacity-S20-{arm}-s{seed}",
                        "stage": "t3_capacity",
                        "arm": arm,
                        "profile": "S20",
                        "seed": seed,
                        "order_seed": seed,
                        "updates": T3_CAPACITY_SPEC["updates"],
                        "epochs": TRAIN_HYPERPARAMS["epochs"],
                        "batch_size": TRAIN_HYPERPARAMS["batch_size"],
                        "micro_batch_size": TRAIN_HYPERPARAMS["micro_batch_size"],
                        "lr": TRAIN_HYPERPARAMS["lr_peak"],
                        "lr_min_ratio": TRAIN_HYPERPARAMS["lr_min_ratio"],
                        "objective": "ce",
                        "group_ce_weight": 0.0,
                        "fresh": spec["fresh"],
                        "coverage": spec["coverage"],
                    }
                )
        return jobs
    raise ValueError(f"unsupported T3 followup branch: {branch!r}")


REGISTRY: dict[str, Any] = {
    "schema": SCHEMA,
    "protocol_version": PROTOCOL_VERSION,
    "status": "PROSPECTIVE_FROZEN",
    "claim_boundary": "Implementation and design contract only; execution alone tests empirical hypotheses.",
    "baseline_source": HITCHHIKER_REVIEWED_BASELINE_COMMIT,
    "historical_weights": {
        "source_commit": HISTORICAL_HH_WEIGHTS_SOURCE,
        "protocol_sha256": HISTORICAL_HH_PROTOCOL_SHA256,
        "run_id": HISTORICAL_HH_RUN_ID,
        "archive_bytes": HISTORICAL_HH_FULL_ARCHIVE_BYTES,
        "archive_sha256": HISTORICAL_HH_FULL_ARCHIVE_SHA256,
        "s3_manifests": HISTORICAL_HH_S3_MANIFESTS,
        "historical_s20_low_lr": HISTORICAL_S20_RESTORATION_STATUS,
    },
    "tokenizer": UTF8.manifest.record(),
    "profiles": _scale_receipts(),
    "prompt_token_bound": PROMPT_TOKEN_BOUND,
    "reserved_target_tokens": RESERVED_TARGET_TOKENS,
    "phases": list(PHASES),
    "phase_ceiling_minutes": PHASE_CEILING_MINUTES,
    "seeds": list(SEEDS),
    "generator_seeds": GENERATOR_SEEDS,
    "holdout_serialization_combination": HOLDOUT_SERIALIZATION_COMBINATION,
    "non_route_rehearsal_counts": NON_ROUTE_REHEARSAL_COUNTS,
    "route_coverage_counts_per_hop": ROUTE_COVERAGE_COUNTS_PER_HOP,
    "train_hyperparams": TRAIN_HYPERPARAMS,
    "t0_spec": T0_SPEC,
    "t1_arms": T1_ARMS,
    "split_quotas": SPLIT_QUOTAS,
    "screening_gate": SCREENING_GATE,
    "confirmation_gate": CONFIRMATION_GATE,
    "bottleneck_criteria": BOTTLENECK_CRITERIA,
    "t3_group_ce_spec": T3_GROUP_CE_SPEC,
    "t3_capacity_spec": T3_CAPACITY_SPEC,
    "t4_future_candidate": T4_FUTURE_CANDIDATE,
    "heuristic_controls": list(HEURISTIC_CONTROLS),
    "deferred_mechanisms": [
        "learned_slots",
        "recurrent_deliberation",
        "self_modification",
        "long_horizon_memory",
        "60M_and_90M_scales",
        "external_benchmarks",
    ],
}

REGISTRY_SHA: str = digest(REGISTRY)
PROTOCOL_SHA: str = REGISTRY_SHA


def registry() -> dict[str, Any]:
    """Return detached canonical Y1 registry and fail if global registry mutated after import."""
    if digest(REGISTRY) != REGISTRY_SHA:
        raise ValueError("Y1 registry mutated after identity binding")
    return json.loads(canonical(REGISTRY))


def effective_protocol_digest(overrides: dict[str, Any] | None = None) -> tuple[str, bool]:
    """Return (digest, is_production) — any override mints a non-production protocol digest."""
    if not overrides:
        return REGISTRY_SHA, True
    active = {k: v for k, v in overrides.items() if v is not None and v is not False}
    if not active:
        return REGISTRY_SHA, True
    amended = {**registry(), "non_production_overrides": active}
    return digest(amended), False
