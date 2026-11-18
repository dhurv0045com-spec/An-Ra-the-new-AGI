"""Avendi V8-Y2 prospective protocol, immutable design registry, and Y1 discovery register."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.planthir.contracts import (
    CausalContrastSpec,
    ExperimentRegistration,
    FalsifierSpec,
    HypothesisSpec,
)
from anra_v5.planthir.registry import PlanthirRegistry


SCHEMA = "anra-avendi-v8-y2-protocol-v1"
DESIGNATION = "Avendi V8-Y2"
BACKEND_ID = "avendi_v8_y2"

BASELINE_HITCHHIKER_COMMIT = "641633a060c2ac84215e149de7dc3d0f4987f748"
Y1_EXECUTED_SOURCE_COMMIT = "60e63d8025259ab4e579f28bb32f66b4189dc0ee"
Y1_PROTOCOL_SHA256 = "3809785b27aa4feca1c0abfc8c81e9ed6dd69e48ff3c4bc0093481a647322cac"
Y1_EXECUTED_RUN_ID = "y1-all-9c80885013"
Y1_SUMMARY_ARCHIVE_SHA256 = "9844ad2015c6c3c9b9bf8ffd8864b8bded9d37ec001c89bc73ac4f64c8cad1bf"
Y1_EVIDENCE_ARCHIVE_SHA256 = "4f76ac0d3a1fc67fac2705325c0909f791325696f9a1162c9e86c783c717173f"
HISTORICAL_HH_FULL_ARCHIVE_SHA256 = "43a0c074cdcc3e12d8d00e0509663a2d47702586414317818ca0876579e487e2"

SYMBOLS: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOP")
ORDERED_UNEQUAL_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (g, h) for g in SYMBOLS for h in SYMBOLS if g != h
)  # 16 * 15 = 240 ordered unequal answer pairs

INIT_SEEDS: tuple[int, ...] = (7, 19, 41)
ORDER_SEED_OFFSET = 1000
ORDER_SEEDS: dict[int, int] = {s: ORDER_SEED_OFFSET + s for s in INIT_SEEDS}

GENERATOR_SEEDS: dict[str, int] = {
    "pilot_reference": 1801,
    "train": 2201,
    "dev": 2202,
    "confirm": 2203,
    "transfer": 2204,
    "p4_train": 2205,
    "p5_eval": 2206,
    "p5_train": 2207,
    "p6_blocks": 2208,
    "bootstrap": 2217,
}

PROMPT_TOKEN_BOUND = 512
RESERVED_TARGET_TOKENS = 6

SPLIT_FAMILY_COUNTS: dict[str, dict[str, int]] = {
    "dev": {"literal_cases": 256, "one_hop_families": 256, "three_hop_families": 256},
    "confirm": {"literal_cases": 256, "one_hop_families": 256, "three_hop_families": 256},
    "transfer": {
        "literal_cases": 256,
        "one_hop_families": 256,
        "three_hop_families": 256,
        "two_hop_families": 128,
        "four_hop_families": 128,
    },
    "train_p3": {"one_hop_families": 1024, "decision_states": 4096},
    "train_p4": {"one_hop_families": 512, "two_hop_families": 512, "decision_states": 4096},
}

PHASE_CEILING_MINUTES: dict[str, int] = {
    "fixture": 30,
    "frozen": 120,
    "acquisition": 240,
    "transition": 240,
    "conditional": 240,
    "correction": 240,
    "all": 360,
}

STAGE_STATUS_VALUES: tuple[str, ...] = (
    "QUALIFIED",
    "FAILED",
    "DIAGNOSTIC",
    "INCONCLUSIVE",
    "BLOCKED",
    "ENGINEERING_FAILURE",
    "MEASUREMENT_FAILURE",
    "NOT_EXECUTED",
)

VERIFICATION_ROW_NAMES: tuple[str, ...] = (
    "byte_integrity",
    "oracle_fixture_correctness",
    "recorded_action_replay",
    "checkpoint_restore",
    "regenerated_per_case_agreement",
    "training_reproduction",
)

VERIFICATION_STATUS_VALUES: tuple[str, ...] = (
    "PASS",
    "FAIL",
    "PARTIAL",
    "UNAVAILABLE",
)

QUALIFICATION_GATES: dict[str, Any] = {
    "p2_frozen": {
        "literal_min_accuracy_per_seed": 0.95,
        "one_hop_all_four_min_per_seed": 0.80,
        "three_hop_all_four_min_per_seed": 0.70,
        "tie_breaker_arm": "F00",
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 2217,
        "fp32_same_runtime_logit_max_abs": 1e-5,
    },
    "p3_acquisition": {
        "min_mean_complete_family_gain": 0.10,
        "require_positive_gain_every_seed": True,
        "min_confirm_one_hop_all_four_per_seed": 0.80,
        "max_same_seed_literal_drop": 0.05,
        "simpler_qualified_default": "CE",
    },
    "p4_transition": {
        "min_raw_three_hop_all_four_per_seed": 0.80,
        "min_eligible_both_correct_families_per_seed": 128,
        "max_identity_swap_drop": 0.01,
        "min_donor_conditional_accuracy": 0.80,
        "min_donor_unconditional_accuracy": 0.70,
        "require_positive_excess_vs_permuted_every_seed": True,
        "min_arch_advantage_mean_gain": 0.10,
        "width_search_grid": [48, 56, 64, 72, 80],
        "param_match_tolerance": 0.10,
    },
    "p5_conditional": {
        "min_first_action_accuracy_per_stratum_seed": 0.80,
        "min_terminal_accuracy_per_stratum_seed": 0.70,
        "families": 256,
        "conditions_per_family": 6,
    },
    "p6_correction": {
        "min_held_out_gain_vs_controls": 0.15,
        "require_positive_every_seed": True,
        "max_old_context_loss": 0.05,
        "blocks": 128,
        "unseen_graphs_per_block": 16,
        "retention_cases_per_block": 16,
    },
}

P3_TRAINING_SPEC: dict[str, Any] = {
    "profile": "S3",
    "parameters": int(MODEL_PROFILES["S3"].parameter_receipt().total),
    "model_sha256": MODEL_PROFILES["S3"].sha256(),
    "width": 256,
    "layers": 4,
    "context_length": 512,
    "vocabulary": UTF8.vocabulary_size,
    "init_seeds": list(INIT_SEEDS),
    "order_seeds": [ORDER_SEEDS[s] for s in INIT_SEEDS],
    "corpus_seed": GENERATOR_SEEDS["train"],
    "one_hop_families": 1024,
    "cells_per_family": 4,
    "decision_states_per_epoch": 4096,
    "epochs": 4,
    "batch_size": 16,
    "families_per_batch": 4,
    "micro_batch_size": 4,
    "total_updates": 1024,
    "optimizer": "adamw",
    "precision": "fp32",
    "betas": [0.9, 0.95],
    "eps": 1e-8,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "lr_peak": 1e-4,
    "lr_min": 1e-5,
    "schedule": "cosine",
    "warmup_updates": 0,
    "arms": {
        "CE": {"lambda_pair_margin": 0.0, "margin": 1.0},
        "CE_PAIR_MARGIN": {"lambda_pair_margin": 0.25, "margin": 1.0},
    },
    "monitor_families": 64,
    "monitor_milestones": [0, 64, 256, 512, 1024],
}

P4_ARCHITECTURE_SPEC: dict[str, Any] = {
    "byte_embedding_width": 64,
    "row_encoder": "GRU(64,64)_single_layer_shared_no_row_pos_embedding",
    "key_value_projection_width": 64,
    "state_cardinality": 16,
    "state_embedding_width": 64,
    "attention_scale": 8.0,
    "readout": "shared_64_to_16_fixed_AP_mapping",
    "controls": [
        "symbolic_bottleneck_soft",
        "symbolic_bottleneck_hard",
        "continuous_recurrent",
        "one_step_attention",
        "analytic_public_oracle",
    ],
    "continuous_width_candidates": [48, 56, 64, 72, 80],
    "train_one_hop_families": 512,
    "train_two_hop_families": 512,
    "withheld_depths": [3, 4],
    "total_updates": 1024,
    "batch_size": 16,
    "micro_batch_size": 4,
}

Y1_BASELINE_LINEAGES: dict[str, dict[str, Any]] = {
    "Y1-t1_factorial-S3-F00-s7": {
        "job_id": "Y1-t1_factorial-S3-F00-s7",
        "arm": "F00",
        "seed": 7,
        "profile": "S3",
        "global_update": 2368,
        "cumulative_tokens": 75776,
        "source_commit": Y1_EXECUTED_SOURCE_COMMIT,
        "protocol_sha256": Y1_PROTOCOL_SHA256,
        "checkpoint_sha256": "57ab8b7292a84689ee32d797257eb4a2c9c8cd3c95421df2ca254801c6cd1706",
        "logits_sidecar_sha256": "30df56eb7a7c6f2048eeabdfaf21607d2cc79f0f7bf44716746d5b5d38b8cd0a",
        "logits_sidecar_bytes": 1030588,
    },
    "Y1-t1_factorial-S3-F00-s19": {
        "job_id": "Y1-t1_factorial-S3-F00-s19",
        "arm": "F00",
        "seed": 19,
        "profile": "S3",
        "global_update": 2368,
        "cumulative_tokens": 75776,
        "source_commit": Y1_EXECUTED_SOURCE_COMMIT,
        "protocol_sha256": Y1_PROTOCOL_SHA256,
        "checkpoint_sha256": "b900fa477499e5a7eca2d5ac4ed717e1f1ff34981a3ae2a85e9aae1a8c3da475",
        "logits_sidecar_sha256": "819f170e1cd53f2ef26f3f05fcf348599443f111caa0efb6fc70765144042eed",
        "logits_sidecar_bytes": 1035895,
    },
    "Y1-t1_factorial-S3-F00-s41": {
        "job_id": "Y1-t1_factorial-S3-F00-s41",
        "arm": "F00",
        "seed": 41,
        "profile": "S3",
        "global_update": 2368,
        "cumulative_tokens": 75776,
        "source_commit": Y1_EXECUTED_SOURCE_COMMIT,
        "protocol_sha256": Y1_PROTOCOL_SHA256,
        "checkpoint_sha256": "4e8891f6f3d8ef1975d949bc7c96393edda9c4dc304b12cf5a5368e5fdffdf95",
        "logits_sidecar_sha256": "962b07c17538ed59ede97f267fb128d00c271c6e2f64294787b6f600b4fc17a4",
        "logits_sidecar_bytes": 1038143,
    },
    "Y1-t1_factorial-S3-F11-s7": {
        "job_id": "Y1-t1_factorial-S3-F11-s7",
        "arm": "F11",
        "seed": 7,
        "profile": "S3",
        "global_update": 2368,
        "cumulative_tokens": 75776,
        "source_commit": Y1_EXECUTED_SOURCE_COMMIT,
        "protocol_sha256": Y1_PROTOCOL_SHA256,
        "checkpoint_sha256": "65c8506d7359d91c76fd5ac3628f0226c68f76a2edf760f1e62d309e9fa0313f",
        "logits_sidecar_sha256": "50c17bc251e249954571126c9fc0dacfefb7700e6b75ba5c14578efba93e7baa",
        "logits_sidecar_bytes": 993189,
    },
    "Y1-t1_factorial-S3-F11-s19": {
        "job_id": "Y1-t1_factorial-S3-F11-s19",
        "arm": "F11",
        "seed": 19,
        "profile": "S3",
        "global_update": 2368,
        "cumulative_tokens": 75776,
        "source_commit": Y1_EXECUTED_SOURCE_COMMIT,
        "protocol_sha256": Y1_PROTOCOL_SHA256,
        "checkpoint_sha256": "96fa4f25b62601b8533d602ad5e1ba6587760bc3cb8bb758a0230a2a7bf686ef",
        "logits_sidecar_sha256": "00c6be40758403f2b42463ba041d92dfbd130e39d4cdb417dabcc787200eb72d",
        "logits_sidecar_bytes": 1025120,
    },
    "Y1-t1_factorial-S3-F11-s41": {
        "job_id": "Y1-t1_factorial-S3-F11-s41",
        "arm": "F11",
        "seed": 41,
        "profile": "S3",
        "global_update": 2368,
        "cumulative_tokens": 75776,
        "source_commit": Y1_EXECUTED_SOURCE_COMMIT,
        "protocol_sha256": Y1_PROTOCOL_SHA256,
        "checkpoint_sha256": "f2ce4dfc965559ee3b2e189a80ce3e8e84786c4e23d1de3a78249b544961babf",
        "logits_sidecar_sha256": "8ef25db3d536052f727ffd640121f6b0c008373d3230fe35e71ae121424b655e",
        "logits_sidecar_bytes": 996887,
    },
}

EVIDENCE_KEYS: dict[str, str] = {
    "Y": "Experiment Result/V8-Y1/y1-all-9c80885013/RESULTS.md",
    "M": "Experiment Result/V8-Y1/y1-all-9c80885013/METRICS.json",
    "B": "Experiment Result/V8-Y1/y1-all-9c80885013/BEHAVIOR.json",
    "H": "Experiment Result/Hitchhiker/hh-stage1-4bfe3a806e3b/RESULTS.md",
    "P": "research/hitchhiker/BRANCH_AND_PALANTIRI_AUDIT.md",
}

PERMANENT_Y1_DISCOVERY_REGISTER: tuple[dict[str, Any], ...] = (
    {"id": "01", "source_keys": ["B"], "observation": "Timestamp rank solves 3,072/3,072 primary presentations.", "live_explanation_or_rival": "Cue exists. Neural use remains open.", "discriminating_test": "Permute timestamps within the identical graph. Measure prediction changes and disagreement with the old rule.", "stage_target": "P0/P2"},
    {"id": "02", "source_keys": ["B"], "observation": "Returned single event contains the terminal value in 512/512 audited cases per split.", "live_explanation_or_rival": "Terminal copying and integration agree.", "discriminating_test": "Hide an interior bridge and move its returned row. Score both terminal outcomes when the bridge is switched.", "stage_target": "P5"},
    {"id": "03", "source_keys": ["B"], "observation": "Development changed targets A:441, B:71.", "live_explanation_or_rival": "Alphabet-order selector favors A.", "discriminating_test": "Balance directed endpoint changes; compare old versus corrected distribution on the same weights.", "stage_target": "P0/P2"},
    {"id": "04", "source_keys": ["Y"], "observation": "Target-bearing frame was compared with bare held-out prompt.", "live_explanation_or_rival": "Overlap assurance failed; actual overlaps unknown.", "discriminating_test": "Compare prefix bytes at the target boundary, full decision states and parent families. Plant a target-only difference and demand rejection.", "stage_target": "P0"},
    {"id": "05", "source_keys": ["M", "B"], "observation": "Successful delivery coexisted with a perfect shortcut.", "live_explanation_or_rival": "A correct runtime can measure an ambiguous task.", "discriminating_test": "Oracle passes and answer-copy/query-blind controls fail family qualification on actual rendered inputs.", "stage_target": "P0"},
    {"id": "06", "source_keys": ["B"], "observation": "F11 produced identical raw outputs on 767/768 changed-query pairs and solved zero pairs.", "live_explanation_or_rival": "Query is lost, weak at readout, or dominated by action choice.", "discriminating_test": "Compare tokenized prompts, signed endpoint odds, raw answers and complete pairs.", "stage_target": "P2/P3"},
    {"id": "07", "source_keys": ["M"], "observation": "F11 mean primary approximately 17.19%, 2.47%, 0.78%, 0.65% at hops 1-4.", "live_explanation_or_rival": "Failure is already visible on one hop; depth and difficulty covary.", "discriminating_test": "Test literal copy, one-hop two-key selection and equal-difficulty paired routes separately.", "stage_target": "P2/P3"},
    {"id": "08", "source_keys": ["B"], "observation": "Gold rank one within A-P only 21-24% for F11.", "live_explanation_or_rival": "Abstention contributes but value discrimination also fails.", "discriminating_test": "Measure raw answer mass, within-answer odds, and exact answer/EOS on identical prompts.", "stage_target": "P2"},
    {"id": "09", "source_keys": ["M"], "observation": "Candidate recipes failed matched gates.", "live_explanation_or_rival": "Data, optimization, decoding and architecture remain alternatives.", "discriminating_test": "A single controlled rescue on new families narrows attribution; failure qualifies only the tested recipe.", "stage_target": "P3"},
    {"id": "10", "source_keys": ["M"], "observation": "F00/F10/F01/F11 mean 31.67/17.90/6.05/5.27%, respectively.", "live_explanation_or_rival": "Static baseline best on this fixture.", "discriminating_test": "Keep F00 as frozen anchor; retest against shortcut-resistant family endpoint.", "stage_target": "P2"},
    {"id": "11", "source_keys": ["Y"], "observation": "Answerable allocation fell from 1,332 to 592 states per hop/epoch.", "live_explanation_or_rival": "Balanced coverage trades away direct acquisition exposure.", "discriminating_test": "Hold direct examples fixed, replace neutral rehearsal with availability examples; compare with a separate direct-dose replacement. Log opportunity costs.", "stage_target": "P5"},
    {"id": "12", "source_keys": ["M", "B"], "observation": "F11 unavailable success ~97.80%, query terminal success ~96.82%, false abstain ~88.02% on answerable base.", "live_explanation_or_rival": "Behavior responds to action priors and/or actual relevance.", "discriminating_test": "Hide an irrelevant edge with the same tool flag as a required missing edge. Score decision and downstream answer.", "stage_target": "P5"},
    {"id": "13", "source_keys": ["Y"], "observation": "F10 lost at every seed; corpus tags differed even in epoch zero.", "live_explanation_or_rival": "Refresh competes with acquisition; initial draw also differs.", "discriminating_test": "Clone the initial corpus/order and checkpoint, then repeat versus refresh exactly matched strata.", "stage_target": "P3"},
    {"id": "14", "source_keys": ["Y"], "observation": "Old C on Y1 primary 9.34%, F01 6.05%, F11 5.27%.", "live_explanation_or_rival": "Multiple training and rendering differences.", "discriminating_test": "Preserve separate historical identities; only a matched intervention can isolate original C.", "stage_target": "P1/P2"},
    {"id": "15", "source_keys": ["Y"], "observation": "S3->S20 follow-up F00 30.40->17.06%, F11 4.69->2.28%.", "live_explanation_or_rival": "Same schedule, different compute or optimizer suitability.", "discriminating_test": "Reopen scale after acquisition; give both scales identical preregistered LR search budgets and common evaluation.", "stage_target": "DEFERRED_SCALE"},
    {"id": "16", "source_keys": ["Y"], "observation": "Final behavior alone admits non-acquisition, interference or shortcut explanations.", "live_explanation_or_rival": "Skills may form and later disappear.", "discriminating_test": "Fixed train and unseen diagnostic families at logged milestones; report onset, loss and post-change time.", "stage_target": "P3"},
    {"id": "17", "source_keys": ["Y"], "observation": "T0 receipt reports 127/128 on familiar families; raw T0 evaluation absent.", "live_explanation_or_rival": "Familiar fit may precede or diverge from transfer.", "discriminating_test": "Export raw records for train memorization, new instances and withheld grammar separately.", "stage_target": "P3"},
    {"id": "18", "source_keys": ["M"], "observation": "F00 primary ranged 22.46-41.89%; multiple trained seeds share a corpus draw.", "live_explanation_or_rival": "Init, order and corpus can each matter.", "discriminating_test": "Block matched intervention by seed and corpus; distinguish family CIs from training-replication uncertainty.", "stage_target": "P2/P3"},
    {"id": "19", "source_keys": ["Y"], "observation": "T1 failed; T2 skipped; T3 ran for failure diagnosis; D2/T4 unestablished.", "live_explanation_or_rival": "A negative screen still permits informative cheap comparisons.", "discriminating_test": "Follow the failure branch and export a row/status for every registered stage.", "stage_target": "P2/P3/P4/P5/P6"},
    {"id": "20", "source_keys": ["Y"], "observation": "Evidence and summary support recorded replay; Y1 full weights were absent from those archives.", "live_explanation_or_rival": "Original state may exist in saved Kaggle outputs.", "discriminating_test": "Match exact final hashes and regenerate behavior from real parents; record missing lineage explicitly.", "stage_target": "P1"},
    {"id": "21", "source_keys": ["M"], "observation": "464,880 episodes replayed, 42,880 updates checked, 39,936 first-logit argmax matches.", "live_explanation_or_rival": "These are distinct verification claims.", "discriminating_test": "Issue six receipts with independent statuses; corrupt one case while preserving aggregate accuracy.", "stage_target": "P0/P1/P2"},
    {"id": "22", "source_keys": ["P"], "observation": "Historical Palantiri audit found four gaps; branch 641633a0 contains later repairs.", "live_explanation_or_rival": "New adapter could still weaken contracts.", "discriminating_test": "Test all four on Avendi payloads plus scientific shortcut negative controls.", "stage_target": "P0"},
    {"id": "23", "source_keys": ["M"], "observation": "Some original-kind scores regressed, including F10 seed 7 memory -85.55 pp.", "live_explanation_or_rival": "Interference and differential acquisition both possible.", "discriminating_test": "Measure the same qualified A checkpoint before/after learning B against equal-work A->A, and independently verify B.", "stage_target": "P5/P6"},
    {"id": "24", "source_keys": ["Y", "H", "P"], "observation": "V6 queries often succeeded while terminal answers failed; V7 blocked changed-payload 73/96; V7.17 LR mattered; H copied the last object on 768 presentations.", "live_explanation_or_rival": "Narrow skills depend on distribution and task structure.", "discriminating_test": "Reuse historical public payload and row-order controls; qualify each skill in its own setting before composing it.", "stage_target": "P2/P3/P4"},
    {"id": "25", "source_keys": ["Y"], "observation": "Y1 provides no measured self-improvement loop.", "live_explanation_or_rival": "Explicit correction, strategy storage, self-diagnosis and weight learning can yield different effects.", "discriminating_test": "Randomize feedback and update route, then test transfer to new entities and retention on previously acquired skills.", "stage_target": "P6"},
)


def derive_seed_int(*parts: Any) -> int:
    """Derive a deterministic 32-bit unsigned integer seed via SHA-256."""
    payload = canonical(list(parts))
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


FROZEN_DESIGN_SPEC: dict[str, Any] = {
    "schema": SCHEMA,
    "designation": DESIGNATION,
    "backend_id": BACKEND_ID,
    "baseline_hitchhiker_commit": BASELINE_HITCHHIKER_COMMIT,
    "y1_executed_source_commit": Y1_EXECUTED_SOURCE_COMMIT,
    "y1_protocol_sha256": Y1_PROTOCOL_SHA256,
    "y1_executed_run_id": Y1_EXECUTED_RUN_ID,
    "y1_summary_archive_sha256": Y1_SUMMARY_ARCHIVE_SHA256,
    "y1_evidence_archive_sha256": Y1_EVIDENCE_ARCHIVE_SHA256,
    "historical_hh_full_archive_sha256": HISTORICAL_HH_FULL_ARCHIVE_SHA256,
    "tokenizer": UTF8.manifest.record(),
    "prompt_token_bound": PROMPT_TOKEN_BOUND,
    "reserved_target_tokens": RESERVED_TARGET_TOKENS,
    "symbols": list(SYMBOLS),
    "ordered_unequal_pair_count": len(ORDERED_UNEQUAL_PAIRS),
    "init_seeds": list(INIT_SEEDS),
    "order_seed_offset": ORDER_SEED_OFFSET,
    "order_seeds": {str(k): v for k, v in ORDER_SEEDS.items()},
    "generator_seeds": GENERATOR_SEEDS,
    "split_family_counts": SPLIT_FAMILY_COUNTS,
    "phase_ceiling_minutes": PHASE_CEILING_MINUTES,
    "qualification_gates": QUALIFICATION_GATES,
    "p3_training_spec": P3_TRAINING_SPEC,
    "p4_architecture_spec": P4_ARCHITECTURE_SPEC,
    "y1_baseline_lineages": Y1_BASELINE_LINEAGES,
    "evidence_keys": EVIDENCE_KEYS,
    "scenario_register": [dict(row) for row in PERMANENT_Y1_DISCOVERY_REGISTER],
    "verification_rows": list(VERIFICATION_ROW_NAMES),
    "stage_status_values": list(STAGE_STATUS_VALUES),
}

DESIGN_SHA: str = digest(FROZEN_DESIGN_SPEC)
REGISTRY_SHA: str = DESIGN_SHA


def frozen_design() -> dict[str, Any]:
    """Return detached copy of the frozen design spec and verify immutability."""
    if digest(FROZEN_DESIGN_SPEC) != DESIGN_SHA:
        raise ValueError("Avendi V8-Y2 frozen design mutated after import")
    return json.loads(canonical(FROZEN_DESIGN_SPEC))


def build_planthir_registration(*, source_commit: str = BASELINE_HITCHHIKER_COMMIT) -> ExperimentRegistration:
    """Construct the Planthir ExperimentRegistration for Avendi V8-Y2."""
    return ExperimentRegistration(
        experiment_id="AVENDI-V8-Y2",
        program="AVENDI",
        architecture_family="Y1-S3-BYTE-AND-P4-STATE",
        parameter_count=int(MODEL_PROFILES["S3"].parameter_receipt().total),
        dataset_identity=DESIGN_SHA,
        source_commit=source_commit,
        hypotheses=(
            HypothesisSpec(
                hypothesis_id="H-P2-FROZEN-CROSSOVER",
                statement="Frozen Y1 S3 F00/F11 models discriminate 1-hop and 3-hop crossed queries/bridges without timestamp or last-row shortcuts.",
                mechanism_target="query_conditioned_value_selection_and_composition",
                falsifiers=(
                    FalsifierSpec(
                        falsifier_id="F-P2-ONE-HOP",
                        description="One-hop complete 4-cell family accuracy across all 3 seeds >= 80%.",
                        metric_name="one_hop_all_four",
                        comparator=">=",
                        threshold=0.80,
                    ),
                    FalsifierSpec(
                        falsifier_id="F-P2-THREE-HOP",
                        description="Three-hop complete 4-cell family accuracy across all 3 seeds >= 70%.",
                        metric_name="three_hop_all_four",
                        comparator=">=",
                        threshold=0.70,
                    ),
                ),
            ),
            HypothesisSpec(
                hypothesis_id="H-P3-PAIR-MARGIN",
                statement="Paired counterfactual endpoint margin loss on 4-cell one-hop families improves complete-family discrimination over matched CE.",
                mechanism_target="paired_counterfactual_endpoint_ranking",
                falsifiers=(
                    FalsifierSpec(
                        falsifier_id="F-P3-GAIN",
                        description="Mean complete-family gain >= 10 pp and positive in every seed with >= 80% confirm one-hop all-four.",
                        metric_name="mean_complete_family_gain",
                        comparator=">=",
                        threshold=0.10,
                    ),
                ),
            ),
            HypothesisSpec(
                hypothesis_id="H-P4-STATE-INTERCHANGE",
                statement="Exchanging the learned hop-1 16-way intermediate state between queries redirects suffix computation to the donor endpoint.",
                mechanism_target="portable_intermediate_entity_state",
                falsifiers=(
                    FalsifierSpec(
                        falsifier_id="F-P4-DONOR",
                        description="Donor-specific outcome accuracy >= 80% conditional and >= 70% unconditional on both original and edited terminal suffixes.",
                        metric_name="donor_unconditional_accuracy",
                        comparator=">=",
                        threshold=0.70,
                    ),
                ),
            ),
        ),
        contrasts=(
            CausalContrastSpec(
                contrast_id="C-P2-F11-VS-F00",
                factor_name="y1_recipe",
                control_arm="F00",
                treatment_arm="F11",
                held_constant=("architecture", "tokenizer", "eval_fixture", "seeds"),
                expected_mechanism_signature="query_and_bridge_sensitivity",
            ),
            CausalContrastSpec(
                contrast_id="C-P3-PAIR-MARGIN-VS-CE",
                factor_name="objective",
                control_arm="CE",
                treatment_arm="CE_PAIR_MARGIN",
                held_constant=("architecture", "init_weights", "corpus_seed_2201", "batch_order", "lr_schedule"),
                expected_mechanism_signature="all_four_signed_margin_flip",
            ),
            CausalContrastSpec(
                contrast_id="C-P4-SYMBOLIC-VS-CONTINUOUS",
                factor_name="architecture",
                control_arm="continuous_recurrent",
                treatment_arm="symbolic_bottleneck_soft",
                held_constant=("row_encoder", "train_families", "optimizer", "updates"),
                expected_mechanism_signature="donor_state_portability_across_terminal_edits",
            ),
        ),
        budget={
            "fixture_minutes": PHASE_CEILING_MINUTES["fixture"],
            "frozen_minutes": PHASE_CEILING_MINUTES["frozen"],
            "acquisition_minutes": PHASE_CEILING_MINUTES["acquisition"],
            "transition_minutes": PHASE_CEILING_MINUTES["transition"],
        },
        dependencies=("V8-Y1",),
        status="REGISTERED",
    )


def build_registry_document(
    *,
    source_commit: str = BASELINE_HITCHHIKER_COMMIT,
    stage_statuses: Mapping[str, str] | None = None,
    scenario_updates: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build the full REGISTRY.json document where design_sha256 is invariant to lifecycle events."""
    reg = PlanthirRegistry()
    exp_reg = build_planthir_registration(source_commit=source_commit)
    reg_sha = reg.register(exp_reg)

    default_stages = {
        "P0_instrument": "NOT_EXECUTED",
        "P1_restoration": "NOT_EXECUTED",
        "P2_frozen_diagnosis": "NOT_EXECUTED",
        "P3_controlled_acquisition": "NOT_EXECUTED",
        "P4_transition_interchange": "NOT_EXECUTED",
        "P5_conditional_action": "NOT_EXECUTED",
        "P6_bounded_correction": "NOT_EXECUTED",
    }
    if stage_statuses:
        for k, v in stage_statuses.items():
            if v not in STAGE_STATUS_VALUES:
                raise ValueError(f"invalid stage status {v!r} for {k!r}")
            default_stages[str(k)] = str(v)

    scenarios_with_events = []
    for row in PERMANENT_Y1_DISCOVERY_REGISTER:
        sid = row["id"]
        updates = list(scenario_updates.get(sid, ())) if scenario_updates else []
        scenarios_with_events.append({
            **row,
            "updates": updates,
        })

    lifecycle = {
        "planthir_registration_sha256": reg_sha,
        "planthir_entry": reg._entries["AVENDI-V8-Y2"],
        "stage_statuses": default_stages,
        "scenario_lifecycle": scenarios_with_events,
    }
    return {
        "schema": "anra-avendi-v8-y2-registry-file-v1",
        "design_sha256": DESIGN_SHA,
        "design": frozen_design(),
        "lifecycle_sha256": digest(lifecycle),
        "lifecycle": lifecycle,
    }


def update_registry_lifecycle(
    doc: Mapping[str, Any],
    *,
    experiment_status: str | None = None,
    result_ref: str | None = None,
    stage_statuses: Mapping[str, str] | None = None,
    scenario_updates: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Update lifecycle/status events without mutating design or design_sha256."""
    if doc.get("design_sha256") != DESIGN_SHA or digest(doc.get("design")) != DESIGN_SHA:
        raise ValueError("registry design_sha256 integrity check failed before lifecycle update")
    out = deepcopy(dict(doc))
    lc = out["lifecycle"]
    if experiment_status is not None:
        entry = lc["planthir_entry"]
        entry["status"] = str(experiment_status)
        if result_ref is not None:
            entry["result_ref"] = str(result_ref)
        events = entry.setdefault("status_events", [])
        events.append({
            "status": str(experiment_status),
            "result_ref": result_ref if result_ref is not None else entry.get("result_ref"),
            "sequence": len(events),
        })
        entry["status_events_sha256"] = digest(events)
    if stage_statuses:
        for k, v in stage_statuses.items():
            if v not in STAGE_STATUS_VALUES:
                raise ValueError(f"invalid stage status {v!r} for {k!r}")
            lc["stage_statuses"][str(k)] = str(v)
    if scenario_updates:
        for srow in lc["scenario_lifecycle"]:
            sid = srow["id"]
            if sid in scenario_updates:
                srow["updates"] = [*srow.get("updates", []), *[dict(u) for u in scenario_updates[sid]]]
    out["lifecycle_sha256"] = digest(lc)
    if out["design_sha256"] != DESIGN_SHA or digest(out["design"]) != DESIGN_SHA:
        raise ValueError("design_sha256 drifted during lifecycle update")
    return out


def save_registry_document(doc: Mapping[str, Any], path: Path | str) -> str:
    """Save and verify registry JSON preserving design_sha256."""
    if doc.get("design_sha256") != DESIGN_SHA or digest(doc.get("design")) != DESIGN_SHA:
        raise ValueError("refusing to save registry with invalid design_sha256")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(doc, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    p.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def load_registry_document(path: Path | str) -> dict[str, Any]:
    """Load and validate registry JSON ensuring design_sha256 and lifecycle_sha256 match."""
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("design_sha256") != DESIGN_SHA or digest(doc.get("design")) != DESIGN_SHA:
        raise ValueError("loaded registry design_sha256 mismatch")
    if doc.get("lifecycle_sha256") != digest(doc.get("lifecycle")):
        raise ValueError("loaded registry lifecycle_sha256 mismatch")
    return doc
