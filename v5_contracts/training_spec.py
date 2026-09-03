"""Decision-complete V5-A implementation candidate and freeze gates.

The constants in this module have no implicit defaults.  The candidate is
frozen for engineering, while the major run remains fail-closed until its
external identities and experimental gates are supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .model_spec import ModelSpec, QK_NORM_EPSILON, V5A_250M
from .run_spec import V5A_RUN_CENTER


SCHEMA = "anra-v5-training-spec/v1.0"


def _source_sha256() -> str:
    normalized = Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode()).hexdigest()


def build_training_spec() -> dict[str, Any]:
    model = V5A_250M
    run = V5A_RUN_CENTER
    run_receipt = run.receipt(model)
    residual_std = 0.02 / math.sqrt(2 * model.layers)
    cognition_fractions = {
        "identity_copy": 0.08,
        "query_binding": 0.16,
        "semantic_state": 0.16,
        "interference_retrieval": 0.10,
        "relational_composition": 0.20,
        "counterfactual_sensitivity": 0.10,
        "heldout_rule_induction": 0.10,
        "missing_information": 0.05,
        "faithful_realization": 0.05,
    }
    cognition_tokens = {
        name: round(750_000_000 * fraction)
        for name, fraction in cognition_fractions.items()
    }
    return {
        "schema": SCHEMA,
        "status": "IMPLEMENTATION_FROZEN_INPUT_AND_EVIDENCE_BLOCKED",
        "main_training_authorized": False,
        "implementation_sha256": _source_sha256(),
        "design_rule": "conventional dense Core; cognition pressure comes from verified data and evaluation",
        "core": {
            **model.canonical(),
            "parameter_count": model.parameter_receipt().total,
            "activation": "swiglu:silu(gate)*up",
            "attention_semantics": "full causal every layer; grouped KV ratio 2:1",
            "attention_kernel": "target-selected only after semantic parity and throughput canary",
            "norm_layout": "pre-rmsnorm attention and ffn; final-rmsnorm",
            "qk_norm_epsilon": QK_NORM_EPSILON,
            "qk_norm_axes": "per head over head_dimension",
            "rope": {
                "layout": "pairwise even-odd over full head dimension",
                "positions": [0, 4095],
                "phase_table_dtype": "float32",
                "scaling": "none",
                "extrapolation_authorized": False,
            },
            "embedding_scale": 1.0,
            "logit_scale": 1.0,
            "logit_cap": None,
            "initialization": {
                "normal_std": 0.02,
                "normal_tensors": ["embedding", "q", "k", "v", "ffn_gate", "ffn_up"],
                "residual_output_std": residual_std,
                "residual_output_tensors": ["attention_output", "ffn_down"],
                "rmsnorm_scale": 1.0,
                "qk_norm_scale": 1.0,
                "biases": "absent",
            },
            "precision": {
                "persistent_parameters": "float32",
                "compute": "bfloat16 autocast",
                "logits_loss_reductions": "float32",
                "grad_norm": "float32 replica-global",
                "optimizer_moments": "float32",
                "persistent_bfloat16_shadow": False,
                "loss_scaler": None,
            },
        },
        "scale_ladder": {
            "p35_recipe": {"layers": 16, "width": 384, "query_heads": 6, "kv_heads": 3, "ffn_width": 1024, "parameters": 35_411_328},
            "m102_recipe": {"layers": 20, "width": 640, "query_heads": 10, "kv_heads": 5, "ffn_width": 1600, "parameters": 101_790_080},
            "v5a": {"layers": 26, "width": 896, "query_heads": 14, "kv_heads": 7, "ffn_width": 2368, "parameters": 250_216_960},
            "ratio_invariant": "2 query heads per KV head",
        },
        "tokenizer": {
            "algorithm": "byte-level BPE with byte fallback",
            "vocabulary_size": 24_576,
            "normalization": "none",
            "pretokenizer": "byte-level; no prefix-space insertion",
            "dropout": 0.0,
            "special_tokens": {"pad": ["<pad>", 0], "unk": ["<unk>", 1], "bos": ["<bos>", 2], "eos": ["<eos>", 3]},
            "unexpected_unknowns_allowed": 0,
            "task_or_difficulty_tokens_allowed": 0,
            "artifact_sha256": None,
            "training_corpus_manifest_sha256": None,
            "classification": "PROVISIONAL_CANDIDATE_E1_IDENTITY_REQUIRED",
        },
        "data": {
            "total_real_nonpad_tokens": 5_000_000_000,
            "mixture_tokens": {"natural": 3_250_000_000, "code_math_formal": 1_000_000_000, "verified_cognition": 750_000_000},
            "mixture_fractions": {"natural": 0.65, "code_math_formal": 0.20, "verified_cognition": 0.15},
            "llm_paraphrase_target_tokens": 0,
            "llm_paraphrase_hard_cap_fraction_total": 0.05,
            "deduplication": "exact plus near-duplicate cluster assignment before split",
            "split_unit": "source-disjoint deduplication cluster",
            "training_manifest_sha256": None,
            "pack_manifest_sha256": None,
            "source_ledger_sha256": None,
        },
        "packing": {
            "sequence_buckets": {"512": 0.25, "1024": 0.25, "2048": 0.30, "4096": 0.20},
            "twenty_microstep_supercycle": [512, 1024, 2048, 4096, 2048, 512, 1024, 2048, 4096, 512, 1024, 2048, 4096, 2048, 512, 1024, 2048, 4096, 512, 1024],
            "supercycle_state_checkpointed": True,
            "share_unit": "real non-padding input tokens",
            "document_format": "BOS content EOS",
            "cross_segment_attention": False,
            "position_ids": "reset to zero per segment",
            "loss_mask": "BOS and PAD excluded; content and EOS included",
            "long_document_policy": "deterministic nonoverlapping chunks; each chunk has BOS/EOS",
            "padding_counted_as_training_tokens": False,
        },
        "cognition": {
            "family_fractions_within_cognition": cognition_fractions,
            "family_tokens_at_15_percent": cognition_tokens,
            "surface_axis": "natural/semi-natural is crossed within every family, never a family label",
            "minimum_natural_or_seminatural_fraction_per_family": 0.25,
            "generator_truth": "mechanically executable; hidden graph and answer never serialized",
            "family_presence": "every family is present throughout training",
            "difficulty_distribution": {"easy": 0.34, "medium": 0.355, "hard": 0.305},
            "difficulty_grids": {
                "binding_cardinality": [2, 4, 8, 16],
                "distractors": [0, 2, 4, 8, 16, 32],
                "state_variables": [1, 2, 4],
                "state_updates": [2, 4, 8],
                "state_queries": ["latest", "intermediate", "rollback", "precedence"],
                "composition_hops": [1, 2, 3],
                "rule_demonstrations": [2, 4, 8],
                "context_position_quartiles": [1, 2, 3, 4],
            },
            "curriculum": "uniform family and difficulty interleaving; staged ordering is an E4 challenger only",
            "sensitivity_groups": ["query_swap", "relevant_fact", "state_intervention"],
            "invariance_groups": ["irrelevant_fact", "serialization_permutation", "surface_paraphrase"],
        },
        "objective": {
            "launch_objective": "causal_cross_entropy_only",
            "reduction": "sum over eligible target positions divided by replica-global eligible-token count",
            "label_smoothing": 0.0,
            "z_loss": 0.0,
            "query_swap_lambda": 0.0,
            "trace_loss_lambda": 0.0,
            "optional_e3_query_swap": {
                "lambdas": [0.0, 0.05, 0.15],
                "margin": 0.0,
                "candidates": 4,
                "counterfactual_queries": 2,
                "candidate_constraint": "identical suffix-token count; byte length and first-token role crossed",
                "candidate_positions": "all rotations balanced",
                "score": "summed suffix log-probability",
                "compute_matching": "measured training FLOPs, not tokens alone",
                "main_run_admission": "new spec version only after fresh raw-Core and natural-transfer pass",
            },
        },
        "optimization": {
            "optimizer": "AdamW",
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-8,
            "weight_decay": 0.1,
            "decay_group": "embedding, attention projections, and FFN matrices (non-normalization parameters)",
            "no_decay_group": "all normalization parameters by semantics: RMSNorm scales and affine QK scales (query_scale/key_scale, shape [heads, head_dim]) never decay regardless of rank",
            "peak_learning_rate": 3e-4,
            "schedule": {
                "index": "pre-update cumulative real non-padding tokens from zero",
                "warmup": {"start": 0, "end": 50_000_000, "start_lr": 0.0, "end_lr": 3e-4},
                "stable": {"start": 50_000_000, "end": 4_500_000_000, "lr": 3e-4},
                "decay": {"start": 4_500_000_000, "end": 5_000_000_000, "start_lr": 3e-4, "end_lr": 3e-5, "shape": "linear"},
                "rewarm_on_resume_or_pack_change": False,
            },
            "global_tokens_per_update": 131_072,
            "full_updates": 38_146,
            "final_partial_update_tokens": 127_488,
            "final_partial_update_plan": {
                "microstep_buckets": [2048, 512, 1024, 2048],
                "global_real_tokens": [32_768, 32_768, 32_768, 29_184],
                "per_replica_real_tokens": [4_096, 4_096, 4_096, 3_648],
                "last_microstep_padding_per_replica": 448,
                "padding_excluded_from_loss_and_token_ledger": True,
            },
            "total_updates": 38_147,
            "gradient_clip_global_l2": 1.0,
            "nonfinite_policy": "abort update and run; do not advance token, cursor, optimizer, or schedule state",
            "zero_grad": "set_to_none",
        },
        "target_topology": {
            "accelerator": "TPU/XLA",
            "replicas": 8,
            "data_parallel": True,
            "global_real_tokens_per_microstep": 32_768,
            "gradient_accumulation_microsteps": 4,
            "real_tokens_per_replica_microstep": 4_096,
            "sequences_per_replica_by_bucket": {"512": 8, "1024": 4, "2048": 2, "4096": 1},
            "gradient_collective": "replica sum with exact global eligible-token denominator at accumulation boundary",
            "activation_checkpointing": "every transformer block",
            "gate": "target preflight, OOM, semantic parity, throughput, and resume canaries must pass",
        },
        "checkpointing": {
            "recovery_threshold_tokens": 10_000_000,
            "recovery_generations_retained": 2,
            "milestone_threshold_tokens": 100_000_000,
            "final_500m_milestone_threshold_tokens": 50_000_000,
            "threshold_semantics": "first completed optimizer update reaching or crossing boundary; record scheduled and actual tokens",
            "milestones_immutable": True,
            "pointer_advance_requires": "upload, redownload hash equality, and clean restore",
            "full_resume_components": ["fp32 parameters", "fp32 Adam moments", "optimizer step", "schedule", "token ledger", "sampler cursor", "all RNG", "rank topology", "all identity manifests"],
            "planning_tensor_bytes_per_parameter": 12,
            "gradients_persisted": False,
        },
        "evaluation": {
            "tier0_tokens": 25_000_000,
            "tier0_cases_per_family": 32,
            "tier1_tokens": 100_000_000,
            "tier1_cases_per_family": 512,
            "sealed_cases_per_family": 1_024,
            "sealed_consumption": "once for the single development-selected checkpoint",
            "fresh_replication_required": True,
            "decoding": {"strategy": "greedy", "temperature": 0.0, "top_p": 1.0, "top_k": None, "max_new_tokens": 64, "stop": "EOS or cap"},
            "native_selection_gate": "candidate-free generation; candidate-set ranking is assisted diagnostic only",
            "representation_metrics": ["gold suffix NLL", "matched-candidate rank", "matched-candidate margin", "q-vs-q-prime lift"],
            "selection_metrics": ["candidate-free exact", "candidate-free semantic", "correct intervention flip", "invariance stability"],
            "realization_metrics": ["raw exact", "whitespace-normalized exact", "semantic", "malformed", "repetition", "conditional on correct unassisted selection"],
            "production_candidate_scoring_mode": None,
        },
        "promotion_gates": {
            "fresh_ood_selection_lcb_above_chance": 0.10,
            "sensitivity_correct_flip_wilson_lcb": 0.80,
            "invariance_stable_both_correct_wilson_lcb": 0.90,
            "state_ood_accuracy": 0.70,
            "two_hop_ood_accuracy": 0.60,
            "three_hop_lcb_above_chance": 0.10,
            "three_hop_max_degradation_from_two_hop": 0.20,
            "conditional_realization": 0.80,
            "missing_information_balanced_accuracy": 0.80,
            "missing_information_false_assertion_max": 0.10,
            "natural_substrate_loss_regression_max": 0.03,
            "code_math_loss_regression_max": 0.05,
            "maximum_family_regression": 0.05,
            "m102_replication_seeds": 2,
            "fresh_natural_paired_lcb_delta_minimum": 0.0,
        },
        "abort_rules": {
            "immediate": ["nonfinite loss or gradient", "token/cursor/optimizer mismatch", "identity/hash/custody mismatch", "resume inequivalence", "candidate or benchmark leakage", "training/evaluation source collision"],
            "pause_preserve": "two consecutive Tier1 worst-family declines >0.05 while LM loss improves",
            "deny_v5a": ["E1 artifact/corpus identity absent", "P35/E3 effect absent on fresh candidate-free and natural transfer", "M102 effect not reproduced", "target TPU or remote restore canary fails"],
        },
        "required_external_identities": ["tokenizer artifact and training corpus", "source/data/pack manifests", "target runtime/container/topology receipt", "remote durability provider receipt", "sealed evaluation commitment"],
        "scientific_gates_before_authorization": ["E1 tokenizer", "E2 2:1 architecture", "E3 CE cognition mixture", "E4 recipe", "M102 two-seed fresh replication", "target TPU join and remote restore"],
        "negative_evidence": {
            "variable_length_candidate_scoring": "sum, token mean, byte mean, DC-PMI, and contextual calibration rejected",
            "development_receipt": "artifacts/e2/scoring_policy_development.json",
            "fresh_scorer_fixture_consumed": False,
        },
        "run_arithmetic_source": run_receipt,
    }


def _ladder_spec(recipe: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        schema="anra-v5-model-spec/v1",
        family="dense-decoder-transformer",
        vocabulary_size=24_576,
        width=int(recipe["width"]),
        layers=int(recipe["layers"]),
        query_heads=int(recipe["query_heads"]),
        kv_heads=int(recipe["kv_heads"]),
        head_dimension=64,
        ffn_width=int(recipe["ffn_width"]),
        context_length=4_096,
        rope_base=10_000.0,
        norm_epsilon=1e-5,
        tied_embeddings=True,
        qk_norm=True,
        qk_norm_affine=True,
        linear_bias=False,
        dropout=0.0,
    )


def validate_training_spec(spec: dict[str, Any]) -> dict[str, bool]:
    ladder = spec["scale_ladder"]
    ladder_counts_exact = all(
        int(ladder[name]["parameters"]) == _ladder_spec(ladder[name]).parameter_receipt().total
        for name in ("p35_recipe", "m102_recipe", "v5a")
    )
    return {
        "ladder_parameter_counts_exact": ladder_counts_exact,
        "schema": spec["schema"] == SCHEMA,
        "main_run_fail_closed": spec["main_training_authorized"] is False,
        "parameter_count_exact": spec["core"]["parameter_count"] == 250_216_960,
        "gqa_ratio_exact": spec["core"]["query_heads"] == 2 * spec["core"]["kv_heads"],
        "model_matches_contract": spec["core"]["parameter_count"] == V5A_250M.parameter_receipt().total,
        "data_tokens_exact": sum(spec["data"]["mixture_tokens"].values()) == 5_000_000_000,
        "cognition_fractions_exact": math.isclose(sum(spec["cognition"]["family_fractions_within_cognition"].values()), 1.0),
        "cognition_tokens_exact": sum(spec["cognition"]["family_tokens_at_15_percent"].values()) == 750_000_000,
        "sequence_mix_exact": math.isclose(sum(spec["packing"]["sequence_buckets"].values()), 1.0),
        "difficulty_mix_exact": math.isclose(sum(spec["cognition"]["difficulty_distribution"].values()), 1.0),
        "update_arithmetic_exact": 38_146 * 131_072 + 127_488 == 5_000_000_000,
        "topology_tokens_exact": 8 * 4_096 * 4 == 131_072,
        "supercycle_exact": spec["packing"]["twenty_microstep_supercycle"].count(512) == 5
        and spec["packing"]["twenty_microstep_supercycle"].count(1024) == 5
        and spec["packing"]["twenty_microstep_supercycle"].count(2048) == 6
        and spec["packing"]["twenty_microstep_supercycle"].count(4096) == 4,
        "final_partial_plan_exact": sum(spec["optimization"]["final_partial_update_plan"]["global_real_tokens"]) == 127_488,
        "launch_auxiliaries_disabled": spec["objective"]["query_swap_lambda"] == 0.0 and spec["objective"]["trace_loss_lambda"] == 0.0,
        "production_scorer_unset": spec["evaluation"]["production_candidate_scoring_mode"] is None,
        "external_identities_unfilled": all(spec["tokenizer"][key] is None for key in ("artifact_sha256", "training_corpus_manifest_sha256")),
    }


def build_receipt() -> dict[str, Any]:
    spec = build_training_spec()
    checks = validate_training_spec(spec)
    return {
        "schema": "anra-v5-training-spec-receipt/v1",
        "status": "PASS_IMPLEMENTATION_SPEC_BLOCKED_MAIN_RUN" if all(checks.values()) else "FAIL",
        "spec": spec,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": receipt["status"]}, sort_keys=True))
    return 0 if receipt["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
