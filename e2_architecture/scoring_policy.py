"""Outcome-free preregistration for the P35 candidate-scoring tournament.

This module deliberately contains policy mathematics and a frozen experiment
contract, but no model result. The preregistration must be committed before
the fresh random-weight tournament is implemented or executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from e0_cognition.scoring_certification import CandidateTrace


PLAN_SCHEMA = "esoes-e2-scoring-policy-preregistration/v2"
DEVELOPMENT_SEEDS = tuple(range(95_101, 95_106))
FRESH_SEEDS = tuple(range(95_201, 95_206))
VOCABULARIES = (16_384, 24_576, 32_768)
NULL_CONTEXTS_PER_PANEL = 4
INDEPENDENT_TRIPLETS = 256
ROTATIONS_PER_TRIPLET = 3
EQUIVALENCE_MARGIN = 0.05
PER_SEED_MARGIN = 0.10
HOLM_FAMILYWISE_ALPHA = 0.01
MAX_GPU_HOURS = 0.5


class Policy(str, Enum):
    SUM = "sum"
    TOKEN_MEAN = "token_mean"
    BYTE_MEAN = "byte_mean"
    DOMAIN_PMI = "domain_pmi"
    CONTEXTUAL_CALIBRATION = "contextual_calibration"


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    candidate: str
    target: CandidateTrace
    neutral: tuple[CandidateTrace, ...]

    def assert_valid(self) -> None:
        self.target.assert_valid()
        if len(self.neutral) != NULL_CONTEXTS_PER_PANEL:
            raise ValueError("candidate evidence requires exactly four neutral traces")
        for trace in self.neutral:
            trace.assert_valid()
            if trace.token_ids != self.target.token_ids:
                raise ValueError("target and neutral candidate tokenizations differ")
        if not self.candidate:
            raise ValueError("candidate text is empty")


def _sha256_file(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _logsumexp(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("logsumexp values must be finite and nonempty")
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def _sequence_log_likelihood(trace: CandidateTrace) -> float:
    trace.assert_valid()
    return math.fsum(trace.token_logprobs)


def score_independent_policy(evidence: CandidateEvidence, policy: Policy) -> float:
    """Score policies whose value does not depend on the other candidates."""

    evidence.assert_valid()
    target = _sequence_log_likelihood(evidence.target)
    if policy is Policy.SUM:
        return target
    if policy is Policy.TOKEN_MEAN:
        return target / len(evidence.target.token_ids)
    if policy is Policy.BYTE_MEAN:
        return target / len(evidence.candidate.encode("utf-8"))
    if policy is Policy.DOMAIN_PMI:
        neutral_mean = math.fsum(
            _sequence_log_likelihood(trace) for trace in evidence.neutral
        ) / len(evidence.neutral)
        return target - neutral_mean
    raise ValueError("contextual calibration requires the complete candidate set")


def score_contextual_calibration(
    candidates: Sequence[CandidateEvidence],
) -> dict[str, float]:
    """Return Zhao-style content-free calibrated log scores in log space."""

    if len(candidates) < 2 or len({item.candidate for item in candidates}) != len(candidates):
        raise ValueError("contextual calibration needs at least two unique candidates")
    for item in candidates:
        item.assert_valid()
    target_totals = [_sequence_log_likelihood(item.target) for item in candidates]
    target_normalizer = _logsumexp(target_totals)
    neutral_normalizers = [
        _logsumexp([_sequence_log_likelihood(item.neutral[index]) for item in candidates])
        for index in range(NULL_CONTEXTS_PER_PANEL)
    ]
    result: dict[str, float] = {}
    for candidate_index, item in enumerate(candidates):
        target_log_probability = target_totals[candidate_index] - target_normalizer
        neutral_log_probabilities = [
            _sequence_log_likelihood(item.neutral[index]) - neutral_normalizers[index]
            for index in range(NULL_CONTEXTS_PER_PANEL)
        ]
        log_prior = _logsumexp(neutral_log_probabilities) - math.log(NULL_CONTEXTS_PER_PANEL)
        result[item.candidate] = target_log_probability - log_prior
    return result


def select(scores: Mapping[str, float]) -> str:
    if len(scores) < 2 or any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("selection scores must be finite and contain at least two candidates")
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if math.isclose(ordered[0][1], ordered[1][1], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("unresolved candidate-score tie")
    return ordered[0][0]


def build_preregistration() -> dict[str, object]:
    """Return the immutable decision protocol, with no model outcomes."""

    return {
        "schema": PLAN_SCHEMA,
        "status": "PREREGISTERED_NO_RESULTS",
        "decision": "candidate-scoring policy eligibility only",
        "primary_policy": Policy.DOMAIN_PMI.value,
        "secondary_policy": Policy.CONTEXTUAL_CALIBRATION.value,
        "negative_controls": [Policy.SUM.value, Policy.TOKEN_MEAN.value, Policy.BYTE_MEAN.value],
        "decision_hierarchy": [
            "Promote domain_pmi only if every gate passes on development and unchanged fresh replication.",
            "Otherwise promote contextual_calibration only if every gate, including decoy stability, passes.",
            "Otherwise keep production_scoring_mode null and block learned E1-E3 comparisons.",
        ],
        "fixtures": {
            "independent_candidate_triplets": INDEPENDENT_TRIPLETS,
            "position_rotations_per_triplet": ROTATIONS_PER_TRIPLET,
            "surface_families": 6,
            "candidate_roles": [
                "unique_shortest_utf8",
                "unique_fewest_tokens",
                "marked_prefix_surface",
                "counterbalanced_hidden_label",
            ],
            "roles_counterbalanced_per_tokenizer": True,
            "development_and_fresh_fixture_hashes_must_differ": True,
        },
        "model": {
            "family": "exact middle P35",
            "vocabularies": list(VOCABULARIES),
            "development_seeds": list(DEVELOPMENT_SEEDS),
            "fresh_seeds": list(FRESH_SEEDS),
            "full_device": "cuda",
            "parity_device": "cpu",
            "training_performed": False,
        },
        "neutral_contexts": {
            "contexts_per_panel": NULL_CONTEXTS_PER_PANEL,
            "panels": 2,
            "candidate_independent": True,
            "tokenizer_specific": True,
            "exact_target_prompt_token_length": True,
            "candidate_suffix_token_ids_identical_to_target": True,
            "panels_have_disjoint_neutral_token_patterns": True,
        },
        "statistics": {
            "primary_unit": "independent candidate triplet",
            "position_rotations_are_not_independent": True,
            "model_seed_is_a_cluster": True,
            "equivalence_margin_from_permutation_chance": EQUIVALENCE_MARGIN,
            "per_seed_maximum_deviation": PER_SEED_MARGIN,
            "equivalence_confidence": 0.90,
            "holm_familywise_alpha": HOLM_FAMILYWISE_ALPHA,
            "axes": [
                "shortest_utf8",
                "fewest_tokens",
                "marked_prefix",
                "surface_family",
                "arbitrary_hidden_label",
            ],
            "operational_hypotheses": {
                "overall": [
                    "winner_is_unique_shortest_utf8_role",
                    "winner_is_unique_fewest_tokens_role",
                    "winner_is_marked_prefix_role",
                    "winner_is_counterbalanced_hidden_label",
                ],
                "surface_family": "winner-role rate for every one of 6 families x 3 roles",
                "total_per_policy_tokenizer": 22,
            },
            "equivalence_test": "parametric cluster-level TOST with Student-t df=4",
            "holm_family": "all 132 TOST p-values: 2 promotable policies x 3 tokenizers x 22 hypotheses",
            "holm_applies_to": "max of the two one-sided p-values for each equivalence hypothesis",
        },
        "gates": {
            "rotation_stability": 1.0,
            "first_position_selection_rate": 1 / 3,
            "nonfinite_scores": 0,
            "unresolved_ties": 0,
            "pooled_equivalence_interval_inside_margin": True,
            "every_seed_inside_per_seed_margin": True,
            "neutral_panel_ranking_agreement_minimum": 0.95,
            "neutral_panel_agreement_definition": "exact full three-candidate ranking",
            "irrelevant_decoy_shared_ranking_stability_minimum": 0.99,
            "decoy_definition": "recompute four-candidate policy, then compare induced full order of original three",
            "synthetic_target_injection_recovery": 1.0,
            "synthetic_target_swap_recovery": 1.0,
            "synthetic_intervention": "replace one valid target token log-probability by -1e-6 before policy scoring; rotate target role",
            "cpu_cuda_winner_mismatches": 0,
            "cpu_cuda_maximum_absolute_score_error": 0.05,
            "cpu_cuda_relative_rms_error": 0.001,
            "must_pass_every_tokenizer": True,
        },
        "freshness_and_leakage": {
            "current_six_group_null_is_consumed_pilot_only": True,
            "policy_and_gates_frozen_before_development_execution": True,
            "selected_policy_written_to_immutable_receipt_before_fresh_execution": True,
            "fresh_results_cannot_change_policy_or_thresholds": True,
            "trained_model_outcomes_cannot_select_or_modify_policy": True,
        },
        "compute": {
            "expected_gpu_hours_upper": 0.25,
            "abort_gpu_hours": MAX_GPU_HOURS,
            "all_five_policies_reuse_identical_model_traces": True,
        },
        "abort_rules": [
            "Neutral prompt length or candidate suffix tokens differ from target.",
            "Candidate roles cannot be uniquely counterbalanced for every tokenizer.",
            "No calibrated policy passes every development gate.",
            "Implementation exceeds the compute abort budget.",
            "Any result is inspected before its governing identity is frozen.",
            "Surface-family x hidden-role contingency differs by more than one count in any cell row.",
        ],
        "limitations": [
            "Random-weight null safety is necessary but does not prove trained-model validity or cognition.",
            "Candidate scoring remains assisted evaluation; candidate-free realization stays separate.",
            "This tournament cannot select tokenizer, architecture, data mixture, objective, or checkpoint.",
        ],
        "primary_sources": [
            "https://aclanthology.org/2021.emnlp-main.564/",
            "https://proceedings.mlr.press/v139/zhao21c.html",
        ],
        "implementation_sha256": _sha256_file(Path(__file__)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_preregistration()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
