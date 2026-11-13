"""Hitchhiker counterfactual evaluation, independent replay, statistics, and gates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import itertools
import math
import random
from typing import Any

from .codec import canonical, digest
from .cognition import (
    Budget,
    CognitiveState,
    OperationCandidate,
    run_episode,
    verify,
)
from .hitchhiker_data import (
    ExternalSlotCanonicalizer,
    public_case_for,
    render_hitchhiker_public,
    score_health_probes,
)
from .hitchhiker_protocol import FORMATION_GATE, PROTOCOL_SHA, SCREENING_GATE, SEEDS
from .v6_evaluation import DEFAULT_BUDGET, QUERY_COST, _case_record, cases_identity, parse_action


ALLOWED_CLAIM_STATUSES = frozenset({"IMPLEMENTED", "OBSERVED", "INFERRED", "INCONCLUSIVE", "PENDING"})


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = len(rows)
    correct = sum(1 for r in rows if r["outcome"]["correct"])
    invalid = sum(1 for r in rows if r["trace"]["status"] == "invalid_action")
    abstained = sum(1 for r in rows if r["trace"]["status"] == "abstained")
    decisions = [d for r in rows for d in r["trace"]["decisions"]]
    pairs: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        if r["pair_id"]:
            pairs[r["pair_id"]].append(bool(r["outcome"]["correct"]))
    complete_pairs = [v for v in pairs.values() if len(v) == 2]
    both_correct = sum(1 for v in complete_pairs if all(v))

    def subset_stats(subset: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(subset)
        c = sum(1 for r in subset if r["outcome"]["correct"])
        inv = sum(1 for r in subset if r["trace"]["status"] == "invalid_action")
        return {
            "episodes": n,
            "correct": c,
            "accuracy": c / n if n else None,
            "invalid": inv,
            "invalid_rate": inv / n if n else None,
        }

    by_panel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_panel[r["panel"]].append(r)
        by_kind[r["kind"]].append(r)

    return {
        "episodes": episodes,
        "correct": correct,
        "accuracy": correct / episodes if episodes else None,
        "invalid": invalid,
        "invalid_rate": invalid / episodes if episodes else None,
        "abstained": abstained,
        "queries": sum(1 for d in decisions if d["operation"] == "QUERY"),
        "retrievals": sum(1 for d in decisions if d["operation"] == "RETRIEVE"),
        "model_calls": sum(d["model_calls"] for d in decisions),
        "model_positions": sum(d["model_positions"] for d in decisions),
        "complete_pairs": len(complete_pairs),
        "both_correct_pairs": both_correct,
        "pair_accuracy": both_correct / len(complete_pairs) if complete_pairs else None,
        "by_panel": {k: subset_stats(v) for k, v in sorted(by_panel.items())},
        "by_kind": {k: subset_stats(v) for k, v in sorted(by_kind.items())},
    }


def evaluate_hitchhiker(
    cases: tuple[Any, ...],
    policy_factory: Any,
    *,
    split: str = "validation",
    budget: Budget = DEFAULT_BUDGET,
    memory_mode: str = "on",
) -> dict[str, Any]:
    """Evaluate cases under strict public-only input isolation and origin attribution."""
    if memory_mode not in ("on", "off", "prefetch"):
        raise ValueError("unknown memory evaluation mode")
    case_list = tuple(cases.cases if hasattr(cases, "cases") else cases)
    if not case_list or len({c.case_id for c in case_list}) != len(case_list):
        raise ValueError("evaluation requires unique nonempty cases")

    rows: list[dict[str, Any]] = []
    for case in case_list:
        memory = None if memory_mode == "off" else case.memory
        is_oracle = getattr(policy_factory, "mode", None) == "oracle"
        public_case = case if is_oracle else public_case_for(case)
        actor = (
            policy_factory.for_case(public_case, memory_available=memory is not None)
            if hasattr(policy_factory, "for_case")
            else policy_factory(public_case)
        )
        if public_case is case and getattr(actor, "origin", None) != "privileged_oracle":
            raise ValueError("truth-bearing oracle configuration cannot be attributed to a model")
        trace = run_episode(
            case.goal,
            case.environment,
            actor,
            budget=budget,
            memory=memory,
            prefetch=(memory_mode == "prefetch"),
        )
        outcome = verify(trace, case.truth)
        first_decision = trace.decisions[0] if trace.decisions else None
        rows.append(
            {
                "case_id": case.case_id,
                "world_id": case.world_id,
                "family": case.family,
                "kind": case.kind,
                "pair_id": case.pair_id,
                "panel": case.panel,
                "trace": asdict(trace),
                "trace_sha256": trace.identity,
                "outcome": asdict(outcome),
                "truth_sha256": digest(asdict(case.truth)),
                "first_action": None if first_decision is None else {
                    "operation": first_decision["operation"] if isinstance(first_decision, dict) else first_decision.operation,
                    "argument": first_decision["argument"] if isinstance(first_decision, dict) else first_decision.argument,
                    "origin": first_decision["origin"] if isinstance(first_decision, dict) else first_decision.origin,
                },
                "prompts": list(getattr(actor, "prompts", [])),
                "overhead_receipts": list(getattr(actor, "overhead_receipts", [])),
            }
        )

    identities = sorted({r["trace"]["model_identity"] for r in rows if r["trace"]["model_identity"] is not None})
    if len(identities) > 1:
        raise ValueError("evaluation mixes model checkpoint identities")

    decoder_schema = getattr(policy_factory, "decoder_schema", getattr(policy_factory, "representation", "utf8-raw"))
    attribution_origin = getattr(policy_factory, "origin", getattr(policy_factory, "mode", "declared_factory"))
    return {
        "schema": "elendril-hitchhiker-evaluation-v1",
        "protocol_sha256": PROTOCOL_SHA,
        "decoder_schema": decoder_schema,
        "attribution_origin": attribution_origin,
        "cases_sha256": cases_identity(case_list),
        "split": split,
        "budget": asdict(budget),
        "memory_mode": memory_mode,
        "model_identity": identities[0] if identities else None,
        "canonicalizer_config": (
            {
                "assignment": getattr(policy_factory, "assignment", "first-mention"),
                "slot_seed": getattr(policy_factory, "slot_seed", 17),
                "strip_aliases": getattr(policy_factory, "strip_aliases", True),
            }
            if attribution_origin == "engineered_canonicalizer"
            else None
        ),
        "metrics": _summarize_rows(rows),
        "episodes": rows,
    }


def independent_rescore_hitchhiker(cases: tuple[Any, ...], record: dict[str, Any]) -> dict[str, Any]:
    """Replay every recorded raw action sequence and verify trace, outcome, and metrics."""
    case_list = tuple(cases.cases if hasattr(cases, "cases") else cases)
    if (
        record.get("schema") != "elendril-hitchhiker-evaluation-v1"
        or record.get("cases_sha256") != cases_identity(case_list)
        or len(record.get("episodes", [])) != len(case_list)
    ):
        raise ValueError("Hitchhiker replay schema, cases hash, or count mismatch")

    indexed = {r["case_id"]: r for r in record["episodes"]}
    if set(indexed) != {c.case_id for c in case_list}:
        raise ValueError("Hitchhiker replay case set mismatch")

    canon_cfg = record.get("canonicalizer_config")
    canonicalizer = (
        ExternalSlotCanonicalizer(
            assignment=canon_cfg["assignment"],
            slot_seed=canon_cfg["slot_seed"],
            strip_aliases=canon_cfg["strip_aliases"],
        )
        if canon_cfg is not None
        else None
    )

    class ReplayFactory:
        decoder_schema = record["decoder_schema"]
        origin = record["attribution_origin"]
        mode = record["attribution_origin"]

        def for_case(self, case: Any, *, memory_available: bool | None = None):
            ref = indexed[case.case_id]
            steps = ref["trace"]["decisions"]

            class Replay:
                model_identity = record["model_identity"]
                origin = ref["trace"]["decisions"][0]["origin"] if steps else record["attribution_origin"]

                def __init__(self) -> None:
                    self.index = 0
                    self.prompts: list[str] = []
                    self.overhead_receipts: list[dict[str, int]] = []
                    self._mapping: dict[str, str] | None = None
                    self._inverse: dict[str, str] | None = None

                def propose(self, state: CognitiveState, legal: frozenset[str], channels: tuple[str, ...], positions: int):
                    if self.index == len(steps):
                        return None
                    step = steps[self.index]
                    self.index += 1
                    self.origin = step["origin"]
                    if self.origin in ("model", "engineered_canonicalizer"):
                        raw_op, raw_arg = parse_action(tuple(step["raw_tokens"]), channels, "utf8-raw")
                        if self.origin == "engineered_canonicalizer":
                            assert canonicalizer is not None
                            if self._mapping is None:
                                self._mapping, self._inverse = canonicalizer.build_mapping(state, case.surface)
                            else:
                                self._mapping, self._inverse = canonicalizer.extend_mapping(self._mapping, self._inverse, state)
                            m_state, m_surf, overhead = canonicalizer.canonicalize_public(state, case.surface, self._mapping)
                            self.overhead_receipts.append(overhead)
                            self.prompts.append(
                                render_hitchhiker_public(m_state, channels, m_surf, bool(memory_available) and "RETRIEVE" in legal)
                            )
                            op, arg = canonicalizer.dereference_output(raw_op, raw_arg, self._inverse)
                        else:
                            self.prompts.append(
                                render_hitchhiker_public(state, channels, case.surface, bool(memory_available) and "RETRIEVE" in legal)
                            )
                            op, arg = raw_op, raw_arg
                        if (op, arg) != (step["operation"], step["argument"]):
                            raise ValueError("replayed raw tokens disagree with recorded decision")
                    elif self.origin in ("fixed_baseline", "privileged_oracle") and record["model_identity"] is None:
                        op, arg = step["operation"], step["argument"]
                    else:
                        raise ValueError("invalid decision origin during Hitchhiker replay")
                    return OperationCandidate(
                        op,
                        arg,
                        step["predicted_success"],
                        step["model_positions"],
                        step["model_calls"],
                        tuple(step["raw_tokens"]),
                    )

            return Replay()

    checked = evaluate_hitchhiker(
        case_list,
        ReplayFactory(),
        split=record["split"],
        budget=Budget(**record["budget"]),
        memory_mode=record["memory_mode"],
    )
    if canonical(checked["metrics"]) != canonical(record["metrics"]) or canonical(checked["episodes"]) != canonical(record["episodes"]):
        raise ValueError("Hitchhiker raw-action replay disagrees with recorded traces or metrics")
    return checked["metrics"]


def evaluate_counterfactual_suite(
    suite: dict[str, Any],
    policy_factory: Any,
    *,
    budget: Budget = DEFAULT_BUDGET,
) -> dict[str, Any]:
    """Evaluate all cohorts in the 7-factor counterfactual suite and summarize contrasts."""
    cohort_records: dict[str, Any] = {}
    cohort_metrics: dict[str, Any] = {}
    for name, cases in suite["cohorts"].items():
        if not cases:
            continue
        rec = evaluate_hitchhiker(cases, policy_factory, split=suite["split"], budget=budget)
        independent_rescore_hitchhiker(cases, rec)
        cohort_records[name] = rec
        cohort_metrics[name] = rec["metrics"]

    factor_summary: dict[str, Any] = {}
    for name, m in cohort_metrics.items():
        if not name.startswith("factor-"):
            continue
        item: dict[str, Any] = {
            "accuracy": m["accuracy"],
            "invalid_rate": m["invalid_rate"],
            "pair_accuracy": m["pair_accuracy"],
            "by_panel": m["by_panel"],
        }
        if name == "factor-distractors":
            p = m["by_panel"]
            a_none = p.get("distractors-none", {}).get("accuracy")
            a_dist = p.get("distractors-matched3", {}).get("accuracy")
            item["accuracy_gap"] = (a_none - a_dist) if (a_none is not None and a_dist is not None) else None
        if name in ("factor-answer-edge-swap", "factor-answer-query-swap"):
            item["both_correct_rate"] = m["pair_accuracy"]
        factor_summary[name] = item

    return {
        "schema": "elendril-hitchhiker-suite-eval-v1",
        "protocol_sha256": PROTOCOL_SHA,
        "split": suite["split"],
        "worlds_per_hop": suite["worlds_per_hop"],
        "attribution_origin": getattr(policy_factory, "origin", getattr(policy_factory, "mode", "declared_factory")),
        "decoder_schema": getattr(policy_factory, "decoder_schema", "utf8-raw"),
        "cohort_metrics": cohort_metrics,
        "factor_summary": factor_summary,
        "records": cohort_records,
    }


def evaluate_health_probes(
    probes: dict[str, Any],
    policy_factory: Any,
    *,
    stage_records: dict[str, Any] | None = None,
    split: str = "validation",
    budget: Budget = DEFAULT_BUDGET,
) -> dict[str, Any]:
    """Evaluate Distractor-Trap, Belief-Revision, Missing-Bridge, and Preservation probes."""
    records: dict[str, Any] = {}
    rows_by_probe: dict[str, list[dict[str, Any]]] = {}
    for name in ("distractor-trap", "belief-revision", "missing-bridge"):
        cases = probes.get(name, ())
        if not cases:
            continue
        rec = evaluate_hitchhiker(cases, policy_factory, split=split, budget=budget)
        independent_rescore_hitchhiker(cases, rec)
        records[name] = rec
        rows_by_probe[name] = rec["episodes"]
    return {
        "schema": "elendril-hitchhiker-probes-eval-v1",
        "scores": score_health_probes(rows_by_probe, stage_records=stage_records),
        "records": records,
    }


def paired_world_bootstrap_contrast(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    seed: int = 8675309,
    resamples: int = 1000,
) -> dict[str, Any]:
    """Cluster episodes by world_id and compute paired bootstrap CI and sign-flip p-value."""
    left_clusters: dict[str, list[int]] = defaultdict(list)
    right_clusters: dict[str, list[int]] = defaultdict(list)
    for row in left_rows:
        left_clusters[row["world_id"]].append(int(bool(row["outcome"]["correct"])))
    for row in right_rows:
        right_clusters[row["world_id"]].append(int(bool(row["outcome"]["correct"])))

    common = sorted(set(left_clusters) & set(right_clusters))
    if not common:
        return {
            "worlds": 0,
            "left_mean": None,
            "right_mean": None,
            "mean_diff": None,
            "ci_95": None,
            "p_value": None,
        }

    world_pairs = [
        (
            sum(left_clusters[w]),
            len(left_clusters[w]),
            sum(right_clusters[w]),
            len(right_clusters[w]),
        )
        for w in common
    ]
    left_mean = sum(x[0] for x in world_pairs) / sum(x[1] for x in world_pairs)
    right_mean = sum(x[2] for x in world_pairs) / sum(x[3] for x in world_pairs)
    obs_diff = left_mean - right_mean

    rng = random.Random(seed)
    diffs: list[float] = []
    n = len(world_pairs)
    for _ in range(resamples):
        drawn = [world_pairs[rng.randrange(n)] for _ in range(n)]
        l_acc = sum(x[0] for x in drawn) / sum(x[1] for x in drawn)
        r_acc = sum(x[2] for x in drawn) / sum(x[3] for x in drawn)
        diffs.append(l_acc - r_acc)
    diffs.sort()
    lo_idx = max(0, int(math.floor(0.025 * (resamples - 1))))
    hi_idx = min(resamples - 1, int(math.ceil(0.975 * (resamples - 1))))

    per_world_diffs = [(lc / ln) - (rc / rn) for lc, ln, rc, rn in world_pairs]
    obs_world_mean = abs(math.fsum(per_world_diffs) / n)
    if n <= 10:
        extreme = 0
        total = 1 << n
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            stat = abs(math.fsum(s * d for s, d in zip(signs, per_world_diffs)) / n)
            if stat >= obs_world_mean - 1e-12:
                extreme += 1
        p_val = extreme / total
    else:
        extreme = 0
        for _ in range(resamples):
            stat = abs(math.fsum((1.0 if rng.randrange(2) else -1.0) * d for d in per_world_diffs) / n)
            if stat >= obs_world_mean - 1e-12:
                extreme += 1
        p_val = (extreme + 1) / (resamples + 1)

    return {
        "worlds": n,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "mean_diff": obs_diff,
        "ci_95": [diffs[lo_idx], diffs[hi_idx]],
        "p_value": p_val,
    }


def holm_adjust(p_values: dict[str, float] | list[float] | tuple[float, ...]) -> dict[str, float] | list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    if isinstance(p_values, dict):
        keys = list(p_values)
        vals = [float(p_values[k]) for k in keys]
        adjusted_list = holm_adjust(vals)
        return dict(zip(keys, adjusted_list))

    raw = [float(p) for p in p_values]
    m = len(raw)
    if m == 0:
        return []
    if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in raw):
        raise ValueError("p_values must be finite probabilities in [0, 1]")
    indexed = sorted(enumerate(raw), key=lambda item: item[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        cand = min(1.0, (m - rank) * p)
        running_max = max(running_max, cand)
        adjusted[orig_idx] = running_max
    return adjusted


def evaluate_screening_gate(
    arm_reports_by_seed: dict[int, dict[str, Any]],
    baseline_reports_by_seed: dict[int, dict[str, Any]],
    *,
    primary_cohort: str = "factor-entity-permutation",
    extra_contrasts: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Apply Section 6 screening criteria across seeds 7/19/41 with world-cluster CI."""
    common_seeds = sorted(set(arm_reports_by_seed) & set(baseline_reports_by_seed))
    if not common_seeds:
        raise ValueError("screening gate requires at least one paired seed")

    seed_diffs: dict[int, float] = {}
    arm_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    max_kind_drop = 0.0

    for seed in common_seeds:
        a_rec = arm_reports_by_seed[seed]["records"][primary_cohort]
        b_rec = baseline_reports_by_seed[seed]["records"][primary_cohort]
        a_acc = a_rec["metrics"]["accuracy"] or 0.0
        b_acc = b_rec["metrics"]["accuracy"] or 0.0
        seed_diffs[seed] = a_acc - b_acc
        arm_rows.extend(a_rec["episodes"])
        base_rows.extend(b_rec["episodes"])

        # Check original task kind regressions if present.
        a_kinds = arm_reports_by_seed[seed].get("original_kind_accuracies", {})
        b_kinds = baseline_reports_by_seed[seed].get("original_kind_accuracies", {})
        for k in set(a_kinds) & set(b_kinds):
            drop = (b_kinds[k] or 0.0) - (a_kinds[k] or 0.0)
            max_kind_drop = max(max_kind_drop, drop)

    mean_gain = math.fsum(seed_diffs.values()) / len(common_seeds)
    all_positive = all(d > 0.0 for d in seed_diffs.values())
    contrast = paired_world_bootstrap_contrast(arm_rows, base_rows)
    ci_positive = bool(contrast["ci_95"] and contrast["ci_95"][0] > 0.0)
    kind_ok = max_kind_drop <= SCREENING_GATE["max_original_kind_regression"]

    if mean_gain >= SCREENING_GATE["min_mean_primary_gain"] and all_positive and kind_ok and ci_positive:
        status = "PASS_SCREENING"
    elif (not kind_ok) or (contrast["ci_95"] is not None and contrast["ci_95"][1] < SCREENING_GATE["min_mean_primary_gain"] and mean_gain <= 0.0):
        status = "FALSIFIED"
    else:
        status = "INCONCLUSIVE"

    return {
        "status": status,
        "primary_cohort": primary_cohort,
        "seeds": common_seeds,
        "seed_differences": seed_diffs,
        "mean_primary_gain": mean_gain,
        "all_seeds_positive": all_positive,
        "max_original_kind_regression": max_kind_drop,
        "world_cluster_contrast": contrast,
        "holm_adjusted_contrasts": holm_adjust(extra_contrasts) if extra_contrasts else None,
    }


def evaluate_formation_gate(
    suite_eval: dict[str, Any],
    *,
    equivariance_report: dict[str, Any] | None = None,
    confirmation_worlds_per_cohort: int | None = None,
) -> dict[str, Any]:
    """Apply Section 6 formation thresholds without promoting undersized mechanical checks."""
    metrics = suite_eval["cohort_metrics"]
    worlds = confirmation_worlds_per_cohort if confirmation_worlds_per_cohort is not None else suite_eval.get("worlds_per_hop", 0)

    hop_accs = {
        h: metrics[h]["accuracy"]
        for h in ("hop-1", "hop-2", "hop-3", "hop-4")
        if h in metrics and metrics[h]["accuracy"] is not None
    }
    stress_accs = {
        k: m["accuracy"]
        for k, m in metrics.items()
        if k.startswith("factor-") and m["accuracy"] is not None
    }
    edge_pair_acc = metrics.get("factor-answer-edge-swap", {}).get("pair_accuracy")
    qswap_pair_acc = metrics.get("factor-answer-query-swap", {}).get("pair_accuracy")
    inv_rates = [m["invalid_rate"] for m in metrics.values() if m.get("invalid_rate") is not None]
    max_invalid = max(inv_rates) if inv_rates else 1.0
    renaming_consistency = equivariance_report.get("argmax_equivariance_rate") if equivariance_report else None

    if worlds < FORMATION_GATE["min_confirmation_worlds_per_cohort"]:
        return {
            "status": "MECHANICAL_ONLY_NOT_FORMATION_QUALIFIED",
            "reason": f"confirmation world count {worlds} below required {FORMATION_GATE['min_confirmation_worlds_per_cohort']}",
            "hop_accuracies": hop_accs,
            "stress_accuracies": stress_accs,
            "edge_pair_accuracy": edge_pair_acc,
            "query_swap_pair_accuracy": qswap_pair_acc,
            "renaming_consistency": renaming_consistency,
            "max_invalid_rate": max_invalid,
        }

    thresholds_met = (
        bool(hop_accs)
        and all(a >= FORMATION_GATE["min_raw_accuracy_per_hop"] for a in hop_accs.values())
        and bool(stress_accs)
        and all(a >= FORMATION_GATE["min_stress_subgroup_accuracy"] for a in stress_accs.values())
        and edge_pair_acc is not None
        and edge_pair_acc >= FORMATION_GATE["min_answer_changing_pair_accuracy"]
        and (qswap_pair_acc is None or qswap_pair_acc >= FORMATION_GATE["min_answer_changing_pair_accuracy"])
        and renaming_consistency is not None
        and renaming_consistency >= FORMATION_GATE["min_renaming_consistency"]
        and max_invalid <= FORMATION_GATE["max_invalid_action_rate"]
    )
    return {
        "status": "FORMATION_QUALIFIED" if thresholds_met else "FALSIFIED",
        "hop_accuracies": hop_accs,
        "stress_accuracies": stress_accs,
        "edge_pair_accuracy": edge_pair_acc,
        "query_swap_pair_accuracy": qswap_pair_acc,
        "renaming_consistency": renaming_consistency,
        "max_invalid_rate": max_invalid,
    }


def _assert_claim_status(status: str) -> str:
    if status not in ALLOWED_CLAIM_STATUSES:
        raise ValueError(f"invalid claim status '{status}'; allowed: {sorted(ALLOWED_CLAIM_STATUSES)}")
    return status


def build_mechanism_decision_table(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build explicit decision table validating every claim status label."""
    table = []
    for item in entries:
        table.append(
            {
                "mechanism": item["mechanism"],
                "status": _assert_claim_status(item["status"]),
                "attribution": item["attribution"],
                "evidence_summary": item["evidence_summary"],
                "next_action": item["next_action"],
            }
        )
    return table


def build_experiment_manifest(
    *,
    source_commit: str,
    stage: str,
    stage0_status: dict[str, Any],
    arm_runs: list[dict[str, Any]],
    decision_table: list[dict[str, Any]],
    claim_status: str = "IMPLEMENTED",
) -> dict[str, Any]:
    """Assemble full Hitchhiker manifest binding source, protocol, costs, and statuses."""
    _assert_claim_status(claim_status)
    return {
        "schema": "elendril-hitchhiker-manifest-v1",
        "protocol_sha256": PROTOCOL_SHA,
        "source_commit": source_commit,
        "stage": stage,
        "claim_status": claim_status,
        "stage0_checkpoint_audit": stage0_status,
        "arm_runs": arm_runs,
        "decision_table": build_mechanism_decision_table(decision_table),
        "capability_claim": "NO_COGNITION_CLAIM_FROM_MECHANICAL_OR_UNEXECUTED_CHECKS",
    }
