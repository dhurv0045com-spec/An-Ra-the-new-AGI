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


ALLOWED_CLAIM_STATUSES = frozenset(
    {
        "IMPLEMENTED",
        "OBSERVED",
        "INFERRED",
        "INCONCLUSIVE",
        "PENDING",
        "NOT_QUALIFIED",
        "INCOMPLETE_NOT_QUALIFIED",
        "GPU_QUALIFIED",
        "EXECUTED",
        "CAPABILITY_QUALIFIED",
    }
)

ORIGINAL_V6_KINDS = ("known", "missing", "conflict", "null", "unavailable", "route", "memory")

REQUIRED_FORMATION_COHORTS = (
    "hop-1",
    "hop-2",
    "hop-3",
    "hop-4",
    "factor-wording",
    "factor-timestamps",
    "factor-row-order",
    "factor-distractors",
    "factor-entity-permutation",
    "factor-alias-clause",
    "factor-answer-edge-swap",
    "factor-answer-query-swap",
)


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
        "unique_worlds": len({r["world_id"] for r in rows}),
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


def evaluate_original_kinds(
    dataset: Any,
    policy_factory: Any,
    *,
    split: str = "validation",
    budget: Budget = DEFAULT_BUDGET,
) -> dict[str, Any]:
    """Evaluate all 7 standard V6 task kinds, replay actions, and return kind accuracies + record."""
    if hasattr(dataset, "split") and dataset.split != split:
        raise ValueError(f"dataset split {dataset.split!r} disagrees with evaluation split {split!r}")
    cases = tuple(dataset.cases if hasattr(dataset, "cases") else dataset)
    rec = evaluate_hitchhiker(cases, policy_factory, split=split, budget=budget)
    independent_rescore_hitchhiker(cases, rec)
    rec["independent_rescore_status"] = "PASS"
    by_kind = rec["metrics"]["by_kind"]
    missing = [k for k in ORIGINAL_V6_KINDS if k not in by_kind or by_kind[k].get("accuracy") is None]
    if missing:
        raise ValueError(f"original V6 evaluation missing required task kinds: {missing}")
    original_kind_accuracies = {k: float(by_kind[k]["accuracy"]) for k in ORIGINAL_V6_KINDS}
    return {
        "schema": "elendril-hitchhiker-original-kinds-eval-v1",
        "protocol_sha256": PROTOCOL_SHA,
        "split": split,
        "attribution_origin": rec["attribution_origin"],
        "decoder_schema": rec["decoder_schema"],
        "model_identity": rec["model_identity"],
        "original_kind_accuracies": original_kind_accuracies,
        "independent_rescore_status": "PASS",
        "record": rec,
    }


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
        rec["independent_rescore_status"] = "PASS"
        cohort_records[name] = rec
        cohort_metrics[name] = rec["metrics"]

    rec_identities = {rec.get("model_identity") for rec in cohort_records.values()}
    if len(rec_identities) > 1:
        raise ValueError("counterfactual suite evaluation mixes model identities")
    suite_model_identity = next(iter(rec_identities)) if rec_identities else None

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
        "worlds_per_hop": suite.get("worlds_per_hop", suite.get("worlds_per_topology")),
        "attribution_origin": getattr(policy_factory, "origin", getattr(policy_factory, "mode", "declared_factory")),
        "decoder_schema": getattr(policy_factory, "decoder_schema", "utf8-raw"),
        "model_identity": suite_model_identity,
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
        rec["independent_rescore_status"] = "PASS"
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


def _is_finite_unit_float(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= float(value) <= 1.0
    )


def _validate_screening_report(
    rep: Any,
    *,
    label: str,
    seed: int,
    primary_cohort: str,
    expected_source_commit: str | None,
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(rep, dict):
        return [f"seed {seed} {label} report is not a dict"]
    if rep.get("split") != "validation":
        reasons.append(f"seed {seed} {label} split is {rep.get('split')!r}, expected 'validation'")
    if rep.get("protocol_sha256") != PROTOCOL_SHA:
        reasons.append(f"seed {seed} {label} protocol_sha256 mismatch")
    if rep.get("attribution_origin") != "model":
        reasons.append(f"seed {seed} {label} attribution_origin is {rep.get('attribution_origin')!r}, expected 'model'")
    if rep.get("decoder_schema") != "utf8-raw":
        reasons.append(f"seed {seed} {label} decoder_schema is {rep.get('decoder_schema')!r}, expected 'utf8-raw'")
    if expected_source_commit is not None and rep.get("source_commit") != expected_source_commit:
        reasons.append(f"seed {seed} {label} source_commit mismatch")

    records = rep.get("records")
    if not isinstance(records, dict) or primary_cohort not in records:
        reasons.append(f"seed {seed} {label} missing primary cohort {primary_cohort!r}")

    kinds = rep.get("original_kind_accuracies")
    if not isinstance(kinds, dict):
        reasons.append(f"seed {seed} {label} missing original_kind_accuracies")
    else:
        missing_kinds = [k for k in ORIGINAL_V6_KINDS if k not in kinds or not _is_finite_unit_float(kinds[k])]
        if missing_kinds:
            reasons.append(f"seed {seed} {label} incomplete original_kind_accuracies: {missing_kinds}")
    return reasons


def _screening_pairing_and_worlds(
    arm_ep: Any,
    base_ep: Any,
    *,
    seed: int,
    min_dev_worlds: int,
) -> list[str]:
    if not isinstance(arm_ep, list) or not isinstance(base_ep, list) or not arm_ep or not base_ep:
        return [f"seed {seed} primary cohort episodes missing or empty"]
    arm_keys = [(e.get("case_id"), e.get("world_id"), e.get("panel")) for e in arm_ep]
    base_keys = [(e.get("case_id"), e.get("world_id"), e.get("panel")) for e in base_ep]
    if any(None in key for key in arm_keys) or any(None in key for key in base_keys):
        return [f"seed {seed} primary cohort episodes missing case_id/world_id/panel"]
    if arm_keys != base_keys:
        return [f"seed {seed} arm/baseline case_id/world_id/panel pairing mismatch"]
    world_count = len({e["world_id"] for e in arm_ep})
    if world_count < min_dev_worlds:
        return [f"seed {seed} independent dev world count {world_count} below required {min_dev_worlds}"]
    return []


def _compute_screening_statistics(
    arm_reports_by_seed: dict[int, dict[str, Any]],
    baseline_reports_by_seed: dict[int, dict[str, Any]],
    seeds: list[int],
    primary_cohort: str,
) -> dict[str, Any]:
    seed_diffs: dict[int, float] = {}
    arm_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    max_kind_drop = 0.0

    for seed in seeds:
        arm_records = arm_reports_by_seed[seed].get("records", {})
        base_records = baseline_reports_by_seed[seed].get("records", {})
        if primary_cohort not in arm_records or primary_cohort not in base_records:
            continue
        a_rec = arm_records[primary_cohort]
        b_rec = base_records[primary_cohort]
        a_ep = a_rec.get("episodes", [])
        b_ep = b_rec.get("episodes", [])
        a_acc = a_rec.get("metrics", {}).get("accuracy")
        b_acc = b_rec.get("metrics", {}).get("accuracy")
        if a_acc is None and a_ep:
            a_acc = sum(1 for e in a_ep if e["outcome"]["correct"]) / len(a_ep)
        if b_acc is None and b_ep:
            b_acc = sum(1 for e in b_ep if e["outcome"]["correct"]) / len(b_ep)
        if a_acc is not None and b_acc is not None:
            seed_diffs[seed] = float(a_acc) - float(b_acc)
        arm_rows.extend(a_ep)
        base_rows.extend(b_ep)

        a_kinds = arm_reports_by_seed[seed].get("original_kind_accuracies") or {}
        b_kinds = baseline_reports_by_seed[seed].get("original_kind_accuracies") or {}
        for k in set(a_kinds) & set(b_kinds):
            if _is_finite_unit_float(a_kinds[k]) and _is_finite_unit_float(b_kinds[k]):
                max_kind_drop = max(max_kind_drop, float(b_kinds[k]) - float(a_kinds[k]))

    mean_gain = math.fsum(seed_diffs.values()) / len(seed_diffs) if seed_diffs else None
    all_positive = all(d > 0.0 for d in seed_diffs.values()) if seed_diffs else False
    contrast = paired_world_bootstrap_contrast(arm_rows, base_rows)
    return {
        "seed_differences": seed_diffs,
        "mean_primary_gain": mean_gain,
        "all_seeds_positive": all_positive,
        "max_original_kind_regression": max_kind_drop,
        "world_cluster_contrast": contrast,
    }


def evaluate_screening_gate(
    arm_reports_by_seed: dict[int, dict[str, Any]],
    baseline_reports_by_seed: dict[int, dict[str, Any]],
    *,
    primary_cohort: str = "factor-entity-permutation",
    extra_contrasts: dict[str, float] | None = None,
    required_seeds: tuple[int, ...] = SEEDS,
    min_dev_worlds_per_hop: int | None = None,
    expected_source_commit: str | None = None,
    enforce_holm_on_secondary: bool = True,
) -> dict[str, Any]:
    """Apply fail-closed Section 6 screening criteria across required seeds with world-cluster CI."""
    if not required_seeds:
        raise ValueError("required_seeds must be nonempty")
    min_dev_worlds = (
        SCREENING_GATE["min_dev_worlds_per_hop"]
        if min_dev_worlds_per_hop is None
        else min_dev_worlds_per_hop
    )
    if type(min_dev_worlds) is not int or min_dev_worlds < 1:
        raise ValueError("min_dev_worlds_per_hop must be a positive integer")

    secondary_contrasts_holm: dict[str, Any] | None = None
    if extra_contrasts:
        adj = holm_adjust(extra_contrasts)
        secondary_contrasts_holm = {
            "adjusted_p_values": adj,
            "alpha": 0.05,
            "enforced": enforce_holm_on_secondary,
            "holm_role": "secondary_exploratory_holm_adjusted",
        }

    reasons: list[str] = []
    missing_arm = [s for s in required_seeds if s not in arm_reports_by_seed]
    missing_base = [s for s in required_seeds if s not in baseline_reports_by_seed]
    if missing_arm:
        reasons.append(f"missing arm reports for seeds {missing_arm}")
    if missing_base:
        reasons.append(f"missing baseline reports for seeds {missing_base}")

    for seed in required_seeds:
        if seed not in arm_reports_by_seed or seed not in baseline_reports_by_seed:
            continue
        arm = arm_reports_by_seed[seed]
        base = baseline_reports_by_seed[seed]
        reasons.extend(
            _validate_screening_report(
                arm,
                label="arm",
                seed=seed,
                primary_cohort=primary_cohort,
                expected_source_commit=expected_source_commit,
            )
        )
        reasons.extend(
            _validate_screening_report(
                base,
                label="baseline",
                seed=seed,
                primary_cohort=primary_cohort,
                expected_source_commit=expected_source_commit,
            )
        )
        if ("source_commit" in arm or "source_commit" in base) and arm.get("source_commit") != base.get("source_commit"):
            reasons.append(f"seed {seed} arm/baseline source_commit mismatch")
        if (
            isinstance(arm.get("records"), dict)
            and isinstance(base.get("records"), dict)
            and primary_cohort in arm["records"]
            and primary_cohort in base["records"]
        ):
            arm_ep = arm["records"][primary_cohort].get("episodes")
            base_ep = base["records"][primary_cohort].get("episodes")
            reasons.extend(
                _screening_pairing_and_worlds(
                    arm_ep,
                    base_ep,
                    seed=seed,
                    min_dev_worlds=min_dev_worlds,
                )
            )

    common_seeds = sorted(set(arm_reports_by_seed) & set(baseline_reports_by_seed))
    stats = _compute_screening_statistics(
        arm_reports_by_seed,
        baseline_reports_by_seed,
        common_seeds,
        primary_cohort,
    )

    if reasons:
        return {
            "status": "INCOMPLETE_NOT_QUALIFIED",
            "reasons": reasons,
            "primary_cohort": primary_cohort,
            "required_seeds": list(required_seeds),
            "seeds": common_seeds,
            "min_dev_worlds_per_hop": min_dev_worlds,
            "holm_role": "secondary_exploratory_holm_adjusted",
            "secondary_contrasts_holm": secondary_contrasts_holm,
            "holm_adjusted_contrasts": secondary_contrasts_holm["adjusted_p_values"] if secondary_contrasts_holm else None,
            "mechanical_statistics": stats,
        }

    mean_gain = stats["mean_primary_gain"]
    all_positive = stats["all_seeds_positive"]
    max_kind_drop = stats["max_original_kind_regression"]
    contrast = stats["world_cluster_contrast"]
    ci_95 = contrast["ci_95"]
    ci_positive = bool(ci_95 is not None and ci_95[0] > 0.0)
    ci_negative = bool(ci_95 is not None and ci_95[1] < 0.0)
    kind_ok = max_kind_drop <= SCREENING_GATE["max_original_kind_regression"]

    if mean_gain >= SCREENING_GATE["min_mean_primary_gain"] and all_positive and kind_ok and ci_positive:
        status = "PASS_SCREENING"
    elif (
        (not kind_ok)
        or ci_negative
        or (ci_95 is not None and ci_95[1] < SCREENING_GATE["min_mean_primary_gain"] and mean_gain <= 0.0)
    ):
        status = "FALSIFIED"
    else:
        status = "INCONCLUSIVE"

    return {
        "status": status,
        "reasons": [],
        "primary_cohort": primary_cohort,
        "required_seeds": list(required_seeds),
        "seeds": common_seeds,
        "seed_differences": stats["seed_differences"],
        "mean_primary_gain": mean_gain,
        "all_seeds_positive": all_positive,
        "max_original_kind_regression": max_kind_drop,
        "world_cluster_contrast": contrast,
        "holm_role": "secondary_exploratory_holm_adjusted",
        "secondary_contrasts_holm": secondary_contrasts_holm,
        "holm_adjusted_contrasts": secondary_contrasts_holm["adjusted_p_values"] if secondary_contrasts_holm else None,
    }


def _recompute_cohort_from_episodes(
    name: str,
    rec: Any,
    *,
    expected_split: str,
    min_worlds: int,
) -> tuple[dict[str, Any] | None, set[str], list[str], list[str]]:
    integrity: list[str] = []
    size: list[str] = []
    identities: set[str] = set()
    if not isinstance(rec, dict):
        return None, identities, [f"cohort {name} record is not a dict"], size

    if rec.get("independent_rescore_status") != "PASS":
        integrity.append(f"cohort {name} lacks independent_rescore_status == 'PASS'")
    if "split" in rec and rec.get("split") != expected_split:
        integrity.append(f"cohort {name} split mismatch: {rec.get('split')!r}")
    if "protocol_sha256" in rec and rec.get("protocol_sha256") != PROTOCOL_SHA:
        integrity.append(f"cohort {name} protocol_sha256 mismatch")
    if "attribution_origin" in rec and rec.get("attribution_origin") != "model":
        integrity.append(f"cohort {name} attribution_origin mismatch: {rec.get('attribution_origin')!r}")
    if "decoder_schema" in rec and rec.get("decoder_schema") != "utf8-raw":
        integrity.append(f"cohort {name} decoder_schema mismatch: {rec.get('decoder_schema')!r}")

    episodes = rec.get("episodes")
    if not isinstance(episodes, (list, tuple)) or not episodes:
        integrity.append(f"cohort {name} has missing or empty episodes")
        return None, identities, integrity, size

    correct = 0
    invalid = 0
    worlds: set[str] = set()
    pairs: dict[str, list[bool]] = defaultdict(list)
    for idx, ep in enumerate(episodes):
        if not isinstance(ep, dict):
            integrity.append(f"cohort {name} episode {idx} is not a dict")
            continue
        wid = ep.get("world_id")
        outcome = ep.get("outcome")
        trace = ep.get("trace")
        if (
            not isinstance(wid, str)
            or not wid
            or not isinstance(outcome, dict)
            or type(outcome.get("correct")) is not bool
            or not isinstance(trace, dict)
            or not isinstance(trace.get("status"), str)
        ):
            integrity.append(f"cohort {name} episode {idx} missing valid world_id/outcome/trace")
            continue
        ep_identity = trace.get("model_identity")
        if not isinstance(ep_identity, str) or not ep_identity:
            integrity.append(f"cohort {name} episode {idx} missing bound model_identity")
        else:
            identities.add(ep_identity)
        if rec.get("model_identity") is not None and ep_identity != rec.get("model_identity"):
            integrity.append(f"cohort {name} episode {idx} model_identity disagrees with record")

        worlds.add(wid)
        if outcome["correct"]:
            correct += 1
        if trace["status"] == "invalid_action":
            invalid += 1
        if name in ("factor-answer-edge-swap", "factor-answer-query-swap"):
            pid = ep.get("pair_id")
            if not isinstance(pid, str) or not pid:
                integrity.append(f"cohort {name} episode {idx} missing pair_id")
            else:
                pairs[pid].append(outcome["correct"])

    world_count = len(worlds)
    if world_count < min_worlds:
        size.append(f"cohort {name} unique world count {world_count} below required {min_worlds}")

    pair_accuracy: float | None = None
    pair_count = 0
    if name in ("factor-answer-edge-swap", "factor-answer-query-swap"):
        incomplete = sum(1 for vals in pairs.values() if len(vals) != 2)
        if not pairs or incomplete > 0:
            integrity.append(f"cohort {name} has incomplete or missing paired evaluations ({incomplete} incomplete)")
        else:
            pair_count = len(pairs)
            both_correct = sum(1 for vals in pairs.values() if all(vals))
            pair_accuracy = both_correct / pair_count
            if pair_count < min_worlds:
                size.append(f"cohort {name} complete pair count {pair_count} below required {min_worlds}")

    recomputed = {
        "episodes": len(episodes),
        "world_count": world_count,
        "accuracy": correct / len(episodes),
        "invalid_rate": invalid / len(episodes),
        "pair_count": pair_count,
        "pair_accuracy": pair_accuracy,
    }
    return recomputed, identities, integrity, size


def evaluate_formation_gate(
    suite_eval: dict[str, Any],
    *,
    equivariance_report: dict[str, Any] | None = None,
    confirmation_worlds_per_cohort: int | None = None,
) -> dict[str, Any]:
    """Apply fail-closed Section 6 formation thresholds from full episode records."""
    del confirmation_worlds_per_cohort  # Untrusted summary override is deliberately ignored.
    min_worlds = FORMATION_GATE["min_confirmation_worlds_per_cohort"]
    integrity_reasons: list[str] = []
    size_reasons: list[str] = []

    if suite_eval.get("protocol_sha256") != PROTOCOL_SHA:
        integrity_reasons.append("suite_eval protocol_sha256 mismatch")
    split = suite_eval.get("split")
    if split not in ("confirmation", "validation"):
        integrity_reasons.append(f"invalid suite_eval split {split!r}")
    if suite_eval.get("attribution_origin") != "model":
        integrity_reasons.append(f"invalid attribution_origin {suite_eval.get('attribution_origin')!r}; expected 'model'")
    if suite_eval.get("decoder_schema") != "utf8-raw":
        integrity_reasons.append(f"invalid decoder_schema {suite_eval.get('decoder_schema')!r}; expected 'utf8-raw'")

    records = suite_eval.get("records")
    if not isinstance(records, dict):
        integrity_reasons.append("suite_eval missing records mapping")
        records = {}

    missing_cohorts = [c for c in REQUIRED_FORMATION_COHORTS if c not in records]
    if missing_cohorts:
        integrity_reasons.append(f"missing required formation cohorts: {missing_cohorts}")

    recomputed: dict[str, dict[str, Any]] = {}
    all_identities: set[str] = set()
    for name in REQUIRED_FORMATION_COHORTS:
        if name not in records:
            continue
        stats, ids, c_int, c_size = _recompute_cohort_from_episodes(
            name,
            records[name],
            expected_split=split if isinstance(split, str) else "confirmation",
            min_worlds=min_worlds,
        )
        integrity_reasons.extend(c_int)
        size_reasons.extend(c_size)
        all_identities.update(ids)
        if stats is not None:
            recomputed[name] = stats

    bound_identity: str | None = None
    if len(all_identities) != 1:
        integrity_reasons.append(f"formation gate requires exactly one bound model_identity, found {sorted(all_identities)}")
    else:
        bound_identity = next(iter(all_identities))
        if suite_eval.get("model_identity") is not None and suite_eval.get("model_identity") != bound_identity:
            integrity_reasons.append("suite_eval model_identity disagrees with episode trace identity")

    renaming_consistency: float | None = None
    if not isinstance(equivariance_report, dict):
        integrity_reasons.append("missing equivariance_report")
    else:
        eq_id = equivariance_report.get("model_identity")
        if not isinstance(eq_id, str) or not eq_id or (bound_identity is not None and eq_id != bound_identity):
            integrity_reasons.append("equivariance_report model_identity missing or disagrees with bound checkpoint")
        perm_rec = records.get("factor-entity-permutation")
        if isinstance(perm_rec, dict) and perm_rec.get("cases_sha256"):
            if equivariance_report.get("cases_sha256") != perm_rec["cases_sha256"]:
                integrity_reasons.append("equivariance_report cases_sha256 disagrees with factor-entity-permutation")
        eq_pairs = equivariance_report.get("complete_pairs")
        if type(eq_pairs) is not int or eq_pairs < 0:
            integrity_reasons.append("equivariance_report complete_pairs missing or invalid")
        elif eq_pairs < min_worlds:
            size_reasons.append(f"equivariance complete_pairs {eq_pairs} below required {min_worlds}")
        eq_rate = equivariance_report.get("argmax_equivariance_rate")
        if not _is_finite_unit_float(eq_rate):
            integrity_reasons.append("equivariance_report argmax_equivariance_rate missing or invalid")
        else:
            renaming_consistency = float(eq_rate)

    hop_accs = {
        h: recomputed[h]["accuracy"]
        for h in ("hop-1", "hop-2", "hop-3", "hop-4")
        if h in recomputed
    }
    stress_accs = {
        k: recomputed[k]["accuracy"]
        for k in REQUIRED_FORMATION_COHORTS
        if k.startswith("factor-") and k in recomputed
    }
    edge_pair_acc = recomputed.get("factor-answer-edge-swap", {}).get("pair_accuracy")
    qswap_pair_acc = recomputed.get("factor-answer-query-swap", {}).get("pair_accuracy")
    inv_rates = [m["invalid_rate"] for m in recomputed.values()]
    max_invalid = max(inv_rates) if inv_rates else 1.0
    cohort_world_counts = {k: m["world_count"] for k, m in recomputed.items()}

    if integrity_reasons:
        status = "NOT_QUALIFIED"
        reasons = integrity_reasons + size_reasons
    elif split == "validation":
        if size_reasons:
            status = "MECHANICAL_ONLY_NOT_FORMATION_QUALIFIED"
            reasons = size_reasons
        else:
            status = "NOT_QUALIFIED"
            reasons = ["formation gate requires split == 'confirmation', got 'validation'"]
    elif size_reasons:
        status = "MECHANICAL_ONLY_NOT_FORMATION_QUALIFIED"
        reasons = size_reasons
    else:
        thresholds_met = (
            len(hop_accs) == 4
            and all(a >= FORMATION_GATE["min_raw_accuracy_per_hop"] for a in hop_accs.values())
            and len(stress_accs) == 8
            and all(a >= FORMATION_GATE["min_stress_subgroup_accuracy"] for a in stress_accs.values())
            and edge_pair_acc is not None
            and edge_pair_acc >= FORMATION_GATE["min_answer_changing_pair_accuracy"]
            and qswap_pair_acc is not None
            and qswap_pair_acc >= FORMATION_GATE["min_answer_changing_pair_accuracy"]
            and renaming_consistency is not None
            and renaming_consistency >= FORMATION_GATE["min_renaming_consistency"]
            and max_invalid <= FORMATION_GATE["max_invalid_action_rate"]
        )
        status = "FORMATION_QUALIFIED" if thresholds_met else "FALSIFIED"
        reasons = [] if thresholds_met else ["formation thresholds not met on complete confirmation suite"]

    return {
        "status": status,
        "reasons": reasons,
        "split": split,
        "model_identity": bound_identity,
        "cohort_world_counts": cohort_world_counts,
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
