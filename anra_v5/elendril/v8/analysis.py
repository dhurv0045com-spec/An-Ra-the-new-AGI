"""V8 statistical comparisons, two-view cost accounting, advancement gates, and reporting."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

from anra_v5.elendril.codec import canonical, digest

from .evaluation import EpisodeRecord
from .protocol import ADVANCEMENT_RULES, SCENARIO_LEDGER


def paired_world_comparison(
    episodes_a: list[EpisodeRecord] | tuple[EpisodeRecord, ...],
    episodes_b: list[EpisodeRecord] | tuple[EpisodeRecord, ...],
    *,
    arm_a: str,
    arm_b: str,
    bootstrap_samples: int = 1000,
    seed: int = 202611,
) -> dict[str, object]:
    """Paired exact comparison clustered by base_world_id on identical world_id sets."""

    map_a = {ep.world_id: ep for ep in episodes_a}
    map_b = {ep.world_id: ep for ep in episodes_b}
    if set(map_a) != set(map_b):
        raise ValueError("paired comparison requires identical world_id evaluation sets")
    if not map_a:
        raise ValueError("paired comparison requires at least one world")

    clusters: dict[str, list[tuple[int, int]]] = defaultdict(list)
    both_correct = 0
    a_only = 0
    b_only = 0
    neither = 0
    diffs: list[float] = []

    for wid in sorted(map_a):
        ea = map_a[wid]
        eb = map_b[wid]
        ca = 1 if ea.outcome.correct else 0
        cb = 1 if eb.outcome.correct else 0
        diff = float(ca - cb)
        diffs.append(diff)
        clusters[ea.base_world_id].append((ca, cb))
        if ca and cb:
            both_correct += 1
        elif ca and not cb:
            a_only += 1
        elif cb and not ca:
            b_only += 1
        else:
            neither += 1

    mean_a = sum(1 for wid in map_a if map_a[wid].outcome.correct) / len(map_a)
    mean_b = sum(1 for wid in map_b if map_b[wid].outcome.correct) / len(map_b)
    mean_diff = mean_a - mean_b

    cluster_keys = sorted(clusters)
    cluster_diffs = [
        sum(ca - cb for ca, cb in clusters[cid]) / len(clusters[cid]) for cid in cluster_keys
    ]
    rng = random.Random(int(digest(["v8-cluster-bootstrap", arm_a, arm_b, seed]), 16))
    k = len(cluster_diffs)
    boot_means: list[float] = []
    for _ in range(bootstrap_samples):
        sample = [cluster_diffs[rng.randrange(k)] for _ in range(k)]
        boot_means.append(sum(sample) / k)
    boot_means.sort()
    ci_low = boot_means[int(0.025 * bootstrap_samples)]
    ci_high = boot_means[min(int(0.975 * bootstrap_samples), bootstrap_samples - 1)]

    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "worlds": len(map_a),
        "base_world_clusters": k,
        "accuracy_a": mean_a,
        "accuracy_b": mean_b,
        "mean_paired_difference": mean_diff,
        "bootstrap_95ci": [ci_low, ci_high],
        "discordant": {"both_correct": both_correct, "a_only": a_only, "b_only": b_only, "neither": neither},
    }


def build_cost_views(run_records: list[dict[str, object]]) -> dict[str, object]:
    """Build both mandatory V8 cost views: equal-exposure and equal-measured-work."""

    by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
    for rec in run_records:
        by_arm[str(rec["arm"])].append(rec)

    equal_exposure: dict[str, dict[str, object]] = {}
    equal_work: dict[str, dict[str, object]] = {}

    for arm, recs in sorted(by_arm.items()):
        r09_raw = [float(r["raw_r09_accuracy"]) for r in recs if r.get("raw_r09_accuracy") is not None]
        raw_all = [float(r["raw_overall_accuracy"]) for r in recs if r.get("raw_overall_accuracy") is not None]
        updates = [int(r.get("completed_updates", 0)) for r in recs]
        wall_s = [float(r.get("wall_seconds", 0.0)) for r in recs]
        eval_pos = [int(r.get("eval_evaluated_positions", 0)) for r in recs]
        calls = [int(r.get("eval_model_calls", 0)) for r in recs]
        params = [int(r.get("total_parameters", 0)) for r in recs]

        mean_wall = sum(wall_s) / len(recs) if recs else 0.0
        mean_calls = sum(calls) / len(recs) if recs else 0.0
        mean_pos = sum(eval_pos) / len(recs) if recs else 0.0
        mean_r09 = sum(r09_raw) / len(r09_raw) if r09_raw else None
        mean_overall = sum(raw_all) / len(raw_all) if raw_all else None

        equal_exposure[arm] = {
            "runs": len(recs),
            "mean_updates": sum(updates) / len(recs) if recs else 0.0,
            "mean_raw_r09_accuracy": mean_r09,
            "mean_raw_overall_accuracy": mean_overall,
            "total_parameters": params[0] if params else None,
        }
        equal_work[arm] = {
            "runs": len(recs),
            "mean_wall_seconds": mean_wall,
            "mean_eval_model_calls": mean_calls,
            "mean_eval_evaluated_positions": mean_pos,
            "r09_accuracy_per_minute": ((mean_r09 / (mean_wall / 60.0)) if (mean_r09 is not None and mean_wall > 0.0) else None),
            "overall_accuracy_per_1k_eval_positions": (
                (mean_overall / (mean_pos / 1000.0)) if (mean_overall is not None and mean_pos > 0.0) else None
            ),
        }

    return {
        "equal_exposure_view": equal_exposure,
        "equal_measured_work_view": equal_work,
    }


def _first_float(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        val = mapping.get(key)
        if val is not None:
            return float(val)
    return None


def _extract_numeric_series(val: object) -> list[float]:
    if val is None:
        return []
    if isinstance(val, dict):
        return [float(v) for _, v in sorted(val.items(), key=lambda kv: str(kv[0])) if v is not None]
    if isinstance(val, (list, tuple)):
        return [float(v) for v in val if v is not None]
    if isinstance(val, (int, float)):
        return [float(val)]
    return []


def _extract_canonical_raw(mapping: dict[str, Any]) -> float | None:
    direct = _first_float(
        mapping,
        "canonical_raw_accuracy",
        "candidate_mean_canonical_raw",
        "mean_canonical_raw",
        "raw_canonical_accuracy",
        "F01_canonical_exact",
    )
    if direct is not None:
        return direct
    ann = mapping.get("annotations")
    if isinstance(ann, dict) and ann.get("F01_canonical_exact") is not None:
        return float(ann["F01_canonical_exact"])
    summary_by_mode = mapping.get("summary_by_mode")
    if isinstance(summary_by_mode, dict):
        raw_sum = summary_by_mode.get("raw")
        if isinstance(raw_sum, dict):
            raw_ann = raw_sum.get("annotations")
            if isinstance(raw_ann, dict) and raw_ann.get("F01_canonical_exact") is not None:
                return float(raw_ann["F01_canonical_exact"])
    return None


def pair_condition_runs(
    runs: Sequence[dict[str, object]],
    *,
    candidate_condition: str,
    comparator_condition: str,
    expected_seed_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Deterministically pair candidate and comparator runs by (init_seed, order_seed) with strict parity checks."""

    if candidate_condition == comparator_condition:
        raise ValueError("candidate_condition and comparator_condition must be distinct")

    indexed: dict[tuple[str, int, int], dict[str, object]] = {}
    for run in runs:
        cond_val = run.get("condition_id")
        if cond_val:
            cond = str(cond_val)
        elif "arm" in run and "dataset" in run:
            cond = f"{run['arm']}-{run['dataset']}"
        else:
            raise ValueError("each run must include condition_id or both arm and dataset")

        if "init_seed" not in run or "order_seed" not in run:
            raise ValueError(f"run for condition {cond!r} missing init_seed or order_seed")
        init_seed = int(run["init_seed"])  # type: ignore[arg-type]
        order_seed = int(run["order_seed"])  # type: ignore[arg-type]

        src = run.get("source_commit")
        if not isinstance(src, str) or not src:
            raise ValueError(f"run ({cond!r}, {init_seed}, {order_seed}) missing source_commit")
        if "data_identity" not in run or run.get("data_identity") is None:
            raise ValueError(f"run ({cond!r}, {init_seed}, {order_seed}) missing data_identity")

        key = (cond, init_seed, order_seed)
        if key in indexed:
            raise ValueError(f"duplicate run for (condition_id, init_seed, order_seed)={key!r}")
        indexed[key] = run

    cand_runs = {(i, o): r for (c, i, o), r in indexed.items() if c == candidate_condition}
    comp_runs = {(i, o): r for (c, i, o), r in indexed.items() if c == comparator_condition}

    if not cand_runs:
        raise ValueError(f"no runs found for candidate_condition={candidate_condition!r}")
    if not comp_runs:
        raise ValueError(f"no runs found for comparator_condition={comparator_condition!r}")
    if set(cand_runs) != set(comp_runs):
        missing_in_cand = sorted(set(comp_runs) - set(cand_runs))
        missing_in_comp = sorted(set(cand_runs) - set(comp_runs))
        raise ValueError(
            f"seed parity mismatch between {candidate_condition!r} and {comparator_condition!r}: "
            f"missing_in_candidate={missing_in_cand}, missing_in_comparator={missing_in_comp}"
        )

    if expected_seed_pairs is not None:
        expected_set = {(int(i), int(o)) for i, o in expected_seed_pairs}
        if set(cand_runs) != expected_set:
            raise ValueError(
                f"matched seed pairs {sorted(cand_runs)} != expected_seed_pairs {sorted(expected_set)}"
            )

    matched_runs = list(cand_runs.values()) + list(comp_runs.values())
    commits = {str(r["source_commit"]) for r in matched_runs}
    if len(commits) != 1:
        raise ValueError(f"source_commit mismatch across paired runs: {sorted(commits)}")
    split_audits = {r.get("split_audit_sha256") for r in matched_runs}
    if len(split_audits) != 1:
        raise ValueError(f"split_audit_sha256 mismatch across paired runs: {sorted(str(x) for x in split_audits)}")

    sorted_seeds = sorted(cand_runs.keys())
    pairs: list[dict[str, Any]] = []
    gains: list[float] = []
    cand_r09s: list[float] = []
    comp_r09s: list[float] = []
    r09_deltas: list[float] = []
    cand_canons: list[float] = []
    comp_canons: list[float] = []
    canon_deltas: list[float] = []

    for init_seed, order_seed in sorted_seeds:
        ra = cand_runs[(init_seed, order_seed)]
        rb = comp_runs[(init_seed, order_seed)]
        cand_ov = _first_float(ra, "raw_overall_accuracy", "overall_accuracy", "exact_episode_accuracy", "accuracy")
        comp_ov = _first_float(rb, "raw_overall_accuracy", "overall_accuracy", "exact_episode_accuracy", "accuracy")
        gain = (cand_ov - comp_ov) if (cand_ov is not None and comp_ov is not None) else 0.0
        gains.append(gain)

        cand_r09 = _first_float(ra, "raw_r09_accuracy", "candidate_raw_r09")
        comp_r09 = _first_float(rb, "raw_r09_accuracy", "comparator_raw_r09")
        r09_delta = (cand_r09 - comp_r09) if (cand_r09 is not None and comp_r09 is not None) else None
        if cand_r09 is not None:
            cand_r09s.append(cand_r09)
        if comp_r09 is not None:
            comp_r09s.append(comp_r09)
        if r09_delta is not None:
            r09_deltas.append(r09_delta)

        cand_canon = _extract_canonical_raw(ra)
        comp_canon = _extract_canonical_raw(rb)
        canon_delta = (cand_canon - comp_canon) if (cand_canon is not None and comp_canon is not None) else None
        if cand_canon is not None:
            cand_canons.append(cand_canon)
        if comp_canon is not None:
            comp_canons.append(comp_canon)
        if canon_delta is not None:
            canon_deltas.append(canon_delta)

        pairs.append(
            {
                "init_seed": init_seed,
                "order_seed": order_seed,
                "candidate_job_id": ra.get("job_id"),
                "comparator_job_id": rb.get("job_id"),
                "candidate_condition": candidate_condition,
                "comparator_condition": comparator_condition,
                "candidate_raw_overall": cand_ov,
                "comparator_raw_overall": comp_ov,
                "paired_gain": gain,
                "delta": gain,
                "candidate_raw_r09": cand_r09,
                "comparator_raw_r09": comp_r09,
                "delta_raw_r09": r09_delta,
                "candidate_relevant_cf_both_correct": _first_float(ra, "relevant_cf_both_correct"),
                "candidate_irrelevant_cf_stability": _first_float(ra, "irrelevant_cf_stability"),
                "candidate_canonical_raw": cand_canon,
                "comparator_canonical_raw": comp_canon,
                "canonical_raw_delta": canon_delta,
                "candidate_data_identity": ra.get("data_identity"),
                "comparator_data_identity": rb.get("data_identity"),
                "candidate_checkpoint_sha256": ra.get("checkpoint_sha256"),
                "comparator_checkpoint_sha256": rb.get("checkpoint_sha256"),
            }
        )

    mean_gain = sum(gains) / len(gains)
    return {
        "schema": "elendril-v8-paired-runs-v1",
        "candidate_condition": candidate_condition,
        "comparator_condition": comparator_condition,
        "source_commit": next(iter(commits)),
        "split_audit_sha256": next(iter(split_audits)),
        "seed_pairs": [[i, o] for i, o in sorted_seeds],
        "seed_count": len(sorted_seeds),
        "pairs": pairs,
        "mean_paired_gain": mean_gain,
        "mean_delta": mean_gain,
        "min_paired_gain": min(gains),
        "max_paired_gain": max(gains),
        "positive_seed_differences": sum(1 for g in gains if g > 0.0),
        "candidate_mean_raw_r09": (sum(cand_r09s) / len(cand_r09s)) if cand_r09s else None,
        "comparator_mean_raw_r09": (sum(comp_r09s) / len(comp_r09s)) if comp_r09s else None,
        "mean_delta_raw_r09": (sum(r09_deltas) / len(r09_deltas)) if r09_deltas else None,
        "candidate_mean_canonical_raw": (sum(cand_canons) / len(cand_canons)) if cand_canons else None,
        "comparator_mean_canonical_raw": (sum(comp_canons) / len(comp_canons)) if comp_canons else None,
        "mean_canonical_raw_delta": (sum(canon_deltas) / len(canon_deltas)) if canon_deltas else None,
    }


def evaluate_advancement_gate(
    summary: list[dict[str, object]] | tuple[dict[str, object], ...] | dict[str, object],
    *,
    baseline_summary: dict[str, object] | None = None,
    confirmation: bool = False,
) -> dict[str, object]:
    """Evaluate fixed-rule development/confirmation advancement gates without post-hoc thresholds."""

    rules = ADVANCEMENT_RULES["confirmation_candidate" if confirmation else "development_candidate"]
    reasons: list[str] = []

    pairs: list[dict[str, Any]] | None = None
    active_dict: dict[str, Any] = {}
    if isinstance(summary, dict):
        active_dict = dict(summary)
        if confirmation:
            split_val = active_dict.get("split_name") or active_dict.get("split")
            if split_val is not None and str(split_val).lower() not in ("confirmation", "conf"):
                reasons.append(f"confirmation gate requires confirmation split results, got {split_val!r}")
            if "is_confirmation" in active_dict and not active_dict["is_confirmation"]:
                reasons.append("confirmation gate rejected non-confirmation summary (is_confirmation=False)")
            phase_val = active_dict.get("phase")
            if phase_val is not None and str(phase_val).lower() not in ("replication", "confirmation"):
                reasons.append(f"confirmation gate requires replication/confirmation phase, got {phase_val!r}")
            if "confirmation_summary" in active_dict:
                conf_sub = active_dict["confirmation_summary"]
                if conf_sub is None:
                    reasons.append("confirmation_summary is required when confirmation=True")
                elif isinstance(conf_sub, dict):
                    active_dict = {**active_dict, **conf_sub}

        raw_pairs = active_dict.get("pairs") or active_dict.get("seed_pairs")
        if isinstance(raw_pairs, (list, tuple)):
            pairs = [dict(p) for p in raw_pairs]
    else:
        pairs = [dict(p) for p in summary]

    if pairs is not None:
        if not pairs:
            return {"qualified": False, "reasons": ["no paired seed runs recorded"], "rules": rules}
        if confirmation:
            for sp in pairs:
                sp_split = sp.get("split_name") or sp.get("split")
                if sp_split is not None and str(sp_split).lower() not in ("confirmation", "conf"):
                    reasons.append(f"confirmation gate requires confirmation split results, got {sp_split!r}")
                    break

        gains = [float(sp.get("paired_gain", sp.get("delta", 0.0))) for sp in pairs]
        r09_vals = [float(sp["candidate_raw_r09"]) for sp in pairs if sp.get("candidate_raw_r09") is not None]
        rel_vals = [
            float(sp["candidate_relevant_cf_both_correct"])
            for sp in pairs
            if sp.get("candidate_relevant_cf_both_correct") is not None
        ]
        irrel_vals = [
            float(sp["candidate_irrelevant_cf_stability"])
            for sp in pairs
            if sp.get("candidate_irrelevant_cf_stability") is not None
        ]
        seed_count = len(pairs)
        pos_diffs = sum(1 for g in gains if g > 0.0)
        mean_gain = sum(gains) / seed_count
        min_gain = min(gains)
        mean_r09 = (sum(r09_vals) / len(r09_vals)) if r09_vals else 0.0
        min_r09 = min(r09_vals) if r09_vals else 0.0
        min_rel_cf = min(rel_vals) if rel_vals else 0.0
        min_irrel_cf = min(irrel_vals) if irrel_vals else 0.0

        canon_deltas = [
            float(sp["canonical_raw_delta"])
            for sp in pairs
            if sp.get("canonical_raw_delta") is not None
        ]
        if canon_deltas:
            canon_regression: float | None = -(sum(canon_deltas) / len(canon_deltas))
        else:
            cand_c = _extract_canonical_raw(active_dict) if active_dict else None
            base_c = _extract_canonical_raw(baseline_summary) if baseline_summary else None
            canon_regression = (base_c - cand_c) if (cand_c is not None and base_c is not None) else None
    else:
        gains = _extract_numeric_series(
            active_dict.get("confirmation_paired_gains") if confirmation and "confirmation_paired_gains" in active_dict
            else (
                active_dict.get("paired_gains")
                or active_dict.get("paired_gains_by_seed")
                or active_dict.get("mean_paired_gain")
                or active_dict.get("paired_improvement_over_comparator")
            )
        )
        r09_vals = _extract_numeric_series(
            active_dict.get("confirmation_r09_by_seed") if confirmation and "confirmation_r09_by_seed" in active_dict
            else (
                active_dict.get("r09_by_seed")
                or active_dict.get("raw_one_hop_by_seed")
                or active_dict.get("raw_one_hop_mean")
                or active_dict.get("candidate_mean_raw_r09")
                or active_dict.get("mean_raw_r09")
                or active_dict.get("raw_r09_accuracy")
            )
        )
        rel_vals = _extract_numeric_series(
            active_dict.get("relevant_cf_by_seed")
            or active_dict.get("relevant_cf_both_correct")
            or active_dict.get("relevant_counterfactual_both_correct")
            or active_dict.get("min_relevant_cf_both_correct")
        )
        irrel_vals = _extract_numeric_series(
            active_dict.get("irrelevant_cf_by_seed")
            or active_dict.get("irrelevant_cf_stability")
            or active_dict.get("irrelevant_counterfactual_stability")
            or active_dict.get("min_irrelevant_cf_stability")
        )
        seed_count = int(active_dict.get("seed_count") or max(len(gains), len(r09_vals), 0))
        if seed_count == 0:
            return {"qualified": False, "reasons": ["no paired seed runs recorded"], "rules": rules}

        pos_diffs = int(
            active_dict["positive_seed_differences"]
            if active_dict.get("positive_seed_differences") is not None
            else sum(1 for g in gains if g > 0.0)
        )
        mean_gain = (sum(gains) / len(gains)) if gains else 0.0
        min_gain = min(gains) if gains else 0.0
        mean_r09 = (sum(r09_vals) / len(r09_vals)) if r09_vals else 0.0
        min_r09 = min(r09_vals) if r09_vals else 0.0
        min_rel_cf = min(rel_vals) if rel_vals else 0.0
        min_irrel_cf = min(irrel_vals) if irrel_vals else 0.0

        direct_drop = _first_float(active_dict, "canonical_raw_regression", "canonical_raw_drop")
        if direct_drop is not None:
            canon_regression = direct_drop
        else:
            cand_c = _extract_canonical_raw(active_dict)
            base_c = (
                _extract_canonical_raw(baseline_summary)
                if baseline_summary is not None
                else _first_float(active_dict, "baseline_canonical_raw_accuracy", "comparator_mean_canonical_raw")
            )
            canon_regression = (base_c - cand_c) if (cand_c is not None and base_c is not None) else None

    if seed_count < int(rules["seed_count"]):
        reasons.append(f"seed_count {seed_count} < {rules['seed_count']}")
    if pos_diffs < int(rules["positive_seed_differences_min"]):
        reasons.append(f"positive_seed_differences {pos_diffs} < {rules['positive_seed_differences_min']}")
    gain_floor = float(rules["mean_gain_min" if confirmation else "paired_improvement_over_comparator_min"])
    if mean_gain < gain_floor:
        reasons.append(f"mean_paired_gain {mean_gain:.4f} < {gain_floor:.4f}")
    if not confirmation and min_gain < float(rules["single_seed_regression_floor"]):
        reasons.append(f"single_seed_regression {min_gain:.4f} < {rules['single_seed_regression_floor']}")
    if mean_r09 < float(rules["raw_one_hop_exact_episode_accuracy_min"]):
        reasons.append(f"mean_raw_r09 {mean_r09:.4f} < {rules['raw_one_hop_exact_episode_accuracy_min']}")
    if "canonical_raw_regression_max" in rules and canon_regression is not None:
        max_drop = float(rules["canonical_raw_regression_max"])
        if canon_regression > max_drop + 1e-12:
            reasons.append(f"canonical_raw_regression {canon_regression:.4f} > {max_drop:.4f}")
    if min_rel_cf < float(rules["relevant_counterfactual_both_correct_min"]):
        reasons.append(
            f"min_relevant_cf_both_correct {min_rel_cf:.4f} < {rules['relevant_counterfactual_both_correct_min']}"
        )
    if min_irrel_cf < float(rules["irrelevant_counterfactual_stability_min"]):
        reasons.append(
            f"min_irrelevant_cf_stability {min_irrel_cf:.4f} < {rules['irrelevant_counterfactual_stability_min']}"
        )

    return {
        "qualified": len(reasons) == 0,
        "reasons": reasons,
        "summary": {
            "seed_count": seed_count,
            "positive_seed_differences": pos_diffs,
            "mean_paired_gain": mean_gain,
            "min_single_seed_gain": min_gain,
            "mean_raw_r09": mean_r09,
            "min_raw_r09": min_r09,
            "canonical_raw_regression": canon_regression,
            "min_relevant_cf_both_correct": min_rel_cf,
            "min_irrelevant_cf_stability": min_irrel_cf,
        },
        "rules": rules,
    }


def load_v717_archive_baseline(repo_root: Path) -> dict[str, object]:
    """Read audited local V7.17 archive facts while explicitly reporting unavailable weights."""

    metrics_path = repo_root / "Experiment Result" / "V7.17" / "f443d3f4141b4163bbc130d46ebd7395" / "METRICS.json"
    if not metrics_path.exists():
        return {"status": "ARCHIVE_MISSING", "weights_status": "PARENT_WEIGHTS_UNAVAILABLE"}
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "status": "ARCHIVE_LOADED_WEIGHTS_UNAVAILABLE",
        "weights_status": "PARENT_WEIGHTS_UNAVAILABLE",
        "metrics_path": str(metrics_path),
        "protocol_sha256": data.get("protocol_sha256"),
        "diagnosis": data.get("diagnosis"),
        "runs": data.get("runs"),
    }


def updated_scenario_ledger(executed_scenarios: set[str], qualified_gates: dict[str, bool] | None = None) -> list[dict[str, object]]:
    """Return a clean scenario ledger updating only scenarios actually run in this artifact."""

    gates = qualified_gates or {}
    out: list[dict[str, object]] = []
    for row in SCENARIO_LEDGER:
        sid = row["id"]
        status = row["status"]
        if sid in executed_scenarios:
            status = "EXECUTED"
        elif gates.get(sid) is True:
            status = "IMPLEMENTED_NOT_RUN"
        out.append({**row, "status": status})
    return out


def build_v8_report_payload(
    *,
    phase: str,
    status: str,
    source_commit: str,
    run_records: list[dict[str, object]],
    paired_comparisons: list[dict[str, object]],
    advancement: dict[str, object] | None,
    executed_scenarios: set[str],
    archive_baseline: dict[str, object] | None = None,
    notes: list[str] | None = None,
) -> tuple[dict[str, object], str]:
    """Produce structured METRICS.json and human-readable RESULTS.md."""

    costs = build_cost_views(run_records)
    ledger = updated_scenario_ledger(executed_scenarios)
    metrics = {
        "schema": "elendril-v8-report-v1",
        "phase": phase,
        "status": status,
        "source_commit": source_commit,
        "archive_baseline": archive_baseline,
        "runs": run_records,
        "paired_comparisons": paired_comparisons,
        "cost_views": costs,
        "advancement": advancement,
        "scenario_ledger": ledger,
        "notes": notes or [],
    }

    lines = [
        f"# ELENDRIL V8 Campaign Report — Phase `{phase}`",
        "",
        f"- **Status:** `{status}`",
        f"- **Source Commit:** `{source_commit}`",
        "",
        "## Equal-Exposure View",
        "",
        "| Arm | Runs | Updates | Mean Raw R09 | Mean Raw Overall | Params |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, row in sorted(costs["equal_exposure_view"].items()):
        r09 = f"{row['mean_raw_r09_accuracy']:.3f}" if row["mean_raw_r09_accuracy"] is not None else "n/a"
        ov = f"{row['mean_raw_overall_accuracy']:.3f}" if row["mean_raw_overall_accuracy"] is not None else "n/a"
        lines.append(f"| {arm} | {row['runs']} | {row['mean_updates']:.0f} | {r09} | {ov} | {row['total_parameters']} |")

    lines.extend([
        "",
        "## Equal-Measured-Work View",
        "",
        "| Arm | Mean Wall (s) | Mean Eval Calls | Mean Eval Positions | R09 Acc / Min |",
        "|---|---:|---:|---:|---:|",
    ])
    for arm, row in sorted(costs["equal_measured_work_view"].items()):
        rpm = f"{row['r09_accuracy_per_minute']:.3f}" if row["r09_accuracy_per_minute"] is not None else "n/a"
        lines.append(
            f"| {arm} | {row['mean_wall_seconds']:.1f} | {row['mean_eval_model_calls']:.1f} | {row['mean_eval_evaluated_positions']:.0f} | {rpm} |"
        )

    if paired_comparisons:
        lines.extend(["", "## Paired World Comparisons", ""])
        for comp in paired_comparisons:
            lo, hi = comp["bootstrap_95ci"]
            lines.append(
                f"- **{comp['arm_a']} vs {comp['arm_b']}** ({comp['worlds']} worlds): "
                f"{comp['accuracy_a']:.3f} vs {comp['accuracy_b']:.3f}, "
                f"diff `{comp['mean_paired_difference']:+.3f}` (95% cluster CI `[{lo:+.3f}, {hi:+.3f}]`)"
            )

    if advancement is not None:
        lines.extend(["", "## Advancement Decision", "", f"- **Qualified:** `{advancement['qualified']}`"])
        for reason in advancement.get("reasons", []):
            lines.append(f"  - Gate failed: {reason}")

    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")

    return metrics, "\n".join(lines) + "\n"


def write_v8_result_template(root: Path, *, source_commit: str) -> dict[str, Path]:
    """Create the pre-run Experiment Result/V8/ template files cleanly without fabricated metrics."""

    v8_dir = root / "Experiment Result" / "V8"
    v8_dir.mkdir(parents=True, exist_ok=True)
    readme_path = v8_dir / "README.md"
    template_metrics_path = v8_dir / "TEMPLATE_METRICS.json"

    readme = """# ELENDRIL V8 Experiment Results (Pending Owner Kaggle Execution)

This directory stores curated V8 campaign artifacts imported via `python -m tools.elendril_v8_export extract-verified` after owner-enabled Kaggle 2xT4 runs complete.

- No GPU research runs were executed locally before owner handoff.
- Expected curated files per completed campaign archive (`ELENDRIL_V8_ANALYSIS.zip`):
  - `<campaign_digest>/FINAL_STATUS.json`
  - `<campaign_digest>/RESULTS.md`
  - `<campaign_digest>/METRICS.json`
"""
    template_metrics = {
        "schema": "elendril-v8-report-v1",
        "phase": "TEMPLATE_NOT_EXECUTED",
        "status": "NOT_RUN",
        "source_commit": source_commit,
        "runs": [],
        "paired_comparisons": [],
        "cost_views": {"equal_exposure_view": {}, "equal_measured_work_view": {}},
        "advancement": None,
        "scenario_ledger": list(SCENARIO_LEDGER),
        "notes": ["Pre-run schema template only; no Kaggle execution metrics populated."],
    }
    readme_path.write_text(readme, encoding="utf-8")
    template_metrics_path.write_text(json.dumps(template_metrics, indent=2) + "\n", encoding="utf-8")
    return {"readme": readme_path, "template_metrics": template_metrics_path}


__all__ = [
    "build_cost_views",
    "build_v8_report_payload",
    "evaluate_advancement_gate",
    "load_v717_archive_baseline",
    "pair_condition_runs",
    "paired_world_comparison",
    "updated_scenario_ledger",
    "write_v8_result_template",
]
