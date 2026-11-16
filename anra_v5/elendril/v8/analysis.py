"""V8 statistical comparisons, two-view cost accounting, advancement gates, and reporting."""

from __future__ import annotations

from collections import defaultdict
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


def evaluate_advancement_gate(
    seed_pairs: list[dict[str, object]],
    *,
    confirmation: bool = False,
) -> dict[str, object]:
    """Evaluate fixed-rule development/confirmation advancement gates without post-hoc thresholds."""

    rules = ADVANCEMENT_RULES["confirmation_candidate" if confirmation else "development_candidate"]
    if not seed_pairs:
        return {"qualified": False, "reasons": ["no paired seed runs recorded"], "rules": rules}

    reasons: list[str] = []
    pos_diffs = sum(1 for sp in seed_pairs if float(sp["paired_gain"]) > 0.0)
    mean_gain = sum(float(sp["paired_gain"]) for sp in seed_pairs) / len(seed_pairs)
    min_gain = min(float(sp["paired_gain"]) for sp in seed_pairs)
    min_r09 = min(float(sp["candidate_raw_r09"]) for sp in seed_pairs)
    min_rel_cf = min(float(sp["candidate_relevant_cf_both_correct"]) for sp in seed_pairs)
    min_irrel_cf = min(float(sp["candidate_irrelevant_cf_stability"]) for sp in seed_pairs)

    if len(seed_pairs) < int(rules["seed_count"]):
        reasons.append(f"seed_count {len(seed_pairs)} < {rules['seed_count']}")
    if pos_diffs < int(rules["positive_seed_differences_min"]):
        reasons.append(f"positive_seed_differences {pos_diffs} < {rules['positive_seed_differences_min']}")
    gain_floor = float(rules["mean_gain_min" if confirmation else "paired_improvement_over_comparator_min"])
    if mean_gain < gain_floor:
        reasons.append(f"mean_paired_gain {mean_gain:.4f} < {gain_floor:.4f}")
    if not confirmation and min_gain < float(rules["single_seed_regression_floor"]):
        reasons.append(f"single_seed_regression {min_gain:.4f} < {rules['single_seed_regression_floor']}")
    if min_r09 < float(rules["raw_one_hop_exact_episode_accuracy_min"]):
        reasons.append(f"min_raw_r09 {min_r09:.4f} < {rules['raw_one_hop_exact_episode_accuracy_min']}")
    if min_rel_cf < float(rules["relevant_counterfactual_both_correct_min"]):
        reasons.append(f"min_relevant_cf_both_correct {min_rel_cf:.4f} < {rules['relevant_counterfactual_both_correct_min']}")
    if min_irrel_cf < float(rules["irrelevant_counterfactual_stability_min"]):
        reasons.append(f"min_irrelevant_cf_stability {min_irrel_cf:.4f} < {rules['irrelevant_counterfactual_stability_min']}")

    return {
        "qualified": len(reasons) == 0,
        "reasons": reasons,
        "summary": {
            "seed_count": len(seed_pairs),
            "positive_seed_differences": pos_diffs,
            "mean_paired_gain": mean_gain,
            "min_single_seed_gain": min_gain,
            "min_raw_r09": min_r09,
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
    "paired_world_comparison",
    "updated_scenario_ledger",
    "write_v8_result_template",
]
