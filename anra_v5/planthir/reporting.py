"""Level-6 campaign reporting, analyst first-page synthesis, archive rescoring & live dashboard."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

from .archives import _stream_file_sha256, safe_extract_planthir_archive


def build_verification_matrix(
    *,
    byte_integrity: str = "PASS",
    oracle_fixture_correctness: str = "PASS",
    recorded_action_replay: str = "PASS",
    checkpoint_restore: str = "MISSING_CHECKPOINT_ARTIFACT",
    regenerated_per_case_agreement: str = "BLOCKED_BY_MISSING_CHECKPOINT",
    training_reproduction: str = "NOT_RUN",
) -> list[dict[str, str]]:
    """Return the canonical 6-row Planthir verification matrix."""
    return [
        {"check": "byte_integrity", "status": str(byte_integrity)},
        {"check": "oracle_fixture_correctness", "status": str(oracle_fixture_correctness)},
        {"check": "recorded_action_replay", "status": str(recorded_action_replay)},
        {"check": "checkpoint_restore", "status": str(checkpoint_restore)},
        {"check": "regenerated_per_case_agreement", "status": str(regenerated_per_case_agreement)},
        {"check": "training_reproduction", "status": str(training_reproduction)},
    ]


def render_campaign_results_markdown(
    *,
    title: str,
    campaign_id: str,
    source_commit: str,
    status: str,
    verification_matrix: Sequence[Mapping[str, Any]],
    stage_summaries: Mapping[str, Any] | None = None,
    first_page_report: Mapping[str, Any] | None = None,
) -> str:
    """Render a standardized `RESULTS.md` report with verification matrix and optional analyst first page."""
    if first_page_report is not None and isinstance(first_page_report.get("markdown"), str):
        base_md = str(first_page_report["markdown"]).rstrip()
        lines = [base_md, "", "## Verification Matrix", "", "| Check | Status |", "|---|---|"]
    else:
        lines = [
            f"# {title}",
            "",
            f"- **Campaign ID:** `{campaign_id}`",
            f"- **Source commit:** `{source_commit}`",
            f"- **Status:** `{status}`",
            "",
            "## Verification Matrix",
            "",
            "| Check | Status |",
            "|---|---|",
        ]
    for row in verification_matrix:
        lines.append(f"| `{row.get('check', '')}` | `{row.get('status', '')}` |")
    if stage_summaries:
        lines.extend(["", "## Stage Summaries", ""])
        for sid, info in sorted(stage_summaries.items()):
            lines.append(f"- `{sid}`: `{json.dumps(info, sort_keys=True)}`")
    lines.append("")
    return "\n".join(lines)


def build_analyst_first_page_report(
    *,
    title: str,
    campaign_id: str,
    source_commit: str,
    decision_summary: str,
    control_arm: str,
    treatment_arm: str,
    paired_seed_bundle: Mapping[str, Any],
    paired_episodes_by_seed: Mapping[int | str, Mapping[str, Any]] | None = None,
    predeclared_slices: Sequence[Mapping[str, Any]] = (),
    interventions: Mapping[str, Mapping[str, Any]] | None = None,
    retrospective_findings: Sequence[str] = (),
    next_experiment: Mapping[str, Any] | None = None,
    analysis_archive: Mapping[str, Any] | None = None,
    full_archive: Mapping[str, Any] | None = None,
    reproduction_command: str = "",
) -> dict[str, Any]:
    """Build an analyst-first report & JSON summary with artifact routing, per-seed pairing, coverage, slices, and trace exemplars."""

    ep_map = {str(k): dict(v) for k, v in dict(paired_episodes_by_seed or {}).items()}
    aa = dict(analysis_archive or {})
    fa = dict(full_archive or {})
    repro = str(reproduction_command or f"python -m anra_v5.planthir rescore-archive --archive {aa.get('name', 'ANALYSIS.zip')} --out rebuilt_report")

    lines = [
        f"# {title}",
        "",
        "## Artifacts, Verification & One-Command Rebuild",
        "",
        f"- **Campaign ID:** `{campaign_id}` · **Source commit:** `{source_commit}`",
        f"- **Answers episode, slice & disagreement questions (compact analysis archive):** `{aa.get('name', 'ANALYSIS.zip')}` (SHA-256 `{aa.get('sha256', 'N/A')}`, verified=`{aa.get('verified', False)}`)",
        f"- **Required for neural probabilities, linear probes & new lesions (full checkpoint archive):** `{fa.get('name', 'FULL.zip')}` (SHA-256 `{fa.get('sha256', 'N/A')}`, verified=`{fa.get('verified', False)}`)",
        f"- **Rebuild report & curated metrics from compact archive in a fresh directory:** `{repro}`",
        "",
        "## Registered Decision & Per-Seed Comparison",
        "",
        f"**Decision.** {decision_summary}",
        "",
        f"| Seed | {control_arm} | {treatment_arm} | {treatment_arm} - {control_arm} | {treatment_arm}-only solved | {control_arm}-only solved | First-op disagreements |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for obs in paired_seed_bundle.get("observations", ()):
        seed_s = str(obs.get("init_seed", ""))
        c_val = float(obs["control_value"])
        t_val = float(obs["treatment_value"])
        ep = ep_map.get(seed_s, {})
        t_only = ep.get("treatment_only_solved", "N/A")
        c_only = ep.get("control_only_solved", "N/A")
        op_dis = ep.get("first_operation_disagreement_count", "N/A")
        lines.append(
            f"| `{obs.get('pair_id', seed_s)}` | {c_val:.4f} | {t_val:.4f} | {t_val - c_val:+.4f} | {t_only} | {c_only} | {op_dis} |"
        )
    missing_pairs = list(paired_seed_bundle.get("missing_pairs", ()))
    if missing_pairs:
        lines.append("")
        lines.append(
            f"- **Missing matched pairs ({paired_seed_bundle.get('status', 'INCOMPLETE_PAIRS_REPORTED')}):** "
            f"completed job records preserved (`{len(paired_seed_bundle.get('preserved_arm_records', ()))}`)."
        )
        for mp in missing_pairs:
            lines.append(f"  - {mp.get('explanation', mp)}")

    if predeclared_slices:
        sorted_slices = sorted(
            predeclared_slices,
            key=lambda s: abs(float(s.get("drop", float(s.get("control_accuracy", 0.0)) - float(s.get("treatment_accuracy", 0.0))))),
            reverse=True,
        )
        lines.extend(
            [
                "",
                "## Predeclared Slice Differences & Contrasting Trace IDs",
                "",
                "| Slice | Tag | Baseline / Control | Target / Treatment | Diff / Drop | Control 1st-op disagree | Treatment 1st-op disagree |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for sl in sorted_slices:
            c_acc = sl.get("baseline_accuracy", sl.get("control_accuracy", 0.0))
            t_acc = sl.get("slice_accuracy", sl.get("treatment_accuracy", 0.0))
            drop = sl.get("drop", float(c_acc) - float(t_acc))
            c_op = sl.get("control_first_op_disagree_rate", sl.get("baseline_first_op_disagree_rate", "N/A"))
            t_op = sl.get("treatment_first_op_disagree_rate", sl.get("slice_first_op_disagree_rate", "N/A"))
            c_op_s = f"{float(c_op):.4f}" if isinstance(c_op, (int, float)) else str(c_op)
            t_op_s = f"{float(t_op):.4f}" if isinstance(t_op, (int, float)) else str(t_op)
            lines.append(
                f"| `{sl.get('slice_name', 'slice')}` | `[PREDECLARED]` | {float(c_acc):.4f} | {float(t_acc):.4f} | {float(drop):+.4f} | {c_op_s} | {t_op_s} |"
            )

    if ep_map:
        lines.extend(["", "### Contrasting Trace IDs", ""])
        for seed_s, ep in sorted(ep_map.items()):
            ex = ep.get("contrasting_trace_ids", {})
            lines.append(
                f"- Seed `{seed_s}` (rule `{ex.get('selection_rule', 'deterministic')}`): "
                f"`{treatment_arm}`-only=`{ex.get('treatment_only_example_id')}`, "
                f"`{control_arm}`-only=`{ex.get('control_only_example_id')}`"
            )

    if interventions:
        lines.extend(["", "## Diagnostic Interventions & Coverage", ""])
        for name, iv in sorted(interventions.items()):
            cov = iv.get("coverage", {})
            pc = iv.get("paired_changes", {})
            delta = iv.get("accuracy_delta", iv.get("delta_vs_baseline", 0.0))
            cov_line = (
                f"worlds={cov.get('distinct_base_world_count', 'N/A')}, "
                f"families={cov.get('distinct_family_count', 'N/A')} ({cov.get('distinct_families', [])}), "
                f"views={cov.get('distinct_view_count', 'N/A')}, "
                f"seeds={cov.get('distinct_seed_count', 'N/A')}, "
                f"target_coverage={cov.get('target_family_coverage_ratio', 'N/A')}"
            )
            pc_line = (
                f"+{pc.get('gained_correct_count', 0)} / -{pc.get('lost_correct_count', 0)}, "
                f"first_op_changed={pc.get('first_operation_changed_count', 0)}"
                if pc
                else "paired_changes=N/A"
            )
            status_note = iv.get("headline") or iv.get("mechanism_status") or cov.get("coverage_summary") or ""
            lines.append(f"- `{name}`: delta={float(delta):+.4f}; {cov_line}; {pc_line}. {status_note}")

    if retrospective_findings:
        lines.extend(["", "## Retrospective Findings", ""])
        for item in retrospective_findings:
            lines.append(f"- `[RETROSPECTIVE]` {item}")

    if next_experiment:
        lines.extend(
            [
                "",
                "## Smallest Discriminating Next Experiment",
                "",
                f"- **Title:** {next_experiment.get('title', '')}",
                f"- **Requires artifact:** `{next_experiment.get('requires_artifact', fa.get('name', 'FULL.zip'))}`",
                f"- **Discriminates between:** `{next_experiment.get('discriminates_between', ())}`",
                f"- **Protocol:** {next_experiment.get('protocol', '')}",
            ]
        )

    lines.append("")
    md_text = "\n".join(lines)
    return {
        "schema": "anra-planthir-analyst-first-page-v1",
        "title": title,
        "campaign_id": campaign_id,
        "source_commit": source_commit,
        "decision_summary": decision_summary,
        "control_arm": control_arm,
        "treatment_arm": treatment_arm,
        "analysis_archive": aa,
        "full_archive": fa,
        "reproduction_command": repro,
        "paired_seed_bundle": dict(paired_seed_bundle),
        "paired_episodes_by_seed": ep_map,
        "predeclared_slices": [dict(s) for s in predeclared_slices],
        "interventions": {str(k): dict(v) for k, v in dict(interventions or {}).items()},
        "retrospective_findings": [str(x) for x in retrospective_findings],
        "next_experiment": dict(next_experiment or {}),
        "markdown": md_text,
    }


def rescore_compact_analysis_archive(
    archive_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Verify a compact analysis ZIP, recompute paired-seed & paired-episode comparisons in a fresh dir, and write RESULTS.md + METRICS.json."""
    from .design import build_paired_seed_observations, compare_paired_episode_records

    zp = Path(archive_path)
    if not zp.is_file():
        raise FileNotFoundError(f"compact analysis archive not found: {zp}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive_sha = _stream_file_sha256(zp)

    v6_verified = False
    with zipfile.ZipFile(zp, "r") as zf:
        has_v6 = "BUNDLE_MANIFEST.json" in zf.namelist()
    if has_v6:
        from tools.elendril_v6_export import verify_zip as v6_verify_zip

        v6_verify_zip(zp)
        v6_verified = True

    with tempfile.TemporaryDirectory(prefix="rsc_", ignore_cleanup_errors=True) as tmp:
        ext = Path(tmp) / "x"
        safe_extract_planthir_archive(zp, ext)
        metrics_path = ext / "METRICS.json"
        if not metrics_path.is_file():
            cands = sorted(ext.rglob("METRICS.json"))
            if not cands:
                raise ValueError(f"METRICS.json not found inside {zp}")
            metrics_path = cands[0]
        raw_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    ctrl_arm = str(raw_metrics.get("control_arm", "P"))
    trt_arm = str(raw_metrics.get("treatment_arm", "R"))
    raw_arm_records = (
        raw_metrics.get("arm_records")
        or raw_metrics.get("paired_seed_bundle", {}).get("preserved_arm_records")
        or []
    )
    if raw_arm_records:
        rebuilt_bundle = build_paired_seed_observations(
            raw_arm_records,
            contrast_id=str(raw_metrics.get("paired_seed_bundle", {}).get("contrast_id", "PRIMARY")),
            control_arm=ctrl_arm,
            treatment_arm=trt_arm,
        )
    else:
        rebuilt_bundle = dict(raw_metrics.get("paired_seed_bundle", {}))

    rebuilt_eps: dict[str, Any] = {}
    ep_by_arm_seed = raw_metrics.get("episodes_by_arm_seed")
    if isinstance(ep_by_arm_seed, Mapping) and ctrl_arm in ep_by_arm_seed and trt_arm in ep_by_arm_seed:
        c_seeds = ep_by_arm_seed[ctrl_arm]
        t_seeds = ep_by_arm_seed[trt_arm]
        for s_key in sorted(set(c_seeds) & set(t_seeds)):
            rebuilt_eps[str(s_key)] = compare_paired_episode_records(c_seeds[s_key], t_seeds[s_key])
    else:
        rebuilt_eps = dict(raw_metrics.get("paired_episodes_by_seed", {}))

    aa_info = dict(raw_metrics.get("analysis_archive") or {})
    aa_info.setdefault("name", zp.name)
    aa_info["sha256"] = archive_sha
    aa_info["verified"] = True

    first_page = build_analyst_first_page_report(
        title=str(raw_metrics.get("title", zp.stem)),
        campaign_id=str(raw_metrics.get("campaign_id", "RESCORED")),
        source_commit=str(raw_metrics.get("source_commit", "0" * 40)),
        decision_summary=str(raw_metrics.get("decision_summary", "")),
        control_arm=ctrl_arm,
        treatment_arm=trt_arm,
        paired_seed_bundle=rebuilt_bundle,
        paired_episodes_by_seed=rebuilt_eps,
        predeclared_slices=raw_metrics.get("predeclared_slices", ()),
        interventions=raw_metrics.get("interventions"),
        retrospective_findings=raw_metrics.get("retrospective_findings", ()),
        next_experiment=raw_metrics.get("next_experiment"),
        analysis_archive=aa_info,
        full_archive=raw_metrics.get("full_archive"),
        reproduction_command=f"python -m anra_v5.planthir rescore-archive --archive {zp.name} --out {out.name}",
    )

    rebuilt_metrics = {
        **{k: v for k, v in raw_metrics.items() if k != "markdown"},
        "paired_seed_bundle": rebuilt_bundle,
        "paired_episodes_by_seed": rebuilt_eps,
        "analysis_archive": aa_info,
        "rebuilt_from_archive_sha256": archive_sha,
    }
    res_md_path = out / "RESULTS.md"
    metrics_out_path = out / "METRICS.json"
    res_md_path.write_text(first_page["markdown"], encoding="utf-8")
    metrics_out_path.write_text(json.dumps(rebuilt_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema": "anra-planthir-rescore-receipt-v1",
        "verified": True,
        "v6_zip_verified": v6_verified,
        "archive_path": str(zp),
        "archive_sha256": archive_sha,
        "output_dir": str(out),
        "results_md": str(res_md_path),
        "metrics_json": str(metrics_out_path),
        "complete_pair_count": rebuilt_bundle.get("complete_pair_count", 0),
        "rescored_seed_count": len(rebuilt_eps),
    }


def render_campaign_dashboard(status_payload: Mapping[str, Any]) -> str:
    """Render a scannable Level-6 Markdown campaign status dashboard from `inspect_campaign_status`."""

    counts = status_payload.get("counts", {})
    lock_info = status_payload.get("confirm_lock") or {}
    smoke_info = status_payload.get("u0000_smoke") or {}
    stage_gates = status_payload.get("stage_gates") or {}
    archives = status_payload.get("archives") or {}

    lines = [
        f"# Palantiri Level-6 Campaign Dashboard (`{status_payload.get('campaign_id', 'unknown')}`)",
        "",
        f"- **Work root:** `{status_payload.get('work_root', '')}`",
        f"- **Campaign status:** `{status_payload.get('status', 'UNKNOWN')}`",
        (
            f"- **Jobs:** total=`{counts.get('total', 0)}`, "
            f"completed=`{counts.get('completed', 0)}`, "
            f"pending=`{counts.get('pending', 0)}`, "
            f"blocked=`{counts.get('blocked', 0)}`, "
            f"failed=`{counts.get('failed', 0)}`"
        ),
        (
            f"- **U0000 smoke:** `{smoke_info.get('passed', 'NOT_RECORDED')}` "
            f"(architectures=`{sorted((smoke_info.get('architectures') or {}).keys())}`)"
        ),
        (
            f"- **Confirm lock:** status=`{lock_info.get('status', 'MISSING')}`, "
            f"pending_confirm_jobs=`{lock_info.get('pending_confirm_jobs', [])}`"
        ),
        f"- **Recommended next action:** `{status_payload.get('recommended_next_action', 'NONE')}`",
    ]
    if stage_gates:
        lines.extend(["", "## Stage Gates", "", "| Gate | Unlocked | Reason |", "|---|---|---|"])
        for gid, ginfo in sorted(stage_gates.items()):
            lines.append(f"| `{gid}` | `{ginfo.get('unlocked')}` | {ginfo.get('reason', '')} |")
    if archives:
        lines.extend(["", "## Exported Archives", ""])
        for kind, apath in sorted(archives.items()):
            lines.append(f"- `{kind}`: `{apath}`")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "build_analyst_first_page_report",
    "build_verification_matrix",
    "render_campaign_dashboard",
    "render_campaign_results_markdown",
    "rescore_compact_analysis_archive",
]
