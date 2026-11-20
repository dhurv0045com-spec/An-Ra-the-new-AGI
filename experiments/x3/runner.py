"""Strictly gated X3 runner executed ONLY after global X2 completion audit passes (§4)."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
import zipfile

from experiments.x2.lineage_runner import load_parent_checkpoint
from experiments.x2.tasks import build_generator_b_multihop_panel, build_selective_edit_panel

from .factorial import (
    build_x3_paired_graph_factorial,
    build_x3_x2_crosswalk,
    eval_x3_shortcut_baselines,
)
from .interventions import (
    evaluate_x3_checkpoint_suite,
    run_x3_pointer_interchange_intervention,
    score_x3_campaign,
)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_gated_x3_campaign(
    *,
    x2_completion_audit: Mapping[str, Any],
    output_dir: Path | str,
    source_commit: str,
    checkpoint_map: Mapping[str, Mapping[int, Path | str]] | None = None,
    device: str = "cpu",
    run_confirmation: bool = False,
    tiny_canary: bool = False,
    dev_base_graphs_per_depth: int | None = None,
    conf_base_graphs_per_depth: int | None = None,
) -> dict[str, Any]:
    """Run X3 strictly gated behind `x2_completion_audit['x2_complete'] is True`, keeping compute/archives separate."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x2_ok = bool(x2_completion_audit.get("x2_complete", False))
    if not x2_ok:
        gate_doc = {
            "schema": "elendril-x3-gate-status-v1",
            "status": "GATED_PENDING_X2_COMPLETE",
            "x3_executed": False,
            "gpu_seconds_spent": 0.0,
            "blocking_x2_status": str(x2_completion_audit.get("global_x2_status", "PARTIAL")),
            "unresolved_mandatory_count": len(x2_completion_audit.get("unresolved_mandatory_ids", ())),
            "unresolved_conditional_count": len(x2_completion_audit.get("unresolved_conditional_ids", ())),
        }
        _atomic_write_json(out_dir / "X3_GATE_STATUS.json", gate_doc)
        return gate_doc

    t0 = time.monotonic()
    x3_work = out_dir / "x3_artifacts"
    x3_work.mkdir(parents=True, exist_ok=True)

    dev_panel = build_x3_paired_graph_factorial(
        split="dev",
        depths=(4, 5),
        base_graphs_per_depth=dev_base_graphs_per_depth,
    )
    dev_baselines = eval_x3_shortcut_baselines(dev_panel)
    x2_ref_worlds = [
        *build_selective_edit_panel(num_histories=4, seed=701, split="dev"),
        *build_generator_b_multihop_panel(split="dev", num_skeletons=2, seed=702, depths=(4, 5), include_exhaustive_perm=False)["episodes"][:16],
    ]
    crosswalk_dev = build_x3_x2_crosswalk(dev_panel, x2_ref_worlds)

    cmap = checkpoint_map or {}
    model_results: dict[str, dict[int, Any]] = {}
    interchanges: dict[str, dict[str, Any]] = {}
    for arch, seed_paths in sorted(cmap.items()):
        model_results[arch] = {}
        interchanges[arch] = {}
        for seed_val, ckpt_path in sorted(seed_paths.items()):
            s_int = int(seed_val)
            loaded = load_parent_checkpoint(
                ckpt_path,
                architecture=arch,
                tiny_canary=tiny_canary,
                device=device,
                require_x1_native=True,
            )
            m = loaded["model"]
            mspec = loaded["model_spec"]
            suite = evaluate_x3_checkpoint_suite(m, mspec, dev_panel, device=device)
            model_results[arch][s_int] = suite
            ic_raw = run_x3_pointer_interchange_intervention(
                m, mspec, dev_panel["aligned_pairs"], device=device, equalize_norm=False
            )
            ic_eq = run_x3_pointer_interchange_intervention(
                m, mspec, dev_panel["aligned_pairs"], device=device, equalize_norm=True
            )
            interchanges[arch][str(s_int)] = {"raw": ic_raw, "norm_equalized": ic_eq}

    dev_score = score_x3_campaign(model_results, stage="dev") if model_results else {
        "schema": "elendril-x3-campaign-score-v1",
        "stage": "dev",
        "gate_passed": False,
        "reason": "NO_CHECKPOINTS_SUPPLIED",
    }

    conf_summary: dict[str, Any] | None = None
    if run_confirmation:
        conf_panel = build_x3_paired_graph_factorial(
            split="confirmation",
            depths=(4, 5),
            base_graphs_per_depth=conf_base_graphs_per_depth,
        )
        conf_baselines = eval_x3_shortcut_baselines(conf_panel)
        crosswalk_conf = build_x3_x2_crosswalk(conf_panel, dev_panel["decisions"])
        conf_results: dict[str, dict[int, Any]] = {}
        for arch, seed_paths in sorted(cmap.items()):
            conf_results[arch] = {}
            for seed_val, ckpt_path in sorted(seed_paths.items()):
                loaded = load_parent_checkpoint(
                    ckpt_path,
                    architecture=arch,
                    tiny_canary=tiny_canary,
                    device=device,
                    require_x1_native=True,
                )
                conf_results[arch][int(seed_val)] = evaluate_x3_checkpoint_suite(
                    loaded["model"], loaded["model_spec"], conf_panel, device=device
                )
        conf_score = score_x3_campaign(conf_results, stage="confirmation") if conf_results else {"gate_passed": False}
        conf_summary = {
            "baselines": conf_baselines,
            "crosswalk": crosswalk_conf,
            "score": {k: v for k, v in conf_score.items() if k != "pair_records"},
        }

    elapsed = round(time.monotonic() - t0, 3)
    metrics_doc = {
        "schema": "elendril-x3-metrics-v1",
        "source_commit": str(source_commit),
        "status": "COMPLETE",
        "x3_executed": True,
        "gpu_seconds_spent": elapsed,
        "x2_unlock_audit_status": x2_completion_audit.get("global_x2_status"),
        "dev_panel_sha256": dev_panel["panel_sha256"],
        "dev_decision_count": dev_panel["decision_count"],
        "dev_shortcut_baselines": dev_baselines,
        "dev_score": dev_score,
        "interchange_interventions": interchanges,
        "confirmation": conf_summary,
    }
    report_lines = [
        "# ELENDRIL V8-X3 Report — Does the Pointer Follow a Changed Relation?",
        "",
        f"- **Status:** `{metrics_doc['status']}` (`x2_unlock={x2_completion_audit.get('global_x2_status')}`)",
        f"- **Dev Panel:** {dev_panel['decision_count']} decisions (`sha={dev_panel['panel_sha256'][:12]}`)",
        f"- **Shortcut Baselines:** first_row={dev_baselines['first_row_accuracy']:.4f}, last_row={dev_baselines['last_row_accuracy']:.4f}, "
        f"freq={dev_baselines['frequency_accuracy']:.4f}, oracle={dev_baselines['oracle_independent_walker_accuracy']:.4f}",
        f"- **Dev Gate Passed:** `{dev_score.get('gate_passed')}`",
    ]
    if "p_vs_fcap" in dev_score:
        pv = dev_score["p_vs_fcap"]
        report_lines.append(
            f"- **P vs F-cap:** mean_delta={pv['mean_delta']:+.4f} (positive_seeds={pv['positive_seeds']}/{len(pv['shared_seeds'])}), "
            f"bootstrap_ci95={pv['bootstrap']['ci95']}"
        )
    if "r_vs_r_noedge" in dev_score:
        rv = dev_score["r_vs_r_noedge"]
        report_lines.append(
            f"- **R vs R-noedge:** mean_delta={rv['mean_delta']:+.4f}, bootstrap_ci95={rv['bootstrap']['ci95']}"
        )
    report_md = "\n".join(report_lines) + "\n"

    _atomic_write_json(x3_work / "X3_METRICS.json", metrics_doc)
    _atomic_write_json(x3_work / "X3_CROSSWALK.json", {"dev": crosswalk_dev, "confirmation": conf_summary["crosswalk"] if conf_summary else None})
    (x3_work / "X3_REPORT.md").write_text(report_md, encoding="utf-8")
    gate_ok = {"schema": "elendril-x3-gate-status-v1", "status": "UNLOCKED_COMPLETE", "x3_executed": True, "gpu_seconds_spent": elapsed}
    _atomic_write_json(out_dir / "X3_GATE_STATUS.json", gate_ok)
    _atomic_write_json(out_dir / "X3_METRICS.json", metrics_doc)
    (out_dir / "X3_REPORT.md").write_text(report_md, encoding="utf-8")

    x3_zip = out_dir / "ELENDRIL_V8_X3_ANALYSIS.zip"
    tmp_zip = out_dir / ".ELENDRIL_V8_X3_ANALYSIS.zip.tmp"
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_dir / "X3_GATE_STATUS.json", "X3_GATE_STATUS.json")
        for fp in sorted(x3_work.rglob("*")):
            if fp.is_file():
                zf.write(fp, fp.relative_to(x3_work).as_posix())
    os.replace(tmp_zip, x3_zip)
    zsha = _sha256_file(x3_zip)
    (out_dir / "ELENDRIL_V8_X3_ANALYSIS.zip.sha256").write_text(f"{zsha}  {x3_zip.name}\n", encoding="utf-8")
    return {**metrics_doc, "archive_path": str(x3_zip), "archive_sha256": zsha}


__all__ = ["run_gated_x3_campaign"]
