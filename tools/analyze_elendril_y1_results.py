"""Offline AN-RA V8-Y1 evidence replay and positive/negative/inconclusive report generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import zipfile

from anra_v5.elendril.codec import canonical
from anra_v5.elendril.y1_data import load_y1_suite
from anra_v5.elendril.y1_evaluation import (
    evaluate_all_heuristics,
    evaluate_t1_or_t2_gate,
    summarize_y1_predictions,
)
from anra_v5.elendril.y1_protocol import PROTOCOL_SHA, SEEDS
from tools import elendril_v6_export as v6


def analyze_y1_archive(archive_path: Path | str, *, allow_non_production: bool = False) -> dict[str, Any]:
    """Verify archive hashes, replay predictions & gates from public suites, and compile METRICS."""
    zp = Path(archive_path)
    zip_check = v6.verify_zip(zp)
    with zipfile.ZipFile(zp, "r") as zf:
        names = set(zf.namelist())
        ident = json.loads(zf.read("analysis/identity.json").decode("utf-8"))
        is_prod = ident.get("protocol_sha256") == PROTOCOL_SHA
        if not is_prod and not allow_non_production:
            raise ValueError(f"protocol SHA mismatch (non-production or modified override): {ident.get('protocol_sha256')} != {PROTOCOL_SHA}")

        dev_suite = load_y1_suite(json.loads(zf.read("prepared/development_suite.json").decode("utf-8")))
        conf_suite = load_y1_suite(json.loads(zf.read("prepared/confirmation_suite.json").decode("utf-8")))
        fol_suite = load_y1_suite(json.loads(zf.read("prepared/followup_suite.json").decode("utf-8")))
        heur_dev = json.loads(zf.read("analysis/heuristics/development.json").decode("utf-8")) if "analysis/heuristics/development.json" in names else evaluate_all_heuristics(dev_suite["cases"], historical_cases=dev_suite["historical_continuity"])

        pred_names = sorted(n for n in names if n.startswith("evidence/predictions/") and not n.endswith("_hist.json"))
        replayed_jobs: dict[str, Any] = {}
        t1_metrics: dict[str, dict[int, dict[str, Any]]] = {}

        for pname in pred_names:
            jid = Path(pname).stem
            pdoc = json.loads(zf.read(pname).decode("utf-8"))
            hname = f"evidence/predictions/{jid}_hist.json"
            hdoc = json.loads(zf.read(hname).decode("utf-8")) if hname in names else None
            rfile = f"analysis/jobs/{jid}/run_receipt.json"
            rec = json.loads(zf.read(rfile).decode("utf-8")) if rfile in names else {}

            if "t2_confirm" in jid:
                spl = conf_suite
            elif "t3_" in jid:
                spl = fol_suite
            else:
                spl = dev_suite

            summ = summarize_y1_predictions(
                spl["cases"],
                pdoc,
                historical_continuity_cases=spl["historical_continuity"],
                historical_pred_doc=hdoc,
            )
            replayed_jobs[jid] = {
                "receipt": rec,
                "primary_endpoint": summ["primary_endpoint"],
                "per_hop_primary": summ["per_hop_primary"],
                "guardrails": summ["guardrails"],
                "edge_change_panel": summ["edge_change_panel"],
                "missing_bridge_panel": summ["missing_bridge_panel"],
                "dual_chain_query_panel": summ["dual_chain_query_panel"],
                "serialization_holdout": summ["serialization_holdout"],
                "preservation_suite": summ["preservation_suite"],
                "first_divergence_counts": summ["first_divergence_counts"],
                "failed_count": summ["failed_count"],
                "representative_failed_case_ids": summ["representative_failed_case_ids"],
                "historical_continuity": summ["historical_continuity"],
            }
            if jid.startswith("Y1-t1_factorial-S3-"):
                arm = rec["arm"]
                seed = int(rec["seed"])
                t1_metrics.setdefault(arm, {})[seed] = summ

        recomputed_t1_gate = None
        if t1_metrics:
            recomputed_t1_gate = evaluate_t1_or_t2_gate(
                stage="t1_factorial",
                arms_metrics=t1_metrics,
                heuristics=heur_dev,
                require_ci_lower_gt_zero=False,
            )

        return {
            "schema": "anra-v8-y1-analysis-v1",
            "archive_path": str(zp),
            "archive_sha256": v6._file_sha(zp)[1],
            "zip_verification": zip_check,
            "identity": ident,
            "production_qualified": is_prod,
            "heuristics_development": heur_dev,
            "jobs": replayed_jobs,
            "recomputed_t1_gate": recomputed_t1_gate,
        }


def render_y1_results_markdown(metrics: dict[str, Any]) -> str:
    """Format full positive/negative/inconclusive Markdown report from offline replayed metrics."""
    ident = metrics["identity"]
    lines = [
        f"# AN-RA V8-Y1 Offline Replayed Results ({ident['phase']})",
        "",
        f"- **Run ID**: `{ident['run_id']}`",
        f"- **Source commit**: `{ident['source']}`",
        f"- **Protocol SHA256**: `{ident['protocol_sha256']}`",
        f"- **Archive SHA256**: `{metrics['archive_sha256']}`",
        "",
        "## Replayed Jobs",
        "",
        "| Job ID | Primary | Edge Both-Correct | Dual-Query Both-Correct | Valid Rate | Unavailable Abstain | Failed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for jid, row in sorted(metrics["jobs"].items()):
        g = row["guardrails"]
        fmt = lambda v: "n/a" if v is None else f"{100.0 * v:.2f}%"
        lines.append(
            f"| {jid} | {fmt(row['primary_endpoint'])} | {fmt(g['edge_both_correct_all_three'])} | "
            f"{fmt(g['dual_query_both_correct'])} | {fmt(g['valid_action_rate'])} | "
            f"{fmt(g['unavailable_abstain_rate'])} | {row['failed_count']} |"
        )
    if metrics.get("recomputed_t1_gate"):
        tg = metrics["recomputed_t1_gate"]
        lines.extend([
            "",
            "## Recomputed T1 Factorial Gate",
            f"- Status: `{tg['status']}` (locked winner: `{tg.get('locked_winner')}`)",
        ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    metrics = analyze_y1_archive(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "METRICS.json").write_bytes(canonical(metrics) + b"\n")
    (args.output / "RESULTS.md").write_text(render_y1_results_markdown(metrics), encoding="utf-8")
    print(json.dumps({"status": "REPLAY_VERIFIED", "output": str(args.output), "jobs": len(metrics["jobs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
