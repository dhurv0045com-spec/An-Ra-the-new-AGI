"""Offline verified analyzer for ELENDRIL V8-X1 session ANALYSIS archives."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any
import zipfile

from anra_v5.elendril.v8x.analysis import (
    build_x1_report_markdown,
    cluster_bootstrap_paired_world_diff,
    evaluate_x1_development_selection,
    summarize_x1_episodes,
)
from anra_v5.elendril.v8x.data import build_x1_d1h_training_dataset, verify_x1_episode
from anra_v5.elendril.v8x.evaluation import run_h01_frozen_intervention
from anra_v5.elendril.v8x.protocol import X1_SESSIONS, session_jobs

from .elendril_v8x_export import FINAL_STATUS_NAME, verify_single_x1_zip


def analyze_x1_archive(
    archive_path: Path,
    *,
    expected_source_commit: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify ANALYSIS ZIP, independently replay episode contracts, and write curated reports."""

    ver = verify_single_x1_zip(archive_path, expected_source_commit=expected_source_commit)
    manifest = ver["manifest"]
    final_status = ver["final_status"]
    session = str(final_status["session"])
    if session not in X1_SESSIONS:
        raise ValueError(f"unsupported X1 session in archive: {session!r}")

    with zipfile.ZipFile(archive_path, "r") as zf:
        ident = json.loads(zf.read("identity.json").decode("utf-8"))
        data_man = json.loads(zf.read("data/manifest.json").decode("utf-8"))
        reg = json.loads(zf.read("registry.json").decode("utf-8"))
        is_cpu = bool(ident.get("cpu_fixtures", False))

        # Regenerate source training identity independently when full-scale (non-fixture) or quota=1
        regen_d1h = build_x1_d1h_training_dataset(quota_per_family=1 if is_cpu else 512)
        if regen_d1h.data_identity != data_man.get("d1h_identity"):
            raise ValueError(
                f"regenerated D1-H identity {regen_d1h.data_identity} != archived {data_man.get('d1h_identity')}"
            )

        replayed_episodes = 0
        contract_violations = 0
        job_summaries: list[dict[str, Any]] = []
        recs_by_job: dict[str, list[dict[str, Any]]] = {}

        names = set(zf.namelist())
        for j in reg.get("jobs", []):
            jid = str(j["job_id"])
            jpath = f"jobs/{jid}/job.json"
            if jpath not in names:
                continue
            jdoc = json.loads(zf.read(jpath).decode("utf-8"))
            ms_paths = sorted(n for n in names if n.startswith(f"jobs/{jid}/milestones/") and n.endswith("/episodes.jsonl"))
            final_sum: dict[str, Any] = {}
            final_recs: list[dict[str, Any]] = []
            for ep_path in ms_paths:
                rows = [json.loads(ln) for ln in zf.read(ep_path).decode("utf-8").splitlines() if ln.strip()]
                replayed_episodes += len(rows)
                for row in rows:
                    if not isinstance(row.get("correct"), bool) or "steps" not in row:
                        contract_violations += 1
                    # Verify each step recorded legal X1 action structure and consumed-tool monotonicity
                    seen_q_used = False
                    seen_m_used = False
                    for st in row.get("steps", ()):
                        op = st.get("executed_op")
                        if op not in ("ANSWER", "QUERY", "RETRIEVE", "ABSTAIN"):
                            contract_violations += 1
                        if seen_q_used and st.get("query_available"):
                            contract_violations += 1
                        if seen_m_used and st.get("memory_available"):
                            contract_violations += 1
                        if op == "QUERY" and st.get("query_available"):
                            seen_q_used = True
                        if op == "RETRIEVE" and st.get("memory_available"):
                            seen_m_used = True
                if ep_path == ms_paths[-1]:
                    final_recs = rows
                    if rows:
                        final_sum = summarize_x1_episodes(rows)
            recs_by_job[jid] = final_recs
            job_summaries.append(
                {
                    "job_id": jid,
                    "architecture": jdoc["architecture"],
                    "objective": jdoc["objective"],
                    "init_seed": jdoc["init_seed"],
                    "completed_updates": jdoc["completed_updates"],
                    "status": jdoc["status"],
                    "summary": final_sum,
                }
            )

    by_cond: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for js in job_summaries:
        by_cond[f"{js['architecture']}:{js['objective']}"][int(js["init_seed"])] = js

    conds = sorted(by_cond)
    comparisons: list[dict[str, Any]] = []
    dev_gate: dict[str, Any] | None = None
    if len(conds) >= 2:
        c_key, t_key = conds[0], conds[1]
        shared = sorted(set(by_cond[c_key]) & set(by_cond[t_key]))
        pairs = []
        ctrl_all, trt_all = [], []
        for s in shared:
            cj, tj = by_cond[c_key][s], by_cond[t_key][s]
            pairs.append(
                {
                    "init_seed": s,
                    "order_seed": s,
                    "ctrl_macro_12": float(cj["summary"].get("macro_12_family_accuracy", 0.0)),
                    "trt_macro_12": float(tj["summary"].get("macro_12_family_accuracy", 0.0)),
                }
            )
            ctrl_all.extend(recs_by_job.get(cj["job_id"], []))
            trt_all.extend(recs_by_job.get(tj["job_id"], []))
        if ctrl_all and trt_all:
            boot = cluster_bootstrap_paired_world_diff(
                ctrl_all,
                trt_all,
                replicates=200 if len(ctrl_all) < 500 else 10_000,
            )
            comparisons.append(
                {
                    "control": c_key,
                    "treatment": t_key,
                    "shared_seeds": shared,
                    "cluster_bootstrap": boot,
                }
            )
            dev_gate = evaluate_x1_development_selection(pairs)

    metrics = {
        "schema": "elendril-x1-curated-metrics-v1",
        "audit_status": "PASS" if contract_violations == 0 else "CONTRACT_VIOLATION",
        "session": session,
        "status": final_status["status"],
        "source_commit": final_status["source_commit"],
        "archive_zip_sha256": ver["zip_sha256"],
        "archive_manifest_sha256": manifest["manifest_sha256"],
        "replayed_episodes": replayed_episodes,
        "contract_violations": contract_violations,
        "expected_jobs": [str(j["job_id"]) for j in session_jobs(session)],
        "completed_jobs": list(final_status.get("completed_jobs", [])),
        "failed_jobs": list(final_status.get("failed_jobs", [])),
        "missing_jobs": list(final_status.get("missing_jobs", [])),
        "jobs": job_summaries,
        "comparisons": comparisons,
        "development_gate": dev_gate,
        "h01_status": run_h01_frozen_intervention(),
        "limitations": [
            "Analysis-only intake checks metadata, trajectories, and splits; tensor verification uses FULL ZIP separately.",
            *(["CPU fixture mode used; cannot be represented as S20 T4 research evidence."] if is_cpu else []),
        ],
    }
    provenance = {
        "schema": "elendril-x1-curated-provenance-v1",
        "source_commit": final_status["source_commit"],
        "session": session,
        "archive_path": str(archive_path),
        "archive_zip_sha256": ver["zip_sha256"],
        "archive_manifest_sha256": manifest["manifest_sha256"],
        "d1h_identity": regen_d1h.data_identity,
        "cpu_fixtures": is_cpu,
    }
    sci_status = (
        "CPU_FIXTURE_ONLY"
        if is_cpu
        else ("HYPOTHESIS_SUPPORTED_DEV" if (dev_gate and dev_gate.get("qualified")) else "OBSERVED_DEV_UNCONFIRMED")
    )
    results_md = build_x1_report_markdown(
        session=session,
        source_commit=str(final_status["source_commit"]),
        execution_status=str(final_status["status"]),
        delivery_status="ANALYSIS_ZIP_VERIFIED",
        scientific_status=sci_status,
        cause_status="REPLAY_PASS" if contract_violations == 0 else "CONTRACT_VIOLATION",
        job_summaries=job_summaries,
        notes=[
            f"Replayed episodes: {replayed_episodes} | Contract violations: {contract_violations}",
            f"Archive SHA-256: `{ver['zip_sha256']}`",
        ],
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "METRICS.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (output_dir / "PROVENANCE.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (output_dir / "RESULTS.md").write_text(results_md, encoding="utf-8")

    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze an ELENDRIL V8-X1 ANALYSIS ZIP offline.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", default=None, help="Expected 40-hex source_commit")
    args = parser.parse_args(argv)

    res = analyze_x1_archive(args.archive, expected_source_commit=args.source, output_dir=args.output)
    print(
        json.dumps(
            {
                "audit_status": res["audit_status"],
                "session": res["session"],
                "status": res["status"],
                "replayed_episodes": res["replayed_episodes"],
                "output_dir": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if res["audit_status"] == "PASS" else 1


__all__ = ["analyze_x1_archive"]

if __name__ == "__main__":
    raise SystemExit(main())
