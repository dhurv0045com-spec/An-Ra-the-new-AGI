"""Audit the returned V7.15 S8 batch and emit curated results without extracting its ZIP."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
import math
from pathlib import Path
import subprocess
from zipfile import ZipFile

from anra_v5.elendril.codec import digest
from anra_v5.elendril.v6_data import make_dataset, read_dataset
from anra_v5.elendril.v7_measurement import verify_record
from anra_v5.elendril.v71_data import build_suite, suite_manifest
from anra_v5.elendril.v71_protocol import PROTOCOL_SHA, jobs
from tools.analyze_elendril_v71_results import compact, panels, portable_replay
from tools.elendril_v6_export import verify_zip

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "e5779922a0072a070f64356f38ea47090e543d6a"
RUN = "3981e34a2ffe4b4b9528b4b4d588883b"
CORE = ROOT / "Experiment Result/V7.1/20c80f66e76e4b589aed2053afbddffc/METRICS.json"


def outcome_pairs(left, right):
    first = {row["case_id"]: row["outcome"]["correct"] for row in left}
    second = {row["case_id"]: row["outcome"]["correct"] for row in right}
    if first.keys() != second.keys() or len(first) != len(left) or len(second) != len(right):
        raise ValueError("paired cases differ")
    counts = Counter((first[key], second[key]) for key in first)
    return {label: counts[pair] for label, pair in (
        ("both", (True, True)), ("left_only", (True, False)),
        ("right_only", (False, True)), ("neither", (False, False)))}


def analyze(path, data_dir=None):
    identity = verify_zip(path)
    if (identity["source"], identity["protocol_sha256"], identity["campaign_status"]) != (SOURCE, PROTOCOL_SHA, "EXECUTED"):
        raise ValueError("unexpected source, protocol or execution status")
    subprocess.run(
        ["git", "diff", "--exit-code", SOURCE, "--", "anra_v5/elendril", "tools/elendril_v7_report.py"],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
    )
    suite = build_suite()
    confirmation = make_dataset("confirmation", worlds=112, seed=909)
    suite["confirmation"] = confirmation.cases
    manifest = suite_manifest(suite)
    core = json.loads(CORE.read_bytes())
    if core["suite"]["identity"] != manifest["suite_sha256"]:
        raise ValueError("S3 and S8 evaluation suites differ")
    expected = {job["job_id"]: job for job in jobs("scale") if job["profile"] == "S8"}
    rounding = []
    models = {}
    confirmation_records = {}
    checked = positive = 0
    example = None

    with ZipFile(path) as archive:
        def read(name):
            return json.loads(archive.read(name))

        names = archive.namelist()
        markers = [name for name in names if name.startswith("campaign-")]
        if len(markers) != 1:
            raise ValueError("expected one campaign marker")
        marker = read(markers[0])
        campaign = marker["campaign"]
        if (marker["run_id"], marker["source"], marker["protocol_sha256"]) != (RUN, SOURCE, PROTOCOL_SHA):
            raise ValueError("campaign marker identity differs")
        if (campaign["run_id"], campaign["source"], campaign["protocol_sha256"],
                campaign["phase"], campaign["status"], campaign["completed_jobs"]) != (
                    RUN, SOURCE, PROTOCOL_SHA, "scale", "EXECUTED", len(expected)):
            raise ValueError("S8 batch execution identity differs")
        if marker["required_jobs"] != list(expected) or campaign["registered_job_ids"] != list(expected):
            raise ValueError("S8 job inventory differs")
        if campaign["predeclared_scale_slice"]["subset_identity"] != digest(
            ["v715-registered-slice-v1", "scale", list(expected)]
        ):
            raise ValueError("registered batch identity differs")
        data_manifest = read("data_manifest.json")
        if (read("suite_manifest.json")["suite"] != manifest or
                data_manifest["confirmation"]["identity"] != confirmation.identity):
            raise ValueError("reconstructed evaluation identity differs")
        qualification = json.loads((ROOT / "research/v7.1/DATA_QUALIFICATION.json").read_bytes())
        if data_manifest["train"]["identity"] != qualification["datasets"]["train"]["identity"]:
            raise ValueError("training identity differs")
        for name in names:
            if name.startswith("controls/") and name.endswith(".json"):
                rounding.extend(portable_replay(suite, read(name)["evaluation"], name=name))
                checked += 1
            elif name.startswith("profiles/") and name.endswith("/timing-evaluation.json"):
                rounding.extend(portable_replay({"base": suite["base"]}, read(name), name=name))
                checked += 1
            elif data_dir and name.startswith("positive/") and name.endswith("/positive.json"):
                report = read(name)
                for field, filename in (("raw", "canary"), ("distinct_validation", "monitor")):
                    data = read_dataset(data_dir / (filename + ".json"))
                    if data.identity != data_manifest[filename]["identity"]:
                        raise ValueError("canary dataset identity differs")
                    verify_record(data, report[field])
                    positive += 1

        if len(campaign["results"]) != len(expected):
            raise ValueError("worker receipt count differs")
        for worker in campaign["results"]:
            job = worker["job"]
            job_id = job["job_id"]
            if job != expected[job_id] or worker["status"] != "COMPLETE" or job_id in models:
                raise ValueError("registered worker did not complete exactly once")
            folder = f"jobs/{job_id}/"
            trajectory = []
            for stage, update in (("initial", 0), ("epoch-1", 592), ("epoch-2", 1184),
                                  ("epoch-4", 2368), ("grammar-final", 2368)):
                name = folder + stage + ".json"
                report = read(name)
                evaluation = report["evaluation"]
                final = stage in ("epoch-4", "grammar-final")
                if (report["update"] != update or report["grammar"] != (stage == "grammar-final") or
                        evaluation["conditions"] != (sorted(suite) if final else ["base", "hop-3"])):
                    raise ValueError("evaluation coverage or mode differs")
                if any(value["raw"]["model_identity"] != report["checkpoint"] for value in evaluation["records"].values()):
                    raise ValueError("checkpoint binding differs")
                rounding.extend(portable_replay(suite, evaluation, name=name))
                checked += 1
                if stage != "grammar-final":
                    trajectory.append({"update": update, "base": compact(evaluation["metrics"]["base"]["all"]),
                                       "hop-3": compact(evaluation["metrics"]["hop-3"]["all"])})
                if stage == "epoch-4":
                    raw_report = report
            grammar = report["evaluation"]
            raw = raw_report["evaluation"]
            if any(report[key] != raw_report[key] for key in ("checkpoint", "weight_sha256")) or worker["checkpoint"] != report["checkpoint"]:
                raise ValueError("final worker and decoder checkpoints differ")
            records = {name: value["raw"]["episodes"] for name, value in raw["records"].items()}
            confirmation_records[job_id] = records["confirmation"]
            logs = sorted(
                (json.loads(line) for name in names if name.startswith(folder + "updates-") and name.endswith(".jsonl")
                 for line in archive.read(name).splitlines()), key=lambda row: row["update"]
            )
            if [row["update"] for row in logs] != list(range(1, 2369)) or any(not math.isfinite(row["loss"]) for row in logs):
                raise ValueError("update log is incomplete or nonfinite")
            work = {key: sum(row[key] for row in logs) for key in (
                "examples", "supervised_tokens", "input_positions", "padded_input_positions", "padded_attention_pairs", "wall_seconds")}
            if work["supervised_tokens"] != worker["supervised_tokens"] or worker["global_update"] != len(logs):
                raise ValueError("update receipt disagrees with logs")
            s3_id = job_id.replace("S8-", "S3-", 1)
            s3_work = next(row for row in core["training"] if row["job"]["job_id"] == s3_id)
            s3_score = next(row["raw"]["correct"] for row in core["rows"]
                            if row["job_id"] == s3_id and row["cohort"] == "confirmation")
            for key in ("examples", "supervised_tokens", "input_positions"):
                if s3_work[key] != work[key]:
                    raise ValueError("comparison exposure differs")
            models[job_id] = {
                "job": job, "checkpoint": report["checkpoint"], "parameters": worker["parameter_count"],
                "updates": len(logs), "work": work,
                "last64_mean_loss": math.fsum(row["loss"] for row in logs[-64:]) / 64,
                "worker_seconds": worker["attempt_wall_seconds"], "peak_allocated_bytes": worker["peak_cuda_allocated_bytes"],
                "s3_confirmation_correct": s3_score, "s3_update_loop_seconds": s3_work["wall_seconds"],
                "trajectory": trajectory,
                "cohorts": {name: {"raw": compact(raw["metrics"][name]["all"]),
                                   "grammar": compact(grammar["metrics"][name]["all"]),
                                   "grammar_pairs": outcome_pairs(records[name], grammar["records"][name]["raw"]["episodes"])}
                            for name in suite},
                "panels": {name: panels(raw["records"][name], suite[name]) for name in (
                    "memory-manipulations", "query-crossed", "hop-2-clamped", "robustness-row-order")},
                "confirmation_by_kind": {kind: compact(value) for kind, value in raw["metrics"]["confirmation"]["by_kind"].items()},
            }
            if job_id == "S8-blocked-ce-s41":
                row = next(row for row in records["hop-3"] if row["trace"]["status"] == "invalid_action")
                case = next(case for case in suite["hop-3"] if case.case_id == row["case_id"])
                example = {"job_id": job_id, "case_id": row["case_id"], "expected": case.truth.expected,
                           "prompt": row["prompts"][0], "first_action": row["trace"]["decisions"][0],
                           "outcome": row["trace"]["status"]}
        if checked != 38 or checked != identity["scientific_rescore"]["reports_replayed"]:
            raise ValueError("unexpected replay report inventory")
        runtime = read(next(name for name in names if name.startswith("environment-")))

    with path.open("rb") as handle:
        archive_hash = file_digest(handle, "sha256").hexdigest()
    return {
        "run_id": RUN, "source": SOURCE, "protocol_sha256": PROTOCOL_SHA,
        "archive": {"bytes": path.stat().st_size, "sha256": archive_hash, "manifest_sha256": identity["manifest_sha256"],
                    "verified_members": identity["files"], "final_delivery_receipt_available": False,
                    "full_checkpoint_zip_available": False},
        "execution": {"registered_jobs": list(expected), "completed_jobs": len(models),
                      "controller_seconds_before_finalization": campaign["wall_seconds"],
                      "gpu_worker_reservation_seconds": campaign["gpu_worker_reservation_seconds"],
                      "capacity_forecast_seconds": campaign["capacity_estimate"]["seconds"],
                      "reported_control_statuses": [row["status"] for row in campaign["positive_controls"] + campaign["profiles"]],
                      "reported_recovery_statuses": [row["recovery"]["status"] for row in campaign["positive_controls"] + campaign["profiles"]],
                      "devices": [row["physical"]["name"] for row in runtime], "runtime": runtime[0]["runtime"]},
        "replay": {"suite_reports": checked, "additional_positive_records": positive, "derived_brier_rounding": rounding},
        "data": {"train_identity": data_manifest["train"]["identity"], "suite_identity": manifest["suite_sha256"],
                 "case_records": manifest["total_cases"], "unique_worlds": len({c.world_id for cases in suite.values() for c in cases})},
        "models": models,
        "confirmation_uniform_vs_blocked": {seed: outcome_pairs(confirmation_records[f"S8-uniform-ce-s{seed}"],
            confirmation_records[f"S8-blocked-ce-s{seed}"]) for seed in (7, 19, 41)},
        "example_failure": example,
        "cause_status": "CAUSE_UNRESOLVED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    result = analyze(args.zip, args.data_dir)
    with args.output.open("x", encoding="utf8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"Verified {result['replay']['suite_reports']} reports and {result['replay']['additional_positive_records']} controls")


if __name__ == "__main__":
    main()
