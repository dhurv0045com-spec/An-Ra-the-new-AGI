"""Owner-operated V7.15 frozen research campaign; no import-time GPU work.

All nine original V7.1 S3 checkpoint generations are restored from the verified
FULL archive. Runs paired-surface and anti-shortcut development diagnostics with
raw neural and engineered grammar decoders. No optimizer updates are permitted.
"""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256, file_digest
import argparse
import json
import os
from pathlib import Path
from queue import Queue, Empty
import shutil
import time
import traceback
from uuid import uuid4
import zipfile

from anra_v5.elendril.v715_protocol import (
    PARENT_SOURCE, PARENT_FULL_SHA256, PROTOCOL_SHA, SEED, registered_jobs, protocol
)
from anra_v5.elendril.v715_diagnostics import build_suite as build_anti, audit as audit_anti
from anra_v5.elendril.v71_followup import build_suite as build_paired, audit as audit_paired
from anra_v5.elendril.v71_data import suite_to_record, suite_from_record
from anra_v5.elendril.v71_protocol import PROTOCOL_SHA as V71_SHA
from anra_v5.elendril.v71_evaluation import verify_suite
from anra_v5.elendril.v6_data import read_dataset
from tools.elendril_cloud import check_source, check_torch, fingerprint
from tools.elendril_v6_campaign import _gpu_records
from tools.elendril_v6_export import immutable_json
from tools.elendril_v71_campaign import invoke
from tools.elendril_v71_export import restore_archive
from v5_training.checkpoint import CheckpointStore

REPO = Path(__file__).resolve().parents[1]
ARCHIVE_NAME = "ELENDRIL_V715_ANALYSIS.zip"


def suite_and_audit(train):
    paired = build_paired(seed=17071)
    anti = build_anti(seed=SEED)
    left, right = audit_paired(paired, seed=17071, train=train), audit_anti(anti, seed=SEED, train=train)
    if left["status"] != "PASS" or right["status"] != "PASS":
        raise ValueError("development diagnostic data failed qualification")
    if left["actual_training_disjoint"]["status"] != "PASS" or right["training_disjoint"]["status"] != "PASS":
        raise ValueError("actual training-world/state disjointness not proven")
    combined = {
        **{"paired/" + k: v for k, v in paired.items()},
        **{"anti/" + k: v for k, v in anti.items()},
    }
    if len(combined) != 56 or sum(len(v) for v in combined.values()) != 896:
        raise ValueError("frozen suite coverage mismatch")
    return combined, {"paired": left, "anti": right}


def _identity(output, source):
    path = output / "analysis" / "identity.json"
    identity = {"schema": "elendril-v715-diagnostic-run-v1", "source": source,
                "protocol_sha256": PROTOCOL_SHA, "parent": PARENT_SOURCE,
                "mode": "diagnostics"}
    if path.exists():
        if json.loads(path.read_bytes()) != identity:
            raise ValueError("resumed output identity mismatch")
    else:
        immutable_json(path, identity)
    return identity


def _parent(parent_archive, parent):
    if parent_archive is None:
        raise ValueError("diagnostics needs the original V7.1 core FULL ZIP as PARENT_ARCHIVE")
    archive = Path(parent_archive)
    if not archive.is_file() or archive.is_symlink():
        raise ValueError("V7.1 parent archive missing or symlinked")
    sha = fingerprint(archive)
    if sha != PARENT_FULL_SHA256:
        raise ValueError("V7.1 FULL archive SHA256 does not match owner receipt")
    expected = {"source": PARENT_SOURCE, "protocol_sha256": V71_SHA, "phase": "core"}
    if parent.exists():
        if parent.is_symlink() or json.loads((parent / "analysis" / "identity.json").read_bytes()) != expected:
            raise ValueError("existing parent restore not verified; never overwrite")
    else:
        restore_archive(archive, parent, source=PARENT_SOURCE, phase="continual", parent=True)
    return {"source": PARENT_SOURCE, "full_zip_sha256": sha,
            "archive_bytes": archive.stat().st_size, "restored_directory": str(parent)}


def _coverage(parent):
    """All nine original core jobs and recorded final generation IDs are required."""
    expected = {}
    for job in registered_jobs():
        path = parent / "analysis" / "jobs" / job["job_id"] / "epoch-4.json"
        report = json.loads(path.read_bytes())
        checkpoint = report["checkpoint"]
        if not isinstance(report.get("weight_sha256"), str) or len(report["weight_sha256"]) != 64:
            raise ValueError("missing original final weight fingerprint")
        state, _ = CheckpointStore(
            parent / "checkpoints" / "jobs" / job["job_id"], job["job_id"]
        ).restore(checkpoint)
        if not state.complete or state.identities.source_commit != PARENT_SOURCE:
            raise ValueError("original V7.1 final generation not source-bound or complete")
        expected[job["job_id"]] = {"checkpoint": checkpoint,
                                   "weight_sha256": report["weight_sha256"]}
    return expected


def _replay(output, suite, expected):
    counts = {"reports": 0, "episodes": 0}
    for job in registered_jobs():
        original = expected[job["job_id"]]
        for decoder in ("raw", "grammar"):
            path = output / "analysis" / "jobs" / job["job_id"] / (decoder + ".json")
            row = json.loads(path.read_bytes())
            if (row["job_id"] != job["job_id"] or row["checkpoint"] != original["checkpoint"]
                    or row["weight_sha256"] != original["weight_sha256"]
                    or row["decoder"] != decoder or row["protocol_sha256"] != PROTOCOL_SHA):
                raise ValueError("V7.15 replay report/checkpoint identity mismatch")
            verify_suite(suite, row["evaluation"])
            if row["evaluation"]["conditions"] != sorted(suite):
                raise ValueError("frozen report missing a registered condition")
            counts["reports"] += 1
            counts["episodes"] += 896
    return {"status": "PASS", **counts,
            "qualification": "Recorded public actions/outcomes replayed; fresh neural probabilities not regenerated."}


def _summary(output):
    rows = []
    for job in registered_jobs():
        for decoder in ("raw", "grammar"):
            report = json.loads(
                (output / "analysis" / "jobs" / job["job_id"] / (decoder + ".json")).read_bytes()
            )
            for cohort, metrics in report["evaluation"]["metrics"].items():
                item = metrics["all"]
                rows.append({
                    "job_id": job["job_id"], "decoder": decoder, "cohort": cohort,
                    "correct": item["correct"], "episodes": item["episodes"],
                    "invalid": item["invalid"], "nonnull_correct": item["nonnull_correct"],
                    "nonnull_total": item["nonnull_total"],
                    "correct_abstentions": item["correct_abstentions"],
                    "null_total": item["null_total"],
                    "both_correct_pairs": item["both_correct_pairs"],
                })
    return {"schema": "elendril-v715-frozen-summary-v1", "rows": rows,
            "qualification": "56 development cohorts and related synthetic worlds; CAUSE UNRESOLVED"}


def _zip(output):
    """Make a small, verified analysis archive; never duplicate parent weights."""
    destination = output / "exports" / ARCHIVE_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for root in ("analysis", "data") for p in (output/root).rglob("*")
                   if p.is_file() and not p.is_symlink())
    if not files:
        raise ValueError("no V7.15 analysis files to export")
    members = []
    for path in files:
        relative = path.relative_to(output).as_posix()
        members.append({"path": relative, "sha256": fingerprint(path), "bytes": path.stat().st_size})
    manifest = {"schema": "elendril-v715-analysis-manifest-v1",
                "protocol_sha256": PROTOCOL_SHA, "members": members}
    if not destination.exists():
        tmp = destination.with_name("." + uuid4().hex + "-pending.zip")
        try:
            with zipfile.ZipFile(tmp, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                                 allowZip64=True) as z:
                for entry in members:
                    z.write(output / entry["path"], entry["path"])
                z.writestr("MANIFEST.json", json.dumps(manifest, sort_keys=True).encode("utf-8"))
            os.replace(tmp, destination)
        finally:
            tmp.unlink(missing_ok=True)
    with zipfile.ZipFile(destination) as z:
        recorded = json.loads(z.read("MANIFEST.json"))
        if recorded != manifest or set(z.namelist()) != ({m["path"] for m in members} | {"MANIFEST.json"}):
            raise ValueError("V7.15 archive inventory changed")
        for member in members:
            data = z.read(member["path"])
            if len(data) != member["bytes"] or sha256(data).hexdigest() != member["sha256"]:
                raise ValueError("V7.15 archive checksum mismatch")
    return {"status": "VERIFIED", "file": ARCHIVE_NAME,
            "bytes": destination.stat().st_size,
            "sha256": fingerprint(destination), "members": len(members)}


def run(*, output, source, parent_archive, minutes=330, owner_enabled=False):
    if owner_enabled is not True:
        raise ValueError("explicit owner-run authorization required")
    if type(minutes) not in (int, float) or not 1 <= minutes <= 330:
        raise ValueError("session minutes out of bounds")
    output = Path(output).resolve()
    if output.is_relative_to(REPO) or output == Path(output.anchor):
        raise ValueError("dedicated output must be outside source")
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    deadline = start + 60*minutes
    campaign = {"schema": "elendril-v715-diagnostic-campaign-v1", "version": "V7.15",
                "source": source, "protocol_sha256": PROTOCOL_SHA, "phase": "diagnostics",
                "status": "PREFLIGHT", "completed_jobs": 0, "expected_jobs": 9,
                "results": [], "scientific_status": "CAUSE UNRESOLVED"}
    try:
        check_source(REPO, source)
        _identity(output, source)
        immutable_json(output / "analysis" / "protocol.json", protocol()) if not (
            output / "analysis" / "protocol.json").exists() else None
        devices = _gpu_records()
        check_torch(REPO, devices)
        if shutil.disk_usage(output).free < 8*(1<<30) and not (output/"parent").exists():
            raise ValueError("NO_GO: need at least 8 GiB free to restore parent checkpoints")
        campaign["parent"] = _parent(parent_archive, output / "parent")
        expected = _coverage(output / "parent")
        train = read_dataset(output / "parent" / "data" / "train.json")
        suite, audits = suite_and_audit(train)
        portable = suite_to_record(suite)
        original = suite_from_record(portable)
        if set(original) != set(suite):
            raise ValueError("portable frozen suite roundtrip failed")
        path = output / "data" / "suite.json"
        if path.exists():
            if json.loads(path.read_bytes()) != portable:
                raise ValueError("existing diagnostic suite identity mismatch")
        else:
            immutable_json(path, portable)
        audit_file = output / "analysis" / "qualification.json"
        if not audit_file.exists():
            immutable_json(audit_file, audits)
        queue = Queue()
        for job in registered_jobs():
            queue.put(job)
        def lane(index):
            records = []
            while time.monotonic() < deadline - 600:
                try: job = queue.get_nowait()
                except Empty: break
                ref = expected[job["job_id"]]
                req = {
                    "source": source, "protocol_sha": PROTOCOL_SHA, "job": job,
                    "output": str(output / "analysis" / "jobs" / job["job_id"]),
                    "parent_output": str(output / "parent"),
                    "suite_file": str(path), "checkpoint": ref["checkpoint"],
                    "expected_weight_sha256": ref["weight_sha256"],
                    "deadline_monotonic": deadline,
                    "timecap_seconds": max(1, min(21600, deadline-time.monotonic()-600)),
                }
                row = invoke("frozen", req, devices[index], module="tools.elendril_v715_worker")
                records.append({**row, "job_id": job["job_id"]})
                if row["status"] != "COMPLETE":
                    break
            return records
        campaign["status"] = "RUNNING"
        with ThreadPoolExecutor(max_workers=2) as executor:
            campaign["results"] = [
                item for lane_rows in executor.map(lane, range(2)) for item in lane_rows
            ]
        campaign["completed_jobs"] = sum(x["status"] == "COMPLETE" for x in campaign["results"])
        if campaign["completed_jobs"] != 9:
            raise RuntimeError("partial frozen results: completion requires all nine jobs")
        replay = _replay(output, suite, expected)
        campaign["replay"] = replay
        immutable_json(output / "analysis" / "SUMMARY.json", _summary(output))
        campaign["status"] = "EXECUTED"
    except Exception as exc:
        campaign.update(status="PARTIAL_OR_NO_GO", error_type=type(exc).__name__,
                        error=str(exc), traceback=traceback.format_exc())
    finally:
        campaign["wall_seconds_before_export"] = time.monotonic() - start
        snapshot = output / "analysis" / "CAMPAIGN_STATUS.json"
        # Write a fresh resumable status and don't falsely mark prior partial runs COMPLETE.
        if snapshot.exists():
            snapshot.unlink()
        immutable_json(snapshot, campaign)
        try:
            delivery = _zip(output)
            campaign["analysis_export"] = delivery
        except Exception as exc:
            campaign["status"] = "DELIVERY_FAILED"
            campaign["export_error"] = f"{type(exc).__name__}: {exc}"
        else:
            if campaign["status"] == "EXECUTED":
                campaign["status"] = "COMPLETE"
        final = output / "exports" / "FINAL_STATUS.json"
        if final.exists():
            final.unlink()
        immutable_json(final, campaign)
    return campaign


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--parent-archive", type=Path, required=True)
    p.add_argument("--minutes", type=int, default=330)
    p.add_argument("--owner-enabled", action="store_true")
    args = p.parse_args(argv)
    campaign = run(output=args.output, source=args.source, parent_archive=args.parent_archive,
                   minutes=args.minutes, owner_enabled=args.owner_enabled)
    print(json.dumps({k: campaign.get(k) for k in (
        "status", "completed_jobs", "analysis_export", "error", "export_error"
    )}, indent=2))
    return 0 if campaign["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
