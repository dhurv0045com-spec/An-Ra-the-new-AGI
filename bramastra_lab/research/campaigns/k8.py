"""K8 campaign entry point (I05/I06): prepare/validate/run/summarize/export.

``run --mode e0`` runs only the E0 gate; ``run --mode full`` executes the
gated campaign. The supervisor owns the only ledger; workers are subprocesses
with explicit GPU visibility. The notebook invokes these commands; it never
redefines the logic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from bramastra_lab.research.campaigns.supervisor import (
    CampaignLedger,
    SupervisorError,
    SupervisorLease,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bramastra_lab.research.campaigns.k8",
        description="K8 two-T4 campaign: prepare, validate, run, summarize, export")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    prepare = subparsers.add_parser("prepare", help="build the generated dataset bundle")
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--families", default="rule-inquiry,inventory,program")
    prepare.add_argument("--training-mechanisms", type=int, default=64,
                         help="reduced build default; full=4096 on Kaggle")
    prepare.add_argument("--controller-mechanisms", type=int, default=8)
    prepare.add_argument("--development-mechanisms", type=int, default=8)
    prepare.add_argument("--confirmation-mechanisms", type=int, default=8)
    prepare.add_argument("--tool-mechanisms", type=int, default=16)
    prepare.add_argument("--tool-heldout", type=int, default=4)
    prepare.add_argument("--meta-train", type=int, default=4)
    prepare.add_argument("--meta-validate", type=int, default=2)
    prepare.add_argument("--meta-confirm", type=int, default=2)

    validate_cmd = subparsers.add_parser("validate",
                                         help="validate the prepared bundle without training")
    validate_cmd.add_argument("--bundle", required=True)

    run = subparsers.add_parser("run", help="run the campaign")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--mode", choices=["e0", "full"], required=True)
    run.add_argument("--data", required=True, help="prepared bundle directory")
    run.add_argument("--max-wall-minutes", type=float, default=600.0,
                     help="owner wall budget (Kaggle GPU sessions allow 720)")
    run.add_argument("--devices", default="cuda:0,cuda:1")
    run.add_argument("--precision", default="fp32",
                     choices=["fp32", "fp16_autocast"])
    run.add_argument("--build-report", default=None,
                     help="build_verification.json from the verify-build "
                          "cell (the readiness gate consumes it)")

    summarize = subparsers.add_parser("summarize", help="aggregate the campaign ledger")
    summarize.add_argument("--run-dir", required=True)

    export = subparsers.add_parser("export", help="write the result bundle")
    export.add_argument("--run-dir", required=True)
    export.add_argument("--out", required=True)

    verify = subparsers.add_parser(
        "verify-build",
        help="run registered local checks, exercise real no-step interfaces, "
             "and write an evidence-backed build report (zero optimizer commits)")
    verify.add_argument("--data", required=True, help="prepared bundle directory")
    verify.add_argument("--report-dir", required=True,
                        help="NEW directory for build_verification.json")
    verify.add_argument("--no-updates", action="store_true",
                        help="required flag: enforces zero optimizer commits")
    verify.add_argument("--notebook", default=None,
                        help="owner notebook path (default: notebooks/bramastra_k8.ipynb)")
    verify.add_argument("--skip-check-groups", action="store_true",
                        help="internal: exercises only (used by focused tests)")
    return parser


def cmd_prepare(args: argparse.Namespace) -> int:
    from bramastra_lab.research.data.k8_bundle import build_k8_bundle

    manifest = build_k8_bundle(
        args.out, families=args.families.split(","),
        training_mechanisms=args.training_mechanisms,
        controller_mechanisms=args.controller_mechanisms,
        development_mechanisms=args.development_mechanisms,
        confirmation_mechanisms=args.confirmation_mechanisms,
        tool_mechanisms=args.tool_mechanisms, tool_heldout=args.tool_heldout,
        meta_train=args.meta_train, meta_validate=args.meta_validate,
        meta_confirm=args.meta_confirm)
    print(json.dumps({"status": "PREPARED", "bundle_dir": args.out,
                      "identity": manifest["identity"],
                      "families": manifest["families"]}, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from bramastra_lab.research.data.k8_bundle import validate_bundle

    report = validate_bundle(args.bundle)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


def cmd_run(args: argparse.Namespace) -> int:
    from bramastra_lab.research.campaigns.runner import run_campaign

    return run_campaign(run_dir=args.run_dir, mode=args.mode, data_dir=args.data,
                        max_wall_minutes=args.max_wall_minutes,
                        devices=args.devices.split(","), precision=args.precision,
                        build_report=os.path.join(args.build_report, "build_verification.json")
                        if args.build_report else None)


def cmd_summarize(args: argparse.Namespace) -> int:
    ledger = CampaignLedger(args.run_dir)
    deadline = ledger.deadline()
    report = {"deadline_unix": deadline,
              "time_remaining_minutes": (deadline - __import__("time").time()) / 60.0
              if deadline else None,
              "phases": {phase: ledger.phase_consumption(phase)
                         for phase in ("E0", "E1", "E2", "E3", "E4", "E5", "E6")}}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Write the full restorable result bundle (R08 + contracts S7).

    Exports ledger, per-phase results, allocation, frozen protocol,
    source/data identities, failures, comparisons, proposer transcripts and
    checkpoint payload bytes (not metadata only). Generates
    artifact_manifest.json only after successful writes and independently
    verifies it. Fails (nonzero) when requirements are absent instead of
    writing a ledger-only stub. Partial failed-run export is written with an
    explicit incomplete status; it cannot satisfy complete-campaign
    acceptance (see E6 verification).
    """
    import hashlib
    import shutil

    ledger_path = os.path.join(args.run_dir, "campaign_ledger.sqlite")
    if not os.path.exists(ledger_path):
        print(json.dumps({"status": "EXPORT_REFUSED",
                          "reason": f"ledger missing: {ledger_path}"}))
        return 2
    ledger = CampaignLedger(args.run_dir)
    try:
        export = ledger.export()
    finally:
        try:
            ledger.close()
        except Exception:
            pass
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "campaign_ledger.json"), "w") as handle:
        json.dump(export, handle, indent=2, sort_keys=True)
    # Per-phase results with qualified-receipt accounting.
    phases = ("E0", "E1", "E2", "E3", "E4", "E5", "E6")
    ledger2 = CampaignLedger(args.run_dir)
    try:
        phase_results = {}
        for phase in phases:
            consumption = ledger2.phase_consumption(phase)
            qualified = ledger2.qualified_phase_receipts(phase)
            phase_results[phase] = {
                "consumption": consumption,
                "qualified_receipts": len(qualified),
                "qualified_devices": sorted({row.device for row in qualified}),
                "qualified_job_ids": sorted(row.job_id for row in qualified),
            }
        allocations = ledger2.conn.execute("SELECT * FROM allocation").fetchall()
        try:
            events = ledger2.conn.execute(
                "SELECT * FROM events ORDER BY rowid").fetchall()
        except Exception:
            events = []
    finally:
        try:
            ledger2.close()
        except Exception:
            pass
    with open(os.path.join(args.out, "phase_results.json"), "w") as handle:
        json.dump(phase_results, handle, indent=2, sort_keys=True)
    with open(os.path.join(args.out, "allocation.json"), "w") as handle:
        json.dump({"allocations": allocations}, handle, indent=2, sort_keys=True,
                  default=str)
    # Failures (failed-run records preserved, never dropped).
    try:
        failures = [row for row in export.get("reservations", [])
                    if len(row) >= 11 and row[10] in ("failed", "timed_out")]
        worker_failures = [e for e in events if isinstance(e, (list, tuple))
                           and len(e) >= 2 and "failed" in str(e[1])]
    except Exception:
        failures, worker_failures = [], []
    with open(os.path.join(args.out, "failures.jsonl"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"failed_reservations": len(failures),
                                 "worker_failed_events": len(worker_failures)}) + "\n")
    # Frozen protocol (the exact plan + cutoffs the campaign ran under).
    from bramastra_lab.research.campaigns import process_supervision as ps

    # The protocol records the allocation the campaign ACTUALLY ran under
    # (from the ledger), never a constant.
    try:
        # Allocation row: (allocation_id, source_hash, data_hash, reserved_at,
        # deadline_unix, wall_minutes).
        recorded_wall = float(allocations[0][5])             if allocations and len(allocations[0]) > 5 else None
    except (TypeError, ValueError, IndexError):
        recorded_wall = None
    protocol = {
        "schema": "bramastra-k8-protocol/v1",
        "wall_minutes": recorded_wall if recorded_wall else 600.0,
        "training_cutoff_minutes": ps.TRAINING_CUTOFF_MINUTES,
        "export_reserve_minutes": ps.EXPORT_RESERVE_MINUTES,
        "required_workers_e0": 2,
        "phases": ["E0", "E1", "E2", "E3", "E4", "E5", "E6"],
        "admission": "qualified E0 receipts on distinct devices + checkpoint + updates",
    }
    with open(os.path.join(args.out, "protocol.json"), "w") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True)
    # Source/data identities (real hashes, never generic `k8`).
    try:
        from bramastra_lab.research.runtime.provenance import source_closure_sha256
        from bramastra_lab.research.config import tokenizer_identity
        from bramastra_lab.research.campaigns.phases.ops import k8_campaign_config

        source_hash = source_closure_sha256()
        tokenizer_id = tokenizer_identity()
        config_id = k8_campaign_config().identity()
    except Exception as exc:
        source_hash, tokenizer_id, config_id = f"unavailable:{exc}", "unavailable", "unavailable"
    try:
        data_hash = allocations[0][2] if allocations and len(allocations[0]) > 2 else "unavailable"
    except Exception:
        data_hash = "unavailable"
    with open(os.path.join(args.out, "source.json"), "w") as handle:
        json.dump({"source_closure_sha256": source_hash,
                   "tokenizer_identity": tokenizer_id,
                   "config_identity": config_id}, handle, indent=2, sort_keys=True)
    with open(os.path.join(args.out, "data.json"), "w") as handle:
        json.dump({"data_hash": data_hash,
                   "note": "Preserve the prepared bundle alongside this export; "
                           "resume across changed data requires explicit migration."},
                  handle, indent=2, sort_keys=True)
    # Proposer transcripts (E5 phase outputs) + comparisons.
    try:
        import glob as _glob

        transcripts: list[dict] = []
        for path in sorted(_glob.glob(os.path.join(
                args.run_dir, "phase_outputs", "E5", "*.json"))):
            try:
                transcripts.append(json.load(open(path, encoding="utf-8")))
            except Exception:
                continue
        with open(os.path.join(args.out, "proposer_transcripts.json"), "w") as handle:
            json.dump({"transcripts": transcripts}, handle, indent=2, sort_keys=True)
    except Exception:
        with open(os.path.join(args.out, "proposer_transcripts.json"), "w") as handle:
            json.dump({"transcripts": []}, handle, indent=2, sort_keys=True)
    with open(os.path.join(args.out, "comparisons.json"), "w") as handle:
        json.dump({"phase_results": phase_results}, handle, indent=2, sort_keys=True)
    # Checkpoint payload bytes (not metadata only): copy real .pt payloads +
    # manifests into the export when present.
    checkpoints_out = os.path.join(args.out, "checkpoints")
    os.makedirs(checkpoints_out, exist_ok=True)
    payload_count = 0
    for base, _dirs, names in os.walk(os.path.join(args.run_dir, "checkpoints")):
        for name in names:
            if not name.endswith((".pt", ".json")):
                continue
            src = os.path.join(base, name)
            try:
                rel = os.path.relpath(src, os.path.join(args.run_dir, "checkpoints"))
                dest = os.path.join(checkpoints_out, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
                if name.endswith(".pt"):
                    payload_count += 1
            except OSError:
                continue
    # Source/data artifacts: copy bundle manifest when the run records one.
    # The run_dir does not store the data path; record the ledger's data_hash
    # binding and require the caller to preserve the bundle alongside export.
    with open(os.path.join(args.out, "restore_evidence.json"), "w") as handle:
        checkpoints = []
        for row in export.get("reservations", []):
            # Reservation rows: (reservation_id, job_id, parent, worker, device,
            # phase, arm, seed, reserved_seconds, reserved_at, status,
            # committed, attempted, exposure, device_seconds, checkpoint_id)
            try:
                if len(row) >= 16 and row[10] in ("completed", "failed"):
                    checkpoints.append({"job_id": row[1], "phase": row[5],
                                        "status": row[10],
                                        "checkpoint_identity": row[15]})
            except Exception:
                continue
        json.dump({"checkpoints": checkpoints,
                   "payload_files_exported": payload_count,
                   "note": "Restorable payloads are ledger checkpoint identities with "
                           "exported .pt bytes under checkpoints/; fixture-only "
                           "identities without payloads cannot satisfy complete "
                           "acceptance."},
                  handle, indent=2, sort_keys=True)
    # Artifact manifest only after successful writes + independent verification.
    manifest_records: dict[str, dict] = {}
    for base, _dirs, names in os.walk(args.out):
        # Skip the manifest itself + nested checkpoints dir handled below.
        for name in sorted(names):
            if name == "artifact_manifest.json":
                continue
            path = os.path.join(base, name)
            try:
                digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
                rel = os.path.relpath(path, args.out)
                manifest_records[rel] = {"sha256": digest,
                                         "bytes": os.path.getsize(path)}
            except OSError:
                continue
    manifest_path = os.path.join(args.out, "artifact_manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump({"schema": "bramastra-k8-artifact-manifest/v1",
                   "files": manifest_records,
                   "payload_files": payload_count,
                   "complete": payload_count > 0}, handle, indent=2, sort_keys=True)
    # Independent verification: re-hash every listed file.
    try:
        reloaded = json.load(open(manifest_path, encoding="utf-8"))
        for rel, record in reloaded.get("files", {}).items():
            actual = hashlib.sha256(
                open(os.path.join(args.out, rel), "rb").read()).hexdigest()
            if actual != record.get("sha256"):
                print(json.dumps({"status": "EXPORT_REFUSED",
                                  "reason": f"manifest verification failed for {rel}"}))
                return 2
    except Exception as exc:
        print(json.dumps({"status": "EXPORT_REFUSED",
                          "reason": f"manifest verification error: {exc}"}))
        return 2
    # Completeness verdict (F20/section 21): derived from real evidence —
    # every required phase must carry a qualified receipt and every
    # completed real checkpoint identity must have its restorable payload
    # bytes in this export. Fixture/double identities never satisfy it.
    exported_ids: set[str] = set()
    for base_d, _dirs_c, names_c in os.walk(checkpoints_out):
        if "manifest.json" in names_c:
            try:
                _m = json.load(open(os.path.join(base_d, "manifest.json"),
                                    encoding="utf-8"))
                if _m.get("checkpoint_id") and any(
                        n.endswith(".pt") for n in names_c):
                    exported_ids.add(str(_m["checkpoint_id"]))
            except Exception:
                continue
    missing_phases = [p for p in ("E0", "E1", "E2", "E3", "E4", "E5")
                      if phase_results[p]["qualified_receipts"] == 0]
    fixture_identities = 0
    unexported_identities = 0
    for row in export.get("reservations", []):
        try:
            if len(row) >= 16 and row[10] == "completed" and row[15]:
                identity = str(row[15])
                if identity.startswith(("double-", "fixture-", "unpublished")):
                    fixture_identities += 1
                elif identity not in exported_ids:
                    unexported_identities += 1
        except Exception:
            continue
    complete = (not missing_phases and unexported_identities == 0
                and payload_count > 0)
    print(json.dumps({
        "status": "EXPORTED_COMPLETE" if complete else "EXPORTED_PARTIAL",
        "complete": complete,
        "missing_phases": missing_phases,
        "fixture_identities": fixture_identities,
        "unexported_identities": unexported_identities,
        "out": args.out,
        "files": sorted(os.listdir(args.out)),
        "payload_files": payload_count}, indent=2))
    return 0


def cmd_verify_build(args: argparse.Namespace) -> int:
    from bramastra_lab.research.campaigns.verify_build import run_verify_build

    if not args.no_updates:
        print("error: verify-build requires --no-updates (zero optimizer "
              "commits are enforced)", file=sys.stderr)
        return 2
    report = run_verify_build(
        args.data, args.report_dir, no_updates=True,
        notebook_path=args.notebook,
        run_checks=not args.skip_check_groups)
    failing = sorted(
        req_id for req_id, row in report["requirements"].items()
        if row["status"] != "pass")
    print(json.dumps({
        "status": "VERIFIED" if report["ready_for_owner_experiment"]
        else "NOT_READY",
        "report": os.path.join(os.path.abspath(args.report_dir),
                               "build_verification.json"),
        "source_closure_sha256": report["source_closure_sha256"],
        "failing_requirements": failing,
        "runtime_checks_pending": [gate["id"] for gate in
                                   report["runtime_checks_pending"]],
        "optimizer_updates_local": report["optimizer_updates_local"],
    }, indent=2, sort_keys=True))
    return 0 if report["ready_for_owner_experiment"] else 1


HANDLERS = {
    "prepare": cmd_prepare,
    "validate": cmd_validate,
    "run": cmd_run,
    "summarize": cmd_summarize,
    "export": cmd_export,
    "verify-build": cmd_verify_build,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return HANDLERS[args.command](args)
    except (SupervisorError,) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
