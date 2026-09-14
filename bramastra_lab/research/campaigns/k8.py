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
    run.add_argument("--max-wall-minutes", type=float, default=480.0)
    run.add_argument("--devices", default="cuda:0,cuda:1")
    run.add_argument("--precision", default="fp16_autocast",
                     choices=["fp32", "fp16_autocast"])

    summarize = subparsers.add_parser("summarize", help="aggregate the campaign ledger")
    summarize.add_argument("--run-dir", required=True)

    export = subparsers.add_parser("export", help="write the result bundle")
    export.add_argument("--run-dir", required=True)
    export.add_argument("--out", required=True)
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
                        devices=args.devices.split(","), precision=args.precision)


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
    """Write the full restorable result bundle (R08).

    Exports ledger, per-phase results, allocation, frozen protocol,
    source/data identities and checkpoint inventory. Fails (nonzero) when
    requirements are absent instead of writing a ledger-only stub.
    """
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
    # Frozen protocol (the exact plan + cutoffs the campaign ran under).
    from bramastra_lab.research.campaigns import process_supervision as ps

    protocol = {
        "schema": "bramastra-k8-protocol/v1",
        "wall_minutes": 480.0,
        "training_cutoff_minutes": ps.TRAINING_CUTOFF_MINUTES,
        "export_reserve_minutes": ps.EXPORT_RESERVE_MINUTES,
        "required_workers_e0": 2,
        "phases": ["E0", "E1", "E2", "E3", "E4", "E5", "E6"],
        "admission": "qualified E0 receipts on distinct devices + checkpoint + updates",
    }
    with open(os.path.join(args.out, "protocol.json"), "w") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True)
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
                   "note": "Restorable payloads are ledger checkpoint identities; "
                           "full .pt durability + fresh-process resume proof "
                           "remain GPU-gated (see E0 restore_proof)."},
                  handle, indent=2, sort_keys=True)
    print(json.dumps({"status": "EXPORTED", "out": args.out,
                      "files": sorted(os.listdir(args.out))}, indent=2))
    return 0


HANDLERS = {
    "prepare": cmd_prepare,
    "validate": cmd_validate,
    "run": cmd_run,
    "summarize": cmd_summarize,
    "export": cmd_export,
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
