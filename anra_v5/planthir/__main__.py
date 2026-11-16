"""CLI entry point for the Planthir universal research observatory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .evidence import (
    DEFAULT_RESULT_ZIP_MAX_BYTES,
    verify_result_bundle,
    verify_single_planthir_zip,
)
from .instrument import run_fault_injection_suite
from .registry import build_historical_atlas
from .validation_study import (
    run_exp_004_live_gpu_cartridge_validation,
    run_validation_study,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="planthir",
        description="AN-RA Planthir Universal Scientific Observatory CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("agent-info", help="Describe the supported agent interface as JSON.")
    p_inspect = sub.add_parser("inspect-run", help="Inspect run metadata without modifying it.")
    p_inspect.add_argument("--work-root", required=True)

    sub.add_parser(
        "calibrate",
        help="Run the 6-fault controlled instrument calibration suite on CPU.",
    )

    p_atlas = sub.add_parser(
        "historical-atlas",
        help="Build the longitudinal V6-V8 historical atlas JSON from Experiment Result/.",
    )
    p_atlas.add_argument("--repo-root", default=".", help="Repository root path.")

    p_val = sub.add_parser(
        "run-validation-study",
        help="Execute bounded CPU validation (EXP-PLANTHIR-001/002/003) and write curated reports.",
    )
    p_val.add_argument("--repo-root", default=".", help="Repository root path.")
    p_val.add_argument(
        "--output-dir",
        default="Experiment Result/Planthir",
        help="Output directory for Tier-1 curated .md and .json artifacts.",
    )

    p_gpu = sub.add_parser(
        "run-gpu-validation",
        help="Execute Level-5 5-Gate Cartridge validation live on GPU (EXP-PLANTHIR-004).",
    )
    p_gpu.add_argument(
        "--output-dir",
        default="Experiment Result/Planthir",
        help="Output directory for curated EXP-PLANTHIR-004 reports and AI handoff brief.",
    )
    p_gpu.add_argument(
        "--updates",
        type=int,
        default=60,
        help="Updates per arm for bounded live validation (default: 60).",
    )
    p_gpu.add_argument(
        "--device",
        default="auto",
        help="Target device ('auto', 'cuda:0', or 'cpu').",
    )

    p_verify = sub.add_parser(
        "verify-bundle",
        help="Verify SHA-256 integrity of a Planthir result.zip / checkpoint_full.zip bundle directory or single archive.",
    )
    target_grp = p_verify.add_mutually_exclusive_group(required=True)
    target_grp.add_argument("--dir", help="Bundle directory containing result.zip, checkpoint_full.zip, and metadata files.")
    target_grp.add_argument("--zip", help="Single Planthir zip archive (result.zip or checkpoint_full.zip) to verify.")
    p_verify.add_argument("--source-commit", default=None, help="Optional expected 40-char git source_commit.")
    p_verify.add_argument(
        "--max-result-bytes",
        type=int,
        default=DEFAULT_RESULT_ZIP_MAX_BYTES,
        help="Maximum allowed size in bytes for result.zip (default: 20 MiB).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command in ("agent-info", "inspect-run"):
        from .agent_tools import InspectionError, agent_info, inspect_run
        try:
            report = agent_info() if args.command == "agent-info" else inspect_run(args.work_root)
        except InspectionError as exc:
            print(json.dumps({"schema": "anra-planthir-inspection-error-v1", "error": {"file": exc.file, "detail": exc.detail}}, sort_keys=True))
            return 1
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "calibrate":
        report = run_fault_injection_suite()
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0 if report["all_detected"] and report["healthy_v8_kernel_passed"] else 1
    if args.command == "historical-atlas":
        atlas = build_historical_atlas(Path(args.repo_root))
        print(json.dumps(atlas, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "run-validation-study":
        summary = run_validation_study(Path(args.repo_root), Path(args.output_dir))
        digest_view = {
            "EXP-PLANTHIR-001": {
                "all_faults_detected": summary["EXP-PLANTHIR-001"]["fault_injection_suite"]["all_detected"],
                "detected_faults": summary["EXP-PLANTHIR-001"]["fault_injection_suite"]["detected_faults"],
                "elapsed_seconds": summary["EXP-PLANTHIR-001"]["elapsed_seconds"],
            },
            "EXP-PLANTHIR-002": {
                "arms": sorted(summary["EXP-PLANTHIR-002"]["arms"].keys()),
                "elapsed_seconds": summary["EXP-PLANTHIR-002"]["elapsed_seconds"],
                "single_architecture_contrast": summary["EXP-PLANTHIR-002"]["checkpoint_diff_J_vs_F"]["single_architecture_contrast"],
            },
            "EXP-PLANTHIR-003": {
                "generations_indexed": summary["EXP-PLANTHIR-003"]["atlas"]["generations_indexed"],
                "total_archived_runs_indexed": summary["EXP-PLANTHIR-003"]["atlas"]["total_archived_runs_indexed"],
            },
        }
        print(json.dumps(digest_view, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "run-gpu-validation":
        print(json.dumps({"schema": "anra-planthir-cli-error-v1", "error": "legacy_launcher_not_qualified", "detail": "This legacy workflow requires migration. Use agent-info for the current Python runner interface."}, sort_keys=True))
        return 1
    if args.command == "verify-bundle":
        if args.dir is not None:
            report = verify_result_bundle(
                Path(args.dir),
                expected_source_commit=args.source_commit,
                max_result_zip_bytes=int(args.max_result_bytes),
            )
        else:
            report = verify_single_planthir_zip(
                Path(args.zip),
                expected_source_commit=args.source_commit,
                max_bytes=int(args.max_result_bytes)
                if Path(args.zip).name == "result.zip"
                else None,
            )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
