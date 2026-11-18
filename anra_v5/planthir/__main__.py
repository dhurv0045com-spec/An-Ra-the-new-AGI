"""Command-line entry point for Planthir declarative experiments, diagnostics, and verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .agent_tools import (
    LEGACY_GPU_ERROR,
    LEGACY_STUDY_CLI_ERROR,
    InspectionError,
    agent_info,
    build_cartridge_from_spec,
    explain_error_code,
    init_spec_template,
    inspect_run,
    planthir_doctor,
    planthir_spec_schema,
    validate_spec_document,
)
from .cartridge import CartridgeRunner
from .evidence import DEFAULT_RESULT_ZIP_MAX_BYTES, resolve_published_generation, verify_result_bundle
from .instrument import run_fault_injection_suite
from .registry import build_historical_atlas
from .replay import build_custody_receipt, independent_replay_bundle
from .runtime import rescore_compact_analysis_archive
from .validation_study import (
    run_exp_004_live_gpu_cartridge_validation,
    run_validation_study,
)


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _load_spec_json(spec_path: str | Path) -> Any:
    path = Path(spec_path)
    if not path.is_file():
        raise FileNotFoundError(f"spec file not found: {path}")
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-JSON constant {c}")),
    )


def _execute_spec(args: argparse.Namespace, *, resume_flag: bool) -> int:
    canary_override = None if getattr(args, "canary", None) is None else bool(args.canary)
    try:
        doc = _load_spec_json(args.spec)
        validation = validate_spec_document(doc, canary_override=canary_override)
        if not validation["valid"]:
            _emit(
                {
                    "status": "INVALID_SPEC",
                    "error": "INVALID_SPEC",
                    "validation": validation,
                    "hint": explain_error_code(validation["errors"][0]["code"]) if validation.get("errors") else None,
                }
            )
            return 1
        cartridge, run_kwargs = build_cartridge_from_spec(doc, canary_override=canary_override)
        if getattr(args, "device", None) is not None:
            run_kwargs["device"] = str(args.device)
        out = CartridgeRunner().run(
            cartridge,
            work_root=Path(args.work_root),
            resume=resume_flag,
            **run_kwargs,
        )
        _emit(
            {
                "status": "COMPLETE",
                "campaign_id": out["campaign_id"],
                "backend_id": cartridge.backend_id,
                "device": out["device"],
                "work_root": str(Path(args.work_root)),
                "generation_id": out["bundle_receipt"].get("generation_id"),
                "bundle_dir": out["bundle_receipt"].get("output_dir"),
                "gates_passed": out["gates"]["all_5_gates_passed"],
                "checkpoint_sha256_by_arm": {
                    aid: info["checkpoint_sha256"] for aid, info in out["arms"].items()
                },
                "ai_handoff_brief": out["ai_handoff_brief"],
            }
        )
        return 0
    except Exception as exc:
        msg = str(exc)
        code = "RESUME_REQUIRED" if "resume required" in msg.lower() else type(exc).__name__
        _emit(
            {
                "status": "FAILED",
                "error": code,
                "detail": msg,
                "work_root": str(getattr(args, "work_root", "")),
                "explanation": explain_error_code(code),
            }
        )
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m anra_v5.planthir",
        description="Planthir Universal Research Infrastructure CLI (declarative data-only JSON workflows)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("agent-info", help="Print read-only contract derived from dataclasses and registered backends.")
    sub.add_parser("doctor", help="Run fast read-only binding & installation integrity self-check (`green_signal`).")
    sub.add_parser("schema", help="Print strict anra-planthir-spec-v1 data-only JSON schema.")

    p_init = sub.add_parser("init", help="Generate a valid canary anra-planthir-spec-v1 JSON template.")
    p_init.add_argument("--backend", default="x1_rd2", help="Registered backend ID (e.g. v8, x1_rd2, x1_s20, or custom).")
    p_init.add_argument("--out", default=None, help="Optional output file path to write JSON template.")
    p_init.add_argument("--no-canary", action="store_true", help="Disable CPU canary flag in template.")

    p_val = sub.add_parser("validate", help="Fail-closed preflight validation of a data-only spec JSON.")
    p_val.add_argument("--spec", required=True)
    p_val.add_argument("--canary", dest="canary", action="store_true", default=None)
    p_val.add_argument("--no-canary", dest="canary", action="store_false")

    for cmd_name, help_text in (
        ("run", "Validate and execute a declarative Planthir experiment spec."),
        ("resume", "Resume an interrupted Planthir run cleanly from committed checkpoints."),
    ):
        p_run = sub.add_parser(cmd_name, help=help_text)
        p_run.add_argument("--spec", required=True)
        p_run.add_argument("--work-root", required=True)
        p_run.add_argument("--device", default=None)
        p_run.add_argument("--canary", dest="canary", action="store_true", default=None)
        p_run.add_argument("--no-canary", dest="canary", action="store_false")
        if cmd_name == "run":
            p_run.add_argument("--resume", action="store_true", default=False)

    p_insp = sub.add_parser("inspect-run", help="Inspect a run directory without modifying files.")
    p_insp.add_argument("--work-root", required=True)

    p_ver = sub.add_parser("verify-bundle", help="Verify exported result.zip, checkpoint_full.zip, CURRENT.json, and custody sidecars.")
    p_ver.add_argument("--bundle-dir", required=True)
    p_ver.add_argument("--expected-source-commit", default=None)
    p_ver.add_argument("--max-result-zip-bytes", type=int, default=DEFAULT_RESULT_ZIP_MAX_BYTES)
    p_ver.add_argument("--raw-evidence-dir", default=None)

    p_rep = sub.add_parser("replay-bundle", help="Extract bundle in isolation and independently replay checkpoints.")
    p_rep.add_argument("--bundle-dir", required=True)
    p_rep.add_argument("--expected-source-commit", default=None)
    p_rep.add_argument("--raw-evidence-dir", default=None)
    p_rep.add_argument("--device", default="cpu")
    p_rep.add_argument("--max-result-zip-bytes", type=int, default=DEFAULT_RESULT_ZIP_MAX_BYTES)
    p_rep.add_argument("--output", default=None)

    p_exp = sub.add_parser("explain-error", help="Explain a Planthir validation/runtime error code and remediation.")
    p_exp.add_argument("--code", required=True)

    sub.add_parser("calibrate", help="Run the 6-fault controlled instrument calibration suite on CPU.")

    p_atlas = sub.add_parser("historical-atlas", help="Print historical AN-RA version registry summary.")
    p_atlas.add_argument("--output-json", default=None)

    p_rescore = sub.add_parser("rescore-archive", help="Verify a compact analysis archive and rebuild RESULTS.md + METRICS.json in a fresh dir.")
    p_rescore.add_argument("--archive", required=True)
    p_rescore.add_argument("--out", required=True)

    sub.add_parser("run-validation-study", help="Deprecated CLI alias; directs callers to schema/init/validate/run.")
    sub.add_parser("run-gpu-validation", help="Legacy command rejected until interactive GPU qualification completes.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "agent-info":
        _emit(agent_info())
        return 0

    if args.command == "doctor":
        rep = planthir_doctor()
        _emit(rep)
        return 0 if rep["green_signal"] else 1

    if args.command == "schema":
        _emit(planthir_spec_schema())
        return 0

    if args.command == "init":
        tpl = init_spec_template(backend_id=args.backend, canary=not bool(args.no_canary))
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(tpl, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _emit({"status": "TEMPLATE_WRITTEN", "out": str(out_path), "spec": tpl})
        else:
            _emit(tpl)
        return 0

    if args.command == "validate":
        canary_override = None if args.canary is None else bool(args.canary)
        try:
            doc = _load_spec_json(args.spec)
            report = validate_spec_document(doc, canary_override=canary_override)
            _emit(report)
            return 0 if report["valid"] else 1
        except Exception as exc:
            _emit({"valid": False, "status": "INVALID_SPEC", "errors": [{"code": type(exc).__name__, "detail": str(exc)}]})
            return 1

    if args.command == "run":
        return _execute_spec(args, resume_flag=bool(args.resume))

    if args.command == "resume":
        return _execute_spec(args, resume_flag=True)

    if args.command == "inspect-run":
        try:
            _emit(inspect_run(args.work_root))
            return 0
        except InspectionError as exc:
            _emit({"error": {"file": exc.file, "detail": exc.detail}})
            return 1

    if args.command == "verify-bundle":
        bdir = Path(args.bundle_dir)
        try:
            if (bdir / "CURRENT.json").is_file():
                ptr = resolve_published_generation(
                    bdir,
                    expected_source_commit=args.expected_source_commit,
                    max_result_zip_bytes=int(args.max_result_zip_bytes),
                    verify=True,
                )
                custody = build_custody_receipt(
                    bdir,
                    expected_source_commit=args.expected_source_commit,
                    max_result_zip_bytes=int(args.max_result_zip_bytes),
                    raw_evidence_dir=args.raw_evidence_dir,
                )
                _emit({"verified": True, "pointer": ptr, "custody": custody})
            else:
                res = verify_result_bundle(
                    bdir,
                    expected_source_commit=args.expected_source_commit,
                    max_result_zip_bytes=int(args.max_result_zip_bytes),
                )
                _emit(res)
            return 0
        except Exception as exc:
            _emit({"verified": False, "error": type(exc).__name__, "detail": str(exc)})
            return 1

    if args.command == "replay-bundle":
        try:
            receipt = independent_replay_bundle(
                args.bundle_dir,
                expected_source_commit=args.expected_source_commit,
                raw_evidence_dir=args.raw_evidence_dir,
                device=str(args.device),
                max_result_zip_bytes=int(args.max_result_zip_bytes),
                output_path=args.output,
                independent_environment=True,
            )
            _emit(receipt)
            return 0 if receipt["verified"] else 1
        except Exception as exc:
            _emit({"verified": False, "status": "REPLAY_FAILED", "error": type(exc).__name__, "detail": str(exc)})
            return 1

    if args.command == "explain-error":
        _emit(explain_error_code(args.code))
        return 0

    if args.command == "calibrate":
        report = run_fault_injection_suite()
        _emit(report)
        return 0 if report["all_detected"] else 1

    if args.command == "historical-atlas":
        report = build_historical_atlas(output_json=Path(args.output_json) if args.output_json else None)
        _emit(report)
        return 0

    if args.command == "rescore-archive":
        try:
            receipt = rescore_compact_analysis_archive(args.archive, args.out)
            _emit(receipt)
            return 0 if receipt["verified"] else 1
        except Exception as exc:
            _emit({"verified": False, "error": type(exc).__name__, "detail": str(exc)})
            return 1

    if args.command == "run-validation-study":
        _emit(
            {
                "error": LEGACY_STUDY_CLI_ERROR,
                "detail": "Hardcoded CLI validation study launcher is deprecated on CLI; use the declarative anra-planthir-spec-v1 workflow.",
                "migrate_to": [
                    "python -m anra_v5.planthir schema",
                    "python -m anra_v5.planthir init --backend x1_rd2 --out spec.json",
                    "python -m anra_v5.planthir validate --spec spec.json",
                    "python -m anra_v5.planthir run --spec spec.json --work-root runs/exp1 --canary",
                ],
                "python_api_note": "anra_v5.planthir.validation_study.run_validation_study remains importable in Python for backward compatibility.",
            }
        )
        return 1

    if args.command == "run-gpu-validation":
        _emit(
            {
                "error": LEGACY_GPU_ERROR,
                "detail": "GPU cloud launcher requires interactive owner GPU qualification first; use `agent-info` / `schema` / `run`.",
            }
        )
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
