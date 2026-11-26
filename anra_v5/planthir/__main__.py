"""Command-line entry point for Planthir declarative experiments, diagnostics, and verification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from .backend import load_backend_modules
from .reporting import rescore_compact_analysis_archive
from .resources import estimate_production_preflight
from .studio import inspect_campaign_status, level_info, scaffold_experiment_module
from .archives import verify_session_archive

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
from .instrument import run_fault_injection_suite, run_level7_runtime_shield_suite
from .registry import build_historical_atlas
from .replay import ReplayVerificationError, build_custody_receipt, independent_replay_bundle
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
    parser.add_argument(
        "--backend-module",
        dest="backend_modules",
        action="append",
        default=[],
        help="Dotted plugin module(s) exposing custom ExperimentModule / register_planthir_backends() (repeatable).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_lvl = sub.add_parser("level", help="Print Level-7 platform architecture, subsystems, backends, and quickstart CLI.")
    p_ai = sub.add_parser("agent-info", help="Print read-only contract derived from dataclasses and registered backends.")
    p_doc = sub.add_parser("doctor", help="Run fast read-only binding & installation integrity self-check (`green_signal`).")
    p_sch = sub.add_parser("schema", help="Print strict anra-planthir-spec-v1 data-only JSON schema.")

    p_stat = sub.add_parser("status", help="Inspect a campaign work directory read-only and print JSON or Markdown dashboard.")
    p_stat.add_argument("--work-root", required=True)
    p_stat.add_argument("--markdown", action="store_true", default=False)

    p_var = sub.add_parser("verify-archive", help="Verify a session ZIP archive and optional SHA-256 sidecar / v6 manifest.")
    p_var.add_argument("--archive", required=True)
    p_var.add_argument("--expected-commit", default=None)
    p_var.add_argument("--max-bytes", type=int, default=None)
    p_var.add_argument("--no-sidecar", action="store_true", default=False)
    p_var.add_argument("--no-v6-manifest", action="store_true", default=False)

    p_scaf = sub.add_parser("scaffold", help="Generate a runnable Level-7 ExperimentModule plugin package and smoke test.")
    p_scaf.add_argument("--name", required=True)
    p_scaf.add_argument("--out-dir", required=True)
    p_scaf.add_argument("--overwrite", action="store_true", default=False)

    p_init = sub.add_parser("init", help="Generate a valid canary anra-planthir-spec-v1 JSON template.")
    p_init.add_argument("--backend", default="x1_rd2", help="Registered backend ID (e.g. v8, x1_rd2, x1_s20, or custom).")
    p_init.add_argument("--out", default=None, help="Optional output file path to write JSON template.")
    p_init.add_argument("--no-canary", action="store_true", help="Disable CPU canary flag in template.")

    p_val = sub.add_parser("validate", help="Fail-closed preflight validation of a data-only spec JSON.")
    p_val.add_argument("--spec", required=True)
    p_val.add_argument("--canary", dest="canary", action="store_true", default=None)
    p_val.add_argument("--no-canary", dest="canary", action="store_false")

    p_pre = sub.add_parser(
        "preflight",
        help="Estimate full total_updates VRAM/prepack bloat, generator scaling, wall-clock, and disk budgets.",
    )
    p_pre.add_argument("--spec", default=None, help="Optional anra-planthir-spec-v1 JSON spec to read schedule defaults.")
    p_pre.add_argument("--total-updates", type=int, default=None, help="Total training updates (default 1024 or spec value).")
    p_pre.add_argument("--num-jobs", type=int, default=None, help="Number of registered campaign jobs (default 1 or spec arms).")
    p_pre.add_argument("--parallel-workers", type=int, default=1, help="Parallel worker lanes (e.g. 2 for Kaggle 2xT4).")
    p_pre.add_argument("--model-params", type=int, default=0, help="Approximate parameters per worker.")
    p_pre.add_argument("--batch-size", type=int, default=None, help="Micro-batch / batch size per worker.")
    p_pre.add_argument("--seq-len", type=int, default=0, help="Maximum sequence length for activation estimation.")
    p_pre.add_argument("--hidden-dim", type=int, default=0, help="Model hidden width for activation estimation.")
    p_pre.add_argument("--num-layers", type=int, default=0, help="Model depth for activation estimation.")
    p_pre.add_argument("--precision", default=None, help="Numeric precision (fp16, amp, bf16, fp32).")
    p_pre.add_argument("--seconds-per-update", type=float, default=None, help="Measured seconds per update.")
    p_pre.add_argument("--prepack-mb", type=float, default=None, help="Prepacked payload / mask bytes per worker in MiB.")
    p_pre.add_argument("--per-job-disk-mb", type=float, default=128.0, help="Estimated disk bytes per job in MiB.")
    p_pre.add_argument("--startup-overhead-seconds", type=float, default=0.0, help="Per-wave startup overhead seconds.")
    p_pre.add_argument("--max-vram-gb", type=float, default=14.0, help="Per-worker VRAM budget in GiB (default 14.0).")
    p_pre.add_argument("--max-prepack-mb", type=float, default=1024.0, help="Per-worker prepack limit in MiB (default 1024.0).")
    p_pre.add_argument("--session-minutes", type=float, default=None, help="Session time ceiling in minutes (default 660.0).")
    p_pre.add_argument("--reserve-minutes", type=float, default=45.0, help="Session reserve minutes (default 45.0).")
    p_pre.add_argument("--max-disk-gb", type=float, default=None, help="Total disk budget in GiB (default 18.0).")
    p_pre.add_argument("--output", default=None, help="Optional file path to write JSON preflight receipt.")

    run_parsers: list[argparse.ArgumentParser] = []
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
        run_parsers.append(p_run)

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
    p_rep.add_argument("--expected-backend", default=None)
    p_rep.add_argument("--require-reference", action="store_true", default=False)
    p_rep.add_argument("--worlds-json", default=None)
    p_rep.add_argument("--raw-evidence-dir", default=None)
    p_rep.add_argument("--device", default="cpu")
    p_rep.add_argument("--max-result-zip-bytes", type=int, default=DEFAULT_RESULT_ZIP_MAX_BYTES)
    p_rep.add_argument("--output", default=None)

    for sp in (p_lvl, p_ai, p_doc, p_sch, p_init, p_val, *run_parsers, p_rep):
        sp.add_argument(
            "--backend-module",
            dest="sub_backend_modules",
            action="append",
            default=[],
            help="Dotted plugin module(s) exposing custom ExperimentModule / register_planthir_backends() (repeatable).",
        )

    p_exp = sub.add_parser("explain-error", help="Explain a Planthir validation/runtime error code and remediation.")
    p_exp.add_argument("--code", required=True)

    sub.add_parser("calibrate", help="Run the 6-fault controlled instrument calibration suite + Level-7 shield on CPU.")
    sub.add_parser("shield", help="Run the adversarial Level-7 runtime shield suite (0xB5 byte decode, FP16 -1e9, JSON sanitizer, archive caps).")

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

    mods = [*getattr(args, "backend_modules", ()), *getattr(args, "sub_backend_modules", ())]
    if mods:
        loaded = load_backend_modules(mods)
        existing = [x.strip() for x in os.environ.get("PLANTHIR_BACKEND_MODULES", "").split(",") if x.strip()]
        os.environ["PLANTHIR_BACKEND_MODULES"] = ",".join(dict.fromkeys([*existing, *loaded]))

    if args.command == "level":
        _emit(level_info())
        return 0

    if args.command == "status":
        rep = inspect_campaign_status(args.work_root)
        if args.markdown:
            print(rep["dashboard_markdown"])
        else:
            _emit(rep)
        return 0

    if args.command == "verify-archive":
        try:
            res = verify_session_archive(
                args.archive,
                max_bytes=args.max_bytes,
                expected_source_commit=args.expected_commit,
                require_sidecar_sha256=not bool(args.no_sidecar),
                require_v6_manifest=not bool(args.no_v6_manifest),
            )
            _emit(res)
            return 0
        except (ValueError, FileNotFoundError, OSError) as exc:
            _emit({"verified": False, "error": type(exc).__name__, "detail": str(exc)})
            return 2

    if args.command == "scaffold":
        try:
            res = scaffold_experiment_module(args.name, args.out_dir, overwrite=bool(args.overwrite))
            _emit(res)
            return 0
        except (ValueError, FileExistsError) as exc:
            _emit({"created": False, "error": type(exc).__name__, "detail": str(exc)})
            return 2

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
            loaded_worlds = (
                json.loads(Path(args.worlds_json).read_text(encoding="utf-8"))
                if args.worlds_json
                else None
            )
            receipt = independent_replay_bundle(
                args.bundle_dir,
                expected_source_commit=args.expected_source_commit,
                worlds=loaded_worlds,
                raw_evidence_dir=args.raw_evidence_dir,
                device=str(args.device),
                max_result_zip_bytes=int(args.max_result_zip_bytes),
                output_path=args.output,
                independent_environment=True,
                expected_backend_id=args.expected_backend,
                require_reference=bool(args.require_reference),
                use_archived_reference=bool(loaded_worlds is not None),
            )
            _emit(receipt)
            return 0 if receipt["verified"] else 1
        except ReplayVerificationError as exc:
            msg = str(exc)
            rec_status = str(exc.receipt.get("status", ""))
            if "INSUFFICIENT_REPLAY_EVIDENCE" in msg or rec_status == "INSUFFICIENT_REPLAY_EVIDENCE":
                err_payload: dict[str, Any] = {
                    "verified": False,
                    "status": "INSUFFICIENT_REPLAY_EVIDENCE",
                    "error": type(exc).__name__,
                    "detail": msg,
                    "defects": exc.defects,
                    "supported_recovery_path": exc.receipt.get(
                        "supported_recovery_path",
                        "Re-export bundle with current Planthir (embedding REPLAY_COHORT.json) "
                        "or supply --worlds-json / explicit reference_predictions.",
                    ),
                }
            else:
                err_payload = {
                    "verified": False,
                    "status": rec_status or "REPLAY_FAILED",
                    "error": type(exc).__name__,
                    "detail": msg,
                    "defects": exc.defects,
                    "mismatch_reasons": exc.mismatch_reasons,
                    "mismatch_counts_by_reason": exc.mismatch_counts_by_reason,
                    "total_mismatch_count": len(exc.mismatches),
                    "mismatches": exc.mismatches[:10],
                    "triage_hints": exc.triage_hints,
                    "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                }
            _emit(err_payload)
            if exc.mismatch_reasons:
                first_m = exc.mismatches[0] if exc.mismatches else {}
                hint = exc.triage_hints[0] if exc.triage_hints else ""
                print(
                    f"[replay-bundle] ORIGINAL_PREDICTION_MISMATCH reasons={exc.mismatch_reasons} "
                    f"first_world={first_m.get('world_id')!r} first_reason={first_m.get('reason')!r} "
                    f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')!r} hint={hint}",
                    file=sys.stderr,
                )
            return 1
        except Exception as exc:
            msg = str(exc)
            if "INSUFFICIENT_REPLAY_EVIDENCE" in msg:
                _emit(
                    {
                        "verified": False,
                        "status": "INSUFFICIENT_REPLAY_EVIDENCE",
                        "error": type(exc).__name__,
                        "detail": msg,
                        "supported_recovery_path": (
                            "Re-export bundle with current Planthir (embedding REPLAY_COHORT.json) "
                            "or supply --worlds-json / explicit reference_predictions."
                        ),
                    }
                )
            else:
                _emit({"verified": False, "status": "REPLAY_FAILED", "error": type(exc).__name__, "detail": msg})
            return 1

    if args.command == "preflight":
        try:
            spec_doc = _load_spec_json(args.spec) if args.spec else {}
            ctrl = spec_doc.get("control_arm") if isinstance(spec_doc.get("control_arm"), dict) else {}
            trts = spec_doc.get("treatment_arms") if isinstance(spec_doc.get("treatment_arms"), list) else []
            ctrl_rec = ctrl.get("recipe") if isinstance(ctrl.get("recipe"), dict) else {}
            exe = spec_doc.get("execution") if isinstance(spec_doc.get("execution"), dict) else {}

            tot_u = int(args.total_updates) if args.total_updates is not None else int(ctrl_rec.get("total_updates", 1024))
            n_jobs = int(args.num_jobs) if args.num_jobs is not None else (1 + len(trts) if ctrl else 1)
            bs = int(args.batch_size) if args.batch_size is not None else int(ctrl_rec.get("batch_size", 1))
            prec = str(args.precision) if args.precision is not None else str(ctrl_rec.get("precision", "fp16"))
            max_vram_gb = (
                float(args.max_vram_gb)
                if args.max_vram_gb is not None
                else (
                    float(exe["max_cuda_allocated_bytes"]) / float(1024**3)
                    if exe.get("max_cuda_allocated_bytes") is not None
                    else 14.0
                )
            )
            sess_min = (
                float(args.session_minutes)
                if args.session_minutes is not None
                else (
                    float(exe["max_wall_seconds"]) / 60.0
                    if exe.get("max_wall_seconds") is not None
                    else 660.0
                )
            )
            max_disk_gb = (
                float(args.max_disk_gb)
                if args.max_disk_gb is not None
                else (
                    float(exe["max_total_disk_bytes"]) / float(1024**3)
                    if exe.get("max_total_disk_bytes") is not None
                    else 18.0
                )
            )
            prepack_bytes = int(round(float(args.prepack_mb) * 1024.0 * 1024.0)) if args.prepack_mb is not None else None
            report = estimate_production_preflight(
                total_updates=tot_u,
                num_jobs=n_jobs,
                parallel_workers=int(args.parallel_workers),
                model_params=int(args.model_params),
                batch_size=bs,
                seq_len=int(args.seq_len),
                hidden_dim=int(args.hidden_dim),
                num_layers=int(args.num_layers),
                precision=prec,
                seconds_per_update=args.seconds_per_update,
                prepack_bytes_per_worker=prepack_bytes,
                per_job_disk_bytes=int(round(float(args.per_job_disk_mb) * 1024.0 * 1024.0)),
                startup_overhead_seconds_per_job=float(args.startup_overhead_seconds),
                max_vram_bytes_per_worker=int(round(max_vram_gb * float(1024**3))),
                max_prepack_bytes_per_worker=int(round(float(args.max_prepack_mb) * 1024.0 * 1024.0)),
                max_session_minutes=sess_min,
                reserve_minutes=float(args.reserve_minutes),
                max_disk_bytes=int(round(max_disk_gb * float(1024**3))),
            )
            if args.output:
                out_path = Path(args.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _emit(report)
            return 0 if report["passed"] else 1
        except Exception as exc:
            _emit({"passed": False, "status": "PREFLIGHT_ERROR", "error": type(exc).__name__, "detail": str(exc)})
            return 1

    if args.command == "explain-error":
        _emit(explain_error_code(args.code))
        return 0

    if args.command == "calibrate":
        report = run_fault_injection_suite()
        _emit(report)
        return 0 if report["all_detected"] else 1

    if args.command == "shield":
        report = run_level7_runtime_shield_suite()
        _emit(report)
        return 0 if report["passed"] else 1

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
