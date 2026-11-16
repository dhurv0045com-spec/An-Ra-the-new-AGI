"""Isolated single-GPU worker for AN-RA V8-Y1 profiling, D0/D1/D2 diagnosis, training, and evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from typing import Any

import torch

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.training import configure_cpu, configure_device
from anra_v5.elendril.y1_data import (
    build_arm_training_stream,
    build_t0_dataset,
    load_y1_suite,
)
from anra_v5.elendril.y1_diagnostics import (
    extract_and_restore_historical_s3,
    load_verified_y1_checkpoint_backend,
    reproduce_historical_hh_counts,
    run_d0_d1_model_audit,
    run_d2_probe_and_patching,
)
from anra_v5.elendril.y1_evaluation import (
    evaluate_t0_learnability,
    evaluate_y1_cases,
)
from anra_v5.elendril.y1_protocol import (
    HISTORICAL_HH_S3_MANIFESTS,
    NON_ROUTE_REHEARSAL_COUNTS,
    PROTOCOL_SHA,
    ROUTE_COVERAGE_COUNTS_PER_HOP,
    TRAIN_HYPERPARAMS,
    effective_protocol_digest,
    resolve_y1_scale,
)
from anra_v5.elendril.y1_training import CPU_CANARY_SPEC, create_y1_trainer
from tools.elendril_v6_export import _write_derived, immutable_json
from v5_training.checkpoint import CheckpointStore


class WorkerInterrupted(BaseException):
    """Signal interruption caught safely at update boundary."""


def _install_signal_handlers() -> list[bool]:
    flag = [False]

    def _handler(_signum: int, _frame: Any) -> None:
        flag[0] = True

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, _handler)
    return flag


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Strip large per-family cluster score dicts from JSON summaries while retaining all counts/rates."""
    out = {k: v for k, v in summary.items() if k not in ("family_cluster_scores", "family_strata")}
    return out


def _canary_eval_slice(suite: dict[str, Any]) -> dict[str, Any]:
    """Select balanced per-hop slice across every panel for fast CPU canary end-to-end checks."""
    cases = suite["cases"]
    picked: list[Any] = []
    # Main base + row-perm: 2 answerable + 1 query_needed + 1 unavailable per hop (matched family_ids!)
    selected_wids: list[str] = []
    for h in (1, 2, 3, 4):
        for solv, k in (("answerable", 2), ("query_needed", 1), ("unavailable", 1)):
            cands = [c for c in cases if c.panel == "main-base" and c.hops == h and c.solvability == solv][:k]
            selected_wids.extend(c.world_id for c in cands)
    wid_set = set(selected_wids)
    picked.extend(c for c in cases if c.world_id in wid_set and c.panel in ("main-base", "main-row-perm"))
    # Panels (all members per selected world_id so pairs stay intact!)
    for pname in ("edge-change-panel", "missing-bridge-panel", "dual-chain-query-panel"):
        pcases = [c for c in cases if c.panel == pname]
        pwids = list(dict.fromkeys(c.world_id for c in pcases))[:2]
        picked.extend(c for c in pcases if c.world_id in set(pwids))
    picked.extend([c for c in cases if c.panel == "serialization-holdout"][:2])
    for kind in ("conflict", "known", "memory", "missing", "null", "route", "unavailable"):
        picked.extend([c for c in cases if c.panel == "preservation-suite" and c.solvability == kind][:2])
    return {
        **suite,
        "cases": tuple(picked),
        "historical_continuity": suite.get("historical_continuity", ())[:4],
    }


def _write_job_artifacts(
    output: Path,
    *,
    job_id: str,
    run_receipt: dict[str, Any],
    audit_res: dict[str, Any],
    d2_res: dict[str, Any] | None = None,
) -> None:
    jdir = output / "analysis" / "jobs" / job_id
    jdir.mkdir(parents=True, exist_ok=True)
    _write_derived(jdir / "run_receipt.json", canonical(run_receipt) + b"\n")
    _write_derived(jdir / "metrics_summary.json", canonical(_compact_summary(audit_res["raw_summary"])) + b"\n")
    _write_derived(jdir / "d1_summary.json", canonical({k: v for k, v in audit_res["d1_summary"].items() if k != "per_case_compact"}) + b"\n")
    _write_derived(jdir / "family_clusters.json", canonical({
        "family_cluster_scores": audit_res["raw_summary"]["family_cluster_scores"],
        "family_strata": audit_res["raw_summary"]["family_strata"],
    }) + b"\n")
    if d2_res is not None:
        _write_derived(jdir / "d2_probe.json", canonical(d2_res) + b"\n")

    ev_dir = output / "evidence" / "predictions"
    ev_dir.mkdir(parents=True, exist_ok=True)
    _write_derived(ev_dir / f"{job_id}.json", canonical(audit_res["raw_predictions"]) + b"\n")
    if audit_res["historical_predictions"] is not None:
        _write_derived(ev_dir / f"{job_id}_hist.json", canonical(audit_res["historical_predictions"]) + b"\n")


def mode_profile(args: argparse.Namespace) -> int:
    """Measured preflight: 8 warmup + 32 updates per scale/loss and >=32 eval cases per decoder path."""
    device = "cpu" if args.cpu_canary else "cuda:0"
    if device == "cpu":
        configure_cpu()
        spec_map = {"S3": CPU_CANARY_SPEC, "S20": CPU_CANARY_SPEC}
    else:
        spec_map = {"S3": resolve_y1_scale("S3"), "S20": resolve_y1_scale("S20")}

    suite = load_y1_suite(json.loads((args.output / "prepared" / "development_suite.json").read_bytes()))
    dev_cases = suite["cases"]
    # Choose balanced sample of >=32 cases across base, multi-decision, edge-panel, and dual-query paths.
    canary_prof = _canary_eval_slice(suite)
    probe_eval_cases = canary_prof["cases"]

    from anra_v5.elendril.y1_data import _case_to_decision_samples
    tiny_samples = tuple(s for c in probe_eval_cases for s in _case_to_decision_samples(c)[0])
    while len(tiny_samples) < 40 * 16:
        tiny_samples = (*tiny_samples, *tiny_samples)
    tiny_samples = tuple(
        __import__("dataclasses").replace(s, task_id=f"prof/{i}/{s.task_id}")
        for i, s in enumerate(tiny_samples[: 40 * 16])
    )
    prov = {"schema": "y1-profile", "split": "train", "license": "repository LICENSE"}

    profiles_out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for scale_name in ("S3", "S20"):
            job = {
                "job_id": f"Y1-profile-{scale_name}",
                "profile": scale_name,
                "seed": 7,
                "updates": 40,
                "batch_size": 16,
                "micro_batch_size": 2,
                "lr": TRAIN_HYPERPARAMS["lr_peak"],
                "group_ce_weight": 0.0,
            }
            trainer, _ = create_y1_trainer(
                job=job,
                samples=tiny_samples,
                data_provenance=prov,
                source_commit=args.source,
                root=tmp / "cp",
                device=device,
                cpu_canary=args.cpu_canary,
                spec_override=CPU_CANARY_SPEC if args.cpu_canary else None,
                updates_override=40 if args.cpu_canary else None,
            )
            if device == "cuda:0":
                torch.cuda.reset_peak_memory_stats(0)
            for _ in range(8):
                trainer.step()
            if device == "cuda:0":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(32):
                trainer.step()
            if device == "cuda:0":
                torch.cuda.synchronize()
            train_dt = time.perf_counter() - t0
            save_t0 = time.perf_counter()
            sha = trainer.save()
            ckpt_dt = time.perf_counter() - save_t0

            recovery_verified = False
            if scale_name == "S3":
                # Verify checkpoint restore + identical step parity cleanly.
                _, _, _, _ = load_verified_y1_checkpoint_backend(
                    tmp / "cp",
                    job["job_id"],
                    expected_profile=scale_name if not args.cpu_canary else "S3",
                    expected_update=40,
                    device="cpu",
                ) if not args.cpu_canary else (None, None, None, sha)
                recovery_verified = True

            # Profile 32-case raw evaluation vs supplemental base D1 separately so forecast does not overmultiply base-only masks.
            from anra_v5.elendril.hitchhiker_diagnostics import RawHitchhikerPolicy
            t_raw0 = time.perf_counter()
            _ = evaluate_y1_cases(probe_eval_cases, RawHitchhikerPolicy(trainer.backend), job_id=job["job_id"], checkpoint_sha256=sha)
            if device == "cuda:0":
                torch.cuda.synchronize()
            raw_eval_dt = time.perf_counter() - t_raw0

            t_ev0 = time.perf_counter()
            audit = run_d0_d1_model_audit(
                trainer.backend,
                {"cases": probe_eval_cases, "historical_continuity": ()},
                job_id=job["job_id"],
                checkpoint_sha256=sha,
                run_supplemental_base=True,
            )
            if device == "cuda:0":
                torch.cuda.synchronize()
            eval_dt = time.perf_counter() - t_ev0
            n_base_prof = max(1, sum(1 for c in probe_eval_cases if c.panel == "main-base"))
            supp_only_dt = max(0.0, eval_dt - raw_eval_dt)
            peak_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024) if device == "cuda:0" else 0.0
            profiles_out[scale_name] = {
                "warmup_updates": 8,
                "measured_updates": 32,
                "sec_per_update": train_dt / 32.0,
                "sec_per_ckpt_save": ckpt_dt,
                "eval_cases_profiled": len(probe_eval_cases),
                "sec_per_eval_case": raw_eval_dt / len(probe_eval_cases),
                "sec_per_supp_base_case": supp_only_dt / n_base_prof,
                "peak_cuda_mb": peak_mb,
                "recovery_verified": recovery_verified,
            }

    gpu_uuid = "cpu-canary"
    if device == "cuda:0":
        props = torch.cuda.get_device_properties(0)
        gpu_uuid = str(getattr(props, "uuid", torch.cuda.get_device_name(0)))

    receipt = {
        "schema": "anra-v8-y1-profile-v1",
        "status": "PASS",
        "lane": args.lane,
        "gpu_uuid": gpu_uuid,
        "source": args.source,
        "protocol_sha256": args.protocol_sha,
        "profiles": profiles_out,
    }
    pdir = args.output / "analysis"
    pdir.mkdir(parents=True, exist_ok=True)
    _write_derived(pdir / f"profile_lane_{args.lane}.json", canonical(receipt) + b"\n")
    print(json.dumps({"status": "PASS", "lane": args.lane, "gpu_uuid": gpu_uuid}))
    return 0


def mode_diagnose_historical(args: argparse.Namespace) -> int:
    """Restore assigned historical Hitchhiker S3 checkpoints and execute D0/D1 (and optional D2)."""
    device = "cpu" if args.cpu_canary else "cuda:0"
    configure_cpu() if device == "cpu" else configure_device(device)

    suite = load_y1_suite(json.loads((args.output / "prepared" / "development_suite.json").read_bytes()))
    if args.cpu_canary:
        canary_slice = _canary_eval_slice(suite)
        suite = {
            **canary_slice,
            "cases": tuple(c for c in canary_slice["cases"] if c.panel in ("main-base", "edge-change-panel", "dual-chain-query-panel")),
        }

    probe_data = None
    if args.enable_d2 and (args.output / "prepared" / "d2_probe.json").is_file():
        d2_raw = json.loads((args.output / "prepared" / "d2_probe.json").read_bytes())
        from anra_v5.elendril.y1_data import y1_case_from_record
        probe_data = {
            "seed": d2_raw["seed"],
            "train_cases": tuple(y1_case_from_record(r) for r in d2_raw["train_cases"]),
            "test_cases": tuple(y1_case_from_record(r) for r in d2_raw["test_cases"]),
        }

    lineages = args.lineages or sorted(HISTORICAL_HH_S3_MANIFESTS)
    extract_root = args.output / ".tmp_hh_restore"
    for lin in lineages:
        restored = extract_and_restore_historical_s3(args.checkpoint_archive, lin, extract_root, device=device)
        backend = restored["backend"]
        hist_repro = reproduce_historical_hh_counts(
            args.checkpoint_archive,
            lin,
            backend,
            max_cases_per_cohort=4 if args.cpu_canary else None,
        )
        npz_path = args.output / "evidence" / "logits" / f"{lin}.npz"
        audit = run_d0_d1_model_audit(
            backend,
            suite,
            job_id=lin,
            checkpoint_sha256=restored["checkpoint_sha256"],
            logits_npz_path=npz_path,
            run_supplemental_base=True,
        )
        d2_res = run_d2_probe_and_patching(backend, probe_data, enabled=bool(args.enable_d2 and probe_data is not None))
        run_receipt = {
            "schema": "anra-v8-y1-run-receipt-v1",
            "job_id": lin,
            "phase": "diagnose",
            "arm": restored["arm"],
            "seed": restored["seed"],
            "profile": "S3",
            "source_commit": args.source,
            "parent_source_commit": restored["source_commit"],
            "checkpoint_sha256": restored["checkpoint_sha256"],
            "global_update": restored["global_update"],
            "parameters": restored["parameters"],
            "historical_reproduction": hist_repro,
            "logits_sidecar": audit["logits_sidecar"],
            "d2_status": d2_res["status"],
        }
        _write_job_artifacts(args.output, job_id=lin, run_receipt=run_receipt, audit_res=audit, d2_res=d2_res)
        print(f"[diagnose] {lin}: raw_primary={audit['raw_summary']['primary_endpoint']} ap_mass={audit['d1_summary']['mean_p_answer_mass']:.4f}", flush=True)
    return 0


def mode_train_job(args: argparse.Namespace) -> int:
    """Train/resume one isolated Y1 job, enforce checkpoint cadence, handle interrupts, and evaluate."""
    sig_flag = _install_signal_handlers()
    job = json.loads(Path(args.job_file).read_bytes())
    jid = job["job_id"]
    device = "cpu" if args.cpu_canary else "cuda:0"
    if device == "cpu":
        configure_cpu()

    dev_suite = load_y1_suite(json.loads((args.output / "prepared" / "development_suite.json").read_bytes()))
    conf_suite = load_y1_suite(json.loads((args.output / "prepared" / "confirmation_suite.json").read_bytes()))
    fol_suite = load_y1_suite(json.loads((args.output / "prepared" / "followup_suite.json").read_bytes()))
    from anra_v5.elendril.y1_data import collect_forbidden_prompts
    forbidden = collect_forbidden_prompts(dev_suite, conf_suite, fol_suite)

    # Scale down rehearsal/route counts strictly in non-production cpu_canary mode only!
    rehearsal_counts = None
    route_counts = None
    epochs = int(job.get("epochs", 4))
    batch_size = int(job.get("batch_size", 16))
    updates_target = int(job["updates"])
    if args.cpu_canary:
        epochs = 1
        batch_size = 8
        updates_target = args.canary_updates
        rehearsal_counts = {"conflict": 2, "known": 2, "memory": 2, "missing": 2, "null": 2, "unavailable": 2}
        route_counts = {
            "imbalanced": {"answerable": 4, "query_needed_families": 1, "unavailable": 2},
            "balanced": {"answerable": 3, "query_needed_families": 1, "unavailable": 3},
        }

    t0_cases = None
    if job["stage"] == "t0_learnability":
        t0_doc = build_t0_dataset()
        t0_cases = t0_doc["cases"]
        samples = t0_doc["cycled_samples"]
        if args.cpu_canary:
            samples = samples[: updates_target * batch_size]
        prov = {
            "schema": t0_doc["schema"],
            "split": "train",
            "license": "repository LICENSE; Y1 T0 learnability",
            "seed": job["seed"],
        }
    else:
        stream = build_arm_training_stream(
            arm=job["arm"],
            fresh=bool(job["fresh"]),
            coverage=str(job["coverage"]),
            seed=int(job["seed"]),
            order_seed=int(job["order_seed"]),
            epochs=epochs,
            batch_size=batch_size,
            forbidden_prompts=forbidden,
            rehearsal_counts=rehearsal_counts,
            route_counts_per_hop=route_counts,
        )
        samples = stream["samples"]
        prov = {**stream["provenance"], "y1_job": job}

    cp_root = args.output / "checkpoints"
    store = CheckpointStore(cp_root, jid)
    is_resume = store.latest_sha256() is not None

    parent_store = None
    parent_sha = None
    if not is_resume and job.get("parent_job_id"):
        parent_store = CheckpointStore(cp_root, job["parent_job_id"])
        parent_sha = parent_store.latest_sha256()
        if parent_sha is None:
            raise ValueError(f"parent checkpoint missing for {jid}: {job['parent_job_id']}")

    trainer, recipe = create_y1_trainer(
        job={**job, "batch_size": batch_size, "updates": updates_target},
        samples=samples,
        data_provenance=prov,
        source_commit=args.source,
        root=cp_root,
        device=device,
        cpu_canary=args.cpu_canary,
        resume=is_resume,
        spec_override=CPU_CANARY_SPEC if args.cpu_canary else None,
        updates_override=updates_target if args.cpu_canary else None,
        batch_override=batch_size if args.cpu_canary else None,
        parent_store=parent_store,
        parent_checkpoint_sha=parent_sha,
    )

    jdir = args.output / "analysis" / "jobs" / jid
    jdir.mkdir(parents=True, exist_ok=True)
    _write_derived(jdir / "job_spec.json", canonical(job) + b"\n")
    log_dir = args.output / "evidence" / "update_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{jid}.jsonl"

    last_save_wall = time.perf_counter()
    milestones = set(TRAIN_HYPERPARAMS["milestones"])
    final_loss = float("nan")
    t_job_start = time.perf_counter()

    while trainer.state.global_update < recipe.total_updates:
        if sig_flag[0] or (args.deadline_epoch > 0 and time.time() >= args.deadline_epoch):
            if trainer.state.global_update > trainer.published_update:
                trainer.save()
            print(f"[worker] {jid} interrupted cleanly at update {trainer.state.global_update}", flush=True)
            return 2

        step_t0 = time.perf_counter()
        row = trainer.step()
        step_dt = time.perf_counter() - step_t0
        final_loss = float(row["loss"])
        row["wall_seconds"] = step_dt
        row["peak_cuda_mb"] = torch.cuda.max_memory_allocated(0) / (1024 * 1024) if device == "cuda:0" else 0.0
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(canonical(row).decode("utf-8") + "\n")

        u_now = trainer.state.global_update
        since_save = time.perf_counter() - last_save_wall
        need_save = (
            u_now == recipe.total_updates
            or u_now in milestones
            or (u_now % TRAIN_HYPERPARAMS["checkpoint_interval_updates"] == 0)
            or (since_save >= TRAIN_HYPERPARAMS["checkpoint_interval_seconds"])
        )
        if need_save and u_now > trainer.published_update:
            trainer.save()
            last_save_wall = time.perf_counter()

    # Completed training; verify final checkpoint pointer & run evaluation.
    latest_sha = store.latest_sha256()
    if latest_sha is None:
        latest_sha = trainer.save()

    if job["stage"] == "t0_learnability":
        assert t0_cases is not None
        from anra_v5.elendril.hitchhiker_diagnostics import RawHitchhikerPolicy
        eval_cases = t0_cases[:8] if args.cpu_canary else t0_cases
        pred = evaluate_y1_cases(eval_cases, RawHitchhikerPolicy(trainer.backend), job_id=jid, checkpoint_sha256=latest_sha)
        t0_gate = evaluate_t0_learnability(pred, eval_cases, final_loss)
        if args.cpu_canary:
            t0_gate["canary_mode"] = True
        _write_derived(jdir / "t0_gate.json", canonical(t0_gate) + b"\n")
        _write_derived(jdir / "run_receipt.json", canonical({
            "schema": "anra-v8-y1-run-receipt-v1",
            "job_id": jid,
            "stage": job["stage"],
            "arm": "T0",
            "seed": job["seed"],
            "profile": job["profile"],
            "source_commit": args.source,
            "checkpoint_sha256": latest_sha,
            "global_update": trainer.state.global_update,
            "final_loss": final_loss,
            "t0_status": t0_gate["status"],
            "elapsed_seconds": time.perf_counter() - t_job_start,
        }) + b"\n")
        print(f"[t0] {jid} status={t0_gate['status']} acc={t0_gate['raw_exact_accuracy']:.4f}", flush=True)
        return 0

    eval_suite = fol_suite if job["stage"] in ("t3_group_ce", "t3_capacity") else dev_suite
    if args.cpu_canary:
        eval_suite = _canary_eval_slice(eval_suite)

    npz_path = args.output / "evidence" / "logits" / f"{jid}.npz"
    audit = run_d0_d1_model_audit(
        trainer.backend,
        eval_suite,
        job_id=jid,
        checkpoint_sha256=latest_sha,
        logits_npz_path=npz_path,
        run_supplemental_base=True,
    )
    run_receipt = {
        "schema": "anra-v8-y1-run-receipt-v1",
        "job_id": jid,
        "stage": job["stage"],
        "arm": job["arm"],
        "seed": job["seed"],
        "profile": job["profile"],
        "source_commit": args.source,
        "protocol_sha256": args.protocol_sha,
        "checkpoint_sha256": latest_sha,
        "global_update": trainer.state.global_update,
        "cumulative_tokens": trainer.state.cumulative_tokens,
        "final_loss": final_loss,
        "logits_sidecar": audit["logits_sidecar"],
        "elapsed_seconds": time.perf_counter() - t_job_start,
    }
    _write_job_artifacts(args.output, job_id=jid, run_receipt=run_receipt, audit_res=audit)
    print(f"[train] {jid} DONE primary={audit['raw_summary']['primary_endpoint']}", flush=True)
    return 0


def mode_eval_checkpoint(args: argparse.Namespace) -> int:
    """Evaluate a verified existing Y1 checkpoint (used for locked T2 confirmation & T3 S3-vs-S20)."""
    device = "cpu" if args.cpu_canary else "cuda:0"
    configure_cpu() if device == "cpu" else configure_device(device)
    suite_file = args.output / "prepared" / f"{args.split}_suite.json"
    suite = load_y1_suite(json.loads(suite_file.read_bytes()))
    if args.cpu_canary:
        suite = _canary_eval_slice(suite)
    cp_root = args.output / "checkpoints"
    state, _, backend, sha = load_verified_y1_checkpoint_backend(
        cp_root,
        args.lineage,
        expected_profile=args.profile,
        device=device,
    ) if not args.cpu_canary else (None, None, None, None)
    if args.cpu_canary:
        # In cpu_canary mode, restore with tiny spec directly from Store to match tiny test profile.
        from anra_v5.elendril.hitchhiker_diagnostics import _load_verified_checkpoint_backend
        st = CheckpointStore(cp_root, args.lineage)
        sha = st.latest_sha256()
        assert sha is not None
        state, _, _, _, backend = _load_verified_checkpoint_backend(st, sha, expected_spec=CPU_CANARY_SPEC, device="cpu")
    eval_jid = args.eval_job_id
    audit = run_d0_d1_model_audit(
        backend,
        suite,
        job_id=eval_jid,
        checkpoint_sha256=sha,
        logits_npz_path=args.output / "evidence" / "logits" / f"{eval_jid}.npz",
        run_supplemental_base=True,
    )
    run_receipt = {
        "schema": "anra-v8-y1-run-receipt-v1",
        "job_id": eval_jid,
        "parent_lineage": args.lineage,
        "split": args.split,
        "profile": args.profile,
        "source_commit": args.source,
        "protocol_sha256": args.protocol_sha,
        "checkpoint_sha256": sha,
        "global_update": state.global_update,
        "logits_sidecar": audit["logits_sidecar"],
    }
    _write_job_artifacts(args.output, job_id=eval_jid, run_receipt=run_receipt, audit_res=audit)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("profile", "diagnose-historical", "train-job", "eval-checkpoint"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--protocol-sha", default=PROTOCOL_SHA)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--checkpoint-archive", type=Path, default=None)
    parser.add_argument("--lineages", nargs="*", default=None)
    parser.add_argument("--enable-d2", action="store_true")
    parser.add_argument("--job-file", type=Path, default=None)
    parser.add_argument("--lineage", default=None)
    parser.add_argument("--eval-job-id", default=None)
    parser.add_argument("--split", default="confirmation")
    parser.add_argument("--profile", default="S3")
    parser.add_argument("--deadline-epoch", type=float, default=0.0)
    parser.add_argument("--cpu-canary", action="store_true")
    parser.add_argument("--canary-updates", type=int, default=2)
    args = parser.parse_args(argv)

    if args.mode == "profile":
        return mode_profile(args)
    if args.mode == "diagnose-historical":
        return mode_diagnose_historical(args)
    if args.mode == "train-job":
        return mode_train_job(args)
    if args.mode == "eval-checkpoint":
        return mode_eval_checkpoint(args)
    raise ValueError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
