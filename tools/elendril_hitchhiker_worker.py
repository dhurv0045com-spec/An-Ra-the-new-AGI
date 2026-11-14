"""Isolated single-GPU worker for Hitchhiker profiling, recovery, Stage 0 audit, and training."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch

from anra_v5.elendril.codec import canonical
from anra_v5.elendril.cognition import (
    ExperienceEvent,
    Goal,
    MemoryVersion,
    PublicEnvironment,
    Truth,
)
from anra_v5.elendril.hitchhiker_data import (
    HitchhikerSurface,
    build_counterfactual_suite,
    build_health_probes,
    build_structural_holdout_suite,
    build_training_stream,
)
from anra_v5.elendril.hitchhiker_diagnostics import (
    ExternalSlotCanonicalizerPolicy,
    RawHitchhikerPolicy,
    admit_historical_s20_checkpoint,
    score_entity_equivariance,
)
from anra_v5.elendril.hitchhiker_evaluation import (
    evaluate_counterfactual_suite,
    evaluate_health_probes,
    evaluate_original_kinds,
)
from anra_v5.elendril.hitchhiker_protocol import (
    AUDITED_V717_COMMIT,
    HISTORICAL_S20_LOWLR_MANIFESTS,
    PROTOCOL_SHA,
    resolve_scale_profile,
    stage_jobs,
    validate_job,
    validate_stage_selection,
)
from anra_v5.elendril.hitchhiker_training import (
    aux_penalty_gradient_probe,
    create_hitchhiker_trainer,
    resolve_exposure_schedule,
    run_hitchhiker_updates,
)
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.training import MODEL, configure_cpu
from anra_v5.elendril.v6_data import Surface, V6Case, make_dataset, read_dataset
from tools.elendril_action_cloud import compare_payloads
from tools.elendril_hitchhiker_runner import (
    _evaluate_backend_decoders,
    write_stage0_evidence_package,
    verify_and_rescore_stage0_package,
)
from tools.elendril_v6_export import _write_derived, immutable_json
from v5_training.checkpoint import CheckpointStore


REPO_ROOT = Path(__file__).resolve().parents[1]


class WorkerInterrupted(BaseException):
    """Raised on SIGTERM/SIGINT so interrupted checkpoints flush cleanly."""


def hitchhiker_case_to_record(case: V6Case) -> dict[str, Any]:
    """Serialize V6Case (with Surface or HitchhikerSurface) losslessly."""
    surf_cls = "HitchhikerSurface" if isinstance(case.surface, HitchhikerSurface) else "Surface"
    return {
        "surface_type": surf_cls,
        "case_id": case.case_id,
        "world_id": case.world_id,
        "family": case.family,
        "kind": case.kind,
        "goal": asdict(case.goal),
        "initial": [asdict(e) for e in case.environment.initial],
        "channels": {k: [asdict(e) for e in v] for k, v in case.environment.channels.items()},
        "truth": asdict(case.truth),
        "surface": asdict(case.surface),
        "memory": None if case.memory is None else asdict(case.memory),
        "pair_id": case.pair_id,
        "panel": case.panel,
    }


def hitchhiker_case_from_record(row: dict[str, Any]) -> V6Case:
    """Deserialize a V6Case with Surface or HitchhikerSurface deterministically."""
    event = lambda item: ExperienceEvent(**{**item, "parents": tuple(item["parents"])})
    goal = Goal(**{**row["goal"], "slots": tuple(row["goal"]["slots"])})
    mem = row["memory"]
    memory = None if mem is None else MemoryVersion(tuple(event(e) for e in mem["events"]), mem["parent"], mem["capacity"])
    surf = HitchhikerSurface(**row["surface"]) if row["surface_type"] == "HitchhikerSurface" else Surface(**row["surface"])
    return V6Case(
        row["case_id"],
        row["world_id"],
        row["family"],
        row["kind"],
        goal,
        PublicEnvironment(
            tuple(event(e) for e in row["initial"]),
            {k: tuple(event(e) for e in v) for k, v in row["channels"].items()},
        ),
        Truth(**{**row["truth"], "admitted_public": tuple(row["truth"]["admitted_public"])}),
        surf,
        memory,
        row["pair_id"],
        row["panel"],
    )


def serialize_suite_bundle(suite: dict[str, Any]) -> dict[str, Any]:
    """Serialize a counterfactual/structural/probe suite with case-level integrity hash."""
    if "cohorts" in suite:
        cohorts = {k: [hitchhiker_case_to_record(c) for c in v] for k, v in suite["cohorts"].items()}
        payload: dict[str, Any] = {
            "kind": "suite",
            "split": suite["split"],
            "seed": suite.get("seed"),
            "cohorts": cohorts,
            "permutations": suite.get("permutations", {}),
        }
    else:
        probes = {k: [hitchhiker_case_to_record(c) for c in v] for k, v in suite.items()}
        payload = {"kind": "probes", "cohorts": probes}
    payload["payload_sha256"] = sha256(canonical(payload["cohorts"])).hexdigest()
    return payload


def load_suite_bundle(path: Path) -> dict[str, Any]:
    """Load and SHA-verify a serialized suite or health-probe bundle."""
    doc = json.loads(Path(path).read_bytes())
    if sha256(canonical(doc["cohorts"])).hexdigest() != doc.get("payload_sha256"):
        raise ValueError(f"serialized suite failed integrity check: {path}")
    if doc["kind"] == "probes":
        return {k: tuple(hitchhiker_case_from_record(r) for r in v) for k, v in doc["cohorts"].items()}
    return {
        "split": doc["split"],
        "seed": doc.get("seed"),
        "cohorts": {k: tuple(hitchhiker_case_from_record(r) for r in v) for k, v in doc["cohorts"].items()},
        "permutations": doc.get("permutations", {}),
    }


def _assert_cuda_isolated(expected_gpu: int = 1) -> None:
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("worker requires isolated CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_available() or torch.cuda.device_count() != expected_gpu:
        raise RuntimeError(f"worker requires {expected_gpu} isolated CUDA device")


def _checkpoint_fingerprint(store: CheckpointStore, sha_id: str) -> dict[str, Any]:
    state, metadata, spec, hashes = store.load(sha_id, expected_commit=metadata_commit(store, sha_id), device="cpu")
    return {"global_update": state.global_update, "hashes": hashes, "spec": spec.sha256()}


def metadata_commit(store: CheckpointStore, sha_id: str) -> str:
    manifest = json.loads((store.object_dir(sha_id) / "manifest.json").read_text(encoding="utf-8"))
    return manifest["source_commit"]


def mode_recover(args: argparse.Namespace) -> dict[str, Any]:
    """Reload saved checkpoint in a fresh process, step forward, and compare exact state."""
    if not args.cpu_canary:
        _assert_cuda_isolated()
    else:
        configure_cpu()
    device = "cpu" if args.cpu_canary else "cuda:0"
    spec = replace(MODEL, vocabulary_size=UTF8.vocabulary_size, context_length=512) if args.cpu_canary else resolve_scale_profile(args.profile)
    base_data = read_dataset(args.prepared / "train_base.json")
    suite = load_suite_bundle(args.prepared / "validation_suite.json")
    job = json.loads(Path(args.job_file).read_text(encoding="utf-8"))
    trainer, _ = create_hitchhiker_trainer(
        job,
        source_commit=args.source,
        root=args.checkpoints,
        base_data=base_data,
        heldout_suites=(suite,),
        resume=True,
        device=device,
        cpu_canary=args.cpu_canary,
        spec_override=spec if args.cpu_canary else None,
        updates_override=args.to_update if args.cpu_canary else None,
        batch_override=4 if args.cpu_canary else None,
    )
    if trainer.state.global_update != args.from_update:
        raise RuntimeError(f"recover expected resumed update {args.from_update}, got {trainer.state.global_update}")
    step_logs = []
    for _ in range(args.from_update, args.to_update):
        step_logs.append(trainer.step())
    ref_sha = args.ref_sha
    resumed_payload = {
        "final_update": trainer.state.global_update,
        "step_losses": [round(float(x["loss"]), 6) for x in step_logs],
        "param_hashes": {
            k: sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k, v in trainer.model.state_dict().items()
        },
    }
    if ref_sha:
        ref_state = json.loads(Path(args.ref_file).read_text(encoding="utf-8"))
        if "model.bin" in ref_state:
            compare_payloads(ref_state, resumed_payload, exact=True)
        elif resumed_payload != ref_state:
            raise ValueError(
                f"recovery mismatch vs ref checkpoint {ref_sha}: "
                f"update={resumed_payload['final_update']} vs {ref_state.get('final_update')}"
            )
    return {"status": "PASS", "recovery": resumed_payload}


def _build_canary_spec() -> tuple[Any, dict[str, Any]]:
    tiny = replace(MODEL, vocabulary_size=UTF8.vocabulary_size, context_length=512)
    return tiny, resolve_scale_profile("S9", {"spec": tiny, "parameters": tiny.parameter_receipt().total, "model_sha256": tiny.sha256()})


def mode_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Measure isolated GPU latency, peak memory, checkpoint IO, and fresh-process recover equivalence."""
    if not args.cpu_canary:
        _assert_cuda_isolated()
    else:
        configure_cpu()
    device = "cpu" if args.cpu_canary else "cuda:0"
    base_data = read_dataset(args.prepared / "train_base.json")
    suite = load_suite_bundle(args.prepared / "validation_suite.json")
    s3_job = [j for j in stage_jobs("stage1_s3_screen") if j["arm"] == "AB" and j["seed"] == 7][0]

    profiles_measured: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for label, scale_name, updates in (("S3", "S3", 8), ("S8", "S8", 4), ("S20", "S20", 4)):
            spec_override = _build_canary_spec()[1] if args.cpu_canary else None
            job_spec = {**s3_job, "profile": scale_name} if scale_name == "S3" else {
                **[j for j in stage_jobs("stage3_scale_confirmation", winning_arm="AB") if j["profile"] == scale_name and j["arm"] == "AB" and j["seed"] == 7][0]
            }
            if args.cpu_canary:
                job_spec = s3_job
            ckpt_dir = tmp / f"ckpt_{label}"
            selection = {"screening_status": "PASS_SCREENING", "split": "validation", "winning_arm": "AB", "model_profile": scale_name} if job_spec["stage"] == "stage3_scale_confirmation" else None
            trainer, _ = create_hitchhiker_trainer(
                job_spec,
                source_commit=args.source,
                root=ckpt_dir,
                base_data=base_data,
                heldout_suites=(suite,),
                device=device,
                cpu_canary=args.cpu_canary,
                spec_override=spec_override,
                updates_override=updates if args.cpu_canary else None,
                batch_override=4 if args.cpu_canary else None,
                winning_arm="AB",
                selection_receipt=selection,
            )
            if not args.cpu_canary:
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            step_logs = []
            ckpt_write_s = 0.0
            mid_sha = None
            for u in range(1, updates + 1):
                res = trainer.step()
                step_logs.append(res)
                if label == "S3" and u in (4, 8):
                    cw0 = time.perf_counter()
                    sha = trainer.save()
                    ckpt_write_s += time.perf_counter() - cw0
                    if u == 4:
                        mid_sha = sha
            wall = time.perf_counter() - t0
            peak_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if not args.cpu_canary else 0.0

            eval_t0 = time.perf_counter()
            sub_n = min(8, len(suite["cohorts"]["hop-1"]))
            evaluate_counterfactual_suite(
                {
                    "split": "validation",
                    "worlds_per_hop": sub_n,
                    "cohorts": {"hop-1": suite["cohorts"]["hop-1"][:sub_n]},
                },
                RawHitchhikerPolicy(trainer.backend),
            )
            eval_s = time.perf_counter() - eval_t0

            recovery_verified = False
            if label == "S3":
                ref_payload = {
                    "final_update": trainer.state.global_update,
                    "step_losses": [round(float(x["loss"]), 6) for x in step_logs[4:8]],
                    "param_hashes": {
                        k: sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                        for k, v in trainer.model.state_dict().items()
                    },
                }
                ref_file = tmp / "ref_s3.json"
                job_file = tmp / "job_s3.json"
                ref_file.write_text(json.dumps(ref_payload), encoding="utf-8")
                job_file.write_text(json.dumps(job_spec), encoding="utf-8")
                # Force LATEST back to update-4 checkpoint for fresh-process recovery check.
                (ckpt_dir / job_spec["job_id"] / "LATEST").write_text(mid_sha + "\n", encoding="ascii")
                rec_out = tmp / "recover_receipt.json"
                cmd = [
                    sys.executable, "-m", "tools.elendril_hitchhiker_worker",
                    "--mode", "recover",
                    "--source", args.source,
                    "--prepared", str(args.prepared),
                    "--checkpoints", str(ckpt_dir),
                    "--job-file", str(job_file),
                    "--profile", "S3",
                    "--from-update", "4",
                    "--to-update", "8",
                    "--ref-sha", str(mid_sha),
                    "--ref-file", str(ref_file),
                    "--receipt", str(rec_out),
                ]
                if args.cpu_canary:
                    cmd.append("--cpu-canary")
                subprocess.run(cmd, cwd=REPO_ROOT, check=True, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
                recovery_verified = json.loads(rec_out.read_text(encoding="utf-8")).get("status") == "PASS"

            ckpt_bytes = sum(p.stat().st_size for p in ckpt_dir.rglob("*") if p.is_file())
            profiles_measured[label] = {
                "updates_measured": updates,
                "wall_seconds": wall,
                "ms_per_update": (wall / updates) * 1000.0,
                "peak_cuda_mb": peak_mb,
                "checkpoint_bytes": ckpt_bytes,
                "checkpoint_write_ms": ckpt_write_s * 1000.0,
                "eval_sample_ms": eval_s * 1000.0,
                "recovery_verified": recovery_verified,
            }
            del trainer
            if not args.cpu_canary:
                torch.cuda.empty_cache()

    return {"status": "PASS", "profiles": profiles_measured}


def mode_stage0(args: argparse.Namespace) -> dict[str, Any]:
    """Audit frozen historical S20 low-lr checkpoint for one seed and export verified Stage 0 package."""
    if not args.cpu_canary and args.checkpoint_root is not None:
        _assert_cuda_isolated()
    else:
        configure_cpu()
    device = "cpu" if args.cpu_canary else ("cuda:0" if args.checkpoint_root else "cpu")
    seed = args.seed
    admission = admit_historical_s20_checkpoint(
        args.checkpoint_root,
        seed,
        expected_data_sha=args.expected_data_sha,
        device=device,
    )
    admission_receipt = {k: v for k, v in admission.items() if k != "backend"}
    out_dir = args.output / "analysis" / "stage0" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if admission["status"] != "ADMITTED":
        summary = {
            "schema": "elendril-hitchhiker-stage0-seed-v1",
            "status": "CHECKPOINT_AUDIT_PENDING",
            "seed": seed,
            "executing_source_commit": args.source,
            "audited_v717_ancestor": AUDITED_V717_COMMIT,
            "admission_receipt": admission_receipt,
        }
        _write_derived(out_dir / "stage0_summary.json", canonical(summary) + b"\n")
        return summary

    suite = load_suite_bundle(args.prepared / "validation_suite.json")
    probes = load_suite_bundle(args.prepared / "validation_probes.json")
    orig_dev = read_dataset(args.prepared / "validation_orig.json")
    struct_suite = load_suite_bundle(args.prepared / "structural_suite.json")
    diag = _evaluate_backend_decoders(
        admission["backend"],
        suite,
        probes,
        original_dev=orig_dev,
        structural_suite=struct_suite,
        full_260_inspection=True,
    )
    pkg_dir = out_dir / "stage0_package"
    pkg_receipt = write_stage0_evidence_package(
        pkg_dir,
        executing_source_commit=args.source,
        historical_seed=seed,
        admission_receipt=admission_receipt,
        diag=diag,
        suite_case_count=sum(len(v) for v in suite["cohorts"].values()),
        probe_case_count=sum(len(v) for v in probes.values()),
    )
    pkg_verify = verify_and_rescore_stage0_package(
        pkg_dir,
        suite=suite,
        probes=probes,
        original_dev=orig_dev,
        structural_suite=struct_suite,
    )
    summary = {
        "schema": "elendril-hitchhiker-stage0-seed-v1",
        "status": "EXECUTED",
        "seed": seed,
        "executing_source_commit": args.source,
        "audited_v717_ancestor": AUDITED_V717_COMMIT,
        "historical_training_source": HISTORICAL_S20_LOWLR_MANIFESTS[seed]["source_commit"],
        "admission_receipt": admission_receipt,
        "diagnostic_classification": diag["diagnostic_classification"],
        "entity_equivariance": diag["entity_equivariance"],
        "decoders": {
            name: {k: v for k, v in dec.items() if not k.startswith("_")}
            for name, dec in diag["decoders"].items()
        },
        "stage0_package": {**pkg_receipt, "verification": pkg_verify},
    }
    _write_derived(out_dir / "stage0_summary.json", canonical(summary) + b"\n")
    return summary


def mode_train(args: argparse.Namespace) -> dict[str, Any]:
    """Execute or resume a single registered Hitchhiker training job and emit complete eval receipts."""
    if not args.cpu_canary:
        _assert_cuda_isolated()
    else:
        configure_cpu()
    device = "cpu" if args.cpu_canary else "cuda:0"
    job = json.loads(Path(args.job_file).read_text(encoding="utf-8"))
    job_id = job["job_id"]
    winning_arm = args.winning_arm or ("AB" if job["stage"] != "stage3_scale_confirmation" else job["arm"])
    validate_job(job, job["stage"], winning_arm=winning_arm)

    selection = None
    if args.selection_file and Path(args.selection_file).is_file():
        selection = json.loads(Path(args.selection_file).read_text(encoding="utf-8"))
    if not args.cpu_canary and job["stage"] in ("stage2_followups", "stage3_scale_confirmation"):
        validate_stage_selection(job["stage"], selection, winning_arm=winning_arm)

    job_dir = args.output / "analysis" / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_derived(job_dir / "job_spec.json", canonical(job) + b"\n")

    base_data = read_dataset(args.prepared / "train_base.json")
    val_suite = load_suite_bundle(args.prepared / "validation_suite.json")
    val_probes = load_suite_bundle(args.prepared / "validation_probes.json")
    val_orig = read_dataset(args.prepared / "validation_orig.json")
    conf_suite = load_suite_bundle(args.prepared / "confirmation_suite.json")
    struct_suite = load_suite_bundle(args.prepared / "structural_suite.json")

    spec_override = _build_canary_spec()[1] if args.cpu_canary else None
    batch_size = 4 if args.cpu_canary else job["batch_size"]
    total_updates = args.updates_override if args.cpu_canary else None

    # Pre-check deterministic skip if final eval + matching LATEST checkpoint already exist.
    ckpt_root = args.output / "checkpoints"
    latest_file = ckpt_root / job_id / "LATEST"
    eval_file = job_dir / "final_eval.json"
    receipt_file = job_dir / "run_receipt.json"
    resume = latest_file.is_file()

    # Build heldout suites ensuring train/eval disjoint custody on every arm.
    heldout = (val_suite, conf_suite, struct_suite, val_probes)
    if job["arm"] in ("B0", "C", "C+penalty"):
        pass

    # If a pre-built serialized stream exists, load it cleanly; otherwise create deterministically.
    stream = None
    if not args.cpu_canary:
        expected_updates = math.ceil(len(base_data.samples) / batch_size) * job["epochs"]
        stream = build_training_stream(
            job["arm"],
            base_data=base_data,
            total_updates=expected_updates,
            batch_size=batch_size,
            data_seed=606,
            order_seed=job["order_seed"],
            relabel_seed=1009,
            heldout_suites=heldout,
        )

    def _on_signal(_signum: int, _frame: Any) -> None:
        raise WorkerInterrupted("received termination signal")

    old_term = signal.signal(signal.SIGTERM, _on_signal)
    trainer = None
    try:
        trainer, active_stream = create_hitchhiker_trainer(
            job,
            source_commit=args.source,
            root=ckpt_root,
            base_data=base_data,
            stream=stream,
            heldout_suites=heldout,
            resume=resume,
            device=device,
            cpu_canary=args.cpu_canary,
            spec_override=spec_override,
            updates_override=total_updates,
            batch_override=batch_size if args.cpu_canary else None,
            winning_arm=winning_arm,
            selection_receipt=selection,
        )
        if (
            resume
            and eval_file.is_file()
            and receipt_file.is_file()
            and trainer.state.global_update == trainer.recipe.total_updates
        ):
            existing = json.loads(eval_file.read_text(encoding="utf-8"))
            if (
                existing.get("protocol_sha256") == PROTOCOL_SHA
                and existing.get("checkpoint_sha256") == trainer.store.latest_sha256()
            ):
                return {
                    "status": "PASS",
                    "job_id": job_id,
                    "arm": job["arm"],
                    "stage": job["stage"],
                    "resumed_skipped": True,
                    "checkpoint_sha256": existing["checkpoint_sha256"],
                    "run_receipt": json.loads(receipt_file.read_text(encoding="utf-8")),
                }

        aux_probe = None
        if job["arm"] == "C+penalty":
            aux_probe = aux_penalty_gradient_probe(trainer)
            if aux_probe["status"] != "PASS":
                raise RuntimeError("aux_penalty_gradient_probe failed on C+penalty")

        save_interval = max(1, min(100, trainer.recipe.total_updates // 4)) if not args.cpu_canary else 1
        t0 = time.perf_counter()
        logs = []
        interrupted = False
        try:
            for _ in range(trainer.state.global_update, trainer.recipe.total_updates):
                step_log = trainer.step()
                logs.append(step_log)
                if step_log["update"] % save_interval == 0 or step_log["update"] == trainer.recipe.total_updates:
                    trainer.save()
                    if trainer.parent:
                        trainer.store.prune(keep={trainer.parent})
                if args.cpu_canary and args.cpu_interrupt_after is not None and step_log["update"] == args.cpu_interrupt_after:
                    raise WorkerInterrupted(f"canary interrupt at update {step_log['update']}")
        except (WorkerInterrupted, KeyboardInterrupt):
            interrupted = True
            if trainer is not None and trainer.state.global_update > trainer.published_update and not trainer.faulted:
                trainer.save()

        if interrupted:
            latest_sha = trainer.store.latest_sha256()
            _write_derived(
                job_dir / "status.json",
                canonical({"status": "INTERRUPTED", "job_id": job_id, "global_update": trainer.state.global_update, "checkpoint_sha256": latest_sha}) + b"\n",
            )
            return {"status": "INTERRUPTED", "job_id": job_id, "global_update": trainer.state.global_update, "checkpoint_sha256": latest_sha}

        if trainer.state.global_update != trainer.recipe.total_updates:
            raise RuntimeError(f"trainer stopped early at {trainer.state.global_update}/{trainer.recipe.total_updates}")
        latest_sha = trainer.store.latest_sha256() or trainer.save()
        if trainer.parent:
            trainer.store.prune(keep={trainer.parent})
        wall = time.perf_counter() - t0

        policy = (
            ExternalSlotCanonicalizerPolicy(trainer.backend, assignment="first-mention")
            if job["arm"] == "D-external"
            else RawHitchhikerPolicy(trainer.backend)
        )
        orig_eval = evaluate_original_kinds(val_orig, policy, split="validation")
        if job["stage"] in ("stage1_s3_screen", "stage2_followups"):
            primary_eval = evaluate_counterfactual_suite(val_suite, policy)
            equivariance = score_entity_equivariance(
                trainer.backend,
                val_suite["cohorts"]["factor-entity-permutation"],
                val_suite["permutations"],
            )
            probes_eval = evaluate_health_probes(val_probes, policy, split="validation")
            structural_eval = None
        else:
            primary_eval = evaluate_counterfactual_suite(conf_suite, policy)
            equivariance = score_entity_equivariance(
                trainer.backend,
                conf_suite["cohorts"]["factor-entity-permutation"],
                conf_suite["permutations"],
            )
            probes_eval = None
            structural_eval = evaluate_counterfactual_suite(struct_suite, policy)

        run_receipt = {
            "schema": "elendril-hitchhiker-job-run-v1",
            "protocol_sha256": PROTOCOL_SHA,
            "source_commit": args.source,
            "job_id": job_id,
            "arm": job["arm"],
            "stage": job["stage"],
            "seed": job["seed"],
            "profile": job["profile"],
            "attribution_origin": job["attribution_origin"],
            "updates_executed": len(logs),
            "global_update": trainer.state.global_update,
            "target_updates": trainer.recipe.total_updates,
            "wall_seconds": wall,
            "supervised_tokens": sum(r["supervised_tokens"] for r in logs),
            "input_positions": sum(r["input_positions"] for r in logs),
            "examples_processed": sum(r["examples"] for r in logs),
            "final_loss": logs[-1]["loss"] if logs else None,
            "checkpoint_sha256": latest_sha,
            "stream_metrics": active_stream["stream_metrics"],
            "aux_penalty_probe": aux_probe,
        }
        eval_bundle = {
            "schema": "elendril-hitchhiker-job-eval-v1",
            "protocol_sha256": PROTOCOL_SHA,
            "source_commit": args.source,
            "job_id": job_id,
            "arm": job["arm"],
            "stage": job["stage"],
            "seed": job["seed"],
            "checkpoint_sha256": latest_sha,
            "model_identity": primary_eval.get("model_identity"),
            "attribution_origin": primary_eval["attribution_origin"],
            "decoder_schema": primary_eval["decoder_schema"],
            "split": primary_eval["split"],
            "cohort_metrics": primary_eval["cohort_metrics"],
            "factor_summary": primary_eval["factor_summary"],
            "original_kind_accuracies": orig_eval["original_kind_accuracies"],
            "equivariance_report": equivariance,
            "probes_scores": probes_eval["scores"] if probes_eval else None,
            "structural_metrics": structural_eval["cohort_metrics"] if structural_eval else None,
            "suite_eval": {
                **primary_eval,
                "source_commit": args.source,
                "original_kind_accuracies": orig_eval["original_kind_accuracies"],
            },
            "original_eval": orig_eval,
            "probes_eval": probes_eval,
            "structural_eval": structural_eval,
        }
        _write_derived(receipt_file, canonical(run_receipt) + b"\n")
        _write_derived(eval_file, canonical(eval_bundle) + b"\n")
        _write_derived(job_dir / "status.json", canonical({"status": "PASS", "job_id": job_id, "checkpoint_sha256": latest_sha}) + b"\n")
        return {
            "status": "PASS",
            "job_id": job_id,
            "arm": job["arm"],
            "stage": job["stage"],
            "resumed_skipped": False,
            "checkpoint_sha256": latest_sha,
            "run_receipt": run_receipt,
        }
    finally:
        signal.signal(signal.SIGTERM, old_term)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("profile", "recover", "stage0", "train"))
    p.add_argument("--source", required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--checkpoints", type=Path, default=None)
    p.add_argument("--checkpoint-root", type=Path, default=None)
    p.add_argument("--job-file", type=Path, default=None)
    p.add_argument("--selection-file", type=Path, default=None)
    p.add_argument("--winning-arm", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--expected-data-sha", default=None)
    p.add_argument("--profile", default="S3")
    p.add_argument("--from-update", type=int, default=4)
    p.add_argument("--to-update", type=int, default=8)
    p.add_argument("--ref-sha", default=None)
    p.add_argument("--ref-file", type=Path, default=None)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--cpu-canary", action="store_true")
    p.add_argument("--updates-override", type=int, default=None)
    p.add_argument("--cpu-interrupt-after", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.mode == "profile":
        result = mode_profile(args)
    elif args.mode == "recover":
        result = mode_recover(args)
    elif args.mode == "stage0":
        result = mode_stage0(args)
    else:
        result = mode_train(args)
    _write_derived(args.receipt, canonical(result) + b"\n")
    print(json.dumps({"status": result.get("status"), "mode": args.mode}, sort_keys=True), flush=True)
    return 0 if result.get("status") in ("PASS", "EXECUTED", "CHECKPOINT_AUDIT_PENDING") else 2 if result.get("status") == "INTERRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
