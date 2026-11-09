"""Isolated V7.15 GPU evaluator for verified V7.1 S3 checkpoints.

Raw and grammar decoding run on exactly the same frozen weights. No optimizer
is built and no training is performed. Gold labels remain evaluator-side.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import time
import traceback

from anra_v5.elendril.v715_protocol import PARENT_SOURCE, PROTOCOL_SHA, registered_jobs
from anra_v5.elendril.v71_data import suite_from_record
from anra_v5.elendril.v71_evaluation import evaluate_suite, verify_suite
from anra_v5.elendril.v71_policy import GrammarPolicy
from anra_v5.elendril.v6_evaluation import NeuralPolicy
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.backend import NeuralBackend, BackendIdentity
from anra_v5.elendril.training import configure_device, _load
from tools.elendril_cloud import check_source
from tools.elendril_v6_export import immutable_json
from tools.elendril_v6_worker import _weights
from v5_model.core import initialize
from v5_training.checkpoint import CheckpointStore

REPO = Path(__file__).resolve().parents[1]


def run(req):
    import torch
    if req["protocol_sha"] != PROTOCOL_SHA or req["job"] not in registered_jobs():
        raise ValueError("V7.15 frozen job/protocol mismatch")
    check_source(REPO, req["source"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one isolated CUDA GPU required")
    job = req["job"]
    parent = Path(req["parent_output"]).resolve()
    root = parent / "checkpoints" / "jobs" / job["job_id"]
    report_path = parent / "analysis" / "jobs" / job["job_id"] / "epoch-4.json"
    source_report = json.loads(report_path.read_bytes())
    checkpoint = source_report["checkpoint"]
    if req["checkpoint"] != checkpoint or source_report["weight_sha256"] != req["expected_weight_sha256"]:
        raise ValueError("V7.1 immutable final report identity mismatch")
    state, payloads = CheckpointStore(root, job["job_id"]).restore(checkpoint)
    spec = MODEL_PROFILES["S3"]
    if (not state.complete or state.identities.source_commit != PARENT_SOURCE
            or state.identities.model_spec_sha256 != spec.sha256()
            or state.identities.tokenizer_sha256 != UTF8.identity):
        raise ValueError("frozen source/spec/tokenizer/complete state mismatch")
    weights = _load(payloads["model.bin"])
    if not all(torch.isfinite(w).all().item() for w in weights.values()):
        raise ValueError("checkpoint contains nonfinite model tensor")
    configure_device("cuda:0")
    model = initialize(spec, job["seed"]).to("cuda:0")
    model.load_state_dict(weights, strict=True)
    model.eval()
    before = _weights(model)
    if before != req["expected_weight_sha256"]:
        raise ValueError("restored checkpoint weights differ from original final report")
    backend = NeuralBackend(
        model, BackendIdentity(spec.sha256(), UTF8.identity, 260, spec.context_length, checkpoint),
        pad_id=UTF8.pad_id, bos_id=UTF8.bos_id, eos_id=UTF8.eos_id
    )
    suite = suite_from_record(json.loads(Path(req["suite_file"]).read_bytes()))
    output = Path(req["output"])
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    started = time.monotonic()
    for label, policy in (("raw", NeuralPolicy(backend)), ("grammar", GrammarPolicy(backend))):
        if time.monotonic() > req["deadline_monotonic"] - 180:
            raise TimeoutError("not enough reserved time for remaining frozen diagnostics")
        target = output / (label + ".json")
        if target.exists():
            row = json.loads(target.read_bytes())
            if (row.get("checkpoint") != checkpoint or row.get("weight_sha256") != before
                    or row.get("protocol_sha256") != PROTOCOL_SHA or row.get("decoder") != label):
                raise ValueError("cached V7.15 evaluation lineage mismatch")
            verify_suite(suite, row["evaluation"])
        else:
            with torch.inference_mode():
                report = evaluate_suite(suite, policy)
            verify_suite(suite, report)
            row = {
                "schema": "elendril-v715-frozen-report-v1",
                "job_id": job["job_id"], "checkpoint": checkpoint,
                "weight_sha256": before, "protocol_sha256": PROTOCOL_SHA,
                "decoder": label, "source": req["source"],
                "parent_source": PARENT_SOURCE,
                "evaluation": report,
            }
            immutable_json(target, row)
        if (row["evaluation"]["conditions"] != sorted(suite)
                or any(e["raw"]["model_identity"] != checkpoint for e in row["evaluation"]["records"].values() if e)):
            raise ValueError("frozen checkpoint binding or cohort coverage mismatch")
        reports.append({"decoder": label, "path": str(target), "cohorts": len(suite),
                        "metrics": row["evaluation"]["metrics"]})
    if _weights(model) != before:
        raise ValueError("frozen evaluation modified neural weights")
    return {
        "status": "COMPLETE", "job_id": job["job_id"],
        "checkpoint": checkpoint, "weight_sha256": before,
        "cohorts": len(suite), "decoders": ["raw", "grammar"],
        "wall_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scientific_status": "CAUSE UNRESOLVED; no optimizer updates; developer synthetic worlds",
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("frozen",))
    p.add_argument("--owner-enabled", action="store_true")
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.owner_enabled:
        p.error("owner authorization required")
    req = json.loads(args.request.read_bytes())
    if args.receipt.exists() or not args.receipt.resolve().is_relative_to(Path(req["output"]).resolve()):
        raise ValueError("refuse to overwrite or misplace a worker receipt")
    def interrupted(_signum, _frame):
        raise InterruptedError("worker interrupted; verified partial reports remain")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        row = run(req)
    except Exception as exc:
        row = {"status": "ERROR", "error_type": type(exc).__name__,
               "message": str(exc), "traceback": traceback.format_exc()}
    row.update(source=req["source"], protocol_sha256=PROTOCOL_SHA, mode=args.mode)
    immutable_json(args.receipt, row)
    return 0 if row["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
