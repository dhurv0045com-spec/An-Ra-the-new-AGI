"""Bounded development experiment, fixture audit and independent rescoring."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time

from .benchmark import FAMILIES, POLICIES, corrupt_binding, fixture, shortcut
from .codec import CODEC_SHA, Prediction, answer_tokens, canonical, digest
from .scoring import score


def write_receipt(path: Path, receipt: dict) -> None:
    # Publish only complete bytes, and refuse to replace any existing receipt.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        pending = Path(handle.name)
    try:
        with pending.open("wb") as handle:
            handle.write(canonical(receipt) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(pending, path)
    finally:
        pending.unlink()


def benchmark_audit() -> dict:
    result = {}
    for family in FAMILIES:
        train, dev = fixture(family, "train"), fixture(family, "development")
        train_clusters = {row.visible.cluster_id for row in train}
        dev_clusters = {row.visible.cluster_id for row in dev}
        train_prompts = {row.visible.prompt for row in train}
        dev_prompts = {row.visible.prompt for row in dev}
        if train_clusters & dev_clusters or train_prompts & dev_prompts:
            raise ValueError("fixture overlap")
        result[family] = {
            "train_rows": len(train), "dev_rows": len(dev), "dev_clusters": len(dev_clusters),
            "development_sha256": digest([asdict(row) for row in dev]),
            "oracle": score(dev, (Prediction(row.visible.task_id, answer_tokens(row.truth.gold))
                                  for row in dev)),
            "controls": {},
        }
        for policy in POLICIES:
            rng = random.Random(141)
            result[family]["controls"][policy] = score(
                dev, (shortcut(row.visible, policy, rng) for row in dev))
    return result


def committed_source() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "*.py"], text=True)
    if status.strip():
        raise ValueError("pilot requires a clean Python source tree")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def pilot(args) -> dict:
    from .training import MODEL, configure_cpu, data_sha, generate, runtime_identity, Trainer

    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("output must be a new file inside an existing directory")
    if not 1 <= args.until <= args.steps:
        raise ValueError("until must be within the training budget")
    source = committed_source()
    configure_cpu()
    started = time.perf_counter()
    cpu_started = time.process_time()
    family = "copy" if args.arm == "copy" else "binding"
    train = fixture(family, "train")
    corruption_seed = args.seed + 100_000
    if args.arm == "binding-permuted":
        train = corrupt_binding(train, corruption_seed)
    dev = fixture(family, "development")
    prompts = tuple((row.visible.task_id, row.visible.prompt) for row in dev)
    trainer = Trainer(train, seed=args.seed, steps=args.steps, root=args.root,
                      lineage=f"{args.arm}-{args.seed}", source_commit=source, resume=args.resume)
    start = trainer.state.global_update
    if start > args.until:
        raise ValueError("checkpoint is beyond the requested stop")
    start_checkpoint = trainer.parent
    max_new = 5 if family == "copy" else 3
    evaluations = []
    updates = []

    def evaluate() -> None:
        begin = time.perf_counter()
        predictions, cost = generate(trainer.model, prompts, max_new_tokens=max_new)
        raw = [asdict(item) for item in predictions]
        evaluations.append({
            "update": trainer.state.global_update, "checkpoint_sha256": trainer.parent,
            "inference": {**cost, "wall_seconds": time.perf_counter() - begin,
                          "emitted_tokens": sum(len(item.tokens) for item in predictions)},
            "predictions": raw, "predictions_sha256": digest(raw), "metrics": score(dev, predictions),
        })

    evaluate()
    while trainer.state.global_update < args.until:
        updates.append(trainer.step())
        step = trainer.state.global_update
        if step in (50, 150) or step == args.until:
            trainer.save()
            evaluate()
    return {
        "schema": "elendril-development-pilot/v1", "source_commit": source,
        "command": ["python", "-m", "anra_v5.elendril", *args.command_line],
        "arm": args.arm, "seed": args.seed, "corruption_seed": corruption_seed if family == "binding"
        and args.arm == "binding-permuted" else None,
        "environment": runtime_identity(), "model_parameters": MODEL.parameter_receipt().total,
        "identity": trainer.state.identities.canonical(), "codec_sha256": CODEC_SHA,
        "training_sha256": data_sha(train), "development_sha256": digest([asdict(row) for row in dev]),
        "benchmark_source_sha256": sha256(Path(__file__).with_name("benchmark.py").read_bytes()).hexdigest(),
        "start_update": start, "start_checkpoint_sha256": start_checkpoint,
        "training_tokens": trainer.state.cumulative_tokens, "final_checkpoint_sha256": trainer.parent,
        "wall_seconds": time.perf_counter() - started, "process_cpu_seconds": time.process_time() - cpu_started,
        "stop_reason": "frozen_update_budget" if trainer.state.complete else "requested_boundary",
        "status": "completed" if trainer.state.complete else "partial", "updates": updates,
        "evaluations": evaluations,
    }


def main(argv: list[str] | None = None) -> None:
    import sys

    command_line = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("pilot")
    run.add_argument("--arm", choices=("copy", "binding", "binding-permuted"), required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--steps", type=int, default=350)
    run.add_argument("--until", type=int)
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    rescore = sub.add_parser("rescore")
    rescore.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(command_line)
    if args.mode == "pilot":
        args.command_line = command_line
        args.until = args.steps if args.until is None else args.until
        write_receipt(args.output, pilot(args))
    elif args.mode == "audit":
        write_receipt(args.output, benchmark_audit())
    else:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        family = "copy" if receipt["arm"] == "copy" else "binding"
        rows = fixture(family, "development")
        if receipt["development_sha256"] != digest([asdict(row) for row in rows]):
            raise ValueError("receipt development identity mismatch")
        for item in receipt["evaluations"]:
            if digest(item["predictions"]) != item["predictions_sha256"]:
                raise ValueError("receipt prediction identity mismatch")
            metrics = score(rows, (Prediction(row["task_id"], tuple(row["tokens"]))
                                   for row in item["predictions"]))
            if metrics != item["metrics"]:
                raise ValueError("receipt metrics disagree with independent scorer")
        print(f"verified {len(receipt['evaluations'])} scoring snapshots")


if __name__ == "__main__":
    main()
