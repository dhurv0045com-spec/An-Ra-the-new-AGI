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

from .benchmark import (BINDING_PANELS, FAMILIES, POLICIES, binding_panel,
                        binding_panel_prompts, corrupt_binding, fixture, shortcut)
from .codec import CODEC_SHA, Prediction, answer_tokens, canonical, digest
from .counterfactual import METHODS
from .join import JOIN_PANELS, join_fixture
from .relational import ROBUST_PANELS, relational_fixture
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


def experiment_rows(surface: str, family: str, split: str):
    if surface == "join-v1":
        if family != "binding":
            raise ValueError("join surface requires a binding arm")
        return join_fixture(split)
    if surface == "relational-v1":
        if family != "binding":
            raise ValueError("the relational surface requires a binding arm")
        return relational_fixture(split)
    if surface != "classic":
        raise ValueError("unknown evaluation surface")
    return fixture(family, split)


def experiment_panels(surface: str):
    if surface == "join-v1":
        return JOIN_PANELS
    return ROBUST_PANELS if surface == "relational-v1" else BINDING_PANELS


def experiment_panel(surface: str, name: str):
    if name == "drop-second-read":
        return join_fixture("development")
    if surface == "join-v1":
        return join_fixture("train" if name == "train" else "development", name)
    if surface == "relational-v1":
        return relational_fixture("train" if name == "train" else "development", name)
    return binding_panel(name)


def pilot(args) -> dict:
    from .training import configure_device, data_sha, generate, runtime_identity, Trainer

    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("output must be a new file inside an existing directory")
    if not 1 <= args.until <= args.steps:
        raise ValueError("until must be within the training budget")
    source = committed_source()
    device = configure_device(args.device)
    if device.type == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    cpu_started = time.process_time()
    family = "copy" if args.arm == "copy" else "binding"
    if family != "binding" and args.method != "reference":
        raise ValueError("counterfactual methods require binding")
    train = experiment_rows(args.surface, family, "train")
    corruption_seed = args.seed + 100_000
    if args.arm == "binding-permuted":
        train = corrupt_binding(train, corruption_seed)
    dev = experiment_rows(args.surface, family, "development")
    prompts = tuple((row.visible.task_id, row.visible.prompt) for row in dev)
    lineage = f"{args.arm}-{args.seed}"
    if args.surface != "classic":
        lineage = f"{args.surface}-{lineage}"
    if args.method != "reference":
        lineage = f"{args.method}-{lineage}"
    if args.profile != "micro":
        lineage = f"{args.profile}-{lineage}"
    trainer = Trainer(train, seed=args.seed, steps=args.steps, root=args.root,
                      lineage=lineage, source_commit=source, resume=args.resume, method=args.method, profile=args.profile, device=device)
    start = trainer.state.global_update
    if start > args.until:
        raise ValueError("checkpoint is beyond the requested stop")
    start_checkpoint = trainer.parent
    max_new = 5 if family == "copy" else 3
    evaluations = []
    updates = []
    training_seconds = 0.0

    def evaluate(rows, visible_prompts, *, remove_second_read=False) -> dict:
        begin = time.perf_counter()
        predictions, cost = generate(trainer.model, visible_prompts, max_new_tokens=max_new,
                                     remove_second_read=remove_second_read)
        raw = [asdict(item) for item in predictions]
        return {
            "update": trainer.state.global_update, "checkpoint_sha256": trainer.parent,
            "segment_training_seconds": training_seconds,
            "inference": {**cost, "wall_seconds": time.perf_counter() - begin,
                          "emitted_tokens": sum(len(item.tokens) for item in predictions)},
            "predictions": raw, "predictions_sha256": digest(raw), "metrics": score(rows, predictions),
        }

    evaluations.append(evaluate(dev, prompts))
    while trainer.state.global_update < args.until:
        begin = time.perf_counter()
        updates.append(trainer.step())
        training_seconds += time.perf_counter() - begin
        step = trainer.state.global_update
        if args.surface == "join-v1":
            scheduled = step % (96 if args.profile == "micro" else 384) == 0
        elif args.surface == "relational-v1":
            scheduled = step % 88 == 0
        else:
            scheduled = step in (50, 150) if args.method == "reference" else step % 84 == 0
        if scheduled or step == args.until:
            trainer.save()
            evaluations.append(evaluate(dev, prompts))
    panels = {}
    if args.method != "reference" and trainer.state.complete:
        names = experiment_panels(args.surface)
        if args.surface == "join-v1" and args.method == "two-read":
            names = (*names, "drop-second-read")
        for name in names:
            rows = experiment_panel(args.surface, name)
            visible_prompts = binding_panel_prompts(name, rows)
            panels[name] = {
                **evaluate(rows, visible_prompts, remove_second_read=name == "drop-second-read"),
                "fixture_sha256": digest([asdict(row) for row in rows]),
                "model_prompts_sha256": digest(visible_prompts),
            }
    accelerator = None
    if device.type == "cuda":
        accelerator = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    return {
        "schema": "elendril-development-pilot/v1", "accelerator": accelerator, "source_commit": source,
        "command": ["python", "-m", "anra_v5.elendril", *args.command_line],
        "arm": args.arm, "seed": args.seed, "corruption_seed": corruption_seed if family == "binding"
        and args.arm == "binding-permuted" else None,
        "method": args.method, "lineage": lineage, "surface": args.surface, "profile": args.profile,
        "environment": runtime_identity(device), "model_parameters": sum(p.numel() for p in trainer.model.parameters()),
        "identity": trainer.state.identities.canonical(), "codec_sha256": CODEC_SHA,
        "training_sha256": data_sha(train), "development_sha256": digest([asdict(row) for row in dev]),
        "benchmark_source_sha256": sha256(Path(__file__).with_name(
            {"join-v1": "join.py", "relational-v1": "relational.py"}.get(args.surface, "benchmark.py")).read_bytes()).hexdigest(),
        "start_update": start, "start_checkpoint_sha256": start_checkpoint,
        "training_tokens": trainer.state.cumulative_tokens, "final_checkpoint_sha256": trainer.parent,
        "wall_seconds": time.perf_counter() - started, "process_cpu_seconds": time.process_time() - cpu_started,
        "stop_reason": "frozen_update_budget" if trainer.state.complete else "requested_boundary",
        "status": "completed" if trainer.state.complete else "partial", "updates": updates,
        "evaluations": evaluations, "panels": panels,
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
    run.add_argument("--method", choices=tuple(METHODS), default="reference")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--surface", choices=("classic", "relational-v1", "join-v1"), default="classic")
    run.add_argument("--profile", choices=("micro", "expanded"), default="micro")
    run.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    run.add_argument("--steps", type=int)
    run.add_argument("--until", type=int)
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    rescore = sub.add_parser("rescore")
    rescore.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(command_line)
    if args.mode == "pilot":
        args.command_line = command_line
        if args.steps is None:
            args.steps = {"join-v1": 384, "relational-v1": 352}.get(args.surface, 350)
        args.until = args.steps if args.until is None else args.until
        write_receipt(args.output, pilot(args))
    elif args.mode == "audit":
        write_receipt(args.output, benchmark_audit())
    else:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        family = "copy" if receipt["arm"] == "copy" else "binding"
        surface = receipt.get("surface", "classic")
        rows = experiment_rows(surface, family, "development")
        if receipt["development_sha256"] != digest([asdict(row) for row in rows]):
            raise ValueError("receipt development identity mismatch")
        def verify(item, fixture_rows):
            if digest(item["predictions"]) != item["predictions_sha256"]:
                raise ValueError("receipt prediction identity mismatch")
            metrics = score(fixture_rows, (Prediction(row["task_id"], tuple(row["tokens"]))
                                           for row in item["predictions"]))
            if metrics != item["metrics"]:
                raise ValueError("receipt metrics disagree with independent scorer")

        for item in receipt["evaluations"]:
            verify(item, rows)
        panels = receipt.get("panels", {})
        expected_panels = set(experiment_panels(surface)) if (receipt.get("method", "reference") != "reference"
                                                 and receipt["status"] == "completed") else set()
        if expected_panels and surface == "join-v1" and receipt["method"] == "two-read":
            expected_panels.add("drop-second-read")
        if panels.keys() != expected_panels:
            raise ValueError("receipt panel coverage mismatch")
        for name, item in panels.items():
            panel_rows = experiment_panel(surface, name)
            if digest([asdict(row) for row in panel_rows]) != item["fixture_sha256"]:
                raise ValueError("receipt panel fixture mismatch")
            if digest(binding_panel_prompts(name, panel_rows)) != item["model_prompts_sha256"]:
                raise ValueError("receipt panel prompt identity mismatch")
            verify(item, panel_rows)
        print(f"verified {len(receipt['evaluations']) + len(panels)} scoring snapshots")


if __name__ == "__main__":
    main()
