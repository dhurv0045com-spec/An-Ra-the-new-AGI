"""Run a reproducible exploratory discovery and retention campaign.

Example: python -m bramastra_lab.discovery.run --output artifacts/bramastra/discovery_dev_701
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import time

import torch

from .curriculum import Demonstrations, build_demonstrations, demonstration_digest
from .evaluation import evaluate
from .learner import Investigator, LearnerConfig
from .statistics import summarize
from .training import Trainer, TrainConfig, continuation_probe, fingerprint
from .worlds import make_worlds, split_worlds


def write_json(path: Path, value) -> None:
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def replay_mix(old: Demonstrations, new: Demonstrations, seed: int) -> Demonstrations:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(old), (len(new),), generator=generator)
    names = ("observations", "lengths", "targets", "labels", "legal", "gains")
    values = [torch.cat((getattr(old, n)[indices], getattr(new, n))) for n in names]
    manifest = {"mix": "equal state counts: prior/new", "seed": seed,
                "old": old.manifest, "new": new.manifest, "states": len(new) * 2,
                "tensor_sha256": demonstration_digest(values)}
    return Demonstrations(*values, manifest)


def train_arm(data, config, model_config, output: Path, parent=None):
    torch.manual_seed(config.seed)
    model = Investigator(model_config)
    if parent is not None:
        model.load_state_dict(parent)
    before = fingerprint(model)
    trainer = Trainer(model, data, config)
    trace = []
    started = time.perf_counter()
    for _ in range(config.steps):
        metrics = trainer.update()
        if trainer.step == 1 or trainer.step % 100 == 0 or trainer.step == config.steps:
            trace.append(metrics)
            print(json.dumps({"arm": output.name, **metrics}), flush=True)
    elapsed = time.perf_counter() - started
    output.mkdir()
    checkpoint = output / "checkpoint.pt"
    continuation = continuation_probe(trainer, checkpoint)
    report = {"specification": model.specification(), "training": config.__dict__,
              "seconds": elapsed, "initial_sha256": before, "final_sha256": fingerprint(model),
              "parameters_changed": before != fingerprint(model), "trace": trace,
              "continuation": continuation, "data": data.manifest,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    write_json(output / "training.json", report)
    return model, report


def run(args) -> dict:
    if args.output.exists():
        raise ValueError("output exists; choose a new run identity")
    if min(args.episodes, args.steps, args.eval_worlds, args.targets, args.threads) <= 0:
        raise ValueError("run counts must be positive")
    if args.adapt_steps < 0 or not 0 <= args.budget <= 2 ** args.bits - 2:
        raise ValueError("invalid adaptation/query budget")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    source_dir = Path(__file__).parent
    snapshots = args.output / "source_snapshot"
    snapshots.mkdir()
    source_hashes = {}
    for source in sorted(source_dir.glob("*.py")):
        shutil.copyfile(source, snapshots / source.name)
        source_hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    # Exclude very imbalanced functions: majority answers must not dominate the study.
    worlds = [w for w in make_worlds(args.bits) if 2 ** args.bits / 4 <= sum(w.table) <= 3 * 2 ** args.bits / 4]
    partitions = split_worlds(worlds)
    training = [w for w in partitions["train"] if w.family != "threshold"]
    acquisition = [w for w in partitions["train"] if w.family == "threshold"]
    rng = random.Random(args.seed + 11)
    evaluation_worlds = []
    # DEV is explicit: these results may guide subsequent design, not confirmation claims.
    for family in sorted({w.family for w in worlds}):
        pool = [w for w in partitions["dev"] if w.family == family]
        evaluation_worlds.extend(rng.sample(pool, min(args.eval_worlds, len(pool))))
    if not training or not acquisition or not evaluation_worlds:
        raise ValueError("selected bit width lacks required training/acquisition/evaluation worlds")
    manifest = {"schema": "bramastra-discovery-campaign/v1", "status": "exploratory development",
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
                "source_sha256": source_hashes, "families": dict(Counter(w.family for w in worlds)),
                "training_ids": [w.world_id for w in training], "acquisition_ids": [w.world_id for w in acquisition],
                "evaluation_ids": [w.world_id for w in evaluation_worlds],
                "untouched_test_ids": [w.world_id for w in partitions["test"]],
                "filter": "positive truth-table fraction between 1/4 and 3/4 inclusive",
                "policy_teacher": "exact entropy reduction on training hypotheses only",
                "confirmation": "NOT_RUN; test world scores are not computed",
                "pretrained_weights": False, "adaptive_code_modification": False}
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "worlds.json", [w.__dict__ for w in worlds])
    data = build_demonstrations(training, episodes=args.episodes, budget=args.budget, seed=args.seed)
    config = LearnerConfig(bits=args.bits, width=args.width)
    model, train_report = train_arm(data, TrainConfig(steps=args.steps, seed=args.seed), config,
                                   args.output / "initial")
    budgets = tuple(sorted({0, min(2, args.budget), min(4, args.budget), args.budget}))
    policies = ("random", "coverage", "learned", "uncertainty", "no_memory")
    rows = evaluate(model, evaluation_worlds, policies=policies, budgets=budgets,
                    targets_per_world=args.targets, seed=args.seed + 19)
    write_json(args.output / "initial" / "evaluation_rows.json", rows)
    initial = summarize(rows, seed=args.seed)
    write_json(args.output / "initial" / "evaluation.json", initial)
    adaptations = {}
    if args.adapt_steps:
        new = build_demonstrations(acquisition, episodes=max(64, args.episodes // 4),
                                   budget=args.budget, seed=args.seed + 1)
        parent = copy.deepcopy(model.state_dict())
        for name, dataset in (("new_only", new), ("replay", replay_mix(data, new, args.seed))):
            child, report = train_arm(dataset, TrainConfig(steps=args.adapt_steps, seed=args.seed + 2),
                                      config, args.output / name, parent=parent)
            child_rows = evaluate(child, evaluation_worlds, policies=policies, budgets=budgets,
                                  targets_per_world=args.targets, seed=args.seed + 19)
            child_summary = summarize(child_rows, seed=args.seed)
            write_json(args.output / name / "evaluation_rows.json", child_rows)
            write_json(args.output / name / "evaluation.json", child_summary)
            adaptations[name] = {"training": report, "evaluation": child_summary}
    result = {"initial": {"training": train_report, "evaluation": initial}, "adaptations": adaptations,
              "elapsed_seconds": time.perf_counter() - started,
              "limits": ["synthetic Boolean rules, not broad AGI", "policy imitates a privileged training-only teacher",
                         "development results, not independent confirmation", "CPU only; no TPU validation",
                         "consolidation resets AdamW for both children; it is not a full resume experiment",
                         "one acquisition round; no autonomous promotion or recursive self-improvement"]}
    write_json(args.output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--bits", type=int, default=6)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--adapt-steps", type=int, default=200)
    parser.add_argument("--eval-worlds", type=int, default=48, help="maximum per family, DEV only")
    parser.add_argument("--targets", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"output": str(args.output), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
