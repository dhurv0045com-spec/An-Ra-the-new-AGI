"""Paired from-scratch terminal-supervision experiment, with bounded compute.

This is research instrumentation. The fixed decision rule is not a learned
self-improvement policy, and a memorized development set is not transfer.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import torch
from torch.nn import functional as F

from .data import BOS, EOS, PAD, Example, assert_disjoint, build_worlds, dataset_hash, decode, digest, encode, make_batch
from .model import BramastraModel, ModelConfig, parameter_count


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tensor_hash(tensors: dict) -> str:
    value = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value.update(name.encode())
        value.update(str(tensor.dtype).encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(pending, path)


def objective(logits, tokens, target_mask, *, terminal_weight: float):
    """Keep answer-gradient weighting identical when adding EOS supervision."""
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), tokens[:, 1:], reduction="none")
    targets = tokens[:, 1:]
    active = target_mask[:, 1:]
    answer_mask = active & (targets != EOS)
    terminal_mask = active & (targets == EOS)
    answer_count = answer_mask.sum()
    terminal_count = terminal_mask.sum()
    if int(answer_count) == 0 or int(terminal_count) != tokens.shape[0]:
        raise ValueError("each row needs answer targets and exactly one supervised terminal")
    answer_loss = (losses * answer_mask).sum() / answer_count
    terminal_loss = (losses * terminal_mask).sum() / terminal_count
    return answer_loss + terminal_weight * terminal_loss


def optimizer_for(model):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 and name != "embedding.weight" else no_decay).append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=3e-4, betas=(0.9, 0.95), eps=1e-8)


def update(model, optimizer, examples, *, config, terminal_weight, device):
    model.train()
    batch = make_batch(examples, max_length=config.max_seq, device=device)
    optimizer.zero_grad(set_to_none=True)
    loss = objective(model(batch["tokens"]), **batch, terminal_weight=terminal_weight)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite loss")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(gradient_norm)


@torch.inference_mode()
def evaluate(model, examples: list[Example], *, device, max_new_tokens: int = 4) -> dict:
    """Generate from prompt only; gold is consulted after the run has stopped."""
    if not examples or max_new_tokens <= 0:
        raise ValueError("nonempty evaluation and a positive generation budget required")
    model.eval()
    groups: dict[int, list[Example]] = defaultdict(list)
    for row in examples:
        prefix = [BOS] + encode(row.prompt)
        if len(prefix) + max_new_tokens > model.config.max_seq:
            raise ValueError("generation budget exceeds context; no gold-length truncation allowed")
        groups[len(prefix)].append(row)
    records = []
    for group in groups.values():
        for offset in range(0, len(group), 64):
            rows = group[offset:offset + 64]
            tokens = torch.tensor([[BOS] + encode(row.prompt) for row in rows], device=device)
            generated: list[list[int]] = [[] for _ in rows]
            reasons: list[str | None] = [None for _ in rows]
            for _ in range(max_new_tokens):
                next_ids = model(tokens)[:, -1].argmax(dim=-1)
                emitted = next_ids.tolist()
                for index, token in enumerate(emitted):
                    if reasons[index] is not None:
                        continue
                    if token == EOS:
                        reasons[index] = "EOS"
                    elif token < 4:
                        reasons[index] = "INVALID_SPECIAL"
                    else:
                        generated[index].append(token)
                tokens = torch.cat((tokens, next_ids[:, None]), dim=1)
                if all(reason is not None for reason in reasons):
                    break
            for row, ids, reason in zip(rows, generated, reasons):
                reason = reason or "MAX_TOKENS"
                try:
                    output = decode(ids)
                except (UnicodeError, ValueError):
                    output = None
                    reason = "INVALID_UTF8"
                correct = reason == "EOS" and output == row.answer
                records.append({**asdict(row), "prediction": output, "generated_ids": ids,
                                "stop_reason": reason, "correct": correct,
                                "answer_prefix_correct": output is not None and output.startswith(row.answer)})
    worlds: dict[str, list[bool]] = defaultdict(list)
    for row in records:
        worlds[row["world_id"]].append(row["correct"])
    n = len(records)
    return {"n_queries": n, "n_worlds": len(worlds),
            "exact_accuracy": sum(row["correct"] for row in records) / n,
            "all_queries_correct_rate": sum(all(values) for values in worlds.values()) / len(worlds),
            "valid_stop_rate": sum(row["stop_reason"] == "EOS" for row in records) / n,
            "answer_prefix_accuracy_diagnostic": sum(row["answer_prefix_correct"] for row in records) / n,
            "stop_histogram": dict(Counter(row["stop_reason"] for row in records)),
            "records": records}


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def trees_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(trees_equal(left[key], right[key]) for key in left)
    if type(left) is not type(right):
        return False
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(trees_equal(a, b) for a, b in zip(left, right))
    return left == right


def resume_check(model, optimizer, sampler, examples, *, completed, batch_size, config,
                 terminal_weight, device, checkpoint_path, manifest_hash):
    """Persist full local state and compare one uninterrupted/restored update."""
    payload = {"schema": "bramastra-local-continuation/v1", "manifest_sha256": manifest_hash,
               "config": asdict(config), "completed_updates": completed,
               "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
               "sampler_state": sampler.getstate(), "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
               "terminal_weight": terminal_weight, "batch_size": batch_size,
               "dataset_sha256": dataset_hash(examples), "schedule": "constant-lr-3e-4"}
    pending = checkpoint_path.with_suffix(".pt.pending")
    torch.save(payload, pending)
    os.replace(pending, checkpoint_path)
    saved_hash = file_hash(checkpoint_path)
    indices = sampler.sample(range(len(examples)), batch_size)
    update(model, optimizer, [examples[i] for i in indices], config=config,
           terminal_weight=terminal_weight, device=device)
    expected_parameters = cpu_tree(model.state_dict())
    expected_optimizer = cpu_tree(optimizer.state_dict())
    if file_hash(checkpoint_path) != saved_hash:
        raise RuntimeError("checkpoint changed before load")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved["manifest_sha256"] != manifest_hash or saved["dataset_sha256"] != dataset_hash(examples):
        raise RuntimeError("checkpoint identity mismatch")
    restored = BramastraModel(ModelConfig(**saved["config"])).to(device)
    restored.load_state_dict(saved["model"])
    restored_optimizer = optimizer_for(restored)
    restored_optimizer.load_state_dict(cpu_tree(saved["optimizer"]))
    restored_sampler = random.Random()
    restored_sampler.setstate(saved["sampler_state"])
    torch.set_rng_state(saved["torch_rng"])
    if device == "cuda":
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    restored_indices = restored_sampler.sample(range(len(examples)), saved["batch_size"])
    update(restored, restored_optimizer, [examples[i] for i in restored_indices], config=config,
           terminal_weight=saved["terminal_weight"], device=device)
    receipt = {"checkpoint_sha256": saved_hash, "checkpoint_update": completed,
               "probe_update": completed + 1, "sampler_equal": indices == restored_indices,
               "parameters_exact": trees_equal(expected_parameters, restored.state_dict()),
               "optimizer_exact": trees_equal(expected_optimizer, restored_optimizer.state_dict()),
               "scope": "local single-process continuation, not remote durability"}
    if not all(receipt[key] for key in ("sampler_equal", "parameters_exact", "optimizer_exact")):
        raise RuntimeError(f"continuation failed: {receipt}")
    # Evaluation refers to the registered final checkpoint, not the probe update.
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    sampler.setstate(saved["sampler_state"])
    torch.set_rng_state(saved["torch_rng"])
    if device == "cuda":
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return receipt


def decide(arms: dict) -> dict:
    """Fixed research triage: transparent code, not a learned diagnosis policy."""
    baseline, terminal = arms["without_terminal"], arms["with_terminal"]
    if baseline["completed_updates"] != terminal["completed_updates"] or any(arm["budget_stopped"] for arm in arms.values()):
        return {"verdict": "INCOMPLETE_COMPARISON", "next_action": "Repeat both arms with a common feasible update budget."}
    gain = terminal["evaluation"]["train"]["exact_accuracy"] - baseline["evaluation"]["train"]["exact_accuracy"]
    stop_gain = terminal["evaluation"]["train"]["valid_stop_rate"] - baseline["evaluation"]["train"]["valid_stop_rate"]
    learned = terminal["evaluation"]["train"]["exact_accuracy"] >= 0.99
    if learned and gain > 0 and stop_gain > 0:
        verdict = "TERMINAL_SUPERVISION_SUPPORTED_ON_DEVELOPMENT"
        next_action = "Keep terminal supervision; test contextual generalization before expanding architecture."
    elif learned:
        verdict = "INSTRUMENT_LEARNABLE_TERMINAL_EFFECT_UNRESOLVED"
        next_action = "Inspect matched stopping and answer behavior; do not infer a terminal-specific effect."
    else:
        verdict = "LEARNABILITY_NOT_ESTABLISHED"
        next_action = "Diagnose answer learning and optimization; do not scale or relax scoring."
    return {"verdict": verdict, "next_action": next_action, "train_exact_gain": gain,
            "train_stop_gain": stop_gain, "controller": "fixed transparent research rule",
            "claim_limit": "One development experiment; no AGI, autonomous discovery, or independent-transfer claim."}


def run(*, output: Path, profile: str = "smoke", seed: int = 601, steps: int = 600,
        max_seconds: float = 120.0, worlds: int = 16, eval_worlds: int = 64,
        batch_size: int = 16, device: str = "cpu", threads: int = 2) -> dict:
    for guard in ("TRIQUETRA_NO_LOCAL_MODEL_COMPUTE", "TRIQUETRA_NO_LOCAL_TRAINING"):
        if os.environ.get(guard) == "1":
            raise RuntimeError(f"local experiment refused by {guard}")
    if profile not in {"smoke", "b0"} or device not in {"cpu", "cuda"}:
        raise ValueError("supported profiles: smoke/b0; verified local backend: cpu, optional cuda")
    if steps < 1 or worlds < 1 or eval_worlds < 1 or not 1 <= batch_size <= 2 * worlds or threads < 1:
        raise ValueError("invalid experiment sizes")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("a finite positive per-arm time bound is required")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing fallback")
    if output.exists():
        raise FileExistsError("immutable experiment output already exists")
    config = ModelConfig(width=64, layers=2, heads=4, ffn=176, max_seq=32) if profile == "smoke" else ModelConfig()
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    train = build_worlds(seed=seed + 1, count=worlds)
    excluded = {row.world_id for row in train}
    development = build_worlds(seed=seed + 2, count=eval_worlds, split="development", exclude_ids=excluded)
    shifted = build_worlds(seed=seed + 3, count=eval_worlds, split="development_shift", style="alternate",
                           exclude_ids=excluded | {row.world_id for row in development})
    assert_disjoint(train, development, shifted)
    datasets = {"train": train, "development": development, "development_shift": shifted}
    # Validate context before creating output or running model compute.
    for rows in datasets.values():
        make_batch(rows, max_length=config.max_seq)
        if any(1 + len(encode(row.prompt)) + 4 > config.max_seq for row in rows):
            raise ValueError("generation cannot fit")
    source_root = Path(__file__).parent
    sources = {path.name: file_hash(path) for path in sorted(source_root.glob("*.py"))}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_root, capture_output=True, text=True)
    manifest = {"schema": "bramastra-terminal-experiment/v1", "profile": profile, "seed": seed,
                "steps_per_arm": steps, "max_training_seconds_per_arm": max_seconds,
                "batch_size": batch_size, "model": asdict(config), "parameters": parameter_count(config),
                "device": device, "threads": threads, "precision": "FP32",
                "torch": str(torch.__version__), "python": platform.python_version(),
                "git_base_revision": revision.stdout.strip() if revision.returncode == 0 else None,
                "source_files_sha256": sources,
                "dataset_sha256": {name: dataset_hash(rows) for name, rows in datasets.items()},
                "objective": "answer_mean_ce + terminal_weight * terminal_mean_ce",
                "terminal_weights": {"without_terminal": 0.0, "with_terminal": 1.0},
                "optimizer": {"name": "AdamW", "lr": 3e-4, "schedule": "constant",
                              "betas": [0.9, 0.95], "epsilon": 1e-8, "matrix_decay": 0.1,
                              "embedding_norm_decay": 0.0, "gradient_clip": 1.0},
                "matching": "same random initialization, examples, sampled batches, update count, answer-loss weight",
                "evaluation": "greedy, max four generated tokens including EOS, exact answer plus EOS",
                "phase": "development; no sealed data", "initialized_from": "random; no pretrained weights"}
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    write_json(output / "datasets.json", {name: [asdict(row) for row in rows] for name, rows in datasets.items()})
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in sources:
        (snapshot / name).write_bytes((source_root / name).read_bytes())
    manifest_hash = digest(manifest)
    arms = {}
    experiment_start = time.monotonic()
    for name, terminal_weight in manifest["terminal_weights"].items():
        torch.manual_seed(seed)
        model = BramastraModel(config).to(device)
        optimizer = optimizer_for(model)
        sampler = random.Random(seed + 4)
        before = tensor_hash(model.state_dict())
        initial = evaluate(model, train, device=device)
        start = time.monotonic()
        curve = []
        counts = {"executed_token_positions": 0, "valid_tokens": 0, "answer_supervised_tokens": 0,
                  "terminal_supervised_tokens": 0}
        completed = 0
        for step in range(1, steps + 1):
            if time.monotonic() - start >= max_seconds:
                break
            indices = sampler.sample(range(len(train)), batch_size)
            rows = [train[index] for index in indices]
            loss, gradient = update(model, optimizer, rows, config=config, terminal_weight=terminal_weight, device=device)
            completed = step
            counts["executed_token_positions"] += batch_size * config.max_seq
            counts["valid_tokens"] += sum(2 + len(encode(row.prompt)) + len(encode(row.answer)) for row in rows)
            counts["answer_supervised_tokens"] += sum(len(encode(row.answer)) for row in rows)
            counts["terminal_supervised_tokens"] += int(terminal_weight > 0) * batch_size
            if step == 1 or step % 100 == 0 or step == steps:
                curve.append({"update": step, "loss": loss, "gradient_norm": gradient})
                print(json.dumps({"arm": name, **curve[-1]}), flush=True)
        training_seconds = time.monotonic() - start
        if completed == 0:
            raise RuntimeError("budget expired before a single update")
        after = tensor_hash(model.state_dict())
        if before == after:
            raise RuntimeError("parameters did not change")
        moment_nonzero = all(bool(state["exp_avg"].abs().sum() > 0) for state in optimizer.state.values())
        if not moment_nonzero:
            raise RuntimeError("optimizer has an inactive parameter state")
        continuation = resume_check(model, optimizer, sampler, train, completed=completed,
                                    batch_size=batch_size, config=config, terminal_weight=terminal_weight,
                                    device=device, checkpoint_path=output / f"{name}.pt", manifest_hash=manifest_hash)
        evaluation = {part: evaluate(model, rows, device=device) for part, rows in datasets.items()}
        arm = {"completed_updates": completed, "budget_stopped": completed != steps,
               "initial_parameter_sha256": before, "final_parameter_sha256": after,
               "optimizer_moments_nonzero": moment_nonzero, "training_seconds": training_seconds,
               "counts": counts, "curve": curve, "initial_train": initial,
               "evaluation": evaluation, "continuation": continuation}
        arms[name] = arm
        write_json(output / f"{name}.json", arm)
        del model, optimizer
    if arms["without_terminal"]["initial_parameter_sha256"] != arms["with_terminal"]["initial_parameter_sha256"]:
        raise RuntimeError("arms did not start from identical parameters")
    result = {"schema": "bramastra-terminal-result/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": manifest_hash, "elapsed_seconds": time.monotonic() - experiment_start,
              "decision": decide(arms), "arms": {name: {"completed_updates": arm["completed_updates"],
                  "training_seconds": arm["training_seconds"], "counts": arm["counts"],
                  "evaluation": {part: {key: value for key, value in report.items() if key != "records"}
                                 for part, report in arm["evaluation"].items()},
                  "continuation": arm["continuation"]} for name, arm in arms.items()}}
    write_json(output / "result.json", result)
    print(json.dumps(result["decision"]), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "b0"), default="smoke")
    parser.add_argument("--seed", type=int, default=601)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--eval-worlds", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(output=args.output, profile=args.profile, seed=args.seed, steps=args.steps,
        max_seconds=args.max_seconds, worlds=args.worlds, eval_worlds=args.eval_worlds,
        batch_size=args.batch_size, device=args.device, threads=args.threads)


if __name__ == "__main__":
    main()
