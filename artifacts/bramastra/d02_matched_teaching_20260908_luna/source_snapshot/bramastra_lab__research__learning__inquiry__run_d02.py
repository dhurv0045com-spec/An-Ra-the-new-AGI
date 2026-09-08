"""Bounded, paired D02 execution for matched one-step/depth-two teaching."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

from bramastra_lab.discovery.evaluation import evaluate
from bramastra_lab.discovery.learner import Investigator, LearnerConfig
from bramastra_lab.discovery.statistics import summarize
from bramastra_lab.discovery.training import TrainConfig, Trainer, fingerprint
from bramastra_lab.discovery.worlds import inputs, make_worlds

from .teaching import build_paired_teaching_data, stratified_semantic_split


class RunTimeout(RuntimeError):
    """Frozen campaign deadline reached; partial evidence is retained."""


PRIMARY = {
    "bits": 4, "episodes": 256, "budget": 2, "width": 32, "steps": 500,
    "batch_size": 32, "learning_rate": 0.001, "policy_weight": 0.25,
    "evaluation_seed": 17, "targets_per_world": 4, "bootstrap_samples": 1000,
    "threads": 2, "seeds": (801, 802), "query_cost": 0.0,
}
_LOCAL_SOURCES = (
    "bramastra_lab/research/learning/inquiry/teaching.py",
    "bramastra_lab/research/learning/inquiry/__init__.py",
    "bramastra_lab/research/learning/inquiry/run_d02.py",
    "bramastra_lab/discovery/__init__.py",
    "bramastra_lab/discovery/curriculum.py",
    "bramastra_lab/discovery/evaluation.py",
    "bramastra_lab/discovery/learner.py",
    "bramastra_lab/discovery/statistics.py",
    "bramastra_lab/discovery/training.py",
    "bramastra_lab/discovery/worlds.py",
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(split: dict[str, list]) -> dict:
    values = {name: [{"world_id": w.world_id, "family": w.family,
                      "bits": w.bits, "table_sha256": hashlib.sha256(bytes(w.table)).hexdigest()}
                     for w in worlds] for name, worlds in split.items()}
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return {"splits": values, "sha256": hashlib.sha256(encoded).hexdigest()}


def _check_paired_data(paired, bits: int) -> None:
    names = ("observations", "lengths", "targets", "labels", "legal")
    for name in names:
        if not torch.equal(getattr(paired.one_step, name), getattr(paired.depth_two, name)):
            raise ValueError(f"paired data mismatch in {name}")
    if torch.equal(paired.one_step.gains, paired.depth_two.gains):
        raise ValueError("teacher intervention has no changed gain tensor")
    if not paired.diagnostics["zero_remaining_both_zero"] or not paired.diagnostics["one_remaining_labels_equal"]:
        raise ValueError("teacher horizon controls failed")
    for data in (paired.one_step, paired.depth_two):
        if bool((~data.legal.any(dim=1)).any()):
            raise ValueError("training state has no legal action")
        candidates = torch.tensor(inputs(bits), dtype=torch.float32)
        target_match = (data.targets[:, None, :] == candidates[None, :, :]).all(dim=-1)
        if not bool(target_match.any(dim=1).all()):
            raise ValueError("target is not a known candidate")
        if bool(data.legal[target_match].any()):
            raise ValueError("scored target is legal in training mask")


def _paired_comparison(left_rows: list[dict], right_rows: list[dict], *, bootstrap: int, seed: int) -> dict:
    key = lambda row: (row["world_id"], row["target"], row["policy"], row["budget"])
    def index(rows):
        result = {}
        for row in rows:
            item_key = key(row)
            if item_key in result:
                raise ValueError("duplicate evaluation key")
            result[item_key] = row
        return result
    left = index(left_rows)
    right = index(right_rows)
    if set(left) != set(right):
        raise ValueError("evaluation keys differ between teaching arms")
    for item_key in left:
        if (left[item_key]["label"], left[item_key]["family"]) != (right[item_key]["label"], right[item_key]["family"]):
            raise ValueError("evaluation labels or families differ between teaching arms")
    primary = [k for k in left if k[2] == "learned" and k[3] == 2]
    by_world: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for world, target, policy, budget in sorted(primary):
        a, b = left[(world, target, policy, budget)], right[(world, target, policy, budget)]
        by_world[world].append((float(b["correct"]) - float(a["correct"]),
                                (b["probability"] - b["label"]) ** 2 - (a["probability"] - a["label"]) ** 2,
                                float(a["correct"]), float(b["correct"])))
    deltas = [sum(v[0] for v in values) / len(values) for values in by_world.values()]
    briers = [sum(v[1] for v in values) / len(values) for values in by_world.values()]
    rng = np.random.default_rng(seed)
    boot = [float(np.mean(rng.choice(deltas, len(deltas), replace=True))) for _ in range(bootstrap)] if deltas else []
    family = defaultdict(list)
    for world, target, policy, budget in sorted(primary):
        family[next(row["family"] for row in left_rows if row["world_id"] == world)].append(
            float(right[(world, target, policy, budget)]["correct"]) - float(left[(world, target, policy, budget)]["correct"]))
    return {"budget": 2, "policy": "learned", "contrast": "depth_two_minus_one_step",
            "nworlds": len(deltas), "accuracy_delta": float(np.mean(deltas)) if deltas else None,
            "brier_delta": float(np.mean(briers)) if briers else None,
            "accuracy_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))] if boot else None,
            "family_accuracy_delta": {k: float(np.mean(v)) for k, v in sorted(family.items())}}


def _train(data, seed: int, output: Path, config: dict, initial_state: dict, deadline: float) -> tuple[Investigator, dict]:
    model = Investigator(LearnerConfig(bits=config["bits"], width=config["width"]))
    model.load_state_dict(initial_state)
    trainer = Trainer(model, data, TrainConfig(steps=config["steps"], batch_size=config["batch_size"],
                                                learning_rate=config["learning_rate"],
                                                policy_weight=config["policy_weight"], seed=seed))
    before = fingerprint(model)
    sampler_initial = hashlib.sha256(bytes(trainer.sampler.get_state().tolist())).hexdigest()
    history = []
    started = time.perf_counter()
    for _ in range(config["steps"]):
        if time.perf_counter() >= deadline:
            report = {"status": "TIMEOUT", "seed": seed, "initial_sha256": before,
                      "sampler_initial_sha256": sampler_initial, "completed_steps": len(history),
                      "seconds": time.perf_counter() - started, "history": history, "data": data.manifest}
            _write(output / "training.json", report)
            raise RunTimeout(f"deadline reached during training seed {seed}")
        history.append(trainer.update())
    elapsed = time.perf_counter() - started
    report = {"status": "COMPLETE", "seed": seed, "initial_sha256": before,
              "sampler_initial_sha256": sampler_initial, "final_sha256": fingerprint(model),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "seconds": elapsed, "history": history, "data": data.manifest}
    _write(output / "training.json", report)
    return model, report


def run(output: str | Path, *, config: dict | None = None, deadline_seconds: float = 300.0) -> dict:
    """Execute both frozen D02 seeds into a new evidence directory."""
    cfg = {**PRIMARY, **(config or {})}
    output = Path(output)
    if output.exists():
        raise ValueError("output exists; choose a new unique d02 directory")
    started = time.perf_counter()
    deadline = started + deadline_seconds
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True)
    source_hashes = {}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for relative in _LOCAL_SOURCES:
        source = Path(relative)
        target = snapshot / relative.replace("/", "__")
        shutil.copyfile(source, target)
        source_hashes[relative] = _sha256(source)
    worlds = make_worlds(cfg["bits"])
    split = stratified_semantic_split(worlds)
    inventory = _inventory(split)
    _write(output / "world_inventory.json", inventory)
    manifest = {"schema": "bramastra-d02/v1", "status": "RUNNING", "configuration": cfg,
                "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                            "numpy": np.__version__, "device": "cpu", "threads": cfg["threads"]},
                "source_sha256_before": source_hashes, "inventory_sha256": inventory["sha256"],
                "seeds": list(cfg["seeds"]), "limitations": [
                    "uniform legal histories test off-policy teacher imitation coverage",
                    "development synthetic Boolean rules; not AGI evidence"]}
    _write(output / "manifest.json", manifest)
    results = {}
    try:
      for seed in cfg["seeds"]:
        if time.perf_counter() >= deadline:
            manifest["status"] = "TIMEOUT"
            manifest["timeout_after_seed"] = seed
            break
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir()
        generation_start = time.perf_counter()
        paired = build_paired_teaching_data(split["train"], episodes=cfg["episodes"], budget=cfg["budget"],
                                             seed=seed, query_cost=cfg["query_cost"])
        _check_paired_data(paired, cfg["bits"])
        generation_seconds = time.perf_counter() - generation_start
        torch.manual_seed(seed)
        initial_model = Investigator(LearnerConfig(bits=cfg["bits"], width=cfg["width"]))
        initial_state = {k: value.detach().clone() for k, value in initial_model.state_dict().items()}
        arm_results = {}
        for name, data in (("one_step", paired.one_step), ("depth_two", paired.depth_two)):
            arm_dir = seed_dir / name
            arm_dir.mkdir()
            model, training = _train(data, seed, arm_dir, cfg, initial_state, deadline)
            eval_start = time.perf_counter()
            rows = evaluate(model, split["dev"], policies=("learned", "random", "coverage", "no_memory"),
                             budgets=(0, 1, 2), targets_per_world=cfg["targets_per_world"],
                             seed=cfg["evaluation_seed"])
            _write(arm_dir / "evaluation_rows.json", rows)
            summary = summarize(rows, bootstrap_samples=cfg["bootstrap_samples"], seed=seed)
            _write(arm_dir / "evaluation.json", summary)
            arm_results[name] = {"training": training, "evaluation_seconds": time.perf_counter() - eval_start,
                                 "summary": summary, "rows": rows, "tensor_sha256": data.manifest["tensor_sha256"]}
        comparison = _paired_comparison(arm_results["one_step"]["rows"], arm_results["depth_two"]["rows"],
                                        bootstrap=cfg["bootstrap_samples"], seed=seed)
        _write(seed_dir / "comparison.json", comparison)
        if arm_results["one_step"]["training"]["initial_sha256"] != arm_results["depth_two"]["training"]["initial_sha256"]:
            raise ValueError("paired arm initialization fingerprints differ")
        if arm_results["one_step"]["training"]["sampler_initial_sha256"] != arm_results["depth_two"]["training"]["sampler_initial_sha256"]:
            raise ValueError("paired arm sampler fingerprints differ")
        results[str(seed)] = {"generation_seconds": generation_seconds, "arms": arm_results,
                              "comparison": comparison, "initial_fingerprint": fingerprint(initial_model)}
    except RunTimeout as exc:
        manifest["status"], manifest["failure"] = "TIMEOUT", {"type": type(exc).__name__, "message": str(exc)}
    except Exception as exc:
        manifest["status"], manifest["failure"] = "FAILED", {"type": type(exc).__name__, "message": str(exc)}
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["results"] = {seed: {"comparison": value["comparison"], "generation_seconds": value["generation_seconds"]}
                                for seed, value in results.items()}
        after = {relative: _sha256(Path(relative)) for relative in _LOCAL_SOURCES}
        manifest["source_sha256_after"] = after
        manifest["source_integrity"] = source_hashes == after
        if manifest["status"] == "RUNNING":
            manifest["status"] = "COMPLETE" if len(results) == len(cfg["seeds"]) and manifest["source_integrity"] else "FAILED"
        _write(output / "manifest.json", manifest)
        _write(output / "results.json", results)
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.output)
