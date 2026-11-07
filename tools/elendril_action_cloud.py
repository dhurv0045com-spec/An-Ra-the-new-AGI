"""Owner-operated action-learning campaigns; import never discovers/starts devices.

Native CUDA PyTorch is qualified rather than overwritten. All workers are pinned
to one physical UUID and address it as cuda:0. No CPU training fallback exists.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from uuid import uuid4

from tools.elendril_cloud import (GIB, admit_devices, blob, check_source, check_torch,
    command, device_env, fingerprint, inventory, read_checkpoint, safe_id, sha_id)


ARMS = ("ce", "action-weighted", "uncertainty-first")
SCHEMA = "elendril-action-cloud-v1"
OFFICIAL_REFERENCES = (
    "https://research.google.com/colaboratory/faq.html",
    "https://www.kaggle.com/docs/notebooks",
    "https://github.com/Kaggle/docker-python",
    "https://github.com/Kaggle/kagglehub",
    "https://docs.pytorch.org/docs/2.14/notes/cuda.html",
    "https://docs.pytorch.org/docs/2.14/notes/serialization.html",
)


@dataclass(frozen=True)
class ActionPlan:
    platform: str
    seeds: tuple[int, ...]
    updates: int
    device_count: int
    train_seconds: int
    export_seconds: int
    fixture_seed: int
    evaluation_split: str

    def jobs(self):
        return tuple((arm, seed) for seed in self.seeds for arm in ARMS)


PLANS = {
    "colab": ActionPlan("colab", (7, 19), 448, 1, 6300, 900, 1729, "validation"),
    "kaggle": ActionPlan("kaggle", (7, 19, 41, 61, 79), 896, 2, 18900, 1800, 2718, "test"),
}


def session_identity() -> str:
    boot = Path("/proc/sys/kernel/random/boot_id")
    if not boot.is_file():
        raise ValueError("owner cloud qualification requires a Linux VM session identity")
    return sha256(blob([socket.gethostname(), boot.read_text().strip()])).hexdigest()


def write_json(path: Path, value, *, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise ValueError(f"refusing to overwrite evidence: {path.name}")
    temporary = path.with_name("." + path.name + "-" + uuid4().hex)
    with temporary.open("xb") as handle:
        handle.write(blob(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if json.loads(path.read_bytes()) != json.loads(blob(value)):
        raise ValueError("evidence readback failed")


@lru_cache(maxsize=2)
def fixture(platform):
    from anra_v5.elendril.action_learning import action_fixture, assert_disjoint
    plan = PLANS[platform]
    train = action_fixture("train", worlds=160, seed=plan.fixture_seed)
    evaluation = action_fixture(plan.evaluation_split, worlds=80, seed=plan.fixture_seed)
    assert_disjoint(train, evaluation)
    if len(train.examples) != 224:
        raise ValueError("registered full-epoch sample count drifted")
    return train, evaluation


def engine(platform, arm, seed, source, root, lineage, *, resume=False, tiny=False):
    from anra_v5.elendril.action_learning import action_fixture
    from anra_v5.elendril.foundation_training import (FoundationTrainer, Recipe, TokenSample,
                                                    legacy_tokenizer_manifest)
    from anra_v5.elendril.training import PROFILES, configure_device
    if arm not in ARMS or seed not in PLANS[platform].seeds:
        raise ValueError("unregistered arm or seed")
    data = action_fixture("train", worlds=5, seed=1729) if tiny else fixture(platform)[0]
    samples = [TokenSample(row.task_id, row.tokens, row.eligible, row.world_id, row.kind)
               for row in data.examples]
    recipe = Recipe(seed=seed, total_updates=64 if tiny else PLANS[platform].updates,
                    batch_size=len(samples) if tiny else 16, micro_batch_size=1 if tiny else 4,
                    objective="action-weighted" if arm == "action-weighted" else "ce",
                    sampling="uncertainty-first" if arm == "uncertainty-first" else "uniform",
                    lr=.003, precision="fp32")
    configure_device("cuda:0")
    return FoundationTrainer(samples, spec=PROFILES["expanded"], recipe=recipe,
        source_commit=source, root=root, lineage=lineage, data_provenance=data.manifest,
        tokenizer_manifest=legacy_tokenizer_manifest(), device="cuda:0", resume=resume)


def model_policy(trainer):
    from anra_v5.elendril.action_learning import DecodedAction, NeuralActionPolicy
    from anra_v5.elendril.codec import BOS, encode
    def decode(prompt, remaining):
        ids = (BOS, *encode(prompt))
        if 2 * len(ids) + 1 > remaining or len(ids) + 1 > trainer.spec.context_length:
            return None
        row = trainer.backend.generate((ids,), max_new_tokens=2)[0]
        return DecodedAction(row.tokens, row.evaluated_positions, row.forward_calls,
                             math.prod(row.token_probabilities))
    return NeuralActionPolicy(decode, trainer.parent or trainer.backend.binding.identity)


def evaluate(trainer, data):
    from anra_v5.elendril.action_learning import evaluate_policy
    return evaluate_policy(data, model_policy(trainer))


def controls(data):
    from anra_v5.elendril.action_learning import ControlPolicy, evaluate_policy
    return {mode: evaluate_policy(data, "oracle" if mode == "oracle" else ControlPolicy(mode))
            for mode in ("fixed", "oracle", "no-query", "always-query", "random", "query-blind", "recency")}


def worker_environment(repo, source):
    check_source(repo, source)
    rows = inventory()
    admit_devices(rows, 1)
    runtime = check_torch(repo, rows)[0]
    import torch
    if torch.cuda.device_count() != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != rows[0].uuid:
        raise ValueError("worker must be isolated by one full physical UUID before importing torch")
    return {"physical": asdict(rows[0]), "runtime": runtime["runtime"], "session": session_identity()}


def restore_mirror(platform, arm, seed, source, root, lineage, mirror, checkpoint=None):
    from v5_training.persistent_store import materialize_local
    fresh = engine(platform, arm, seed, source, root, lineage)
    materialize_local(mirror, fresh.store, checkpoint_sha256=checkpoint)
    return engine(platform, arm, seed, source, root, lineage, resume=True)


def compare_payloads(reference, actual, *, exact):
    """All tensor/state values and counters; different GPU restore is numerical."""
    import torch
    from anra_v5.elendril.training import _load
    def compare(left, right, trail="state"):
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(f"recovery tensor shape/dtype mismatch: {trail}")
            if exact or trail.endswith("/step"):
                if not torch.equal(left.cpu(), right.cpu()):
                    raise ValueError(f"recovery tensor differs: {trail}")
            else:
                torch.testing.assert_close(left.cpu(), right.cpu(), rtol=1e-5, atol=1e-6, equal_nan=False)
        elif isinstance(left, dict):
            if not isinstance(right, dict) or set(left) != set(right):
                raise ValueError(f"recovery keys mismatch: {trail}")
            for key in left:
                compare(left[key], right[key], f"{trail}/{key}")
        elif isinstance(left, (tuple, list)):
            if type(left) is not type(right) or len(left) != len(right):
                raise ValueError(f"recovery sequence mismatch: {trail}")
            for i, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f"{trail}/{i}")
        elif type(left) is not type(right) or left != right:
            raise ValueError(f"recovery scalar mismatch: {trail}")
    for name in ("model.bin", "optimizer.bin"):
        compare(_load(reference[name]), _load(actual[name]), name)
    for name in ("rng.bin", "scheduler.json", "cursor.json", "ledger.json", "training_state.json"):
        if reference[name] != actual[name]:
            raise ValueError(f"recovery state differs: {name}")


def _live_payloads(trainer):
    from anra_v5.elendril.training import _dump
    from anra_v5.elendril.codec import canonical
    return {"model.bin": _dump(trainer.model.state_dict()),
            "optimizer.bin": _dump(trainer.optimizer.state_dict()), "rng.bin": trainer.rng_bytes(),
            "scheduler.json": canonical({"metadata": trainer.metadata, "scaler": trainer.scaler.state_dict()}),
            "cursor.json": canonical(asdict(trainer.state.cursor)),
            "ledger.json": canonical(dict(trainer.state.tokens_by_source)),
            "training_state.json": canonical(trainer.state.canonical())}


def worker_canary(args, environment):
    from anra_v5.elendril.action_learning import action_fixture
    from v5_training.persistent_store import mirror_checkpoint
    import torch
    started = time.monotonic()
    output, mirror = args.output, args.durable
    output.mkdir(parents=True, exist_ok=False)
    # Formation control is owner-enabled CUDA work, never part of local tests.
    positive = engine(args.platform, "ce", args.seed, args.source, output / "positive", "formation", tiny=True)
    tiny = action_fixture("train", worlds=5, seed=1729)
    history = [positive.step() for _ in range(64)]
    positive_checkpoint = positive.save()
    mirror_checkpoint(positive.store, mirror, checkpoint_sha256=positive_checkpoint)
    learned = evaluate(positive, tiny)
    write_json(mirror / ("formation-" + uuid4().hex + ".json"),
               {"updates": history, "evaluation": learned, "checkpoint": positive_checkpoint})
    if learned["totals"]["all"]["correct"] != 5 or history[-1]["loss"] >= .2 * history[0]["loss"]:
        raise ValueError("NO-GO: 64-update action formation control did not overfit five worlds")
    results = []
    for arm in ARMS:
        lineage = safe_id(f"canary-{args.platform}-{arm}-{args.seed}")
        trainer = engine(args.platform, arm, args.seed, args.source, output / arm, lineage)
        rows = []
        torch.cuda.synchronize()
        before = time.monotonic()
        for _ in range(7):
            rows.append(trainer.step())
        head = trainer.save()
        mirror_checkpoint(trainer.store, mirror, checkpoint_sha256=head)
        for _ in range(7):
            rows.append(trainer.step())
        torch.cuda.synchronize()
        train_seconds = time.monotonic() - before
        expected_live = _live_payloads(trainer)
        expected_checkpoint = trainer.save()
        mirror_checkpoint(trainer.store, mirror, checkpoint_sha256=expected_checkpoint)
        resumed = restore_mirror(args.platform, arm, args.seed, args.source,
            output / (arm + "-restart"), lineage, mirror, head)
        for _ in range(7):
            resumed.step()
        compare_payloads(expected_live, _live_payloads(resumed), exact=True)
        _, expected = trainer.store.restore(expected_checkpoint)
        challenge = {"schema": SCHEMA + "/challenge", "source": args.source,
            "platform": args.platform, "arm": arm, "seed": args.seed, "lineage": lineage,
            "environment": environment, "checkpoint": head, "expected_checkpoint": expected_checkpoint,
            "start_update": 7, "end_update": 14, "mirror": str(mirror.resolve()),
            "runtime_training": trainer.runtime}
        challenge_path = mirror / (arm + "-challenge-" + uuid4().hex + ".json")
        write_json(challenge_path, challenge)
        copy_root = output / (arm + "-fresh-process")
        proof_path = output / (arm + "-fresh-process.json")
        command([sys.executable, "-m", "tools.elendril_action_cloud", "recover", "--owner-enabled",
                 "--source", args.source, "--platform", args.platform, "--output", str(copy_root),
                 "--receipt", str(proof_path), "--challenge", str(challenge_path),
                 "--mirror", str(mirror)], cwd=args.repo, env=device_env(inventory()[0]), timeout=240)
        before = time.monotonic()
        evaluation = evaluate(trainer, fixture(args.platform)[1])
        torch.cuda.synchronize()
        evaluation_seconds = time.monotonic() - before
        results.append({"arm": arm, "updates": rows, "train_seconds": train_seconds,
            "evaluation_seconds": evaluation_seconds, "challenge": str(challenge_path),
            "challenge_sha256": fingerprint(challenge_path), "evaluation": evaluation,
            "checkpoint": expected_checkpoint, "same_device_exact": True,
            "fresh_process": json.loads(proof_path.read_bytes())})
    summary = {"schema": SCHEMA + "/canary", "source": args.source, "platform": args.platform,
        "environment": environment, "results": results, "formation": learned,
        "wall_seconds": time.monotonic() - started, "status": "PASS",
        "off_session_persistence": "UNVERIFIED_OWNER_REATTACHMENT_REQUIRED"}
    write_json(args.receipt, summary)
    return summary


def worker_recover(args, environment):
    challenge = json.loads(args.challenge.read_bytes())
    if challenge["schema"] != SCHEMA + "/challenge" or challenge["source"] != args.source:
        raise ValueError("recovery source/challenge schema mismatch")
    previous = challenge["environment"]
    if (environment["runtime"] != previous["runtime"]
            or environment["physical"].get("driver") != previous["physical"].get("driver")):
        raise ValueError("exact runtime required for this recovery recipe")
    cross_session = environment["session"] != previous["session"]
    if args.off_session_ack and not cross_session:
        raise ValueError("fresh-session qualification requires a different VM identity")
    exact = environment["physical"]["uuid"] == previous["physical"]["uuid"]
    trainer = restore_mirror(challenge["platform"], challenge["arm"], challenge["seed"],
        args.source, args.output, challenge["lineage"], args.mirror, challenge["checkpoint"])
    for _ in range(challenge["end_update"] - challenge["start_update"]):
        trainer.step()
    # Decode expected committed generation independently from the copied mirror.
    from v5_training.checkpoint import CheckpointStore
    from v5_training.persistent_store import materialize_local
    reference = CheckpointStore(args.output / "reference", challenge["lineage"])
    materialize_local(args.mirror, reference, checkpoint_sha256=challenge["expected_checkpoint"])
    _, expected = reference.restore(challenge["expected_checkpoint"])
    compare_payloads(expected, _live_payloads(trainer), exact=exact)
    report = {"schema": SCHEMA + "/recovery", "source": args.source,
        "challenge_sha256": fingerprint(args.challenge), "challenge_document": challenge,
        "original_environment": previous,
        "environment": environment, "classification": "bitwise-identical" if exact else "numerical-equivalent",
        "rtol": 0 if exact else 1e-5, "atol": 0 if exact else 1e-6,
        "restored_update": trainer.state.global_update, "checkpoint": challenge["checkpoint"],
        "expected_checkpoint": challenge["expected_checkpoint"], "off_session_owner_ack": args.off_session_ack,
        "fresh_session": cross_session, "status": "PASS"}
    write_json(args.receipt, report)
    return report


def worker_train(args, environment):
    job_started, cpu_started = time.monotonic(), time.process_time()
    from v5_training.persistent_store import mirror_checkpoint, read_mirror_pointer
    plan = PLANS[args.platform]
    lineage = safe_id(f"{args.run_id}-{args.arm}-{args.seed}")
    head = read_mirror_pointer(args.durable, lineage_id=lineage)
    args.output.mkdir(parents=True, exist_ok=True)
    if head:
        trainer = restore_mirror(args.platform, args.arm, args.seed, args.source,
            args.output / ("restore-" + uuid4().hex), lineage, args.durable, head)
    else:
        trainer = engine(args.platform, args.arm, args.seed, args.source,
            args.output / "checkpoints", lineage)
    start_update = trainer.state.global_update
    started, rows = time.monotonic(), []
    args.durable.mkdir(parents=True, exist_ok=True)
    log = args.durable / ("updates-" + uuid4().hex + ".jsonl")
    try:
        with log.open("xb") as handle:
            while not trainer.state.complete:
                row = trainer.step()
                rows.append(row)
                handle.write(blob(row)); handle.flush(); os.fsync(handle.fileno())
                if trainer.state.global_update % 56 == 0 or trainer.state.complete:
                    checkpoint = trainer.save()
                    mirror_checkpoint(trainer.store, args.durable, checkpoint_sha256=checkpoint)
                if time.monotonic() - started > args.seconds:
                    raise TimeoutError("training cap reached; latest mirrored generation is recoverable")
        training_seconds = time.monotonic() - started
        evaluation_started = time.monotonic()
        evaluation = evaluate(trainer, fixture(args.platform)[1])
        evaluation_seconds = time.monotonic() - evaluation_started
        baseline_reports = controls(fixture(args.platform)[1])
        report = {"schema": SCHEMA + "/result", "source": args.source, "platform": args.platform,
            "arm": args.arm, "seed": args.seed, "lineage": lineage, "environment": environment,
            "recipe": asdict(trainer.recipe), "model_sha256": trainer.spec.sha256(),
            "tokenizer_sha256": trainer.codec_sha, "data_sha256": trainer.data_sha,
            "training_fixture": fixture(args.platform)[0].manifest,
            "checkpoint": trainer.parent, "global_update": trainer.state.global_update,
            "start_update": start_update, "wall_seconds": time.monotonic() - job_started,
            "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
            "process_cpu_seconds": time.process_time() - cpu_started,
            "measurement_scope": "worker initialization through model/control evaluation; CUDA lane reserved, not GPU utilization",
            "segment_input_positions": sum(row["input_positions"] for row in rows),
            "segment_supervised_tokens": sum(row["supervised_tokens"] for row in rows),
            "lifetime_supervised_tokens": trainer.state.cumulative_tokens,
            "evaluation_model_positions": evaluation["totals"]["all"]["model_positions"],
            "evaluation_model_calls": evaluation["totals"]["all"]["model_calls"],
            "updates": rows, "evaluation": evaluation, "controls": baseline_reports,
            "peak_cuda_bytes": __import__("torch").cuda.max_memory_allocated(), "status": "COMPLETE"}
        write_json(args.receipt, report)
        return report
    except BaseException as exc:
        write_json(args.durable / ("failure-" + uuid4().hex + ".json"), {
            "schema": SCHEMA + "/failure", "source": args.source, "arm": args.arm, "seed": args.seed,
            "exception": type(exc).__name__, "message": str(exc), "updates": rows,
            "checkpoint": trainer.parent, "global_update": trainer.state.global_update,
            "wall_seconds": time.monotonic() - started, "status": "INCOMPLETE"})
        raise


def recovery_gate(paths, *, source, environments):
    records = [json.loads(Path(path).read_bytes()) for path in paths]
    for environment in environments:
        matches = [row for row in records if row.get("schema") == SCHEMA + "/recovery"
                   and row.get("source") == source and row.get("status") == "PASS"
                   and row.get("fresh_session") is True and row.get("off_session_owner_ack") is True
                   and row.get("environment", {}).get("physical", {}).get("uuid") == environment["physical"]["uuid"]
                   and row.get("environment", {}).get("physical", {}).get("driver") == environment["physical"].get("driver")
                   and row.get("environment", {}).get("runtime") == environment["runtime"]
                   and row.get("environment", {}).get("session") == session_identity()]
        if not matches:
            raise ValueError("NO-GO: every current GPU lane needs source/runtime-bound owner-reattached fresh-session recovery")
        for row in matches:
            sha_id(row["challenge_sha256"])
            challenge = row.get("challenge_document", {})
            exact = row["original_environment"]["physical"]["uuid"] == row["environment"]["physical"]["uuid"]
            if (sha256(blob(challenge)).hexdigest() != row["challenge_sha256"]
                    or challenge.get("schema") != SCHEMA + "/challenge" or challenge.get("source") != source
                    or challenge.get("environment") != row["original_environment"]
                    or challenge.get("checkpoint") != row["checkpoint"]
                    or challenge.get("expected_checkpoint") != row["expected_checkpoint"]
                    or challenge.get("end_update") != row["restored_update"]
                    or row["original_environment"]["session"] == row["environment"]["session"]
                    or row["original_environment"]["runtime"] != row["environment"]["runtime"]
                    or row["original_environment"]["physical"].get("driver") != row["environment"]["physical"].get("driver")
                    or row["classification"] != ("bitwise-identical" if exact else "numerical-equivalent")
                    or row["rtol"] != (0 if exact else 1e-5) or row["atol"] != (0 if exact else 1e-6)):
                raise ValueError("unqualified recovery classification")
    return records


def forecast(plan, canaries, *, elapsed=0.):
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed time must be finite and nonnegative")
    if len(canaries) != plan.device_count:
        raise ValueError("each lane needs its measured representative canary")
    lane_seconds = [0.] * plan.device_count
    for index, (arm, _) in enumerate(plan.jobs()):
        lane = index % plan.device_count
        rows = [row for row in canaries[lane]["results"] if row["arm"] == arm]
        if (len(rows) != 1 or any(type(rows[0][key]) not in (int,float)
                or not math.isfinite(rows[0][key]) or rows[0][key] <= 0
                for key in ("train_seconds", "evaluation_seconds"))):
            raise ValueError("missing or invalid representative arm measurement")
        row = rows[0]
        # Includes repeated publication overhead and a conservative twofold eval margin.
        lane_seconds[lane] += 1.5 * row["train_seconds"] * plan.updates / 14 + 2 * row["evaluation_seconds"]
    seconds = max(lane_seconds)
    if elapsed + seconds > plan.train_seconds:
        raise ValueError("NO-GO: full matched plan cannot fit measured remaining training budget")
    return {"predicted_seconds": seconds, "elapsed_seconds": elapsed,
            "training_cap": plan.train_seconds, "export_reserve": plan.export_seconds,
            "basis": "all-arm 14-update real CUDA canaries; 1.5x training and 2x evaluation margin"}


def canary_gate(canaries, *, source, platform, environments):
    from anra_v5.elendril.action_learning import action_fixture
    if len(canaries) != len(environments):
        raise ValueError("one current canary required per lane")
    for canary, environment in zip(canaries, environments):
        actual = canary.get("environment", {})
        if (canary.get("schema") != SCHEMA + "/canary" or canary.get("status") != "PASS"
                or canary.get("source") != source or canary.get("platform") != platform
                or actual.get("physical", {}).get("uuid") != environment["physical"]["uuid"]
                or actual.get("physical", {}).get("driver") != environment["physical"].get("driver")
                or actual.get("runtime") != environment["runtime"] or actual.get("session") != session_identity()
                or [row.get("arm") for row in canary.get("results", [])] != list(ARMS)):
            raise ValueError("NO-GO: canary source/platform/runtime/device/session mismatch")
        formation = rescore_evaluation(action_fixture("train", worlds=5, seed=1729), canary["formation"])
        if formation["all"]["correct"] != 5:
            raise ValueError("NO-GO: neural formation control failed")
        for row in canary["results"]:
            if ([step.get("update") for step in row.get("updates", [])] != list(range(1,15))
                    or row.get("same_device_exact") is not True
                    or row.get("fresh_process", {}).get("status") != "PASS"):
                raise ValueError("NO-GO: representative canary missing updates/exact/fresh-process proof")
            rescore_evaluation(fixture(platform)[1], row["evaluation"])


def screen_gate(path, source, *, owner_reviewed):
    if not owner_reviewed:
        raise ValueError("Kaggle requires explicit owner review of Colab screen")
    summary = json.loads(Path(path).read_bytes())
    if (summary.get("schema") != SCHEMA + "/campaign" or summary.get("platform") != "colab"
            or summary.get("source") != source or summary.get("status") != "COMPLETE"
            or len(summary.get("results", [])) != 6):
        raise ValueError("complete pinned-source Colab screening bundle required")
    seen, scores = set(), {}
    for item in summary["results"]:
        result_path = Path(path).parent / item["file"]
        if result_path.resolve().parent != Path(path).resolve().parent or fingerprint(result_path) != item["sha256"]:
            raise ValueError("unsafe or corrupt Colab result reference")
        result = json.loads(result_path.read_bytes())
        job = (result["arm"], result["seed"])
        if (job not in PLANS["colab"].jobs() or job in seen or result["source"] != source
                or result["status"] != "COMPLETE" or result["global_update"] != 448
                or result["evaluation"]["dataset_sha256"] != fixture("colab")[1].identity):
            raise ValueError("screen arm/source/data/completion mismatch")
        verify_result_checkpoint(Path(path).parent, result)
        verified = rescore_evaluation(fixture("colab")[1], result["evaluation"])
        fixed = controls(fixture("colab")[1])["fixed"]["totals"]["all"]
        total = verified["all"]
        formed = (total["correct"] / total["episodes"] >= .9 and total["invalid"] == 0
                  and all(verified[kind]["correct"] / verified[kind]["episodes"] >= .8
                          for kind in verified if kind != "all")
                  and total["requests"] / total["episodes"] <= fixed["requests"] / fixed["episodes"] + .05)
        scores[job] = {"formed": formed, "totals": verified}
        seen.add(job)
    qualified = [arm for arm in ARMS if all(scores[(arm, seed)]["formed"] for seed in PLANS["colab"].seeds)]
    if not qualified:
        raise ValueError("NO-GO: no arm passed the frozen formation gate at both Colab seeds")
    signals = []
    for arm in ARMS[1:]:
        def improvement(seed):
            candidate = scores[(arm, seed)]["totals"]
            control = scores[("ce", seed)]["totals"]
            gain = candidate["all"]["utility"] / 80 - control["all"]["utility"] / 80
            no_regression = all(candidate[kind]["correct"] / candidate[kind]["episodes"]
                >= control[kind]["correct"] / control[kind]["episodes"] - .05 for kind in candidate if kind != "all")
            return gain >= .05 and no_regression
        if all(improvement(seed) for seed in PLANS["colab"].seeds):
            signals.append(arm)
    return {"screen_sha256": fingerprint(Path(path)), "owner_reviewed": True,
            "formation_qualified": qualified, "alternative_signal": signals}


def verify_result_checkpoint(directory, result):
    """Receipt and committed payload must name the same qualified experiment."""
    from anra_v5.elendril.codec import CODEC_SHA, digest
    from anra_v5.elendril.foundation_training import Recipe, TokenSample
    from anra_v5.elendril.training import PROFILES
    platform, arm, seed = result["platform"], result["arm"], result["seed"]
    plan = PLANS[platform]
    if (arm, seed) not in plan.jobs():
        raise ValueError("unregistered checkpoint result")
    expected_recipe = asdict(Recipe(seed=seed, total_updates=plan.updates, batch_size=16, micro_batch_size=4,
        objective="action-weighted" if arm == "action-weighted" else "ce",
        sampling="uncertainty-first" if arm == "uncertainty-first" else "uniform", lr=.003, precision="fp32"))
    train = fixture(platform)[0]
    samples = [TokenSample(row.task_id, row.tokens, row.eligible, row.world_id, row.kind) for row in train.examples]
    data_sha = digest({"samples": [asdict(row) for row in samples], "provenance": train.manifest})
    model_sha = PROFILES["expanded"].sha256()
    if (result.get("recipe") != expected_recipe or result.get("training_fixture") != train.manifest
            or result.get("data_sha256") != data_sha or result.get("model_sha256") != model_sha
            or result.get("tokenizer_sha256") != CODEC_SHA):
        raise ValueError("result recipe/model/data/tokenizer identity mismatch")
    checkpoint = sha_id(result["checkpoint"])
    lineage = safe_id(result["lineage"])
    target = directory / "checkpoints" / lineage / checkpoint
    read_checkpoint(target, checkpoint)
    state = json.loads((target / "training_state.json").read_bytes())
    metadata = json.loads((target / "scheduler.json").read_bytes())["metadata"]
    if (state["identities"]["source_commit"] != result["source"]
            or state["identities"]["tokenizer_sha256"] != CODEC_SHA
            or state["identities"]["model_spec_sha256"] != model_sha
            or state["identities"]["pack_manifest_sha256"] != data_sha
            or state["lineage_id"] != lineage or state["global_update"] != plan.updates
            or state["optimizer_step_max"] != plan.updates or state["cumulative_tokens"] != state["token_budget"]
            or state["cursor"]["sequence_ordinal"] != plan.updates * 16
            or metadata["recipe"] != expected_recipe or metadata["provenance"] != train.manifest):
        raise ValueError("committed generation disagrees with result/recipe/provenance")
    return checkpoint


def rescore_evaluation(data, report):
    """Re-execute recorded raw actions against fresh public worlds and truth.

    This verifies recorded outcomes/controller costs without invoking model
    weights. It does not authenticate that a remote machine ran those weights.
    """
    from anra_v5.elendril.action_learning import DEFAULT_BUDGET, QUERY_COST, KINDS, DecodedAction, NeuralActionPolicy
    from anra_v5.elendril.cognition import run_episode, verify
    if (report.get("dataset_sha256") != data.identity or report.get("budget") != asdict(DEFAULT_BUDGET)
            or report.get("query_cost") != QUERY_COST or len(report.get("episodes", [])) != len(data.cases)
            or set(report.get("totals", {})) != {"all", *KINDS}):
        raise ValueError("evaluation fixture/budget/cost/denominator mismatch")
    cases = {case.world_id: case for case in data.cases}
    seen, rows = set(), []
    for row in report["episodes"]:
        world = row["world_id"]
        if world not in cases or world in seen or row["kind"] != cases[world].kind:
            raise ValueError("evaluation world duplicate/missing/kind mismatch")
        seen.add(world)
        case = cases[world]
        steps = list(row["trace"]["decisions"])
        consumed = 0
        def decode(prompt, remaining):
            nonlocal consumed
            if consumed == len(steps):
                return None
            step = steps[consumed]
            consumed += 1
            if step["origin"] != "model":
                raise ValueError("model arm contains a scripted decision")
            return DecodedAction(tuple(step["raw_tokens"]), step["model_positions"],
                                 step["model_calls"], step["predicted_success"])
        policy = NeuralActionPolicy(decode, row["trace"]["model_identity"])
        trace = run_episode(case.goal, case.environment, policy, budget=DEFAULT_BUDGET)
        outcome = verify(trace, case.truth)
        utility = float(outcome.correct) - QUERY_COST * trace.requests
        if (consumed != len(steps) or blob(asdict(trace)) != blob(row["trace"])
                or trace.identity != row["trace_sha256"] or blob(asdict(outcome)) != blob(row["outcome"])
                or row["requests"] != trace.requests or abs(row["utility"] - utility) > 1e-12):
            raise ValueError("raw-action replay disagrees with recorded trace/outcome/cost")
        rows.append((case.kind, trace, outcome, utility))
    totals = {}
    for kind in report["totals"]:
        subset = [row for row in rows if kind == "all" or row[0] == kind]
        totals[kind] = {"episodes": len(subset), "correct": sum(row[2].correct for row in subset),
            "requests": sum(row[1].requests for row in subset), "utility": sum(row[3] for row in subset),
            "invalid": sum(row[1].status == "invalid_action" for row in subset),
            "model_calls": sum(step.model_calls for row in subset for step in row[1].decisions),
            "model_positions": sum(step.model_positions for row in subset for step in row[1].decisions)}
    if blob(totals) != blob(report["totals"]):
        raise ValueError("rescored complete-action totals disagree")
    return totals


def run_canaries(repo, platform, source, workspace, durable):
    check_source(repo, source)
    devices = inventory(); admit_devices(devices, PLANS[platform].device_count)
    check_torch(repo, devices)
    durable.mkdir(parents=True, exist_ok=True)
    def lane(index):
        output = workspace / f"canary-lane-{index}-{uuid4().hex}"
        receipt = durable / f"canary-lane-{index}-{uuid4().hex}.json"
        mirror = durable / f"canary-mirror-{index}-{uuid4().hex}"
        command([sys.executable, "-m", "tools.elendril_action_cloud", "canary", "--owner-enabled",
            "--source", source, "--platform", platform, "--output", str(output),
            "--durable", str(mirror), "--receipt", str(receipt),
            "--seed", "7"], cwd=repo, env=device_env(devices[index]), timeout=1200,
            log=durable / ("canary-worker-" + uuid4().hex + ".log"))
        return json.loads(receipt.read_bytes())
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        return list(pool.map(lane, range(len(devices))))


def run_campaign(repo, platform, source, workspace, durable, canaries, *,
                 run_id, recovery_receipts, screen=None, owner_reviewed=False, elapsed=0.):
    check_source(repo, source)
    plan = PLANS[platform]
    devices = inventory(); admit_devices(devices, plan.device_count)
    environments = check_torch(repo, devices)
    canary_gate(canaries, source=source, platform=platform, environments=environments)
    recovery_gate(recovery_receipts, source=source, environments=environments)
    if platform == "kaggle":
        screen_gate(screen, source, owner_reviewed=owner_reviewed)
    prediction = forecast(plan, canaries, elapsed=elapsed)
    safe_id(run_id)
    durable.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def lane(index):
        results = []
        for jobindex, (arm, seed) in enumerate(plan.jobs()):
            if jobindex % len(devices) != index:
                continue
            remaining = plan.train_seconds - elapsed - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("campaign cap reached before next matched job")
            receipt = durable / f"result-{arm}-{seed}.json"
            if receipt.exists():
                raise ValueError("new campaign requires unused result names; preserve prior evidence")
            command([sys.executable, "-m", "tools.elendril_action_cloud", "train", "--owner-enabled",
                "--source", source, "--platform", platform, "--output", str(workspace / f"job-{arm}-{seed}"),
                "--durable", str(durable / "checkpoints"), "--receipt", str(receipt),
                "--seed", str(seed), "--arm", arm, "--run-id", run_id, "--seconds", str(max(1, int(remaining)))],
                cwd=repo, env=device_env(devices[index]), timeout=max(1, int(remaining)),
                log=durable / (f"worker-{arm}-{seed}-" + uuid4().hex + ".log"))
            results.append({"file": receipt.name, "sha256": fingerprint(receipt)})
        return results
    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            results = sum(list(pool.map(lane, range(len(devices)))), [])
        report = {"schema": SCHEMA + "/campaign", "source": source, "platform": platform,
            "plan": asdict(plan), "forecast": prediction, "canaries": canaries,
            "wall_seconds": time.monotonic() - started, "results": results,
            "status": "COMPLETE", "promotion": "NONE; owner scientific review required"}
        write_json(durable / "campaign.json", report)
        return report
    except BaseException as exc:
        write_json(durable / ("campaign-failure-" + uuid4().hex + ".json"), {
            "schema": SCHEMA + "/failure", "source": source, "platform": platform,
            "exception": type(exc).__name__, "message": str(exc), "status": "INCOMPLETE",
            "partial_results": [p.name for p in durable.glob("result-*.json")]})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("canary", "recover", "train"))
    parser.add_argument("--owner-enabled", action="store_true")
    parser.add_argument("--source", required=True)
    parser.add_argument("--platform", choices=tuple(PLANS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--durable", type=Path)
    parser.add_argument("--mirror", type=Path)
    parser.add_argument("--challenge", type=Path)
    parser.add_argument("--off-session-ack", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--arm", choices=ARMS, default="ce")
    parser.add_argument("--run-id", default="action-v5")
    parser.add_argument("--seconds", type=int, default=6300)
    args = parser.parse_args(argv)
    if not args.owner_enabled:
        parser.error("explicit --owner-enabled required; no default compute launch")
    args.repo = Path(__file__).resolve().parents[1]
    write_targets = [args.output, args.receipt, *([args.durable] if args.durable else [])]
    if any(path.resolve().is_relative_to(args.repo) for path in write_targets):
        parser.error("cloud outputs must be outside the immutable source checkout")
    if args.receipt.exists():
        parser.error("receipt already exists; preserve evidence and use a new attempt path")
    if ((args.mode in ("canary", "train") and args.durable is None)
            or (args.mode == "recover" and (args.mirror is None or args.challenge is None))
            or not 1 <= args.seconds <= PLANS[args.platform].train_seconds):
        parser.error("missing storage arguments or out-of-bounds cap")
    environment = worker_environment(args.repo, args.source)
    return {"canary": worker_canary, "recover": worker_recover, "train": worker_train}[args.mode](args, environment)


if __name__ == "__main__":
    main()
