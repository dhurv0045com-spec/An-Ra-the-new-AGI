"""Owner-operated V7.16 mechanism workers over the qualified training engine.

Modes: train (mechanism arms, promotion arms and native ark017-sparse replay),
profile (production-shaped timing plus exact recovery where the registration
allows resume) and recover (transactional re-drive). H4 never fabricates
traces: the shipped provider executes bounded episodes in the real environment
with a seeded grammar-restricted decoder and substitutes only RLOO-winning
traces whose rendered frame matches the registered sample shape exactly.
Importing this module runs no training.
"""
import argparse
from dataclasses import asdict, replace
import gc
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback
from uuid import uuid4

import torch

from anra_v5.elendril import v716_protocol
from anra_v5.elendril.codec import digest
from anra_v5.elendril.v716_protocol import PROTOCOL_SHA
from anra_v5.elendril.v716_training import (arm_requires_provider, create_trainer,
                                            _continual_provenance, _continual_recipe)
from anra_v5.elendril.v716_mechanisms import (OutcomeCoupledDeliberationTrainer,
                                              RolloutRecord)
from anra_v5.elendril.cognition import OperationCandidate, run_episode, verify
from anra_v5.elendril.foundation_training import StageTransition, TokenSample
from anra_v5.elendril.representation import UTF8, SpecialToken
from anra_v5.elendril.v6_data import action_text, read_dataset
from anra_v5.elendril.v6_evaluation import (DEFAULT_BUDGET, NeuralPolicy, PublicCase,
                                            parse_action)
from anra_v5.elendril.v71_data import suite_from_record
from anra_v5.elendril.v71_evaluation import evaluate_suite, verify_suite
from tools.elendril_cloud import check_source
from tools.elendril_v6_export import immutable_json
from tools.elendril_v6_worker import _weights, append_json
from tools.elendril_v71_worker import evaluation
from v5_training.checkpoint import CheckpointStore

REPO = Path(__file__).resolve().parents[1]

# Byte-token ids a V6 action may render through parse_action: EOS, "!", "A".."P",
# "Q", "R" and "~". Rollout decoding samples inside this support only; that is a
# registered H4 design decision, recorded in the rollout audit of every update.
ROLLOUT_SUPPORT = (UTF8.eos_id, ord("!") + 4,
                   *(ord(letter) + 4 for letter in "ABCDEFGHIJKLMNOP"),
                   ord("Q") + 4, ord("R") + 4, ord("~") + 4)
ROLLOUTS_REGISTERED = 2
RESUME_REFUSED = ("H5 pointer and ark017-sparse lineages are fresh-only in V7.16 v1; "
                  "exact resume is not checkpoint-qualified")


class SamplingPolicy(NeuralPolicy):
    """NeuralPolicy with seeded multinomial decoding inside the action support.

    Identical prompt rendering, budget accounting and no-repair parsing to the
    qualified evaluation policy; only argmax decoding is replaced, because
    identical rollouts make the RLOO baseline degenerate.
    """

    def __init__(self, backend, surface=None, *, memory_available=False,
                 max_new_tokens=4, seed=0):
        super().__init__(backend, surface, memory_available=memory_available,
                         max_new_tokens=max_new_tokens)
        if type(seed) is not int or not 0 <= seed < 2 ** 63:
            raise ValueError("rollout seed must be a bounded nonnegative integer")
        self.seed = seed
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def for_case(self, case, *, memory_available=None):
        return type(self)(self.backend, case.surface,
                          memory_available=case.memory is not None
                          if memory_available is None else memory_available,
                          max_new_tokens=self.max_new_tokens, seed=self.seed)

    def propose(self, state, legal, channels, positions):
        from anra_v5.elendril.v6_data import render_public

        if self.surface is None:
            raise ValueError("bind a public case surface before decoding")
        prompt = render_public(state, channels, self.surface,
                               self.memory_available and "RETRIEVE" in legal)
        ids = [UTF8.bos_id, *UTF8.encode(prompt), UTF8.special(SpecialToken.ANSWER)]
        if len(ids) > self.backend.binding.context_length or len(ids) > positions:
            return None
        raw, probabilities, used = [], [], 0
        support = torch.tensor(ROLLOUT_SUPPORT, dtype=torch.int64)
        for _ in range(self.max_new_tokens):
            if len(ids) > self.backend.binding.context_length or used + len(ids) > positions:
                break
            distribution = self.backend.next_distribution(ids).cpu()
            mass = distribution[support]
            if not torch.isfinite(mass).all() or float(mass.sum()) <= 0.:
                break
            token = int(support[int(torch.multinomial(
                mass / mass.sum(), 1, generator=self.generator))])
            probabilities.append(float(distribution[token]))
            used += len(ids)
            raw.append(token)
            if token in (UTF8.eos_id, UTF8.pad_id, UTF8.bos_id,
                         UTF8.special(SpecialToken.ANSWER)):
                break
            ids.append(token)
        self.prompts.append(prompt)
        operation, argument = parse_action(raw, channels)
        return OperationCandidate(operation, argument, math.prod(probabilities),
                                  used, len(raw), tuple(raw))


def _sampling_policy(backend, seed, k=0):
    """Registered production rollout decoder: seeded multinomial, action support.

    The factory seam exists so CPU canaries can inject a bounded stub; the
    shipped configuration always resolves to this function. ``k`` is the
    rollout index and only matters for injected stubs.
    """
    del k
    return SamplingPolicy(backend, max_new_tokens=4, seed=seed)


class ShapedRolloutProvider:
    """Executed-rollout seam for H4 (registered rollouts K=2, U(tau) costs).

    For every chosen (case, decision) task the provider executes K bounded
    episodes in the real environment, scores each trace with the registered
    cost parameters, keeps traces above the RLOO baseline, and substitutes the
    winning trace only when its action text and rendered frame reproduce the
    registered sample shape exactly (same total tokens, same eligible count).
    All rejections are ledgered to the rollout audit; unqualified tasks keep
    the baseline row, so the registered batch, update count and supervised
    token ledger can never move.
    """

    schema = "elendril-v716-executed-rollout-provider-v1"

    def __init__(self, data, *, rollouts=ROLLOUTS_REGISTERED, query_cost=0.1,
                 deliberate_cost=0.0, illegal_cost=0.05, policy_factory=None):
        if rollouts not in (1, 2, 4):
            raise ValueError("registered rollout budget is 1, 2 or 4")
        self.cases = {case.case_id: case for case in data.cases}
        if len(self.cases) != len(data.cases):
            raise ValueError("rollout provider requires unique case ids")
        self.rollouts = rollouts
        self.policy_factory = policy_factory or _sampling_policy
        self.selector = OutcomeCoupledDeliberationTrainer(
            rollouts=rollouts, query_cost=query_cost,
            deliberate_cost=deliberate_cost, illegal_cost=illegal_cost)
        self.backend = None
        self.seed_base = None
        self.audit = []
        self.audit_path = None

    def bind(self, trainer):
        if self.backend is not None:
            raise ValueError("rollout provider is already bound to a trainer")
        self.backend = trainer.backend
        self.seed_base = trainer.recipe.seed

    def __call__(self, update, chosen):
        if self.backend is None:
            raise ValueError("rollout provider used before trainer binding")
        if len({sample.task_id for sample in chosen}) != len(chosen):
            raise ValueError("one update must not repeat a rollout task")
        groups, executed, rejections = {}, {}, []
        for sample in chosen:
            base_task = sample.task_id.split("@")[0]
            case_id = base_task.split("/")[0]
            slot = int(base_task.split("/")[1])
            case = self.cases.get(case_id)
            if case is None:
                raise ValueError(f"rollout task {sample.task_id} has no registered case")
            outcomes = [self._execute(case, sample, slot, k, update)
                        for k in range(self.rollouts)]
            traces = [trace for trace in outcomes if isinstance(trace, tuple)]
            rejections.extend(trace for trace in outcomes if isinstance(trace, str))
            executed[sample.task_id] = traces
            if len(traces) == self.rollouts:
                groups[sample.task_id] = [record for record, _, _ in traces]
            else:
                rejections.append({"task_id": sample.task_id,
                                   "reason": "incomplete executed rollout group; "
                                             "baseline row kept"})
        kept = self.selector.select(groups) if groups else {}
        winners = {}
        for task, records in kept.items():
            winners[task] = max((triple for triple in executed[task]
                                 if triple[0] in records),
                                key=lambda triple: (triple[2], triple[0].trace_id))
        replacements = []
        for sample in chosen:
            candidate = self._frame(sample, winners.get(sample.task_id), rejections)
            replacements.append(sample if candidate is None else candidate)
        audit = self.selector.audit(groups) if groups else {
            "schema": self.selector.schema, "prompts": 0, "rollouts": 0, "kept": 0,
            "kept_fraction": 0., "mean_utility": 0.,
            "query_cost": self.selector.query_cost,
            "rollouts_per_prompt": self.rollouts}
        row = {"schema": self.schema, "update": update, "selection": audit,
               "substituted": sum(1 for sample, replacement in zip(chosen, replacements)
                                  if replacement is not sample),
               "rejections": rejections}
        self.audit.append(row)
        if self.audit_path is not None:
            append_json(self.audit_path, row)
        return replacements

    def _execute(self, case, sample, slot, k, update):
        """One executed episode; returns (record, prompt, utility) or a reason."""
        seed = int(digest([self.schema, self.seed_base, update, sample.task_id, k]), 16) \
            % (2 ** 63)
        policy = self.policy_factory(self.backend, seed, k)
        actor = policy.for_case(PublicCase(case.case_id, case.surface,
                                           case.memory is not None),
                                memory_available=case.memory is not None)
        trace = run_episode(case.goal, case.environment, actor, budget=DEFAULT_BUDGET,
                            memory=case.memory)
        outcome = verify(trace, case.truth)
        if slot >= len(trace.decisions):
            return f"{sample.task_id}-k{k}: episode ended before decision slot"
        decision = trace.decisions[slot]
        if decision.origin != "model":
            return f"{sample.task_id}-k{k}: decision did not originate from the model"
        try:
            actions = action_text(decision.operation, decision.argument)
        except ValueError:
            return f"{sample.task_id}-k{k}: action outside the registered V6 grammar"
        prompt = actor.prompts[slot] if slot < len(actor.prompts) else None
        if prompt is None:
            return f"{sample.task_id}-k{k}: rendered prompt unavailable"
        record = RolloutRecord(
            task_id=sample.task_id, trace_id=f"{sample.task_id}-k{k}", actions=actions,
            correct=bool(outcome.correct), knowable=bool(outcome.admitted_public),
            verified_queries=sum(item.operation == "QUERY"
                                 and item.argument in case.environment.channels
                                 for item in trace.decisions),
            deliberations=sum(item.operation == "DELIBERATE" for item in trace.decisions),
            illegal=1 if trace.status == "invalid_action" else 0,
            supervised_tokens=sum(sample.eligible))
        return record, prompt, self.selector.utility(record)

    def _frame(self, sample, winner, rejections):
        """Shape-match the winning kept trace; ledger every rejection reason."""
        if winner is None:
            return None
        record, prompt, _ = winner
        if len(UTF8.encode(record.actions)) + 1 != sum(sample.eligible):
            rejections.append({"task_id": sample.task_id, "trace_id": record.trace_id,
                               "reason": "answer byte length breaks the shape contract"})
            return None
        try:
            frame = UTF8.frame(prompt, record.actions, max_length=len(sample.tokens),
                               pad_to=len(sample.tokens))
        except ValueError as exc:
            rejections.append({"task_id": sample.task_id, "trace_id": record.trace_id,
                               "reason": f"frame rejected: {exc}"})
            return None
        replacement = TokenSample(sample.task_id, frame.ids, frame.loss_mask,
                                  sample.world_id, sample.category)
        if len(replacement.tokens) != len(sample.tokens) \
                or sum(replacement.eligible) != sum(sample.eligible):
            rejections.append({"task_id": sample.task_id, "trace_id": record.trace_id,
                               "reason": "post-frame shape mismatch"})
            return None
        return replacement


def create(data, job, *, source, root, resume=False, transition=None, parent_store=None,
           stage_samples=None, replay_samples=None, rollout_provider=None):
    return create_trainer(data, job, source=source, root=Path(root),
                          resume=resume, transition=transition,
                          parent_store=parent_store, stage_samples=stage_samples,
                          replay_samples=replay_samples,
                          rollout_provider=rollout_provider)


def stage_transition(data, job, stage_samples, replay_samples, parent, payloads):
    """Bind the ark017-sparse stage transition to the exact child plan."""
    _, _, recipe = _continual_recipe(job, stage_samples)
    provenance, _ = _continual_provenance(data, job, stage_samples, replay_samples)
    child_data = digest({"samples": [asdict(sample) for sample in stage_samples],
                         "provenance": provenance})
    parent_recipe = digest(json.loads(payloads["scheduler.json"])["metadata"]["recipe"])
    return StageTransition(parent["checkpoint"], parent_recipe, digest(asdict(recipe)),
                           child_data)


def train(req, deadline):
    job = req["job"]
    data = read_dataset(req["data_file"])
    root = Path(req["checkpoint_root"])
    output = Path(req["output"])
    suite = suite_from_record(json.loads(Path(req["suite_file"]).read_bytes()))
    provider = ShapedRolloutProvider(data) if arm_requires_provider(job) else None
    resume = (root / job["job_id"] / "LATEST").exists()
    transition, parent_store, stage_samples, replay_samples = None, None, None, None
    if job["mechanism"] == "continual":
        if resume:
            raise ValueError(RESUME_REFUSED)
        pool_data = read_dataset(req["replay_file"])
        stage_samples = tuple(data.samples)
        replay_samples = tuple(sample for sample in pool_data.samples
                               if sample.category in ("memory", "missing"))
        if not replay_samples:
            raise ValueError("replay pool is empty; Skill-A donor episodes required")
        parent = req["parent"]
        parent_store = CheckpointStore(Path(parent["root"]), parent["lineage"])
        _, payloads = parent_store.restore(parent["checkpoint"])
        transition_file = output / "transition.json"
        if transition_file.exists():
            transition = StageTransition(**json.loads(transition_file.read_bytes()))
        else:
            transition = stage_transition(data, job, stage_samples, replay_samples,
                                          parent, payloads)
            immutable_json(transition_file, asdict(transition))
    trainer = create(data, job, source=req["source"], root=root, resume=resume,
                     transition=transition, parent_store=parent_store,
                     stage_samples=stage_samples, replay_samples=replay_samples,
                     rollout_provider=provider)
    if provider is not None:
        provider.bind(trainer)
        provider.audit_path = output / "rollout-audit.jsonl"
    before = trainer.state.global_update
    rows_per_epoch = len(trainer.epoch_samples[0]) if trainer.epoch_samples is not None \
        else len(trainer.samples)
    epoch_updates = math.ceil(rows_per_epoch / trainer.recipe.batch_size)
    marks = {epoch_updates * e for e in range(1, job["epochs"] + 1) if e in (1, 2, 4)}
    initial = "after-transfer-before-B.json" if job["mechanism"] == "continual" \
        else "initial.json"
    if not resume:
        evaluation(trainer, suite, output / initial, conditions=["base", "hop-3"])

    def inspect():
        update = trainer.state.global_update
        if update not in marks:
            return
        trainer.store.record_milestone(threshold_tokens=trainer.state.cumulative_tokens,
                                       checkpoint_sha256=trainer.parent)
        final = update == trainer.recipe.total_updates
        evaluation(trainer, suite, output / f"epoch-{update // epoch_updates}.json",
                   conditions=None if final else ["base", "hop-3"])

    if resume:
        inspect()
    log = output / f"updates-{uuid4().hex}.jsonl"
    interrupted = None
    try:
        while not trainer.state.complete and time.monotonic() < deadline - 180:
            started = time.monotonic()
            row = trainer.step()
            row["wall_seconds"] = time.monotonic() - started
            chosen = [trainer.samples[i] for i in trainer.plan[row["update"] - 1]]
            chunks = [chosen[i:i + trainer.recipe.micro_batch_size]
                      for i in range(0, len(chosen), trainer.recipe.micro_batch_size)]
            row["padded_input_positions"] = sum(
                len(chunk) * max(len(sample.tokens) for sample in chunk)
                for chunk in chunks)
            row["padded_attention_pairs"] = sum(
                len(chunk) * max(len(sample.tokens) for sample in chunk) ** 2
                for chunk in chunks)
            append_json(log, row)
            if trainer.state.global_update % 64 == 0 or trainer.state.global_update in marks:
                trainer.save()
                inspect()
                trainer.store.prune(keep={trainer.parent})
    except (KeyboardInterrupt, InterruptedError) as exc:
        interrupted = type(exc).__name__
    if not trainer.faulted and trainer.state.global_update > trainer.published_update:
        trainer.save()
    if trainer.state.complete and not interrupted:
        evaluation(trainer, suite, output / "grammar-final.json", grammar=True)
    result = {"status": "COMPLETE" if trainer.state.complete and not interrupted
              else "PARTIAL", "interrupted": interrupted,
              "recovery_required": bool(trainer.faulted), "job": job,
              "source": req["source"], "checkpoint": trainer.parent,
              "checkpoint_root": str(root), "start_update": before,
              "global_update": trainer.state.global_update,
              "total_updates": trainer.recipe.total_updates,
              "supervised_tokens": trainer.state.cumulative_tokens,
              "metadata": trainer.metadata,
              "parameter_count": sum(parameter.numel()
                                     for parameter in trainer.model.parameters())}
    if provider is not None:
        result["rollout_audit_updates"] = len(provider.audit)
        result["rollout_substitutions"] = sum(row["substituted"] for row in provider.audit)
        result["rollout_kept_traces"] = sum(row["selection"]["kept"]
                                            for row in provider.audit)
        result["rollout_executed_rollouts"] = sum(row["selection"]["rollouts"]
                                                  for row in provider.audit)
    if trainer.replay_buffer is not None:
        result["replay_history"] = trainer.replay_history()
    return result


def profile(req, deadline):
    job = req["job"]
    if job["mechanism"] == "continual":
        raise ValueError("continual capacity is forecast from measured mechanism "
                         "profiles; continual arms are fresh-only and not profiled")
    root = Path(req["checkpoint_root"])
    output = Path(req["output"])
    data = read_dataset(req["data_file"])
    provider = ShapedRolloutProvider(data) if arm_requires_provider(job) else None
    trainer = create(data, job, source=req["source"], root=root / "expected",
                     rollout_provider=provider)
    if provider is not None:
        provider.bind(trainer)
    init_weights = _weights(trainer.model)
    timings, head2, head4 = [], None, None
    for update in range(1, 9):
        if time.monotonic() > deadline:
            raise TimeoutError("profile deadline")
        started = time.monotonic()
        trainer.step()
        if trainer.device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.monotonic() - started)
        if update in (2, 4):
            trainer.save()
            if update == 2:
                head2 = trainer.parent
            else:
                head4 = trainer.parent
    if _weights(trainer.model) == init_weights:
        raise ValueError("profile weights did not update")
    suite = suite_from_record(json.loads(Path(req["suite_file"]).read_bytes()))
    trainer.backend.binding = replace(trainer.backend.binding,
                                      checkpoint_sha256=_weights(trainer.model))
    started = time.monotonic()
    report = evaluate_suite({"base": suite["base"]}, NeuralPolicy(trainer.backend))
    verify_suite({"base": suite["base"]}, report)
    episode_seconds = (time.monotonic() - started) / len(suite["base"])
    immutable_json(output / "timing-evaluation.json", report)
    resumable = job["arm"] != "H5-POINTER-READ"
    recovery = {"status": "NOT_QUALIFIED",
                "reason": RESUME_REFUSED}
    if resumable:
        challenge = {"source": req["source"], "protocol_sha256": PROTOCOL_SHA,
                     "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
                     "expected_root": str(root / "expected"), "head2": head2,
                     "head4": head4}
        immutable_json(output / "challenge.json", challenge)
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        child = {**req, "checkpoint_root": str(root / "recovery"),
                 "output": str(output / "recovery"),
                 "challenge": str(output / "challenge.json"),
                 "timecap_seconds": max(1, int(deadline - time.monotonic()))}
        request = output / "recovery-request.json"
        receipt = output / "recovery" / "receipt.json"
        immutable_json(request, child)
        process = subprocess.run(
            [sys.executable, "-m", "tools.elendril_v716_worker", "recover",
             "--owner-enabled", "--request", str(request), "--receipt", str(receipt)],
            cwd=REPO, timeout=max(1, deadline - time.monotonic()))
        recovered = json.loads(receipt.read_bytes())
        if process.returncode or recovered["status"] != "PASS":
            raise ValueError("profile recovery failed: " + str(recovered))
        transient = (root / "recovery").resolve()
        if not transient.is_relative_to(root.resolve()) or transient == root.resolve():
            raise ValueError("unsafe transient root")
        shutil.rmtree(transient)
        recovery = recovered
    return {"status": "PASS", "profile": job["profile"], "mechanism": job["mechanism"],
            "seconds_per_update": sum(timings) / len(timings),
            "evaluation_seconds_per_episode": episode_seconds,
            "recovery": recovery, "init_weights_sha256": init_weights,
            "head2": head2, "head4": head4, "checkpoint_root": str(root / "expected"),
            "qualification": "production-shaped timing with mechanism seams active; "
                             "no skill claim"}


def recover(req):
    from tools.elendril_action_cloud import compare_payloads

    job = req["job"]
    challenge = json.loads(Path(req["challenge"]).read_bytes())
    if (challenge["source"] != req["source"]
            or challenge["protocol_sha256"] != PROTOCOL_SHA
            or challenge["gpu_uuid"] != os.environ["CUDA_VISIBLE_DEVICES"]):
        raise ValueError("challenge mismatch")
    reference = CheckpointStore(Path(challenge["expected_root"]), job["job_id"])
    state, payloads = reference.restore(challenge["head2"])
    destination = CheckpointStore(Path(req["checkpoint_root"]), job["job_id"])
    published = destination.publish(state=state, payloads=payloads,
                                    expected_parent_sha256=None)
    if published != challenge["head2"]:
        raise ValueError("republished generation differs")
    data = read_dataset(req["data_file"])
    provider = ShapedRolloutProvider(data) if arm_requires_provider(job) else None
    trainer = create(data, job, source=req["source"], root=Path(req["checkpoint_root"]),
                     resume=True, rollout_provider=provider)
    if provider is not None:
        provider.bind(trainer)
    if trainer.state.global_update != 2:
        raise ValueError("recovery cursor mismatch")
    while trainer.state.global_update < 4:
        trainer.step()
    trainer.save()
    _, left = reference.restore(challenge["head4"])
    _, right = trainer.store.restore(trainer.parent)
    compare_payloads(left, right, exact=True)
    return {"status": "PASS", "checkpoint": trainer.parent, "global_update": 4,
            "comparison": "exact full payload comparison"}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("profile", "recover", "train"))
    parser.add_argument("--owner-enabled", action="store_true")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if not arguments.owner_enabled:
        parser.error("owner execution required")
    request = json.loads(arguments.request.read_bytes())
    cap = request["timecap_seconds"]
    if request["protocol_sha"] != PROTOCOL_SHA or type(cap) not in (int, float) \
            or not math.isfinite(cap) or not 1 <= cap <= 21600:
        raise ValueError("protocol or wall cap mismatch")
    check_source(REPO, request["source"])
    out = Path(request["output"]).resolve()
    cp = Path(request.get("checkpoint_root", out / "unused")).resolve()
    if out.is_relative_to(REPO) or cp.is_relative_to(REPO) \
            or not arguments.receipt.resolve().is_relative_to(out) \
            or arguments.receipt.exists():
        raise ValueError("outside-source output and new contained receipt required")
    out.mkdir(parents=True, exist_ok=True)
    if arguments.mode in ("train", "profile", "recover"):
        v716_protocol.validate(request["job"], request["phase"])
    started = time.monotonic()

    def stop(_signum, _frame):
        raise InterruptedError("worker termination requested; preserve verified checkpoint")

    signal.signal(signal.SIGTERM, stop)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated CUDA GPU required")
        handler = {"profile": profile, "recover": recover, "train": train}[arguments.mode]
        result = handler(request, started + cap)
        result.update(gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],
                      wall_seconds=time.monotonic() - started,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as exc:
        result = {"status": "ERROR", "error_type": type(exc).__name__,
                  "message": str(exc), "traceback": traceback.format_exc(),
                  "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES")}
    result.update(source=request["source"], protocol_sha256=PROTOCOL_SHA,
                  mode=arguments.mode)
    immutable_json(arguments.receipt, result)
    return 0 if result["status"] in ("PASS", "COMPLETE", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
