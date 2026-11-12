"""V7.16 exposure-locked trainer factory for the six mechanism arms.

One factor changes at a time against the blocked cross-entropy baseline while
epochs, optimizer updates, supervised tokens and unpadded input positions are
held exactly equal. The factory reuses the qualified FoundationTrainer engine
(transactional checkpoints, exact resume, deterministic plans) and adds only
the mechanism seams documented here. Importing this module runs no training.

Replay accounting rule (H6): the registered batch ledger and the state token
budget stay bound to the baseline batch; every swapped-in Skill-A episode and
its supervised-token delta are reported separately in telemetry, so replay
exposure can never be silently inflated or denied.
"""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import re

import torch

from . import v716_protocol
from .backend import BackendIdentity, NeuralBackend
from .codec import canonical, digest
from .foundation_training import Recipe, TokenSample, tensor_inventory
from .representation import UTF8
from .training import batch_tensors, configure_cpu, configure_device
from .v6_data import _supervision
from .v6_registry import MODEL_PROFILES
from .v7_training import V7Trainer, ordered_indices
from .v716_mechanisms import (Ark014SurfaceAugmenter, Ark017SurpriseReplayBuffer,
                              CanonicalRowNormalizer, CausalPointerTwoReadDecoder,
                              CoAdaptedControllerMaskLoss, CounterfactualMemoryGenerator,
                              public_state_specs)

CONTRASTIVE_WEIGHT = 0.1
CONTRASTIVE_MARGIN = 0.5
REPLAY_FRACTION = 0.1
MECHANISM_JOB_KEYS = {"job_id", "arm", "profile", "seed", "epochs", "mechanism", "exposure"}
CONTINUAL_JOB_KEYS = MECHANISM_JOB_KEYS | {"stage"}
CONTINUAL_STAGES = v716_protocol.CONTINUAL_ARMS
POINTER_ARCHITECTURE_SUFFIX = CausalPointerTwoReadDecoder.VERSION


def _suffix(sample, epoch):
    return TokenSample(f"{sample.task_id}@e{epoch}", sample.tokens, sample.eligible,
                       sample.world_id, sample.category)


def _spec_key(task_id):
    return task_id.split("@")[0]


def _epoch_rows(data, epochs):
    """Baseline rows replicated per epoch with epoch-suffixed unique task ids."""
    return [tuple(_suffix(sample, epoch) for sample in data.samples)
            for epoch in range(1, epochs + 1)]


def _cases_to_rows(cases, epoch):
    examples = tuple(example for case in cases for example in _supervision(case)[0])
    return tuple(_suffix(example.sample, epoch) for example in examples)


class V716Trainer(V7Trainer):
    """Blocked-CE engine with explicit, contract-checked mechanism seams.

    ``epoch_samples`` supplies one prebuilt sample tuple per epoch (unique task
    ids through the ``@e<epoch>`` suffix). The attribute is bound before the
    superclass plan is computed, so the registered plan, counts and token
    budget already reflect the epoch datasets.
    """

    def __init__(self, samples, *, spec, recipe, source_commit, root, lineage,
                 data_provenance, tokenizer_manifest, device="cpu", resume=False,
                 transition=None, parent_store=None, method="blocked",
                 epoch_samples=None, state_specs=None, mask=None, pointer=False,
                 contrastive_pairs=None, replay_buffer=None, rollout_provider=None):
        self.epoch_samples = None if epoch_samples is None else tuple(
            tuple(rows) for rows in epoch_samples)
        self.state_specs = None if state_specs is None else dict(state_specs)
        self.mask = mask
        self.pointer = bool(pointer)
        if self.pointer:
            # The superclass resume path loads a saved state dict into the bare
            # core, which cannot carry pointer tensors; and stage migration
            # would silently change the freshly-installed architecture.
            if resume:
                raise ValueError("H5 pointer lineages are fresh-only in V7.16 v1; "
                                 "crash recovery is not yet checkpoint-qualified")
            if transition is not None or parent_store is not None:
                raise ValueError("H5 pointer lineages reject stage migration in V7.16 v1")
        self.contrastive_pairs = () if contrastive_pairs is None else tuple(contrastive_pairs)
        self.replay_buffer = replay_buffer
        self.rollout_provider = rollout_provider
        self.replay_extra_tokens = 0
        self._samples_by_task = {}
        self._replay_ledger = []
        super().__init__(samples, spec=spec, recipe=recipe, source_commit=source_commit,
                         root=root, lineage=lineage, data_provenance=data_provenance,
                         tokenizer_manifest=tokenizer_manifest, device=device,
                         resume=resume, transition=transition, parent_store=parent_store,
                         method=method)
        if self.pointer:
            self._install_pointer(spec, recipe)
        self._samples_by_task = {sample.task_id: sample for sample in self.samples}
        if self.contrastive_pairs:
            missing = sorted({task_id for pair in self.contrastive_pairs
                              for task_id in (pair[1], pair[2])
                              if task_id not in self._samples_by_task})
            if missing:
                raise ValueError(f"contrastive pairs reference unknown samples: {missing[:4]}")

    # -- epoch-addressed plan -------------------------------------------------
    def indices(self, update):
        if self.epoch_samples is None:
            return super().indices(update)
        count = len(self.epoch_samples[0])
        per_epoch = math.ceil(count / self.recipe.batch_size)
        epoch, batch = divmod(update, per_epoch)
        if epoch >= len(self.epoch_samples):
            raise ValueError("plan exceeds the registered epoch datasets")
        key = ("epoch", epoch)
        if key not in self._epoch_orders:
            local = ordered_indices(self.epoch_samples[epoch], self.recipe.seed, epoch, self.method)
            self._epoch_orders[key] = tuple(epoch * count + index for index in local)
        order = self._epoch_orders[key]
        start = batch * self.recipe.batch_size
        return order[start:start + self.recipe.batch_size]

    def sample_cursor(self, completed_updates):
        if self.epoch_samples is None:
            return super().sample_cursor(completed_updates)
        count = len(self.epoch_samples[0])
        per_epoch = math.ceil(count / self.recipe.batch_size)
        epoch, batch = divmod(completed_updates, per_epoch)
        epoch = min(epoch, len(self.epoch_samples) - 1)
        return epoch * count + min(batch * self.recipe.batch_size, count)

    # -- H5 pointer installation ---------------------------------------------
    def _install_pointer(self, spec, recipe):
        from v5_training.optimizer import build_adamw_optimizer, optimizer_group_receipt

        if self.parent is not None:
            raise ValueError("H5 pointer lineages are fresh-only in V7.16 v1")
        self.model = CausalPointerTwoReadDecoder(self.model)
        architecture = digest([spec.sha256(), POINTER_ARCHITECTURE_SUFFIX])
        self.backend = NeuralBackend(self.model,
            BackendIdentity(architecture, self.codec_sha, spec.vocabulary_size,
                            spec.context_length),
            pad_id=self.pad, bos_id=self.bos, eos_id=self.eos)
        self.optimizer = build_adamw_optimizer(self.model, lr=recipe.lr)
        self.metadata["tensors"] = tensor_inventory(self.model)
        self.metadata = json.loads(canonical(self.metadata))
        identities = replace(self.state.identities,
                             model_spec_sha256=architecture,
                             optimizer_spec_sha256=optimizer_group_receipt(
                                 self.model, self.optimizer)["sha256"],
                             run_spec_sha256=digest(self.metadata))
        self.state = replace(self.state, identities=identities)
        self.receipt = self.model.parameter_receipt()

    # -- one update with mechanism seams --------------------------------------
    def step(self):
        if self.faulted or self.state.complete:
            raise ValueError("complete or faulted engine requires a fresh restore")
        self.faulted = True
        update = self.state.global_update
        plan_indices = tuple(self.plan[update])
        chosen = [self.samples[index] for index in plan_indices]
        telemetry = {"replay_swaps": 0, "replay_extra_supervised_tokens": 0}
        swapped_in, swapped_out, pool_positions = [], [], ()
        if self.replay_buffer is not None:
            pool, swapped_out, ledger = self.replay_buffer.plan_update(plan_indices)
            pool_positions = pool
            swapped_in = [self.replay_buffer.samples[position] for position in pool]
            if pool:
                removed = set(swapped_out)
                chosen = [sample for index, sample in enumerate(chosen)
                          if plan_indices[index] not in removed] + swapped_in
                telemetry["replay_swaps"] = len(pool)
                telemetry["replay_ledger"] = ledger
        chosen = self._apply_rollout_provider(update, chosen)
        weights = [[float(flag) for flag in sample.eligible] for sample in chosen]
        denominator = sum(map(sum, weights))
        if denominator <= 0:
            raise ValueError("update batch carries no supervised tokens")
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        learning_rate = self.learning_rate(update)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        numerator = 0.
        pending_pairs = self._epoch_pairs(update)
        for start in range(0, len(chosen), self.recipe.micro_batch_size):
            chunk = chosen[start:start + self.recipe.micro_batch_size]
            tokens, segments, _ = batch_tensors(
                [(sample.tokens, sample.eligible) for sample in chunk], self.device)
            loss_weights = torch.tensor(
                [row + [0.] * (tokens.shape[1] - len(row))
                 for row in weights[start:start + len(chunk)]], device=self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.recipe.precision == "fp16"):
                logits = self.backend.logits(
                    tokens, segments,
                    activation_checkpointing=self.recipe.activation_checkpointing)
                if self.mask is not None:
                    logits = self.mask(logits, tokens,
                                       self._eligible_tensor(chunk, tokens.shape[1]),
                                       self._specs_for(chunk))
                losses = self._token_losses(logits, tokens)
                value = (losses * loss_weights[:, 1:]).sum()
            if not torch.isfinite(logits).all() or not torch.isfinite(value):
                raise FloatingPointError("nonfinite forward; update rejected")
            self.scaler.scale(value / denominator).backward()
            numerator += value.item()
        if pending_pairs:
            penalty = self._contrastive_penalty(pending_pairs)
            self.scaler.scale(CONTRASTIVE_WEIGHT * penalty).backward()
            telemetry["contrastive_penalty"] = float(penalty.detach())
        self.scaler.unscale_(self.optimizer)
        from v5_training.optimizer import validate_parameter_ownership
        validate_parameter_ownership(self.model, self.optimizer)
        if any(parameter.grad is None for parameter in self.model.parameters()):
            raise ValueError("silent missing gradient")
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe.clip,
                                              error_if_nonfinite=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if len(self.optimizer.state) != len(list(self.model.parameters())) or any(
                slot["step"].item() != update + 1 for slot in self.optimizer.state.values()):
            raise ValueError("optimizer skipped or duplicated the proposed update")
        if any(not torch.isfinite(parameter).all() for parameter in self.model.parameters()):
            raise FloatingPointError("nonfinite parameter; restore committed parent")
        if swapped_in:
            extra = sum(sum(sample.eligible) for sample in swapped_in) \
                - sum(sum(self.samples[index].eligible) for index in swapped_out)
            self.replay_extra_tokens += extra
            mean_loss = numerator / denominator
            for position, sample in zip(pool_positions, swapped_in):
                self.replay_buffer.observe(position, mean_loss)
            telemetry["replay_extra_supervised_tokens"] = extra
        from hashlib import sha256
        self.state = replace(self.state, generation=update + 1, global_update=update + 1,
                             cumulative_tokens=self.state.cumulative_tokens + self.counts[update],
                             tokens_by_source={"action": self.state.cumulative_tokens
                                               + self.counts[update]},
                             optimizer_step_max=update + 1,
                             schedule_tokens=self.state.cumulative_tokens + self.counts[update],
                             cursor=replace(self.state.cursor,
                                            sequence_ordinal=self.sample_cursor(update + 1)),
                             rng_state_sha256=sha256(self.rng_bytes()).hexdigest(),
                             parent_checkpoint_sha256=self.parent)
        self.state.assert_valid()
        self.faulted = False
        return {"update": update + 1, "loss": numerator / denominator,
                "weighted_denominator": denominator,
                "supervised_tokens": self.counts[update],
                "input_positions": sum(len(sample.tokens) for sample in chosen),
                "examples": len(chosen),
                "microbatches": math.ceil(len(chosen) / self.recipe.micro_batch_size),
                "pre_clip": float(norm), "lr": learning_rate,
                "indices": list(plan_indices), **telemetry}

    def _swapped_positions(self, plan_indices):
        """Legacy hook retained for symmetry; the pool is captured in step()."""
        return ()

    def _token_losses(self, logits, tokens):
        if self.pointer:
            targets = tokens[:, 1:]
            return -logits[:, :-1, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            tokens[:, 1:].reshape(-1), reduction="none").reshape_as(tokens[:, 1:])

    def _eligible_tensor(self, chunk, length):
        return torch.tensor([[float(flag) for flag in sample.eligible]
                             + [0.] * (length - len(sample.eligible)) for sample in chunk],
                            device=self.device)

    def _specs_for(self, chunk):
        if self.state_specs is None:
            raise ValueError("mechanism masking requires public state specs")
        specs = []
        for sample in chunk:
            spec = self.state_specs.get(_spec_key(sample.task_id))
            if spec is None:
                raise ValueError(f"missing public state spec for {sample.task_id}")
            specs.append(spec)
        return specs

    def _apply_rollout_provider(self, update, chosen):
        if self.rollout_provider is None:
            return chosen
        replacement = self.rollout_provider(update, tuple(chosen))
        if len(replacement) != len(chosen) or any(
                not isinstance(sample, TokenSample) for sample in replacement):
            raise ValueError("rollout provider must return one TokenSample per chosen row")
        for original, candidate in zip(chosen, replacement):
            if len(candidate.tokens) != len(original.tokens) \
                    or sum(candidate.eligible) != sum(original.eligible):
                raise ValueError("rollout replacement breaks the exposure ledger; "
                                 "shape-matched substitution is required")
        return list(replacement)

    def _epoch_pairs(self, update):
        """Pairs penalized at this update: every epoch pair at its first update.

        Deterministic and bounded: one extra forward/backward over the epoch's
        twin pairs at the first update of each epoch, independent of batch
        membership, so the penalty always applies exactly once per epoch.
        """
        if not self.contrastive_pairs or self.epoch_samples is None:
            return ()
        count = len(self.epoch_samples[0])
        per_epoch = math.ceil(count / self.recipe.batch_size)
        if update % per_epoch:
            return ()
        epoch = update // per_epoch + 1
        return tuple(pair for pair in self.contrastive_pairs if pair[0] == epoch)

    def _contrastive_penalty(self, pending_pairs):
        from v5_model.core import packed_layout

        from .v716_mechanisms import OrderContrastiveLoss
        loss = OrderContrastiveLoss(margin=CONTRASTIVE_MARGIN)
        left_rows, right_rows = [], []
        for _, task_left, task_right in pending_pairs:
            left_rows.append(self._samples_by_task[task_left])
            right_rows.append(self._samples_by_task[task_right])
        hidden = []
        for rows in (left_rows, right_rows):
            tokens, segments, _ = batch_tensors(
                [(sample.tokens, sample.eligible) for sample in rows], self.device)
            positions, mask = packed_layout(segments, torch_module=torch)
            model = self.model.core if self.pointer else self.model
            states = model.forward_hidden(tokens, positions, mask)
            picked = []
            for row, sample in enumerate(rows):
                first = sample.eligible.index(True)
                picked.append(states[row, first - 1, :])
            hidden.append(torch.stack(picked))
        return loss(hidden[0], hidden[1])

    def save(self, *, inject_crash_at=None):
        published = super().save(inject_crash_at=inject_crash_at)
        if self.replay_buffer is not None:
            self._replay_ledger.append(self.replay_buffer.ledger())
        return published

    def replay_history(self):
        return tuple(self._replay_ledger)


def _surface_seed(job):
    return int(digest([job["job_id"], "v716-surface-seed"]), 16) % (2 ** 32)


def _arm_rows(job, data, *, reserved_world_ids=frozenset()):
    """Build per-epoch sample tuples plus arm-specific trainer seams and ledger."""
    arm = job["arm"]
    epochs = job["epochs"]
    if arm == "H1-COADAPT-MASK":
        return (_epoch_rows(data, epochs),
                {"mask": CoAdaptedControllerMaskLoss(),
                 "state_specs": public_state_specs(data.cases)},
                {"mask_policy": CoAdaptedControllerMaskLoss.schema,
                 "note": "public-state grammar only; no target leakage"})
    if arm == "H2-ARK014-INVARIANCE":
        augmenter = Ark014SurfaceAugmenter(data.cases, seed=_surface_seed(job),
                                           epochs=epochs,
                                           reserved_world_ids=reserved_world_ids)
        rows = [_cases_to_rows(augmenter.epoch_cases(epoch), epoch)
                for epoch in range(1, epochs + 1)]
        return rows, {}, {"surface_audit": augmenter.audit()}
    if arm == "H2B-NORM":
        canonical = tuple(CanonicalRowNormalizer.supervision(case) for case in data.cases)
        rows = []
        for epoch in range(1, epochs + 1):
            examples = tuple(example for group in canonical for example in group)
            rows.append(tuple(_suffix(example.sample, epoch) for example in examples))
        return rows, {}, {"normalizer": CanonicalRowNormalizer.schema,
                          "targets_unchanged": True}
    if arm in ("H3-CONTRAST-MEM", "H3B-CONTRASTIVE"):
        generator = CounterfactualMemoryGenerator(seed=_surface_seed(job))
        rows, epoch_audits, pairs = [], {}, ()
        removed_pool = sorted((case for case in data.cases
                               if CounterfactualMemoryGenerator.eligible(case)),
                              key=lambda case: case.case_id)
        for epoch in range(1, epochs + 1):
            replacements, audit = generator.epoch_replacements(
                data.cases, epoch=epoch, reserved_world_ids=reserved_world_ids)
            cases = [replacements.get(case.case_id, case) for case in data.cases]
            twin_pairs = []
            if arm == "H3B-CONTRASTIVE":
                selected = set(replacements)
                removed_candidates = [case for case in removed_pool
                                      if case.case_id not in selected]
                twin_cases, removals = [], []
                for index, (_, variant) in enumerate(sorted(replacements.items())):
                    if index >= len(removed_candidates):
                        break
                    twin = replace(variant,
                                   case_id=digest([variant.case_id, "v716-h3b-twin"]),
                                   surface=replace(variant.surface, order="reverse"),
                                   panel="v716-h3b-twin")
                    twin_cases.append(twin)
                    removals.append(removed_candidates[index])
                    twin_pairs.append((epoch, f"{variant.case_id}/0@e{epoch}",
                                       f"{twin.case_id}/0@e{epoch}"))
                dropped = {case.case_id for case in removals}
                cases = [case for case in cases if case.case_id not in dropped] + twin_cases
                audit = {**audit, "twins": len(twin_cases), "removed": len(removals)}
            rows.append(_cases_to_rows(cases, epoch))
            epoch_audits[epoch] = audit
            pairs = pairs + tuple(twin_pairs)
        extra = {"epoch_audits": epoch_audits}
        if arm == "H3B-CONTRASTIVE":
            extra.update({"contrastive_weight": CONTRASTIVE_WEIGHT,
                          "contrastive_margin": CONTRASTIVE_MARGIN,
                          "contrastive_pairs": len(pairs)})
            return rows, {"contrastive_pairs": pairs}, extra
        return rows, {}, extra
    if arm in ("H4-OUTCOME-DELIB", "H5-POINTER-READ"):
        rows = _epoch_rows(data, epochs)
        seams, ledger = {}, {}
        if arm == "H5-POINTER-READ":
            profile = MODEL_PROFILES[job["profile"]]
            projected = profile.parameter_receipt().total + 2 * profile.width + 1
            if projected > CausalPointerTwoReadDecoder.PARAMETER_CEILING:
                raise ValueError("pointer head exceeds the preregistered parameter ceiling")
            seams["pointer"] = True
            ledger.update({"pointer_version": POINTER_ARCHITECTURE_SUFFIX,
                           "projected_parameters": projected,
                           "parameter_ceiling": CausalPointerTwoReadDecoder.PARAMETER_CEILING})
        return rows, seams, ledger
    raise ValueError(f"arm {arm} is not a mechanism-phase arm")


def create_trainer(data, job, *, source, root, device="cuda:0", resume=False,
                   cpu_canary=False, spec_override=None, updates_override=None,
                   parent_store=None, transition=None, reserved_world_ids=frozenset(),
                   stage_samples=None, replay_samples=None, rollout_provider=None):
    """Build one exposure-locked V7.16 trainer for a registered job."""
    if not isinstance(job, dict):
        raise ValueError("V7.16 jobs are dictionaries")
    phase = "continual" if job.get("mechanism") == "continual" else "mechanism"
    expected = CONTINUAL_JOB_KEYS if phase == "continual" else MECHANISM_JOB_KEYS
    if set(job) != expected:
        raise ValueError(f"V7.16 job keys must be exactly {sorted(expected)}")
    try:
        v716_protocol.validate(job, phase)
    except ValueError as error:
        raise ValueError(f"job outside the preregistered V7.16 matrix: {error}") from None
    if phase == "continual":
        return _continual_trainer(data, job, source=source, root=root, device=device,
                                  resume=resume, cpu_canary=cpu_canary,
                                  spec_override=spec_override,
                                  updates_override=updates_override,
                                  parent_store=parent_store, transition=transition,
                                  stage_samples=stage_samples, replay_samples=replay_samples)
    if parent_store is not None or transition is not None \
            or stage_samples is not None or replay_samples is not None:
        raise ValueError("V7.16 mechanism lineages are fresh; stage migration "
                         "arguments belong to the continual phase path")
    if data.split != "train" or data.manifest.get("scratch") or data.manifest.get("augmentation"):
        raise ValueError("V7.16 mechanism arms require untransformed training data")
    if device != "cuda:0" and not (device == "cpu" and cpu_canary is True):
        raise ValueError("V7.16 production requires isolated CUDA; no CPU fallback")
    if device != "cpu" and (cpu_canary or spec_override is not None or updates_override is not None):
        raise ValueError("overrides are reserved for bounded CPU canaries")
    if arm_requires_provider(job) and not callable(rollout_provider):
        raise ValueError("H4 requires an executed-rollout provider")
    spec = spec_override or MODEL_PROFILES[job["profile"]]
    rows, seams, ledger = _arm_rows(job, data, reserved_world_ids=reserved_world_ids)
    samples = tuple(sample for epoch_rows in rows for sample in epoch_rows)
    batch = min(16, len(rows[0]))
    total_updates = math.ceil(len(rows[0]) / batch) * job["epochs"]
    if device == "cpu":
        if len(rows[0]) > 128 or spec.parameter_receipt().total > 100_000:
            raise ValueError("CPU canary exceeds sample or parameter bound")
        if type(updates_override) is not int or not 1 <= updates_override <= 8:
            raise ValueError("CPU canary requires 1..8 updates")
        total_updates = updates_override
    if total_updates > 9216:
        raise ValueError("training plan exceeds the engine update bound")
    tokens = [sum(sum(sample.eligible) for sample in epoch_rows) for epoch_rows in rows]
    positions = [sum(len(sample.tokens) for sample in epoch_rows) for epoch_rows in rows]
    if len(set(tokens)) != 1:
        raise ValueError("per-epoch supervised-token parity violated before training")
    baseline_tokens = sum(sum(sample.eligible) for sample in data.samples)
    if tokens[0] != baseline_tokens:
        raise ValueError("arm exposure drifts from the blocked-CE baseline ledger")
    # Unpadded input positions are recorded per epoch. They are parity-exact
    # for H1/H2A/H2B/H4/H5 and may legitimately differ where the mechanism
    # changes rendered content (H3 counterfactual evidence); the ledger below
    # keeps every epoch visible either way.
    recipe = Recipe(seed=job["seed"], total_updates=total_updates, batch_size=batch,
                    micro_batch_size=min(2, batch), lr=3e-4, objective="ce",
                    sampling="uniform", schedule="cosine", precision="fp32",
                    activation_checkpointing=False)
    provenance = {**data.manifest, "v716_job": dict(job),
                  "v716_protocol_sha256": v716_protocol.PROTOCOL_SHA,
                  "v716_parent_dataset": data.identity,
                  "v716_supervision": {
                      "mechanism": job["mechanism"], "worlds": len(data.cases),
                      "samples_per_epoch": len(rows[0]),
                      "supervised_tokens_per_epoch": tokens[0],
                      "baseline_supervised_tokens_per_epoch": baseline_tokens,
                      "input_positions_per_epoch": positions,
                      "input_positions_parity": len(set(positions)) == 1,
                      "exposure_parity": "exact",
                      "ledger": ledger,
                      "qualification": "single preregistered mechanism factor; exposure locked"}}
    if device == "cpu":
        configure_cpu()
    else:
        configure_device(device)
    return V716Trainer(samples, spec=spec, recipe=recipe, source_commit=source,
                       root=Path(root), lineage=job["job_id"],
                       data_provenance=provenance,
                       tokenizer_manifest=UTF8.manifest.record(), device=device,
                       resume=resume, method="blocked", epoch_samples=rows,
                       state_specs=seams.get("state_specs"), mask=seams.get("mask"),
                       pointer=seams.get("pointer", False),
                       contrastive_pairs=seams.get("contrastive_pairs"),
                       replay_buffer=None, rollout_provider=rollout_provider)


def arm_requires_provider(job):
    return isinstance(job, dict) and job.get("arm") == "H4-OUTCOME-DELIB"


def _continual_recipe(job, stage_samples):
    """Registered continual recipe; shared by the trainer and the transition builder."""
    batch = min(16, len(stage_samples))
    total_updates = math.ceil(len(stage_samples) / batch) * job["epochs"]
    recipe = Recipe(seed=job["seed"], total_updates=total_updates, batch_size=batch,
                    micro_batch_size=min(2, batch), lr=3e-4, objective="ce",
                    sampling="uniform", schedule="cosine", precision="fp32",
                    activation_checkpointing=False)
    return batch, total_updates, recipe


def _continual_provenance(data, job, stage_samples, replay_samples):
    """Replay-ledger provenance; deterministic, so the transition binds it too."""
    buffer = Ark017SurpriseReplayBuffer(tuple(replay_samples), fraction=REPLAY_FRACTION)
    provenance = {**data.manifest, "v716_job": dict(job),
                  "v716_protocol_sha256": v716_protocol.PROTOCOL_SHA,
                  "v716_stage_samples": len(stage_samples),
                  "v716_replay_pool": len(replay_samples),
                  "v716_replay_fraction": REPLAY_FRACTION,
                  "v716_supervision": {
                      "mechanism": "continual", "stage": job["stage"],
                      "updates_fixed_to_baseline": True,
                      "ledger": buffer.ledger(),
                      "qualification": "donor replay swaps samples inside registered batches; "
                                       "extra Skill-A exposure is ledgered, never hidden"}}
    return provenance, buffer


def _continual_trainer(data, job, *, source, root, device, resume, cpu_canary,
                       spec_override, updates_override, parent_store, transition,
                       stage_samples, replay_samples):
    stage = job["stage"]
    if stage not in CONTINUAL_STAGES:
        raise ValueError("unregistered V7.16 continual stage")
    if stage in ("b-only", "replay", "fresh-b"):
        raise NotImplementedError(
            "b-only/replay/fresh-b are the registered v7.15 continual control arms; "
            "run them through the qualified continual phase path unchanged")
    if resume:
        raise ValueError("ark017-sparse replay-buffer state is not yet checkpoint-qualified; "
                         "fresh lineages only in V7.16 v1")
    if stage_samples is None or replay_samples is None or parent_store is None \
            or transition is None:
        raise ValueError("ark017-sparse requires stage samples, a replay pool, a verified "
                         "parent store and an explicit stage transition")
    if not callable(getattr(transition, "validate", None)):
        raise ValueError("stage transition must be a StageTransition")
    if device != "cuda:0" and not (device == "cpu" and cpu_canary is True):
        raise ValueError("V7.16 production requires isolated CUDA; no CPU fallback")
    spec = spec_override or MODEL_PROFILES[job["profile"]]
    stage_samples = tuple(stage_samples)
    replay_samples = tuple(replay_samples)
    if not stage_samples or not replay_samples:
        raise ValueError("empty stage or replay samples")
    batch, total_updates, recipe = _continual_recipe(job, stage_samples)
    if device == "cpu":
        if len(stage_samples) > 128 or spec.parameter_receipt().total > 100_000:
            raise ValueError("CPU canary exceeds sample or parameter bound")
        if type(updates_override) is not int or not 1 <= updates_override <= 8:
            raise ValueError("CPU canary requires 1..8 updates")
        total_updates = updates_override
        recipe = replace(recipe, total_updates=total_updates)
    provenance, buffer = _continual_provenance(data, job, stage_samples, replay_samples)
    if device == "cpu":
        configure_cpu()
    else:
        configure_device(device)
    return V716Trainer(stage_samples, spec=spec, recipe=recipe, source_commit=source,
                       root=Path(root), lineage=job["job_id"],
                       data_provenance=provenance,
                       tokenizer_manifest=UTF8.manifest.record(), device=device,
                       method="blocked", epoch_samples=None,
                       transition=transition, parent_store=parent_store,
                       replay_buffer=buffer)
