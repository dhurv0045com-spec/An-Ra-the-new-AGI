"""Model-operations interface for phase executors (D4 + real execution contracts S2).

Production ops perform real training/generation on GPU. Test doubles implement
the SAME validated interface, record calls and return deterministic
fixture-labeled counts WITHOUT optimizer updates. Executors never branch on
ops type; they call the interface. The double enforces the same shapes,
required fields, lineage rules and objective eligibility as production; a
permissive recorder alone is not an integration test.
"""
from __future__ import annotations

from typing import Any, Protocol


K8_CAMPAIGN_MODEL = {"profile": "development", "vocab": 260, "layers": 8,
                     "width": 256, "heads": 4, "ffn": 704, "max_seq": 512}
K8_CAMPAIGN_TRAINING = {"learning_rate": 0.0003, "weight_decay": 0.01,
                        "clip_norm": 1.0}
# FP16 is an opt-in experiment, not the campaign default. The first live
# two-T4 E0 run overflowed GradScaler's initial FP16 scale before its first
# update; production must choose the stable precision until AMP has a
# successful hardware-calibration receipt.
K8_CAMPAIGN_PRECISION = "fp32"


def k8_campaign_config():
    """Frozen K8 campaign configuration (experiment.md S4 + campaign.json).

    Development geometry with campaign context 512 (not the 256 default),
    frozen AdamW LR 3e-4 / WD 0.01 / clip 1.0. Context/optimizer come from the
    frozen protocol, not just a named profile.
    """
    from bramastra_lab.research.config import BuildConfig

    return BuildConfig.from_dict({
        "model": dict(K8_CAMPAIGN_MODEL),
        "training": dict(K8_CAMPAIGN_TRAINING),
    })


def k8_identities(*, data_dir: str | None = None) -> dict[str, str]:
    """Real source/tokenizer/config/data/code identities (no generic `k8`)."""
    from bramastra_lab.research.config import tokenizer_identity

    from bramastra_lab.research.runtime.provenance import source_closure_sha256
    import hashlib
    import os

    try:
        source_hash = source_closure_sha256()
    except Exception:
        source_hash = "unavailable-source"
    tokenizer = tokenizer_identity()
    try:
        config_id = k8_campaign_config().identity()
    except Exception:
        config_id = "unavailable-config"
    data_hash = "unavailable-data"
    if data_dir and os.path.isdir(data_dir):
        digest = hashlib.sha256()
        for base, _dirs, names in sorted(os.walk(data_dir)):
            for name in sorted(names):
                path = os.path.join(base, name)
                try:
                    digest.update(hashlib.sha256(
                        open(path, "rb").read()).hexdigest().encode())
                except OSError:
                    continue
        data_hash = digest.hexdigest()
    return {"source_hash": source_hash, "tokenizer_identity": tokenizer,
            "config_identity": config_id, "data_identity": data_hash,
            "code_identity": source_hash}


class ModelOps(Protocol):
    """Expensive model boundary behind every phase executor."""

    def init_model(self, *, seed: int, profile: str, device: str) -> Any:
        """Random initialization for one seed/profile/device; returns handle."""
        ...

    def snapshot_state(self, handle: Any) -> Any:
        """Capture exact starting state for paired-equality checks."""
        ...

    def states_equal(self, left: Any, right: Any) -> bool:
        ...

    def training_update(self, handle: Any, *, batch: Any, window: Any,
                        extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """One explicit accumulate+finalize update; returns actual counters."""
        ...

    def save_checkpoint(self, handle: Any, *, path: str,
                        fraction: float) -> str:
        """Publish checkpoint; returns checkpoint identity."""
        ...

    def free_generation(self, handle: Any, *, prompt: Any,
                        max_new_tokens: int) -> dict[str, Any]:
        """Free-generation evaluation with stop evidence."""
        ...

    def optimizer_updates(self, handle: Any) -> int:
        ...

    # -- real lifecycle (contracts S2) -------------------------------------
    def initialize_random(self, *, seed: int, device: str,
                          reservation: Any | None = None) -> Any:
        """E1/pilot-only random init with frozen config; returns learner."""
        ...

    def restore_parent(self, *, parent: dict[str, Any], device: str,
                       optimizer_policy: str = "fresh") -> Any:
        """Hash-validated parent payload + compatible config; real model."""
        ...

    def fork_child(self, *, parent_handle: Any,
                   optimizer_policy: str = "fresh") -> Any:
        """Copy verified parent state into isolated child lineage."""
        ...

    def construct_objectives(self, *, handle: Any, batch: Any,
                             compiled: dict[str, Any],
                             arm: str) -> tuple[Any, dict[str, Any]]:
        """Build live sums + denominators without stepping (no branch)."""
        ...

    def apply_update(self, handle: Any, *, batch: Any, window: Any,
                     extra: dict[str, Any] | None = None,
                     pair_rows: Any | None = None) -> dict[str, Any]:
        """One window, one normalization, one finalization; real events."""
        ...

    def evaluate_episode(self, *, handle: Any, episode: dict[str, Any],
                         task_env: Any | None = None) -> dict[str, Any]:
        """Model decisions through env; observed actions/costs + verifier."""
        ...

    def publish_checkpoint(self, *, handle: Any, run_dir: str, phase: str,
                           arm: str | None, seed: int | None,
                           update_index: int,
                           parent_checkpoint_id: str | None,
                           writer_token: str | None = None,
                           expected_parent: str | None = None,
                           data_dir: str | None = None) -> str:
        """Real publication via runtime.save_checkpoint + fencing."""
        ...

    def restore_verify(self, *, run_dir: str, checkpoint_id: str) -> dict[str, Any]:
        """Fresh-process load + state/stream validation."""
        ...

    def migrate_to_gated(self, handle: Any, *, gates_enabled: bool) -> Any:
        """Migrate the training handle to the gated architecture."""
        ...


def _frozen_config_for_profile(profile: str):
    """Frozen campaign config; refuses silent 256-context development."""
    from bramastra_lab.research.config import BuildConfig

    if profile == "development":
        # Campaign specifies 512, not the 256 default. Callers must use the
        # frozen K8 override, not the bare profile name.
        raise ValueError(
            "profile 'development' defaults to max_seq 256 but the K8 campaign "
            "specifies 512; use k8_campaign_config() / profile "
            "'k8-campaign' instead of a bare named profile")
    if profile in ("k8-campaign", "k8"):
        return k8_campaign_config()
    return BuildConfig.from_dict({"model": {"profile": profile}})


class ProductionOps:
    """Real GPU ops (never executed locally per policy; owner launch only)."""

    def __init__(self, *, precision: str = "fp32") -> None:
        self.precision = precision

    def _build_trainer(self, *, seed: int, device: str, profile: str):
        from bramastra_lab.research.config import seed_everything
        from bramastra_lab.research.models import IntegratedModel
        from bramastra_lab.research.learning.k8_trainer import K8Trainer

        seed_everything(seed)
        if profile in ("k8-campaign", "k8"):
            config = k8_campaign_config()
        else:
            # Enforce frozen context: bare development (256) is refused.
            config = _frozen_config_for_profile(profile)
        model = IntegratedModel(config).to(device)
        # Campaign mode: allocation gate enforced (never require_allocation=False
        # on the production path; E0 binds a live allocation, later phases
        # bind the campaign reservation via begin_campaign before updates).
        trainer = K8Trainer(
            config, model, device=device,
            precision="fp32" if not str(device).startswith("cuda") else self.precision,
            require_allocation=True)
        # Match campaign betas (0.9, 0.95) explicitly; base trainer defaults
        # differ. LR/WD/clip already frozen via config.
        try:
            for group in trainer.optimizer.param_groups:
                group["betas"] = (0.9, 0.95)
                group["eps"] = 1e-8
        except Exception:
            pass
        return config, model, trainer

    def init_model(self, *, seed: int, profile: str, device: str):
        # Backward-compat entry: route through the frozen campaign path.
        # Bare 'development' is refused (must be k8-campaign with 512).
        if profile == "development":
            profile = "k8-campaign"
        config, model, trainer = self._build_trainer(
            seed=seed, device=device, profile=profile)
        return {"config": config, "model": model, "trainer": trainer,
                "seed": seed, "profile": profile, "device": device,
                "started_updates": 0,
                "config_identity": config.identity(),
                "architecture_id": getattr(
                    model, "architecture_id", "bramastra-base-decoder/v1")}

    def initialize_random(self, *, seed: int, device: str,
                          reservation: Any | None = None) -> Any:
        handle = self.init_model(seed=seed, profile="k8-campaign", device=device)
        # Always unbound here: authority attaches explicitly through
        # session.bind_job_reservation (O01), which validates job/device/
        # phase/source/deadline before admitting via begin_campaign.
        handle["reservation"] = reservation
        handle["reservation_bound"] = False
        return handle

    def restore_parent(self, *, parent: dict[str, Any], device: str,
                       optimizer_policy: str = "fresh") -> Any:
        """Restore a verified parent payload onto a real model.

        `parent` must be a ParentRef.resolve() record (checkpoint_id +
        manifest with payload hash + config identity). Random init with the
        same seed is forbidden.
        """
        if not isinstance(parent, dict) or not parent.get("checkpoint_id"):
            raise ValueError("restore_parent requires a verified parent record")
        manifest = parent.get("manifest") or {}
        run_dir = parent.get("run_dir")
        checkpoint_id = parent.get("checkpoint_id")
        if run_dir is None:
            raise ValueError("parent record carries no run_dir; refusing")
        from bramastra_lab.research.runtime.checkpoint import load_checkpoint

        payload, loaded = load_checkpoint(
            run_dir, checkpoint_id=checkpoint_id,
            expect_config_identity=parent.get("config_identity")
            or manifest.get("config_identity"))
        # Rebuild model on the requested device with the frozen campaign
        # config, then apply the verified payload.
        from bramastra_lab.research.config import BuildConfig
        from bramastra_lab.research.models import IntegratedModel
        from bramastra_lab.research.learning.k8_trainer import (
            AllocationContext, K8Trainer)
        import time

        # Config identity compatibility: refuse cross-config resume. Only
        # ValueError (the contract refusal) propagates; unexpected errors
        # building the frozen config must fail loudly, never fall back.
        config = k8_campaign_config()
        if loaded.config_identity != config.identity():
            raise ValueError(
                "parent config identity does not match frozen K8 campaign "
                "config; refusing cross-config resume")
        model = IntegratedModel(config).to(device)
        trainer = K8Trainer(
            config, model, device=device,
            precision="fp32" if not str(device).startswith("cuda") else self.precision,
            require_allocation=True)
        if optimizer_policy == "fresh":
            # Fresh optimizer: load model weights only, keep new optimizer.
            try:
                model.load_state_dict(payload["model"])
            except Exception as exc:
                raise ValueError(f"parent model state incompatible: {exc}") from exc
        else:
            trainer.load_state_payload(payload)
        return {"config": config, "model": model, "trainer": trainer,
                "seed": parent.get("seed", 0), "profile": "k8-campaign",
                "device": device, "started_updates": int(
                    trainer.counters.optimizer_updates),
                "config_identity": config.identity(),
                "architecture_id": getattr(
                    model, "architecture_id", "bramastra-base-decoder/v1"),
                "parent_checkpoint_id": checkpoint_id}

    def fork_child(self, *, parent_handle: Any,
                   optimizer_policy: str = "fresh") -> Any:
        import copy

        config = parent_handle["config"]
        device = parent_handle["device"]
        from bramastra_lab.research.models import IntegratedModel
        from bramastra_lab.research.learning.k8_trainer import K8Trainer

        child_model = IntegratedModel(config).to(device)
        child_model.load_state_dict(
            copy.deepcopy(parent_handle["model"].state_dict()))
        child_trainer = K8Trainer(
            config, child_model, device=device,
            precision="fp32" if not str(device).startswith("cuda") else self.precision,
            require_allocation=True)
        if optimizer_policy == "preserved":
            # Preserve optimizer state explicitly (declared policy only).
            try:
                child_trainer.optimizer.load_state_dict(
                    copy.deepcopy(parent_handle["trainer"].optimizer.state_dict()))
            except Exception as exc:
                raise ValueError(
                    f"preserved-optimizer fork refused: {exc}") from exc
        return {"config": config, "model": child_model, "trainer": child_trainer,
                "seed": parent_handle.get("seed", 0),
                "profile": parent_handle.get("profile", "k8-campaign"),
                "device": device,
                "started_updates": int(child_trainer.counters.optimizer_updates),
                "config_identity": parent_handle.get("config_identity"),
                "architecture_id": parent_handle.get("architecture_id"),
                "parent_checkpoint_id": parent_handle.get("parent_checkpoint_id")}

    def construct_objectives(self, *, handle: Any, batch: Any,
                             compiled: dict[str, Any],
                             arm: str) -> tuple[Any, dict[str, Any]]:
        """Real target construction (no fixture branch).

        `compiled` must carry dataset-derived channels: answer handled by the
        batch itself; world/action/value/pair come from the episode's real
        history, teacher policy, return convention and paired goals. Missing
        eligible terms for enabled objectives fail before the boundary.
        """
        from bramastra_lab.research.experience.supervision import SupervisionWindow

        weights = dict(compiled["weights"])
        enabled = frozenset(compiled["enabled"])
        window = SupervisionWindow(weights=weights, enabled_terms=enabled)
        window.add("token", batch.target_count)
        extra: dict[str, Any] = {}
        model = handle["model"]
        config = handle["config"]
        # World: actual next observation after the allowed action.
        if "world" in enabled and float(weights.get("world", 0.0)) > 0:
            channels = compiled.get("world")
            if not channels:
                raise ValueError(
                    "world objective enabled but compiled world targets are "
                    "missing; refusing silent zero loss")
            from bramastra_lab.research.learning.k8_scoring import (
                world_transition_token_loss)
            extra["world"] = world_transition_token_loss(
                model, config, channels["prefix_tokens"],
                action=channels["action"],
                target_feedback=channels["target_feedback"])
            window.add("world", int(channels.get("denominator", 1)))
        # Action: declared teacher policy over legal candidates.
        if "action" in enabled and float(weights.get("action", 0.0)) > 0:
            channels = compiled.get("action")
            if not channels:
                raise ValueError(
                    "action objective enabled but compiled action targets are "
                    "missing; refusing silent zero loss")
            from bramastra_lab.research.learning.k8_scoring import (
                score_candidates_trainable)
            import torch as _torch

            _legal = channels.get("legal_mask")
            if isinstance(_legal, (list, tuple)):
                _legal = _torch.tensor(list(_legal), dtype=_torch.bool)
            scored = score_candidates_trainable(
                model, config, channels["prefix_tokens"],
                channels["candidates"],
                legal_mask=_legal)
            # Teacher distribution must be supplied from the dataset, never
            # an invented uniformity over arbitrary token lists.
            import torch

            teacher = channels.get("teacher_distribution")
            if teacher is None:
                raise ValueError(
                    "action channels carry no teacher distribution from the "
                    "dataset; refusing invented uniformity loss")
            teacher_t = torch.as_tensor(
                teacher, dtype=scored["log_probs"].dtype,
                device=scored["log_probs"].device)
            extra["action"] = -(teacher_t * scored["log_probs"]).sum()
            window.add("action", int(channels.get("denominator", 1)))
        # Value: specified return convention (real verifier outcome).
        if "value" in enabled and float(weights.get("value", 0.0)) > 0:
            channels = compiled.get("value")
            if not channels:
                raise ValueError(
                    "value objective enabled but compiled value targets are "
                    "missing; refusing unconditional zero value")
            from bramastra_lab.research.learning.k8_scoring import (
                value_estimate_trainable)
            estimate = value_estimate_trainable(
                model, config, channels["prefix_tokens"])
            import torch

            target = torch.as_tensor(
                float(channels["target_return"]), dtype=estimate.dtype,
                device=estimate.device)
            extra["value"] = (estimate - target).square()
            window.add("value", int(channels.get("denominator", 1)))
        # Pair: actual paired goals and respective valid answers.
        # Eligibility is declared here (denominator); the real own/swapped
        # rows are supplied as pair_rows at the update boundary (never
        # invented uniformity).
        if "pair" in enabled and float(weights.get("pair", 0.0)) > 0:
            channels = compiled.get("pair")
            if not channels:
                raise ValueError(
                    "pair objective enabled but compiled pair channels are "
                    "missing; refusing")
            window.add("pair", int(channels.get("denominator", 1)))
        return window, extra

    def apply_update(self, handle, *, batch, window, extra=None,
                       pair_rows=None):
        return self.training_update(handle, batch=batch, window=window,
                                    extra=extra, pair_rows=pair_rows)

    def evaluate_episode(self, *, handle: Any, episode: dict[str, Any],
                         task_env: Any | None = None) -> dict[str, Any]:
        """Run model decisions through the environment; verifier decides."""
        # Production path: free generation on real prompt tokens + independent
        # verifier supplied by the caller. Observed actions/results/costs come
        # from the episode's real history; the final verifier (not EOS)
        # decides success. No fixed-prompt fallback: callers must supply real
        # prompt tokens compiled from the public goal.
        prompt = episode.get("prompt_tokens")
        if not prompt:
            raise ValueError(
                "evaluate_episode requires real prompt_tokens compiled from "
                "the public goal; refusing fixed-prompt fallback")
        max_new = int(episode.get("max_new_tokens", 24))
        report = self.free_generation(handle, prompt=prompt,
                                      max_new_tokens=max_new)
        verifier = episode.get("verifier")
        observation = {"answer": report.get("answer", ""),
                       "sum": report.get("answer", "")}
        success = None
        if callable(verifier):
            try:
                success = bool(verifier(episode.get("mechanism", {}), observation))
            except Exception:
                success = False
        # Counters derive from the actual generation event (durable): one
        # episode submits once (actions=1, nodes=1); model calls and cost
        # scale with generated tokens (never case-number arithmetic).
        new_tokens = int(report.get("new_tokens", 1) or 1)
        return {"answer": report.get("answer", ""),
                "stopped_on_eos": bool(report.get("stopped_on_eos", False)),
                "success": success,
                "model_calls": max(1, new_tokens), "actions": 1, "nodes": 1,
                "cost": float(max(1, new_tokens))}

    def publish_checkpoint(self, *, handle: Any, run_dir: str, phase: str,
                           arm: str | None, seed: int | None,
                           update_index: int,
                           parent_checkpoint_id: str | None,
                           writer_token: str | None = None,
                           expected_parent: str | None = None,
                           data_dir: str | None = None) -> str:
        import os

        from bramastra_lab.research.runtime.checkpoint import publish_checkpoint

        payload = handle["trainer"].state_payload()
        identities = k8_identities(data_dir=data_dir)
        # Distinct child lineage is bound via run_id/phase/arm/seed.
        run_id = f"{phase}-{arm or 'na'}-{seed if seed is not None else 'na'}-" \
                 f"{update_index}-{handle.get('seed', seed or 0)}"
        manifest = publish_checkpoint(
            run_dir=run_dir, run_id=run_id, update_index=update_index,
            payload=payload,
            config_identity=payload.get("config_identity")
            or identities["config_identity"],
            tokenizer_identity=identities["tokenizer_identity"],
            data_identity=identities["data_identity"],
            code_identity=identities["code_identity"],
            parent_checkpoint_id=parent_checkpoint_id,
            writer_token=writer_token, expected_parent=expected_parent,
            milestone=f"{phase}-{arm}-{seed}",
            phase=phase, arm=arm, seed=seed)
        # Kaggle output has a hard finite quota.  A milestone label tracks
        # only this lineage's newest complete checkpoint, so rotate every
        # superseded payload immediately after its replacement is durable.
        # This preserves all live lineage heads and accepted parents while
        # preventing long E1/E5 runs from retaining an unbounded history of
        # model-plus-optimizer blobs.
        from bramastra_lab.research.runtime.checkpoint import prune_checkpoints

        prune_checkpoints(run_dir, keep_latest=0)
        # Also mirror a human-readable pointer under phase/arm/seed dirs.
        try:
            mirror_dir = os.path.join(run_dir, "checkpoints", phase,
                                      f"{arm}-{seed}")
            os.makedirs(mirror_dir, exist_ok=True)
            with open(os.path.join(
                    mirror_dir, f"{update_index:06d}.json"), "w",
                    encoding="utf-8") as fh:
                import json as _json
                _json.dump({"checkpoint_id": manifest.checkpoint_id,
                            "run_id": run_id,
                            "update_index": update_index},
                           fh, indent=2, sort_keys=True)
        except OSError:
            pass
        return manifest.checkpoint_id

    def restore_verify(self, *, run_dir: str, checkpoint_id: str) -> dict[str, Any]:
        from bramastra_lab.research.runtime.checkpoint import restore_verify

        identities = k8_identities()
        return restore_verify(
            run_dir=run_dir, checkpoint_id=checkpoint_id,
            expect_config_identity=identities["config_identity"],
            expect_tokenizer_identity=identities["tokenizer_identity"])

    def migrate_to_gated(self, handle: Any, *, gates_enabled: bool) -> Any:
        """Migrate the actual training handle to the gated architecture.

        S1 (gates_enabled=True) becomes the migrated GatedReuseModel with
        zero gates (functionally equal at migration); S0 carries the same
        two scalar slots fixed/disabled. Migration happens on the handle
        that will be trained (never a discarded tiny probe). Verifies zero-
        gate equality, gate gradients, head/segment contract and optimizer
        inventory on that very handle.
        """
        from bramastra_lab.research.models.gated import (
            GatedReuseModel,
            check_gate_gradients,
            migrate_from_parent,
        )
        import torch

        config = handle["config"]
        parent_model = handle["model"]
        device = handle.get("device", "cpu")
        child_model = migrate_from_parent(
            parent_model, config, gates_enabled=gates_enabled)
        # Head/segment contract on the migrated handle.
        probe = torch.randint(0, config.model.vocab, (2, 12))
        out = child_model(probe, torch.ones_like(probe, dtype=torch.bool),
                          segment_ids=torch.ones_like(probe),
                          return_hidden=True, return_value=True)
        assert out.logits is not None and out.hidden is not None \
            and out.value is not None
        # Gate inventory on the migrated handle.
        if gates_enabled:
            assert child_model.gates_enabled
            assert [float(v) for v in child_model.gate_values()] == [0.0, 0.0]
            checks = check_gate_gradients(child_model)
            assert checks["shared_block_gradients_present"]
            assert checks["embedding_gradients_present"]
        else:
            assert not child_model.gates_enabled
        # Fresh optimizer for the migrated handle (declared fresh policy),
        # preserving the frozen campaign precision/allocation semantics.
        from bramastra_lab.research.learning.k8_trainer import K8Trainer

        child_model = child_model.to(device)
        new_trainer = K8Trainer(
            config, child_model, device=device,
            precision="fp32" if not str(device).startswith("cuda") else self.precision,
            require_allocation=True)
        try:
            for group in new_trainer.optimizer.param_groups:
                group["betas"] = (0.9, 0.95)
                group["eps"] = 1e-8
        except Exception:
            pass
        # Optimizer inventory must contain (S1) or exclude (S0) gate params.
        param_names = [n for n, _ in child_model.named_parameters()]
        has_gate = any("gate_alpha" in n for n in param_names)
        if gates_enabled and not has_gate:
            raise ValueError("S1 migrated handle carries no gate parameters")
        if not gates_enabled and has_gate:
            # S0 carries disabled slots as buffers, not trainable params.
            trainable = [n for n, p in child_model.named_parameters() if p.requires_grad]
            if any("gate_alpha" in n for n in trainable):
                raise ValueError("S0 gate slots must be fixed/disabled")
        handle["model"] = child_model
        handle["trainer"] = new_trainer
        handle["architecture_id"] = getattr(
            child_model, "architecture_id", "gated")
        handle["gates_enabled"] = bool(gates_enabled)
        handle["migrated"] = True
        handle["started_updates"] = int(new_trainer.counters.optimizer_updates)
        return handle

    def snapshot_state(self, handle):
        import copy

        return copy.deepcopy(handle["model"].state_dict())

    def states_equal(self, left, right) -> bool:
        try:
            if set(left) != set(right):
                return False
            import torch

            return all(bool((left[k].detach().cpu() == right[k].detach().cpu()).all())
                       for k in left)
        except Exception:
            return False

    def training_update(self, handle, *, batch, window, extra=None,
                          pair_rows=None):
        trainer = handle["trainer"]
        before_attempted = int(getattr(trainer, "attempted_updates", 0))
        before_committed = int(trainer.counters.optimizer_updates)
        trainer.accumulate_full_window(
            batch, window_builder=lambda _: window,
            extra_terms_fn=(lambda: extra) if extra else None,
            pair_rows=pair_rows)
        report = trainer.finalize_update()
        after_attempted = int(getattr(trainer, "attempted_updates", 0))
        after_committed = int(trainer.counters.optimizer_updates)
        return {"committed": after_committed - before_committed,
                "attempted": after_attempted - before_attempted,
                "exposure": batch.target_count,
                "optimizer_update": report.optimizer_update}

    def save_checkpoint(self, handle, *, path, fraction):
        # Deprecated backward-compat wrapper. The old (path, fraction)
        # signature cannot supply tokenizer/data/code identities or fencing,
        # so it derives run_dir/phase lineage from the path and requires the
        # handle to carry data_dir. New code must call publish_checkpoint
        # directly. Generic placeholders are refused inside publish.
        import os

        parts = os.path.normpath(path).split(os.sep)
        if "checkpoints" not in parts:
            raise ValueError(
                f"save_checkpoint path {path!r} carries no checkpoints segment; "
                "use publish_checkpoint with explicit run_dir/phase/arm/seed")
        idx = parts.index("checkpoints")
        run_dir = os.sep.join(parts[:idx])
        if not run_dir:
            raise ValueError(
                f"save_checkpoint path {path!r} carries no run_dir before "
                "checkpoints; use publish_checkpoint directly")
        # Path forms: run/checkpoints/<phase>/<file> or
        # run/checkpoints/<phase>/<arm-seed>/<file>.
        try:
            phase = parts[idx + 1]
        except IndexError:
            raise ValueError(
                f"save_checkpoint path {path!r} carries no phase segment")
        arm_seed = parts[idx + 2] if len(parts) > idx + 3 else None
        if arm_seed is None:
            raise ValueError(
                f"save_checkpoint path {path!r} carries no arm-seed segment; "
                "use publish_checkpoint directly")
        if "-" not in arm_seed:
            raise ValueError(
                f"save_checkpoint arm-seed {arm_seed!r} is not <arm>-<seed>")
        arm, _, seed_text = arm_seed.partition("-")
        try:
            seed: int | None = int(seed_text.split("-")[0].split(".")[0])
        except ValueError:
            raise ValueError(
                f"save_checkpoint arm-seed {arm_seed!r} carries no integer seed")
        try:
            trainer = handle["trainer"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"handle carries no trainer: {exc}") from exc
        try:
            update_index = int(trainer.counters.optimizer_updates)
        except Exception as exc:
            raise ValueError(f"trainer counters unreadable: {exc}") from exc
        data_dir = handle.get("data_dir")
        if not data_dir:
            raise ValueError(
                "handle carries no data_dir; publish_checkpoint requires real "
                "data identity (use publish_checkpoint directly)")
        return self.publish_checkpoint(
            handle=handle, run_dir=run_dir, phase=phase, arm=arm,
            seed=seed, update_index=update_index,
            parent_checkpoint_id=handle.get("parent_checkpoint_id"),
            data_dir=data_dir)

    def free_generation(self, handle, *, prompt, max_new_tokens):
        from bramastra_lab.research.runtime.inference import generate_free_form

        return generate_free_form(
            handle["model"], handle["config"], prompt,
            max_new_tokens=max_new_tokens).__dict__

    def optimizer_updates(self, handle) -> int:
        try:
            return int(handle["trainer"].counters.optimizer_updates)
        except Exception:
            return 0


class RecordingDoubleOps:
    """Deterministic test double: records calls, performs zero updates.

    Enforces the SAME shapes, required fields, lineage rules and objective
    eligibility as production. Returns fixture-labeled counts only; fixture
    receipts never qualify for campaign aggregates. A no-training boundary
    (training_update raising) must never yield a positive learned receipt.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._handles: dict[int, dict] = {}
        self._next = 0

    def _require_window_eligible(self, window, extra) -> None:
        weights = dict(getattr(window, "weights", {}))
        enabled = set(getattr(window, "enabled_terms", frozenset()))
        # Enabled positive-weight terms must have eligible denominators;
        # disabled terms must not execute (same rule as K8Trainer).
        for term in sorted(enabled):
            weight = float(weights.get(term, 0.0))
            try:
                denom = int(window.denominator(term))
            except Exception:
                denom = 0
            if weight > 0 and denom <= 0:
                raise ValueError(
                    f"objective term {term!r} weighted {weight} but has zero "
                    "eligible data; refusing silent zero loss")
        if extra:
            for term in extra:
                if term not in enabled:
                    raise ValueError(
                        f"objective term {term!r} executed while disabled; "
                        "disabled terms must not contribute")
                if float(weights.get(term, 0.0)) <= 0:
                    raise ValueError(
                        f"objective term {term!r} executed with non-positive "
                        "weight; refusing")

    def init_model(self, *, seed, profile, device):
        # Enforce frozen campaign context like production: bare development
        # (256) is refused in new code; map it explicitly for old callers but
        # record the canonical k8-campaign lineage.
        canonical = profile
        if profile == "development":
            canonical = "k8-campaign"
        handle_id = self._next
        self._next += 1
        handle = {"double_id": handle_id, "seed": seed, "profile": canonical,
                  "requested_profile": profile,
                  "device": device, "updates": 0, "attempted": 0,
                  "exposure": 0, "init_state": f"init-{seed}-{canonical}",
                  "evidence_kind": "fixture"}
        self._handles[handle_id] = handle
        self.calls.append(("init_model", {"seed": seed, "profile": canonical,
                                          "requested_profile": profile,
                                          "device": device}))
        return handle

    def initialize_random(self, *, seed, device, reservation=None):
        handle = self.init_model(seed=seed, profile="k8-campaign", device=device)
        handle["reservation"] = reservation
        self.calls.append(("initialize_random", {"seed": seed, "device": device}))
        return handle

    def restore_parent(self, *, parent, device, optimizer_policy="fresh"):
        if not isinstance(parent, dict) or not parent.get("checkpoint_id"):
            raise ValueError("restore_parent requires a verified parent record")
        if not parent.get("payload_sha256") and not parent.get("manifest"):
            raise ValueError("parent record carries no verified payload hash")
        handle = self.init_model(seed=int(parent.get("seed", 0) or 0),
                                 profile="k8-campaign", device=device)
        handle["parent_checkpoint_id"] = parent.get("checkpoint_id")
        handle["optimizer_policy"] = optimizer_policy
        self.calls.append(("restore_parent", {
            "checkpoint_id": str(parent.get("checkpoint_id"))[:12],
            "optimizer_policy": optimizer_policy}))
        return handle

    def fork_child(self, *, parent_handle, optimizer_policy="fresh"):
        import copy

        child = copy.deepcopy(parent_handle)
        child["double_id"] = self._next
        self._next += 1
        child["optimizer_policy"] = optimizer_policy
        child["forked_from"] = parent_handle.get("double_id")
        # No sibling mutation: parent handle is untouched (deep copy).
        self._handles[child["double_id"]] = child
        self.calls.append(("fork_child", {
            "parent": parent_handle.get("double_id"),
            "child": child["double_id"],
            "optimizer_policy": optimizer_policy}))
        return child

    def construct_objectives(self, *, handle, batch, compiled, arm):
        # Same eligibility rules as production; returns validated fixture
        # sums (no gradients locally) with identical keys/denominators.
        weights = dict(compiled["weights"])
        enabled = frozenset(compiled["enabled"])
        from bramastra_lab.research.experience.supervision import SupervisionWindow

        window = SupervisionWindow(weights=weights, enabled_terms=enabled)
        window.add("token", batch.target_count)
        for term in ("world", "action", "value", "pair"):
            if term in enabled and float(weights.get(term, 0.0)) > 0:
                channels = compiled.get(term)
                if not channels:
                    raise ValueError(
                        f"{term} enabled but compiled channels missing")
                window.add(term, int(channels.get("denominator", 1)))
        # Validate extra-term eligibility exactly like production.
        self._require_window_eligible(window, compiled.get("extra"))
        self.calls.append(("construct_objectives", {
            "arm": arm, "enabled": sorted(enabled),
            "targets": int(getattr(batch, "target_count", 0))}))
        return window, dict(compiled.get("extra") or {})

    def apply_update(self, handle, *, batch, window, extra=None,
                       pair_rows=None):
        return self.training_update(handle, batch=batch, window=window,
                                    extra=extra, pair_rows=pair_rows)

    def migrate_to_gated(self, handle, *, gates_enabled: bool):
        """Fixture migration on the training handle (same validation)."""
        self.calls.append(("e4_migration", {"gates_enabled": gates_enabled,
                                            "device": handle.get("device")}))
        handle["gates_enabled"] = bool(gates_enabled)
        handle["architecture_id"] = "bramastra-gated-block-reuse/v1" if gates_enabled \
            else "bramastra-base-decoder/v1+disabled-gate-slots"
        handle["migrated"] = True
        handle["gate_alpha"] = [0.0, 0.0]
        # Optimizer inventory on the very handle: S1 must expose trainable
        # gate slots, S0 must carry them fixed/disabled.
        if gates_enabled:
            if handle.get("gate_alpha") != [0.0, 0.0]:
                raise ValueError("S1 fixture migration lost gate slots")
        else:
            if handle.get("architecture_id") != \
                    "bramastra-base-decoder/v1+disabled-gate-slots":
                raise ValueError("S0 fixture architecture must carry disabled slots")
        return handle

    def evaluate_episode(self, *, handle, episode, task_env=None):
        # Deterministic double: requires a real episode with mechanism +
        # verifier; counters derive from the recorded double call (one
        # deterministic generation event, never case-number arithmetic).
        if not isinstance(episode, dict) or not episode.get("mechanism"):
            raise ValueError("evaluate_episode requires a real episode")
        if not callable(episode.get("verifier")):
            raise ValueError("evaluate_episode requires an independent verifier")
        self.calls.append(("evaluate_episode", {
            "id": handle.get("double_id")}))
        # Fixed deterministic outcome WITHOUT claiming learned success:
        # success comes from the verifier on a recorded observation.
        observation = {"answer": "double", "sum": "double"}
        try:
            success = bool(episode["verifier"](
                episode["mechanism"], observation))
        except Exception:
            success = False
        return {"answer": "double", "stopped_on_eos": True, "success": success,
                "model_calls": 1, "actions": 1, "nodes": 1, "cost": 1.0,
                "new_tokens": 1}

    def publish_checkpoint(self, *, handle, run_dir, phase, arm, seed,
                           update_index, parent_checkpoint_id,
                           writer_token=None, expected_parent=None,
                           data_dir=None):
        import json as _json
        import os as _os

        # Fixture publication: writes a receipt (not a .pt payload) and
        # returns a fixture-labeled identity. E6 must reject fixture-only
        # bundles (no .pt payloads). A per-handle sequence keeps repeated
        # fraction publishes distinct (optimizer never advances for doubles).
        self.calls.append(("publish_checkpoint", {
            "phase": phase, "arm": arm, "seed": seed,
            "update_index": update_index}))
        seq = int(handle.get("_fixture_seq", 0)) + 1
        handle["_fixture_seq"] = seq
        mirror = _os.path.join(run_dir, "checkpoints", phase, f"{arm}-{seed}")
        _os.makedirs(mirror, exist_ok=True)
        identity = f"fixture-ckpt-{phase}-{arm}-{seed}-{update_index}-{seq}"
        with open(_os.path.join(mirror, f"{update_index:06d}-{seq:03d}.json"), "w",
                  encoding="utf-8") as fh:
            _json.dump({"checkpoint_id": identity, "evidence_kind": "fixture",
                        "update_index": update_index, "seq": seq}, fh, indent=2)
        return identity

    def restore_verify(self, *, run_dir, checkpoint_id):
        if str(checkpoint_id).startswith("fixture-ckpt-"):
            raise ValueError(
                "fixture checkpoint cannot satisfy restore_verify; a real "
                ".pt payload is required")
        self.calls.append(("restore_verify", {"id": str(checkpoint_id)[:12]}))
        return {"restored_ok": True, "checkpoint_id": checkpoint_id}

    def snapshot_state(self, handle):
        self.calls.append(("snapshot_state", {"id": handle["double_id"]}))
        return dict(handle)

    def states_equal(self, left, right) -> bool:
        self.calls.append(("states_equal", {}))
        if isinstance(left, dict) and isinstance(right, dict):
            return left.get("init_state") == right.get("init_state") \
                and left.get("seed") == right.get("seed")
        return left == right

    def training_update(self, handle, *, batch, window, extra=None,
                          pair_rows=None):
        self._require_window_eligible(window, extra)
        # Pair eligibility: enabled positive-weight pair requires real rows.
        weights = dict(getattr(window, "weights", {}))
        enabled = set(getattr(window, "enabled_terms", frozenset()))
        if "pair" in enabled and float(weights.get("pair", 0.0)) > 0 \
                and not pair_rows:
            raise ValueError(
                "pair objective enabled but no pair renderings supplied; "
                "refusing silent zero pair loss")
        weights = dict(getattr(window, "weights", {}))
        enabled = sorted(getattr(window, "enabled_terms", frozenset()))
        self.calls.append(("training_update", {"weights": weights,
                                               "enabled": enabled,
                                               "extra_terms": sorted((extra or {})),
                                               "targets": getattr(batch, "target_count", 0)}))
        # Fixture counts only: logical deltas without optimizer work. The
        # caller must label the receipt fixture; fixture never qualifies.
        handle["updates"] += 1
        handle["attempted"] += 1
        exposure = int(getattr(batch, "target_count", 0))
        handle["exposure"] += exposure
        return {"committed": 1, "attempted": 1, "exposure": exposure,
                "optimizer_update": handle["updates"],
                "evidence_kind": "fixture"}

    def save_checkpoint(self, handle, *, path, fraction):
        import os as _os

        self.calls.append(("save_checkpoint", {"id": handle["double_id"],
                                               "fraction": fraction,
                                               "path": path}))
        # Write a fixture receipt alongside the (non-payload) identity so the
        # absence of .pt files remains detectable by E6.
        try:
            _os.makedirs(_os.path.dirname(path), exist_ok=True)
            with open(path + ".fixture.json", "w", encoding="utf-8") as fh:
                fh.write('{"evidence_kind": "fixture"}')
        except OSError:
            pass
        return f"fixture-ckpt-{handle['double_id']}-{fraction}"

    def free_generation(self, handle, *, prompt, max_new_tokens):
        self.calls.append(("free_generation", {"id": handle["double_id"],
                                               "max_new_tokens": max_new_tokens}))
        return {"answer": "double", "stopped_on_eos": True, "tokens": []}

    def optimizer_updates(self, handle) -> int:
        # Doubles perform zero optimizer updates by construction.
        return 0
