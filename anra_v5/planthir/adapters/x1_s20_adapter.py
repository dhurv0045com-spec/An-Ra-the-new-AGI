"""PlanthirBackend adapter binding S20-backed ELENDRIL V8-X1 (R, R-noedge, P-capR, U, D1-H/D2) into Palantiri."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn

from anra_v5.elendril.codec import digest
from anra_v5.elendril.v8x.contracts import (
    X1ModelSpec,
    X1_ARCHITECTURES,
    X1_D2_DATASET_SCHEMA,
    X1_DATASET_SCHEMA,
    X1_OBJECTIVES,
)
from anra_v5.elendril.v8x.data import (
    X1D1HDataset,
    X1D2Dataset,
    X1D2PairedGroup,
    X1SupervisedState,
    build_x1_d1h_training_dataset,
    build_x1_d2_paired_dataset,
)
from anra_v5.elendril.v8x.evaluation import X1S20EnvironmentKernel, X1S20ObservablePolicy
from anra_v5.elendril.v8x.objectives import compute_x1_objective_batch
from anra_v5.elendril.v8x.training import X1S20Trainer, X1TrainRecipe

from ..backend import BACKEND_CONTRACT_VERSION, BackendCapabilities
from ..contracts import EnvironmentKernel, ObservablePolicy


class X1S20Backend:
    """First-class PlanthirBackend executing full ELENDRIL V8-X1 architectures on V5 S20/tiny specs."""

    backend_id = "x1_s20"
    backend_version = "anra-x1-s20-1.0"
    contract_version = BACKEND_CONTRACT_VERSION

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            backend_version=self.backend_version,
            contract_version=self.contract_version,
            supported_architectures=X1_ARCHITECTURES,
            supported_objectives=X1_OBJECTIVES,
            supported_samplings=("uniform", "interleaved"),
            supported_devices=("cpu", "cuda"),
            supports_grouped_loss=True,
            supports_resume=True,
            supports_fp16=False,
        )

    def resolve_model_spec(self, arm: Any) -> X1ModelSpec:
        spec = getattr(arm, "model_spec", None)
        if not isinstance(spec, X1ModelSpec):
            raise ValueError("x1_s20 backend requires arm.model_spec to be an X1ModelSpec")
        spec.assert_valid()
        return spec

    def canonical_model_spec(self, model_spec: Any) -> dict[str, Any]:
        if not isinstance(model_spec, X1ModelSpec):
            raise ValueError("expected X1ModelSpec")
        return model_spec.canonical()

    def model_parameter_count(self, model_spec: Any) -> int:
        if not isinstance(model_spec, X1ModelSpec):
            raise ValueError("expected X1ModelSpec")
        return model_spec.total_parameters()

    def canonical_factor_config(self, arm: Any) -> dict[str, Any]:
        return self.resolve_model_spec(arm).canonical()

    def validate_kernel(self, kernel: EnvironmentKernel) -> None:
        if not isinstance(kernel, X1S20EnvironmentKernel):
            raise ValueError("x1_s20 backend requires X1S20EnvironmentKernel")

    def validate_arm(self, arm: Any, train_data: Any, *, device: str, cpu_canary: bool) -> dict[str, Any]:
        caps = self.capabilities()
        if getattr(arm, "trainer_factory", None) is not None:
            raise ValueError("custom trainer_factory is unsupported on qualified x1_s20 backend")
        spec = self.resolve_model_spec(arm)
        if arm.architecture_family != spec.architecture:
            raise ValueError("arm.architecture_family differs from X1ModelSpec.architecture")
        if spec.architecture not in caps.supported_architectures:
            raise ValueError(f"unsupported X1 architecture: {spec.architecture!r}")
        arm.recipe.validate()
        if arm.recipe.objective not in caps.supported_objectives or arm.recipe.sampling not in caps.supported_samplings:
            raise ValueError(f"unsupported X1 objective/sampling: {(arm.recipe.objective, arm.recipe.sampling)!r}")
        if not isinstance(train_data, (X1D1HDataset, X1D2Dataset)) or arm.dataset_identity != train_data.data_identity:
            raise ValueError("X1 S20 dataset identity or schema mismatch")
        if arm.recipe.objective == "standard" and not isinstance(train_data, X1D1HDataset):
            raise ValueError("standard objective requires X1D1HDataset")
        if arm.recipe.objective != "standard" and not isinstance(train_data, X1D2Dataset):
            raise ValueError("paired objectives require X1D2Dataset")
        if any(type(u) is not int or not 1 <= u <= arm.recipe.total_updates for u in arm.milestone_updates):
            raise ValueError("milestone updates must fall inside total_updates")

        n_samples = len(train_data.states) if isinstance(train_data, X1D1HDataset) else sum(len(g.samples) for g in train_data.groups)
        if torch.device(device).type == "cpu" and (
            not cpu_canary
            or n_samples > caps.max_cpu_canary_samples
            or spec.total_parameters() > caps.max_cpu_canary_params
            or arm.recipe.total_updates > caps.max_cpu_canary_updates
        ):
            raise ValueError("CPU canary limits exceeded for x1_s20 arm")
        return {
            "backend_id": self.backend_id,
            "architecture": spec.architecture,
            "parameter_count": spec.total_parameters(),
            "sample_count": n_samples,
        }

    def validate_dataset_and_worlds(
        self,
        train_data: Any,
        train_worlds: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(train_data, (X1D1HDataset, X1D2Dataset)):
            raise ValueError("x1_s20 requires X1D1HDataset or X1D2Dataset")
        if digest(train_data.manifest) != train_data.data_identity:
            raise ValueError("X1 dataset manifest checksum mismatch")
        if not train_worlds:
            raise ValueError("x1_s20 requires non-empty train_worlds for transitivity audit")
        return {
            "dataset_name": train_data.dataset_name,
            "data_identity": train_data.data_identity,
            "content_sha256": train_data.data_identity,
            "world_count": len(train_worlds),
            "resolved_train_worlds": tuple(train_worlds),
        }

    def create_trainer(
        self,
        arm: Any,
        train_data: Any,
        *,
        source_commit: str,
        root: Path,
        lineage: str,
        device: str,
        cpu_canary: bool,
        resume: bool,
    ) -> X1S20Trainer:
        spec = self.resolve_model_spec(arm)
        arm_resume = resume and (root / lineage / "LATEST").is_file()
        is_cpu = torch.device(device).type == "cpu"
        if isinstance(train_data, X1D2Dataset):
            return X1S20Trainer(
                x1_spec=spec,
                recipe=arm.recipe,
                order_seed=arm.order_seed,
                source_commit=source_commit,
                root=root,
                lineage=lineage,
                data_provenance=train_data.manifest,
                groups=train_data.groups,
                device=device,
                cpu_canary=(is_cpu and cpu_canary),
                resume=arm_resume,
            )
        return X1S20Trainer(
            x1_spec=spec,
            recipe=arm.recipe,
            order_seed=arm.order_seed,
            source_commit=source_commit,
            root=root,
            lineage=lineage,
            data_provenance=train_data.manifest,
            states=train_data.states,
            interleaved_schedule=train_data.interleaved_schedule,
            device=device,
            cpu_canary=(is_cpu and cpu_canary),
            resume=arm_resume,
        )

    def sample_gradient_loss_fn(
        self,
        trainer: X1S20Trainer,
        arm: Any,
        train_data: X1D1HDataset | X1D2Dataset,
        device: torch.device,
    ) -> Callable[[nn.Module], torch.Tensor]:
        from anra_v5.elendril.training import batch_tensors
        import torch.nn.functional as F

        if isinstance(train_data, X1D2Dataset):
            # Include only groups that have ANSWER targets when checking pointer/head grads
            chosen = [g for g in train_data.groups if any(s.gold_op == 0 for s in g.samples)][:2] or list(train_data.groups[:1])

            def _loss_d2(m: nn.Module) -> torch.Tensor:
                from anra_v5.planthir.adapters.x1_rd2_adapter import compute_x1_d2_grouped_loss

                flat = [s for g in chosen for s in g.samples]
                tokens, segments, _ = batch_tensors([(s.tokens, s.eligible) for s in flat], device)
                apos = torch.tensor([s.action_position for s in flat], device=device, dtype=torch.int64)
                out = m.forward_action(tokens, segments, apos)  # type: ignore[attr-defined]
                eq_a, eq_b, eq_perm, eq_mask = [], [], [], []
                cf_a, cf_b, cf_ta, cf_tb, cf_mask = [], [], [], [], []
                off = 0
                for g in chosen:
                    for ep in g.eq_pairs:
                        eq_a.append(off + ep.idx_a)
                        eq_b.append(off + ep.idx_b)
                        eq_perm.append(list(ep.perm_16))
                        eq_mask.append(bool(ep.eligible))
                    for cp in g.cf_pairs:
                        cf_a.append(off + cp.idx_a)
                        cf_b.append(off + cp.idx_b)
                        cf_ta.append(int(cp.target_a))
                        cf_tb.append(int(cp.target_b))
                        cf_mask.append(bool(cp.eligible))
                    off += len(g.samples)
                tot, _ = compute_x1_d2_grouped_loss(
                    out.log_probs_19,
                    out.log_p_op,
                    out.log_p_val,
                    gold_ops=torch.tensor([s.gold_op for s in flat], device=device, dtype=torch.int64),
                    gold_args=torch.tensor([s.gold_arg for s in flat], device=device, dtype=torch.int64),
                    state_mask=torch.tensor([bool(s.state_mask) for s in flat], device=device, dtype=torch.bool),
                    eq_a=torch.tensor(eq_a, device=device, dtype=torch.int64),
                    eq_b=torch.tensor(eq_b, device=device, dtype=torch.int64),
                    eq_perm_16=torch.tensor(eq_perm, device=device, dtype=torch.int64) if eq_perm else torch.zeros((0, 16), device=device, dtype=torch.int64),
                    eq_mask=torch.tensor(eq_mask, device=device, dtype=torch.bool),
                    cf_a=torch.tensor(cf_a, device=device, dtype=torch.int64),
                    cf_b=torch.tensor(cf_b, device=device, dtype=torch.int64),
                    cf_target_a=torch.tensor(cf_ta, device=device, dtype=torch.int64),
                    cf_target_b=torch.tensor(cf_tb, device=device, dtype=torch.int64),
                    cf_mask=torch.tensor(cf_mask, device=device, dtype=torch.bool),
                    eq_weight=float(getattr(arm.recipe, "eq_weight", 0.1)) if arm.recipe.objective in ("PAIR-EQ", "PAIR-EQ-CF") else 0.0,
                    cf_weight=float(getattr(arm.recipe, "cf_weight", 0.1)) if arm.recipe.objective == "PAIR-EQ-CF" else 0.0,
                    cf_margin=float(getattr(arm.recipe, "cf_margin", 0.5)),
                )
                return tot

            return _loss_d2

        states = [s for s in train_data.states if s.gold_op == 0][:4] or list(train_data.states[:4])
        if trainer.x1_spec.architecture == "B0":
            def _loss_b0(m: nn.Module) -> torch.Tensor:
                from v5_model.core import packed_layout

                tokens, segments, eligible = batch_tensors([(s.tokens, s.eligible) for s in states], device)
                positions, mask_attn = packed_layout(segments, torch_module=torch)
                logits = m(tokens, positions, mask_attn)  # type: ignore[operator]
                losses = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), reduction="none").reshape_as(tokens[:, 1:])
                return (losses * eligible[:, 1:].to(losses.dtype)).mean()

            return _loss_b0

        def _loss_std(m: nn.Module) -> torch.Tensor:
            tokens, segments, _ = batch_tensors([(s.tokens, s.eligible) for s in states], device)
            apos = torch.tensor([s.action_position for s in states], device=device, dtype=torch.int64)
            gold_ops = torch.tensor([s.gold_op for s in states], device=device, dtype=torch.int64)
            gold_args = torch.tensor([s.gold_arg for s in states], device=device, dtype=torch.int64)
            out = m.forward_action(tokens, segments, apos)  # type: ignore[attr-defined]
            op_nll = -out.log_p_op.float().gather(1, gold_ops.unsqueeze(1)).squeeze(1)
            is_ans = (gold_ops == 0).to(op_nll.dtype)
            val_nll = -out.log_p_val.float().gather(1, gold_args.clamp(0, 15).unsqueeze(1)).squeeze(1) * is_ans
            return (op_nll + val_nll).mean()

        return _loss_std

    def create_policy(
        self,
        model: Any,
        model_spec: Any,
        *,
        model_identity: str,
        mode: str = "raw",
        copy_enabled_override: bool | None = None,
    ) -> ObservablePolicy:
        if not isinstance(model_spec, X1ModelSpec):
            raise ValueError("expected X1ModelSpec")
        return X1S20ObservablePolicy(
            model,
            model_spec,
            model_identity=model_identity,
            mode=mode,
            copy_enabled_override=copy_enabled_override,
        )


def prepare_x1_s20_canary_data(*, objective: str = "standard", quota_per_family: int = 1, max_states: int = 24) -> tuple[X1D1HDataset | X1D2Dataset, tuple[dict[str, Any], ...]]:
    """Prepare bounded CPU canary X1 D1-H / D2 corpus within Planthir CPU limits (<= 32 samples)."""

    d1h = build_x1_d1h_training_dataset(quota_per_family=max(1, quota_per_family))
    if objective == "standard":
        limited = d1h.states[:max_states]
        # Ensure at least one ANSWER state is present so pointer/value heads receive active gradients
        if not any(s.gold_op == 0 for s in limited):
            ans_s = [s for s in d1h.states if s.gold_op == 0][:4]
            limited = (*ans_s, *limited[: max_states - len(ans_s)])
        manifest = {**d1h.manifest, "state_count": len(limited), "states_sha256": digest([s.task_id for s in limited])}
        ds = X1D1HDataset(
            dataset_name=d1h.dataset_name,
            split=d1h.split,
            data_identity=digest(manifest),
            worlds=d1h.worlds,
            states=tuple(limited),
            interleaved_schedule=tuple(range(len(limited))),
            manifest=manifest,
        )
        return ds, d1h.worlds
    max_groups = max(1, max_states // 4)
    d2 = build_x1_d2_paired_dataset(base_worlds=d1h.worlds[:max_groups])
    groups = d2.groups[:max_groups]
    manifest = {**d2.manifest, "group_count": len(groups), "sample_count": sum(len(g.samples) for g in groups), "groups_sha256": digest([g.group_id for g in groups])}
    ds2 = X1D2Dataset(
        dataset_name=d2.dataset_name,
        split=d2.split,
        data_identity=digest(manifest),
        worlds=d2.worlds[:max_groups],
        groups=tuple(groups),
        manifest=manifest,
    )
    return ds2, d1h.worlds[:max_groups]


__all__ = ["X1S20Backend", "prepare_x1_s20_canary_data"]
