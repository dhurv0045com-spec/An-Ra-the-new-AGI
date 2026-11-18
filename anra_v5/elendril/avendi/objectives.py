"""Avendi V8-Y2 P3 controlled acquisition: paired counterfactual endpoint margin loss vs same-batch CE."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.foundation_training import Recipe, TokenSample
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.training import MODEL, batch_tensors, configure_device
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v6_training import V6Trainer
from v5_contracts.model_spec import ModelSpec
from v5_training.checkpoint import CheckpointStore
from .frozen import evaluate_backend_on_avendi_split
from .metrics import AP_TOKEN_IDS, bootstrap_paired_family_contrasts
from .protocol import (
    INIT_SEEDS,
    ORDER_SEEDS,
    P3_TRAINING_SPEC,
    QUALIFICATION_GATES,
    derive_seed_int,
)
from .worlds import AvendiCase, AvendiFamily


CPU_CANARY_S3_SPEC = replace(MODEL, vocabulary_size=UTF8.vocabulary_size, context_length=512)


@dataclass(frozen=True)
class AvendiMarginSample(TokenSample):
    """Framed byte-token sample carrying gold and rival endpoint token IDs for P3 margin loss."""

    family_id: str = ""
    cell_key: str = ""
    gold_token_id: int = 0
    rival_token_id: int = 0


def case_to_margin_sample(case: AvendiCase, *, epoch: int = 0) -> AvendiMarginSample:
    """Convert an AvendiCase into a framed AvendiMarginSample with eligible mask on action+EOS only."""
    n_prompt = case.target_boundary_index
    n_total = len(case.full_tokens)
    eligible = tuple(i >= n_prompt for i in range(n_total))
    return AvendiMarginSample(
        task_id=f"e{epoch}/{case.case_id}",
        tokens=case.full_tokens,
        eligible=eligible,
        world_id=case.family_id,
        category="route",
        family_id=case.family_id,
        cell_key=case.cell_key,
        gold_token_id=AP_TOKEN_IDS[case.expected_answer],
        rival_token_id=AP_TOKEN_IDS[case.rival_answer],
    )


def build_p3_family_aligned_stream(
    one_hop_families: Sequence[AvendiFamily],
    *,
    order_seed: int,
    epochs: int = 4,
) -> dict[str, Any]:
    """Build 4-epoch stream shuffling whole 4-cell families identically across CE and CE+pair_margin arms."""
    samples: list[AvendiMarginSample] = []
    family_schedule: list[str] = []
    for ep in range(epochs):
        rng = random.Random(derive_seed_int("avendi-p3-epoch-order", order_seed, ep))
        fams = list(one_hop_families)
        rng.shuffle(fams)
        for fam in fams:
            family_schedule.append(f"e{ep}:{fam.family_id}")
            for cell in fam.cells:
                samples.append(case_to_margin_sample(cell, epoch=ep))

    schedule_sha = digest(family_schedule)
    return {
        "samples": tuple(samples),
        "family_schedule": tuple(family_schedule),
        "family_schedule_sha256": schedule_sha,
        "provenance": {
            "schema": "anra-avendi-v8-y2-p3-stream-v1",
            "split": "train",
            "license": "repository LICENSE; Avendi V8-Y2 P3 1-hop crossover",
            "order_seed": order_seed,
            "epochs": epochs,
            "families_per_epoch": len(one_hop_families),
            "total_samples": len(samples),
            "family_schedule_sha256": schedule_sha,
        },
    }


def compute_ce_and_pair_margin_terms(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    loss_weights: torch.Tensor,
    chunk: Sequence[AvendiMarginSample],
    *,
    margin: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Compute eligible-token CE sum and counterfactual pair margin sum on a microbatch."""
    ce_losses = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        tokens[:, 1:].reshape(-1),
        reduction="none",
    ).reshape_as(tokens[:, 1:])
    ce_sum = (ce_losses * loss_weights[:, 1:]).sum()
    eligible_tokens = int(loss_weights[:, 1:].sum().item())

    margin_terms: list[torch.Tensor] = []
    for row_idx, sample in enumerate(chunk):
        first_pos = sample.eligible.index(True)
        step_logits = logits[row_idx, first_pos - 1].float()
        s_i = step_logits[sample.gold_token_id] - step_logits[sample.rival_token_id]
        margin_terms.append(F.relu( float(margin) - s_i ))
    margin_sum = torch.stack(margin_terms).sum() if margin_terms else ce_sum.new_tensor(0.0)
    return ce_sum, margin_sum, eligible_tokens, len(margin_terms)


def audit_pair_margin_gradient_sign_reversal() -> dict[str, Any]:
    """Negative control: swapping gold and rival pair labels on a fixture reverses the signed auxiliary gradient."""
    torch.manual_seed(7)
    logits = torch.zeros((1, 4, 260), dtype=torch.float32, requires_grad=True)
    tokens = torch.tensor([[UTF8.bos_id, 69, 70, UTF8.eos_id]], dtype=torch.int64)
    weights = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)

    s_orig = AvendiMarginSample(
        task_id="neg_orig",
        tokens=(UTF8.bos_id, 69, 70, UTF8.eos_id),
        eligible=(False, False, True, True),
        world_id="fam0",
        category="route",
        family_id="fam0",
        cell_key="00",
        gold_token_id=AP_TOKEN_IDS["G"],
        rival_token_id=AP_TOKEN_IDS["H"],
    )
    s_swap = replace(
        s_orig,
        task_id="neg_swap",
        gold_token_id=AP_TOKEN_IDS["H"],
        rival_token_id=AP_TOKEN_IDS["G"],
    )

    _, m_orig, _, _ = compute_ce_and_pair_margin_terms(logits, tokens, weights, [s_orig], margin=1.0)
    g_orig = torch.autograd.grad(m_orig, logits, retain_graph=True)[0].detach().clone()

    _, m_swap, _, _ = compute_ce_and_pair_margin_terms(logits, tokens, weights, [s_swap], margin=1.0)
    g_swap = torch.autograd.grad(m_swap, logits, retain_graph=True)[0].detach().clone()

    id_g = AP_TOKEN_IDS["G"]
    id_h = AP_TOKEN_IDS["H"]
    grad_g_orig = float(g_orig[0, 1, id_g].item())
    grad_h_orig = float(g_orig[0, 1, id_h].item())
    grad_g_swap = float(g_swap[0, 1, id_g].item())
    grad_h_swap = float(g_swap[0, 1, id_h].item())

    reversed_ok = (
        grad_g_orig < 0.0
        and grad_h_orig > 0.0
        and math.isclose(grad_g_swap, -grad_g_orig, rel_tol=1e-6)
        and math.isclose(grad_h_swap, -grad_h_orig, rel_tol=1e-6)
    )
    return {
        "schema": "anra-avendi-v8-y2-margin-neg-control-v1",
        "status": "PASS" if reversed_ok else "FAIL",
        "grad_g_orig": grad_g_orig,
        "grad_h_orig": grad_h_orig,
        "grad_g_swapped": grad_g_swap,
        "grad_h_swapped": grad_h_swap,
    }


class AvendiP3Trainer(V6Trainer):
    """Deterministic S3 trainer for P3 comparing ordinary CE (lambda=0) vs CE+pair_margin (lambda=0.25)."""

    def __init__(
        self,
        samples: tuple[AvendiMarginSample, ...],
        *,
        spec: ModelSpec,
        recipe: Recipe,
        source_commit: str,
        root: Path,
        lineage: str,
        data_provenance: dict[str, Any],
        lambda_pair_margin: float = 0.0,
        margin: float = 1.0,
        device: str = "cpu",
        resume: bool = False,
    ) -> None:
        if not math.isfinite(lambda_pair_margin) or lambda_pair_margin < 0.0:
            raise ValueError("lambda_pair_margin must be finite and non-negative")
        self.lambda_pair_margin = float(lambda_pair_margin)
        self.margin = float(margin)
        self.update1_gradient_audit: dict[str, Any] | None = None
        prov = {
            **data_provenance,
            "avendi_lambda_pair_margin": self.lambda_pair_margin,
            "avendi_margin": self.margin,
        }
        super().__init__(
            samples,
            spec=spec,
            recipe=recipe,
            source_commit=source_commit,
            root=root,
            lineage=lineage,
            data_provenance=prov,
            tokenizer_manifest=UTF8.manifest.record(),
            device=device,
            resume=resume,
        )

    def indices(self, update: int) -> tuple[int, ...]:
        start = update * self.recipe.batch_size
        end = start + self.recipe.batch_size
        if end > len(self.samples):
            raise ValueError(f"update {update} exceeds P3 stream ({len(self.samples)})")
        return tuple(range(start, end))

    def sample_cursor(self, completed_updates: int) -> int:
        return completed_updates * self.recipe.batch_size

    def step(self) -> dict[str, Any]:
        if self.faulted or self.state.complete:
            raise ValueError("complete or faulted engine requires a fresh restore")
        self.faulted = True
        u = self.state.global_update
        chosen = [self.samples[i] for i in self.plan[u]]
        batch_len = len(chosen)
        weights = [[float(x) for x in s.eligible] for s in chosen]
        total_eligible_tokens = sum(sum(row[1:]) for row in weights)
        if total_eligible_tokens <= 0:
            raise ValueError("P3 batch contains zero supervised target tokens")

        self.model.train()
        lr = self.learning_rate(u)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)

        total_ce = 0.0
        total_margin = 0.0
        total_loss = 0.0
        eligible_margin_cells = 0

        for start in range(0, batch_len, self.recipe.micro_batch_size):
            chunk = chosen[start : start + self.recipe.micro_batch_size]
            chunk_w = weights[start : start + self.recipe.micro_batch_size]
            tokens, segments, _ = batch_tensors([(s.tokens, s.eligible) for s in chunk], self.device)
            loss_weights = torch.tensor(
                [row + [0.0] * (tokens.shape[1] - len(row)) for row in chunk_w],
                device=self.device,
            )
            logits = self.backend.logits(
                tokens, segments, activation_checkpointing=self.recipe.activation_checkpointing
            )
            ce_sum, margin_sum, _, m_cells = compute_ce_and_pair_margin_terms(
                logits, tokens, loss_weights, chunk, margin=self.margin
            )
            eligible_margin_cells += m_cells

            # Audit both objective gradient norms on shared logits at update 1 (u == 0, first microbatch)
            if u == 0 and start == 0 and self.update1_gradient_audit is None:
                ce_g = torch.autograd.grad(ce_sum / total_eligible_tokens, logits, retain_graph=True)[0]
                m_g = torch.autograd.grad(margin_sum / batch_len, logits, retain_graph=True)[0]
                self.update1_gradient_audit = {
                    "update": 1,
                    "ce_logits_grad_norm": float(torch.linalg.vector_norm(ce_g).item()),
                    "pair_margin_logits_grad_norm": float(torch.linalg.vector_norm(m_g).item()),
                    "lambda_pair_margin": self.lambda_pair_margin,
                    "eligible_ce_tokens_in_batch": int(total_eligible_tokens),
                    "eligible_margin_cells_in_batch": batch_len,
                }

            loss = (ce_sum / total_eligible_tokens) + self.lambda_pair_margin * (margin_sum / batch_len)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite P3 loss")
            self.scaler.scale(loss).backward()
            total_ce += float((ce_sum / total_eligible_tokens).detach().item())
            total_margin += float((margin_sum / batch_len).detach().item())
            total_loss += float(loss.detach().item())

        self.scaler.unscale_(self.optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe.clip).item())
        if not math.isfinite(grad_norm):
            raise FloatingPointError("non-finite P3 gradient norm")
        self.scaler.step(self.optimizer)
        self.scaler.update()

        next_u = u + 1
        new_tokens = self.state.cumulative_tokens + self.counts[u]
        self.state = replace(
            self.state,
            generation=next_u,
            global_update=next_u,
            cumulative_tokens=new_tokens,
            tokens_by_source={"action": new_tokens},
            optimizer_step_max=next_u,
            schedule_tokens=new_tokens,
            cursor=replace(self.state.cursor, sequence_ordinal=self.sample_cursor(next_u)),
            rng_state_sha256=sha256(self.rng_bytes()).hexdigest(),
            parent_checkpoint_sha256=self.parent,
        )
        self.state.assert_valid()
        self.faulted = False
        return {
            "update": next_u,
            "loss": total_loss,
            "ce_loss": total_ce,
            "pair_margin_loss": total_margin,
            "grad_norm": grad_norm,
            "lr": lr,
            "supervised_tokens": self.counts[u],
            "eligible_margin_cells": eligible_margin_cells,
        }


def evaluate_monitor_slice(backend: Any, families: Sequence[AvendiFamily]) -> dict[str, Any]:
    """Evaluate a 64-family train or dev monitor slice during P3 training."""
    cases = [c for f in families for c in f.cells]
    fake_split = {"split": "monitor", "literal_cases": (), "one_hop_families": families, "three_hop_families": ()}
    res = evaluate_backend_on_avendi_split(backend, fake_split, job_id="monitor", checkpoint_sha256="0" * 64)
    oh = res["one_hop"]
    return {
        "families": len(families),
        "cases": len(cases),
        "raw_exact_accuracy": oh["raw_exact_accuracy"],
        "all_four_raw_rate": oh["all_four_raw_rate"],
        "latent_flip_rate": oh["latent_flip_rate"],
        "mean_signed_margin": oh["mean_signed_margin"],
    }


def run_p3_job(
    *,
    arm: str,
    seed: int,
    suite: Mapping[str, Any],
    output_dir: Path,
    source_commit: str,
    device: str = "cpu",
    cpu_canary: bool = False,
    canary_updates: int = 2,
    max_steps_this_call: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Train or resume one P3 job (CE or CE_PAIR_MARGIN) and evaluate on dev + confirm."""
    if arm not in P3_TRAINING_SPEC["arms"]:
        raise ValueError(f"unsupported P3 arm: {arm!r}")
    arm_cfg = P3_TRAINING_SPEC["arms"][arm]
    lam = float(arm_cfg["lambda_pair_margin"])
    order_seed = ORDER_SEEDS.get(seed, 1000 + seed)
    job_id = f"AV-p3_acquisition-S3-{arm}-s{seed}"

    train_fams = suite["splits"]["train_p3"]["one_hop_families"]
    if cpu_canary:
        train_fams = train_fams[:8]
    stream = build_p3_family_aligned_stream(
        train_fams,
        order_seed=order_seed,
        epochs=1 if cpu_canary else int(P3_TRAINING_SPEC["epochs"]),
    )

    spec = CPU_CANARY_S3_SPEC if cpu_canary else MODEL_PROFILES["S3"]
    batch_size = 8 if cpu_canary else int(P3_TRAINING_SPEC["batch_size"])
    micro_batch = 4
    total_updates = canary_updates if cpu_canary else int(P3_TRAINING_SPEC["total_updates"])

    recipe = Recipe(
        seed=seed,
        total_updates=total_updates,
        batch_size=batch_size,
        micro_batch_size=micro_batch,
        lr=float(P3_TRAINING_SPEC["lr_peak"]),
        clip=float(P3_TRAINING_SPEC["grad_clip"]),
        objective="ce",
        sampling="uniform",
        schedule="cosine",
        precision="fp32",
    )

    cp_root = output_dir / "checkpoints"
    store = CheckpointStore(cp_root, job_id)
    is_resume = bool(resume and store.latest_sha256() is not None)

    configure_device(device)
    trainer = AvendiP3Trainer(
        stream["samples"],
        spec=spec,
        recipe=recipe,
        source_commit=source_commit,
        root=cp_root,
        lineage=job_id,
        data_provenance=stream["provenance"],
        lambda_pair_margin=lam,
        margin=float(arm_cfg["margin"]),
        device=device,
        resume=is_resume,
    )

    # Record epoch-0 initial weight digest to prove CE and CE_PAIR_MARGIN share identical init per seed
    init_param_sha = digest([
        [round(float(v), 6) for v in p.detach().cpu().flatten()[:16].tolist()]
        for p in trainer.model.parameters()
    ])

    monitor_train_fams = train_fams[: min(len(train_fams), 2 if cpu_canary else int(P3_TRAINING_SPEC["monitor_families"]))]
    dev_fams = suite["splits"]["dev"]["one_hop_families"]
    monitor_dev_fams = dev_fams[: min(len(dev_fams), 2 if cpu_canary else int(P3_TRAINING_SPEC["monitor_families"]))]
    milestones = {0, total_updates} if cpu_canary else set(P3_TRAINING_SPEC["monitor_milestones"])

    monitors: dict[str, Any] = {}
    if trainer.state.global_update == 0 and 0 in milestones:
        monitors["0"] = {
            "train": evaluate_monitor_slice(trainer.backend, monitor_train_fams),
            "dev": evaluate_monitor_slice(trainer.backend, monitor_dev_fams),
        }

    step_logs: list[dict[str, Any]] = []
    steps_taken = 0
    while trainer.state.global_update < recipe.total_updates:
        if max_steps_this_call is not None and steps_taken >= max_steps_this_call:
            break
        row = trainer.step()
        step_logs.append(row)
        steps_taken += 1
        u_now = trainer.state.global_update
        if u_now in milestones:
            monitors[str(u_now)] = {
                "train": evaluate_monitor_slice(trainer.backend, monitor_train_fams),
                "dev": evaluate_monitor_slice(trainer.backend, monitor_dev_fams),
            }
        if u_now == recipe.total_updates or (u_now % 64 == 0) or cpu_canary:
            trainer.save()
            trainer.store.prune(keep={trainer.parent} if trainer.parent else set())

    latest_sha = trainer.store.latest_sha256() or trainer.save()
    trainer.store.prune(keep={trainer.parent} if trainer.parent else set())

    dev_split = suite["splits"]["dev"]
    conf_split = suite["splits"]["confirm"]
    if cpu_canary:
        dev_split = {
            **dev_split,
            "literal_cases": dev_split["literal_cases"][:8],
            "one_hop_families": dev_split["one_hop_families"][:4],
            "three_hop_families": dev_split["three_hop_families"][:2],
        }
        conf_split = {
            **conf_split,
            "literal_cases": conf_split["literal_cases"][:8],
            "one_hop_families": conf_split["one_hop_families"][:4],
            "three_hop_families": conf_split["three_hop_families"][:2],
        }

    dev_eval = evaluate_backend_on_avendi_split(trainer.backend, dev_split, job_id=f"{job_id}_dev", checkpoint_sha256=latest_sha)
    conf_eval = evaluate_backend_on_avendi_split(trainer.backend, conf_split, job_id=f"{job_id}_confirm", checkpoint_sha256=latest_sha)

    jdir = output_dir / "analysis" / "jobs" / job_id
    jdir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "anra-avendi-v8-y2-p3-job-receipt-v1",
        "job_id": job_id,
        "arm": arm,
        "seed": seed,
        "order_seed": order_seed,
        "lambda_pair_margin": lam,
        "checkpoint_sha256": latest_sha,
        "global_update": trainer.state.global_update,
        "family_schedule_sha256": stream["family_schedule_sha256"],
        "init_param_probe_sha256": init_param_sha,
        "update1_gradient_audit": trainer.update1_gradient_audit,
        "monitors": monitors,
        "dev_summary": {k: v for k, v in dev_eval.items() if k != "case_scores"},
        "confirm_summary": {k: v for k, v in conf_eval.items() if k != "case_scores"},
    }
    (jdir / "run_receipt.json").write_bytes(canonical(receipt) + b"\n")
    return receipt


def evaluate_p3_acquisition_gate(
    job_receipts_by_arm_seed: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Evaluate P3 support gate: >= 10 pp mean 1-hop family gain, positive every seed, >= 80% confirm 1-hop, literal drop <= 5 pp."""
    cfg = QUALIFICATION_GATES["p3_acquisition"]
    ce_map = job_receipts_by_arm_seed.get("CE", {})
    pm_map = job_receipts_by_arm_seed.get("CE_PAIR_MARGIN", {})
    seeds = sorted(set(ce_map) & set(pm_map))
    if not seeds:
        return {"schema": "anra-avendi-v8-y2-p3-gate-v1", "status": "INCOMPLETE", "reason": "missing_paired_seeds"}

    dev_gains: dict[int, float] = {}
    conf_pm_1hop: dict[int, float] = {}
    conf_ce_1hop: dict[int, float] = {}
    lit_drops: dict[int, float] = {}

    t_fam_by_seed: dict[int, Mapping[str, Any]] = {}
    c_fam_by_seed: dict[int, Mapping[str, Any]] = {}

    for s in seeds:
        ce_dev = ce_map[s]["dev_summary"]
        pm_dev = pm_map[s]["dev_summary"]
        ce_conf = ce_map[s]["confirm_summary"]
        pm_conf = pm_map[s]["confirm_summary"]

        dev_gains[s] = float(pm_dev["one_hop"]["all_four_raw_rate"]) - float(ce_dev["one_hop"]["all_four_raw_rate"])
        conf_pm_1hop[s] = float(pm_conf["one_hop"]["all_four_raw_rate"])
        conf_ce_1hop[s] = float(ce_conf["one_hop"]["all_four_raw_rate"])
        lit_drops[s] = float(ce_dev["literal_copy"]["raw_exact_accuracy"]) - float(pm_dev["literal_copy"]["raw_exact_accuracy"])

        t_fam_by_seed[s] = pm_dev["one_hop"]["family_scores"]
        c_fam_by_seed[s] = ce_dev["one_hop"]["family_scores"]

    mean_gain = sum(dev_gains.values()) / len(dev_gains)
    pos_every_seed = all(g > 0.0 for g in dev_gains.values())
    pm_conf_pass = all(v >= float(cfg["min_confirm_one_hop_all_four_per_seed"]) for v in conf_pm_1hop.values())
    ce_conf_pass = all(v >= float(cfg["min_confirm_one_hop_all_four_per_seed"]) for v in conf_ce_1hop.values())
    lit_ok = all(d <= float(cfg["max_same_seed_literal_drop"]) for d in lit_drops.values())

    pair_margin_supported = (
        mean_gain >= float(cfg["min_mean_complete_family_gain"])
        and pos_every_seed
        and pm_conf_pass
        and lit_ok
    )
    if pair_margin_supported:
        selected_arm = "CE_PAIR_MARGIN"
        stage_status = "QUALIFIED"
    elif ce_conf_pass:
        # "If both qualify equally, use CE for the simpler qualified operation."
        selected_arm = "CE"
        stage_status = "QUALIFIED"
    else:
        selected_arm = None
        stage_status = "FAILED"

    boot = bootstrap_paired_family_contrasts(t_fam_by_seed, c_fam_by_seed, metric_key="all_four_raw_correct")
    return {
        "schema": "anra-avendi-v8-y2-p3-gate-v1",
        "status": stage_status,
        "pair_margin_supported": pair_margin_supported,
        "ce_standalone_qualified": ce_conf_pass,
        "selected_arm": selected_arm,
        "mean_dev_one_hop_family_gain": mean_gain,
        "dev_gain_by_seed": {str(k): v for k, v in dev_gains.items()},
        "confirm_one_hop_all_four_pm_by_seed": {str(k): v for k, v in conf_pm_1hop.items()},
        "confirm_one_hop_all_four_ce_by_seed": {str(k): v for k, v in conf_ce_1hop.items()},
        "same_seed_literal_drop_by_seed": {str(k): v for k, v in lit_drops.items()},
        "bootstrap_pm_minus_ce": boot,
    }
