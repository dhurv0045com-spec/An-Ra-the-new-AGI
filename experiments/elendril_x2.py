"""Registered Palantiri / Planthir plugin backend `elendril_x2` (PLANTHIR_BACKEND_MODULES=experiments.elendril_x2)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from anra_v5.elendril.codec import digest
from anra_v5.elendril.v8.contracts import SYMBOLS
from anra_v5.planthir.runtime import ExperimentModule, GenericDatasetBundle

from .x2.models import (
    OP_TO_INDEX,
    X2_ARCHITECTURES,
    X2TypedModel,
    build_s20_base_spec,
    build_tiny_canary_base_spec,
    build_x2_model,
    build_x2_model_spec,
)
from .x2.tasks import (
    T13VersionSpaceSolver,
    X2EnvironmentKernel,
    X2ObservablePolicy,
    build_generator_b_multihop_panel,
    build_selective_edit_panel,
    build_t13_decisive_fixture,
    build_t14_panel,
    build_t14_price_crossover_fixture,
    encode_x2_text,
    solve_t14_public_dp,
)

X2_BACKEND_ID = "elendril_x2"
X2_OBJECTIVES = ("standard", "PAIR-CE", "PAIR-EQ", "PAIR-EQ-CF", "T13_ADAPT", "T14_BELIEF")


def _encode_batch_samples(samples: Sequence[Mapping[str, Any]], device: torch.device, max_ctx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = [encode_x2_text(str(s["rendered_text"]), max_length=max_ctx) for s in samples]
    max_len = max(len(t) for t, _ in encoded)
    toks = torch.zeros((len(encoded), max_len), dtype=torch.long, device=device)
    segs = torch.full((len(encoded), max_len), -1, dtype=torch.long, device=device)
    apos = torch.zeros((len(encoded),), dtype=torch.long, device=device)
    for i, (seq, ap) in enumerate(encoded):
        toks[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        segs[i, : len(seq)] = 0
        apos[i] = ap
    return toks, segs, apos


def compute_x2_grouped_loss(
    log_probs_19: torch.Tensor,
    log_p_op: torch.Tensor,
    log_p_val: torch.Tensor,
    *,
    gold_ops: torch.Tensor,
    gold_args: torch.Tensor,
    state_mask: torch.Tensor,
    eq_pairs: Sequence[tuple[int, int, Sequence[int], bool]] = (),
    cf_pairs: Sequence[tuple[int, int, int, int, bool]] = (),
    eq_weight: float = 0.1,
    cf_weight: float = 0.1,
    cf_margin: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Vectorized X2 grouped D2 loss (CE + JS equivariance + CF margin) preserving exact empty-pair & mask semantics."""
    device = log_probs_19.device
    sm = state_mask.to(dtype=torch.float32)
    active_states = float(sm.sum().item())
    zero = log_probs_19.sum() * 0.0

    op_nll = -log_p_op.float().gather(1, gold_ops.unsqueeze(1)).squeeze(1)
    is_ans = (gold_ops == 0).to(dtype=torch.float32)
    val_nll = -log_p_val.float().gather(1, gold_args.clamp(0, 15).unsqueeze(1)).squeeze(1) * is_ans
    ce_loss = ((op_nll + val_nll) * sm).sum() / max(active_states, 1.0) if active_states > 0.0 else zero

    eq_loss = zero
    elig_eq = [p for p in eq_pairs if p[3]]
    if elig_eq and eq_weight > 0.0:
        ia = torch.tensor([p[0] for p in elig_eq], dtype=torch.long, device=device)
        ib = torch.tensor([p[1] for p in elig_eq], dtype=torch.long, device=device)
        perm = torch.tensor([list(p[2]) for p in elig_eq], dtype=torch.long, device=device)
        p19 = log_probs_19.float().exp().clamp_min(1e-12)
        pa, pb = p19[ia], p19[ib]
        pa_ans = torch.zeros_like(pa[:, :16])
        pa_ans.scatter_add_(1, perm, pa[:, :16])
        pa_perm = torch.cat([pa_ans, pa[:, 16:]], dim=-1).clamp_min(1e-12)
        pa_perm = pa_perm / pa_perm.sum(dim=-1, keepdim=True)
        pb_norm = pb / pb.sum(dim=-1, keepdim=True)
        mid = 0.5 * (pa_perm + pb_norm)
        js = 0.5 * (
            (pa_perm * (pa_perm.log() - mid.log())).sum(dim=-1)
            + (pb_norm * (pb_norm.log() - mid.log())).sum(dim=-1)
        )
        eq_loss = js.mean()

    cf_loss = zero
    elig_cf = [p for p in cf_pairs if p[4]]
    if elig_cf and cf_weight > 0.0:
        ia = torch.tensor([p[0] for p in elig_cf], dtype=torch.long, device=device)
        ib = torch.tensor([p[1] for p in elig_cf], dtype=torch.long, device=device)
        ta = torch.tensor([p[2] for p in elig_cf], dtype=torch.long, device=device)
        tb = torch.tensor([p[3] for p in elig_cf], dtype=torch.long, device=device)
        lp = log_probs_19.float()
        m_a = F.relu(float(cf_margin) - (lp[ia, ta] - lp[ia, tb]))
        m_b = F.relu(float(cf_margin) - (lp[ib, tb] - lp[ib, ta]))
        cf_loss = 0.5 * (m_a + m_b).mean()

    total = ce_loss + float(eq_weight) * eq_loss + float(cf_weight) * cf_loss
    return total, {
        "ce_loss": float(ce_loss.detach().item()),
        "eq_loss": float(eq_loss.detach().item()),
        "cf_loss": float(cf_loss.detach().item()),
        "active_states": active_states,
        "eligible_eq_pairs": float(len(elig_eq)),
        "eligible_cf_pairs": float(len(elig_cf)),
    }


def compute_x2_grouped_loss_scalar_reference(
    log_probs_19: torch.Tensor,
    log_p_op: torch.Tensor,
    log_p_val: torch.Tensor,
    *,
    gold_ops: Sequence[int],
    gold_args: Sequence[int],
    state_mask: Sequence[bool],
    eq_pairs: Sequence[tuple[int, int, Sequence[int], bool]] = (),
    cf_pairs: Sequence[tuple[int, int, int, int, bool]] = (),
    eq_weight: float = 0.1,
    cf_weight: float = 0.1,
    cf_margin: float = 0.5,
) -> torch.Tensor:
    """Explicit element-by-element scalar reference for §0.2/§9 loss & gradient parity testing."""
    ce_terms = []
    for i, active in enumerate(state_mask):
        if not active:
            continue
        term = -log_p_op[i, int(gold_ops[i])]
        if int(gold_ops[i]) == 0:
            term = term - log_p_val[i, int(gold_args[i])]
        ce_terms.append(term)
    ce_loss = sum(ce_terms) / len(ce_terms) if ce_terms else (log_probs_19.sum() * 0.0)

    eq_terms = []
    p19 = log_probs_19.float().exp().clamp_min(1e-12)
    for ia, ib, perm, elig in eq_pairs:
        if not elig:
            continue
        pa = p19[ia]
        pb = p19[ib]
        pa_ans = [log_probs_19.sum() * 0.0 for _ in range(16)]
        for src_i, dst_i in enumerate(perm):
            pa_ans[int(dst_i)] = pa_ans[int(dst_i)] + pa[src_i]
        pa_perm = torch.cat([torch.stack(pa_ans), pa[16:]], dim=0).clamp_min(1e-12)
        pa_perm = pa_perm / pa_perm.sum()
        pb_norm = pb / pb.sum()
        mid = 0.5 * (pa_perm + pb_norm)
        js = 0.5 * ((pa_perm * (pa_perm.log() - mid.log())).sum() + (pb_norm * (pb_norm.log() - mid.log())).sum())
        eq_terms.append(js)
    eq_loss = sum(eq_terms) / len(eq_terms) if (eq_terms and eq_weight > 0.0) else (log_probs_19.sum() * 0.0)

    cf_terms = []
    for ia, ib, ta, tb, elig in cf_pairs:
        if not elig:
            continue
        m_a = F.relu(float(cf_margin) - (log_probs_19[ia, ta] - log_probs_19[ia, tb]))
        m_b = F.relu(float(cf_margin) - (log_probs_19[ib, tb] - log_probs_19[ib, ta]))
        cf_terms.append(0.5 * (m_a + m_b))
    cf_loss = sum(cf_terms) / len(cf_terms) if (cf_terms and cf_weight > 0.0) else (log_probs_19.sum() * 0.0)

    return ce_loss + float(eq_weight) * eq_loss + float(cf_weight) * cf_loss


def _world_to_supervised_sample(
    kernel: X2EnvironmentKernel,
    world: Mapping[str, Any],
    *,
    t13_solver: T13VersionSpaceSolver | None = None,
) -> dict[str, Any]:
    st = kernel.initial_state(world)
    obs = kernel.observe(st)
    tt = str(world.get("task_type", "selective_edit"))

    if tt == "t14_costly":
        dp = solve_t14_public_dp(world)
        op_name = dp["optimal_operation"]
        gold_op = OP_TO_INDEX[op_name]
        active_labels = list(world["public_belief"]["active_labels"])
        gold_arg = SYMBOLS.index(active_labels[0]) if active_labels[0] in SYMBOLS else 0
        prior = list(world["public_belief"]["prior_probs"])
        soft_val = [0.0] * 16
        for lbl, p in zip(active_labels, prior):
            if lbl in SYMBOLS:
                soft_val[SYMBOLS.index(lbl)] = float(p)
        cert_target = 1.0 if bool(world["public_belief"].get("certified", False)) else 0.0
        # 4 forecast branches (Q, R, Q->R, R->Q) -> 4 outcome classes (0=no_cert, 1=cert_A, 2=cert_B, 3=cert_other)
        h = world.get("hidden", {})
        q_cls = 1 if h.get("query_certifies") else 0
        r_cls = 2 if h.get("retrieve_certifies") else 0
        forecast_targets = [q_cls, r_cls, max(q_cls, r_cls), max(q_cls, r_cls)]
        return {
            "world_id": str(world["world_id"]),
            "task_type": tt,
            "rendered_text": obs.rendered_text,
            "gold_op": int(gold_op),
            "gold_arg": int(gold_arg),
            "soft_val_probs": soft_val,
            "certifiable_target": float(cert_target),
            "forecast_targets": forecast_targets,
            "ambiguous": not bool(cert_target > 0.5),
        }

    hidden = world.get("hidden", {})
    exp_op = str(hidden.get("expected_operation", "ANSWER"))
    exp_ans = str(hidden.get("expected_answer", "A"))
    gold_op = OP_TO_INDEX.get(exp_op, 0)
    gold_arg = SYMBOLS.index(exp_ans) if exp_ans in SYMBOLS else 0
    soft_val = [0.0] * 16
    ambiguous = False
    if tt == "t13_fst" and t13_solver is not None:
        post = t13_solver.posterior(world.get("support", ()), world["query"])
        ambiguous = not bool(post["identifiable"])
        for idx, p in enumerate(post["output_probs"]):
            soft_val[idx] = float(p)
    else:
        soft_val[gold_arg] = 1.0
    return {
        "world_id": str(world["world_id"]),
        "task_type": tt,
        "rendered_text": obs.rendered_text,
        "gold_op": int(gold_op),
        "gold_arg": int(gold_arg),
        "soft_val_probs": soft_val,
        "certifiable_target": 0.0 if ambiguous else 1.0,
        "forecast_targets": [0, 0, 0, 0],
        "ambiguous": ambiguous,
    }


def x2_model_factory(architecture: str, model_config: Mapping[str, Any], seed: int) -> X2TypedModel:
    """Registered §0.2 model_factory(architecture, model_config, seed)."""
    cfg = dict(model_config)
    use_s20 = bool(cfg.pop("s20_backbone", False)) or int(cfg.get("width", 16)) >= 256
    if use_s20:
        base_spec = build_s20_base_spec(context_length=int(cfg.get("context_length", 640)))
    else:
        base_spec = build_tiny_canary_base_spec(
            width=int(cfg.get("width", 16)),
            layers=int(cfg.get("layers", 1)),
            query_heads=int(cfg.get("query_heads", 2)),
            kv_heads=int(cfg.get("kv_heads", 1)),
            head_dimension=int(cfg.get("head_dimension", 8)),
            ffn_width=int(cfg.get("ffn_width", 32)),
            context_length=int(cfg.get("context_length", 256)),
        )
    spec = build_x2_model_spec(architecture, base_spec=base_spec, tiny_canary=not use_s20, **cfg)
    return build_x2_model(spec, int(seed), device="cpu")


def x2_dataset_factory(dataset_config: Mapping[str, Any], cpu_canary: bool) -> dict[str, Any]:
    """Registered §0.2 dataset_factory with complete §0.3 CPU fixture and distinct canary vs production hashes."""
    kernel = X2EnvironmentKernel()
    t13_solver = T13VersionSpaceSolver(max_laws=512)
    seed = int(dataset_config.get("seed", 7))
    ds_name = str(dataset_config.get("dataset_name", "x2_canary_fixture" if cpu_canary else "x2_production_stream"))

    if cpu_canary:
        # Complete §0.3 fixture: binding + correction + dependent + untouched sibling + T13 witness + T14 crossover
        sel_train = build_selective_edit_panel(num_histories=4, seed=seed + 11, split="train")
        t13_fix = build_t13_decisive_fixture(witness_position="late")
        t14_fix = build_t14_price_crossover_fixture()

        train_worlds = [
            *sel_train[:6],
            {**t13_fix["worlds"]["id_disc"], "world_id": "tr_t13_id", "sig": "tr_t13_id"},
            {**t13_fix["worlds"]["swap_disc"], "world_id": "tr_t13_swap", "sig": "tr_t13_swap"},
            {**t13_fix["worlds"]["ambiguous_disc_a"], "world_id": "tr_t13_ambig", "sig": "tr_t13_ambig"},
            {**t14_fix["cheap_cert"], "world_id": "tr_t14_cheap", "sig": "tr_t14_cheap"},
            {**t14_fix["expensive_nocert"], "world_id": "tr_t14_exp", "sig": "tr_t14_exp"},
            {**t14_fix["cheap_nocert"], "world_id": "tr_t14_cheap_nc", "sig": "tr_t14_cheap_nc"},
        ]
        sel_eval = build_selective_edit_panel(num_histories=2, seed=seed + 509, split="dev")
        eval_worlds = [
            sel_eval[0],  # edited dependent
            sel_eval[1],  # untouched sibling
            t13_fix["worlds"]["id_inv"],
            t14_fix["cheap_cert"],
            t14_fix["expensive_cert"],
        ]
        gen_b = build_generator_b_multihop_panel(split="dev", num_skeletons=1, seed=seed + 907, depths=(3, 4), include_exhaustive_perm=False)
        multihop_worlds = gen_b["episodes"][:4]
        probe_train_worlds = build_selective_edit_panel(num_histories=2, seed=seed + 313, split="probe")
        samples = tuple(_world_to_supervised_sample(kernel, w, t13_solver=t13_solver) for w in train_worlds)
        data_id = digest(["elendril-x2-cpu-canary-v1", ds_name, seed, [w["world_id"] for w in train_worlds]])
    else:
        n_sel = int(dataset_config.get("num_selective_histories", 64))
        n_t14 = int(dataset_config.get("t14_per_stratum", 8))
        sel_train = build_selective_edit_panel(num_histories=n_sel, seed=seed + 1001, split="train")
        t14_train = build_t14_panel(split="train", histories_per_stratum=n_t14, seed=seed + 2001)
        train_worlds = [*sel_train, *t14_train]
        sel_eval = build_selective_edit_panel(num_histories=16, seed=seed + 5001, split="dev")
        t14_eval = build_t14_panel(split="dev", histories_per_stratum=4, seed=seed + 6001)
        eval_worlds = [*sel_eval, *t14_eval]
        gen_b = build_generator_b_multihop_panel(split="dev", num_skeletons=4, seed=seed + 7001, depths=(3, 4, 5), include_exhaustive_perm=False)
        multihop_worlds = gen_b["episodes"][:16]
        probe_train_worlds = build_selective_edit_panel(num_histories=8, seed=seed + 3001, split="probe")
        samples = tuple(_world_to_supervised_sample(kernel, w, t13_solver=t13_solver) for w in train_worlds)
        data_id = digest(["elendril-x2-production-v1", ds_name, seed, len(train_worlds), len(eval_worlds)])

    return {
        "train_data": GenericDatasetBundle(
            items=samples,
            dataset_name=ds_name,
            data_identity=data_id,
            manifest={
                "schema": "elendril-x2-dataset-v1",
                "dataset_name": ds_name,
                "cpu_canary": bool(cpu_canary),
                "data_identity": data_id,
                "num_train_samples": len(samples),
                "num_eval_worlds": len(eval_worlds),
                "license": "internal-research",
            },
        ),
        "train_worlds": train_worlds,
        "eval_worlds": eval_worlds,
        "multihop_worlds": multihop_worlds,
        "probe_train_worlds": probe_train_worlds,
    }


def x2_loss_fn(model: X2TypedModel, batch: Sequence[Mapping[str, Any]], recipe: Any) -> torch.Tensor:
    """Registered §0.2 loss_fn handling standard, grouped PAIR-*, T13_ADAPT, and T14_BELIEF objectives."""
    dev = next(model.parameters()).device
    max_ctx = int(model.x2_spec.base_spec.context_length)
    toks, segs, apos = _encode_batch_samples(batch, dev, max_ctx)
    out = model.forward_action(toks, segs, apos)

    gold_ops = torch.tensor([int(b["gold_op"]) for b in batch], dtype=torch.long, device=dev)
    gold_args = torch.tensor([int(b["gold_arg"]) for b in batch], dtype=torch.long, device=dev)
    soft_val = torch.tensor([list(b["soft_val_probs"]) for b in batch], dtype=torch.float32, device=dev)
    cert_tgt = torch.tensor([float(b["certifiable_target"]) for b in batch], dtype=torch.float32, device=dev)
    ambig_mask = torch.tensor([bool(b.get("ambiguous", False)) for b in batch], dtype=torch.bool, device=dev)

    # Identifiable action NLL + soft-target loss for ambiguous T13 items (§6)
    op_nll = -out.log_p_op.float().gather(1, gold_ops.unsqueeze(1)).squeeze(1)
    is_ans = (gold_ops == 0).to(torch.float32)
    hard_val_nll = -out.log_p_val.float().gather(1, gold_args.clamp(0, 15).unsqueeze(1)).squeeze(1)
    soft_val_nll = -(soft_val * out.log_p_val.float()).sum(dim=-1)
    val_nll = torch.where(ambig_mask, soft_val_nll, hard_val_nll) * is_ans
    action_loss = (op_nll + val_nll).mean()

    # Proper categorical provisional belief loss + separate certifiability BCE (§7, §11)
    prov_logp = F.log_softmax(out.provisional_val_logits.float(), dim=-1)
    prov_loss = -(soft_val * prov_logp).sum(dim=-1).mean()
    cert_loss = F.binary_cross_entropy_with_logits(out.certifiable_logit.float(), cert_tgt)

    loss = action_loss + 0.1 * prov_loss + 0.1 * cert_loss

    # Predictive-state public-outcome forecast supervision (identical across PS and PS-ctrl per §4)
    if out.forecast_logits is not None:
        ftgt = torch.tensor([list(b.get("forecast_targets", [0, 0, 0, 0])) for b in batch], dtype=torch.long, device=dev)
        f_loss = F.cross_entropy(out.forecast_logits.reshape(-1, out.forecast_logits.shape[-1]), ftgt.reshape(-1))
        loss = loss + 0.2 * f_loss

    obj = str(getattr(recipe, "objective", "standard"))
    if obj in ("PAIR-EQ", "PAIR-EQ-CF") and len(batch) >= 2:
        eq_w = float(getattr(recipe, "eq_weight", 0.1)) or 0.1
        cf_w = float(getattr(recipe, "cf_weight", 0.1)) if obj == "PAIR-EQ-CF" else 0.0
        cf_m = float(getattr(recipe, "cf_margin", 0.5))
        id_perm = list(range(16))
        eq_pairs = [(0, 1, id_perm, True)] if gold_ops[0].item() == gold_ops[1].item() else []
        cf_pairs = [(0, 1, int(gold_args[0].item()), int(gold_args[1].item()), True)] if gold_args[0].item() != gold_args[1].item() else []
        grp_loss, _ = compute_x2_grouped_loss(
            out.log_probs_19,
            out.log_p_op,
            out.log_p_val,
            gold_ops=gold_ops,
            gold_args=gold_args,
            state_mask=torch.ones(len(batch), dtype=torch.bool, device=dev),
            eq_pairs=eq_pairs,
            cf_pairs=cf_pairs,
            eq_weight=eq_w,
            cf_weight=cf_w,
            cf_margin=cf_m,
        )
        loss = grp_loss + 0.1 * prov_loss + 0.1 * cert_loss

    return loss


def x2_kernel_factory() -> X2EnvironmentKernel:
    return X2EnvironmentKernel(max_actions=4)


def x2_policy_factory(
    model: X2TypedModel,
    model_spec: Any,
    *,
    model_identity: str = "elendril-x2-policy",
    mode: str = "raw",
    copy_enabled_override: bool | None = None,
) -> X2ObservablePolicy:
    return X2ObservablePolicy(
        model,
        model_spec,
        model_identity=model_identity,
        mode=mode,
        copy_enabled_override=copy_enabled_override,
    )


def build_elendril_x2_module() -> ExperimentModule:
    return ExperimentModule(
        backend_id=X2_BACKEND_ID,
        backend_version="2.0",
        program="ELENDRIL-V8-X2",
        architectures=X2_ARCHITECTURES,
        objectives=X2_OBJECTIVES,
        model_factory=x2_model_factory,
        dataset_factory=x2_dataset_factory,
        loss_fn=x2_loss_fn,
        kernel_factory=x2_kernel_factory,
        policy_factory=x2_policy_factory,
        default_dataset_config={"dataset_name": "x2_canary_fixture", "seed": 7},
        default_model_config={"width": 16, "layers": 1, "query_heads": 2, "kv_heads": 1, "head_dimension": 8, "ffn_width": 32, "context_length": 256, "mem_dim": 16},
        treatment_architecture="DM",
        treatment_model_config={"width": 16, "layers": 1, "query_heads": 2, "kv_heads": 1, "head_dimension": 8, "ffn_width": 32, "context_length": 256, "mem_dim": 16},
        extra_model_keys=(
            "s20_backbone",
            "context_length",
            "ffn_width",
            "head_dimension",
            "kv_heads",
            "layers",
            "query_heads",
            "width",
            "pointer_dim",
            "num_roles",
            "dense_hidden",
            "slot_dim",
            "num_slots",
            "slot_edge_hidden",
            "pool_hidden",
            "u_dim",
            "u_steps",
            "u_par_hidden",
            "mem_dim",
            "forecast_hidden",
            "forecast_branches",
            "forecast_classes",
        ),
        extra_dataset_keys=("dataset_name", "seed", "num_selective_histories", "t14_per_stratum"),
        extra_recipe_keys=("eq_weight", "cf_weight", "cf_margin"),
        supports_grouped_loss=True,
    )


def register_planthir_backends() -> ExperimentModule:
    """Entry point invoked automatically by Palantiri when PLANTHIR_BACKEND_MODULES=experiments.elendril_x2."""
    mod = build_elendril_x2_module()
    mod.register()
    return mod


__all__ = [
    "X2_BACKEND_ID",
    "X2_OBJECTIVES",
    "build_elendril_x2_module",
    "compute_x2_grouped_loss",
    "compute_x2_grouped_loss_scalar_reference",
    "register_planthir_backends",
    "x2_dataset_factory",
    "x2_kernel_factory",
    "x2_loss_fn",
    "x2_model_factory",
    "x2_policy_factory",
]
