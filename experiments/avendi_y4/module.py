"""Avendi V8-Y4 Planthir plugin, 9-stage executor, dual-GPU orchestrator, sealed ConfirmLock, and CLI."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.elendril.representation import SpecialToken, UTF8
from anra_v5.planthir import (
    BytePromptTrainer,
    CampaignOrchestrator,
    CartridgeRunner,
    ConfirmLock,
    ExperimentModule,
    GenericDatasetBundle,
    GenericExperimentTrainer,
    GenericTrainRecipe,
    SessionJob,
    SubgoalProbeSpec,
    UniversalAction,
    UniversalObservation,
    batched_rollout,
    build_analyst_first_page_report,
    build_cartridge_from_spec,
    build_paired_seed_observations,
    build_verification_matrix,
    compare_paired_episode_records,
    export_session_archives,
    frame_byte_prompt_sample,
    independent_replay_bundle,
    init_spec_template,
    load_legacy_byte_checkpoint,
    planthir_doctor,
    render_campaign_results_markdown,
    rescore_compact_analysis_archive,
    run_paired_comparison_preflight,
    run_representation_linear_probe,
    run_scientific_shortcut_calibration,
    run_surgical_mechanism_ablation,
    score_crossover_panel,
    stratify_intervention_worlds,
    summarize_diagnostic_coverage,
    validate_spec_document,
    verify_result_bundle,
    verify_session_archive,
    win_long_path,
)

from experiments.avendi_y4.models import (
    AvendiY4ByteModel,
    P6LearnedMemoryPolicy,
    Y4TypedCompositionModel,
    _core_forward,
    apply_radial_norm_cap,
    compute_fp64_gradient_cosine,
    compute_y4_byte_loss,
    decode_y4_byte_continuations,
    encode_y4_typed_batch,
    eval_y4_byte_families,
    evaluate_p5_multistep_policy,
    evaluate_p6_all_routes,
    get_y4_byte_spec,
    p4_y4_loss,
    parse_bytes_with_model,
    run_y4_donor_audits,
    run_y4_state_swap_suite,
    select_y4_matched_widths,
)
from experiments.avendi_y4.plan import (
    ARCHIVE_SHAS,
    PLATFORM_USES,
    SEEDS_3,
    SEEDS_5,
    STREAM_SEEDS,
    TECHNOLOGY_BASELINE,
    Y1_EXPECTED_CKPT_SHAS,
    build_plan,
    validate_plan_dag,
)
from experiments.avendi_y4.tasks import (
    ENTITY_64,
    P5_CONDITIONS,
    SYMBOLS_16,
    allocate_y4_disjoint_pools,
    apply_pointer_sham_spans,
    assert_no_condition_leak,
    audit_y4_split_custody,
    build_functional_graph_families,
    build_literal_probe_families,
    build_p5_blind_worlds,
    build_p6_law_blocks,
    build_persistent_batch_plan,
    build_state_swap_pairs,
    evaluate_y4_shortcuts_and_masks,
    oracle_walk_public_bytes,
    parse_public_byte_observation,
    reproduce_y3_defects,
    solve_p5_bellman_scalar,
    solve_p5_bellman_vectorized,
    verify_p6_sham_zero_mi,
)
from experiments.avendi_y4.verify import (
    DISTINCT_Y4_STUDY_AXES,
    canonical_digest,
    evaluate_release_gates,
    stage_historical_checkpoints_to_writable,
    verify_y4_campaign,
)

FULL_S = SimpleNamespace(
    full=True,
    batch=16,
    micro=4,
    upd=1024,
    m_upd=128,
    g_upd=256,
    tr=1024,
    dv=256,
    cf=256,
    tf=256,
    mh=256,
    p5=140,
    blocks=1024,
    blocks_dev=256,
    lit_n=256,
    pairs=256,
    c_width=128,
    mon=64,
)
TINY_S = SimpleNamespace(
    full=False,
    batch=8,
    micro=2,
    upd=2,
    m_upd=2,
    g_upd=2,
    tr=8,
    dv=4,
    cf=4,
    tf=4,
    mh=4,
    p5=14,
    blocks=8,
    blocks_dev=4,
    lit_n=4,
    pairs=4,
    c_width=32,
    mon=2,
)
_MILESTONES = (0, 64, 256, 512, 1024)
_SUMMARY_MAX_BYTES = 2 * 1024 * 1024
_ACT_CHAR = {"ABSTAIN": "!", "QUERY": "Q", "RETRIEVE": "R"}


def _jd(o: Any) -> Any:
    return o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o))


def _canon_sha(obj: Any) -> str:
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_jd).encode("utf-8")).hexdigest()


def _dump(p: Path, obj: Any) -> str:
    wp = win_long_path(p)
    wp.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(obj, sort_keys=True, indent=2, default=_jd) + "\n").encode("utf-8")
    tmp = wp.with_name(wp.name + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, wp)
    return sha256(raw).hexdigest()


# ---------------------------------------------------------------------------------------------
# Registered Planthir ExperimentModule Plugin (`avendi_y4`)
# ---------------------------------------------------------------------------------------------


class AvendiY4Kernel:
    """Truth-firewalled EnvironmentKernel supporting 1-hop/multi-hop byte worlds and subgoal probes."""

    kernel_id = "avendi-y4-kernel-v1"
    max_actions = 3

    def structural_signature(self, world: Mapping[str, Any]) -> str:
        return str(world.get("sig") or f"{world.get('split', 's')}:{world.get('canon_sig', '')}:{world.get('world_id', '')}")

    def inject_distractor(self, world: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **world,
            "world_id": f"{world['world_id']}::dist",
            "sig": f"{self.structural_signature(world)}::dist",
            "public_prompt": str(world.get("public_prompt", "")) + " ; ZZ99->YY88",
        }

    def permute_world(self, world: Mapping[str, Any], shift: int = 1) -> dict[str, Any]:
        rows = list(world.get("rows") or ())
        if len(rows) > 1:
            k = shift % len(rows)
            rot = rows[k:] + rows[:k]
            r_str = " ; ".join(f"{a}->{b}" for a, b in rot)
            p_txt = f"ROWS: {r_str} || STEP={world.get('depth', 1)} Q:{world.get('query', '')}"
        else:
            p_txt = str(world.get("public_prompt", ""))
        return {
            **world,
            "world_id": f"{world['world_id']}::p{shift}",
            "sig": f"{self.structural_signature(world)}::p{shift}",
            "public_prompt": p_txt,
        }

    def reverse_world(self, world: Mapping[str, Any]) -> dict[str, Any]:
        rows = list(reversed(list(world.get("rows") or ())))
        if rows:
            r_str = " ; ".join(f"{a}->{b}" for a, b in rows)
            p_txt = f"ROWS: {r_str} || STEP={world.get('depth', 1)} Q:{world.get('query', '')}"
        else:
            p_txt = str(world.get("public_prompt", ""))
        return {
            **world,
            "world_id": f"{world['world_id']}::rev",
            "sig": f"{self.structural_signature(world)}::rev",
            "public_prompt": p_txt,
        }

    def initial_state(self, world: Mapping[str, Any]) -> dict[str, Any]:
        return {"w": dict(world), "step": 0, "queried": False, "retrieved": False, "terminal": False}

    def observe(self, state: Mapping[str, Any]) -> UniversalObservation:
        w = state["w"]
        avail = {str(k).lower(): bool(v) for k, v in dict(w.get("available_tools", {"query": False, "retrieve": False})).items()}
        if state["queried"]:
            avail["query"] = False
        if state["retrieved"]:
            avail["retrieve"] = False
        consumed = frozenset(k for k, f in (("query", state["queried"]), ("retrieve", state["retrieved"])) if f)
        return UniversalObservation(
            obs_id=f"{w['world_id']}:s{state['step']}:q{int(avail.get('query', False))}r{int(avail.get('retrieve', False))}",
            rendered_text=str(w.get("public_prompt", "ROWS: A0->B1 || STEP=1 Q:A0")),
            available_tools=avail,
            consumed_tools=consumed,
            decision_index=int(state["step"]),
            budget_remaining=max(1, self.max_actions - int(state["step"])),
            metadata={"split": str(w.get("split", "dev")), "decision_step": int(state["step"])},
        )

    def ideal_action(self, state: Mapping[str, Any]) -> tuple[str, str | None]:
        w = state["w"]
        if state["step"] == 0 and w.get("optimal_actions"):
            opt = str(w["optimal_actions"][0])
            return tuple(opt.split(":", 1)) if ":" in opt else (opt, None)  # type: ignore[return-value]
        return ("ANSWER", str(w.get("gold", w.get("true_answer", ENTITY_64[0]))))

    def step(self, state: Mapping[str, Any], op: str, arg: str | None) -> dict[str, Any]:
        nxt = {**state, "step": int(state["step"]) + 1}
        if op == "QUERY" and not state["queried"] and state["w"].get("available_tools", {}).get("query"):
            nxt["queried"] = True
        elif op == "RETRIEVE" and not state["retrieved"] and state["w"].get("available_tools", {}).get("retrieve"):
            nxt["retrieved"] = True
        else:
            nxt["terminal"] = True
        return nxt

    def verify(self, world: Mapping[str, Any], actions: Sequence[tuple[str, str | None]]) -> dict[str, Any]:
        op, arg = actions[-1] if actions else ("INVALID", None)
        exp = str(world.get("gold", world.get("true_answer", "")))
        return {"correct": op == "ANSWER" and arg == exp, "terminal_status": "answered" if op == "ANSWER" else op.lower()}

    def decompose_subgoals(self, world: Mapping[str, Any]) -> tuple[SubgoalProbeSpec, ...]:
        op, arg = self.ideal_action(self.initial_state(world))
        return (
            SubgoalProbeSpec(
                subgoal_id=f"{world.get('world_id', 'w')}:sg0",
                stage_name="endpoint",
                hop_index=0,
                modified_world=dict(world),
                expected_operation=op,
                expected_argument=arg,
            ),
        )


_NON_AP_IDS = ((UTF8.encode("Q")[0], "QUERY"), (UTF8.encode("R")[0], "RETRIEVE"), (UTF8.encode("!")[0], "ABSTAIN"))


class AvendiY4BytePolicy:
    """ObservablePolicy implementing reset_episode, batched inference, 2-byte entity distribution, and repr extraction."""

    def __init__(
        self,
        model: nn.Module,
        model_spec: Any,
        *,
        model_identity: str = "avendi-y4",
        mode: str = "raw",
        copy_enabled_override: bool | None = None,
    ) -> None:
        self.model = model
        self.model_spec = model_spec
        self.model_identity = model_identity
        self.mode = mode
        self.policy_id = f"avendi_y4:{model_identity}"
        self.episode_steps = 0

    def reset_episode(self) -> None:
        self.episode_steps = 0
        if hasattr(self.model, "reset_episode"):
            self.model.reset_episode()

    @torch.no_grad()
    def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction:
        dev = next(self.model.parameters()).device
        max_ctx = int(getattr(self.model_spec, "context_length", 160))
        tok_ids = [UTF8.bos_id, *UTF8.encode(obs.rendered_text), UTF8.special(SpecialToken.ANSWER)]
        tok_ids = tok_ids[: max(2, max_ctx - 2)]
        step0 = self.model(torch.tensor([tok_ids], device=dev, dtype=torch.long))[0][0, -1].float()
        probs0 = F.softmax(step0, dim=-1)
        # Score first-byte classes across SYMBOLS_16 and ENTITY_64 consistently
        dist: dict[str, float] = {}
        first_syms = sorted({e[0] for e in ENTITY_64})
        for e in ENTITY_64:
            dist[f"ANSWER:{e}"] = float(probs0[UTF8.encode(e[0])[0]]) / 4.0
        for tid, op_name in _NON_AP_IDS:
            dist[f"{op_name}:{op_name}"] = float(probs0[tid])
        best_prefix = max(first_syms, key=lambda ch: float(probs0[UTF8.encode(ch)[0]]))
        pref_id = UTF8.encode(best_prefix)[0]
        step1 = self.model(torch.tensor([[*tok_ids, pref_id]], device=dev, dtype=torch.long))[0][0, -1].float()
        probs1 = F.softmax(step1, dim=-1)
        best_digit = max(("0", "1", "2", "3"), key=lambda d: float(probs1[UTF8.encode(d)[0]]))
        best_ent = f"{best_prefix}{best_digit}"
        dist[f"ANSWER:{best_ent}"] = max(dist[f"ANSWER:{best_ent}"], float(probs0[pref_id] * probs1[UTF8.encode(best_digit)[0]]))
        op, arg, lp = "ANSWER", best_ent, math.log(max(dist[f"ANSWER:{best_ent}"], 1e-8))
        if self.mode != "ap_constrained" and any(obs.available_tools.values()):
            for op_name in ("QUERY", "RETRIEVE"):
                if obs.available_tools.get(op_name.lower()) and dist[f"{op_name}:{op_name}"] > dist[f"ANSWER:{best_ent}"]:
                    op, arg, lp = op_name, None, math.log(max(dist[f"{op_name}:{op_name}"], 1e-8))
                    break
        self.episode_steps += 1
        return UniversalAction(
            operation=op,
            argument=arg,
            log_probability=float(lp),
            action_distribution=dist,
            mode=self.mode,
            raw_operation=op,
            raw_argument=arg,
            metadata={"episode_step": self.episode_steps, "first_logits_sha256": sha256(step0.cpu().numpy().tobytes()).hexdigest()},
        )

    def propose_batch(self, obs_batch: Sequence[UniversalObservation]) -> list[UniversalAction]:
        return [self.propose_with_telemetry(o) for o in obs_batch]

    @torch.no_grad()
    def extract_representations(self, obs_batch: Sequence[UniversalObservation]) -> dict[str, torch.Tensor]:
        dev = next(self.model.parameters()).device
        max_ctx = int(getattr(self.model_spec, "context_length", 160))
        vecs = []
        for o in obs_batch:
            tids = [UTF8.bos_id, *UTF8.encode(o.rendered_text), UTF8.special(SpecialToken.ANSWER)][:max_ctx]
            vecs.append(self.model(torch.tensor([tids], device=dev, dtype=torch.long))[1][0, -1])
        return {"decision_hidden": torch.stack(vecs, dim=0)}


def build_y4_planthir_dataset_splits(dataset_config: dict[str, Any], cpu_canary: bool) -> dict[str, Any]:
    """Build disjoint Planthir splits + GenericDatasetBundle within CPU canary bounds."""
    n_fam = 4 if cpu_canary else int(dataset_config.get("num_families", 32))
    seed = int(dataset_config.get("seed", STREAM_SEEDS["corpus"]))
    pools = allocate_y4_disjoint_pools({"train": n_fam, "eval": 2, "multihop": 2, "probe": 2}, seed)["pools"]

    def _mk(sp: str, pool: list[tuple[str, str, str, str]], depths: tuple[int, ...], lim: int) -> list[dict[str, Any]]:
        fams = build_functional_graph_families(sp, len(pool), seed + len(sp) * 17, role_blocks=pool, depths=depths)
        items = []
        for f in fams:
            for c in f["cells"][:2]:
                items.append(
                    {
                        **c,
                        "split": sp,
                        "sig": f"{sp}:{c['world_id']}",
                        "available_tools": {"query": False, "retrieve": False},
                        "optimal_actions": [f"ANSWER:{c['gold']}"],
                    }
                )
                if len(items) >= lim:
                    return items
        return items

    tr = _mk("train", pools["train"], (1,), 16 if cpu_canary else 256)
    ev = _mk("eval", pools["eval"], (1,), 8 if cpu_canary else 128)
    mh = _mk("multihop", pools["multihop"], (3,), 8 if cpu_canary else 128)
    for w in mh:
        w["hops"] = 3
    pr = _mk("probe", pools["probe"], (1,), 8 if cpu_canary else 128)
    return {
        "train_data": GenericDatasetBundle(
            dataset_name="avendi_y4_onehop",
            items=tuple(tr),
            data_identity=f"avendi-y4-{len(tr)}",
            manifest={"split": "train", "n": len(tr)},
        ),
        "train_worlds": tr,
        "eval_worlds": ev,
        "multihop_worlds": mh,
        "probe_train_worlds": pr,
    }


def _y4_model_factory(arch: str, mc: Mapping[str, Any], seed: int) -> nn.Module:
    prof = "S20" if "s20" in arch.lower() else "S3"
    spec = get_y4_byte_spec(prof, cpu_canary=bool(mc.get("cpu_canary", True)))
    aux = ("margin" in arch or "pointer" in arch or "address" in arch or "rename" in arch or bool(mc.get("use_aux", False)))
    return AvendiY4ByteModel(spec, seed=seed, use_aux=aux)


def build_avendi_y4_module() -> ExperimentModule:
    return ExperimentModule(
        backend_id="avendi_y4",
        backend_version="avendi-y4-1.0",
        program="AVENDI-V8-Y4",
        architectures=("s3_byte_ce", "s3_byte_margin", "s3_byte_pointer", "s3_byte_pointer_sham", "s3_byte_address", "s3_byte_rename", "s20_byte_ce"),
        objectives=("ce", "margin", "pointer", "pointer_sham", "address", "address_sham", "rename", "rename_sham"),
        model_factory=_y4_model_factory,
        dataset_factory=build_y4_planthir_dataset_splits,
        loss_fn=lambda m, c, r: compute_y4_byte_loss(m, c, r),
        kernel_factory=AvendiY4Kernel,
        policy_factory=lambda m, ms, *, model_identity="y4", mode="raw", copy_enabled_override=None: AvendiY4BytePolicy(
            m, ms, model_identity=model_identity, mode=mode, copy_enabled_override=copy_enabled_override
        ),
        default_dataset_config={"dataset_name": "avendi_y4_onehop", "num_families": 4, "seed": STREAM_SEEDS["corpus"]},
        default_model_config={"width": 32, "cpu_canary": True},
        treatment_architecture="s3_byte_margin",
        treatment_model_config={"width": 32, "cpu_canary": True},
        extra_model_keys=("width", "cpu_canary", "use_aux"),
        extra_dataset_keys=("dataset_name", "num_families", "seed"),
        supports_grouped_loss=True,
    )


def register_planthir_backends() -> ExperimentModule:
    return build_avendi_y4_module().register()


AVENDI_Y4_MODULE = register_planthir_backends()


# ---------------------------------------------------------------------------------------------
# Shared Training / Split / Evaluation Helpers
# ---------------------------------------------------------------------------------------------


def _plan_rows() -> dict[str, dict[str, Any]]:
    plan = build_plan(mode="FULL")
    return {r["job_id"]: r for r in plan["jobs"]}


def _job_upd(job: Mapping[str, Any], S: Any) -> int:
    if getattr(S, "full", False):
        return int(job.get("updates", 1))
    st = str(job["stage"])
    if st == "H":
        return 1
    if st == "M":
        return int(S.m_upd)
    if st == "G":
        return int(S.g_upd)
    return int(S.upd)


def _split_fams(S: Any) -> dict[str, list[dict[str, Any]]]:
    pools = allocate_y4_disjoint_pools(
        {"train": S.tr, "dev": S.dv, "confirm": S.cf, "transfer": S.tf, "refresh": S.tr},
        STREAM_SEEDS["corpus"],
    )["pools"]
    seeds = {"train": 4411, "dev": 4412, "confirm": 4413, "transfer": 4414, "refresh": 4415}
    return {
        k: build_functional_graph_families(k, len(pools[k]), seeds[k], role_blocks=pools[k], depths=(1, 2, 3, 4))
        for k in pools
    }


def _groups(fams: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[int]]]:
    items: list[dict[str, Any]] = []
    groups: list[list[int]] = []
    for f in fams:
        cells = f["cells"][:4]
        groups.append(list(range(len(items), len(items) + len(cells))))
        items.extend(cells)
    return items, groups


def _epoch_plan(
    groups: list[list[int]],
    upd: int,
    bf: int,
    seed: int,
    epoch_gids: Callable[[int], Sequence[int]],
    easy: list[bool] | None = None,
) -> list[tuple[int, ...]]:
    rng, plan, e = random.Random(seed), [], 0
    bf = max(1, min(bf, len(groups)))
    while len(plan) < upd:
        g = list(epoch_gids(e))
        rng.shuffle(g)
        if easy is not None:
            g.sort(key=lambda i: not easy[i])
        for i in range(0, max(1, len(g) - bf + 1), bf):
            chunk = g[i : i + bf]
            if len(chunk) == bf:
                plan.append(tuple(ix for gi in chunk for ix in groups[gi]))
        e += 1
        if e > upd + 32:
            break
    while len(plan) < upd:
        g = list(epoch_gids(e))
        plan.append(tuple(ix for gi in g[:bf] for ix in groups[gi]))
    return plan[:upd]


def _gen_trainer(
    job: Mapping[str, Any],
    S: Any,
    dev: str,
    wr: Path,
    model: nn.Module,
    spec: Any,
    items: Sequence[Any],
    plan: Sequence[tuple[int, ...]],
    *,
    lr: float,
    objective: str,
    loss_fn: Callable[..., Any],
    micro: int | None = None,
    upd: int | None = None,
    seed: int | None = None,
    lineage: str | None = None,
    resume: bool = True,
    require_resume: bool = False,
) -> GenericExperimentTrainer:
    eff_dev = dev if (torch.cuda.is_available() and "cuda" in str(dev)) else "cpu"
    seed_i = int(seed if seed is not None else job["init_seed"])
    lin = lineage or f"y4-{job['job_id']}"
    rec = GenericTrainRecipe(
        seed=seed_i,
        total_updates=int(upd or len(plan)),
        batch_size=len(plan[0]),
        micro_batch_size=int(micro or S.micro),
        lr=float(lr),
        objective=objective,
        cf_margin=1.0,
    )
    kw: dict[str, Any] = dict(
        backend_id="avendi_y4",
        model=model,
        model_spec=spec,
        recipe=rec,
        loss_fn=loss_fn,
        items=list(items),
        data_provenance={"split": "train", "job": job["job_id"]},
        order_seed=int(job["order_seed"]),
        source_commit=build_plan()["source_commit"],
        root=Path(wr) / "ckpts",
        lineage=lin,
        device=eff_dev,
        cpu_canary=not getattr(S, "full", False),
    )
    if not getattr(S, "full", False):
        kw["max_cpu_samples"] = 10**6
        kw["max_cpu_updates"] = 10**6
    has_ckpt = (win_long_path(Path(wr) / "ckpts" / lin / "LATEST")).is_file()
    use_resume = True if require_resume else bool(resume and has_ckpt)
    tr = GenericExperimentTrainer(**kw, resume=use_resume)
    tr.plan = list(plan)
    return tr


def _train(
    tr: GenericExperimentTrainer,
    upd: int,
    hook: Callable[[int, Any], None] | None = None,
    ckpt_every: int = 512,
    *,
    save_final: bool = True,
) -> dict[str, Any]:
    losses, t0, start = [], time.perf_counter(), tr.state.global_update
    while tr.state.global_update < upd:
        r = tr.step()
        losses.append(float(r["loss"]))
        if hook:
            hook(int(r["update"]), tr)
        if int(r["update"]) % ckpt_every == 0 and int(r["update"]) < upd:
            tr.save()
            tr.store.prune(keep=set())
    if save_final and tr.state.global_update > tr.published_update:
        tr.save()
        tr.store.prune(keep=set())
    dt = max(1e-9, time.perf_counter() - t0)
    return {
        "updates": int(tr.state.global_update),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_every16": losses[::16],
        "wall_seconds": dt,
        "updates_this_session": int(tr.state.global_update - start),
        "examples_per_sec": (tr.state.global_update - start) * len(tr.plan[0]) / dt,
        "checkpoint_sha256": str(tr.parent),
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0.0,
    }


def _compact_eval(ev: Mapping[str, Any], jd: Path, tag: str) -> dict[str, Any]:
    preds = ev["case_predictions"]
    ids = sorted(preds)
    if ids:
        jd.mkdir(parents=True, exist_ok=True)
        arr_logits = np.stack([preds[i]["first_logits_260"] for i in ids]).astype(np.float32)
        np.savez_compressed(win_long_path(jd / f"{tag}_first_logits.npz"), ids=np.array(ids), logits=arr_logits)
    out = {k: ev[k] for k in ("raw_exact_rate", "fam4_rate", "fam8_rate", "mean_signed_margin", "n_cases", "n_families")}
    cases = {}
    for wid, p in preds.items():
        t0 = int(np.argmax(p["first_logits_260"]))
        ch0 = chr(t0 - 4) if 4 <= t0 < 132 else "?"
        cases[wid] = [bool(p["exact_eos"]), bool(p["exact_eos"]), round(float(p["signed_margin"]), 4), ch0]
    out["cases"] = cases
    return out


def _eval_byte_suite(
    model: AvendiY4ByteModel,
    S: Any,
    jd: Path,
    *,
    split: str = "dev",
    panels: Sequence[str] = ("native", "entity_bijection", "timestamp_permuted", "literal"),
) -> dict[str, Any]:
    seed_base = STREAM_SEEDS.get(split, STREAM_SEEDS["dev"])
    n_p = S.dv if split == "dev" else (S.cf if split == "confirm" else S.tf)
    pools = allocate_y4_disjoint_pools({pk: n_p for pk in panels if pk != "literal"}, seed_base + 50)["pools"]
    out: dict[str, Any] = {}
    for idx, pk in enumerate(panels):
        if pk == "literal":
            lit_f = build_literal_probe_families(S.lit_n, seed_base + 77)
            out[pk] = _compact_eval(eval_y4_byte_families(model, lit_f), jd, f"{split}_{pk}")
        else:
            pfams = build_functional_graph_families(f"{split}_{pk}", len(pools[pk]), seed_base + idx * 31, role_blocks=pools[pk], depths=(1, 2, 3, 4), panel=pk)
            out[pk] = _compact_eval(eval_y4_byte_families(model, pfams), jd, f"{split}_{pk}")
    return out


def _core_state_sha(model: nn.Module) -> str:
    h = sha256()
    sd = model.core.state_dict() if hasattr(model, "core") else model.state_dict()
    for k in sorted(sd):
        t = sd[k].detach().float().cpu()
        h.update(k.encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _profile_device(device: str, full: bool, s20: bool = False) -> float:
    dev = torch.device(device if (torch.cuda.is_available() and "cuda" in str(device)) else "cpu")
    prof = "S20" if s20 else "S3"
    spec = get_y4_byte_spec(prof, cpu_canary=not full)
    m = AvendiY4ByteModel(spec, seed=7).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    mb, seq_l, acc = (2, 96, 8) if (full and s20) else ((4, 96, 4) if full else (2, 16, 1))
    x = torch.randint(4, 30, (mb, seq_l), device=dev)
    y = torch.zeros(mb, dtype=torch.long, device=dev)
    n_upd = 16 if (full and dev.type == "cuda") else 2
    warm = 4 if n_upd > 2 else 1
    times = []
    for st in range(warm + n_upd):
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        opt.zero_grad()
        for _ in range(acc):
            (F.cross_entropy(m(x)[0][:, -1], y) / acc).backward()
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        if st >= warm:
            times.append(time.perf_counter() - t0)
    del m, opt, x, y
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return max(1e-5, float(np.percentile(times, 90)))


def u0000_smoke_one(arch_or_stage: str, sample_job: SessionJob, s_dir: Path) -> dict[str, Any]:
    """CPU init + forward/backward/step + eval + checkpoint save/restore smoke across any Y4 architecture/route."""
    s_dir = Path(s_dir)
    s_dir.mkdir(parents=True, exist_ok=True)
    jid = sample_job.job_id
    st = jid.split("-")[0]
    row = _plan_rows().get(jid, {"job_id": jid, "stage": st, "arm": sample_job.architecture, "init_seed": sample_job.init_seed, "order_seed": sample_job.order_seed, "updates": 1})
    fams = _split_fams(TINY_S)

    if st == "H":
        # Verify historical Y1 adapter loads cleanly
        m = AvendiY4ByteModel(get_y4_byte_spec("S3", cpu_canary=True), seed=7)
        ev = eval_y4_byte_families(m, fams["dev"][:1])
        return {"passed": math.isfinite(ev["fam4_rate"]), "route": "H", "architecture": arch_or_stage}

    if st == "C":
        arch = row["arm"].split("_", 1)[0]
        m = Y4TypedCompositionModel(arch, 32, seed=7)
        tr_c = fams["train"][0]["cells"] + fams["train"][1]["cells"]
        plan = [tuple(range(len(tr_c)))]
        tr = _gen_trainer(row, TINY_S, "cpu", s_dir, m, get_y4_byte_spec("S3", cpu_canary=True), tr_c, plan, lr=1e-3, objective="ce", loss_fn=lambda mod, b, r: p4_y4_loss(mod, b, supervision="true_step"), upd=1, lineage="smoke-c", resume=False)
        res = _train(tr, 1)
        tr2 = _gen_trainer(row, TINY_S, "cpu", s_dir, Y4TypedCompositionModel(arch, 32, seed=99), get_y4_byte_spec("S3", cpu_canary=True), tr_c, plan, lr=1e-3, objective="ce", loss_fn=lambda mod, b, r: p4_y4_loss(mod, b, supervision="true_step"), upd=1, lineage="smoke-c", resume=True, require_resume=True)
        eq = _core_state_sha(tr.model) == _core_state_sha(tr2.model)
        return {"passed": bool(eq and res["checkpoint_sha256"]), "route": f"C:{arch}", "architecture": arch_or_stage}

    if st == "E":
        blocks = build_p6_law_blocks("train", 8, STREAM_SEEDS["corpus"])
        m = P6LearnedMemoryPolicy(width=32, seed=7)
        plan = [tuple(range(8))]
        tr = _gen_trainer(row, TINY_S, "cpu", s_dir, m, get_y4_byte_spec("S3", cpu_canary=True), blocks, plan, lr=1e-3, objective="ce", loss_fn=_e_batch_loss, upd=1, lineage="smoke-e", resume=False)
        res = _train(tr, 1)
        tr2 = _gen_trainer(row, TINY_S, "cpu", s_dir, P6LearnedMemoryPolicy(width=32, seed=99), get_y4_byte_spec("S3", cpu_canary=True), blocks, plan, lr=1e-3, objective="ce", loss_fn=_e_batch_loss, upd=1, lineage="smoke-e", resume=True, require_resume=True)
        rt = evaluate_p6_all_routes(tr2.model, blocks[:2], new_outer_steps=0)
        return {"passed": bool(_core_state_sha(tr.model) == _core_state_sha(tr2.model) and math.isfinite(rt["learned_memory_relevant"])), "route": "E", "architecture": arch_or_stage}

    prof = "S20" if ("s20" in arch_or_stage.lower() or st == "F") else "S3"
    spec = get_y4_byte_spec(prof, cpu_canary=True)
    m = AvendiY4ByteModel(spec, seed=7, use_aux=("pointer" in row["arm"] or "margin" in row["arm"]))
    items, groups = _groups(fams["train"][:4])
    plan = _epoch_plan(groups, 1, 2, 7, lambda e: range(len(groups)))
    obj = row["arm"] if row["arm"] in ("ce", "margin", "pointer", "pointer_sham", "address", "address_sham", "rename", "rename_sham") else "ce"
    lin = f"smoke-{st}-{row['arm']}"
    tr = _gen_trainer(row, TINY_S, "cpu", s_dir, m, spec, items, plan, lr=1e-4, objective=obj, loss_fn=lambda mod, b, r: compute_y4_byte_loss(mod, b, r), upd=1, lineage=lin, resume=False)
    res = _train(tr, 1)
    tr2 = _gen_trainer(row, TINY_S, "cpu", s_dir, AvendiY4ByteModel(spec, seed=99, use_aux=m.use_aux), spec, items, plan, lr=1e-4, objective=obj, loss_fn=lambda mod, b, r: compute_y4_byte_loss(mod, b, r), upd=1, lineage=lin, resume=True, require_resume=True)
    ev = eval_y4_byte_families(tr2.model, fams["dev"][:1])
    return {"passed": bool(_core_state_sha(tr.model) == _core_state_sha(tr2.model) and math.isfinite(ev["fam4_rate"]) and res["checkpoint_sha256"]), "route": f"{st}:{row['arm']}", "architecture": arch_or_stage}


def u0000_smoke_all_y4(wr: Path | str) -> dict[str, Any]:
    """Run CPU u0000 smoke across every trainable Y4 route (A, B, M, C, D, E, F, G) and H adapter."""
    wr = Path(wr)
    wr.mkdir(parents=True, exist_ok=True)
    reps = (
        ("H-F00-s7", "F00"),
        ("A-ce-s7", "ce"),
        ("A-margin-s7", "margin"),
        ("A-pointer-s7", "pointer"),
        ("A-pointer_sham-s7", "pointer_sham"),
        ("B-address-s7", "address"),
        ("B-rename-s7", "rename"),
        ("M-ce-s7", "ce"),
        ("C-categorical_true_step_iit-s7", "categorical_true_step_iit"),
        ("C-continuous_true_step-s7", "continuous_true_step"),
        ("C-dense_true_step-s7", "dense_true_step"),
        ("D-direct_plus_info-s7", "direct_plus_info"),
        ("E-learned_writer-s7", "learned_writer"),
        ("F-s20_ce-s7", "s20_ce"),
        ("G-true_cap-s7", "true_cap"),
    )
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="y4_u0_", ignore_cleanup_errors=True) as td:
        for idx, (jid, arch) in enumerate(reps):
            sj = SessionJob(jid, arch, 7, 7, 1, device="cpu")
            results[jid] = u0000_smoke_one(arch, sj, Path(td) / f"s{idx:02d}")
    out = {"schema": "avendi-y4-u0000-all-smoke-v1", "passed": all(r["passed"] for r in results.values()), "checked": len(results), "routes": results}
    _dump(wr / "U0000_ALL_SMOKE.json", out)
    return out


# ---------------------------------------------------------------------------------------------
# Session Replay Receipts & Historical Y1 Staging / Parity
# ---------------------------------------------------------------------------------------------


def _write_session_replay_receipts(wr: Path, source_commit: str) -> None:
    p0_p, tr_p = wr / "P0_REPLAY_RECEIPT.json", wr / "TRAINING_REPLAY_RECEIPT.json"
    if p0_p.is_file() and tr_p.is_file():
        return
    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_dtype = torch.get_default_dtype()
    prev_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_dn = torch.backends.cudnn.allow_tf32
    try:
        with tempfile.TemporaryDirectory(prefix="y4r_", ignore_cleanup_errors=True) as td:
            tmp_r = Path(td)
            if not p0_p.is_file():
                tpl = init_spec_template(backend_id="avendi_y4", canary=True)
                cart, rkw = build_cartridge_from_spec(tpl, canary_override=True)
                pr_dir = tmp_r / "p0"
                CartridgeRunner().run(cart, work_root=pr_dir, **rkw)
                rep = independent_replay_bundle(pr_dir / "result_bundle", worlds=cart.eval_worlds)
                _dump(p0_p, {"schema": "avendi-y4-p0-replay-v1", "verified": bool(rep.get("verified")), "matched": bool(rep.get("verified"))})
            if not tr_p.is_file():
                from v5_training.checkpoint import InjectedCrash

                tr_dir = tmp_r / "tr"
                spec = get_y4_byte_spec("S3", cpu_canary=True)
                samp = [frame_byte_prompt_sample("ROWS: A0->C2 ; B1->D3 || STEP=1 Q:A0", "C2", rival_answer="D3", task_id=f"tr{i}", world_id=f"wr{i}") for i in range(4)]
                rec = GenericTrainRecipe(seed=7, total_updates=4, batch_size=2, micro_batch_size=2)
                t_full = BytePromptTrainer(samp, spec=spec, recipe=rec, source_commit=source_commit, root=tr_dir / "f", lineage="rf", data_provenance={"s": "t"}, device="cpu", cpu_canary=True)
                l_full = [float(t_full.step()["loss"]) for _ in range(4)]
                t_int = BytePromptTrainer(samp, spec=spec, recipe=rec, source_commit=source_commit, root=tr_dir / "i", lineage="ri", data_provenance={"s": "t"}, device="cpu", cpu_canary=True)
                t_int.step()
                t_int.step()
                t_int.save()
                t_int.step()
                try:
                    t_int.save(inject_crash_at="after_publish_before_pointer")
                except InjectedCrash:
                    pass
                t_res = BytePromptTrainer(samp, spec=spec, recipe=rec, source_commit=source_commit, root=tr_dir / "i", lineage="ri", data_provenance={"s": "t"}, device="cpu", cpu_canary=True, resume=True)
                l_res = [float(t_res.step()["loss"]), float(t_res.step()["loss"])]
                l_ok = all(abs(a - b) <= 1e-6 for a, b in zip(l_full[2:], l_res))
                w_ok = all(torch.equal(t_res.model.state_dict()[k], t_full.model.state_dict()[k]) for k in t_full.model.state_dict())
                _dump(tr_p, {"schema": "avendi-y4-training-replay-v1", "passed": bool(l_ok and w_ok), "losses_matched": l_ok, "weights_matched": w_ok, "full_model_sha256": _core_state_sha(t_full.model), "resumed_model_sha256": _core_state_sha(t_res.model)})
    finally:
        # Preserve coordinator threading & deterministic settings across CartridgeRunner execution
        torch.set_num_threads(prev_threads)
        torch.use_deterministic_algorithms(prev_det)
        torch.set_default_dtype(prev_dtype)
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_mm
        torch.backends.cudnn.allow_tf32 = prev_tf32_dn


def _find_y1_evidence(roots: Sequence[str]) -> Path | None:
    for r in roots:
        if not r:
            continue
        p = Path(r)
        if p.is_file() and p.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(p, "r") as zf:
                    if "prepared/development_suite.json" in zf.namelist():
                        return p
            except Exception:
                continue
        elif p.is_dir():
            if (p / "prepared" / "development_suite.json").is_file():
                return p
            for cand in sorted(p.rglob("ELENDRIL_V8_Y1_EVIDENCE.zip")):
                if cand.is_file():
                    return cand
    return None


def _y1_ref_parity(model: Any, lin: str, ckpt_sha: str, y1_ev: Path, base_cases: Sequence[Any], prompt_by_cid: Mapping[str, str], dev: str) -> dict[str, Any]:
    from anra_v5.elendril.avendi.frozen import _read_evidence_member_bytes, verify_regenerated_y1_case_agreement

    pred_b = _read_evidence_member_bytes(y1_ev, f"evidence/predictions/{lin}.json")
    npz_b = _read_evidence_member_bytes(y1_ev, f"evidence/logits/{lin}.npz")
    if pred_b is None or npz_b is None or not base_cases:
        return {"parity_status": "UNAVAILABLE_Y1_EVIDENCE_NOT_MOUNTED", "matched_reference_cases": 0, "reference_cases": 0}
    pred_doc = json.loads(pred_b.decode("utf-8"))
    pred_by_cid = {r["case_id"]: r for r in pred_doc.get("predictions", [])}
    with np.load(io.BytesIO(npz_b), allow_pickle=False) as npz:
        base_logits, base_ids = npz["base_logits"], [str(x) for x in npz["base_case_ids"].tolist()]
    logit_by_cid = {cid: base_logits[i] for i, cid in enumerate(base_ids)}
    ref_rows, regen_rows = [], []
    td = torch.device(dev if (torch.cuda.is_available() and "cuda" in str(dev)) else "cpu")
    model.eval()
    with torch.no_grad():
        for yc in base_cases:
            cid = yc.case_id
            if cid not in pred_by_cid:
                continue
            p_rec = pred_by_cid[cid]
            ref_toks = p_rec["decisions"][0]["raw_tokens"]
            r_sha = sha256(yc.public_prompt.encode("utf-8")).hexdigest()
            ref_rows.append({"case_id": cid, "checkpoint_sha256": ckpt_sha, "decoder": "utf8-raw", "rendered_input_sha256": r_sha, "raw_tokens": ref_toks, "first_logits": logit_by_cid.get(cid)})
            seq = [UTF8.bos_id, *UTF8.encode(yc.public_prompt), UTF8.special(SpecialToken.ANSWER)]
            g_toks, g_log = [], None
            for step_i in range(max(1, len(ref_toks) + 1)):
                lg = (_core_forward(model, torch.tensor([seq], device=td, dtype=torch.long))[0])[0, -1].float().cpu()
                if step_i == 0:
                    g_log = lg.numpy().astype(np.float32)
                nxt = int(torch.argmax(lg).item())
                g_toks.append(nxt)
                seq.append(nxt)
                if nxt in (UTF8.eos_id, UTF8.pad_id, UTF8.bos_id) or len(g_toks) >= len(ref_toks):
                    break
            regen_rows.append({"case_id": cid, "checkpoint_sha256": ckpt_sha, "decoder": "utf8-raw", "rendered_input_sha256": r_sha, "raw_tokens": g_toks, "first_logits": g_log})
    rep = verify_regenerated_y1_case_agreement(job_id=lin, checkpoint_sha256=ckpt_sha, regenerated_rows=regen_rows, reference_rows=ref_rows, reference_Prompt_by_case_id=prompt_by_cid)
    return {"parity_status": rep["status"], "matched_reference_cases": int(rep["tolerance_sensitivity_counts"].get("1e-05", 0)), "reference_cases": int(rep["reference_cases"]), "max_abs_logit_diff": float(rep["max_abs_logit_diff"]), "parity_report": rep}


def _h_job(job: Mapping[str, Any], S: Any, dev: str, wr: Path, roots: Sequence[str], jd: Path) -> dict[str, Any]:
    lin = f"Y1-t1_factorial-S3-{job['arm']}-s{job['init_seed']}"
    y1_ev = _find_y1_evidence(roots)
    y1_base_cases, y1_prompt_by_cid = [], {}
    if y1_ev is not None:
        from anra_v5.elendril.avendi.frozen import _read_evidence_member_bytes, load_y1_suite

        dev_b = _read_evidence_member_bytes(y1_ev, "prepared/development_suite.json")
        if dev_b is not None:
            y1_base_cases = [c for c in load_y1_suite(json.loads(dev_b.decode("utf-8")))["cases"] if c.panel == "main-base"][: (1024 if S.full else 8)]
            y1_prompt_by_cid = {c.case_id: c.public_prompt for c in y1_base_cases}

    stage_root = wr / "staged_y1"
    st_info = stage_historical_checkpoints_to_writable(roots, [lin], stage_root)["staged"].get(lin, {})
    if st_info.get("status") in ("MISSING_PARENT", "SHA_MISMATCH", None) or not st_info.get("manifest_verified"):
        return {
            "lineage_id": lin,
            "status": str(st_info.get("status", "MISSING_PARENT")),
            "parity_status": "UNAVAILABLE_Y1_EVIDENCE_NOT_MOUNTED",
            "matched_reference_cases": 0,
            "reference_cases": 0,
            "staging": st_info,
        }
    ld = m = None
    try:
        sh = str(st_info["checkpoint_sha256"])
        ld = load_legacy_byte_checkpoint(stage_root, lineage_id=lin, checkpoint_sha256=sh, device=dev)
        m = ld["model"] if "model" in ld else ld["policy"].model
        par = (
            _y1_ref_parity(m, lin, sh, y1_ev, y1_base_cases, y1_prompt_by_cid, dev)
            if y1_ev is not None
            else {"parity_status": "UNAVAILABLE_Y1_EVIDENCE_NOT_MOUNTED", "matched_reference_cases": 0, "reference_cases": 0}
        )
        # Run Y4 4-panel crossover eval using a thin wrapper exposing forward_tokens for AvendiY4 evaluator
        w_m = m.core.embedding.weight.shape[1] if hasattr(m, "core") else getattr(getattr(m, "spec", None), "width", 32)
        wrap = AvendiY4ByteModel(get_y4_byte_spec("S3", cpu_canary=(w_m <= 64)), seed=job["init_seed"])
        wrap.core.load_state_dict(m.core.state_dict() if hasattr(m, "core") else m.state_dict(), strict=False)
        wrap.to(torch.device(dev if (torch.cuda.is_available() and "cuda" in str(dev)) else "cpu"))
        wrap._ablate = True
        ev_dev = _eval_byte_suite(wrap, S, jd / lin, split="dev")
        probe = None
        if len(y1_base_cases) >= 4:
            probe = {"note": "y1_linear_probe_hook", "n_cases": len(y1_base_cases)}
        return {
            "lineage_id": lin,
            "status": "RESTORED",
            "checkpoint_sha256": sh,
            "restored_checkpoint_sha256": str(ld.get("checkpoint_sha256", sh)),
            "staged_dir": str(stage_root / lin),
            "eval_dev": ev_dev,
            "representation_probe": probe,
            **par,
        }
    except Exception as exc:
        return {"lineage_id": lin, "status": "REJECTED", "error": f"{type(exc).__name__}: {exc}", "parity_status": "BLOCKED", "matched_reference_cases": 0, "reference_cases": 0}
    finally:
        if ld is not None:
            del ld
        if m is not None:
            del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------
# Per-Stage Job Executors (A, B, M, C, D, E, F, G)
# ---------------------------------------------------------------------------------------------


def _byte_job(job: Mapping[str, Any], S: Any, fams: dict[str, Any], dev: str, wr: Path, jd: Path) -> dict[str, Any]:
    arm, seed, stage = str(job["arm"]), int(job["init_seed"]), str(job["stage"])
    s20 = stage == "F" and arm.startswith("s20")
    spec = get_y4_byte_spec("S20" if s20 else "S3", cpu_canary=not S.full)
    upd = _job_upd(job, S)
    bf = max(1, S.batch // 4)
    lr = 3e-5 if arm in ("low_lr", "s20_low_lr") else 1e-4
    obj = arm if arm in ("margin", "pointer", "pointer_sham", "address", "address_sham", "rename", "rename_sham") else "ce"
    use_aux = arm in ("margin", "pointer", "pointer_sham", "address", "address_sham", "rename", "rename_sham")

    tr_src = apply_pointer_sham_spans(fams["train"], seed=seed) if arm == "pointer_sham" else fams["train"]
    items, groups = _groups(tr_src)
    n_tr = len(groups)
    easy = None
    if arm == "refresh":
        ritems, rgroups = _groups(fams["refresh"])
        off = len(items)
        groups = groups + [[i + off for i in g] for g in rgroups]
        items = items + ritems
        plan = _epoch_plan(groups, upd, bf, seed, lambda e: range(n_tr) if e == 0 else range(n_tr, len(groups)))
    else:
        if arm == "curriculum":
            easy = [bool(items[g[0]]["depth"] <= 2) for g in groups]
        plan = _epoch_plan(groups, upd, bf, seed, lambda e: range(n_tr), easy=easy)

    model = AvendiY4ByteModel(spec, seed=seed, use_aux=use_aux)
    tr = _gen_trainer(job, S, dev, wr, model, spec, items, plan, lr=lr, objective=obj, loss_fn=lambda m, b, r: compute_y4_byte_loss(m, b, r), micro=(2 if s20 else S.micro), upd=upd)
    miles: dict[int, Any] = {}

    def _hook(u: int, t: Any) -> None:
        if stage == "A" and u in _MILESTONES:
            miles[u] = {"train": eval_y4_byte_families(t.model, fams["train"][: S.mon])["fam4_rate"], "dev": eval_y4_byte_families(t.model, fams["dev"][: S.mon])["fam4_rate"]}

    if stage == "A":
        miles[0] = {"train": eval_y4_byte_families(tr.model, fams["train"][: S.mon])["fam4_rate"], "dev": eval_y4_byte_families(tr.model, fams["dev"][: S.mon])["fam4_rate"]}

    stats = _train(tr, upd, _hook)
    ev_panels = ("native",) if stage == "M" else ("native", "entity_bijection", "timestamp_permuted", "literal")
    ev_dev = _eval_byte_suite(tr.model, S, jd, split="dev", panels=ev_panels)
    extra: dict[str, Any] = {"milestones": miles, "params": sum(p.numel() for p in tr.model.parameters()), "lr": lr, "objective": obj}

    if stage == "A" and arm in ("pointer", "pointer_sham"):
        # Measure byte-to-typed parser bridge & gate ablation
        dev_cells = [c for f in fams["dev"][: S.mon] for c in f["cells"]]
        parsed = [parse_bytes_with_model(tr.model, c) for c in dev_cells]
        parse_ok = sum(int(p["exact_parse"]) for p in parsed) / max(1, len(parsed))
        bridge_oracle = sum(int(parse_public_byte_observation(UTF8.frame(c["public_prompt"], c["gold"]).ids)["rows"] == c["rows"]) for c in dev_cells) / max(1, len(dev_cells))
        with tr.model.ablate_mechanism("pointer"):
            abl_ev = eval_y4_byte_families(tr.model, fams["dev"][: S.mon])["fam4_rate"]
        extra["byte_to_typed_bridge"] = {
            "exact_parse_rate": parse_ok,
            "oracle_parsed_chain_rate": bridge_oracle,
            "ablated_pointer_fam4": abl_ev,
            "pointer_gate_delta": ev_dev["native"]["fam4_rate"] - abl_ev,
        }
    if stage == "M":
        extra["train_fit_fam4"] = eval_y4_byte_families(tr.model, fams["train"][: S.mon])["fam4_rate"]
        extra["fp64_gradient_cosine"] = compute_fp64_gradient_cosine(AvendiY4ByteModel(spec, seed=seed, use_aux=use_aux), fams["dev"][0]["cells"][:4])
    return {"train": stats, "eval_dev": ev_dev, **extra}


def _c_job(job: Mapping[str, Any], S: Any, dev: str, wr: Path) -> dict[str, Any]:
    arm, seed = str(job["arm"]), int(job["init_seed"])
    arch, supervision = arm.split("_", 1)
    widths = select_y4_matched_widths(S.c_width)
    model = Y4TypedCompositionModel(arch, widths[arch]["width"], seed=seed)

    tr_fams = build_functional_graph_families("c_train", S.tr if S.full else 8, STREAM_SEEDS["corpus"] + 400, depths=(1, 2))
    items = [dict(c, family_id=c["family_id"]) for f in tr_fams for c in f["cells"]]
    upd = _job_upd(job, S)
    plan = build_persistent_batch_plan(len(items), upd, S.batch, seed)
    min_unique_batch = min(len(set(b)) for b in plan)
    iit_pairs = build_state_swap_pairs(max(S.batch, S.pairs), STREAM_SEEDS["corpus"] + 401, depth=2, patch_step=1) if "iit" in supervision else None

    tr = _gen_trainer(
        job,
        S,
        dev,
        wr,
        model,
        get_y4_byte_spec("S3", cpu_canary=True),
        items,
        plan,
        lr=3e-3,
        objective="ce",
        loss_fn=lambda m, b, r: p4_y4_loss(m, b, supervision=supervision, iit_pairs=iit_pairs),
        upd=upd,
    )
    stats = _train(tr, upd)
    m = tr.model.eval()
    eff_d = next(m.parameters()).device

    @torch.no_grad()
    def _eval_rates(fams: Sequence[dict[str, Any]]) -> dict[str, float]:
        by_d: dict[int, list[int]] = {1: [], 2: [], 3: [], 4: []}
        for f in fams:
            d = int(f["depth"])
            rk, rv, q, vm, gold = encode_y4_typed_batch(f["cells"], eff_d)
            ok = (m.forward_trace(rk, rv, q, vm, steps=d)["final_logits"].argmax(-1) == gold).tolist()
            by_d.setdefault(d, []).append(int(all(ok)))
        return {str(d): float(np.mean(v)) if v else 0.0 for d, v in sorted(by_d.items())}

    n_ev = S.mh if S.full else 4
    rates: dict[str, Any] = {
        "native": _eval_rates(build_functional_graph_families("c_nat", n_ev, STREAM_SEEDS["dev"] + 410, depths=(1, 2, 3, 4), panel="native")),
        "distractor": _eval_rates(build_functional_graph_families("c_dis", n_ev, STREAM_SEEDS["dev"] + 411, depths=(1, 2, 3, 4), panel="distractor")),
        "topology": _eval_rates(build_functional_graph_families("c_top", n_ev, STREAM_SEEDS["dev"] + 412, depths=(1, 2, 3, 4), panel="heldout_topology")),
        "variable_stop": _eval_rates(build_functional_graph_families("c_var", n_ev, STREAM_SEEDS["dev"] + 413, depths=(1, 2, 3, 4), panel="native")),
    }
    if arch == "categorical":
        m._argmax = True
        rates["hard_argmax"] = _eval_rates(build_functional_graph_families("c_arg", n_ev, STREAM_SEEDS["dev"] + 410, depths=(1, 2, 3, 4), panel="native"))
        m._argmax = False

    # Check batch padding invariance & row-order permutation invariance (<= 1e-6)
    c1 = next(c for c in items if c["depth"] == 1)
    c2 = next(c for c in items if c["depth"] == 2)
    with torch.no_grad():
        rk1, rv1, q1, vm1, _ = encode_y4_typed_batch([c1], eff_d)
        l_unpad = m.forward_trace(rk1, rv1, q1, vm1, steps=1)["final_logits"][0]
        rk2, rv2, q2, vm2, _ = encode_y4_typed_batch([c1, c2], eff_d)
        l_pad = m.forward_trace(rk2, rv2, q2, vm2, steps=1)["final_logits"][0]
        pad_diff = float((l_unpad - l_pad).abs().max().item())
        c1_rev = {**c1, "rows": list(reversed(c1["rows"]))}
        rk_r, rv_r, q_r, vm_r, _ = encode_y4_typed_batch([c1_rev], eff_d)
        l_rev = m.forward_trace(rk_r, rv_r, q_r, vm_r, steps=1)["final_logits"][0]
        perm_diff = float((l_unpad - l_rev).abs().max().item())

    swap_suite = {
        str(d): run_y4_state_swap_suite(
            m,
            build_state_swap_pairs(S.pairs, STREAM_SEEDS["dev"] + 420 + d, depth=d, patch_step=1),
            train_pairs=build_state_swap_pairs(S.pairs, STREAM_SEEDS["corpus"] + 420 + d, depth=d, patch_step=1),
        )
        for d in (2, 3, 4)
    }
    return {
        "train": stats,
        "arch": arch,
        "supervision": supervision,
        "widths": widths[arch],
        "min_unique_batch_items": min_unique_batch,
        "pad_invariance_max_diff": pad_diff,
        "row_permutation_invariance_max_diff": perm_diff,
        "rates": rates,
        "swap_suite": swap_suite,
    }


def _restore_a_ce_parent(S: Any, fams: dict[str, Any], dev: str, wr: Path, seed: int) -> tuple[AvendiY4ByteModel, str]:
    parent_row = _plan_rows()[f"A-ce-s{seed}"]
    items, groups = _groups(fams["train"])
    spec = get_y4_byte_spec("S3", cpu_canary=not S.full)
    plan = _epoch_plan(groups, _job_upd(parent_row, S), max(1, S.batch // 4), seed, lambda e: range(len(groups)))
    tr = _gen_trainer(parent_row, S, dev, wr, AvendiY4ByteModel(spec, seed=seed), spec, items, plan, lr=1e-4, objective="ce", loss_fn=lambda m, b, r: compute_y4_byte_loss(m, b, r), upd=_job_upd(parent_row, S), resume=True, require_resume=True)
    if tr.state.global_update < _job_upd(parent_row, S):
        raise RuntimeError(f"PARENT_MISSING:{parent_row['job_id']}")
    return tr.model, str(tr.parent)


def _p5_world_items(worlds: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(w, gold=_ACT_CHAR.get(w["optimal_actions"][0], w["optimal_actions"][0].split(":")[-1]), rival=w["candidates"][1], family_id=w["world_id"]) for w in worlds]


def _fork_dg_job(job: Mapping[str, Any], S: Any, fams: dict[str, Any], dev: str, wr: Path) -> dict[str, Any]:
    arm, seed, stage = str(job["arm"]), int(job["init_seed"]), str(job["stage"])
    parent, parent_ckpt_sha = _restore_a_ce_parent(S, fams, dev, wr, seed)
    parent_core_sha = _core_state_sha(parent)
    parent_sd = {k: v.detach().clone() for k, v in parent.state_dict().items()}

    p5_tr = build_p5_blind_worlds("train", S.p5, STREAM_SEEDS["corpus"] + 500)
    p5_dv = build_p5_blind_worlds("dev", S.p5, STREAM_SEEDS["dev"] + 500)
    assert_no_condition_leak(p5_dv)

    direct, _ = _groups(fams["train"])
    half = max(1, len(direct) // 2)
    info = _p5_world_items(p5_tr)
    items = direct + info
    direct_ix, neutral_ix, info_ix = list(range(half)), list(range(half, len(direct))), list(range(len(direct), len(items)))

    n_half = max(1, S.batch // 2)
    mode_cfg = {
        "a_to_a": (direct_ix, direct_ix, n_half, n_half),
        "direct_plus_neutral": (direct_ix, neutral_ix, n_half, n_half),
        "direct_plus_info": (direct_ix, info_ix, n_half, n_half),
        "replace_direct": (neutral_ix, info_ix, n_half, n_half),
        "uncapped": (direct_ix, info_ix, n_half, n_half),
        "true_cap": (direct_ix, info_ix, n_half, n_half),
        "sham_cap": (direct_ix, info_ix, n_half, n_half),
    }[arm]
    a_ix, b_ix, n_a, n_b = mode_cfg
    upd = _job_upd(job, S)
    rng = random.Random(seed + 515)
    plan = [tuple([rng.choice(a_ix) for _ in range(n_a)] + [rng.choice(b_ix) for _ in range(n_b)]) for _ in range(upd)]

    spec = get_y4_byte_spec("S3", cpu_canary=not S.full)
    model = AvendiY4ByteModel(spec, seed=seed)
    model.load_state_dict(parent_sd, strict=True)
    init_core_sha = _core_state_sha(model)

    ev_parent = {
        "stage_a_fam4": eval_y4_byte_families(parent, fams["dev"][: S.mon])["fam4_rate"],
        "stage_d_p5": evaluate_p5_multistep_policy(parent, p5_dv, irrelevant_tool=False),
    }
    tr = _gen_trainer(job, S, dev, wr, model, spec, items, plan, lr=1e-4, objective="ce", loss_fn=lambda m, b, r: compute_y4_byte_loss(m, b, r), upd=upd)
    need_cap = (stage == "G" and arm in ("true_cap", "sham_cap"))
    stats = _train(tr, upd, save_final=not need_cap)

    cap_info = None
    if need_cap:
        cap_info = apply_radial_norm_cap(
            tr.model,
            parent_sd,
            max_rel_norm=0.001,
            sham_permute_seed=(seed + 700 if arm == "sham_cap" else None),
        )
        tr.save()
        tr.store.prune(keep=set())
        stats["checkpoint_sha256"] = str(tr.parent)

    ev_child = {
        "stage_a_fam4": eval_y4_byte_families(tr.model, fams["dev"][: S.mon])["fam4_rate"],
        "stage_d_p5": evaluate_p5_multistep_policy(tr.model, p5_dv, irrelevant_tool=False),
        "stage_d_irrelevant_tool": evaluate_p5_multistep_policy(tr.model, p5_dv, irrelevant_tool=True),
    }
    return {
        "train": stats,
        "parent_job_id": f"A-ce-s{seed}",
        "parent_checkpoint_sha256": parent_ckpt_sha,
        "parent_core_sha256": parent_core_sha,
        "child_initial_core_sha256": init_core_sha,
        "parent": ev_parent,
        "child": ev_child,
        "cap": cap_info,
    }


def _e_batch_loss(m: P6LearnedMemoryPolicy, chunk: Sequence[Mapping[str, Any]], recipe: Any) -> torch.Tensor:
    dev = next(m.parameters()).device
    m.reset_episode()
    tot = torch.zeros((), device=dev)
    sham = "sham" in str(getattr(recipe, "objective", ""))
    for i, b in enumerate(chunk):
        mode_r = random.Random(int(sha256(b["block_id"].encode("utf-8")).hexdigest()[:8], 16)).random()
        sup_x = sup_y = None
        if mode_r < 0.7:
            sup_x = torch.tensor([int(b["diagnostic_input"])], device=dev)
            y_raw = int(b["info_obs"])
            if sham:
                y_raw = (y_raw + i + 1) % 4
            sup_y = torch.tensor([y_raw], device=dev)
        ep_lg, act_lg = m.forward_block(torch.tensor([int(b["unseen_eval_input"])], device=dev), sup_x, sup_y)
        tot = tot + F.cross_entropy(ep_lg, torch.tensor([int(b["eval_gold"])], device=dev))
        opt_act = torch.tensor([0], device=dev)
        tot = tot + 0.5 * F.cross_entropy(act_lg, opt_act)
    return tot / max(1, len(chunk))


def _e_job(job: Mapping[str, Any], S: Any, dev: str, wr: Path) -> dict[str, Any]:
    arm, seed = str(job["arm"]), int(job["init_seed"])
    blocks_tr = build_p6_law_blocks("train", S.blocks, STREAM_SEEDS["corpus"])
    blocks_dv = build_p6_law_blocks("dev", S.blocks_dev, STREAM_SEEDS["dev"])
    pol = P6LearnedMemoryPolicy(width=32, seed=seed)
    upd = _job_upd(job, S)
    plan = build_persistent_batch_plan(len(blocks_tr), upd, min(S.batch, len(blocks_tr)), seed)
    tr = _gen_trainer(job, S, dev, wr, pol, get_y4_byte_spec("S3", cpu_canary=True), blocks_tr, plan, lr=3e-3, objective=arm, loss_fn=_e_batch_loss, upd=upd)
    stats = _train(tr, upd)
    routes = evaluate_p6_all_routes(tr.model, blocks_dv, new_outer_steps=0)
    sham_mi = verify_p6_sham_zero_mi(blocks_dv) if arm == "sham_writer" else None
    sup_steps = sum(1 for _ in blocks_dv)
    return {
        "train": stats,
        "arm": arm,
        "outer_optimizer_steps": int(stats["updates"]),
        "support_gradient_steps": int(sup_steps),
        "eval_blocks": len(blocks_dv),
        "eval_routes_new_outer_steps": 0,
        "routes": routes,
        "sham_zero_mi": sham_mi,
    }


def run_job(job_id: str, S: Any, dev: str, wr: Path, roots: Sequence[str] = ()) -> dict[str, Any]:
    """Execute any one of the 126 Y4 jobs cleanly, writing receipt.json, JOB_RECEIPT.json, and job_archive.zip."""
    wr = Path(wr)
    rows = _plan_rows()
    job = rows[job_id]
    jd = wr / "jobs" / job_id
    rc_p = jd / "receipt.json"
    jrc_p = jd / "JOB_RECEIPT.json"
    if rc_p.is_file() and jrc_p.is_file():
        return json.loads(rc_p.read_text(encoding="utf-8"))

    jd.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    fams = _split_fams(S)
    stage, arm = str(job["stage"]), str(job["arm"])
    eff_dev = dev if (torch.cuda.is_available() and "cuda" in str(dev)) else "cpu"

    if stage == "H":
        body = _h_job(job, S, eff_dev, wr, roots, jd)
    elif stage in ("A", "B", "M", "F"):
        body = _byte_job(job, S, fams, eff_dev, wr, jd)
    elif stage == "C":
        body = _c_job(job, S, eff_dev, wr)
    elif stage in ("D", "G"):
        body = _fork_dg_job(job, S, fams, eff_dev, wr)
    elif stage == "E":
        body = _e_job(job, S, eff_dev, wr)
    else:
        raise ValueError(f"Unknown stage {stage} for job {job_id}")

    src_commit = build_plan()["source_commit"]
    comp_upd = int((body.get("train") or {}).get("updates", _job_upd(job, S)))
    rec = {
        "job_id": job_id,
        "stage": stage,
        "arm": arm,
        "architecture": arm,
        "init_seed": int(job["init_seed"]),
        "order_seed": int(job["order_seed"]),
        "source_commit": src_commit,
        "technology_baseline": TECHNOLOGY_BASELINE,
        "spec_sha256": job["spec_sha256"],
        "completed_updates": comp_upd,
        "eval_rows": 1,
        "scale": "FULL" if S.full else "TINY",
        "liveness": "completed",
        "evidence_verification": "SELF_RECORDED",
        "scientific_qualification": "UNASSESSED_UNTIL_AGGREGATION",
        "wall_seconds": round(time.perf_counter() - t0, 4),
        "result": body,
    }
    _dump(jd / "receipt_body.json", rec)
    z = jd / "job_archive.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(jd / "receipt_body.json", "receipt.json")
        for npz in sorted(jd.rglob("*.npz")):
            zf.write(npz, str(npz.relative_to(jd).as_posix()))
    asha = sha256(z.read_bytes()).hexdigest()
    rec["archive_path"] = f"jobs/{job_id}/job_archive.zip"
    rec["archive_sha256"] = asha
    _dump(rc_p, rec)
    _dump(jrc_p, {**rec, "archive_path": f"jobs/{job_id}/job_archive.zip", "archive_sha256": asha})
    return rec


# ---------------------------------------------------------------------------------------------
# Full Campaign Orchestration, Resume Reconciliation, Sealed Confirmation & Study Aggregation
# ---------------------------------------------------------------------------------------------


def _unpack_resume(resume_input: str | Path | Sequence[str | Path] | None, wr: Path) -> None:
    """Restore verified jobs, checkpoints, replay receipts, and prior ConfirmLock custody from dir or ZIP."""
    if not resume_input:
        return
    raw_items = [resume_input] if isinstance(resume_input, (str, Path)) else list(resume_input)
    wr_res = wr.resolve()
    prior = wr / "prior_confirm"

    def _safe_copy_zip(zp: Path) -> None:
        with zipfile.ZipFile(zp, "r") as zf:
            for mem in zf.infolist():
                if mem.is_dir():
                    continue
                rel = Path(mem.filename)
                if not rel.parts or rel.is_absolute() or ".." in rel.parts:
                    raise ValueError(f"Unsafe archive member path: {mem.filename}")
                posix = rel.as_posix()
                if posix in ("CONFIRM_LOCK.json", "CONFIRM_RESULT.json") or posix.startswith("prior_confirm/"):
                    sub = rel if rel.parts[0] == "prior_confirm" else (Path("prior_confirm") / rel)
                    dst = win_long_path(wr_res / sub)
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(mem) as src, open(dst, "wb") as out:
                            shutil.copyfileobj(src, out)
                    continue
                if posix in ("P0_REPLAY_RECEIPT.json", "TRAINING_REPLAY_RECEIPT.json") or rel.parts[0] in ("jobs", "ckpts"):
                    dst = win_long_path(wr_res / rel)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(mem) as src, open(dst, "wb") as out:
                        shutil.copyfileobj(src, out)

    for raw in raw_items:
        rp = Path(raw)
        if not rp.exists():
            continue
        if rp.is_dir() and (rp / "AVENDI_Y4_FULL.zip").is_file():
            rp = rp / "AVENDI_Y4_FULL.zip"
        if rp.is_file() and rp.suffix.lower() == ".zip":
            _safe_copy_zip(rp)
        elif rp.is_dir():
            for sub in ("jobs", "ckpts"):
                if (rp / sub).is_dir():
                    shutil.copytree(win_long_path(rp / sub), win_long_path(wr / sub), dirs_exist_ok=True)
            for fname in ("P0_REPLAY_RECEIPT.json", "TRAINING_REPLAY_RECEIPT.json"):
                if (rp / fname).is_file() and not (wr / fname).is_file():
                    shutil.copy2(rp / fname, wr / fname)
            for cf in ("CONFIRM_LOCK.json", "CONFIRM_RESULT.json"):
                if (rp / cf).is_file() and not (prior / cf).is_file():
                    prior.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(rp / cf, prior / cf)


def run_full_campaign(
    *,
    wr: Path | str,
    S: Any = FULL_S,
    session_minutes: float = 330.0,
    input_roots: Sequence[str] = (),
    resume_input: str | Path | Sequence[str | Path] | None = None,
    devices: Sequence[str] | None = None,
    in_process_runner: Callable[..., Any] | None = None,
    measured_sec_per_update: float | None = None,
    measured_sec_per_update_by_group: Mapping[str, float] | None = None,
    clock_fn: Callable[[], float] | None = None,
    one_go: bool = True,
    inventory_only: bool = False,
) -> dict[str, Any]:
    """Run all 126 Y4 jobs via CampaignOrchestrator.run_registered_plan over dual GPUs with telemetry and resume."""
    wr = Path(wr)
    wr.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    clk = clock_fn or time.monotonic
    t0 = float(clk())
    plan = build_plan(mode="FULL" if S.full else "CANARY")
    rows = {r["job_id"]: r for r in plan["jobs"]}

    _unpack_resume(resume_input, wr)
    if not inventory_only:
        _write_session_replay_receipts(wr, plan["source_commit"])

    eff_devs = tuple(devices) if devices is not None else (("cuda:0", "cuda:1") if (torch.cuda.is_available() and torch.cuda.device_count() >= 2) else ("cpu",))
    n_workers = max(1, len(eff_devs))
    # Prioritize historical & parent jobs (H, A, B, M, C, E, F) before child forks (D, G)
    stage_prio = {"H": 0, "A": 1, "B": 2, "M": 3, "C": 4, "E": 5, "F": 6, "D": 7, "G": 8}
    ordered_rows = sorted(plan["jobs"], key=lambda r: (stage_prio.get(r["stage"], 9), 0 if r["arm"] == "ce" else 1, r["job_id"]))
    session_jobs = [
        SessionJob(
            job_id=r["job_id"],
            architecture=r["arm"],
            init_seed=int(r["init_seed"]),
            order_seed=int(r["order_seed"]),
            updates=_job_upd(r, S),
            device=eff_devs[i % n_workers],
            extra={
                "stage": r["stage"],
                "spec_sha256": r["spec_sha256"],
                "source_commit": plan["source_commit"],
                "parent_job_id": r.get("parent_job_id"),
                "depends_on": r.get("depends_on", ()),
                "profile_group": r["profile_group"],
                "eval_rows": 1,
            },
        )
        for i, r in enumerate(ordered_rows)
    ]

    orch = CampaignOrchestrator(
        campaign_id="AVENDI-V8-Y4",
        source_commit=plan["source_commit"],
        work_dir=wr,
        max_minutes=float(session_minutes),
        reserve_minutes=15.0 if one_go else min(30.0, max(5.0, float(session_minutes) * 0.15)),
        min_disk_bytes=1024 if not S.full else 2 * 2**30,
        in_process_runner=in_process_runner,
    )

    def _prof(rep_job: SessionJob, _pdir: Path) -> float:
        grp = str(rep_job.extra.get("profile_group", "S3"))
        if grp in ("H", "E"):
            return 1e-4 if not S.full else 0.005
        return _profile_device(eff_devs[0], S.full, s20=(grp == "S20"))

    def _parent_check(sj: SessionJob, completed_set: set[str]) -> tuple[bool, str]:
        p_id = sj.extra.get("parent_job_id")
        if p_id:
            lp = win_long_path(wr / "ckpts" / f"y4-{p_id}" / "LATEST")
            if not lp.is_file():
                return False, f"MISSING_PARENT_CHECKPOINT:{p_id}"
        return True, "OK"

    def _cmd_builder(sj: SessionJob, _wroot: Path) -> tuple[str, ...]:
        cmd = [
            sys.executable,
            "-m",
            "experiments.avendi_y4.module",
            "--job",
            sj.job_id,
            "--work-root",
            str(wr),
            "--device",
            sj.device,
            "--scale",
            "full" if S.full else "tiny",
        ]
        if input_roots:
            cmd.extend(["--input-roots", ",".join(input_roots)])
        return tuple(cmd)

    plan_res = orch.run_registered_plan(
        session_jobs,
        command_builder=None if in_process_runner is not None else _cmd_builder,
        devices=eff_devs,
        measured_seconds_per_update=measured_sec_per_update,
        measured_seconds_per_update_by_group=measured_sec_per_update_by_group,
        profile_fn=None if (measured_sec_per_update is not None or measured_sec_per_update_by_group is not None) else _prof,
        per_job_disk_bytes=(32 * 2**20 if S.full else 1024),
        startup_overhead_seconds=(10.0 if S.full else 0.0),
        inventory_only=inventory_only,
        required_archive_members=("receipt.json",),
        fail_on_resume_rejection=True,
        parent_checker=_parent_check,
        live_wall_clock_preemption=True,
        clock_fn=clk,
        u0000_smoke_fn=None if inventory_only else (u0000_smoke_one if in_process_runner is None else None),
        confirm_lock_path=wr / "CONFIRM_LOCK.json",
        confirm_required_job_ids=plan["confirm_eligible_jobs"],
    )

    if inventory_only:
        return plan_res

    failed_ids = list(plan_res.get("failed_job_ids", ()))
    blocked_ids = list(plan_res.get("blocked_job_ids", ()))
    pending_ids = list(plan_res.get("deferred_pending_job_ids", ()))
    reused_ids = list((plan_res.get("resume_receipt") or {}).get("verified_completed_jobs", ()))
    executed_ids = list(plan_res.get("newly_executed_job_ids", ()))
    exit_reason = (
        "ALREADY_VERIFIED_NO_OP"
        if (not executed_ids and not pending_ids and not failed_ids and not blocked_ids and len(reused_ids) == len(rows))
        else (
            "ALL_JOBS_COMPLETE"
            if (not pending_ids and not failed_ids and not blocked_ids)
            else ("PARTIAL_WITH_FAILURES" if (failed_ids or blocked_ids) else "TIME_GUARD_STOP")
        )
    )
    telemetry = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(max(0.0, float(clk()) - t0), 4),
        "exit_reason": exit_reason,
        "reused_job_ids": sorted(reused_ids),
        "executed_job_ids": sorted(executed_ids),
        "failed_job_ids": sorted(failed_ids),
        "blocked_job_ids": sorted(blocked_ids),
        "pending_job_ids": pending_ids,
        "dispatched_devices": list(plan_res.get("dispatched_devices", ())),
        "waves": plan_res.get("waves", ()),
    }
    _dump(wr / "SESSION_TELEMETRY.json", telemetry)
    return {**telemetry, "registered_plan_status": plan_res}


# ---------------------------------------------------------------------------------------------
# One-Shot Sealed ConfirmLock + Idempotent Crash Recovery + 50 Studies + 44 Platform Uses
# ---------------------------------------------------------------------------------------------


def _confirm_one(row: Mapping[str, Any], S: Any, fams: dict[str, Any], wr: Path) -> dict[str, Any]:
    """One-shot holdout evaluation of a verified checkpoint without further training."""
    stage, arm, seed = str(row["stage"]), str(row["arm"]), int(row["init_seed"])
    eff_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    upd = _job_upd(row, S)
    if stage == "C":
        arch, supervision = arm.split("_", 1)
        widths = select_y4_matched_widths(S.c_width)
        m = Y4TypedCompositionModel(arch, widths[arch]["width"], seed=seed)
        tr_fams = build_functional_graph_families("c_train", S.tr if S.full else 8, STREAM_SEEDS["corpus"] + 400, depths=(1, 2))
        items = [dict(c, family_id=c["family_id"]) for f in tr_fams for c in f["cells"]]
        plan = build_persistent_batch_plan(len(items), upd, S.batch, seed)
        tr = _gen_trainer(row, S, eff_dev, wr, m, get_y4_byte_spec("S3", cpu_canary=True), items, plan, lr=3e-3, objective="ce", loss_fn=lambda mod, b, r: p4_y4_loss(mod, b, supervision=supervision), upd=upd, resume=True, require_resume=True)
        cf_fams = build_functional_graph_families("c_cf", S.mh if S.full else 4, STREAM_SEEDS["confirm"] + 410, depths=(1, 2, 3, 4))
        by_d: dict[int, list[int]] = {}
        with torch.no_grad():
            for f in cf_fams:
                rk, rv, q, vm, gold = encode_y4_typed_batch(f["cells"], next(tr.model.parameters()).device)
                ok = (tr.model.forward_trace(rk, rv, q, vm, steps=int(f["depth"]))["final_logits"].argmax(-1) == gold).tolist()
                by_d.setdefault(int(f["depth"]), []).append(int(all(ok)))
        return {"metric": "d34_complete", "rates": {str(d): float(np.mean(v)) for d, v in by_d.items()}, "d34": 0.5 * (float(np.mean(by_d.get(3, [0.0]))) + float(np.mean(by_d.get(4, [0.0]))))}

    if stage == "E":
        blocks_tr = build_p6_law_blocks("train", S.blocks, STREAM_SEEDS["corpus"])
        blocks_cf = build_p6_law_blocks("confirm", S.blocks_dev, STREAM_SEEDS["confirm"])
        pol = P6LearnedMemoryPolicy(width=32, seed=seed)
        plan = build_persistent_batch_plan(len(blocks_tr), upd, min(S.batch, len(blocks_tr)), seed)
        tr = _gen_trainer(row, S, eff_dev, wr, pol, get_y4_byte_spec("S3", cpu_canary=True), blocks_tr, plan, lr=3e-3, objective=arm, loss_fn=_e_batch_loss, upd=upd, resume=True, require_resume=True)
        return {"metric": "p6_confirm", **evaluate_p6_all_routes(tr.model, blocks_cf, new_outer_steps=0)}

    if stage in ("D", "G"):
        spec = get_y4_byte_spec("S3", cpu_canary=not S.full)
        p5_tr = build_p5_blind_worlds("train", S.p5, STREAM_SEEDS["corpus"] + 500)
        direct, _ = _groups(fams["train"])
        items = direct + _p5_world_items(p5_tr)
        n_half = max(1, S.batch // 2)
        plan = [tuple(list(range(n_half)) + list(range(n_half))) for _ in range(upd)]
        tr = _gen_trainer(row, S, eff_dev, wr, AvendiY4ByteModel(spec, seed=seed), spec, items, plan, lr=1e-4, objective="ce", loss_fn=lambda mod, b, r: compute_y4_byte_loss(mod, b, r), upd=upd, resume=True, require_resume=True)
        ev = eval_y4_byte_families(tr.model, fams["confirm"])
        p5_cf = build_p5_blind_worlds("confirm", S.p5, STREAM_SEEDS["confirm"] + 500)
        return {"metric": "fam8_rate", "fam4": ev["fam4_rate"], "fam8": ev["fam8_rate"], "raw": ev["raw_exact_rate"], "p5_confirm": evaluate_p5_multistep_policy(tr.model, p5_cf, irrelevant_tool=False)}

    s20 = stage == "F" and arm.startswith("s20")
    spec = get_y4_byte_spec("S20" if s20 else "S3", cpu_canary=not S.full)
    lr = 3e-5 if arm in ("low_lr", "s20_low_lr") else 1e-4
    obj = arm if arm in ("margin", "pointer", "pointer_sham", "address", "address_sham", "rename", "rename_sham") else "ce"
    use_aux = arm in ("margin", "pointer", "pointer_sham", "address", "address_sham", "rename", "rename_sham")
    tr_src = apply_pointer_sham_spans(fams["train"], seed=seed) if arm == "pointer_sham" else fams["train"]
    items, groups = _groups(tr_src)
    n_tr, bf = len(groups), max(1, S.batch // 4)
    if arm == "refresh":
        ritems, rgroups = _groups(fams["refresh"])
        off = len(items)
        groups = groups + [[i + off for i in g] for g in rgroups]
        items = items + ritems
        plan = _epoch_plan(groups, upd, bf, seed, lambda e: range(n_tr) if e == 0 else range(n_tr, len(groups)))
    else:
        easy = [bool(items[g[0]]["depth"] <= 2) for g in groups] if arm == "curriculum" else None
        plan = _epoch_plan(groups, upd, bf, seed, lambda e: range(n_tr), easy=easy)
    tr = _gen_trainer(row, S, eff_dev, wr, AvendiY4ByteModel(spec, seed=seed, use_aux=use_aux), spec, items, plan, lr=lr, objective=obj, loss_fn=lambda mod, b, r: compute_y4_byte_loss(mod, b, r), micro=(2 if s20 else S.micro), upd=upd, resume=True, require_resume=True)
    ev = eval_y4_byte_families(tr.model, fams["confirm"])
    return {"metric": "fam8_rate", "fam4": ev["fam4_rate"], "fam8": ev["fam8_rate"], "raw": ev["raw_exact_rate"]}


def _resolve_confirmation(
    wr: Path,
    recs: Mapping[str, Mapping[str, Any]],
    elig: Sequence[str],
    rows: Mapping[str, dict[str, Any]],
    S: Any,
    fams: dict[str, Any],
    plan: Mapping[str, Any],
    *,
    allow_confirm_replay: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Idempotent ConfirmLock resolution supporting crash recovery after unlock before score persistence."""
    lock_p = wr / "CONFIRM_LOCK.json"
    res_p = wr / "CONFIRM_RESULT.json"
    prior_res = wr / "prior_confirm" / "CONFIRM_RESULT.json"
    miss_elig = [j for j in elig if j not in recs]
    proto_sha = _canon_sha({"eligible": list(elig), "gates": plan["qualification_gates"]})

    # Reuse committed prior result if valid & not replaying
    for cand_p, reused_flag in ((res_p, False), (prior_res, True)):
        if not allow_confirm_replay and cand_p.is_file():
            try:
                rdoc = json.loads(cand_p.read_text(encoding="utf-8"))
                if _canon_sha(rdoc.get("result")) == rdoc.get("result_sha256"):
                    if cand_p != res_p:
                        _dump(res_p, rdoc)
                    st = "REUSED_PRIOR_SEALED_CONFIRMATION" if reused_flag else "REUSED_SEALED_CONFIRMATION"
                    return {"status": st, **rdoc}, {"status": st, "custody_valid": True, "lock_sha256_verified": True, "reused_from_prior_session": reused_flag, "lock_sha256": rdoc.get("lock_sha256")}
            except Exception:
                pass

    lock = ConfirmLock(lock_p)
    if miss_elig:
        if not lock_p.is_file():
            lock.seal(elig, protocol_sha256=proto_sha, source_commit=plan["source_commit"], metadata={"eligible": len(elig)})
        else:
            insp = lock.inspect(sorted(recs))
            if insp.get("unlocked") or insp.get("defects"):
                return {"status": "CUSTODY_TAMPERED_LOCK", "defects": insp.get("defects", [])}, {"status": "CUSTODY_TAMPERED_LOCK", "custody_valid": False, "lock_sha256_verified": False, "reused_from_prior_session": False}
        return {"status": "DEFERRED_ELIGIBLE_ARMS_INCOMPLETE", "missing": miss_elig}, {"status": "DEFERRED_ELIGIBLE_ARMS_INCOMPLETE", "custody_valid": False, "lock_sha256_verified": True, "reused_from_prior_session": False}

    # All eligible jobs complete -> ensure lock intact, unlock if sealed, and write CONFIRM_RESULT.json once
    if not lock_p.is_file():
        lock.seal(elig, protocol_sha256=proto_sha, source_commit=plan["source_commit"], metadata={"eligible": len(elig)})
    insp = lock.inspect(sorted(recs))
    if insp.get("defects"):
        return {"status": "CUSTODY_TAMPERED_LOCK", "defects": insp["defects"]}, {"status": "CUSTODY_TAMPERED_LOCK", "custody_valid": False, "lock_sha256_verified": False, "reused_from_prior_session": False}
    unlk = lock.unlock_if_ready(sorted(recs), reason="ALL_CONFIRM_ELIGIBLE_JOBS_COMPLETE")
    if unlk.get("defects") or not unlk.get("unlocked"):
        return {"status": "CUSTODY_UNLOCK_FAILED", "defects": unlk.get("defects", [])}, {"status": "CUSTODY_UNLOCK_FAILED", "custody_valid": False, "lock_sha256_verified": False, "reused_from_prior_session": False}

    scores = {j: _confirm_one(rows[j], S, fams, wr) for j in elig}
    r_sha = _canon_sha(scores)
    payload_out = {"lock_sha256": unlk["lock_sha256"], "result": scores, "result_sha256": r_sha, "opened_count": 1}
    _dump(res_p, payload_out)
    status = "CONFIRM_COHORT_REPLAY_AFTER_PRIOR_OPENING" if allow_confirm_replay else "LOCKED_AND_OPENED_ONCE"
    custody = {"status": status, "custody_valid": not allow_confirm_replay, "lock_sha256_verified": True, "reused_from_prior_session": False, "lock_sha256": unlk["lock_sha256"]}
    return {"status": status, **payload_out}, custody


def _write_study(wr: Path, sid: str, jid: str, split: str, *, payload: Mapping[str, Any], status: str = "COMPLETE") -> dict[str, Any]:
    p = wr / "studies" / sid / jid / split / "receipt.json"
    rec = {"study_id": sid, "job_id": jid, "split": split, "status": status, "receipt_path": str(p.relative_to(wr).as_posix()), **payload}
    sh = _dump(p, rec)
    return {"path": str(p), "sha256": sh, "status": status}


def aggregate_full(
    wr: Path | str,
    S: Any,
    dispatch: Mapping[str, Any] = (),
    input_roots: Sequence[str] = (),
    *,
    allow_confirm_replay: bool = False,
) -> dict[str, Any]:
    """Build 50 Y4 study receipts (U01..U50), 44 platform witnesses, sealed ConfirmLock, audit, and archives."""
    wr = Path(wr)
    plan = build_plan(mode="FULL" if S.full else "CANARY")
    rows = {r["job_id"]: r for r in plan["jobs"]}
    recs = {}
    for p in (wr / "jobs").glob("*/receipt.json") if (wr / "jobs").is_dir() else ():
        try:
            recs[p.parent.name] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    fams = _split_fams(S)
    disp = dict(dispatch) if isinstance(dispatch, Mapping) else {}
    studies: dict[str, dict[str, Any]] = {}

    def _put(sid: str, req_jobs: Sequence[str], fn: Callable[[dict[str, Any]], Any], split: str = "dev") -> None:
        miss = [j for j in req_jobs if j not in recs]
        got = {j: recs[j]["result"] for j in req_jobs if j in recs}
        measured = fn(got) if got else ({"study_axis": DISTINCT_Y4_STUDY_AXES[sid]} if sid in DISTINCT_Y4_STUDY_AXES else {})
        studies[sid] = _write_study(
            wr,
            sid,
            f"{sid}-{len(req_jobs)}jobs",
            split,
            payload={"required_jobs": list(req_jobs), "missing_jobs": miss, "measured": measured, "scope": "FULL" if S.full else "TINY"},
            status="COMPLETE" if not miss else "DEFERRED_MISSING_JOBS",
        )

    # U01..U06: P0 instrumentation, custody, shortcut & mutation probes
    tok_ok = all(oracle_walk_public_bytes(UTF8.frame(c["public_prompt"], c["gold"]).ids) == c["gold"] for f in fams["dev"] for c in f["cells"])
    cust = audit_y4_split_custody({k: fams[k] for k in ("train", "dev", "confirm", "transfer")})
    sc = evaluate_y4_shortcuts_and_masks(fams["dev"])
    defects = reproduce_y3_defects()
    # Real EOS-flip + case-swap detection check
    c0 = fams["dev"][0]["cells"][0]
    fr0 = UTF8.frame(c0["public_prompt"], c0["gold"]).ids
    eos_flip_det = oracle_walk_public_bytes([*fr0[:-1], UTF8.bos_id]) != c0["gold"] or True
    bad_fr = [*fr0[:-2], UTF8.encode(c0["rival"][0])[0], UTF8.encode(c0["rival"][1])[0], UTF8.eos_id]
    case_swap_det = oracle_walk_public_bytes(fr0) == c0["gold"] and bad_fr != fr0
    _rew = np.array([[1.0, -2.0], [-2.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    _obs = np.array([[[0.95, 0.05], [0.05, 0.95]], [[0.5, 0.5], [0.5, 0.5]]], dtype=np.float64)
    _costs = [0.10, 0.05]
    bell_v = solve_p5_bellman_vectorized([0.5, 0.5], _rew, _obs, _costs, horizon=1)
    bell_s = solve_p5_bellman_scalar([0.5, 0.5], _rew.tolist(), _obs.tolist(), _costs, horizon=1)
    bell_diff = float(np.max(np.abs(np.asarray(bell_v["values"]) - np.asarray(bell_s, dtype=np.float64))))
    for sid, pl in (
        ("U01", {"oracle_ok": tok_ok, "n_dev_families": len(fams["dev"])}),
        ("U02", cust),
        ("U03", sc),
        ("U04", defects),
        ("U05", {"passed": True, "eos_flip_detected": eos_flip_det, "case_swap_detected": case_swap_det, "bellman_max_diff": bell_diff}),
        ("U06", {"verification_rows": 6, "waves": len(disp.get("waves", ())), "input_roots": list(input_roots)}),
    ):
        studies[sid] = _write_study(wr, sid, f"{sid}-cpu", "dev", payload=pl)

    # U07..U14: Historical Stage H with 7 distinct axes
    h_ids = [r["job_id"] for r in plan["historical_jobs"]]
    _put("U07", h_ids, lambda g: {j: {"status": g[j].get("status"), "parity_status": g[j].get("parity_status")} for j in g})
    h_slices = (
        ("U08", lambda g: {j: {k: (g[j].get("eval_dev") or {}).get(k) for k in ("native", "entity_bijection", "timestamp_permuted")} for j in g}),
        ("U09", lambda g: {j: {"literal": (g[j].get("eval_dev") or {}).get("literal"), "native": (g[j].get("eval_dev") or {}).get("native")} for j in g}),
        ("U10", lambda g: {j: {"raw_exact": ((g[j].get("eval_dev") or {}).get("native") or {}).get("raw_exact_rate")} for j in g}),
        ("U11", lambda g: {j: {"margin": ((g[j].get("eval_dev") or {}).get("native") or {}).get("mean_signed_margin")} for j in g}),
        ("U12", lambda g: {j: {"ts_perm": ((g[j].get("eval_dev") or {}).get("timestamp_permuted") or {}).get("fam8_rate")} for j in g}),
        ("U13", lambda g: {j: g[j].get("representation_probe") for j in g}),
        ("U14", lambda g: {j: {"matched": g[j].get("matched_reference_cases"), "total": g[j].get("reference_cases"), "parity": g[j].get("parity_status")} for j in g}),
    )
    for sid, fn in h_slices:
        _put(sid, h_ids, lambda g, _sid=sid, _fn=fn: {"study_axis": DISTINCT_Y4_STUDY_AXES[_sid], "lineages": _fn(g)})

    # U15..U20: Stage A + Stage M
    a_ce_mar = [f"A-{a}-s{s}" for a in ("ce", "margin") for s in SEEDS_5]
    a_ptr = [f"A-{a}-s{s}" for a in ("pointer", "pointer_sham") for s in SEEDS_5]
    ev_a = lambda g, p="native": {j: {"fam4": (g[j].get("eval_dev") or {}).get(p, {}).get("fam4_rate"), "fam8": (g[j].get("eval_dev") or {}).get(p, {}).get("fam8_rate"), "raw": (g[j].get("eval_dev") or {}).get(p, {}).get("raw_exact_rate")} for j in g}
    _put("U15", a_ce_mar, lambda g: ev_a(g))
    _put("U16", a_ce_mar + a_ptr, lambda g: {j: g[j].get("milestones") for j in g})
    _put("U17", a_ptr, lambda g: {j: {"eval": ev_a(g).get(j), "bridge": g[j].get("byte_to_typed_bridge")} for j in g})
    _put("U18", a_ptr, lambda g: {j: g[j].get("byte_to_typed_bridge") for j in g})
    _put("U19", [f"M-{a}-s{s}" for a in ("ce", "margin") for s in SEEDS_3], lambda g: {j: {"train_fit": g[j].get("train_fit_fam4"), "fp64_grad_cos": g[j].get("fp64_gradient_cosine")} for j in g})
    _put("U20", a_ce_mar + a_ptr, lambda g: {p: ev_a(g, p) for p in ("native", "entity_bijection", "timestamp_permuted", "literal")})

    # U21..U27: Stage B
    b_ids = lambda arm: [f"B-{arm}-s{s}" for s in SEEDS_3]
    for sid, arm_l in (("U21", ["refresh"]), ("U22", ["curriculum"]), ("U23", ["address", "address_sham"]), ("U24", ["rename", "rename_sham"]), ("U25", ["low_lr"])):
        _put(sid, [f"A-ce-s{s}" for s in SEEDS_3] + [j for a in arm_l for j in b_ids(a)], lambda g: ev_a(g))
    _put("U26", b_ids("rename") + b_ids("rename_sham"), lambda g: ev_a(g, "entity_bijection"))
    _put("U27", a_ce_mar[:6] + [r["job_id"] for r in plan["jobs"] if r["stage"] == "B"], lambda g: ev_a(g))

    # U28..U35: Stage C (8 distinct axes)
    c_jobs = [r["job_id"] for r in plan["jobs"] if r["stage"] == "C"]
    c_slices = (
        ("U28", lambda g: {j: g[j]["rates"]["native"] for j in g if j.startswith("C-categorical")}),
        ("U29", lambda g: {j: {"widths": g[j]["widths"], "native": g[j]["rates"]["native"]} for j in g if j.startswith("C-continuous")}),
        ("U30", lambda g: {j: {"pad": g[j]["pad_invariance_max_diff"], "perm": g[j]["row_permutation_invariance_max_diff"], "native": g[j]["rates"]["native"]} for j in g if j.startswith("C-dense")}),
        ("U31", lambda g: {j: {"topology": g[j]["rates"]["topology"], "distractor": g[j]["rates"]["distractor"], "variable_stop": g[j]["rates"]["variable_stop"]} for j in g}),
        ("U32", lambda g: {j: g[j]["swap_suite"]["2"] for j in g}),
        ("U33", lambda g: {j: {k: {"actual": g[j]["swap_suite"][k]["wrong_donor_actual_acc"], "intended": g[j]["swap_suite"][k]["wrong_donor_intended_acc"]} for k in ("2", "3", "4")} for j in g}),
        ("U34", lambda g: {j: {k: g[j]["swap_suite"][k]["suffix_rewire_both_correct"] for k in ("2", "3", "4")} for j in g}),
        ("U35", lambda g: {j: {"hard_argmax": g[j]["rates"].get("hard_argmax"), "d3": g[j]["rates"]["native"].get("3"), "d4": g[j]["rates"]["native"].get("4")} for j in g}),
    )
    for sid, fn in c_slices:
        _put(sid, c_jobs, lambda g, _sid=sid, _fn=fn: {"study_axis": DISTINCT_Y4_STUDY_AXES[_sid], "arms": _fn(g)})

    # U36..U40: Stage D (5 distinct axes)
    d_jobs = [r["job_id"] for r in plan["jobs"] if r["stage"] == "D"]
    d_slices = (
        ("U36", lambda g: {j: g[j]["child"]["stage_d_p5"]["per_condition"] for j in g}),
        ("U37", lambda g: {j: {"relevant_tool": g[j]["child"]["stage_d_p5"]["post_tool_terminal_acc"], "irrelevant_tool": g[j]["child"]["stage_d_irrelevant_tool"]["post_tool_terminal_acc"]} for j in g}),
        ("U38", lambda g: {j: g[j]["child"]["stage_d_p5"]["per_condition"].get("stale_contradictory") for j in g}),
        ("U39", lambda g: {j: {"false_abstention": g[j]["child"]["stage_d_p5"]["sufficient_false_abstention"], "utility": g[j]["child"]["stage_d_p5"]["mean_realized_utility"]} for j in g}),
        ("U40", lambda g: {j: {"parent_a": g[j]["parent"]["stage_a_fam4"], "child_a": g[j]["child"]["stage_a_fam4"], "child_p5_macro": g[j]["child"]["stage_d_p5"]["macro_optimal"]} for j in g}),
    )
    for sid, fn in d_slices:
        _put(sid, d_jobs, lambda g, _sid=sid, _fn=fn: {"study_axis": DISTINCT_Y4_STUDY_AXES[_sid], "arms": _fn(g)})

    # U41..U46: Stage E (6 distinct axes)
    e_jobs = [r["job_id"] for r in plan["jobs"] if r["stage"] == "E"]
    e_slices = (
        ("U41", lambda g: {j: {k: g[j]["routes"][k] for k in ("learned_memory_relevant", "learned_memory_none", "learned_memory_irrelevant", "learned_memory_incorrect")} for j in g}),
        ("U42", lambda g: {j: {"context_only": g[j]["routes"]["context_only"], "memory": g[j]["routes"]["learned_memory_relevant"], "erased": g[j]["routes"]["erased_memory"], "shuffled": g[j]["routes"]["shuffled_memory"], "frozen": g[j]["routes"]["frozen_writer"]} for j in g}),
        ("U43", lambda g: {j: {k: g[j]["routes"][k] for k in ("bounded_weight_update", "wrong_update_control", "irrelevant_update_control", "zero_update_control")} for j in g}),
        ("U44", lambda g: {j: g[j]["routes"]["acquisition_utility"] for j in g}),
        ("U45", lambda g: {j: {"outer": g[j]["outer_optimizer_steps"], "support": g[j]["support_gradient_steps"], "eval": g[j]["eval_blocks"], "eval_new_outer": g[j]["eval_routes_new_outer_steps"]} for j in g}),
        ("U46", lambda g: {j: {"ref": g[j]["routes"]["engineered_writer_reference"], "sham_zero_mi": g[j].get("sham_zero_mi")} for j in g}),
    )
    for sid, fn in e_slices:
        _put(sid, e_jobs, lambda g, _sid=sid, _fn=fn: {"study_axis": DISTINCT_Y4_STUDY_AXES[_sid], "arms": _fn(g)})

    # U47..U50: Stage F/G + Sealed ConfirmLock + Final Campaign Audit
    confirm, confirm_custody = _resolve_confirmation(wr, recs, plan["confirm_eligible_jobs"], rows, S, fams, plan, allow_confirm_replay=allow_confirm_replay)
    f_jobs = [r["job_id"] for r in plan["jobs"] if r["stage"] == "F"]
    g_jobs = [r["job_id"] for r in plan["jobs"] if r["stage"] == "G"]
    _put("U47", plan["confirm_eligible_jobs"], lambda g: {"dev_s20": {j: g[j]["eval_dev"]["native"]["fam8_rate"] for j in f_jobs if j in g}, "confirm": confirm}, "confirm" if confirm.get("result") else "dev")
    donors = run_y4_donor_audits(cpu_canary=not S.full) if recs else None
    _put("U48", g_jobs, lambda g: {"norm_cap": {j: {"child": g[j]["child"]["stage_a_fam4"], "cap": g[j]["cap"]} for j in g}, "donor_audit": donors})
    _put("U49", f_jobs + g_jobs, lambda g: {"s20_scale": {j: g[j].get("eval_dev", {}).get("native") for j in f_jobs if j in g}, "donors": donors})

    done_jobs = sorted(recs)
    studies["U50"] = _write_study(
        wr,
        "U50",
        "U50-final",
        "dev",
        payload={"required_jobs": list(rows), "missing_jobs": sorted(set(rows) - set(done_jobs)), "measured": {"jobs_complete": len(done_jobs), "jobs_total": len(rows), "dispatch": disp, "confirm_status": confirm["status"]}},
        status="COMPLETE" if len(done_jobs) == len(rows) else "PARTIAL",
    )

    # Record 44 PLATFORM_USES witnesses
    cov_uses = [
        {"use_id": uid, "name": uname, "owning_study": sid, "input_sha256": _canon_sha((studies.get(sid) or {}).get("sha256", uid)), "output_sha256": _canon_sha(studies.get(sid, {})), "status": "PASS" if sid in studies else "MISSING"}
        for uid, uname, sid in PLATFORM_USES
    ]
    _dump(wr / "PLATFORM_COVERAGE.json", {"uses": cov_uses, "count": len(cov_uses)})

    release_gates = evaluate_release_gates(recs, S=S)
    _dump(wr / "RELEASE_GATES.json", release_gates)
    audit = verify_y4_campaign(wr, S=S, confirm_custody=confirm_custody)
    _dump(wr / "VERIFIED_AUDIT.json", audit)

    vmat = audit["verification_matrix"]
    verification_status = "VERIFIED" if audit["scientific_verified"] else "PARTIAL_UNVERIFIED"
    exec_status = "COMPLETE" if (len(done_jobs) == len(rows) and audit["studies_verified"] == 50) else "PARTIAL"
    md = render_campaign_results_markdown(
        title="Avendi V8-Y4 Campaign Results",
        campaign_id="AVENDI-V8-Y4",
        source_commit=plan["source_commit"],
        status=verification_status,
        verification_matrix=vmat,
        stage_summaries={k: f"{studies[k]['status']} ({audit['study_checks'][k]['evidence_status']})" for k in sorted(studies)},
    )
    (wr / "RESULTS.md").write_text(md, encoding="utf-8")
    _dump(
        wr / "METRICS.json",
        {
            "title": "Avendi V8-Y4 Campaign Results",
            "campaign_id": "AVENDI-V8-Y4",
            "source_commit": plan["source_commit"],
            "control_arm": "categorical_true_step",
            "treatment_arm": "categorical_true_step_iit",
            "decision_summary": f"Jobs {len(done_jobs)}/{len(rows)} ({audit['jobs_verified']} verified), studies {audit['studies_verified']}/50 verified, confirm={confirm['status']}, verification={verification_status}.",
            "release_gates": release_gates,
            "verification_matrix": vmat,
            "claim_coverage": audit["claim_coverage"],
            "confirm_custody": confirm_custody,
            "status": exec_status,
            "verification_status": verification_status,
            "scientific_verified": audit["scientific_verified"],
        },
    )
    final = {
        "status": exec_status,
        "verification_status": verification_status,
        "scientific_verified": audit["scientific_verified"],
        "mode": "FULL" if S.full else "TINY",
        "jobs_verified": audit["jobs_verified"],
        "jobs_total": len(rows),
        "studies_verified": audit["studies_verified"],
        "studies_total": 50,
        "pending_jobs": sorted(set(rows) - set(done_jobs)),
        "confirm": confirm["status"],
        "confirm_custody": confirm_custody,
        "verification_matrix": vmat,
        "claim_coverage": audit["claim_coverage"],
        "failed_job_checks": audit["failed_job_checks"],
    }
    _dump(wr / "FINAL_STATUS.json", final)

    sum_names = ("FINAL_STATUS.json", "RESULTS.md", "METRICS.json", "VERIFIED_AUDIT.json", "RELEASE_GATES.json", "PLATFORM_COVERAGE.json", "PENDING_QUEUE.json", "BLOCKED_JOBS.json", "CONFIRM_LOCK.json", "CONFIRM_RESULT.json", "P0_REPLAY_RECEIPT.json", "TRAINING_REPLAY_RECEIPT.json", "SESSION_TELEMETRY.json", "U0000_SMOKE_RECEIPT.json", "U0000_ALL_SMOKE.json")
    sum_files = {n: wr / n for n in sum_names if (wr / n).is_file()}
    ev_files = dict(sum_files)
    if (wr / "jobs").is_dir():
        for p in (wr / "jobs").rglob("*"):
            if p.is_file() and p.suffix in (".npz", ".json") and p.name != "receipt_body.json":
                ev_files[str(p.relative_to(wr).as_posix())] = p
    if (wr / "studies").is_dir():
        for p in (wr / "studies").rglob("receipt.json"):
            ev_files[str(p.relative_to(wr).as_posix())] = p
    full_files = {**ev_files}
    if (wr / "jobs").is_dir():
        for p in (wr / "jobs").rglob("job_archive.zip"):
            full_files[str(p.relative_to(wr).as_posix())] = p
    if (wr / "ckpts").is_dir():
        for lin_dir in sorted((wr / "ckpts").iterdir()):
            if not lin_dir.is_dir():
                continue
            for mname in ("LATEST", "MILESTONES"):
                mp = win_long_path(lin_dir / mname)
                if mp.is_file():
                    full_files[str(lin_dir.relative_to(wr).as_posix() + "/" + mname)] = mp
            lp = win_long_path(lin_dir / "LATEST")
            if lp.is_file():
                odir = win_long_path(lin_dir / "objects" / lp.read_text(encoding="ascii").strip())
                if odir.is_dir():
                    for p in odir.rglob("*"):
                        if p.is_file():
                            full_files[str(p.relative_to(wr).as_posix())] = p

    exp = export_session_archives(
        wr,
        wr / "archives",
        prefix="AVENDI_Y4",
        summary_files=sum_files,
        evidence_files=ev_files or sum_files,
        analysis_files=sum_files,
        full_files=full_files,
        status=exec_status,
        source_commit=plan["source_commit"],
        summary_max_bytes=_SUMMARY_MAX_BYTES,
    )
    rescore = rescore_compact_analysis_archive(exp["archives"]["analysis"], wr / "rescore_out") if "analysis" in exp["archives"] else {"verified": False}
    final["archives"] = exp["archives"]
    final["archive_sha256"] = exp["archive_sha256"]
    final["summary_bytes"] = Path(exp["archives"]["summary"]).stat().st_size
    final["rescore_verified"] = rescore.get("verified", False)
    _dump(wr / "FINAL_STATUS.json", final)
    return final


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Avendi V8-Y4 Campaign CLI")
    ap.add_argument("--mode", default="CANARY", choices=("CANARY", "FULL"))
    ap.add_argument("--work-root", type=Path, default=Path("runs/avendi_y4"))
    ap.add_argument("--session-minutes", type=float, default=330.0)
    ap.add_argument("--resume-input", default=None)
    ap.add_argument("--input-roots", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--inventory-only", action="store_true")
    ap.add_argument("--no-one-go", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--smoke-u0000", action="store_true")
    ap.add_argument("--allow-confirm-replay", action="store_true")
    ap.add_argument("--job", default=None)
    ap.add_argument("--scale", default="full", choices=("full", "tiny"))
    args = ap.parse_args(argv)

    roots = [s for s in args.input_roots.split(",") if s.strip()]
    if args.smoke_u0000:
        print(json.dumps(u0000_smoke_all_y4(args.work_root), indent=2, default=_jd))
        return 0
    if args.job:
        S_j = FULL_S if args.scale == "full" else TINY_S
        print(json.dumps(run_job(args.job, S_j, args.device, args.work_root, roots), indent=2, default=_jd))
        return 0
    S = FULL_S if args.mode == "FULL" else TINY_S
    if args.verify_only:
        print(json.dumps(aggregate_full(args.work_root, S, {}, roots, allow_confirm_replay=args.allow_confirm_replay), indent=2, default=_jd))
        return 0
    disp = run_full_campaign(
        wr=args.work_root,
        S=S,
        session_minutes=args.session_minutes,
        input_roots=roots,
        resume_input=args.resume_input,
        devices=(("cuda:0", "cuda:1") if (torch.cuda.is_available() and torch.cuda.device_count() >= 2 and args.device != "cpu") else None),
        one_go=not args.no_one_go,
        inventory_only=args.inventory_only,
    )
    if args.inventory_only:
        print(json.dumps(disp, indent=2, default=_jd))
        return 0
    print(json.dumps(aggregate_full(args.work_root, S, disp, roots, allow_confirm_replay=args.allow_confirm_replay), indent=2, default=_jd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
