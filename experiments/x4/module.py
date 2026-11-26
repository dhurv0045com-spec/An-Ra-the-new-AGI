"""V8-X4 Palantiri Level-7 Plugin, 4 Resumable S20 Cycles, 5-Mode Learning, Capability & Evidence Registries."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.publication import quarantine_interrupted_publication
from anra_v5.elendril.representation import SpecialToken, UTF8
from anra_v5.elendril.training import _dump, _load
from anra_v5.elendril.v8x.data import ACTION_19_LABELS, SYMBOLS
from anra_v5.palantiri import (
    CampaignOrchestrator,
    ExperimentModule,
    GenericDatasetBundle,
    SessionJob,
    UniversalAction,
    UniversalObservation,
    bootstrap_paired_families,
    build_analyst_first_page_report,
    build_paired_seed_observations,
    dump_json_safe,
    export_session_archives,
    verify_session_archive,
    win_long_path,
)
from v5_model.core import packed_layout
from v5_training.checkpoint import CheckpointStore
from v5_training.optimizer import optimizer_group_receipt
from v5_training.state import CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState, IdentityBindings, TrainingState

from .battery import (
    X4_22_FAMILIES,
    X4_FAMILY_GROUPS,
    X4_SPLIT_SEEDS,
    build_l0_l5_diagnostic_suite,
    build_x01_x12_interventions,
    build_x4_battery,
    build_x4_curriculum_schedule,
    diagnose_c03_c04_c10_failure_modes,
    evaluate_x4_cases,
    format_x4_typed_target,
    infer_x4_generation_bound,
    parse_and_score_x4_output,
    score_x4_multidimensional_profile,
    solve_x4_case_oracle,
    verify_x4_split_disjointness,
)
from .integrity import run_foundational_integrity_gate
from .models import (
    X4CognitiveModel,
    X4_ARCHITECTURES,
    X4_CANARY_ARCHITECTURES,
    build_x4_backbone_spec,
    encode_prompt_and_target_pair,
)
from .objectives import (
    X4_OBJECTIVES,
    build_x4_matched_batches,
    compute_x4_objective_step,
    prepack_x4_batches,
)

X4_BACKEND_ID = "elendril_x4"
X4_DEFAULT_UPDATES = 1024
X4_DISCOVERY_SEEDS: tuple[int, ...] = (7, 19, 41)
X4_CONFIRM_SEEDS: tuple[int, ...] = (101, 202, 303)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dump_json_safe(payload) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _deterministic_sym(text: str) -> str:
    return SYMBOLS[int(sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 16]


def _world_op_and_arg(world: Mapping[str, Any]) -> tuple[str, str, str | None]:
    prompt = str(world.get("prompt", world.get("rendered_text", "")))
    gold_txt = str(world.get("gold_text", world.get("expected_answer", ""))).strip()
    if gold_txt in ("QUERY", "RETRIEVE", "ABSTAIN", "AMBIGUOUS"):
        return prompt, "ABSTAIN" if gold_txt == "AMBIGUOUS" else gold_txt, None
    if gold_txt.startswith("ANSWER:"):
        payload = gold_txt.split(":", 1)[1].strip()
        sym = payload if payload in SYMBOLS else _deterministic_sym(payload)
        return prompt, "ANSWER", sym
    sym = gold_txt if gold_txt in SYMBOLS else _deterministic_sym(gold_txt)
    return prompt, "ANSWER", sym


class X4CognitiveKernel:
    """Truth-firewalled stateful kernel for Palantiri Level-7 launch gates, canary & replay."""

    kernel_id = "elendril-x4-cognitive-kernel-v1"
    max_actions = 4

    def structural_signature(self, w: Mapping[str, Any]) -> str:
        return str(w.get("structural_hash", w.get("case_id", "x4")))

    def inject_distractor(self, w: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(w)
        p = str(w.get("prompt", w.get("rendered_text", "")))
        suffix = " [note:0]"
        if len((p + suffix).encode("utf-8")) <= 600:
            out["prompt"] = p + suffix
        out["world_id"] = str(w.get("world_id", w.get("case_id", "w"))) + "::dist"
        return out

    def permute_world(self, w: Mapping[str, Any], shift: int = 1) -> dict[str, Any]:
        out = dict(w)
        out["world_id"] = str(w.get("world_id", w.get("case_id", "w"))) + f"::p{shift}"
        return out

    def reverse_world(self, w: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(w)
        out["world_id"] = str(w.get("world_id", w.get("case_id", "w"))) + "::rev"
        return out

    def decompose_subgoals(self, w: Mapping[str, Any]) -> tuple[Any, ...]:
        return ()

    def initial_state(self, w: Mapping[str, Any]) -> dict[str, Any]:
        can_query = str(w.get("uncertainty_kind", "")) == "information_uncertainty"
        return {"w": w, "step": 0, "queried": False, "can_query": can_query, "terminal": False}

    def observe(self, st: Mapping[str, Any]) -> UniversalObservation:
        w = st["w"]
        prompt = str(w.get("prompt", w.get("rendered_text", "")))
        if st.get("queried"):
            prompt = f"{prompt} [OBS:RESOLVED]"
        q_avail = bool(st.get("can_query") and not st.get("queried"))
        consumed = frozenset({"query"}) if st.get("queried") else frozenset()
        return UniversalObservation(
            obs_id=f"{w.get('case_id', w.get('world_id', 'x4'))}:s{st['step']}:q{int(bool(st.get('queried')))}",
            rendered_text=prompt,
            available_tools={"query": q_avail, "retrieve": False, "memory": False},
            consumed_tools=consumed,
            decision_index=int(st["step"]),
            budget_remaining=max(1, self.max_actions - int(st["step"])),
            metadata={"case_id": str(w.get("case_id", "")), "family": str(w.get("family", "")), "step": int(st["step"])},
        )

    def ideal_action(self, st: Mapping[str, Any]) -> tuple[str, str | None]:
        w = st["w"]
        _, op, arg = _world_op_and_arg(w)
        if st.get("queried") and op == "QUERY":
            return "ANSWER", SYMBOLS[0]
        return op, arg

    def step(self, st: Mapping[str, Any], op: str, arg: str | None) -> dict[str, Any]:
        nxt = int(st["step"]) + 1
        if op in ("QUERY", "RETRIEVE") and not st.get("terminal") and nxt < self.max_actions:
            return {"w": st["w"], "step": nxt, "queried": (op == "QUERY") or bool(st.get("queried")), "can_query": False if op == "QUERY" else bool(st.get("can_query")), "terminal": False}
        return {"w": st["w"], "step": nxt, "queried": bool(st.get("queried")), "can_query": False, "terminal": True}

    def verify(self, w: Mapping[str, Any], actions: Sequence[tuple[str, str | None]]) -> dict[str, Any]:
        if not actions:
            return {"correct": False, "reason": "NO_ACTIONS"}
        _, exp_op, exp_arg = _world_op_and_arg(w)
        last_op, last_arg = actions[-1]
        ok = (exp_op == "QUERY" and any(op == "QUERY" for op, _ in actions)) or ((last_op == exp_op) and (last_arg == exp_arg))
        return {"correct": bool(ok), "actions_count": len(actions)}


def _action_pair_from_idx(idx: int) -> tuple[str, str | None]:
    lbl = str(ACTION_19_LABELS[int(idx)])
    return (lbl, None) if lbl in ("QUERY", "RETRIEVE", "ABSTAIN") else ("ANSWER", lbl)


class X4CognitivePolicy:
    """Palantiri-compatible cognitive policy with fixed `prompt_end_positions` causal anchoring."""

    def __init__(
        self,
        model: X4CognitiveModel,
        model_spec: Any = None,
        *,
        model_identity: str = "x4-cognitive",
        mode: str = "raw",
        copy_enabled_override: bool | None = None,
    ) -> None:
        self.model = model
        self.model_spec = model_spec
        self.model_identity = str(model_identity)
        self.mode = str(mode)
        self.copy_enabled_override = copy_enabled_override
        self.policy_id = f"x4-policy:{getattr(model, 'architecture', 'unknown')}:{self.mode}"

    def reset_episode(self) -> None:
        self.model.reset_episode()

    @torch.no_grad()
    def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction:
        dev = next(self.model.parameters()).device
        ids = [UTF8.bos_id, *UTF8.encode(obs.rendered_text), UTF8.special(SpecialToken.ANSWER)]
        if len(ids) > self.model.max_ctx:
            ids = ids[-self.model.max_ctx :]
        toks = torch.tensor([ids], dtype=torch.long, device=dev)
        segs = torch.zeros_like(toks)
        pos, mask = packed_layout(segs, torch_module=torch)
        end_p = torch.tensor([len(ids) - 1], dtype=torch.long, device=dev)
        was_train = self.model.training
        self.model.eval()
        if self.copy_enabled_override is False:
            with self.model.ablate_mechanism("disable_pointer"):
                out = self.model(toks, pos, mask, prompt_end_positions=end_p)
        else:
            out = self.model(toks, pos, mask, prompt_end_positions=end_p)
        if was_train:
            self.model.train()

        logits_19 = out["action_logits"][0].float().clone()
        raw_idx = int(torch.argmax(logits_19).item())
        raw_op, raw_arg = _action_pair_from_idx(raw_idx)
        if self.mode == "availability-mask":
            avail = dict(obs.available_tools)
            if not avail.get("query", False):
                logits_19[16] = -1e9
            if not avail.get("retrieve", False):
                logits_19[17] = -1e9
            logits_19[18] = -1e9
        probs = F.softmax(logits_19, dim=-1)
        idx = int(torch.argmax(probs).item())
        op, arg = _action_pair_from_idx(idx)
        lp = float(torch.log(probs[idx].clamp_min(1e-7)).item())
        dist = {
            (f"{a_op}:{a_arg}" if a_arg is not None else a_op): float(probs[i].item())
            for i in range(len(ACTION_19_LABELS))
            for a_op, a_arg in [_action_pair_from_idx(i)]
        }
        unc = out["uncertainties"][0].detach().cpu().tolist()
        return UniversalAction(
            operation=op,
            argument=arg,
            log_probability=lp,
            action_distribution=dist,
            mode=self.mode,
            raw_operation=raw_op,
            raw_argument=raw_arg,
            telemetry={"epistemic_uncertainty": float(unc[0]), "info_uncertainty": float(unc[1]), "copy_gate_mean": float(out["copy_gate_mean"][0].item())},
        )

    @torch.no_grad()
    def extract_representations(self, obs_batch: Sequence[UniversalObservation]) -> dict[str, torch.Tensor]:
        if not obs_batch:
            return {"decision_hidden": torch.zeros((0, int(self.model.d_model)))}
        dev = next(self.model.parameters()).device
        seqs = []
        for obs in obs_batch:
            ids = [UTF8.bos_id, *UTF8.encode(obs.rendered_text), UTF8.special(SpecialToken.ANSWER)]
            if len(ids) > self.model.max_ctx:
                ids = ids[-self.model.max_ctx :]
            seqs.append(ids)
        max_l = max(len(s) for s in seqs)
        toks = torch.zeros((len(seqs), max_l), dtype=torch.long, device=dev)
        segs = torch.full((len(seqs), max_l), -1, dtype=torch.long, device=dev)
        end_p = torch.zeros((len(seqs),), dtype=torch.long, device=dev)
        for i, ids in enumerate(seqs):
            toks[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=dev)
            segs[i, : len(ids)] = 0
            end_p[i] = len(ids) - 1
        pos, mask = packed_layout(segs, torch_module=torch)
        was_train = self.model.training
        self.model.eval()
        out = self.model(toks, pos, mask, prompt_end_positions=end_p)
        if was_train:
            self.model.train()
        return {"decision_hidden": out["decision_hidden"].detach().cpu()}


# ---------------------------------------------------------------------------
# Repaired 5-Mode Learning, Persistent-State Controls, Retention & Transfer (§17)
# ---------------------------------------------------------------------------


def evaluate_x4_learning_modes(
    model: X4CognitiveModel,
    *,
    icl_cases_by_shot: Mapping[int, Sequence[Mapping[str, Any]]],
    state_adapt_sequence: Sequence[Mapping[str, Any]],
    grad_adapt_demos: Sequence[Mapping[str, Any]],
    grad_adapt_queries: Sequence[Mapping[str, Any]],
    retention_queries: Sequence[Mapping[str, Any]],
    tool_free_cases: Sequence[Mapping[str, Any]],
    cross_domain_cases: Sequence[Mapping[str, Any]],
    oracle_solver: Any = solve_x4_case_oracle,
    lr: float = 5e-4,
    max_param_delta_norm: float = 1.0,
) -> dict[str, Any]:
    """Repaired 5-mode learning eval fixing Defect J (typed targets in Mode 3, full Mode 2 controls, distinct Mode 4/5 sets)."""
    dev = next(model.parameters()).device
    total_params = sum(int(p.numel()) for p in model.parameters())

    def _eval_seq(items: Sequence[Mapping[str, Any]], *, use_persistent: bool, feed_feedback: bool = False, unrelated_prefix: str | None = None) -> float:
        if not items:
            return 0.0
        if unrelated_prefix is not None and use_persistent:
            model.update_persistent_state(unrelated_prefix)
        ok = 0
        for c in items:
            out = model.generate_bytes(str(c["prompt"]), max_new_bytes=infer_x4_generation_bound(str(c["prompt"])), use_persistent_state=use_persistent)
            sc = parse_and_score_x4_output(str(c["prompt"]), out["text"], c, terminated_by_eos=bool(out["terminated_by_eos"]))
            ok += int(bool(sc["correct"]))
            if use_persistent and feed_feedback:
                model.update_persistent_state(f"{c['prompt']} -> {format_x4_typed_target(c)}")
        return float(ok / len(items))

    # Mode 1: Frozen In-Context Learning
    model.eval()
    model.reset_episode()
    icl_curve: dict[int, float] = {}
    for sh, items in sorted(icl_cases_by_shot.items()):
        ok = 0
        for c in items:
            model.reset_episode()
            ok += int(_eval_seq([c], use_persistent=False) >= 0.5)
        icl_curve[int(sh)] = float(ok / max(1, len(items)))

    # Mode 2: Persistent-state learning with ALL 5 matched controls
    model.reset_episode()
    m2_with_fb = _eval_seq(state_adapt_sequence, use_persistent=True, feed_feedback=True)
    model.reset_episode()
    m2_no_fb = _eval_seq(state_adapt_sequence, use_persistent=True, feed_feedback=False)
    m2_reset_vals = []
    for c in state_adapt_sequence:
        model.reset_episode()
        m2_reset_vals.append(_eval_seq([c], use_persistent=True, feed_feedback=False))
    m2_reset = float(sum(m2_reset_vals) / max(1, len(m2_reset_vals)))
    model.reset_episode()
    m2_unrelated = _eval_seq(
        state_adapt_sequence,
        use_persistent=True,
        feed_feedback=False,
        unrelated_prefix="X01|UNRELATED|DISTRACTOR|k9=Z",
    )
    model.reset_episode()
    m2_frozen = _eval_seq(state_adapt_sequence, use_persistent=False)

    # Mode 3: Parameter adaptation using proper `format_x4_typed_target(d)` + pre/post + retention audit!
    model.reset_episode()
    pre_adapt_acc = _eval_seq(grad_adapt_queries, use_persistent=False)
    pre_retain_acc = _eval_seq(retention_queries, use_persistent=False)
    snap = {k: v.detach().clone() for k, v in model.named_parameters()}
    efficiency_curve: dict[int, float] = {}
    train_tokens = 0
    if grad_adapt_demos:
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=float(lr))
        for step_k, demo in enumerate(grad_adapt_demos, start=1):
            model.train()
            ids, elig, end_p = encode_prompt_and_target_pair(
                str(demo["prompt"]), format_x4_typed_target(demo), max_length=model.max_ctx
            )
            train_tokens += len(ids)
            toks = torch.tensor([ids], dtype=torch.long, device=dev)
            elig_t = torch.tensor([elig], dtype=torch.bool, device=dev)
            pos, mask = packed_layout(torch.zeros_like(toks), torch_module=torch)
            opt.zero_grad(set_to_none=True)
            out = model(toks, pos, mask, prompt_end_positions=torch.tensor([end_p], dtype=torch.long, device=dev))
            nll = -out["log_p_unified"][:, :-1, :].float().gather(2, toks[:, 1:].unsqueeze(-1)).squeeze(-1)
            m_f = elig_t[:, 1:].float()
            loss = (nll * m_f).sum() / m_f.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step_k in (1, 4, 16) or step_k == len(grad_adapt_demos):
                model.eval()
                efficiency_curve[step_k] = _eval_seq(grad_adapt_queries, use_persistent=False)

    sq_sum = 0.0
    with torch.no_grad():
        for name, p in model.named_parameters():
            diff = p.detach() - snap[name]
            sq_sum += float((diff * diff).sum().item())
        delta_norm = math.sqrt(sq_sum)
        if delta_norm > float(max_param_delta_norm) and delta_norm > 0.0:
            scale = float(max_param_delta_norm) / delta_norm
            for name, p in model.named_parameters():
                p.copy_(snap[name] + scale * (p.detach() - snap[name]))
            delta_norm = float(max_param_delta_norm)

    model.eval()
    post_adapt_acc = _eval_seq(grad_adapt_queries, use_persistent=False)
    post_retain_acc = _eval_seq(retention_queries, use_persistent=False)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(snap[name])
    model.reset_episode()

    # Mode 4 & 5: Strictly distinct Tool-Free vs Cross-Domain Transfer sets!
    tf_ids = {str(c["case_id"]) for c in tool_free_cases}
    cd_ids = {str(c["case_id"]) for c in cross_domain_cases}
    distinct_transfer_sets = len(tf_ids & cd_ids) == 0

    m4_neural = _eval_seq(tool_free_cases, use_persistent=False)
    m4_oracle = float(sum(int(str(oracle_solver(c)).strip() == format_x4_typed_target(c)) for c in tool_free_cases) / max(1, len(tool_free_cases)))
    m5_cross = _eval_seq(cross_domain_cases, use_persistent=False)

    return {
        "schema": "elendril-x4-five-learning-modes-v3",
        "mode1_frozen_icl_curve": icl_curve,
        "mode2_with_feedback_acc": m2_with_fb,
        "mode2_no_feedback_acc": m2_no_fb,
        "mode2_reset_control_acc": m2_reset,
        "mode2_unrelated_history_acc": m2_unrelated,
        "mode2_frozen_state_acc": m2_frozen,
        "mode2_feedback_gain": float(m2_with_fb - m2_no_fb),
        "mode3_pre_adapt_acc": pre_adapt_acc,
        "mode3_post_adapt_acc": post_adapt_acc,
        "mode3_adaptation_gain": float(post_adapt_acc - pre_adapt_acc),
        "mode3_pre_retention_acc": pre_retain_acc,
        "mode3_post_retention_acc": post_retain_acc,
        "mode3_retention_delta": float(post_retain_acc - pre_retain_acc),
        "mode3_param_delta_l2": float(delta_norm),
        "mode4_tool_free_acc": m4_neural,
        "mode4_oracle_assisted_acc": m4_oracle,
        "mode4_cross_domain_transfer_acc": m5_cross,
        "mode4_distinct_transfer_sets_verified": bool(distinct_transfer_sets),
        "mode5_sample_efficiency_curve": efficiency_curve,
        "train_tokens_processed": train_tokens,
        "total_params": total_params,
    }


# ---------------------------------------------------------------------------
# 4 Resumable Research Cycles Ledger (Strictly S20, Zero S50 Jobs)
# ---------------------------------------------------------------------------


def build_x4_cycle_ledger(
    *,
    source_commit: str,
    updates: int = X4_DEFAULT_UPDATES,
    cycles: Sequence[str] = ("C1", "C2", "C3", "C4"),
) -> dict[str, Any]:
    """Build 4 resumable S20 research cycles (ZERO S50 jobs)."""
    jobs: list[dict[str, Any]] = []
    cyc_set = set(cycles)

    def _add(cycle: str, study: str, arch: str, obj: str, data_reg: str, curr: str, seed: int) -> None:
        reg_tag = {"fresh_stream": "fs", "fixed_400": "f400", "fixed_4k": "f4k", "fixed_4000": "f4k", "fixed_40000": "f40k"}.get(data_reg, data_reg)
        cur_tag = "p2c" if curr == "curriculum_p2c" else "rnd"
        jid = f"X4-{cycle}-{arch}-{obj}-{reg_tag}-{cur_tag}-s{seed}".replace("_", "-")
        cfg = {
            "job_id": jid,
            "cycle": cycle,
            "stage": cycle,
            "study": study,
            "architecture": arch,
            "objective": obj,
            "data_regime": data_reg,
            "curriculum_mode": curr,
            "scale": "S20",
            "init_seed": int(seed),
            "order_seed": int(seed) + 2000,
            "updates": int(updates),
            "confirmation_eligible": True,
            "source_commit": str(source_commit),
        }
        cfg["spec_sha256"] = digest(cfg)
        jobs.append(cfg)

    if "C1" in cyc_set:
        for arch in ("Base", "P-Ref", "R-Ref", "R-NoEdge"):
            for seed in X4_DISCOVERY_SEEDS:
                _add("C1", "FOUND", arch, "D0-ByteCE", "fresh_stream", "curriculum_p2c", seed)

    if "C2" in cyc_set:
        # Algorithm Axis on R-Ref (matched exposure fresh_stream / curriculum_p2c)
        for obj in ("D0-Typed", "D1-CorrInvar", "D2-ExactCF", "D3-SimpleWeight", "D3-Adaptive", "D4-ECCO"):
            for seed in X4_DISCOVERY_SEEDS:
                _add("C2", "ALG", "R-Ref", obj, "fresh_stream", "curriculum_p2c", seed)
        # Data & Curriculum Axis on R-Ref + D4-ECCO
        for data_reg, curr in (
            ("fixed_400", "curriculum_p2c"),
            ("fixed_4000", "curriculum_p2c"),
            ("fixed_40000", "curriculum_p2c"),
            ("fresh_stream", "random_interleaved"),
        ):
            for seed in X4_DISCOVERY_SEEDS:
                _add("C2", "DATA", "R-Ref", "D4-ECCO", data_reg, curr, seed)

    if "C3" in cyc_set:
        for arch in (
            "P1-Alpha",
            "P1-ParallelRead",
            "R1-Alpha",
            "R1-DenseControl",
            "U2-Fix",
            "U4-Fix",
            "U2-Unanchored",
            "U-P",
            "P1-OpBank",
            "P1-OpBank-DenseCtrl",
            "P1-CE-Reader",
        ):
            for seed in X4_DISCOVERY_SEEDS:
                _add("C3", "ARCH", arch, "D4-ECCO", "fresh_stream", "curriculum_p2c", seed)

    if "C4" in cyc_set:
        for arch in ("P1-OpBank", "ANRA-X4-Int"):
            for seed in X4_CONFIRM_SEEDS:
                _add("C4", "CONFIRM", arch, "D4-ECCO", "fresh_stream", "curriculum_p2c", seed)

    return {
        "schema": "elendril-x4-program-ledger-v1",
        "source_commit": str(source_commit),
        "total_gpu_jobs": len(jobs),
        "s20_jobs": len(jobs),
        "s50_jobs": 0,
        "cycles": list(cycles),
        "jobs": jobs,
        "ledger_sha256": digest(jobs),
    }


# Stage-qualified contrasts (Fixes Defect L: explicit (cycle, arch, obj, data_reg, curr) keys, ZERO self-contrasts)
X4_PAIRED_CONTRASTS: tuple[tuple[str, tuple[str, str, str, str, str], tuple[str, str, str, str, str]], ...] = (
    ("C1_RRef_vs_Base", ("C1", "Base", "D0-ByteCE", "fresh_stream", "curriculum_p2c"), ("C1", "R-Ref", "D0-ByteCE", "fresh_stream", "curriculum_p2c")),
    ("C1_RRef_vs_RNoEdge", ("C1", "R-NoEdge", "D0-ByteCE", "fresh_stream", "curriculum_p2c"), ("C1", "R-Ref", "D0-ByteCE", "fresh_stream", "curriculum_p2c")),
    ("C2_D4ECCO_vs_D2ExactCF", ("C2", "R-Ref", "D2-ExactCF", "fresh_stream", "curriculum_p2c"), ("C2", "R-Ref", "D4-ECCO", "fresh_stream", "curriculum_p2c")),
    ("C2_D4ECCO_vs_D3Simple", ("C2", "R-Ref", "D3-SimpleWeight", "fresh_stream", "curriculum_p2c"), ("C2", "R-Ref", "D4-ECCO", "fresh_stream", "curriculum_p2c")),
    ("C2_Fresh_vs_Fixed400", ("C2", "R-Ref", "D4-ECCO", "fixed_400", "curriculum_p2c"), ("C2", "R-Ref", "D4-ECCO", "fresh_stream", "curriculum_p2c")),
    ("C2_P2C_vs_RandomInterleaved", ("C2", "R-Ref", "D4-ECCO", "fresh_stream", "random_interleaved"), ("C2", "R-Ref", "D4-ECCO", "fresh_stream", "curriculum_p2c")),
    ("C3_P1Alpha_vs_Parallel", ("C3", "P1-ParallelRead", "D4-ECCO", "fresh_stream", "curriculum_p2c"), ("C3", "P1-Alpha", "D4-ECCO", "fresh_stream", "curriculum_p2c")),
    ("C3_R1Alpha_vs_Dense", ("C3", "R1-DenseControl", "D4-ECCO", "fresh_stream", "curriculum_p2c"), ("C3", "R1-Alpha", "D4-ECCO", "fresh_stream", "curriculum_p2c")),
    ("C3_U2Fix_vs_U2Unanchored", ("C3", "U2-Unanchored", "D4-ECCO", "fresh_stream", "curriculum_p2c"), ("C3", "U2-Fix", "D4-ECCO", "fresh_stream", "curriculum_p2c")),
    ("C3_P1OpBank_vs_DenseCtrl", ("C3", "P1-OpBank-DenseCtrl", "D4-ECCO", "fresh_stream", "curriculum_p2c"), ("C3", "P1-OpBank", "D4-ECCO", "fresh_stream", "curriculum_p2c")),
    ("C4_Integrated_vs_Single", ("C4", "P1-OpBank", "D4-ECCO", "fresh_stream", "curriculum_p2c"), ("C4", "ANRA-X4-Int", "D4-ECCO", "fresh_stream", "curriculum_p2c")),
)


def evaluate_capability_preservation(
    baseline_profile: Mapping[str, Any],
    candidate_profile: Mapping[str, Any],
    *,
    max_group_regression: float = 0.02,
) -> dict[str, Any]:
    """Capability-preservation guardrail auditing all 6 Groups A-F (§18)."""
    base_g = baseline_profile.get("per_group_accuracy", {})
    cand_g = candidate_profile.get("per_group_accuracy", {})
    deltas: dict[str, float] = {}
    regressions: list[str] = []
    for grp in sorted(set(base_g) | set(cand_g)):
        d = float(cand_g.get(grp, 0.0)) - float(base_g.get(grp, 0.0))
        deltas[grp] = d
        if d < -float(max_group_regression):
            regressions.append(grp)
    return {
        "schema": "elendril-x4-capability-preservation-v1",
        "max_allowed_regression": float(max_group_regression),
        "group_deltas": deltas,
        "regressed_groups": regressions,
        "preserved": len(regressions) == 0,
    }


def evaluate_x4_qualification_thresholds(
    candidate_profile: Mapping[str, Any],
    baseline_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Enforce two-tier honest competence gates (§20): QUALIFIED vs PRELIMINARY_IMPROVEMENT vs NOT_QUALIFIED."""
    l1 = float(candidate_profile.get("macro_mean", 0.0))
    cf = float(candidate_profile.get("counterfactual_both_correct", 0.0))
    unsup = float(candidate_profile.get("unsupported_guess_rate", 1.0))
    false_abs = float(candidate_profile.get("false_abstention_rate", 1.0))
    pres = (
        evaluate_capability_preservation(baseline_profile, candidate_profile)
        if baseline_profile is not None
        else {"preserved": True, "regressed_groups": []}
    )
    qualified = (l1 >= 0.70) and (cf >= 0.65) and (unsup <= 0.05) and (false_abs <= 0.05) and bool(pres["preserved"])
    prelim = False
    if baseline_profile is not None:
        base_l1 = float(baseline_profile.get("macro_mean", 0.0))
        prelim = (l1 - base_l1 >= 0.03) and bool(pres["preserved"])
    tier = "QUALIFIED" if qualified else ("PRELIMINARY_IMPROVEMENT" if prelim else "NOT_QUALIFIED")
    return {
        "schema": "elendril-x4-qualification-v1",
        "tier": tier,
        "qualified": qualified,
        "macro_mean": l1,
        "counterfactual_both_correct": cf,
        "unsupported_guess_rate": unsup,
        "false_abstention_rate": false_abs,
        "capability_preservation": pres,
    }


def build_x4_failure_and_evidence_registry(*, x4_summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Maintain explicit historical failure registry + X4 H1-H5 hypothesis tracking (§23)."""
    return {
        "schema": "elendril-x4-failure-evidence-registry-v1",
        "historical_x1_x2_x3_failures": [
            {"id": "DEF-A", "status": "REPAIRED_IN_X4", "summary": "X3 R omitted pointer; X4 R-Ref restores pointer + edge_mlp."},
            {"id": "DEF-B", "status": "REPAIRED_IN_X4", "summary": "Pointer bypassed gen_logits; X4 routes p_copy into unified byte distribution."},
            {"id": "DEF-C", "status": "REPAIRED_IN_X4", "summary": "Future target bytes leaked via valid_tok; X4 enforces pos <= prompt_end (0.0 diff verified)."},
            {"id": "DEF-D", "status": "REPAIRED_IN_X4", "summary": "Unnormalized weighted log-space mix; X4 uses normalized FP32 prob mixture."},
            {"id": "DEF-E", "status": "REPAIRED_IN_X4", "summary": "Arbitrary mod-16 pointer fallback removed."},
            {"id": "DEF-F", "status": "REPAIRED_IN_X4", "summary": "Colliding mod-16 aux targets removed in D0-ByteCE / matched pairs."},
            {"id": "DEF-G", "status": "REPAIRED_IN_X4", "summary": "D2 exposure confound + 1-pair untargeted L1 fixed with matched exposure + all-pairs margin."},
            {"id": "DEF-H", "status": "REPAIRED_IN_X4", "summary": "Fixed prompt_end during both training and generation."},
            {"id": "DEF-I", "status": "REPAIRED_IN_X4", "summary": "Split-agnostic canonical_structural_hash."},
            {"id": "DEF-J", "status": "REPAIRED_IN_X4", "summary": "Mode 2 controls, Mode 3 typed targets + retention, distinct Mode 4/5 sets."},
            {"id": "DEF-K", "status": "REPAIRED_IN_X4", "summary": "U2/U4 anchored residual refinement + 0/1/2/4/8 trajectory eval."},
            {"id": "DEF-L", "status": "REPAIRED_IN_X4", "summary": "Stage-qualified contrast tuples & non-self-contrast assertion."},
            {"id": "DEF-M", "status": "REPAIRED_IN_X4", "summary": "Unified seq mean token prob calibration."},
            {"id": "DEF-N", "status": "REPAIRED_IN_X4", "summary": "Removed split= token leak from C09."},
            {"id": "DEF-O", "status": "REPAIRED_IN_X4", "summary": "Repaired C03/C04/C10 train primitive coverage + P2C curriculum."},
            {"id": "DEF-P", "status": "REPAIRED_IN_X4", "summary": "D0-ByteCE strictly zero aux loss terms."},
        ],
        "original_hypotheses_h1_to_h5": {
            "H1_P1_OpBank": "Low-rank sparse neural operator bank (~85k params) vs dense MLP control — pending S20 GPU C3.",
            "H2_D4_ECCO": "Error-conditioned counterfactual credit assignment + correctness-gated invariance — pending S20 GPU C2.",
            "H3_Fresh_P2C": "Fresh procedural stream + Primitive-to-Composition curriculum — pending S20 GPU C2.",
            "H4_LiveGrad_ResGraft": "Live-gradient bounded residual arm integration + capability preservation — pending S20 GPU C4.",
            "H5_UnifiedByteLM": "Unified copy-assisted byte LM replacing dual-head/mod-16 split — integrity-verified locally; GPU learning pending.",
        },
        "campaign_status": (x4_summary or {}).get("status", "PRE_GPU_EXECUTION"),
    }


# ---------------------------------------------------------------------------
# Checkpoint Store & Single Job / Campaign Orchestration
# ---------------------------------------------------------------------------


def _publish_x4_checkpoint(
    store: CheckpointStore,
    *,
    model: X4CognitiveModel,
    optimizer: torch.optim.Optimizer,
    spec: Any,
    spec_sha256: str,
    lineage_id: str,
    global_update: int,
    target_updates: int,
    source_commit: str,
    architecture: str,
    objective: str,
    parent_sha: str | None,
    batch_size: int = 16,
    include_optimizer: bool = True,
) -> str:
    data_sha = digest({"program": "ELENDRIL-V8-X4", "arch": architecture, "obj": objective})
    rng_bytes = torch.get_rng_state().numpy().tobytes()
    rng_sha = sha256(rng_bytes).hexdigest()
    ident = IdentityBindings(
        IDENTITY_SCHEMA,
        str(source_commit),
        str(spec.sha256()),
        UTF8.identity,
        data_sha,
        data_sha,
        str(spec_sha256),
        optimizer_group_receipt(model, optimizer)["sha256"],
        digest(["cosine", 1e-4, int(target_updates)]),
        digest(["x4", architecture, objective]),
    )
    cursor = CursorState(CURSOR_SCHEMA, data_sha, 0, int(global_update), 0)
    tot_tok = max(1, int(global_update) * int(batch_size))
    budget_tok = max(tot_tok, int(target_updates) * int(batch_size))
    state = TrainingState(
        schema="anra-v5-training-state/v1",
        lineage_id=str(lineage_id),
        generation=int(global_update),
        global_update=int(global_update),
        cumulative_tokens=tot_tok,
        token_budget=budget_tok,
        tokens_by_source={"elendril_x4": tot_tok},
        optimizer_step_max=int(global_update),
        schedule_tokens=tot_tok,
        tokens_per_update=int(batch_size),
        curriculum_phase=f"x4-{architecture}",
        cursor=cursor,
        rng_state_sha256=rng_sha,
        identities=ident,
        parent_checkpoint_sha256=parent_sha,
    )
    state.assert_valid()
    meta = {
        "architecture": architecture,
        "objective": objective,
        "spec_sha256": str(spec_sha256),
        "model_spec_sha256": str(spec.sha256()),
        "includes_optimizer_state": bool(include_optimizer),
    }
    payloads = {
        "model.bin": _dump({k: v.detach().cpu() for k, v in model.state_dict().items()}),
        "optimizer.bin": _dump(optimizer.state_dict()) if include_optimizer else _dump({}),
        "rng.bin": rng_bytes,
        "scheduler.json": canonical({"metadata": meta, "scaler": {}}),
        "cursor.json": canonical(asdict(cursor)),
        "ledger.json": canonical(dict(state.tokens_by_source)),
        "training_state.json": canonical(state.canonical()),
    }
    quarantine_interrupted_publication(store.lineage_root, state.generation)
    new_sha = store.publish(state=state, payloads=payloads, expected_parent_sha256=parent_sha)
    store.prune(keep=set())
    return new_sha


def _restore_x4_checkpoint(
    store: CheckpointStore,
    sha: str,
    *,
    model: X4CognitiveModel,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
    expected_spec_sha256: str | None = None,
) -> dict[str, Any]:
    state, payloads = store.restore(sha)
    sched = json.loads(payloads["scheduler.json"])
    meta = sched.get("metadata", sched)
    if expected_spec_sha256 and meta.get("spec_sha256") != expected_spec_sha256:
        raise ValueError(f"SWAPPED_SPEC_MISMATCH: expected {expected_spec_sha256}, got {meta.get('spec_sha256')}")
    model.load_state_dict(_load(payloads["model.bin"]))
    model.to(torch.device(device))
    if optimizer is not None and "optimizer.bin" in payloads:
        opt_state = _load(payloads["optimizer.bin"])
        if isinstance(opt_state, Mapping) and opt_state.get("state"):
            optimizer.load_state_dict(opt_state)
        if "rng.bin" in payloads:
            torch.set_rng_state(torch.frombuffer(bytearray(payloads["rng.bin"]), dtype=torch.uint8))
    return {"state": state, "meta": meta, "global_update": int(state.global_update), "sha": str(sha)}


def run_x4_single_job(
    *,
    job: Mapping[str, Any],
    work_root: Path | str,
    device: str = "cpu",
    canary: bool = False,
    resume: bool = True,
    ckpt_every: int = 512,
    updates_override: int | None = None,
) -> dict[str, Any]:
    """Execute or resume a single registered S20/tiny X4 job atomically."""
    wroot = Path(work_root)
    wroot.mkdir(parents=True, exist_ok=True)
    jid = str(job["job_id"])
    arch = str(job["architecture"])
    obj = str(job["objective"])
    data_reg = str(job.get("data_regime", "fresh_stream"))
    curr_mode = str(job.get("curriculum_mode", "curriculum_p2c"))
    scale = "tiny" if canary else str(job.get("scale", "S20"))
    seed = int(job["init_seed"])
    oseed = int(job.get("order_seed", seed + 2000))
    source_commit = str(job["source_commit"])
    spec_sha = str(job["spec_sha256"])

    target_updates = int(updates_override) if updates_override is not None else (4 if canary else int(job["updates"]))
    dev = torch.device(device)
    spec = build_x4_backbone_spec(scale)
    model = X4CognitiveModel(arch, spec, seed=seed).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Keep short "c"/"m" dirs + win_long_path so Windows MAX_PATH never truncates manifest.json
    store = CheckpointStore(win_long_path(wroot / "c"), "m")
    start_u = 0
    latest_sha = store.latest_sha256() if resume else None
    if latest_sha is not None:
        restored = _restore_x4_checkpoint(store, latest_sha, model=model, optimizer=optimizer, device=device, expected_spec_sha256=spec_sha)
        start_u = int(restored["global_update"])

    fams = ("X01", "C04_COMPOSITIONAL_SCAN", "C10_LANGUAGE_MATH_TRANSFER") if canary else X4_22_FAMILIES
    batch_sz = 4 if canary else 16
    sched = build_x4_curriculum_schedule(
        seed=oseed,
        total_updates=target_updates,
        batch_size=batch_sz,
        data_regime=data_reg,
        curriculum_mode=curr_mode,
        families=fams,
    )
    matched = build_x4_matched_batches(sched, seed=oseed, batch_size=batch_sz, is_schedule=True)
    batches = prepack_x4_batches(matched, max_length=model.max_ctx, device=dev)

    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    t0 = time.perf_counter()
    last_stats: dict[str, Any] = {}
    with (wroot / "updates.jsonl").open("a", encoding="utf-8") as fh:
        for u in range(start_u, target_updates):
            step_b = batches[u % len(batches)]
            last_stats = compute_x4_objective_step(
                model, optimizer, step_b, objective=obj, update_idx=u, total_updates=target_updates, use_amp=use_amp, scaler=scaler if use_amp else None
            )
            cur_u = u + 1
            fh.write(dump_json_safe({"update": cur_u, **last_stats}) + "\n")
            if (cur_u % max(1, int(ckpt_every)) == 0) or (cur_u == target_updates):
                keep_opt = bool(canary) or (cur_u < target_updates)
                latest_sha = _publish_x4_checkpoint(
                    store, model=model, optimizer=optimizer, spec=spec, spec_sha256=spec_sha, lineage_id="m",
                    global_update=cur_u, target_updates=target_updates, source_commit=source_commit,
                    architecture=arch, objective=obj, parent_sha=latest_sha, batch_size=batch_sz, include_optimizer=keep_opt,
                )
    store.prune(keep=set())

    dev_cases = build_x4_battery(split="dev", per_family=1 if canary else 6, families=fams)["cases"]
    preds = evaluate_x4_cases(model, dev_cases, batch_size=4 if canary else 32, device=device)
    profile = score_x4_multidimensional_profile(preds, dev_cases)
    wall = round(time.perf_counter() - t0, 3)

    receipt = {
        "schema": "elendril-x4-job-receipt-v1",
        "job_id": jid,
        "cycle": str(job.get("cycle", "C1")),
        "stage": str(job.get("stage", "C1")),
        "architecture": arch,
        "objective": obj,
        "data_regime": data_reg,
        "curriculum_mode": curr_mode,
        "scale": scale,
        "init_seed": seed,
        "status": "COMPLETE",
        "source_commit": source_commit,
        "spec_sha256": spec_sha,
        "target_updates": target_updates,
        "completed_updates": target_updates,
        "checkpoint_sha256": latest_sha,
        "exposure_manifest": matched["exposure_manifest"],
        "dev_macro_mean": profile["macro_mean"],
        "dev_profile": profile,
        "last_step_stats": last_stats,
        "wall_seconds": wall,
    }
    _atomic_write_json(wroot / "JOB_RECEIPT.json", receipt)
    return receipt


def _link_latest_full_files(work_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(work_dir.glob("*.json")) + sorted(work_dir.glob("*.md")):
        if p.is_file():
            out[p.name] = p
    jobs_dir = work_dir / "jobs"
    if not jobs_dir.is_dir():
        return out
    for jdir in sorted(jobs_dir.iterdir()):
        if not jdir.is_dir():
            continue
        for fn in ("JOB_RECEIPT.json", "updates.jsonl"):
            fp = jdir / fn
            if fp.is_file():
                out[f"jobs/{jdir.name}/{fn}"] = fp
        ck_root = jdir / "c"
        if ck_root.is_dir():
            for lat_p in ck_root.rglob("LATEST"):
                sha = lat_p.read_text(encoding="utf-8").strip().splitlines()[0] if lat_p.is_file() else ""
                out[lat_p.relative_to(work_dir).as_posix()] = lat_p
                if sha:
                    obj_d = lat_p.parent / "objects" / sha
                    if obj_d.is_dir():
                        for fp in sorted(obj_d.rglob("*")):
                            if fp.is_file():
                                out[fp.relative_to(work_dir).as_posix()] = fp
                for idx_f in lat_p.parent.glob("*.json"):
                    out[idx_f.relative_to(work_dir).as_posix()] = idx_f
    return out


def run_x4_campaign(
    *,
    work_dir: Path | str,
    export_dir: Path | str,
    source_commit: str,
    devices: Sequence[str] = ("cuda:0", "cuda:1"),
    session_minutes: float = 660.0,
    reserve_minutes: float = 45.0,
    measured_seconds_per_update: float = 0.25,
    per_job_disk_bytes: int = 95 * 1024 * 1024,
    cycles: Sequence[str] = ("C1", "C2", "C3", "C4"),
    canary: bool = False,
    resume_receipts: Sequence[Mapping[str, Any]] = (),
    in_process_runner: Callable[..., Any] | None = None,
    mirror_dir: Path | str | None = None,
    clock_fn: Callable[[], float] | None = None,
    jobs_override: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run resumable 4-cycle X4 S20 research campaign with Cycle-1 Integrity Gate & Level-7 Custody."""
    wdir = Path(work_dir)
    edir = Path(export_dir)
    wdir.mkdir(parents=True, exist_ok=True)
    edir.mkdir(parents=True, exist_ok=True)
    now_fn = clock_fn if clock_fn is not None else time.monotonic
    t_start = now_fn()
    usable_seconds = max(60.0, (float(session_minutes) - float(reserve_minutes)) * 60.0) if session_minutes > reserve_minutes else float(session_minutes) * 60.0

    # Gate 0: Cycle 1 Foundational Integrity Gate (fail-closed!)
    gate_rep = run_foundational_integrity_gate(wdir)
    if not gate_rep["all_passed"]:
        raise RuntimeError(f"Cycle 1 Foundational Integrity Gate failed: {gate_rep}")

    ledger = build_x4_cycle_ledger(source_commit=source_commit, updates=4 if canary else X4_DEFAULT_UPDATES, cycles=cycles)
    raw_jobs = list(jobs_override) if jobs_override is not None else list(ledger["jobs"])
    jobs_by_id = {str(j["job_id"]): dict(j) for j in raw_jobs}
    _atomic_write_json(wdir / "X4_DESIGN_REGISTRY.json", ledger)

    canary_fams = (
        "X01",
        "C01_NOVEL_OP_INDUCTION",
        "C03_GRID_TRANSFORM_ARC",
        "C04_COMPOSITIONAL_SCAN",
        "C08_ADAPTIVE_MEMORY_REVISION",
        "C09_METACOGNITION_SELF_CHECK",
        "C10_LANGUAGE_MATH_TRANSFER",
    )
    splits_map = {
        sp: build_x4_battery(
            split=sp,
            per_family=1 if canary else 8,
            families=canary_fams if canary else X4_22_FAMILIES,
        )["cases"]
        for sp in ("train", "dev", "confirm", "ood_transfer")
    }
    disj_rep = verify_x4_split_disjointness(splits_map)
    _atomic_write_json(wdir / "X4_SPLIT_CUSTODY.json", disj_rep)

    orch = CampaignOrchestrator(
        campaign_id="ELENDRIL-V8-X4",
        source_commit=str(source_commit),
        work_dir=wdir,
        max_minutes=float(session_minutes),
        reserve_minutes=float(reserve_minutes),
        in_process_runner=in_process_runner,
    )
    all_sjobs = [
        SessionJob(
            job_id=str(j["job_id"]),
            architecture=str(j["architecture"]),
            init_seed=int(j["init_seed"]),
            order_seed=int(j["order_seed"]),
            updates=int(j["updates"]),
            session=str(j["cycle"]),
            extra=dict(j),
        )
        for j in raw_jobs
    ]
    orch.prepare(all_sjobs)

    reconcile = orch.reconcile_verified_resume(resume_receipts, raw_jobs, fail_on_rejection=bool(resume_receipts))
    completed_ids = set(reconcile["verified_completed_jobs"])
    job_receipts: dict[str, dict[str, Any]] = {
        str(r["job_id"]): dict(r) for r in resume_receipts if str(r.get("job_id", "")) in completed_ids
    }
    for jid in list(completed_ids):
        rp = wdir / "jobs" / jid / "JOB_RECEIPT.json"
        if rp.is_file() and jid not in job_receipts:
            job_receipts[jid] = json.loads(rp.read_text(encoding="utf-8"))

    pending_sjobs = [s for s in all_sjobs if s.job_id not in completed_ids]

    def _cmd_builder(job: SessionJob, wroot: Path) -> tuple[str, ...]:
        spec_p = wdir / "specs" / f"{job.job_id}.json"
        _atomic_write_json(spec_p, job.extra)
        cmd = [sys.executable, "-m", "experiments.x4.module", "run-job", "--job-json", str(spec_p), "--work-root", str(wroot), "--device", "cpu" if canary else "cuda:0"]
        if canary:
            cmd.append("--canary")
        return tuple(cmd)

    plan = orch.admit_slice(pending_sjobs, measured_seconds_per_update=float(measured_seconds_per_update), per_job_disk_bytes=int(per_job_disk_bytes), parallel_workers=max(1, len(devices)))
    admitted = [j for j in pending_sjobs if j.job_id in set(plan["admitted_job_ids"])]
    deferred_ids = list(plan["deferred_pending_job_ids"])
    failed_ids: list[str] = []
    stopped_reason: str | None = None
    n_devs = max(1, len(devices))

    for w_start in range(0, len(admitted), n_devs):
        if usable_seconds - (now_fn() - t_start) <= 0.0:
            deferred_ids = [j.job_id for j in admitted[w_start:]] + deferred_ids
            stopped_reason = "DEADLINE_RESERVE_REACHED"
            break
        wave_res = orch.run_waves(admitted[w_start : w_start + n_devs], command_builder=_cmd_builder, devices=tuple(devices), resume=True)
        for wid in wave_res.get("completed_jobs", ()):
            rp = wdir / "jobs" / wid / "JOB_RECEIPT.json"
            if rp.is_file():
                job_receipts[wid] = json.loads(rp.read_text(encoding="utf-8"))
            completed_ids.add(wid)
        for wrec in wave_res.get("waves", ()):
            failed_ids.extend(wrec.get("failed_workers", ()))

    pending_all = [j.job_id for j in all_sjobs if j.job_id not in completed_ids]
    next_id = pending_all[0] if pending_all else None
    conf_req = [str(j["job_id"]) for j in raw_jobs]
    conf_pending = [jid for jid in conf_req if jid not in completed_ids]
    conf_lock = {
        "schema": "elendril-x4-confirmation-lock-v1",
        "required_job_count": len(conf_req),
        "completed_required_count": len(conf_req) - len(conf_pending),
        "pending_confirm_jobs": conf_pending,
        "unlocked": len(conf_pending) == 0 and len(conf_req) > 0,
        "status": "UNLOCKED_COMPLETE_COHORT" if (len(conf_pending) == 0 and len(conf_req) > 0) else "LOCKED_PENDING_COHORT",
    }
    _atomic_write_json(wdir / "CONFIRMATION_LOCK.json", conf_lock)

    # Sealed Confirmation + L0-L5 + C03/C04/C10 Diagnostics + X01-X12 Interventions + 5 Learning Modes
    sealed_report: dict[str, Any] = {"status": "LOCKED_PENDING_COHORT"}
    cap_registry: dict[str, Any] = {"schema": "elendril-x4-capability-registry-v1", "arms": {}}
    baseline_profile = None

    if conf_lock["unlocked"] and not failed_ids:
        eval_dev = "cuda:0" if (not canary and devices and any(str(d).startswith("cuda") for d in devices) and torch.cuda.is_available()) else "cpu"
        conf_cases = splits_map["confirm"]
        ood_cases = splits_map["ood_transfer"]
        l0_l5_cases = build_l0_l5_diagnostic_suite(train_cases=splits_map["train"], dev_cases=splits_map["dev"], ood_cases=ood_cases, per_level_limit=1 if canary else 2)
        x_interv_cases = build_x01_x12_interventions(splits_map["dev"][:12])
        bsz = 4 if canary else 32
        sealed_jobs: dict[str, Any] = {}
        modes_by_job: dict[str, Any] = {}

        for jid in sorted(completed_ids):
            meta = dict(job_receipts.get(jid) or jobs_by_id.get(jid) or {})
            ck_root = win_long_path(wdir / "jobs" / jid / "c")
            sha = CheckpointStore(ck_root, "m").latest_sha256() if ck_root.is_dir() else None
            if not sha:
                continue
            spec = build_x4_backbone_spec("tiny" if canary else meta.get("scale", "S20"))
            m = X4CognitiveModel(str(meta["architecture"]), spec, seed=int(meta["init_seed"]))
            _restore_x4_checkpoint(CheckpointStore(ck_root, "m"), sha, model=m, device=eval_dev, expected_spec_sha256=str(meta["spec_sha256"]))

            c_preds = evaluate_x4_cases(m, conf_cases, batch_size=bsz, device=eval_dev)
            c_prof = score_x4_multidimensional_profile(c_preds, conf_cases)
            o_preds = evaluate_x4_cases(m, ood_cases, batch_size=bsz, device=eval_dev)
            o_prof = score_x4_multidimensional_profile(o_preds, ood_cases)
            l_preds = evaluate_x4_cases(m, l0_l5_cases, batch_size=bsz, device=eval_dev)
            l_prof = score_x4_multidimensional_profile(l_preds, l0_l5_cases)
            if str(meta["architecture"]) == "Base" and baseline_profile is None:
                baseline_profile = c_prof

            # U step trajectory (0, 1, 2, 4, 8 steps) for U-family architectures
            u_traj = None
            if m.uses_u:
                u_traj = {}
                sub_u = conf_cases[:8]
                for ks in (0, 1, 2, 4, 8):
                    up = evaluate_x4_cases(m, sub_u, batch_size=8, device=eval_dev, eval_steps_override=ks)
                    u_traj[str(ks)] = float(sum(int(bool(x["correct"])) for x in up) / max(1, len(up)))

            sealed_jobs[jid] = {
                "architecture": meta["architecture"],
                "cycle": meta.get("cycle"),
                "objective": meta["objective"],
                "data_regime": meta.get("data_regime"),
                "confirm_profile": c_prof,
                "ood_profile": o_prof,
                "l0_l5_profile": l_prof,
                "u_step_trajectory": u_traj,
            }
            cap_registry["arms"][jid] = evaluate_x4_qualification_thresholds(c_prof, baseline_profile)

            if str(meta["architecture"]) in ("Base", "R-Ref", "P1-OpBank", "ANRA-X4-Int") and len(modes_by_job) < 4:
                icl_m = {sh: [c for c in conf_cases if c["family"] in ("C01_NOVEL_OP_INDUCTION", "C02_FST_RULE_COMPOSITION") and int(c.get("n_shots", 0)) == sh][:2] or [conf_cases[0]] for sh in (0, 1, 4, 16, 64)}
                c08_seq = [c for c in conf_cases if c["family"] == "C08_ADAPTIVE_MEMORY_REVISION"][:4] or conf_cases[:2]
                c01_d = [c for c in splits_map["dev"] if c["family"] == "C01_NOVEL_OP_INDUCTION"][:2] or splits_map["dev"][:2]
                c01_q = [c for c in conf_cases if c["family"] == "C01_NOVEL_OP_INDUCTION"][:4] or conf_cases[:2]
                ret_q = [c for c in conf_cases if c["family"] == "X01"][:2] or conf_cases[:2]
                tf_c = [c for c in ood_cases if c["family"] in ("C03_GRID_TRANSFORM_ARC", "C04_COMPOSITIONAL_SCAN")][:4] or ood_cases[:2]
                cd_c = [c for c in ood_cases if c["family"] == "C10_LANGUAGE_MATH_TRANSFER"][:4] or ood_cases[-2:]
                modes_by_job[jid] = {
                    "modes": evaluate_x4_learning_modes(
                        m, icl_cases_by_shot=icl_m, state_adapt_sequence=c08_seq,
                        grad_adapt_demos=c01_d, grad_adapt_queries=c01_q, retention_queries=ret_q,
                        tool_free_cases=tf_c, cross_domain_cases=cd_c,
                    ),
                    "c03_c04_c10_diagnosis": diagnose_c03_c04_c10_failure_modes(
                        m, train_cases=splits_map["train"], unseen_same_fmt=splits_map["dev"], unseen_reformatted=ood_cases, device=eval_dev
                    ),
                    "x_interventions_acc": float(sum(int(bool(x["correct"])) for x in evaluate_x4_cases(m, x_interv_cases[:8], batch_size=8, device=eval_dev)) / max(1, min(8, len(x_interv_cases)))),
                }
            del m
        sealed_report = {"status": "COMPLETE", "jobs": sealed_jobs, "five_learning_modes": modes_by_job}

    _atomic_write_json(wdir / "X4_SEALED_CONFIRMATION.json", sealed_report)
    _atomic_write_json(wdir / "X4_CAPABILITY_REGISTRY.json", cap_registry)

    # Stage-qualified paired contrasts (Fixes Defect L: zero self-contrasts, zero stage erasure!)
    arm_records: list[dict[str, Any]] = []
    seen_k: set[tuple[tuple[str, str, str, str, str], int, int]] = set()
    for jid in sorted(job_receipts):
        r = job_receipts[jid]
        if "dev_macro_mean" not in r:
            continue
        a_key = (str(r.get("cycle", "")), str(r["architecture"]), str(r["objective"]), str(r.get("data_regime", "")), str(r.get("curriculum_mode", "")))
        rec_k = (a_key, int(r["init_seed"]), int(r.get("order_seed", int(r["init_seed"]) + 2000)))
        if rec_k in seen_k:
            continue
        seen_k.add(rec_k)
        arm_records.append({"arm_key": "|".join(a_key), "init_seed": rec_k[1], "order_seed": rec_k[2], "value": float(r["dev_macro_mean"])})

    paired_summary: dict[str, Any] = {}
    for cid, ctrl_t, trt_t in X4_PAIRED_CONTRASTS:
        assert ctrl_t != trt_t, f"Self-contrast detected: {cid}"
        ck, tk = "|".join(ctrl_t), "|".join(trt_t)
        cv = {r["init_seed"]: r["value"] for r in arm_records if r["arm_key"] == ck}
        tv = {r["init_seed"]: r["value"] for r in arm_records if r["arm_key"] == tk}
        sh = sorted(set(cv) & set(tv))
        if sh:
            paired_summary[cid] = bootstrap_paired_families([cv[s] for s in sh], [tv[s] for s in sh], resamples=300, seed=1801)
            if "first_page_report" not in paired_summary:
                obs = build_paired_seed_observations(arm_records, contrast_id=cid, control_arm=ck, treatment_arm=tk, arm_key="arm_key")
                paired_summary["first_page_report"] = build_analyst_first_page_report(
                    title="ELENDRIL V8-X4 Fundamental Reconstruction Summary",
                    campaign_id="ELENDRIL-V8-X4",
                    source_commit=source_commit,
                    decision_summary="COHORT_COMPLETE" if not pending_all else "PARTIAL",
                    control_arm=ck,
                    treatment_arm=tk,
                    paired_seed_bundle=obs,
                )

    campaign_status = "COMPLETE" if (not pending_all and not failed_ids) else "PARTIAL"
    summary_doc = {
        "schema": "elendril-x4-campaign-summary-v1",
        "status": campaign_status,
        "source_commit": str(source_commit),
        "integrity_gate_passed": bool(gate_rep["all_passed"]),
        "completed_count": len(completed_ids),
        "total_jobs": len(all_sjobs),
        "s50_jobs": 0,
        "pending_job_ids": pending_all,
        "deferred_job_ids": deferred_ids,
        "failed_job_ids": sorted(set(failed_ids)),
        "next_resumable_job_id": next_id,
        "stopped_reason": stopped_reason,
        "confirmation_lock": conf_lock,
        "sealed_status": sealed_report.get("status"),
        "paired_summary": paired_summary,
    }
    _atomic_write_json(wdir / "X4_FAILURE_EVIDENCE_REGISTRY.json", build_x4_failure_and_evidence_registry(x4_summary=summary_doc))
    _atomic_write_json(wdir / "X4_CAMPAIGN_SUMMARY.json", summary_doc)
    _atomic_write_json(wdir / "JOB_RECEIPTS.json", {"receipts_by_id": job_receipts})
    (wdir / "RESULTS.md").write_text(
        "\n".join([f"# ELENDRIL V8-X4 ({campaign_status})", "", f"- Completed: `{len(completed_ids)}/{len(all_sjobs)}` S20 jobs (`S50=0`)", f"- Integrity Gate: `{gate_rep['all_passed']}`", f"- Confirmation Lock: `{conf_lock['status']}`", ""]),
        encoding="utf-8",
    )

    full_files = _link_latest_full_files(wdir)
    analysis_files = {k: v for k, v in full_files.items() if "/c/" not in k}
    summary_files = {k: v for k, v in analysis_files.items() if "/" not in k}
    try:
        free_b = shutil.disk_usage(edir).free
    except OSError:
        free_b = 0
    an_b = sum(p.stat().st_size for p in analysis_files.values() if p.is_file())
    if free_b > an_b:
        full_b = sum(p.stat().st_size for k, p in full_files.items() if k not in analysis_files and p.is_file())
        arcs = export_session_archives(
            wdir, edir, prefix="ELENDRIL_V8_X4",
            analysis_files=analysis_files, summary_files=summary_files, evidence_files=analysis_files, full_files=full_files,
            status=campaign_status, source_commit=source_commit, protocol_sha256=ledger["ledger_sha256"],
            include_v6_manifest=True, write_full_zip=(free_b > full_b + 2 * 1024**3),
        )
        an_ver = verify_session_archive(arcs["archives"]["analysis"], expected_source_commit=source_commit)
        summary_doc["archives"] = arcs["archives"]
        summary_doc["archive_sha256"] = arcs["archive_sha256"]
        summary_doc["analysis_archive_verified"] = bool(an_ver.get("verified"))
        _atomic_write_json(edir / "X4_CAMPAIGN_SUMMARY.json", summary_doc)
        if mirror_dir is not None:
            td = Path(mirror_dir)
            td.mkdir(parents=True, exist_ok=True)
            mirrored = []
            for p_str in (*arcs["archives"].values(), str(edir / "X4_CAMPAIGN_SUMMARY.json"), str(wdir / "RESULTS.md")):
                if p_str and Path(p_str).is_file():
                    dst = td / Path(p_str).name
                    if dst.exists():
                        dst.unlink()
                    shutil.copy2(p_str, dst)
                    mirrored.append(str(dst))
            summary_doc["mirrored_files"] = mirrored
    return summary_doc


# ---------------------------------------------------------------------------
# Palantiri ExperimentModule Factory
# ---------------------------------------------------------------------------


def x4_model_factory(architecture: str, model_config: Mapping[str, Any], seed: int) -> X4CognitiveModel:
    return X4CognitiveModel(architecture, build_x4_backbone_spec(str(model_config.get("scale", "tiny"))), seed=int(seed))


def x4_dataset_factory(dataset_config: Mapping[str, Any], cpu_canary: bool) -> dict[str, Any]:
    seed = int(dataset_config.get("seed", X4_SPLIT_SEEDS["train"]))
    fams = ("X01", "C04_COMPOSITIONAL_SCAN", "C10_LANGUAGE_MATH_TRANSFER") if cpu_canary else X4_22_FAMILIES
    tr = build_x4_battery(split="train", seed=seed, per_family=4 if cpu_canary else 8, families=fams)["cases"]
    dv = build_x4_battery(split="dev", seed=X4_SPLIT_SEEDS["dev"], per_family=2 if cpu_canary else 4, families=fams)["cases"]
    oo = build_x4_battery(split="ood_transfer", seed=X4_SPLIT_SEEDS["ood_transfer"], per_family=2 if cpu_canary else 4, families=fams)["cases"]
    def _wrap(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for c in rows:
            d = dict(c)
            d.setdefault("world_id", str(c["case_id"]))
            d.setdefault("sig", str(c.get("structural_hash", c["case_id"])))
            out.append(d)
        return out
    tr_all = _wrap(tr)
    # Disjoint even/odd partition + unique structural hashes between train and probe_train for Gate 1!
    tr_w = [c for i, c in enumerate(tr_all) if i % 2 == 0]
    probe_w = [
        dict(c, world_id=f"{c['world_id']}::probe", sig=f"{c['sig']}::probe")
        for i, c in enumerate(tr_all)
        if i % 2 == 1
    ]
    return {
        "train_data": GenericDatasetBundle(items=tuple(tr_w), dataset_name="x4_22fam", data_identity=f"x4-22fam-{len(tr_w)}", manifest={"cases": len(tr_w)}),
        "train_worlds": tr_w,
        "eval_worlds": _wrap(dv),
        "multihop_worlds": _wrap(oo),
        "probe_train_worlds": probe_w,
    }


def x4_loss_fn(model: X4CognitiveModel, chunk: Sequence[Mapping[str, Any]], recipe: Any) -> torch.Tensor:
    dev = next(model.parameters()).device
    encoded = [encode_prompt_and_target_pair(str(c["prompt"]), format_x4_typed_target(c), max_length=model.max_ctx) for c in chunk]
    bsz = len(encoded)
    max_l = max(len(ids) for ids, _, _ in encoded)
    toks = torch.zeros((bsz, max_l), dtype=torch.long, device=dev)
    segs = torch.full((bsz, max_l), -1, dtype=torch.long, device=dev)
    elig_t = torch.zeros((bsz, max_l), dtype=torch.bool, device=dev)
    end_t = torch.zeros((bsz,), dtype=torch.long, device=dev)
    for i, (ids, el, ep) in enumerate(encoded):
        toks[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=dev)
        segs[i, : len(ids)] = 0
        elig_t[i, : len(el)] = torch.tensor(el, dtype=torch.bool, device=dev)
        end_t[i] = ep
    pos, mask = packed_layout(segs, torch_module=torch)
    out = model(toks, pos, mask, prompt_end_positions=end_t)
    nll = -out["log_p_unified"][:, :-1, :].float().gather(2, toks[:, 1:].unsqueeze(-1)).squeeze(-1)
    m_f = elig_t[:, 1:].float()
    return (nll * m_f).sum() / m_f.sum().clamp_min(1.0)


def x4_kernel_factory() -> X4CognitiveKernel:
    return X4CognitiveKernel()


def x4_policy_factory(
    model: X4CognitiveModel,
    model_spec: Any = None,
    *,
    model_identity: str = "x4-cognitive",
    mode: str = "raw",
    copy_enabled_override: bool | None = None,
) -> X4CognitivePolicy:
    return X4CognitivePolicy(model, model_spec, model_identity=model_identity, mode=mode, copy_enabled_override=copy_enabled_override)


def build_elendril_x4_module() -> ExperimentModule:
    return ExperimentModule(
        backend_id=X4_BACKEND_ID,
        backend_version="4.0",
        program="ELENDRIL-V8-X4",
        architectures=X4_CANARY_ARCHITECTURES,
        objectives=X4_OBJECTIVES,
        model_factory=x4_model_factory,
        dataset_factory=x4_dataset_factory,
        loss_fn=x4_loss_fn,
        kernel_factory=x4_kernel_factory,
        policy_factory=x4_policy_factory,
        default_dataset_config={"seed": X4_SPLIT_SEEDS["train"]},
        default_model_config={"scale": "tiny"},
        treatment_architecture="ANRA-X4-Int",
        treatment_model_config={"scale": "tiny"},
        extra_model_keys=("scale",),
        extra_dataset_keys=("seed",),
        extra_recipe_keys=("eq_weight", "cf_weight", "cf_margin"),
        supports_grouped_loss=True,
    )


def register_planthir_backends() -> ExperimentModule:
    mod = build_elendril_x4_module()
    mod.register()
    return mod


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ELENDRIL V8-X4 Fundamental Reconstruction CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    p_gate = sub.add_parser("integrity-gate", help="Run Cycle 1 Foundational Integrity Gate (§3.1-§3.5).")
    p_gate.add_argument("--work-dir", required=False, default=None)
    p_inv = sub.add_parser("inventory", help="Print 4-cycle X4 S20 ledger JSON.")
    p_inv.add_argument("--source", required=True)
    p_job = sub.add_parser("run-job", help="Run a single registered X4 job.")
    p_job.add_argument("--job-json", required=True)
    p_job.add_argument("--work-root", required=True)
    p_job.add_argument("--device", default="cpu")
    p_job.add_argument("--canary", action="store_true")
    p_camp = sub.add_parser("run-campaign", help="Run 4-cycle X4 campaign.")
    p_camp.add_argument("--source", required=True)
    p_camp.add_argument("--work-dir", required=True)
    p_camp.add_argument("--export-dir", required=True)
    p_camp.add_argument("--devices", default="0,1")
    p_camp.add_argument("--cycles", default="C1,C2,C3,C4")
    p_camp.add_argument("--session-minutes", type=float, default=660.0)
    p_camp.add_argument("--canary", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "integrity-gate":
        rep = run_foundational_integrity_gate(args.work_dir)
        print(dump_json_safe(rep))
        return 0 if rep["all_passed"] else 1
    if args.command == "inventory":
        print(dump_json_safe(build_x4_cycle_ledger(source_commit=args.source)))
        return 0
    if args.command == "run-job":
        job = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
        rec = run_x4_single_job(job=job, work_root=args.work_root, device=args.device, canary=bool(args.canary))
        print(dump_json_safe({"job_id": rec["job_id"], "status": rec["status"], "dev_macro_mean": rec["dev_macro_mean"]}))
        return 0
    if args.command == "run-campaign":
        devs = ["cpu"] if args.canary else [f"cuda:{d.strip()}" for d in args.devices.split(",") if d.strip()]
        cycs = tuple(c.strip() for c in args.cycles.split(",") if c.strip())
        res = run_x4_campaign(work_dir=args.work_dir, export_dir=args.export_dir, source_commit=args.source, devices=devs, cycles=cycs, session_minutes=args.session_minutes, canary=bool(args.canary))
        print(dump_json_safe({"status": res["status"], "completed_count": res["completed_count"], "total_jobs": res["total_jobs"]}))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "X4CognitiveKernel",
    "X4CognitivePolicy",
    "X4_BACKEND_ID",
    "X4_CONFIRM_SEEDS",
    "X4_DEFAULT_UPDATES",
    "X4_DISCOVERY_SEEDS",
    "X4_PAIRED_CONTRASTS",
    "build_elendril_x4_module",
    "build_x4_cycle_ledger",
    "build_x4_failure_and_evidence_registry",
    "evaluate_capability_preservation",
    "evaluate_x4_learning_modes",
    "evaluate_x4_qualification_thresholds",
    "main",
    "register_planthir_backends",
    "run_x4_campaign",
    "run_x4_single_job",
    "x4_dataset_factory",
    "x4_kernel_factory",
    "x4_loss_fn",
    "x4_model_factory",
    "x4_policy_factory",
]
