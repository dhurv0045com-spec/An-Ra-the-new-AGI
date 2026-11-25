"""V8-X3 Palantiri Plugin, Batched Evaluator, 57-Job DAG Ledger, Confirmation Lock, & 2xT4 Campaign (§2-§15)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
import torch.nn as nn
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
    WorkerJobSpec,
    bootstrap_paired_families,
    build_analyst_first_page_report,
    build_paired_seed_observations,
    export_session_archives,
    verify_session_archive,
)
from v5_model.core import packed_layout
from v5_training.checkpoint import CheckpointStore
from v5_training.optimizer import optimizer_group_receipt
from v5_training.state import CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState, IdentityBindings, TrainingState

from .cognitive_battery import (
    CONFIRM_COMMITMENT_SEED,
    DEV_BANK_SEED,
    MANDATORY_22_FAMILIES,
    SCAN_VALID_ACTIONS,
    SPLIT_SEEDS,
    STRUCTURAL_GROUP_QUOTAS,
    TRAIN_BANK_SEED,
    TRANSFER_BANK_SEED,
    VALID_TYPED_OPS,
    X3_COGNITIVE_FAMILIES,
    _apply_grid_op,
    _eval_listops_node,
    _eval_scan_cmd,
    _grid_to_str,
    _solve_c06_horn,
    _solve_c08,
    audit_legacy_class_collisions,
    build_c_structural_group_suite,
    build_x3_cognitive_battery,
    canonical_structural_hash,
    compute_public_version_space_posterior,
    eval_x3_battery_shortcuts,
    format_public_typed_target,
    infer_family_from_public_prompt,
    infer_public_generation_bound,
    normalize_c10_rational_unit,
    parse_and_score_public_typed_output,
    score_cognitive_battery_profile,
    solve_cognitive_case_oracle,
    verify_c09_budget_reversal_oracle,
    verify_x3_destructive_masks,
    verify_x3_split_disjointness,
)
from .d3_objectives import (
    D3_OBJECTIVES,
    apply_pcgrad_conflict_surgery,
    audit_d1h_d2_schedule_and_preceding_observations,
    audit_x1_sampler_exposure,
    build_d3_16world_pack_minibatches,
    build_d3_hybrid_batches,
    compute_d3_objective_step,
    d3_warmup_ramp_weights,
    prepack_d3_batches,
    verify_fixed_weight_4microbatch_ce_equivalence,
)
from .factorial import independent_walk_x3_graph
from .models import (
    S50_MODEL_SPEC,
    TINY_CANARY_MODEL_SPEC,
    X3CognitiveModel,
    X3_ARCHITECTURES,
    X3_CONTROL_PAIRS,
    X3_CORE_ARCHITECTURES,
    X3_EXTENDED_ARCHITECTURES,
    X3_VERSIONED_CONTROL_PAIRS,
    build_legacy_x1_scaled_model,
    build_x3_backbone_spec,
    encode_prompt_and_target_pair,
    evaluate_five_learning_modes,
)
from .verify import (
    RUN3_DEFAULT_NEXT_JOB,
    build_r0_r7_program_ledger,
    reconcile_r0_run3_obligations,
    run_four_five_hop_interior_edge_swap,
    run_frozen_u_and_evidence_discrimination,
    summarize_paired_arm_statistics,
)

X3_BACKEND_ID = "elendril_x3"
X3_DEFAULT_UPDATES = 1024
X3_SEEDS: tuple[int, ...] = (7, 19, 41)
X3_CANARY_ARCHITECTURES: tuple[str, ...] = (
    "P-R-Iter-Fixed",
    "P-R-Iter",
    "P-R-Iter-NoRel",
    "P-R-Iter-AddMem",
    "P-R-Iter-NoCopy",
    "Base",
    "P",
    "R",
    "R-noedge",
)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _world_prompt_and_ideal(world: Mapping[str, Any]) -> tuple[str, str, str | None]:
    prompt = str(world.get("prompt", world.get("rendered_text", "")))
    gold_txt = str(world.get("gold_text", world.get("expected_answer", "")))
    if gold_txt in ("QUERY", "RETRIEVE", "ABSTAIN"):
        return prompt, gold_txt, None
    g_cls = int(world.get("gold_class", 0)) % 16
    return prompt, "ANSWER", SYMBOLS[g_cls]


class X3CognitiveKernel:
    """Truth-firewalled stateful kernel for Planthir Launch Gates, Canary, & Replay."""

    kernel_id = "elendril-x3-cognitive-kernel-v1"
    max_actions = 4

    def structural_signature(self, w: Mapping[str, Any]) -> str:
        return str(w.get("canon_sig", w.get("world_id", "x3")))

    def inject_distractor(self, w: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(w)
        p = str(w.get("prompt", w.get("rendered_text", "")))
        suffix = " [note:0]"
        if len((p + suffix).encode("utf-8")) <= 600:
            out["prompt"] = p + suffix
        out["world_id"] = str(w.get("world_id", "w")) + "::dist"
        out["canon_sig"] = str(w.get("canon_sig", "c")) + "::dist"
        return out

    def permute_world(self, w: Mapping[str, Any], shift: int = 1) -> dict[str, Any]:
        out = dict(w)
        out["world_id"] = str(w.get("world_id", "w")) + f"::p{shift}"
        out["canon_sig"] = str(w.get("canon_sig", "c")) + f"::p{shift}"
        return out

    def reverse_world(self, w: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(w)
        out["world_id"] = str(w.get("world_id", "w")) + "::rev"
        out["canon_sig"] = str(w.get("canon_sig", "c")) + "::rev"
        return out

    def decompose_subgoals(self, w: Mapping[str, Any]) -> tuple[Any, ...]:
        return ()

    def initial_state(self, w: Mapping[str, Any]) -> dict[str, Any]:
        can_query = str(w.get("uncertainty_kind", "")) == "information_uncertainty"
        return {
            "w": w,
            "step": 0,
            "queried": False,
            "can_query": can_query,
            "terminal": False,
        }

    def observe(self, st: Mapping[str, Any]) -> UniversalObservation:
        w = st["w"]
        prompt = str(w.get("prompt", w.get("rendered_text", "")))
        if st.get("queried"):
            prompt = f"{prompt} [OBS:RESOLVED]"
        q_avail = bool(st.get("can_query") and not st.get("queried"))
        consumed = frozenset({"query"}) if st.get("queried") else frozenset()
        # Truth Firewall: NEVER put gold/expected/hidden/truth/ground_truth keys in metadata!
        return UniversalObservation(
            obs_id=f"{w.get('case_id', w.get('world_id', 'x3'))}:s{st['step']}:q{int(bool(st.get('queried')))}",
            rendered_text=prompt,
            available_tools={"query": q_avail, "retrieve": False, "memory": False},
            consumed_tools=consumed,
            decision_index=int(st["step"]),
            budget_remaining=max(1, self.max_actions - int(st["step"])),
            metadata={
                "case_id": str(w.get("case_id", w.get("world_id", ""))),
                "family": str(w.get("family", "")),
                "split": str(w.get("split", "")),
                "step": int(st["step"]),
            },
        )

    def ideal_action(self, st: Mapping[str, Any]) -> tuple[str, str | None]:
        w = st["w"]
        _, op, arg = _world_prompt_and_ideal(w)
        if st.get("queried") and op == "QUERY":
            g_cls = int(w.get("gold_class", 0)) % 16
            return "ANSWER", SYMBOLS[g_cls]
        return op, arg

    def step(self, st: Mapping[str, Any], op: str, arg: str | None) -> dict[str, Any]:
        nxt_step = int(st["step"]) + 1
        if op in ("QUERY", "RETRIEVE") and not st.get("terminal") and nxt_step < self.max_actions:
            return {
                "w": st["w"],
                "step": nxt_step,
                "queried": True if op == "QUERY" else bool(st.get("queried")),
                "can_query": False if op == "QUERY" else bool(st.get("can_query")),
                "terminal": False,
            }
        return {
            "w": st["w"],
            "step": nxt_step,
            "queried": bool(st.get("queried")),
            "can_query": False,
            "terminal": True,
        }

    def verify(self, w: Mapping[str, Any], actions: Sequence[tuple[str, str | None]]) -> dict[str, Any]:
        if not actions:
            return {"correct": False, "reason": "NO_ACTIONS"}
        _, exp_op, exp_arg = _world_prompt_and_ideal(w)
        last_op, last_arg = actions[-1]
        if exp_op == "QUERY" and any(op == "QUERY" for op, _ in actions):
            ok = True
        else:
            ok = (last_op == exp_op) and (last_arg == exp_arg)
        return {"correct": bool(ok), "actions_count": len(actions)}


def _encode_obs_prompt(prompt: str, max_ctx: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = [UTF8.bos_id, *UTF8.encode(prompt), UTF8.special(SpecialToken.ANSWER)]
    if len(ids) > max_ctx:
        ids = ids[-max_ctx:]
    toks = torch.tensor([ids], dtype=torch.long, device=device)
    segs = torch.zeros_like(toks)
    pos, mask = packed_layout(segs, torch_module=torch)
    apos = torch.tensor([len(ids) - 1], dtype=torch.long, device=device)
    return toks, pos, mask, apos


def _action_pair_from_idx(idx: int) -> tuple[str, str | None]:
    lbl = str(ACTION_19_LABELS[int(idx)])
    if lbl in ("QUERY", "RETRIEVE", "ABSTAIN"):
        return lbl, None
    return "ANSWER", lbl


def _label_str(op: str, arg: str | None) -> str:
    return f"{op}:{arg}" if arg is not None else op


class X3CognitivePolicy:
    """Planthir-compatible cognitive policy supporting raw, masked, and batched evaluation."""

    def __init__(
        self,
        model: X3CognitiveModel,
        model_spec: Any = None,
        *,
        model_identity: str = "x3-cognitive",
        mode: str = "raw",
        copy_enabled_override: bool | None = None,
    ) -> None:
        self.model = model
        self.model_spec = model_spec
        self.model_identity = str(model_identity)
        self.mode = str(mode)
        self.copy_enabled_override = copy_enabled_override
        self.policy_id = f"x3-policy:{getattr(model, 'architecture', 'unknown')}:{self.mode}"

    def reset_episode(self) -> None:
        self.model.reset_episode()

    @torch.no_grad()
    def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction:
        dev = next(self.model.parameters()).device
        toks, pos, mask, apos = _encode_obs_prompt(obs.rendered_text, self.model.max_ctx, dev)
        was_train = self.model.training
        self.model.eval()
        if self.copy_enabled_override is False:
            with self.model.ablate_mechanism("disable_pointer"):
                out = self.model(toks, pos, mask, action_positions=apos)
        else:
            out = self.model(toks, pos, mask, action_positions=apos)
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
            logits_19[18] = -1e9  # ABSTAIN masked in strict tool-availability eval unless needed
        probs = F.softmax(logits_19, dim=-1)
        idx = int(torch.argmax(probs).item())
        op, arg = _action_pair_from_idx(idx)
        lp = float(torch.log(probs[idx].clamp_min(1e-7)).item())
        dist = {
            _label_str(*_action_pair_from_idx(i)): float(probs[i].item())
            for i in range(len(ACTION_19_LABELS))
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
            telemetry={
                "epistemic_uncertainty": float(unc[0]),
                "info_uncertainty": float(unc[1]),
                "ponder_cost": float(out["ponder_cost"][0].item()),
            },
        )

    @torch.no_grad()
    def extract_representations(self, obs_batch: Sequence[UniversalObservation]) -> dict[str, torch.Tensor]:
        if not obs_batch:
            d = int(self.model.d_model)
            return {"decision_hidden": torch.zeros((0, d))}
        dev = next(self.model.parameters()).device
        seqs: list[list[int]] = []
        for obs in obs_batch:
            ids = [UTF8.bos_id, *UTF8.encode(obs.rendered_text), UTF8.special(SpecialToken.ANSWER)]
            if len(ids) > self.model.max_ctx:
                ids = ids[-self.model.max_ctx :]
            seqs.append(ids)
        max_l = max(len(s) for s in seqs)
        toks = torch.zeros((len(seqs), max_l), dtype=torch.long, device=dev)
        segs = torch.full((len(seqs), max_l), -1, dtype=torch.long, device=dev)
        apos = torch.zeros((len(seqs),), dtype=torch.long, device=dev)
        for i, ids in enumerate(seqs):
            toks[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=dev)
            segs[i, : len(ids)] = 0
            apos[i] = len(ids) - 1
        pos, mask = packed_layout(segs, torch_module=torch)
        was_train = self.model.training
        self.model.eval()
        out = self.model(toks, pos, mask, action_positions=apos)
        if was_train:
            self.model.train()
        return {"decision_hidden": out["decision_hidden"].detach().cpu()}


class PublicOracleTypedDecoder(nn.Module):
    """Artificially perfect public-input decoder reconstructing answers strictly from public prompt bytes (§1, §4)."""

    def __init__(self, *, max_ctx: int = 640) -> None:
        super().__init__()
        self.max_ctx = int(max_ctx)
        self._dummy = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _solve_from_public_prompt(prompt: str) -> str:
        import re as _re
        from fractions import Fraction as _Fraction
        import numpy as _np

        p = str(prompt).strip()
        fam = infer_family_from_public_prompt(p)
        if fam == "C01_NOVEL_OP_INDUCTION":
            vs = compute_public_version_space_posterior(p)
            if vs["is_ambiguous"] or len(vs["unique_target_values"]) != 1:
                return "ABSTAIN:AMBIGUOUS"
            return f"ANSWER:{vs['unique_target_values'][0]}"

        if fam == "C02_FST_RULE_COMPOSITION":
            vs = compute_public_version_space_posterior(p)
            if vs["unique_target_values"]:
                return f"ANSWER:{vs['unique_target_values'][0]}"
            return "ABSTAIN:AMBIGUOUS"

        if fam == "C03_GRID_TRANSFORM_ARC":
            parts = p.split("|")
            demos = []
            for seg in parts[2:-1]:
                if "->" in seg:
                    a_s, b_s = seg.strip().split("->", 1)
                    ga = _np.array([[int(c) for c in r] for r in a_s.strip().split("/")], dtype=_np.int64)
                    gb = _np.array([[int(c) for c in r] for r in b_s.strip().split("/")], dtype=_np.int64)
                    demos.append((ga, gb))
            q_s = parts[-1].strip().split(":", 1)[1].split("->", 1)[0].strip()
            q_in = _np.array([[int(c) for c in r] for r in q_s.split("/")], dtype=_np.int64)
            id_cmap = {i: i for i in range(10)}
            for op_name in ("flip_h", "flip_v", "rot90", "transpose", "gravity_down", "border_one", "tile_xor_rot180"):
                cmap: dict[int, int] = {}
                ok = True
                for ga, gb in demos:
                    raw_out = _apply_grid_op(ga, op_name, id_cmap)
                    if raw_out.shape != gb.shape:
                        ok = False
                        break
                    for src_v, dst_v in zip(raw_out.reshape(-1).tolist(), gb.reshape(-1).tolist()):
                        if src_v in cmap and cmap[src_v] != dst_v:
                            ok = False
                            break
                        cmap[src_v] = dst_v
                    if not ok:
                        break
                if ok and demos:
                    q_out = _apply_grid_op(q_in, op_name, cmap)
                    return f"ANSWER:{_grid_to_str(q_out)}"
            return f"ANSWER:{_grid_to_str(q_in)}"

        if fam == "C04_COMPOSITIONAL_SCAN":
            cmd = p.split("|")[2]
            return f"ANSWER:{_eval_scan_cmd(cmd)}"

        if fam == "C05_RELATIONAL_GRAPH_CF":
            edges: list[dict[str, Any]] = []
            if p.startswith("CONF-TRACE "):
                m = _re.search(r"from=([^\s]+)\s+depth=(\d+)\s+ep>=(\d+)", p)
                assert m is not None
                goal, hops, min_ep = m.group(1), int(m.group(2)), int(m.group(3))
                body = p.split("\n", 1)[1] if "\n" in p else ""
                for seg in body.split("|"):
                    em = _re.search(r"([^\s=]+)=>([^\s(]+)\(e(\d+),a(\d+)\)", seg.strip())
                    if em:
                        edges.append({"subj": em.group(1), "val": em.group(2), "epoch": int(em.group(3)), "authority": int(em.group(4))})
            elif p.startswith("Trace "):
                m = _re.search(r"Trace\s+(\d+)\s+steps\s+from\s+([^\s]+)\s+\(ep>=(\d+)", p)
                assert m is not None
                hops, goal, min_ep = int(m.group(1)), m.group(2), int(m.group(3))
                for line in p.splitlines()[1:]:
                    em = _re.search(r"ep\s+(\d+):\s+node\s+([^\s]+)\s+links\s+to\s+([^\s.]+)", line)
                    if em:
                        edges.append({"subj": em.group(2), "val": em.group(3), "epoch": int(em.group(1)), "authority": 1})
            else:
                m = _re.search(r"GOAL=([^\s]+)\s+HOPS=(\d+)\s+MIN_EP=(\d+)", p)
                assert m is not None
                goal, hops, min_ep = m.group(1), int(m.group(2)), int(m.group(3))
                for line in p.splitlines()[1:]:
                    em = _re.search(r"^E\s+([^\s-]+)->([^\s]+)\s+ep=(\d+)\s+auth=(\d+)", line)
                    if em:
                        edges.append({"subj": em.group(1), "val": em.group(2), "epoch": int(em.group(3)), "authority": int(em.group(4))})
            w = independent_walk_x3_graph(edges, goal, hops=hops, min_epoch=min_ep)
            return f"ANSWER:{w['terminal_value']}"

        if fam == "C06_OPEN_WORLD_DEDUCTION":
            m = _re.search(r"\|F:([^|]*)\|R:([^|]*)\|Q:([^\?]+)\?", p)
            assert m is not None
            facts = {kv.split("=")[0]: (kv.split("=")[1] == "T") for kv in m.group(1).split() if "=" in kv}
            rules = []
            for r_s in m.group(2).split(";"):
                if "->" in r_s:
                    lhs, rhs = r_s.split("->", 1)
                    head, pol = rhs.split(":", 1)
                    rules.append((tuple(lhs.split("+")), head, pol == "T"))
            return f"ANSWER:{_solve_c06_horn(facts, rules, m.group(3))}"

        if fam == "C07_HIERARCHICAL_LISTOPS":
            expr_s = p.split("|", 3)[3].rsplit("=R?", 1)[0]
            toks = expr_s.replace("[", " [ ").replace("]", " ] ").split()
            pos_i = [0]

            def _parse_node() -> Any:
                tok = toks[pos_i[0]]
                pos_i[0] += 1
                if tok == "[":
                    op_name = toks[pos_i[0]]
                    pos_i[0] += 1
                    children = []
                    while toks[pos_i[0]] != "]":
                        children.append(_parse_node())
                    pos_i[0] += 1
                    return [op_name, *children]
                return int(tok)

            tree = _parse_node()
            return f"ANSWER:R{_eval_listops_node(tree)}"

        if fam == "C08_ADAPTIVE_MEMORY_REVISION":
            m = _re.search(r"^C08\|MEM\|([^|]+)\|Q:([^=]+)=\?", p)
            assert m is not None
            evs = []
            for seg in m.group(1).split(";"):
                kind, rest = seg.split(":", 1)
                ent, rest2 = rest.split("=", 1)
                val, ts = rest2.split("@", 1)
                evs.append({"kind": kind, "ent": ent, "val": val, "ts": int(ts)})
            return f"ANSWER:{_solve_c08(evs, m.group(2))}"

        if fam == "C09_METACOGNITION_SELF_CHECK":
            if "|MISSING|" in p:
                mpq = float(_re.search(r"pQ=([0-9.]+)", p).group(1))
                mpr = float(_re.search(r"pR=([0-9.]+)", p).group(1))
                mc = float(_re.search(r"c=([0-9.]+)", p).group(1))
                ev_q = mpq - mc
                ev_r = mpr - mc
                if ev_q >= ev_r and ev_q > 1e-12:
                    return "QUERY"
                if ev_r > ev_q and ev_r > 1e-12:
                    return "RETRIEVE"
                return "ABSTAIN"
            m_e = _re.search(r"X=(\d+)\+(\d+)", p)
            if m_e:
                return f"ANSWER:{int(m_e.group(1)) + int(m_e.group(2))}"
            m_h = _re.search(r"for\s+(\d+)\*(\d+)\+(\d+)-3", p)
            assert m_h is not None
            return f"ANSWER:{int(m_h.group(1)) * int(m_h.group(2)) + int(m_h.group(3)) - 3}"

        if fam == "C10_LANGUAGE_MATH_TRANSFER":
            m1 = _re.search(r"earns\s+(\d+)\s+([A-Za-z_]+)\s+per\s+cycle\s+over\s+(\d+)\s+cycles,\s+adds\s+(\d+)\s+reserve\s+[A-Za-z_]+,\s+and\s+receives\s+(\d+)\s+adjustment", p)
            if m1:
                num = int(m1.group(1)) * int(m1.group(3)) + int(m1.group(4)) + int(m1.group(5))
                return f"ANSWER:{num} {m1.group(2)}"
            m2 = _re.search(r"splits\s+\((\d+)\s+([A-Za-z_]+)\s+over\s+(\d+)\s+cycles\s+plus\s+(\d+)\s+base\s+and\s+(\d+)\s+bonus\)\s+across\s+(\d+)\s+shares", p)
            if m2:
                num = int(m2.group(1)) * int(m2.group(3)) + int(m2.group(4)) + int(m2.group(5))
                fr = _Fraction(num, int(m2.group(6)))
                val_s = str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"
                return f"ANSWER:{val_s} {m2.group(2)}"
        raise ValueError(f"Unsupported public prompt for PublicOracleTypedDecoder: {p[:120]!r}")

    @torch.no_grad()
    def forward(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        action_positions: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        bsz = token_ids.shape[0]
        dev = token_ids.device
        return {
            "action_logits": torch.zeros((bsz, 19), device=dev),
            "uncertainties": torch.full((bsz, 2), 0.5, device=dev),
            "ponder_cost": torch.zeros((bsz,), device=dev),
        }

    @torch.no_grad()
    def generate_bytes(
        self,
        prompt_text: str,
        *,
        max_new_bytes: int | None = None,
        use_persistent_state: bool = False,
        stop_chars: Sequence[str] = ("\n",),
    ) -> dict[str, Any]:
        _ = (use_persistent_state, stop_chars)
        bound = int(max_new_bytes) if max_new_bytes is not None else infer_public_generation_bound(prompt_text)
        txt = self._solve_from_public_prompt(prompt_text)
        overflow = len(txt.encode("utf-8")) > bound
        return {
            "text": txt[:bound] if overflow else txt,
            "terminated_by_eos": not overflow,
            "bound_bytes": bound,
            "generated_bytes": min(bound, len(txt.encode("utf-8"))),
            "epistemic_uncertainty": 0.1,
            "info_uncertainty": 0.1,
            "ponder_cost": 0.0,
        }


def verify_perfect_public_decoder_through_evaluator(
    cases: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Prove the live evaluator + exact public typed scorer yields 1.0 on determinate & k=0 ambiguous cases (§1, §4)."""
    decoder = PublicOracleTypedDecoder()
    # Filter to cases verifiable from public prompt alone (determinate instances + C01 k=0 ambiguity)
    eligible = [
        c for c in cases
        if not (c["family"] == "C02_FST_RULE_COMPOSITION" and int(c.get("n_shots", 0)) < 4)
        and not (c["family"] == "C01_NOVEL_OP_INDUCTION" and 0 < int(c.get("n_shots", 0)) < 4 and bool(c.get("public_ambiguous", False)))
    ]
    preds = evaluate_x3_batched(decoder, eligible, batch_size=batch_size, device="cpu", use_generative_bytes=True)
    profile = score_cognitive_battery_profile(preds, eligible)
    return {
        "schema": "elendril-x3-perfect-public-decoder-check-v1",
        "cases_evaluated": len(eligible),
        "macro_mean": float(profile["macro_mean"]),
        "micro_mean": float(profile["micro_mean"]),
        "worst_dimension_floor": float(profile["worst_dimension_floor"]),
        "passed": bool(profile["micro_mean"] == 1.0 and profile["worst_dimension_floor"] == 1.0),
    }


@torch.no_grad()
def evaluate_x3_batched(
    model: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    use_generative_bytes: bool = True,
) -> list[dict[str, Any]]:
    """Right-padded causal batched evaluation with public typed exact-output scoring (no mod-16 class shortcut)."""
    dev = torch.device(device)
    model = model.to(dev)
    was_train = model.training
    model.eval()
    max_ctx = int(getattr(model, "max_ctx", 640))
    results: list[dict[str, Any]] = []
    for start in range(0, len(cases), max(1, int(batch_size))):
        chunk = cases[start : start + int(batch_size)]
        seqs: list[list[int]] = []
        for c in chunk:
            ids = [UTF8.bos_id, *UTF8.encode(str(c["prompt"])), UTF8.special(SpecialToken.ANSWER)]
            if len(ids) > max_ctx:
                ids = ids[-max_ctx:]
            seqs.append(ids)
        max_l = max(len(s) for s in seqs)
        toks = torch.zeros((len(seqs), max_l), dtype=torch.long, device=dev)
        segs = torch.full((len(seqs), max_l), -1, dtype=torch.long, device=dev)
        apos = torch.zeros((len(seqs),), dtype=torch.long, device=dev)
        for i, ids in enumerate(seqs):
            toks[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=dev)
            segs[i, : len(ids)] = 0
            apos[i] = len(ids) - 1
        pos, mask = packed_layout(segs, torch_module=torch)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
            out = model(toks, pos, mask, action_positions=apos)
        act_logits = out["action_logits"].float().detach().cpu()
        probs = F.softmax(act_logits, dim=-1)
        unc = out["uncertainties"].float().detach().cpu()
        ponder = out["ponder_cost"].float().detach().cpu()

        for i, c in enumerate(chunk):
            prompt = str(c["prompt"])
            bound = infer_public_generation_bound(prompt)
            pred_idx = int(torch.argmax(probs[i]).item())
            op, arg = _action_pair_from_idx(pred_idx)
            conf = float(probs[i, pred_idx].item())
            if use_generative_bytes and hasattr(model, "generate_bytes"):
                gen = model.generate_bytes(prompt, max_new_bytes=bound)
                pred_text = str(gen["text"])
                term_eos = bool(gen.get("terminated_by_eos", True))
            else:
                pred_text = f"ANSWER:{arg}" if (op == "ANSWER" and arg is not None) else op
                term_eos = True
            scored = parse_and_score_public_typed_output(
                prompt, pred_text, c, max_new_bytes=bound, terminated_by_eos=term_eos
            )
            results.append(
                {
                    "case_id": str(c["case_id"]),
                    "family": str(c["family"]),
                    "predicted_text": pred_text,
                    "predicted_idx": pred_idx,
                    "action_logits": act_logits[i],
                    "confidence": conf,
                    "epistemic_uncertainty": float(unc[i, 0].item()),
                    "info_uncertainty": float(unc[i, 1].item()),
                    "ponder_cost": float(ponder[i].item()),
                    **scored,
                }
            )
    if was_train:
        model.train()
    return results


# ---------------------------------------------------------------------------
# ExperimentModule Factory & Registration (PLANTHIR_BACKEND_MODULES=experiments.x3.module)
# ---------------------------------------------------------------------------


def x3_model_factory(architecture: str, model_config: Mapping[str, Any], seed: int) -> X3CognitiveModel:
    scale = str(model_config.get("scale", "tiny"))
    base_spec = build_x3_backbone_spec(scale)
    return X3CognitiveModel(architecture, base_spec, seed=int(seed))


def _wrap_planthir_worlds(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    wrapped = []
    for c in cases:
        w = dict(c)
        w.setdefault("world_id", str(c["case_id"]))
        w.setdefault("sig", str(c.get("canon_sig", c["case_id"])))
        spec = c.get("problem_spec", {})
        w.setdefault("hops", int(spec.get("hops", 1)) if isinstance(spec, Mapping) else 1)
        wrapped.append(w)
    return wrapped


def x3_dataset_factory(dataset_config: Mapping[str, Any], cpu_canary: bool) -> dict[str, Any]:
    seed = int(dataset_config.get("seed", SPLIT_SEEDS["train"]))
    per_fam = 2 if cpu_canary else int(dataset_config.get("per_family", 20))
    tr_bat = build_x3_cognitive_battery(split="train", seed=seed, per_family=per_fam)
    ev_bat = build_x3_cognitive_battery(split="dev", seed=SPLIT_SEEDS["dev"], per_family=1 if cpu_canary else 10)
    mh_bat = build_x3_cognitive_battery(split="ood_transfer", seed=SPLIT_SEEDS["ood_transfer"], per_family=1 if cpu_canary else 4)
    tr_all = _wrap_planthir_worlds(tr_bat["cases"])
    tr_items = [c for i, c in enumerate(tr_all) if i % 2 == 0][: (10 if cpu_canary else len(tr_all))]
    probe_items = [c for i, c in enumerate(tr_all) if i % 2 == 1][: (6 if cpu_canary else len(tr_all))]
    ev_items = _wrap_planthir_worlds(ev_bat["cases"][:8] if cpu_canary else ev_bat["cases"])
    mh_items = _wrap_planthir_worlds([c for c in mh_bat["cases"] if c["family"] in ("C05_RELATIONAL_GRAPH_CF", "C06_OPEN_WORLD_DEDUCTION", "C07_HIERARCHICAL_LISTOPS")][: (3 if cpu_canary else 24)])
    return {
        "train_data": GenericDatasetBundle(
            items=tuple(tr_items),
            dataset_name="x3_cognitive_10fam",
            data_identity=f"x3-10fam-{tr_bat['battery_sha256'][:12]}",
            manifest={"split": "train", "cases": len(tr_items), "battery_sha256": tr_bat["battery_sha256"]},
        ),
        "train_worlds": tr_items,
        "eval_worlds": ev_items,
        "multihop_worlds": mh_items,
        "probe_train_worlds": probe_items,
    }


def x3_loss_fn(model: X3CognitiveModel, chunk: Sequence[Mapping[str, Any]], recipe: Any) -> torch.Tensor:
    dev = next(model.parameters()).device
    samples = []
    g_cls = []
    t_uncert = []
    for c in chunk:
        ids, elig, ap = encode_prompt_and_target_pair(
            str(c["prompt"]), format_public_typed_target(c), max_length=model.max_ctx
        )
        samples.append((ids, elig, ap))
        g_cls.append(max(0, min(18, int(c.get("gold_class", 0)))))
        uk = str(c.get("uncertainty_kind", ""))
        if uk == "information_uncertainty":
            t_uncert.append([0.2, 0.9])
        elif uk == "epistemic_reasoning":
            t_uncert.append([0.9, 0.2])
        else:
            t_uncert.append([0.1, 0.1])

    max_l = max(len(ids) for ids, _, _ in samples)
    bsz = len(samples)
    toks = torch.zeros((bsz, max_l), dtype=torch.long, device=dev)
    segs = torch.full((bsz, max_l), -1, dtype=torch.long, device=dev)
    elig_t = torch.zeros((bsz, max_l), dtype=torch.bool, device=dev)
    apos = torch.zeros((bsz,), dtype=torch.long, device=dev)
    for i, (ids, el, ap) in enumerate(samples):
        toks[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=dev)
        segs[i, : len(ids)] = 0
        elig_t[i, : len(el)] = torch.tensor(el, dtype=torch.bool, device=dev)
        apos[i] = ap
    pos, mask = packed_layout(segs, torch_module=torch)
    out = model(toks, pos, mask, action_positions=apos)

    targets = toks[:, 1:].reshape(-1)
    logits_flat = out["logits"][:, :-1, :].reshape(-1, out["logits"].shape[-1])
    mask_flat = elig_t[:, 1:].reshape(-1).to(logits_flat.dtype)
    byte_ce = (F.cross_entropy(logits_flat, targets, reduction="none") * mask_flat).sum() / mask_flat.sum().clamp_min(1.0)
    act_ce = F.nll_loss(out["action_logits"], torch.tensor(g_cls, dtype=torch.long, device=dev))
    meta_loss = F.mse_loss(out["uncertainties"], torch.tensor(t_uncert, dtype=torch.float32, device=dev))
    ponder_loss = out["ponder_cost"].mean()
    return byte_ce + 0.25 * act_ce + 0.1 * meta_loss + 0.02 * ponder_loss


def x3_kernel_factory() -> X3CognitiveKernel:
    return X3CognitiveKernel()


def x3_policy_factory(
    model: X3CognitiveModel,
    model_spec: Any = None,
    *,
    model_identity: str = "x3-cognitive",
    mode: str = "raw",
    copy_enabled_override: bool | None = None,
) -> X3CognitivePolicy:
    return X3CognitivePolicy(
        model,
        model_spec,
        model_identity=model_identity,
        mode=mode,
        copy_enabled_override=copy_enabled_override,
    )


def build_elendril_x3_module() -> ExperimentModule:
    return ExperimentModule(
        backend_id=X3_BACKEND_ID,
        backend_version="3.0",
        program="ELENDRIL-V8-X3",
        architectures=X3_CANARY_ARCHITECTURES,
        objectives=D3_OBJECTIVES,
        model_factory=x3_model_factory,
        dataset_factory=x3_dataset_factory,
        loss_fn=x3_loss_fn,
        kernel_factory=x3_kernel_factory,
        policy_factory=x3_policy_factory,
        default_dataset_config={"seed": SPLIT_SEEDS["train"], "per_family": 2},
        default_model_config={"scale": "tiny"},
        treatment_architecture="P-R-Iter",
        treatment_model_config={"scale": "tiny"},
        extra_model_keys=("scale",),
        extra_dataset_keys=("seed", "per_family"),
        extra_recipe_keys=("eq_weight", "cf_weight", "cf_margin"),
        supports_grouped_loss=True,
    )


def register_planthir_backends() -> ExperimentModule:
    """Entry point invoked by Palantiri when PLANTHIR_BACKEND_MODULES=experiments.x3.module."""
    mod = build_elendril_x3_module()
    mod.register()
    return mod


# ---------------------------------------------------------------------------
# Preregistered 51-Job Multi-Stage Ledger, S50 Gate, & Confirmation Lock
# ---------------------------------------------------------------------------


def build_x3_program_ledger(*, source_commit: str, updates: int = X3_DEFAULT_UPDATES) -> dict[str, Any]:
    """Build complete preregistered 57-job multi-stage X3 cognitive research ledger."""
    jobs: list[dict[str, Any]] = []

    def _add(stage: str, study: str, arch: str, obj: str, scale: str, gated_by: str | None) -> None:
        for seed in X3_SEEDS:
            jid = f"X3-{stage}-{study}-{scale}-{arch}-{obj}-s{seed}".replace("_", "-")
            cfg = {
                "job_id": jid,
                "stage": stage,
                "study": study,
                "architecture": arch,
                "objective": obj,
                "scale": scale,
                "init_seed": int(seed),
                "order_seed": int(seed) + 1000,
                "updates": int(updates),
                "gated_by": gated_by,
                "confirmation_eligible": stage != "S4",
                "source_commit": str(source_commit),
            }
            cfg["spec_sha256"] = digest(cfg)
            jobs.append(cfg)

    # Stage 0 (6 jobs): R vs R-noedge on D2 (crosses cleanly with completed X1 D1/A0 runs)
    for arch in ("R", "R-noedge"):
        _add("S0", "REDGE", arch, "D2", "S20", None)

    # Stage 1 (12 jobs): D3 algorithm investigation (P, R x D3-Hybrid, D3-NoSurgery, D3-NoAux)
    for arch in ("P", "R"):
        for obj in ("D3-Hybrid", "D3-NoSurgery", "D3-NoAux"):
            _add("S1", "D3", arch, obj, "S20", None)

    # Stage 2 (15 jobs): P-R-Iter + 4 matched active controls on D3-Hybrid (S20)
    for arch in ("P-R-Iter", "P-R-Iter-Fixed", "P-R-Iter-NoRel", "P-R-Iter-AddMem", "P-R-Iter-NoCopy"):
        _add("S2", "ARCH", arch, "D3-Hybrid", "S20", None)

    # Stage 3 (6 jobs): Integration & dual copy/generate interface across D3-Hybrid vs D2 (S20)
    for obj in ("D3-Hybrid", "D2"):
        _add("S3", "INTEG", "P-R-Iter", obj, "S20", "S2")

    # Stage 4 (12 jobs): Evidence-Gated S50 (50.55M core) Scaling Law
    for arch, obj in (("Base", "D3-Hybrid"), ("P", "D3-Hybrid"), ("P-R-Iter", "CE"), ("P-R-Iter", "D3-Hybrid")):
        _add("S4", "SCALING", arch, obj, "S50", "S20_SCALING_GATE")

    assert len(jobs) == 57, f"Expected 57 GPU jobs, got {len(jobs)}"
    return {
        "schema": "elendril-x3-program-ledger-v1",
        "source_commit": str(source_commit),
        "total_gpu_jobs": len(jobs),
        "s20_jobs": sum(1 for j in jobs if j["scale"] == "S20"),
        "s50_jobs": sum(1 for j in jobs if j["scale"] == "S50"),
        "jobs": jobs,
        "ledger_sha256": digest(jobs),
    }


def evaluate_s50_scaling_gate(
    s20_stage_scores: Mapping[str, Mapping[str, float]],
    *,
    min_macro_gain_over_p: float = 0.08,
    min_absent_gain_over_p: float = 0.15,
    min_meta_auroc: float = 0.70,
) -> dict[str, Any]:
    """Evidence gate determining whether S50 (50.55M) scaling runs or stays gated at 0.0 GPU s (§11)."""
    trt = s20_stage_scores.get("P-R-Iter")
    ref_p = s20_stage_scores.get("P")
    if not trt or not ref_p:
        return {
            "schema": "elendril-x3-s50-gate-v1",
            "status": "GATED_PENDING_S20_QUALIFICATION",
            "unlocked": False,
            "gpu_seconds_spent": 0.0,
            "reason": "MISSING_S20_EVIDENCE",
        }
    macro_delta = float(trt.get("macro_mean", 0.0) - ref_p.get("macro_mean", 0.0))
    absent_delta = float(trt.get("absent_from_input_accuracy", 0.0) - ref_p.get("absent_from_input_accuracy", 0.0))
    auroc = float(trt.get("epistemic_vs_info_auroc", 0.0))
    beats_controls = all(
        float(trt.get("macro_mean", 0.0)) > float(s20_stage_scores.get(c, {}).get("macro_mean", -1.0))
        for c in ("P-R-Iter-Fixed", "P-R-Iter-NoRel", "P-R-Iter-AddMem", "P-R-Iter-NoCopy")
        if c in s20_stage_scores
    )
    passed = (
        macro_delta >= float(min_macro_gain_over_p)
        and absent_delta >= float(min_absent_gain_over_p)
        and auroc >= float(min_meta_auroc)
        and beats_controls
    )
    return {
        "schema": "elendril-x3-s50-gate-v1",
        "status": "UNLOCKED" if passed else "GATED_S20_THRESHOLD_NOT_MET",
        "unlocked": bool(passed),
        "gpu_seconds_spent": 0.0 if not passed else None,
        "macro_delta_over_p": macro_delta,
        "absent_delta_over_p": absent_delta,
        "epistemic_vs_info_auroc": auroc,
        "beats_active_controls": bool(beats_controls),
    }


def evaluate_confirmation_lock(
    *,
    required_job_ids: Sequence[str],
    verified_completed_ids: Sequence[str],
    confirm_battery: Mapping[str, Any] | None = None,
    lock_out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Complete-cohort Confirmation Lock: NEVER unseals confirm during PARTIAL/deferred sessions (§2)."""
    bat = confirm_battery if confirm_battery is not None else build_x3_cognitive_battery(split="confirm", per_family=20)
    done = set(str(x) for x in verified_completed_ids)
    req = [str(x) for x in required_job_ids]
    pending = [jid for jid in req if jid not in done]
    unlocked = len(pending) == 0 and len(req) > 0
    doc = {
        "schema": "elendril-x3-confirmation-lock-v1",
        "confirm_split": "confirm",
        "confirm_battery_sha256": str(bat["battery_sha256"]),
        "required_job_count": len(req),
        "completed_required_count": len(req) - len(pending),
        "pending_confirm_jobs": pending,
        "unlocked": bool(unlocked),
        "status": "UNLOCKED_COMPLETE_COHORT" if unlocked else "LOCKED_PENDING_COHORT",
    }
    if lock_out_path is not None:
        _atomic_write_json(Path(lock_out_path), doc)
    return doc


# ---------------------------------------------------------------------------
# CPU Every-Architecture u0000 Smoke & Production Job Worker / Campaign Runner
# ---------------------------------------------------------------------------


def _publish_x3_checkpoint(
    store: CheckpointStore,
    *,
    model: X3CognitiveModel,
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
    """Publish atomic SHA-256 content-addressed checkpoint via v5_training.CheckpointStore & prune old generations."""
    data_sha = digest({"program": "ELENDRIL-V8-X3", "arch": architecture, "obj": objective})
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
        digest(["d3", architecture, objective]),
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
        tokens_by_source={"elendril_x3": tot_tok},
        optimizer_step_max=int(global_update),
        schedule_tokens=tot_tok,
        tokens_per_update=int(batch_size),
        curriculum_phase=f"x3-{architecture}",
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
    opt_bytes = _dump(optimizer.state_dict()) if include_optimizer else _dump({})
    payloads = {
        "model.bin": _dump({k: v.detach().cpu() for k, v in model.state_dict().items()}),
        "optimizer.bin": opt_bytes,
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


def _restore_x3_checkpoint(
    store: CheckpointStore,
    sha: str,
    *,
    model: X3CognitiveModel,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
    expected_spec_sha256: str | None = None,
) -> dict[str, Any]:
    state, payloads = store.restore(sha)
    sched = json.loads(payloads["scheduler.json"])
    meta = sched.get("metadata", sched)
    if expected_spec_sha256 and meta.get("spec_sha256") != expected_spec_sha256:
        raise ValueError(
            f"SWAPPED_SPEC_MISMATCH:{store.lineage_id}: expected {expected_spec_sha256}, got {meta.get('spec_sha256')}"
        )
    weights = _load(payloads["model.bin"])
    model.load_state_dict(weights)
    model.to(torch.device(device))
    if optimizer is not None and "optimizer.bin" in payloads:
        opt_state = _load(payloads["optimizer.bin"])
        if isinstance(opt_state, Mapping) and opt_state.get("state"):
            optimizer.load_state_dict(opt_state)
        if "rng.bin" in payloads:
            torch.set_rng_state(torch.frombuffer(bytearray(payloads["rng.bin"]), dtype=torch.uint8))
    return {"state": state, "meta": meta, "global_update": int(state.global_update), "sha": str(sha)}


def run_all_architectures_u0000_smoke(
    work_dir: Path | str,
    *,
    source_commit: str,
    include_s50_construction: bool = False,
) -> dict[str, Any]:
    """Mandatory CPU u0000 init + 1-train-step + 1-eval-step + checkpoint save/restore across ALL X3 & legacy B0 paths."""
    wroot = Path(work_dir)
    wroot.mkdir(parents=True, exist_ok=True)
    tiny_spec = build_x3_backbone_spec("tiny")
    bat = build_x3_cognitive_battery(split="train", seed=SPLIT_SEEDS["train"], per_family=2)["cases"]
    batches = build_d3_hybrid_batches(bat, seed=7, batch_size=4)
    results: dict[str, Any] = {}

    for arch in X3_ARCHITECTURES:
        m = X3CognitiveModel(arch, tiny_spec, seed=7)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        step0_eval = evaluate_x3_batched(m, bat[:4], batch_size=4, device="cpu", use_generative_bytes=True)
        step_stats = compute_d3_objective_step(m, opt, batches[0], objective="D3-Hybrid")
        dead_grads = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
        assert not dead_grads, f"Dead grads on {arch}: {dead_grads}"
        assert math.isfinite(float(step_stats["total_loss"])), f"Non-finite loss on {arch}"

        lin_id = f"smoke-{arch}".replace("_", "-")
        store = CheckpointStore(wroot / "u0000_ckpts", lin_id)
        sha = _publish_x3_checkpoint(
            store,
            model=m,
            optimizer=opt,
            spec=tiny_spec,
            spec_sha256=tiny_spec.sha256(),
            lineage_id=lin_id,
            global_update=1,
            target_updates=1,
            source_commit=source_commit,
            architecture=arch,
            objective="D3-Hybrid",
            parent_sha=store.latest_sha256(),
            batch_size=4,
        )
        m2 = X3CognitiveModel(arch, tiny_spec, seed=19)
        _restore_x3_checkpoint(store, sha, model=m2, device="cpu", expected_spec_sha256=tiny_spec.sha256())
        ev1 = evaluate_x3_batched(m, bat[:2], batch_size=2, device="cpu", use_generative_bytes=False)
        ev2 = evaluate_x3_batched(m2, bat[:2], batch_size=2, device="cpu", use_generative_bytes=False)
        max_diff = max(float((a["action_logits"] - b["action_logits"]).abs().max().item()) for a, b in zip(ev1, ev2))
        assert max_diff < 1e-6, f"Checkpoint restore parity failed on {arch}: {max_diff}"
        results[arch] = {
            "params": sum(p.numel() for p in m.parameters()),
            "u0000_eval_cases": len(step0_eval),
            "u0001_loss": round(float(step_stats["total_loss"]), 4),
            "restore_max_diff": max_diff,
            "checkpoint_sha": sha,
            "passed": True,
        }

    # Compulsory legacy B0 regression smoke on tiny proxy
    b0_rec = build_legacy_x1_scaled_model("B0", scale="tiny", init_seed=7)
    b0_m = b0_rec["model"]
    b0_opt = torch.optim.AdamW(b0_m.parameters(), lr=1e-4)
    b0_toks = torch.tensor([[1, 65, 66, 3, 67, 2]], dtype=torch.long)
    b0_segs = torch.zeros_like(b0_toks)
    b0_pos, b0_mask = packed_layout(b0_segs, torch_module=torch)
    b0_h = b0_m.forward_hidden(b0_toks, b0_pos, b0_mask)
    b0_logits = F.linear(b0_h, b0_m.embedding.weight)
    b0_loss = F.cross_entropy(b0_logits[:, :-1, :].reshape(-1, b0_logits.shape[-1]), b0_toks[:, 1:].reshape(-1))
    b0_loss.backward()
    b0_opt.step()
    b0_dead = [n for n, p in b0_m.named_parameters() if p.requires_grad and p.grad is None]
    assert not b0_dead and math.isfinite(float(b0_loss.item()))
    results["legacy-B0"] = {
        "params": b0_rec["total_parameters"],
        "u0001_loss": round(float(b0_loss.item()), 4),
        "passed": True,
    }

    s50_counts: dict[str, int] = {}
    if include_s50_construction:
        s50_spec = build_x3_backbone_spec("S50")
        for arch in ("Base", "P", "P-R-Iter"):
            ms = X3CognitiveModel(arch, s50_spec, seed=7)
            s50_counts[arch] = sum(p.numel() for p in ms.parameters())
            del ms
        for leg_arch in ("P-50M", "R-50M"):
            lrec = build_legacy_x1_scaled_model(leg_arch, scale="S50", init_seed=7)
            s50_counts[f"legacy-{leg_arch}"] = lrec["total_parameters"]
            del lrec

    report = {
        "schema": "elendril-x3-u0000-all-arch-smoke-v2",
        "architectures_tested": [k for k in results if k != "legacy-B0"],
        "legacy_b0_tested": True,
        "all_passed": all(k in results for k in X3_ARCHITECTURES) and results["legacy-B0"]["passed"],
        "s50_constructed_params": s50_counts,
        "details": results,
    }
    _atomic_write_json(wroot / "U0000_ALL_ARCH_SMOKE.json", report)
    return report


def run_x3_single_job(
    *,
    job: Mapping[str, Any],
    work_root: Path | str,
    device: str = "cpu",
    canary: bool = False,
    resume: bool = True,
    ckpt_every: int = 512,
    updates_override: int | None = None,
) -> dict[str, Any]:
    """Execute or resume a single registered X3 job atomically with dependency-aware retention & missing-parent isolation."""
    wroot = Path(work_root)
    wroot.mkdir(parents=True, exist_ok=True)
    jid = str(job["job_id"])
    arch = str(job["architecture"])
    obj = str(job["objective"])
    scale = "tiny" if canary else str(job.get("scale", "S20"))
    seed = int(job["init_seed"])
    oseed = int(job.get("order_seed", seed + 1000))
    source_commit = str(job["source_commit"])
    spec_sha = str(job["spec_sha256"])

    # Isolate missing parent checkpoints without crashing independent jobs in the wave (§0, §5)
    parent_id = job.get("parent_job_id")
    if parent_id:
        pdir = Path(job.get("parent_checkpoint_dir") or (wroot.parent / str(parent_id)))
        has_cm = (pdir / "c").is_dir() and bool(CheckpointStore(pdir / "c", "m").latest_sha256())
        has_main = (pdir / "checkpoints").is_dir() and bool(CheckpointStore(pdir / "checkpoints", "main").latest_sha256())
        if not (has_cm or has_main):
            blocked = {
                "schema": "elendril-x3-job-receipt-v1",
                "job_id": jid,
                "architecture": arch,
                "objective": obj,
                "scale": scale,
                "init_seed": seed,
                "status": "BLOCKED_MISSING_PARENT",
                "missing_parent_job_id": str(parent_id),
                "source_commit": source_commit,
                "spec_sha256": spec_sha,
            }
            _atomic_write_json(wroot / "JOB_RECEIPT.json", blocked)
            return blocked

    target_updates = int(updates_override) if updates_override is not None else (4 if canary else int(job["updates"]))
    full_job_updates = 4 if canary else max(target_updates, int(job.get("updates", target_updates)))
    preserve_final_opt = bool(job.get("preserve_final_optimizer", False))
    dev = torch.device(device)
    spec = build_x3_backbone_spec(scale)
    model = X3CognitiveModel(arch, spec, seed=seed).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Keep short "c"/"m" dirs inside per-job folder so nested Windows paths stay safely < MAX_PATH
    store = CheckpointStore(wroot / "c", "m")
    start_u = 0
    latest_sha = store.latest_sha256() if resume else None
    if latest_sha is not None:
        restored = _restore_x3_checkpoint(
            store, latest_sha, model=model, optimizer=optimizer, device=device, expected_spec_sha256=spec_sha
        )
        start_u = int(restored["global_update"])

    tr_bat = build_x3_cognitive_battery(split="train", seed=SPLIT_SEEDS["train"], per_family=2 if canary else 40)
    dev_bat = build_x3_cognitive_battery(split="dev", seed=SPLIT_SEEDS["dev"], per_family=2 if canary else 12)
    batch_sz = 4 if canary else 16
    raw_batches = build_d3_hybrid_batches(tr_bat["cases"], seed=oseed, batch_size=batch_sz)
    batches = prepack_d3_batches(raw_batches, max_length=model.max_ctx, device=dev, objective=obj)
    use_cuda_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    t0 = time.perf_counter()
    last_stats: dict[str, Any] = {}
    updates_log = wroot / "updates.jsonl"
    with updates_log.open("a", encoding="utf-8") as fh:
        for u in range(start_u, target_updates):
            step_b = batches[u % len(batches)]
            last_stats = compute_d3_objective_step(
                model,
                optimizer,
                step_b,
                objective=obj,
                update_idx=u,
                use_amp=use_cuda_amp,
                scaler=scaler if use_cuda_amp else None,
            )
            cur_u = u + 1
            fh.write(json.dumps({"update": cur_u, **last_stats}, sort_keys=True) + "\n")
            if (cur_u % max(1, int(ckpt_every)) == 0) or (cur_u == target_updates):
                # Keep full AdamW state mid-run, in canary resume tests, or on continuation/adaptation parents!
                keep_opt = bool(canary) or (cur_u < full_job_updates) or preserve_final_opt
                latest_sha = _publish_x3_checkpoint(
                    store,
                    model=model,
                    optimizer=optimizer,
                    spec=spec,
                    spec_sha256=spec_sha,
                    lineage_id="m",
                    global_update=cur_u,
                    target_updates=full_job_updates,
                    source_commit=source_commit,
                    architecture=arch,
                    objective=obj,
                    parent_sha=latest_sha,
                    batch_size=batch_sz,
                    include_optimizer=keep_opt,
                )
    store.prune(keep=set())

    preds = evaluate_x3_batched(model, dev_bat["cases"], batch_size=4 if canary else 32, device=device, use_generative_bytes=True)
    profile = score_cognitive_battery_profile(preds, dev_bat["cases"])
    pred_by_id = {str(p["case_id"]): p for p in preds}
    dev_mistakes = [
        {
            "case_id": str(c["case_id"]),
            "family": str(c["family"]),
            "prompt": str(c["prompt"])[:220],
            "gold_answer": str(c["gold_text"]),
            "pred_answer": str(pred_by_id[str(c["case_id"])]["predicted_text"]),
            "absent_from_input": bool(c.get("absent_from_input")),
        }
        for c in dev_bat["cases"]
        if str(c["case_id"]) in pred_by_id and not bool(pred_by_id[str(c["case_id"])]["correct"])
    ][:40]
    wall = round(time.perf_counter() - t0, 3)

    receipt = {
        "schema": "elendril-x3-job-receipt-v1",
        "job_id": jid,
        "stage": str(job.get("stage", "")),
        "architecture": arch,
        "objective": obj,
        "scale": scale,
        "init_seed": seed,
        "status": "COMPLETE",
        "source_commit": source_commit,
        "spec_sha256": spec_sha,
        "target_updates": target_updates,
        "completed_updates": target_updates,
        "preserve_final_optimizer": preserve_final_opt,
        "eval_rows": len(preds),
        "checkpoint_sha256": latest_sha,
        "dev_macro_mean": profile["macro_mean"],
        "dev_worst_floor": profile["worst_dimension_floor"],
        "dev_profile": profile,
        "dev_mistakes": dev_mistakes,
        "last_step_stats": last_stats,
        "wall_seconds": wall,
    }
    _atomic_write_json(wroot / "JOB_RECEIPT.json", receipt)
    return receipt


def _mirror_artifacts_to_dir(files: Sequence[Path], target_dir: Path | str) -> list[str]:
    """Zero-copy os.link mirror (fallback shutil.copy2) of top-level outputs into /kaggle/working."""
    td = Path(target_dir)
    td.mkdir(parents=True, exist_ok=True)
    mirrored: list[str] = []
    for src in files:
        if not src.is_file():
            continue
        dst = td / src.name
        if src.resolve() == dst.resolve():
            mirrored.append(str(dst))
            continue
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        mirrored.append(str(dst))
    return mirrored


def _link_latest_full_files(work_dir: Path) -> dict[str, Path]:
    """Pack ONLY LATEST checkpoints + receipts/JSONs into *_FULL.zip to stay safely under 19.5 GiB."""
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
        for fname in ("JOB_RECEIPT.json", "updates.jsonl"):
            fp = jdir / fname
            if fp.is_file():
                out[f"jobs/{jdir.name}/{fname}"] = fp
        ck_root = jdir / "c"
        if not ck_root.is_dir():
            ck_root = jdir / "checkpoints"
        if ck_root.is_dir():
            for lat_p in ck_root.rglob("LATEST"):
                sha = lat_p.read_text(encoding="utf-8").strip().splitlines()[0] if lat_p.is_file() else ""
                out[lat_p.relative_to(work_dir).as_posix()] = lat_p
                if sha:
                    obj_dir = lat_p.parent / "objects" / sha
                    if obj_dir.is_dir():
                        for fp in sorted(obj_dir.rglob("*")):
                            if fp.is_file():
                                out[fp.relative_to(work_dir).as_posix()] = fp
                # Include index manifests alongside LATEST
                for idx_f in lat_p.parent.glob("*.json"):
                    out[idx_f.relative_to(work_dir).as_posix()] = idx_f
    return out


X3_PAIRED_CONTRASTS: tuple[tuple[str, str, str, str], ...] = (
    ("R_vs_Rnoedge_D2", "R-noedge", "D2", "R"),
    ("D3_Hybrid_vs_NoSurgery_R", "R", "D3-NoSurgery", "R"),
    ("D3_Hybrid_vs_NoAux_R", "R", "D3-NoAux", "R"),
    ("P-R-Iter_vs_Fixed", "P-R-Iter-Fixed", "D3-Hybrid", "P-R-Iter"),
    ("P-R-Iter_vs_NoRel", "P-R-Iter-NoRel", "D3-Hybrid", "P-R-Iter"),
    ("P-R-Iter_vs_AddMem", "P-R-Iter-AddMem", "D3-Hybrid", "P-R-Iter"),
    ("P-R-Iter_vs_NoCopy", "P-R-Iter-NoCopy", "D3-Hybrid", "P-R-Iter"),
    ("P-R-Iter_vs_P_D3", "P", "D3-Hybrid", "P-R-Iter"),
)


def _run_stage5_sealed_evaluations(
    *,
    wdir: Path,
    completed_ids: set[str],
    job_receipts: Mapping[str, Mapping[str, Any]],
    jobs_by_id: Mapping[str, Mapping[str, Any]],
    splits_map: Mapping[str, Sequence[Mapping[str, Any]]],
    eval_device: str,
    canary: bool,
) -> dict[str, Any]:
    """Run Stage 5 sealed confirm, OOD transfer, and 5 learning modes once the cohort unseals."""
    confirm_cases = list(splits_map["confirm"])
    ood_cases = list(splits_map["ood_transfer"])
    dev_cases = list(splits_map.get("dev", confirm_cases))

    icl_map = {
        sh: [
            c
            for c in confirm_cases
            if c["family"] in ("C01_NOVEL_OP_INDUCTION", "C02_FST_RULE_COMPOSITION")
            and int(c.get("n_shots", 0)) == sh
        ][: (2 if canary else 4)]
        or [confirm_cases[0]]
        for sh in (0, 1, 4, 16, 64)
    }
    state_seq = [c for c in confirm_cases if c["family"] == "C08_ADAPTIVE_MEMORY_REVISION"][: (2 if canary else 8)] or confirm_cases[:2]
    grad_demos = [c for c in dev_cases if c["family"] == "C01_NOVEL_OP_INDUCTION"][: (2 if canary else 4)] or dev_cases[:2]
    grad_queries = [c for c in confirm_cases if c["family"] == "C01_NOVEL_OP_INDUCTION"][: (2 if canary else 8)] or confirm_cases[:2]
    transfer_sub = [
        c
        for c in ood_cases
        if c["family"] in ("C10_LANGUAGE_MATH_TRANSFER", "C03_GRID_TRANSFORM_ARC")
    ][: (2 if canary else 10)] or ood_cases[:2]

    conf_by_job: dict[str, Any] = {}
    ood_by_job: dict[str, Any] = {}
    modes_by_job: dict[str, Any] = {}
    rep_archs_seen: set[str] = set()
    rep_targets = ("P-R-Iter", "P", "R", "P-R-Iter-Fixed", "Base")
    bsz = 4 if canary else 32

    def _collect_mistakes(preds: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        p_map = {str(p["case_id"]): p for p in preds}
        wrong = []
        for c in cases:
            cid = str(c["case_id"])
            pr = p_map.get(cid)
            if pr is not None and not bool(pr.get("correct")):
                wrong.append(
                    {
                        "case_id": cid,
                        "family": str(c["family"]),
                        "prompt": str(c["prompt"])[:200],
                        "gold_answer": str(c["gold_text"]),
                        "pred_answer": str(pr.get("predicted_text", "")),
                        "absent_from_input": bool(c.get("absent_from_input")),
                    }
                )
            if len(wrong) >= 25:
                break
        return wrong

    for jid in sorted(completed_ids):
        meta = dict(job_receipts.get(jid) or jobs_by_id.get(jid) or {})
        arch = str(meta.get("architecture", ""))
        obj = str(meta.get("objective", ""))
        scale_val = "tiny" if canary else str(meta.get("scale", "S20"))
        ck_root = wdir / "jobs" / jid / "c"
        lin_id = "m"
        if not ck_root.is_dir():
            ck_root = wdir / "jobs" / jid / "checkpoints"
            lin_id = "main"
        sha = CheckpointStore(ck_root, lin_id).latest_sha256() if ck_root.is_dir() else None
        if not sha or not arch:
            conf_by_job[jid] = {"job_id": jid, "weights_available": False, "architecture": arch, "objective": obj}
            ood_by_job[jid] = {"job_id": jid, "weights_available": False, "architecture": arch, "objective": obj}
            continue

        store = CheckpointStore(ck_root, lin_id)
        spec = build_x3_backbone_spec(scale_val)
        m = X3CognitiveModel(arch, spec, seed=int(meta.get("init_seed", 7)))
        _restore_x3_checkpoint(store, sha, model=m, device=eval_device, expected_spec_sha256=str(meta.get("spec_sha256", "")))

        c_preds = evaluate_x3_batched(m, confirm_cases, batch_size=bsz, device=eval_device, use_generative_bytes=True)
        c_prof = score_cognitive_battery_profile(c_preds, confirm_cases)
        conf_by_job[jid] = {
            "job_id": jid,
            "weights_available": True,
            "architecture": arch,
            "objective": obj,
            "scale": scale_val,
            "init_seed": int(meta.get("init_seed", 7)),
            "checkpoint_sha256": sha,
            "profile": c_prof,
            "mistakes": _collect_mistakes(c_preds, confirm_cases),
        }

        o_preds = evaluate_x3_batched(m, ood_cases, batch_size=bsz, device=eval_device, use_generative_bytes=True)
        o_prof = score_cognitive_battery_profile(o_preds, ood_cases)
        ood_by_job[jid] = {
            "job_id": jid,
            "weights_available": True,
            "architecture": arch,
            "objective": obj,
            "scale": scale_val,
            "init_seed": int(meta.get("init_seed", 7)),
            "checkpoint_sha256": sha,
            "profile": o_prof,
            "mistakes": _collect_mistakes(o_preds, ood_cases),
        }

        if arch in rep_targets and arch not in rep_archs_seen:
            rep_archs_seen.add(arch)
            modes_by_job[jid] = {
                "job_id": jid,
                "architecture": arch,
                "objective": obj,
                "scale": scale_val,
                "modes": evaluate_five_learning_modes(
                    m,
                    icl_cases_by_shot=icl_map,
                    state_adapt_sequence=state_seq,
                    grad_adapt_demos=grad_demos,
                    grad_adapt_queries=grad_queries,
                    transfer_cases=transfer_sub,
                    oracle_solver=solve_cognitive_case_oracle,
                ),
            }

        del m
        if str(eval_device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    conf_doc = {
        "schema": "elendril-x3-sealed-confirm-results-v1",
        "split": "confirm",
        "cases": len(confirm_cases),
        "evaluated_jobs": sum(1 for v in conf_by_job.values() if v.get("weights_available")),
        "jobs": conf_by_job,
    }
    ood_doc = {
        "schema": "elendril-x3-ood-transfer-results-v1",
        "split": "ood_transfer",
        "cases": len(ood_cases),
        "evaluated_jobs": sum(1 for v in ood_by_job.values() if v.get("weights_available")),
        "jobs": ood_by_job,
    }
    modes_doc = {
        "schema": "elendril-x3-five-modes-summary-v1",
        "evaluated_models": len(modes_by_job),
        "models": modes_by_job,
    }
    _atomic_write_json(wdir / "X3_SEALED_CONFIRMATION_RESULTS.json", conf_doc)
    _atomic_write_json(wdir / "X3_OOD_TRANSFER_RESULTS.json", ood_doc)
    _atomic_write_json(wdir / "X3_FIVE_LEARNING_MODES.json", modes_doc)
    return {
        "sealed_confirm_evaluated_jobs": conf_doc["evaluated_jobs"],
        "ood_transfer_evaluated_jobs": ood_doc["evaluated_jobs"],
        "five_learning_modes_evaluated": modes_doc["evaluated_models"],
    }


def run_x3_campaign(
    *,
    work_dir: Path | str,
    export_dir: Path | str,
    source_commit: str,
    devices: Sequence[str] = ("cuda:0", "cuda:1"),
    session_minutes: float = 660.0,
    reserve_minutes: float = 25.0,
    measured_seconds_per_update: float = 0.28,
    per_job_disk_bytes: int = 95 * 1024 * 1024,
    canary: bool = False,
    resume_receipts: Sequence[Mapping[str, Any]] = (),
    in_process_runner: Callable[..., Any] | None = None,
    mirror_dir: Path | str | None = None,
    force_s50_gate: bool | None = None,
    clock_fn: Callable[[], float] | None = None,
    jobs_override: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Orchestrate X3 multi-stage DAG campaign using Palantiri CampaignOrchestrator, stats & verified archives."""
    wdir = Path(work_dir)
    edir = Path(export_dir)
    wdir.mkdir(parents=True, exist_ok=True)
    edir.mkdir(parents=True, exist_ok=True)
    now_fn = clock_fn if clock_fn is not None else time.monotonic
    t_start = now_fn()
    usable_seconds = max(60.0, (float(session_minutes) - float(reserve_minutes)) * 60.0) if session_minutes > reserve_minutes else float(session_minutes) * 60.0

    # Stage 0 CPU empirical sampler exposure audit (0 GPU seconds)
    sampler_audit = audit_x1_sampler_exposure(quota_per_family=24)
    _atomic_write_json(wdir / "X3_SAMPLER_AUDIT.json", sampler_audit)

    ledger = build_x3_program_ledger(source_commit=source_commit, updates=4 if canary else X3_DEFAULT_UPDATES)
    r0_r7_ledger = build_r0_r7_program_ledger(source_commit=source_commit)
    raw_jobs = list(jobs_override) if jobs_override is not None else list(ledger["jobs"])
    jobs_by_id = {str(j["job_id"]): dict(j) for j in raw_jobs}
    _atomic_write_json(wdir / "X3_DESIGN_REGISTRY.json", ledger)
    _atomic_write_json(wdir / "X3_R0_R7_LEDGER.json", r0_r7_ledger)

    # Disjoint split custody verification across all 4 splits
    splits_map = {sp: build_x3_cognitive_battery(split=sp, per_family=6 if canary else 20)["cases"] for sp in ("train", "dev", "confirm", "ood_transfer")}
    disjoint_rep = verify_x3_split_disjointness(splits_map)
    shortcut_rep = eval_x3_battery_shortcuts(splits_map["dev"])
    mask_rep = verify_x3_destructive_masks(splits_map["dev"])
    _atomic_write_json(wdir / "X3_CUSTODY_PREFLIGHT.json", {"disjoint": disjoint_rep, "shortcuts": shortcut_rep, "destructive_masks": mask_rep})

    orch = CampaignOrchestrator(
        campaign_id="ELENDRIL-V8-X3",
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
            session=str(j["stage"]),
            extra=dict(j),
        )
        for j in raw_jobs
    ]
    orch.prepare(all_sjobs)

    reconcile = orch.reconcile_verified_resume(
        resume_receipts,
        raw_jobs,
        fail_on_rejection=bool(resume_receipts),
    )
    completed_ids = set(reconcile["verified_completed_jobs"])
    job_receipts: dict[str, dict[str, Any]] = {
        str(r["job_id"]): dict(r)
        for r in resume_receipts
        if str(r.get("job_id", "")) in completed_ids
    }
    # Load any existing local JOB_RECEIPT.json already matching completed IDs
    for jid in list(completed_ids):
        rp = wdir / "jobs" / jid / "JOB_RECEIPT.json"
        if rp.is_file() and jid not in job_receipts:
            job_receipts[jid] = json.loads(rp.read_text(encoding="utf-8"))

    # Execute Stages S0..S3 (all S20 jobs) BEFORE evaluating S50 gate
    s20_jobs = [s for s in all_sjobs if s.extra.get("scale", "S20") != "S50" and s.job_id not in completed_ids]
    s50_jobs_all = [s for s in all_sjobs if s.extra.get("scale", "S20") == "S50"]

    def _cmd_builder(job: SessionJob, wroot: Path) -> tuple[str, ...]:
        spec_p = wdir / "specs" / f"{job.job_id}.json"
        _atomic_write_json(spec_p, job.extra)
        cmd = [
            sys.executable,
            "-m",
            "experiments.x3.module",
            "run-job",
            "--job-json",
            str(spec_p),
            "--work-root",
            str(wroot),
            "--device",
            "cpu" if canary else "cuda:0",
        ]
        if canary:
            cmd.append("--canary")
        return tuple(cmd)

    # Dual-clock admit_slice for S20 batch + wave-by-wave execution with live remaining-time check
    plan = orch.admit_slice(
        s20_jobs,
        measured_seconds_per_update=float(measured_seconds_per_update),
        per_job_disk_bytes=int(per_job_disk_bytes),
        parallel_workers=max(1, len(devices)),
    )
    admitted_set = set(plan["admitted_job_ids"])
    admitted_s20 = [j for j in s20_jobs if j.job_id in admitted_set]
    deferred_ids = list(plan["deferred_pending_job_ids"])

    failed_ids: list[str] = []
    blocked_ids: list[str] = []
    stopped_reason: str | None = None
    n_devs = max(1, len(devices))
    for w_start in range(0, len(admitted_s20), n_devs):
        rem_sec = usable_seconds - (now_fn() - t_start)
        if rem_sec <= 0.0:
            deferred_ids = [j.job_id for j in admitted_s20[w_start:]] + deferred_ids
            stopped_reason = "DEADLINE_RESERVE_REACHED"
            break
        wave_jobs = admitted_s20[w_start : w_start + n_devs]
        wave_res = orch.run_waves(wave_jobs, command_builder=_cmd_builder, devices=tuple(devices), resume=True)
        for wid in wave_res.get("completed_jobs", ()):
            rp = wdir / "jobs" / wid / "JOB_RECEIPT.json"
            if rp.is_file():
                rec_doc = json.loads(rp.read_text(encoding="utf-8"))
                job_receipts[wid] = rec_doc
                if rec_doc.get("status") == "BLOCKED_MISSING_PARENT":
                    blocked_ids.append(wid)
                    completed_ids.discard(wid)
                    continue
            completed_ids.add(wid)
        for wrec in wave_res.get("waves", ()):
            failed_ids.extend(wrec.get("failed_workers", ()))

    # Summarize S20 stage scores for evidence-gated S50 promotion
    by_arch_scores: dict[str, dict[str, list[float]]] = {}
    for rec in job_receipts.values():
        if str(rec.get("scale", "S20")) != "S50" and "dev_profile" in rec:
            prof = rec["dev_profile"]
            bucket = by_arch_scores.setdefault(str(rec["architecture"]), {"macro": [], "absent": [], "auroc": []})
            bucket["macro"].append(float(prof.get("macro_mean", 0.0)))
            bucket["absent"].append(float(prof.get("absent_from_input_accuracy", 0.0)))
            bucket["auroc"].append(float(prof.get("epistemic_vs_info_auroc", 0.5)))
    s20_means = {
        a: {
            "macro_mean": sum(d["macro"]) / max(1, len(d["macro"])),
            "absent_from_input_accuracy": sum(d["absent"]) / max(1, len(d["absent"])),
            "epistemic_vs_info_auroc": sum(d["auroc"]) / max(1, len(d["auroc"])),
        }
        for a, d in by_arch_scores.items()
    }
    s50_gate = evaluate_s50_scaling_gate(s20_means)
    if force_s50_gate is not None:
        s50_gate["unlocked"] = bool(force_s50_gate)
        s50_gate["status"] = "UNLOCKED" if force_s50_gate else "GATED_PENDING_S20_QUALIFICATION"
    _atomic_write_json(wdir / "X3_S50_SCALING_GATE.json", s50_gate)

    pending_s50 = [j for j in s50_jobs_all if j.job_id not in completed_ids]
    gated_s50_ids: list[str] = []
    if s50_gate["unlocked"] and not deferred_ids and not failed_ids:
        for w_start in range(0, len(pending_s50), n_devs):
            rem_sec = usable_seconds - (now_fn() - t_start)
            if rem_sec <= 0.0:
                deferred_ids.extend(j.job_id for j in pending_s50[w_start:])
                stopped_reason = "DEADLINE_RESERVE_REACHED"
                break
            wave_jobs = pending_s50[w_start : w_start + n_devs]
            wave_res = orch.run_waves(wave_jobs, command_builder=_cmd_builder, devices=tuple(devices), resume=True)
            for wid in wave_res.get("completed_jobs", ()):
                completed_ids.add(wid)
                rp = wdir / "jobs" / wid / "JOB_RECEIPT.json"
                if rp.is_file():
                    job_receipts[wid] = json.loads(rp.read_text(encoding="utf-8"))
            for wrec in wave_res.get("waves", ()):
                failed_ids.extend(wrec.get("failed_workers", ()))
    elif pending_s50:
        if not s50_gate["unlocked"]:
            # Scientifically gated Stage 4 S50 jobs stay at 0.0 GPU s cleanly without blocking campaign completion
            gated_s50_ids = [j.job_id for j in pending_s50]
        else:
            deferred_ids.extend(j.job_id for j in pending_s50 if j.job_id not in deferred_ids)

    gated_s50_set = set(gated_s50_ids)
    pending_all = [j.job_id for j in all_sjobs if j.job_id not in completed_ids and j.job_id not in gated_s50_set]
    next_id = pending_all[0] if pending_all else None
    _atomic_write_json(
        wdir / "PENDING_QUEUE.json",
        {
            "schema": "elendril-x3-pending-queue-v1",
            "pending_job_ids": pending_all,
            "deferred_job_ids": deferred_ids,
            "gated_s50_job_ids": gated_s50_ids,
            "failed_job_ids": sorted(set(failed_ids)),
            "next_resumable_job_id": next_id,
            "stopped_reason": stopped_reason,
        },
    )
    _atomic_write_json(wdir / "JOB_RECEIPTS.json", {"receipts_by_id": job_receipts})

    # Complete-cohort Confirmation Lock: NEVER opens confirm if ANY confirmation-eligible job is unfinished
    conf_req_ids = [str(j["job_id"]) for j in raw_jobs if bool(j.get("confirmation_eligible", True))]
    conf_lock = evaluate_confirmation_lock(
        required_job_ids=conf_req_ids,
        verified_completed_ids=sorted(completed_ids),
        confirm_battery={"battery_sha256": digest(splits_map["confirm"])},
        lock_out_path=wdir / "CONFIRMATION_LOCK.json",
    )

    stage5_summary: dict[str, Any] = {"status": "LOCKED_PENDING_COHORT"}
    if conf_lock["unlocked"] and not failed_ids:
        eval_dev = (
            "cuda:0"
            if (not canary and devices and any(str(d).startswith("cuda") for d in devices) and torch.cuda.is_available())
            else "cpu"
        )
        stage5_summary = {
            "status": "COMPLETE",
            **_run_stage5_sealed_evaluations(
                wdir=wdir,
                completed_ids=completed_ids,
                job_receipts=job_receipts,
                jobs_by_id=jobs_by_id,
                splits_map=splits_map,
                eval_device=eval_dev,
                canary=bool(canary),
            ),
        }

    # Paired bootstrap summary using Palantiri primitives across completed treatment/control arms
    # Deduplicate (arm_id, init_seed, order_seed) deterministically across stages (e.g. S2 vs S3 sharing P-R-Iter D3-Hybrid)
    arm_records: list[dict[str, Any]] = []
    seen_arm_keys: set[tuple[str, int, int]] = set()
    for jid in sorted(job_receipts):
        r = job_receipts[jid]
        if "dev_macro_mean" not in r:
            continue
        arm_id = f"{r['architecture']}_{r['objective']}_{r['scale']}"
        i_seed = int(r["init_seed"])
        o_seed = int(r.get("order_seed", i_seed + 1000))
        key = (arm_id, i_seed, o_seed)
        if key in seen_arm_keys:
            continue
        seen_arm_keys.add(key)
        arm_records.append(
            {
                "arm_id": arm_id,
                "init_seed": i_seed,
                "order_seed": o_seed,
                "value": float(r.get("dev_macro_mean", 0.0)),
            }
        )
    paired_summary: dict[str, Any] = {}
    s20_scale_tag = "tiny" if canary else "S20"
    contrast_specs: list[tuple[str, str, str]] = [
        (cid, f"{ca}_{obj}_{s20_scale_tag}", f"{ta}_{obj}_{s20_scale_tag}")
        for cid, ca, obj, ta in X3_PAIRED_CONTRASTS
    ]
    if not canary:
        contrast_specs.extend(
            [
                ("S50_P-R-Iter_vs_P", "P_D3-Hybrid_S50", "P-R-Iter_D3-Hybrid_S50"),
                ("S50_P-R-Iter_vs_Base", "Base_D3-Hybrid_S50", "P-R-Iter_D3-Hybrid_S50"),
            ]
        )
    for cid, ctrl_key, trt_key in contrast_specs:
        ctrl_vals = {r["init_seed"]: r["value"] for r in arm_records if r["arm_id"] == ctrl_key}
        trt_vals = {r["init_seed"]: r["value"] for r in arm_records if r["arm_id"] == trt_key}
        shared_seeds = sorted(set(ctrl_vals) & set(trt_vals))
        if shared_seeds:
            c_vec = [ctrl_vals[s] for s in shared_seeds]
            t_vec = [trt_vals[s] for s in shared_seeds]
            paired_summary[cid] = bootstrap_paired_families(c_vec, t_vec, resamples=500, seed=1801)
            if cid == "P-R-Iter_vs_Fixed":
                obs = build_paired_seed_observations(
                    arm_records, contrast_id=cid, control_arm=ctrl_key, treatment_arm=trt_key, arm_key="arm_id"
                )
                paired_summary["first_page_report"] = build_analyst_first_page_report(
                    title="ELENDRIL V8-X3 Cognitive Summary",
                    campaign_id="ELENDRIL-V8-X3",
                    source_commit=source_commit,
                    decision_summary="CPU_OR_PARTIAL" if pending_all else "COHORT_COMPLETE",
                    control_arm=ctrl_key,
                    treatment_arm=trt_key,
                    paired_seed_bundle=obs,
                )

    campaign_status = "COMPLETE" if (not pending_all and not failed_ids) else "PARTIAL"
    summary_doc = {
        "schema": "elendril-x3-campaign-summary-v1",
        "status": campaign_status,
        "source_commit": str(source_commit),
        "completed_count": len(completed_ids),
        "total_jobs": len(all_sjobs),
        "pending_job_ids": pending_all,
        "blocked_missing_parent_job_ids": sorted(set(blocked_ids)),
        "gated_s50_job_ids": gated_s50_ids,
        "failed_job_ids": sorted(set(failed_ids)),
        "next_resumable_job_id": next_id,
        "s50_scaling_gate": s50_gate,
        "confirmation_lock": conf_lock,
        "stage5_summary": stage5_summary,
        "sampler_audit_ratio": sampler_audit["base_world_diversity_ratio_d1h_over_d2"],
        "paired_summary": paired_summary,
    }
    _atomic_write_json(wdir / "X3_CAMPAIGN_SUMMARY.json", summary_doc)
    (wdir / "RESULTS.md").write_text(
        "\n".join(
            [
                f"# ELENDRIL V8-X3 Results ({campaign_status})",
                "",
                f"- Completed: `{len(completed_ids)}/{len(all_sjobs)}` jobs (`gated_s50={len(gated_s50_ids)}`, `next_resumable={next_id}`)",
                f"- S50 Gate: `{s50_gate['status']}` (`unlocked={s50_gate['unlocked']}`)",
                f"- Confirmation Lock: `{conf_lock['status']}` (`pending={len(conf_lock['pending_confirm_jobs'])}`)",
                f"- Stage 5 Sealed Confirm / OOD / 5-Modes: `{stage5_summary.get('status')}` (`confirm_jobs={stage5_summary.get('sealed_confirm_evaluated_jobs', 0)}`)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    full_files = _link_latest_full_files(wdir)
    analysis_files = {k: v for k, v in full_files.items() if "/c/" not in k and "/checkpoints/" not in k}
    summary_files = {k: v for k, v in analysis_files.items() if "/" not in k}
    full_only_bytes = sum(p.stat().st_size for k, p in full_files.items() if k not in analysis_files and p.is_file())
    try:
        free_export_bytes = shutil.disk_usage(edir).free
    except OSError:
        free_export_bytes = 0
    analysis_bytes = sum(p.stat().st_size for p in analysis_files.values() if p.is_file())
    if free_export_bytes <= analysis_bytes:
        summary_doc["export_error"] = "DISK_FULL_INSUFFICIENT_EXPORT_SPACE"
        summary_doc["archives"] = {}
        summary_doc["archive_sha256"] = {}
        summary_doc["analysis_archive_verified"] = False
        _atomic_write_json(wdir / "X3_CAMPAIGN_SUMMARY.json", summary_doc)
        return summary_doc
    can_write_full = free_export_bytes > (full_only_bytes + 2 * 1024**3)
    archives = export_session_archives(
        wdir,
        edir,
        prefix="ELENDRIL_V8_X3",
        analysis_files=analysis_files,
        summary_files=summary_files,
        evidence_files=analysis_files,
        full_files=full_files,
        status=campaign_status,
        source_commit=source_commit,
        protocol_sha256=ledger["ledger_sha256"],
        include_v6_manifest=True,
        write_full_zip=can_write_full,
    )
    an_path = archives["archives"]["analysis"]
    an_ver = verify_session_archive(an_path, expected_source_commit=source_commit)
    summary_doc["archives"] = archives["archives"]
    summary_doc["archive_sha256"] = archives["archive_sha256"]
    summary_doc["analysis_archive_verified"] = bool(an_ver.get("verified"))
    _atomic_write_json(edir / "X3_CAMPAIGN_SUMMARY.json", summary_doc)

    if mirror_dir is not None:
        arc_paths = archives["archives"]
        mirror_targets = [
            Path(arc_paths["analysis"]),
            Path(arc_paths["analysis"] + ".sha256"),
            Path(arc_paths["summary"]),
            Path(arc_paths["evidence"]),
            edir / "X3_CAMPAIGN_SUMMARY.json",
            wdir / "RESULTS.md",
        ]
        if arc_paths.get("full"):
            mirror_targets.append(Path(arc_paths["full"]))
        summary_doc["mirrored_files"] = _mirror_artifacts_to_dir(mirror_targets, mirror_dir)

    return summary_doc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ELENDRIL V8-X3 Cognitive Research Program CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_smk = sub.add_parser("u0000-smoke", help="Execute u0000 init+train+eval+restore across all 9 architectures.")
    p_smk.add_argument("--work-dir", required=True)
    p_smk.add_argument("--source", required=True)
    p_smk.add_argument("--include-s50", action="store_true")

    p_inv = sub.add_parser("inventory", help="Print preregistered 57-job X3 multi-stage ledger JSON.")
    p_inv.add_argument("--source", required=True)

    p_job = sub.add_parser("run-job", help="Run a single X3 registered job JSON.")
    p_job.add_argument("--job-json", required=True)
    p_job.add_argument("--work-root", required=True)
    p_job.add_argument("--device", default="cpu")
    p_job.add_argument("--canary", action="store_true")
    p_job.add_argument("--updates-override", type=int, default=None)

    p_camp = sub.add_parser("run-campaign", help="Run multi-GPU X3 campaign.")
    p_camp.add_argument("--source", required=True)
    p_camp.add_argument("--work-dir", required=True)
    p_camp.add_argument("--export-dir", required=True)
    p_camp.add_argument("--devices", default="0,1")
    p_camp.add_argument("--session-minutes", type=float, default=660.0)
    p_camp.add_argument("--canary", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "u0000-smoke":
        res = run_all_architectures_u0000_smoke(
            args.work_dir, source_commit=args.source, include_s50_construction=bool(args.include_s50)
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0 if res["all_passed"] else 1
    if args.command == "inventory":
        print(json.dumps(build_x3_program_ledger(source_commit=args.source), indent=2, sort_keys=True))
        return 0
    if args.command == "run-job":
        job = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
        rec = run_x3_single_job(
            job=job,
            work_root=args.work_root,
            device=args.device,
            canary=bool(args.canary),
            updates_override=args.updates_override,
        )
        print(json.dumps({"job_id": rec["job_id"], "status": rec["status"], "dev_macro_mean": rec.get("dev_macro_mean")}, indent=2))
        return 0
    if args.command == "run-campaign":
        devs = ["cpu"] if args.canary else [f"cuda:{d.strip()}" for d in args.devices.split(",") if d.strip()]
        res = run_x3_campaign(
            work_dir=args.work_dir,
            export_dir=args.export_dir,
            source_commit=args.source,
            devices=devs,
            session_minutes=args.session_minutes,
            canary=bool(args.canary),
        )
        print(json.dumps({"status": res["status"], "completed_count": res["completed_count"], "total_jobs": res["total_jobs"]}, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PublicOracleTypedDecoder",
    "X3CognitiveKernel",
    "X3CognitivePolicy",
    "X3_BACKEND_ID",
    "X3_CANARY_ARCHITECTURES",
    "X3_DEFAULT_UPDATES",
    "X3_SEEDS",
    "build_elendril_x3_module",
    "build_r0_r7_program_ledger",
    "build_x3_program_ledger",
    "evaluate_confirmation_lock",
    "evaluate_s50_scaling_gate",
    "evaluate_x3_batched",
    "main",
    "reconcile_r0_run3_obligations",
    "register_planthir_backends",
    "run_all_architectures_u0000_smoke",
    "run_x3_campaign",
    "run_x3_single_job",
    "verify_perfect_public_decoder_through_evaluator",
    "x3_dataset_factory",
    "x3_kernel_factory",
    "x3_loss_fn",
    "x3_model_factory",
    "x3_policy_factory",
]
