"""X1 EnvironmentKernel, ObservablePolicy, batched GPU evaluator, lesions, depth scaling, and H01 check."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from anra_v5.elendril.codec import digest
from anra_v5.elendril.representation import UTF8, SpecialToken
from anra_v5.elendril.training import batch_tensors
from anra_v5.elendril.v8.contracts import ACTION_19_LABELS, POLICY_MODES, SYMBOLS, PolicyProposal
from anra_v5.elendril.v8.policies import V8BytePolicy, _apply_mode_to_distribution, _index_to_action
from anra_v5.planthir.causal import stratify_intervention_worlds, summarize_diagnostic_coverage
from anra_v5.planthir.contracts import SubgoalProbeSpec, UniversalAction, UniversalObservation

from .contracts import X1EvaluationOutcome, X1ModelSpec, X1_OBSERVATION_VERSION, X1_U_ARCHITECTURES

_R_MECHANISTIC_FAMILIES = ("X02", "X03", "X05", "X06", "X07", "X08", "X09")
from .data import (
    PublicFactX1,
    render_public_x1,
    resolve_public_x1_belief,
    verify_x1_episode,
    x1_teacher_decide,
)


@dataclass(frozen=True, slots=True)
class X1KernelState:
    """Immutable public environment state shared across teacher, rollout, and verifier."""

    world: dict[str, Any]
    public_facts: tuple[PublicFactX1, ...]
    seen_facts: frozenset[PublicFactX1]
    query_avail: bool
    mem_avail: bool
    consumed_tools: frozenset[str]
    decision: int
    max_actions: int
    executed_actions: tuple[tuple[str, str | None], ...]
    terminal: bool = False


class X1S20EnvironmentKernel:
    """Truth-firewalled X1 EnvironmentKernel enforcing consumed tools and epistemic verification."""

    kernel_id = "elendril-x1-kernel-v1"

    def __init__(self, *, max_actions: int = 4, renderer_override: int | None = None) -> None:
        if type(max_actions) is not int or max_actions < 1:
            raise ValueError("max_actions must be a positive integer")
        self.max_actions = max_actions
        self.renderer_override = renderer_override

    def canonical_config(self) -> dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "max_actions": self.max_actions,
            "renderer_override": self.renderer_override,
        }

    def initial_state(self, world: Mapping[str, Any]) -> X1KernelState:
        facts = tuple(PublicFactX1(**r) for r in world["initial_facts"])
        tools = world["allowed_public_tools"]
        return X1KernelState(
            world=deepcopy(dict(world)),
            public_facts=facts,
            seen_facts=frozenset(facts),
            query_avail=bool(tools.get("query_available", False)),
            mem_avail=bool(tools.get("memory_available", False)),
            consumed_tools=frozenset(),
            decision=0,
            max_actions=self.max_actions,
            executed_actions=(),
            terminal=False,
        )

    def observe(self, state: X1KernelState) -> UniversalObservation:
        w = state.world
        rid = int(self.renderer_override) if self.renderer_override is not None else int(w.get("renderer_template", 0))
        budget_rem = max(1, state.max_actions - state.decision)
        text = render_public_x1(
            ordered_facts=state.public_facts,
            goal_subject=str(w["goal_subject"]),
            hops=int(w["hops"]),
            min_epoch=int(w.get("min_epoch", 1)),
            query_available=state.query_avail,
            memory_available=state.mem_avail,
            budget_remaining=budget_rem,
            renderer_id=rid,
            row_order_seed=int(w.get("row_order_seed", 0)),
        )
        obs_id = digest([X1_OBSERVATION_VERSION, state.decision, text, state.query_avail, state.mem_avail, budget_rem])[:16]
        return UniversalObservation(
            obs_id=obs_id,
            rendered_text=text,
            available_tools={"query": state.query_avail, "retrieve": state.mem_avail},
            consumed_tools=state.consumed_tools,
            decision_index=state.decision,
            budget_remaining=budget_rem,
            metadata={"hops": int(w["hops"]), "min_epoch": int(w.get("min_epoch", 1)), "renderer_id": rid},
        )

    def ideal_action(self, state: X1KernelState) -> tuple[str, str | None]:
        w = state.world
        return x1_teacher_decide(
            state.public_facts,
            str(w["goal_subject"]),
            int(w["hops"]),
            int(w.get("min_epoch", 1)),
            state.query_avail,
            state.mem_avail,
        )

    def step(self, state: X1KernelState, operation: str, argument: str | None) -> X1KernelState:
        if state.terminal:
            raise ValueError("cannot step terminal X1KernelState")
        facts = list(state.public_facts)
        seen = set(state.seen_facts)
        qa, ma = state.query_avail, state.mem_avail
        consumed = set(state.consumed_tools)
        executed = (*state.executed_actions, (operation, argument))
        next_dec = state.decision + 1
        terminal = next_dec >= state.max_actions

        if operation == "QUERY":
            if not qa:
                terminal = True
            else:
                consumed.add("query")
                qa = False
                for r in state.world["hidden"]["query_payload"]:
                    f = PublicFactX1(**r)
                    if f not in seen:
                        seen.add(f)
                        facts.append(f)
        elif operation == "RETRIEVE":
            if not ma:
                terminal = True
            else:
                consumed.add("retrieve")
                ma = False
                for r in state.world["hidden"]["memory_payload"]:
                    f = PublicFactX1(**r)
                    if f not in seen:
                        seen.add(f)
                        facts.append(f)
        else:
            terminal = True

        return X1KernelState(
            world=state.world,
            public_facts=tuple(facts),
            seen_facts=frozenset(seen),
            query_avail=qa,
            mem_avail=ma,
            consumed_tools=frozenset(consumed),
            decision=next_dec,
            max_actions=state.max_actions,
            executed_actions=executed,
            terminal=terminal,
        )

    def verify(self, world: Mapping[str, Any], actions: Sequence[tuple[str, str | None]]) -> dict[str, Any]:
        out = verify_x1_episode(world, actions, action_budget=self.max_actions)
        return {
            "correct": out.correct,
            "terminal_status": out.terminal_status,
            "charged_cost": out.charged_cost,
            "epistemic_supported": out.epistemic_supported,
            "lucky_unsupported_guess": out.lucky_unsupported_guess,
            "verifier_identity": out.verifier_identity,
        }

    def inject_oracle_information(self, world: Mapping[str, Any]) -> dict[str, Any]:
        w = deepcopy(dict(world))
        merged = list(w["initial_facts"])
        if w["allowed_public_tools"].get("query_available"):
            merged.extend(dict(r) for r in w["hidden"].get("query_payload", []))
        if w["allowed_public_tools"].get("memory_available"):
            merged.extend(dict(r) for r in w["hidden"].get("memory_payload", []))
        w["initial_facts"] = merged
        w["allowed_public_tools"] = {"query_available": False, "memory_available": False}
        w["hidden"]["query_payload"] = []
        w["hidden"]["memory_payload"] = []
        w["world_id"] = f"{world['world_id']}::oracle_injected"
        return w

    def decompose_subgoals(self, world: Mapping[str, Any]) -> tuple[SubgoalProbeSpec, ...]:
        hops = int(world.get("hops", 1))
        exp = world.get("hidden", {}).get("expected")
        if hops < 2 or exp is None:
            return ()
        ow = self.inject_oracle_information(world)
        goal = str(world["goal_subject"])
        min_ep = int(world.get("min_epoch", 1))
        st1, mid = resolve_public_x1_belief(ow["initial_facts"], goal, 1, min_ep)
        if st1 == "derived" and mid is not None:
            return ()  # hop 1 must yield intermediate entity, not symbol
        # Resolve directly using raw precedence map
        facts = [PublicFactX1(**r) for r in ow["initial_facts"] if int(r.get("epoch", 1)) >= min_ep]
        by_subj: dict[str, tuple[tuple[int, int], str | None]] = {}
        from .contracts import DEFAULT_X1_SOURCE_RANKS

        for f in facts:
            key = (f.epoch, DEFAULT_X1_SOURCE_RANKS[f.source])
            if f.subject not in by_subj or key > by_subj[f.subject][0]:
                by_subj[f.subject] = (key, f.value)
        if goal not in by_subj or by_subj[goal][1] is None:
            return ()
        mid_ent = str(by_subj[goal][1])
        h2_ok, h2_val = resolve_public_x1_belief(ow["initial_facts"], mid_ent, hops - 1, min_ep)
        if h2_ok != "derived" or h2_val != exp:
            return ()
        h2 = deepcopy(ow)
        h2["world_id"] = f"{world['world_id']}::hop2_bridge"
        h2["hops"] = hops - 1
        h2["goal_subject"] = mid_ent
        h2["hidden"]["expected"] = h2_val
        return (
            SubgoalProbeSpec(f"{world['world_id']}::sg_bridge", "hop_2_given_bridge", 2, h2, "ANSWER", h2_val, "X1 suffix bridge"),
        )


class X1S20ObservablePolicy:
    """ObservablePolicy and batched evaluator interface for X1S20 / tiny X1 models."""

    def __init__(
        self,
        model: Any,
        x1_spec: X1ModelSpec,
        *,
        model_identity: str,
        mode: str = "raw",
        copy_enabled_override: bool | None = None,
        lesion_mode: str | None = None,
        recurrent_steps_override: int | None = None,
    ) -> None:
        if mode not in POLICY_MODES:
            raise ValueError(f"unsupported policy mode: {mode!r}")
        self.model = model
        self.x1_spec = x1_spec
        self.v8_spec = None
        self.model_identity = model_identity
        self.mode = mode
        self.copy_enabled_override = copy_enabled_override
        self.lesion_mode = lesion_mode
        self.recurrent_steps_override = recurrent_steps_override
        self.policy_id = f"{model_identity}:{x1_spec.architecture}:{mode}"
        self._b0 = (
            V8BytePolicy(model, type("_S", (), {"architecture": "B0", "max_sequence_length": x1_spec.max_sequence_length})(), model_identity=model_identity, mode=mode)
            if x1_spec.architecture == "B0"
            else None
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _encode_prompt(self, text: str) -> list[int]:
        ids = [UTF8.bos_id, *UTF8.encode(text), UTF8.special(SpecialToken.ANSWER)]
        if len(ids) > self.x1_spec.max_sequence_length:
            raise ValueError(f"X1 prompt len {len(ids)} > max_sequence_length {self.x1_spec.max_sequence_length}")
        return ids

    def propose_batch(self, obs_list: Sequence[UniversalObservation]) -> list[UniversalAction]:
        if not obs_list:
            return []
        if self._b0 is not None:
            from anra_v5.elendril.v8.contracts import PublicInput

            pubs = [
                PublicInput(o.rendered_text, bool(o.available_tools.get("query", False)), bool(o.available_tools.get("retrieve", False)), max(1, o.budget_remaining))
                for o in obs_list
            ]
            props = self._b0.propose_batch(pubs)
            out_actions: list[UniversalAction] = []
            for (ex, rw), obs in zip(props, obs_list):
                dist = {lbl: (1.0 if (lbl == ex.argument or lbl == ex.operation) else 0.0) for lbl in ACTION_19_LABELS}
                out_actions.append(
                    UniversalAction(
                        operation=ex.operation,
                        argument=ex.argument,
                        log_probability=ex.joint_log_probability,
                        action_distribution=dist,
                        mode=self.mode,
                        raw_operation=rw.operation,
                        raw_argument=rw.argument,
                        telemetry={"model_calls": float(ex.model_calls), "evaluated_positions": float(ex.evaluated_positions)},
                    )
                )
            return out_actions

        encoded = [self._encode_prompt(o.rendered_text) for o in obs_list]
        samples = [(tuple(ids), (False,) * len(ids)) for ids in encoded]
        tokens, segments, _ = batch_tensors(samples, self.device)
        apos = torch.tensor([len(ids) - 1 for ids in encoded], device=self.device, dtype=torch.int64)
        modes = [(m, m.training) for m in self.model.modules()]
        self.model.eval()
        try:
            with torch.inference_mode():
                out = self.model.forward_action(
                    tokens,
                    segments,
                    apos,
                    copy_enabled_override=self.copy_enabled_override,
                    lesion_mode=self.lesion_mode,
                    recurrent_steps_override=self.recurrent_steps_override,
                )
                probs = out.log_probs_19.exp().cpu()
                gates = out.gate.cpu() if out.gate is not None else None
                ents = out.attention_entropy.cpu() if out.attention_entropy is not None else None
                cands = out.has_pointer_candidates.cpu() if out.has_pointer_candidates is not None else None
        finally:
            for m, was in modes:
                m.training = was

        results: list[UniversalAction] = []
        for idx, obs in enumerate(obs_list):
            qa = bool(obs.available_tools.get("query", False))
            ma = bool(obs.available_tools.get("retrieve", False))
            p_row = probs[idx]
            raw_idx, _ = _apply_mode_to_distribution(p_row, mode="raw", query_available=qa, memory_available=ma)
            raw_op, raw_arg, _ = _index_to_action(raw_idx, query_available=qa)
            if self.mode in ("raw", "grammar"):
                ex_idx = raw_idx
                ex_lp = math.log(max(float(p_row[raw_idx].item()), 1e-30))
            else:
                ex_idx, ex_lp = _apply_mode_to_distribution(p_row, mode=self.mode, query_available=qa, memory_available=ma)
            ex_op, ex_arg, _ = _index_to_action(ex_idx, query_available=qa)
            tel: dict[str, float] = {"model_calls": 1.0, "evaluated_positions": float(len(encoded[idx]))}
            if gates is not None:
                tel["pointer_gate"] = float(gates[idx].item())
            if ents is not None:
                tel["pointer_attention_entropy"] = float(ents[idx].item())
            if cands is not None:
                tel["has_pointer_candidates"] = 1.0 if bool(cands[idx].item()) else 0.0
            results.append(
                UniversalAction(
                    operation=ex_op,
                    argument=ex_arg,
                    log_probability=ex_lp,
                    action_distribution={lbl: float(p_row[j].item()) for j, lbl in enumerate(ACTION_19_LABELS)},
                    mode=self.mode,
                    raw_operation=raw_op,
                    raw_argument=raw_arg,
                    telemetry=tel,
                )
            )
        return results

    def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction:
        return self.propose_batch([obs])[0]

    def extract_representations(self, obs_batch: Sequence[UniversalObservation]) -> dict[str, torch.Tensor]:
        if not obs_batch:
            raise ValueError("obs_batch cannot be empty")
        encoded = [self._encode_prompt(o.rendered_text) for o in obs_batch]
        samples = [(tuple(ids), (False,) * len(ids)) for ids in encoded]
        tokens, segments, _ = batch_tensors(samples, self.device)
        apos = torch.tensor([len(ids) - 1 for ids in encoded], device=self.device, dtype=torch.int64)
        b_idx = torch.arange(len(encoded), device=self.device, dtype=torch.int64)
        modes = [(m, m.training) for m in self.model.modules()]
        self.model.eval()
        try:
            with torch.inference_mode():
                if self.x1_spec.architecture == "B0":
                    from v5_model.core import packed_layout

                    pos, mask = packed_layout(segments, torch_module=torch)
                    H = self.model.forward_hidden(tokens, pos, mask)
                    return {"decision_hidden": H[b_idx, apos].float().cpu()}
                out = self.model.forward_action(tokens, segments, apos)
                res: dict[str, torch.Tensor] = {
                    "decision_hidden": out.decision_hidden.cpu(),
                    "log_probs_19": out.log_probs_19.cpu(),
                }
                if out.gate is not None:
                    res["pointer_gate"] = out.gate.cpu()
                if out.attention_entropy is not None:
                    res["pointer_entropy"] = out.attention_entropy.cpu()
                return res
        finally:
            for m, was in modes:
                m.training = was


def classify_first_divergence_x1(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify the first divergence using Section 9.1's 6-category taxonomy."""

    saw_tool_acquired = False
    for st in steps:
        ex_op = str(st["executed_op"])
        ex_arg = st.get("executed_arg")
        id_op = str(st["ideal_op"])
        id_arg = st.get("ideal_arg")
        match = (ex_op == id_op) and (ex_op != "ANSWER" or ex_arg == id_arg)
        if not match:
            if int(st["step_index"]) == 0 and ex_op != id_op:
                cat = "1_wrong_initial_action" if id_op in ("QUERY", "RETRIEVE") else "1_unnecessary_initial_action"
            elif id_op in ("QUERY", "RETRIEVE") and ex_op in ("QUERY", "RETRIEVE"):
                cat = "2_inappropriate_or_repeated_tool"
            elif ex_op == "ANSWER" and id_op == "ANSWER":
                cat = "4_wrong_value_same_op"
            elif saw_tool_acquired:
                cat = "3_post_tool_answer_failure"
            else:
                cat = "5_stale_or_contradiction_failure"
            return {"decision": int(st["step_index"]), "category": cat, "executed": [ex_op, ex_arg], "ideal": [id_op, id_arg]}
        if ex_op in ("QUERY", "RETRIEVE"):
            saw_tool_acquired = True
    return {"decision": None, "category": "none"}


def evaluate_x1_worlds_batched(
    policy: X1S20ObservablePolicy,
    kernel: X1S20EnvironmentKernel,
    worlds: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Run lockstep batched X1 episode rollouts matching sequential kernel transitions."""

    if not worlds:
        return []
    chunk = min(batch_size, 16) if policy.x1_spec.architecture == "B0" else batch_size
    records: list[dict[str, Any]] = []

    for start in range(0, len(worlds), chunk):
        sub = worlds[start : start + chunk]
        states = [kernel.initial_state(w) for w in sub]
        traces: list[list[dict[str, Any]]] = [[] for _ in sub]
        for _dec in range(kernel.max_actions):
            open_idxs = [i for i, st in enumerate(states) if not st.terminal]
            if not open_idxs:
                break
            obs_list = [kernel.observe(states[i]) for i in open_idxs]
            actions = policy.propose_batch(obs_list)
            for idx, obs, act in zip(open_idxs, obs_list, actions):
                st = states[idx]
                id_op, id_arg = kernel.ideal_action(st)
                traces[idx].append(
                    {
                        "step_index": int(obs.decision_index),
                        "obs_id": obs.obs_id,
                        "public_text_sha256": sha256(obs.rendered_text.encode("utf-8")).hexdigest(),
                        "query_available": bool(obs.available_tools["query"]),
                        "memory_available": bool(obs.available_tools["retrieve"]),
                        "budget_remaining": int(obs.budget_remaining),
                        "raw_op": act.raw_operation,
                        "raw_arg": act.raw_argument,
                        "executed_op": act.operation,
                        "executed_arg": act.argument,
                        "ideal_op": id_op,
                        "ideal_arg": id_arg,
                        "log_probability": float(act.log_probability),
                        "telemetry": dict(act.telemetry),
                    }
                )
                states[idx] = kernel.step(st, act.operation, act.argument)

        for w, st, tr in zip(sub, states, traces):
            out = verify_x1_episode(w, st.executed_actions, action_budget=kernel.max_actions)
            records.append(
                {
                    "world_id": str(w["world_id"]),
                    "base_world_id": str(w["base_world_id"]),
                    "family": str(w["family"]),
                    "form": str(w["form"]),
                    "hops": int(w["hops"]),
                    "mode": policy.mode,
                    "correct": bool(out.correct),
                    "epistemic_supported": bool(out.epistemic_supported),
                    "lucky_unsupported_guess": bool(out.lucky_unsupported_guess),
                    "terminal_status": out.terminal_status,
                    "charged_cost": float(out.charged_cost),
                    "first_divergence": classify_first_divergence_x1(tr),
                    "cf_transition_class": w.get("cf_transition_class"),
                    "steps": tr,
                }
            )
    return records


def run_x1_lesion_and_intervention_battery(
    model: Any,
    x1_spec: X1ModelSpec,
    dev_views: Sequence[Mapping[str, Any]],
    *,
    model_identity: str,
    u_sweep_worlds: Sequence[Mapping[str, Any]] | None = None,
    stratify_sample: bool = True,
    max_sample_views: int = 48,
    target_families: Sequence[str] = _R_MECHANISTIC_FAMILIES,
    seed_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Run Section 9.2 lesion, renderer, and depth-extrapolation probes on a frozen X1 checkpoint."""

    kernel = X1S20EnvironmentKernel()
    base_pol = X1S20ObservablePolicy(model, x1_spec, model_identity=model_identity, mode="raw")
    sample = (
        stratify_intervention_worlds(dev_views, max_worlds=max_sample_views, target_families=target_families)
        if stratify_sample
        else list(dev_views[:max_sample_views])
    )
    cov = summarize_diagnostic_coverage(sample, target_families=target_families, seed_ids=seed_ids)
    base_recs = evaluate_x1_worlds_batched(base_pol, kernel, sample)
    base_acc = sum(1 for r in base_recs if r["correct"]) / max(len(base_recs), 1)

    report: dict[str, Any] = {
        "baseline_accuracy": float(base_acc),
        "sample_worlds": len(sample),
        "coverage": cov,
        "interventions": {},
    }

    # Pointer / relational lesions per architecture
    lesions: tuple[str, ...]
    if x1_spec.architecture in ("R", "R-noedge"):
        lesions = ("zero_r", "permute_edges", "sham", "disable_copy", "permute_val_bins") if x1_spec.architecture == "R" else ("zero_r", "sham", "disable_copy")
    elif x1_spec.architecture in ("P", "P-capR"):
        lesions = ("disable_copy", "permute_val_bins")
    else:
        lesions = ()

    for lmode in lesions:
        l_pol = X1S20ObservablePolicy(model, x1_spec, model_identity=f"{model_identity}:{lmode}", mode="raw", lesion_mode=lmode)
        l_recs = evaluate_x1_worlds_batched(l_pol, kernel, sample)
        l_acc = sum(1 for r in l_recs if r["correct"]) / max(len(l_recs), 1)
        gained = 0
        lost = 0
        unchanged_ok = 0
        unchanged_bad = 0
        first_op_changed = 0
        first_act_changed = 0
        changed_ids: list[str] = []
        for br, lr in zip(base_recs, l_recs):
            b_ok = bool(br["correct"])
            l_ok = bool(lr["correct"])
            if l_ok and not b_ok:
                gained += 1
            elif b_ok and not l_ok:
                lost += 1
            elif b_ok and l_ok:
                unchanged_ok += 1
            else:
                unchanged_bad += 1
            bs0 = br["steps"][0] if br.get("steps") else {}
            ls0 = lr["steps"][0] if lr.get("steps") else {}
            b_fa = (bs0.get("executed_op"), bs0.get("executed_arg"))
            l_fa = (ls0.get("executed_op"), ls0.get("executed_arg"))
            if b_fa[0] != l_fa[0]:
                first_op_changed += 1
            if b_fa != l_fa:
                first_act_changed += 1
            if (b_ok != l_ok or b_fa != l_fa) and len(changed_ids) < 32:
                changed_ids.append(str(br["world_id"]))
        paired_changes = {
            "total_episodes": len(sample),
            "gained_correct_count": gained,
            "lost_correct_count": lost,
            "unchanged_correct_count": unchanged_ok,
            "unchanged_wrong_count": unchanged_bad,
            "net_correct_delta": gained - lost,
            "first_operation_changed_count": first_op_changed,
            "first_action_changed_count": first_act_changed,
            "changed_world_ids": changed_ids,
        }
        delta = float(l_acc - base_acc)
        if not cov["coverage_complete"]:
            mech_status = "MECHANISM_UNRESOLVED_MISSING_TARGET_FAMILIES"
        elif abs(delta) <= 1e-6 and first_act_changed == 0:
            mech_status = "NO_NET_EFFECT_ON_COVERED_FAMILIES"
        else:
            mech_status = "ACTIVE_INTERVENTION_EFFECT"
        report["interventions"][lmode] = {
            "accuracy": float(l_acc),
            "delta_vs_baseline": delta,
            "coverage": cov,
            "paired_changes": paired_changes,
            "mechanism_status": mech_status,
        }

    # Test-time U depth sweep (1, 2, 4, 8 iterations) per Section 8.3
    if x1_spec.architecture in X1_U_ARCHITECTURES:
        target_u = list(u_sweep_worlds or sample)
        u_grid: dict[str, float] = {}
        for k in (1, 2, 4, 8):
            u_pol = X1S20ObservablePolicy(model, x1_spec, model_identity=f"{model_identity}:u{k}", mode="raw", recurrent_steps_override=k)
            u_recs = evaluate_x1_worlds_batched(u_pol, kernel, target_u)
            u_grid[str(k)] = float(sum(1 for r in u_recs if r["correct"]) / max(len(u_recs), 1))
        report["u_depth_sweep"] = u_grid

    return report


def run_h01_frozen_intervention(checkpoint_paths: Sequence[Any] | None = None) -> dict[str, Any]:
    """Emit H01 historical frozen-weight status without inventing or re-labeling non-S20 checkpoints."""

    if not checkpoint_paths:
        return {
            "hypothesis": "H01",
            "status": "HISTORICAL_WEIGHTS_NOT_AVAILABLE",
            "detail": "Six V8 B0/S20 D0/D1 checkpoints were not mounted locally; original archive audit preserved and X1 core proceeds.",
        }
    return {
        "hypothesis": "H01",
        "status": "MOUNTED_CHECKPOINTS_DETECTED_USE_V8_FROZEN_ADAPTER",
        "checkpoint_count": len(checkpoint_paths),
    }


__all__ = [
    "X1KernelState",
    "X1S20EnvironmentKernel",
    "X1S20ObservablePolicy",
    "classify_first_divergence_x1",
    "evaluate_x1_worlds_batched",
    "run_h01_frozen_intervention",
    "run_x1_lesion_and_intervention_battery",
]
