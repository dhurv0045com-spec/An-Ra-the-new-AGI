"""V8/Elendril adapter implementing Planthir EnvironmentKernel and ObservablePolicy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from anra_v5.elendril.codec import digest
from anra_v5.elendril.representation import UTF8, SpecialToken
from anra_v5.elendril.training import batch_tensors
from anra_v5.elendril.v8.contracts import (
    ACTION_19_LABELS,
    POLICY_MODES,
    SYMBOLS,
    PublicInput,
    V8ModelSpec,
)
from anra_v5.elendril.v8.data import render_public_v8
from anra_v5.elendril.v8.policies import V8BytePolicy, _apply_mode_to_distribution, _index_to_action
from anra_v5.elendril.v8.teacher import PublicFact, resolve_public_belief, teacher_decide
from anra_v5.elendril.v8.verifier import verify_episode_trace
from v5_model.core import packed_layout

from ..contracts import SubgoalProbeSpec, UniversalAction, UniversalObservation


@dataclass(frozen=True, slots=True)
class V8KernelState:
    """Immutable state snapshot inside V8EnvironmentKernel."""

    world: dict[str, Any]
    ordered_facts: tuple[PublicFact, ...]
    seen_facts: frozenset[PublicFact]
    query_avail: bool
    mem_avail: bool
    consumed_tools: frozenset[str]
    decision: int
    max_actions: int
    executed_actions: tuple[tuple[str, str | None], ...]
    terminal: bool = False


class V8EnvironmentKernel:
    """Unified V8 state machine shared by teacher, interactive rollout, and verifier."""

    def __init__(
        self,
        *,
        render_fn: Callable[..., str] = render_public_v8,
        max_actions: int = 4,
        enforce_tool_consumption: bool = True,
    ) -> None:
        self.render_fn = render_fn
        self.max_actions = max_actions
        self.enforce_tool_consumption = enforce_tool_consumption
        self.kernel_id = (
            "elendril-v8-kernel-v1"
            if enforce_tool_consumption
            else "elendril-v8-kernel-fault-unconsumed-tool"
        )

    def initial_state(self, world: Mapping[str, Any]) -> V8KernelState:
        facts = tuple(PublicFact(**row) for row in world["initial_facts"])
        q_avail = bool(world["allowed_public_tools"]["query_available"])
        m_avail = bool(world["allowed_public_tools"]["memory_available"])
        return V8KernelState(
            world=deepcopy(dict(world)),
            ordered_facts=facts,
            seen_facts=frozenset(facts),
            query_avail=q_avail,
            mem_avail=m_avail,
            consumed_tools=frozenset(),
            decision=0,
            max_actions=self.max_actions,
            executed_actions=(),
            terminal=False,
        )

    def observe(self, state: V8KernelState) -> UniversalObservation:
        w = state.world
        text = self.render_fn(
            ordered_facts=state.ordered_facts,
            goal_subject=str(w["goal_subject"]),
            hops=int(w["hops"]),
            query_available=state.query_avail,
            memory_available=state.mem_avail,
            template_id=int(w["renderer_template"]),
            alias_prefix=str(w["alias_prefix"]),
            order_seed=int(w["row_order_seed"]),
            decision=state.decision,
        )
        obs_id = digest(["v8-obs", state.decision, text, state.query_avail, state.mem_avail])[:16]
        return UniversalObservation(
            obs_id=obs_id,
            rendered_text=text,
            available_tools={"query": state.query_avail, "retrieve": state.mem_avail},
            consumed_tools=state.consumed_tools,
            decision_index=state.decision,
            budget_remaining=max(1, state.max_actions - state.decision),
            metadata={
                "hops": int(w["hops"]),
                "template_id": int(w["renderer_template"]),
                "alias_prefix": str(w["alias_prefix"]),
            },
        )

    def ideal_action(self, state: V8KernelState) -> tuple[str, str | None]:
        w = state.world
        return teacher_decide(
            state.ordered_facts,
            str(w["goal_subject"]),
            int(w["hops"]),
            state.query_avail,
            state.mem_avail,
        )

    def step(self, state: V8KernelState, operation: str, argument: str | None) -> V8KernelState:
        if state.terminal:
            raise ValueError("cannot step a terminal V8KernelState")
        ordered = list(state.ordered_facts)
        seen = set(state.seen_facts)
        q_avail = state.query_avail
        m_avail = state.mem_avail
        consumed = set(state.consumed_tools)
        executed = (*state.executed_actions, (operation, argument))
        next_decision = state.decision + 1
        terminal = next_decision >= state.max_actions

        if operation == "QUERY":
            if not q_avail:
                terminal = True
            else:
                consumed.add("query")
                for row in state.world["hidden"]["query_payload"]:
                    fact = PublicFact(**row)
                    if fact not in seen:
                        seen.add(fact)
                        ordered.append(fact)
                if self.enforce_tool_consumption:
                    q_avail = False
        elif operation == "RETRIEVE":
            if not m_avail:
                terminal = True
            else:
                consumed.add("retrieve")
                for row in state.world["hidden"]["memory_payload"]:
                    fact = PublicFact(**row)
                    if fact not in seen:
                        seen.add(fact)
                        ordered.append(fact)
                if self.enforce_tool_consumption:
                    m_avail = False
        elif operation in ("ANSWER", "ABSTAIN", "INVALID"):
            terminal = True
        else:
            terminal = True

        return V8KernelState(
            world=state.world,
            ordered_facts=tuple(ordered),
            seen_facts=frozenset(seen),
            query_avail=q_avail,
            mem_avail=m_avail,
            consumed_tools=frozenset(consumed),
            decision=next_decision,
            max_actions=state.max_actions,
            executed_actions=executed,
            terminal=terminal,
        )

    def verify(
        self, world: Mapping[str, Any], actions: Sequence[tuple[str, str | None]]
    ) -> dict[str, Any]:
        outcome = verify_episode_trace(dict(world), list(actions), action_budget=self.max_actions)
        return {
            "correct": outcome.correct,
            "terminal_status": outcome.terminal_status,
            "charged_cost": outcome.charged_cost,
            "verifier_identity": outcome.verifier_identity,
        }

    def inject_oracle_information(self, world: Mapping[str, Any]) -> dict[str, Any]:
        """Create a matched counterfactual world with hidden payloads already visible."""

        cloned = deepcopy(dict(world))
        seen = {PublicFact(**r) for r in cloned["initial_facts"]}
        merged = list(cloned["initial_facts"])
        if cloned["allowed_public_tools"]["query_available"]:
            for row in cloned["hidden"]["query_payload"]:
                f = PublicFact(**row)
                if f not in seen:
                    seen.add(f)
                    merged.append(dict(row))
        if cloned["allowed_public_tools"]["memory_available"]:
            for row in cloned["hidden"]["memory_payload"]:
                f = PublicFact(**row)
                if f not in seen:
                    seen.add(f)
                    merged.append(dict(row))
        cloned["initial_facts"] = merged
        cloned["allowed_public_tools"] = {"query_available": False, "memory_available": False}
        cloned["hidden"]["query_payload"] = []
        cloned["hidden"]["memory_payload"] = []
        cloned["world_id"] = f"{world['world_id']}::oracle_injected"
        return cloned

    def decompose_subgoals(self, world: Mapping[str, Any]) -> tuple[SubgoalProbeSpec, ...]:
        """Decompose a multi-hop V8 world into intermediate single-hop causal probes."""

        hops = int(world["hops"])
        expected_final = world["hidden"]["expected"]
        if hops < 2 or expected_final is None:
            return ()

        oracle_world = self.inject_oracle_information(world)
        all_facts = tuple(PublicFact(**row) for row in oracle_world["initial_facts"])
        goal = str(world["goal_subject"])

        status1, mid = resolve_public_belief(all_facts, goal)
        if status1 != "derived" or mid is None:
            return ()
        status2, end_val = resolve_public_belief(all_facts, mid)
        if status2 != "derived" or end_val != expected_final:
            return ()

        hop1_world = deepcopy(oracle_world)
        hop1_world["world_id"] = f"{world['world_id']}::hop1"
        hop1_world["hops"] = 1
        hop1_world["goal_subject"] = goal
        hop1_world["hidden"]["expected"] = mid

        hop2_world = deepcopy(oracle_world)
        hop2_world["world_id"] = f"{world['world_id']}::hop2_bridge"
        hop2_world["hops"] = 1
        hop2_world["goal_subject"] = mid
        hop2_world["hidden"]["expected"] = end_val

        return (
            SubgoalProbeSpec(
                subgoal_id=f"{world['world_id']}::subgoal_hop1",
                stage_name="hop_1",
                hop_index=1,
                modified_world=hop1_world,
                expected_operation="ANSWER",
                expected_argument=mid,
                description=f"Resolve intermediate hop 1: {goal} -> {mid}",
            ),
            SubgoalProbeSpec(
                subgoal_id=f"{world['world_id']}::subgoal_hop2",
                stage_name="hop_2_given_bridge",
                hop_index=2,
                modified_world=hop2_world,
                expected_operation="ANSWER",
                expected_argument=end_val,
                description=f"Resolve hop 2 given bridge: {mid} -> {end_val}",
            ),
        )


def _distribution_dict(probs_19: torch.Tensor) -> dict[str, float]:
    return {label: float(probs_19[i].item()) for i, label in enumerate(ACTION_19_LABELS)}


class V8ObservablePolicy:
    """Planthir ObservablePolicy adapter for V8 B0, J, F, P, F-cap, and U models."""

    def __init__(
        self,
        model: Any,
        v8_spec: V8ModelSpec,
        *,
        model_identity: str,
        mode: str = "raw",
        copy_enabled_override: bool | None = None,
    ) -> None:
        if mode not in POLICY_MODES:
            raise ValueError(f"unsupported policy mode: {mode!r}")
        self.model = model
        self.v8_spec = v8_spec
        self.model_identity = model_identity
        self.mode = mode
        self.copy_enabled_override = copy_enabled_override
        self.policy_id = f"{model_identity}:{v8_spec.architecture}:{mode}"
        self._b0_policy = (
            V8BytePolicy(model, v8_spec, model_identity=model_identity, mode=mode)
            if v8_spec.architecture == "B0"
            else None
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _encode_prompt(self, text: str) -> list[int]:
        ids = [UTF8.bos_id, *UTF8.encode(text), UTF8.special(SpecialToken.ANSWER)]
        if len(ids) > self.v8_spec.max_sequence_length:
            raise ValueError(
                f"observation length {len(ids)} exceeds max_sequence_length {self.v8_spec.max_sequence_length}"
            )
        return ids

    def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction:
        q_avail = bool(obs.available_tools.get("query", False))
        m_avail = bool(obs.available_tools.get("retrieve", False))
        prompt_ids = self._encode_prompt(obs.rendered_text)

        if self._b0_policy is not None:
            pub = PublicInput(
                text_utf8=obs.rendered_text,
                query_available=q_avail,
                memory_available=m_avail,
                action_budget=max(1, obs.budget_remaining),
            )
            exec_prop, raw_prop = self._b0_policy.propose(pub)
            dist0 = self._b0_policy.backend.next_distribution(prompt_ids).cpu()
            label_probs: dict[str, float] = {}
            char_map = {sym: sym for sym in SYMBOLS}
            char_map["QUERY"] = "Q"
            char_map["RETRIEVE"] = "R"
            char_map["ABSTAIN"] = "!"
            for label in ACTION_19_LABELS:
                tok = UTF8.encode(char_map[label])[0]
                label_probs[label] = float(dist0[tok].item())
            total = sum(label_probs.values())
            if total > 0.0:
                label_probs = {k: v / total for k, v in label_probs.items()}
            return UniversalAction(
                operation=exec_prop.operation,
                argument=exec_prop.argument,
                log_probability=exec_prop.joint_log_probability,
                action_distribution=label_probs,
                mode=self.mode,
                raw_operation=raw_prop.operation,
                raw_argument=raw_prop.argument,
                telemetry={"model_calls": float(exec_prop.model_calls), "evaluated_positions": float(exec_prop.evaluated_positions)},
            )

        modes = [(m, m.training) for m in self.model.modules()]
        self.model.eval()
        try:
            with torch.inference_mode():
                tokens = torch.tensor([prompt_ids], device=self.device, dtype=torch.int64)
                segments = torch.zeros_like(tokens)
                action_pos = torch.tensor([len(prompt_ids) - 1], device=self.device, dtype=torch.int64)
                out = self.model.forward_action(
                    tokens,
                    segments,
                    action_pos,
                    copy_enabled_override=self.copy_enabled_override,
                )
                probs_19 = out.log_probs_19[0].exp().cpu()
                telemetry: dict[str, float] = {
                    "model_calls": 1.0,
                    "evaluated_positions": float(len(prompt_ids)),
                }
                if out.gate is not None:
                    telemetry["pointer_gate"] = float(out.gate[0].item())
                if out.attention_entropy is not None:
                    telemetry["pointer_attention_entropy"] = float(out.attention_entropy[0].item())
                if out.has_pointer_candidates is not None:
                    telemetry["has_pointer_candidates"] = 1.0 if bool(out.has_pointer_candidates[0].item()) else 0.0
                if self.v8_spec.architecture == "U":
                    telemetry.update(self._recurrent_telemetry(tokens, segments, action_pos))
        finally:
            for m, was in modes:
                m.training = was

        raw_idx, _ = _apply_mode_to_distribution(
            probs_19,
            mode="raw",
            query_available=q_avail,
            memory_available=m_avail,
        )
        raw_op, raw_arg, _ = _index_to_action(raw_idx, query_available=q_avail)
        if self.mode in ("raw", "grammar"):
            exec_idx = raw_idx
            exec_logp = math.log(max(float(probs_19[raw_idx].item()), 1e-30))
        else:
            exec_idx, exec_logp = _apply_mode_to_distribution(
                probs_19,
                mode=self.mode,
                query_available=q_avail,
                memory_available=m_avail,
            )
        exec_op, exec_arg, _ = _index_to_action(exec_idx, query_available=q_avail)
        return UniversalAction(
            operation=exec_op,
            argument=exec_arg,
            log_probability=exec_logp,
            action_distribution=_distribution_dict(probs_19),
            mode=self.mode,
            raw_operation=raw_op,
            raw_argument=raw_arg,
            telemetry=telemetry,
        )

    def _recurrent_telemetry(
        self,
        tokens: torch.Tensor,
        segments: torch.Tensor,
        action_pos: torch.Tensor,
    ) -> dict[str, float]:
        H = self.model.forward_hidden_only(tokens, segments)
        b_idx = torch.arange(tokens.shape[0], device=tokens.device, dtype=torch.int64)
        z_state = H[b_idx, action_pos].float()
        L = tokens.shape[1]
        pos_grid = torch.arange(L, device=tokens.device).unsqueeze(0)
        prefix_mask = (pos_grid <= action_pos.unsqueeze(1)) & (segments >= 0)
        k_all = self.model.u_k(H.float())
        v_all = self.model.u_v(H.float())
        scale = math.sqrt(self.v8_spec.recurrent_dim)
        deltas: list[float] = []
        for _ in range(self.v8_spec.recurrent_steps):
            q_t = self.model.u_q(z_state)
            attn_scores = torch.bmm(q_t.unsqueeze(1), k_all.transpose(1, 2)).squeeze(1) / scale
            attn_weights = F.softmax(attn_scores.masked_fill(~prefix_mask, float("-inf")), dim=-1)
            read_vec = torch.bmm(attn_weights.unsqueeze(1), v_all).squeeze(1)
            next_state = self.model.u_gru(self.model.u_out(read_vec), z_state)
            deltas.append(float((next_state - z_state).norm(dim=-1).mean().item()))
            z_state = next_state
        return {
            "recurrent_mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
            "recurrent_final_delta": deltas[-1] if deltas else 0.0,
        }

    def extract_representations(
        self, obs_batch: Sequence[UniversalObservation]
    ) -> dict[str, torch.Tensor]:
        if not obs_batch:
            raise ValueError("obs_batch cannot be empty")
        encoded = [self._encode_prompt(obs.rendered_text) for obs in obs_batch]
        samples = [(tuple(ids), (False,) * len(ids)) for ids in encoded]
        tokens, segments, _ = batch_tensors(samples, self.device)
        action_pos = torch.tensor([len(ids) - 1 for ids in encoded], device=self.device, dtype=torch.int64)
        b_idx = torch.arange(len(encoded), device=self.device, dtype=torch.int64)

        modes = [(m, m.training) for m in self.model.modules()]
        self.model.eval()
        try:
            with torch.inference_mode():
                if self.v8_spec.architecture == "B0":
                    positions, mask = packed_layout(segments, torch_module=torch)
                    H = self.model.forward_hidden(tokens, positions, mask)
                    decision_hidden = H[b_idx, action_pos].float().cpu()
                    return {"decision_hidden": decision_hidden}

                H = self.model.forward_hidden_only(tokens, segments)
                decision_hidden = H[b_idx, action_pos].float().cpu()
                out = self.model.forward_action(
                    tokens,
                    segments,
                    action_pos,
                    copy_enabled_override=self.copy_enabled_override,
                )
                result: dict[str, torch.Tensor] = {
                    "decision_hidden": decision_hidden,
                    "log_probs_19": out.log_probs_19.cpu(),
                }
                if out.gate is not None:
                    result["pointer_gate"] = out.gate.cpu()
                if out.attention_entropy is not None:
                    result["pointer_entropy"] = out.attention_entropy.cpu()
                return result
        finally:
            for m, was in modes:
                m.training = was


__all__ = ["V8EnvironmentKernel", "V8KernelState", "V8ObservablePolicy"]
