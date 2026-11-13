"""V8 actor policies operating strictly on PublicInput (truth-firewalled)."""

from __future__ import annotations

import math
from typing import Any

import torch

from anra_v5.elendril.backend import BackendIdentity, NeuralBackend
from anra_v5.elendril.representation import UTF8, SpecialToken
from anra_v5.elendril.v6_evaluation import parse_action

from .contracts import ACTION_19_LABELS, POLICY_MODES, SYMBOLS, PolicyProposal, PublicInput, V8ModelSpec

_CHANNEL = "request-current-status"


def _ensure_public_input(obs: PublicInput) -> None:
    if not isinstance(obs, PublicInput):
        raise TypeError(f"V8 actor strictly requires PublicInput, got {type(obs).__name__}")


def _apply_mode_to_distribution(
    probs_19: torch.Tensor,
    *,
    mode: str,
    query_available: bool,
    memory_available: bool,
) -> tuple[int, float]:
    """Select index in 0..18 and log-prob under the declared policy mode."""

    if mode not in POLICY_MODES:
        raise ValueError(f"unsupported policy mode: {mode!r}")
    p = probs_19.clone()
    if mode == "availability-mask":
        if not query_available:
            p[16] = 0.0
        if not memory_available:
            p[17] = 0.0
    elif mode == "direct-answer-only":
        p[16:] = 0.0

    total = float(p.sum().item())
    if total <= 0.0 or not math.isfinite(total):
        # Fallback if mask zeroed everything (impossible under availability-mask since A-P + ABSTAIN remain)
        idx = int(probs_19.argmax().item())
        prob = float(probs_19[idx].item())
        return idx, math.log(max(prob, 1e-30))
    p = p / total
    idx = int(p.argmax().item())
    prob = float(p[idx].item())
    return idx, math.log(max(prob, 1e-30))


def _index_to_action(idx: int, *, query_available: bool) -> tuple[str, str | None, str]:
    label = ACTION_19_LABELS[idx]
    if label in SYMBOLS:
        return "ANSWER", label, label
    if label == "QUERY":
        return "QUERY", (_CHANNEL if query_available else None), "Q"
    if label == "RETRIEVE":
        return "RETRIEVE", None, "R"
    if label == "ABSTAIN":
        return "ABSTAIN", None, "!"
    raise AssertionError("unreachable action index")


class V8TypedPolicy:
    """Actor policy for J, F, P, F-cap, and U models receiving PublicInput only."""

    origin = "model"

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
        if v8_spec.architecture == "B0":
            raise ValueError("use V8BytePolicy for B0")
        self.model = model
        self.v8_spec = v8_spec
        self.model_identity = model_identity
        self.mode = mode
        self.copy_enabled_override = copy_enabled_override

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def propose(self, obs: PublicInput) -> tuple[PolicyProposal, PolicyProposal]:
        """Return (executed_mode_proposal, unconstrained_raw_proposal)."""

        _ensure_public_input(obs)
        prompt_ids = [UTF8.bos_id, *UTF8.encode(obs.text_utf8), UTF8.special(SpecialToken.ANSWER)]
        if len(prompt_ids) > self.v8_spec.max_sequence_length:
            raise ValueError("PublicInput exceeds model max_sequence_length")

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
        finally:
            for m, was_training in modes:
                m.training = was_training

        raw_idx, raw_logp = _apply_mode_to_distribution(
            probs_19,
            mode="raw",
            query_available=obs.query_available,
            memory_available=obs.memory_available,
        )
        raw_op, raw_arg, raw_txt = _index_to_action(raw_idx, query_available=obs.query_available)
        raw_proposal = PolicyProposal(
            operation=raw_op,
            argument=raw_arg,
            joint_log_probability=raw_logp,
            raw_output=raw_txt,
            model_calls=1,
            evaluated_positions=len(prompt_ids),
            mode="raw",
            model_identity=self.model_identity,
        )

        if self.mode in ("raw", "grammar"):
            # Typed heads cannot emit malformed syntax; grammar == raw on 19-way space.
            exec_proposal = PolicyProposal(
                operation=raw_op,
                argument=raw_arg,
                joint_log_probability=raw_logp,
                raw_output=raw_txt,
                model_calls=1,
                evaluated_positions=len(prompt_ids),
                mode=self.mode,
                model_identity=self.model_identity,
            )
            return exec_proposal, raw_proposal

        exec_idx, exec_logp = _apply_mode_to_distribution(
            probs_19,
            mode=self.mode,
            query_available=obs.query_available,
            memory_available=obs.memory_available,
        )
        exec_op, exec_arg, exec_txt = _index_to_action(exec_idx, query_available=obs.query_available)
        exec_proposal = PolicyProposal(
            operation=exec_op,
            argument=exec_arg,
            joint_log_probability=exec_logp,
            raw_output=raw_txt,
            model_calls=1,
            evaluated_positions=len(prompt_ids),
            mode=self.mode,
            model_identity=self.model_identity,
        )
        return exec_proposal, raw_proposal


class V8BytePolicy:
    """Actor policy for B0 (unchanged byte policy) supporting raw/grammar/availability-mask/direct-answer-only."""

    origin = "model"

    def __init__(
        self,
        model: Any,
        v8_spec: V8ModelSpec,
        *,
        model_identity: str,
        mode: str = "raw",
        max_new_tokens: int = 4,
    ) -> None:
        if mode not in POLICY_MODES:
            raise ValueError(f"unsupported policy mode: {mode!r}")
        if v8_spec.architecture != "B0":
            raise ValueError("V8BytePolicy is for B0 only")
        self.model = model
        self.v8_spec = v8_spec
        self.model_identity = model_identity
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.backend = NeuralBackend(
            model,
            BackendIdentity(
                v8_spec.base_spec.sha256(),
                UTF8.identity,
                v8_spec.vocabulary_size,
                v8_spec.base_spec.context_length,
                checkpoint_sha256=model_identity if len(model_identity) == 64 else None,
            ),
            pad_id=UTF8.pad_id,
            bos_id=UTF8.bos_id,
            eos_id=UTF8.eos_id,
        )

    def propose(self, obs: PublicInput) -> tuple[PolicyProposal, PolicyProposal]:
        _ensure_public_input(obs)
        prompt_ids = [UTF8.bos_id, *UTF8.encode(obs.text_utf8), UTF8.special(SpecialToken.ANSWER)]
        if len(prompt_ids) > self.v8_spec.max_sequence_length:
            raise ValueError("PublicInput exceeds B0 max_sequence_length")

        # First measure unconstrained raw autoregressive completion
        raw_ids = list(prompt_ids)
        raw_tokens: list[int] = []
        raw_logp = 0.0
        positions = 0
        for _ in range(self.max_new_tokens):
            dist = self.backend.next_distribution(raw_ids)
            positions += len(raw_ids)
            tok = int(dist.argmax().item())
            raw_logp += math.log(max(float(dist[tok].item()), 1e-30))
            raw_tokens.append(tok)
            if tok in (UTF8.eos_id, UTF8.pad_id, UTF8.bos_id, UTF8.special(SpecialToken.ANSWER)):
                break
            raw_ids.append(tok)

        channels = (_CHANNEL,) if obs.query_available else ()
        raw_op, raw_arg = parse_action(raw_tokens, channels)
        try:
            raw_str = UTF8.decode([t for t in raw_tokens if t >= 4])
        except ValueError:
            raw_str = repr(raw_tokens)
        raw_proposal = PolicyProposal(
            operation=raw_op,
            argument=raw_arg,
            joint_log_probability=raw_logp,
            raw_output=raw_str,
            model_calls=len(raw_tokens),
            evaluated_positions=positions,
            mode="raw",
            model_identity=self.model_identity,
        )
        if self.mode == "raw":
            return raw_proposal, raw_proposal

        # Single-step distribution over valid grammar actions at first token
        dist0 = self.backend.next_distribution(prompt_ids)
        dist1 = None  # Only compute EOS check over valid 1-byte actions
        valid_labels = list(SYMBOLS) + ["ABSTAIN"]
        if self.mode != "direct-answer-only":
            if self.mode == "grammar" or obs.query_available:
                valid_labels.append("QUERY")
            if self.mode == "grammar" or obs.memory_available:
                valid_labels.append("RETRIEVE")

        best_idx = 0
        best_logp = float("-inf")
        best_op, best_arg = "ABSTAIN", None
        char_map = {sym: sym for sym in SYMBOLS}
        char_map["ABSTAIN"] = "!"
        char_map["QUERY"] = "Q"
        char_map["RETRIEVE"] = "R"
        for i, lbl in enumerate(valid_labels):
            byte_tok = UTF8.encode(char_map[lbl])[0]
            lp0 = math.log(max(float(dist0[byte_tok].item()), 1e-30))
            if dist1 is None:
                # Approximate 1-token+EOS score by probing EOS after the chosen byte
                pass
            next_dist = self.backend.next_distribution([*prompt_ids, byte_tok])
            lp_eos = math.log(max(float(next_dist[UTF8.eos_id].item()), 1e-30))
            total_lp = lp0 + lp_eos
            if total_lp > best_logp:
                best_logp = total_lp
                best_idx = i
                if lbl in SYMBOLS:
                    best_op, best_arg = "ANSWER", lbl
                elif lbl == "QUERY":
                    best_op, best_arg = "QUERY", (_CHANNEL if obs.query_available else None)
                elif lbl == "RETRIEVE":
                    best_op, best_arg = "RETRIEVE", None
                else:
                    best_op, best_arg = "ABSTAIN", None

        exec_proposal = PolicyProposal(
            operation=best_op,
            argument=best_arg,
            joint_log_probability=best_logp,
            raw_output=raw_str,
            model_calls=1 + len(valid_labels),
            evaluated_positions=len(prompt_ids) + len(valid_labels) * (len(prompt_ids) + 1),
            mode=self.mode,
            model_identity=self.model_identity,
        )
        return exec_proposal, raw_proposal


def make_policy(
    model: Any,
    v8_spec: V8ModelSpec,
    *,
    model_identity: str,
    mode: str = "raw",
    copy_enabled_override: bool | None = None,
) -> V8TypedPolicy | V8BytePolicy:
    if v8_spec.architecture == "B0":
        return V8BytePolicy(model, v8_spec, model_identity=model_identity, mode=mode)
    return V8TypedPolicy(
        model,
        v8_spec,
        model_identity=model_identity,
        mode=mode,
        copy_enabled_override=copy_enabled_override,
    )


__all__ = [
    "V8BytePolicy",
    "V8TypedPolicy",
    "make_policy",
]
