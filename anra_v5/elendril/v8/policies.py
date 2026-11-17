"""V8 actor policies operating strictly on PublicInput (truth-firewalled)."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from anra_v5.elendril.backend import BackendIdentity, NeuralBackend
from anra_v5.elendril.representation import UTF8, SpecialToken
from anra_v5.elendril.v6_evaluation import parse_action

from .contracts import ACTION_19_LABELS, POLICY_MODES, SYMBOLS, PolicyProposal, PublicInput, V8ModelSpec

_CHANNEL = "request-current-status"
_ACTION_19_CHARS = tuple(SYMBOLS) + ("Q", "R", "!")
_ACTION_19_BYTE_TOKENS = tuple(UTF8.encode(ch)[0] for ch in _ACTION_19_CHARS)
_B0_STOP_TOKENS = frozenset((UTF8.eos_id, UTF8.pad_id, UTF8.bos_id, UTF8.special(SpecialToken.ANSWER)))


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


def _build_typed_proposals(
    probs_19: torch.Tensor,
    obs: PublicInput,
    *,
    evaluated_positions: int,
    mode: str,
    model_identity: str,
) -> tuple[PolicyProposal, PolicyProposal]:
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
        evaluated_positions=evaluated_positions,
        mode="raw",
        model_identity=model_identity,
    )

    if mode in ("raw", "grammar"):
        # Typed heads cannot emit malformed syntax; grammar == raw on 19-way space.
        exec_proposal = PolicyProposal(
            operation=raw_op,
            argument=raw_arg,
            joint_log_probability=raw_logp,
            raw_output=raw_txt,
            model_calls=1,
            evaluated_positions=evaluated_positions,
            mode=mode,
            model_identity=model_identity,
        )
        return exec_proposal, raw_proposal

    exec_idx, exec_logp = _apply_mode_to_distribution(
        probs_19,
        mode=mode,
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
        evaluated_positions=evaluated_positions,
        mode=mode,
        model_identity=model_identity,
    )
    return exec_proposal, raw_proposal


def _valid_action_19_indices(
    mode: str,
    *,
    query_available: bool,
    memory_available: bool,
) -> list[int]:
    if mode not in POLICY_MODES:
        raise ValueError(f"unsupported policy mode: {mode!r}")
    # Preserve A..P, ABSTAIN, QUERY, RETRIEVE tie-breaking order.
    indices = list(range(16)) + [18]
    if mode != "direct-answer-only":
        if mode == "grammar" or query_available:
            indices.append(16)
        if mode == "grammar" or memory_available:
            indices.append(17)
    return indices


def _build_b0_exec_proposal(
    *,
    raw_proposal: PolicyProposal,
    dist0_cpu: torch.Tensor,
    lp_eos_19_cpu: torch.Tensor,
    obs: PublicInput,
    prompt_len: int,
    mode: str,
    model_identity: str,
) -> PolicyProposal:
    if mode == "raw":
        return raw_proposal
    valid_indices = _valid_action_19_indices(
        mode,
        query_available=obs.query_available,
        memory_available=obs.memory_available,
    )
    best_logp = float("-inf")
    best_op, best_arg = "ABSTAIN", None
    min_logp = math.log(1e-30)
    for idx in valid_indices:
        byte_tok = _ACTION_19_BYTE_TOKENS[idx]
        lp0 = math.log(max(float(dist0_cpu[byte_tok].item()), 1e-30))
        lp_eos = max(float(lp_eos_19_cpu[idx].item()), min_logp)
        total_lp = lp0 + lp_eos
        if total_lp > best_logp:
            best_logp = total_lp
            lbl = ACTION_19_LABELS[idx]
            if lbl in SYMBOLS:
                best_op, best_arg = "ANSWER", lbl
            elif lbl == "QUERY":
                best_op, best_arg = "QUERY", (_CHANNEL if obs.query_available else None)
            elif lbl == "RETRIEVE":
                best_op, best_arg = "RETRIEVE", None
            else:
                best_op, best_arg = "ABSTAIN", None

    return PolicyProposal(
        operation=best_op,
        argument=best_arg,
        joint_log_probability=best_logp,
        raw_output=raw_proposal.raw_output,
        model_calls=2,
        evaluated_positions=prompt_len + len(valid_indices) * (prompt_len + 1),
        mode=mode,
        model_identity=model_identity,
    )


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

    def extract_probs_batch(self, obs_list: Sequence[PublicInput]) -> tuple[torch.Tensor, list[int]]:
        """Run one batched forward pass and return (probs_19_cpu, prompt_lengths)."""

        if not obs_list:
            return torch.empty((0, 19), dtype=torch.float32), []
        prompt_ids_list: list[list[int]] = []
        for obs in obs_list:
            _ensure_public_input(obs)
            prompt_ids = [UTF8.bos_id, *UTF8.encode(obs.text_utf8), UTF8.special(SpecialToken.ANSWER)]
            if len(prompt_ids) > self.v8_spec.max_sequence_length:
                raise ValueError("PublicInput exceeds model max_sequence_length")
            prompt_ids_list.append(prompt_ids)

        prompt_lens = [len(ids) for ids in prompt_ids_list]
        max_len = max(prompt_lens)
        dev = self.device
        padded_tokens = [ids + [UTF8.pad_id] * (max_len - len(ids)) for ids in prompt_ids_list]
        padded_segments = [[0] * len(ids) + [-1] * (max_len - len(ids)) for ids in prompt_ids_list]

        modes = [(m, m.training) for m in self.model.modules()]
        self.model.eval()
        try:
            with torch.inference_mode():
                tokens = torch.tensor(padded_tokens, device=dev, dtype=torch.int64)
                segments = torch.tensor(padded_segments, device=dev, dtype=torch.int64)
                action_pos = torch.tensor([n - 1 for n in prompt_lens], device=dev, dtype=torch.int64)
                out = self.model.forward_action(
                    tokens,
                    segments,
                    action_pos,
                    copy_enabled_override=self.copy_enabled_override,
                )
                probs_19_batch = out.log_probs_19.exp().cpu()
        finally:
            for m, was_training in modes:
                m.training = was_training
        return probs_19_batch, prompt_lens

    def proposals_from_probs(
        self,
        probs_19: torch.Tensor,
        obs: PublicInput,
        *,
        evaluated_positions: int,
        mode: str | None = None,
    ) -> tuple[PolicyProposal, PolicyProposal]:
        """Build (executed_mode_proposal, raw_proposal) on CPU from cached 19-way probabilities."""

        return _build_typed_proposals(
            probs_19,
            obs,
            evaluated_positions=evaluated_positions,
            mode=self.mode if mode is None else mode,
            model_identity=self.model_identity,
        )

    def propose_batch(self, obs_list: Sequence[PublicInput]) -> list[tuple[PolicyProposal, PolicyProposal]]:
        """Batched equivalent of propose(obs) returning (executed_mode_proposal, raw_proposal) per item."""

        if not obs_list:
            return []
        probs_19_batch, prompt_lens = self.extract_probs_batch(obs_list)
        return [
            _build_typed_proposals(
                probs_19_batch[i],
                obs,
                evaluated_positions=prompt_lens[i],
                mode=self.mode,
                model_identity=self.model_identity,
            )
            for i, obs in enumerate(obs_list)
        ]

    def propose(self, obs: PublicInput) -> tuple[PolicyProposal, PolicyProposal]:
        """Return (executed_mode_proposal, unconstrained_raw_proposal)."""

        return self.propose_batch([obs])[0]


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

    def _run_batch_inference(
        self,
        obs_list: Sequence[PublicInput],
        *,
        label_indices_per_row: list[list[int]] | None,
        max_cand_batch_seqs: int = 64,
    ) -> tuple[list[PolicyProposal], list[torch.Tensor], list[torch.Tensor] | None, list[int]]:
        if type(max_cand_batch_seqs) is not int or max_cand_batch_seqs <= 0:
            raise ValueError("max_cand_batch_seqs must be a positive integer")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if not obs_list:
            return [], [], ([] if label_indices_per_row is not None else None), []

        prompt_ids_list: list[list[int]] = []
        for obs in obs_list:
            _ensure_public_input(obs)
            prompt_ids = [UTF8.bos_id, *UTF8.encode(obs.text_utf8), UTF8.special(SpecialToken.ANSWER)]
            if len(prompt_ids) > self.v8_spec.max_sequence_length:
                raise ValueError("PublicInput exceeds B0 max_sequence_length")
            prompt_ids_list.append(prompt_ids)

        n = len(prompt_ids_list)
        prompt_lens = [len(ids) for ids in prompt_ids_list]
        cur_ids = [list(ids) for ids in prompt_ids_list]
        raw_tokens: list[list[int]] = [[] for _ in range(n)]
        raw_logp = [0.0] * n
        positions = [0] * n
        dist0_list: list[torch.Tensor | None] = [None] * n
        active = list(range(n))
        dev = self.backend.device
        lp_eos_19_list = (
            [torch.full((19,), float("-inf"), dtype=torch.float32) for _ in range(n)]
            if label_indices_per_row is not None
            else None
        )

        modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            with torch.inference_mode():
                for step in range(self.max_new_tokens):
                    if not active:
                        break
                    seqs = [cur_ids[i] for i in active]
                    max_len = max(len(s) for s in seqs)
                    padded_toks = [s + [UTF8.pad_id] * (max_len - len(s)) for s in seqs]
                    padded_segs = [[0] * len(s) + [-1] * (max_len - len(s)) for s in seqs]
                    tok_tensor = torch.tensor(padded_toks, device=dev, dtype=torch.int64)
                    seg_tensor = torch.tensor(padded_segs, device=dev, dtype=torch.int64)
                    logits = self.backend.logits(tok_tensor, seg_tensor)
                    last_pos = torch.tensor([len(s) - 1 for s in seqs], device=dev, dtype=torch.int64)
                    row_arange = torch.arange(len(seqs), device=dev, dtype=torch.int64)
                    dists_cpu = torch.softmax(logits[row_arange, last_pos, :], dim=-1).cpu()

                    next_active: list[int] = []
                    for b, row_idx in enumerate(active):
                        dist = dists_cpu[b]
                        if step == 0:
                            dist0_list[row_idx] = dist
                        positions[row_idx] += len(cur_ids[row_idx])
                        tok = int(dist.argmax().item())
                        raw_logp[row_idx] += math.log(max(float(dist[tok].item()), 1e-30))
                        raw_tokens[row_idx].append(tok)
                        if tok not in _B0_STOP_TOKENS:
                            cur_ids[row_idx].append(tok)
                            next_active.append(row_idx)
                    active = next_active

                if label_indices_per_row is not None and lp_eos_19_list is not None:
                    cand_items: list[tuple[int, int, list[int]]] = []
                    for row_idx, label_indices in enumerate(label_indices_per_row):
                        p_ids = prompt_ids_list[row_idx]
                        for label_idx in label_indices:
                            cand_items.append((row_idx, label_idx, [*p_ids, _ACTION_19_BYTE_TOKENS[label_idx]]))
                    for start in range(0, len(cand_items), max_cand_batch_seqs):
                        sub = cand_items[start : start + max_cand_batch_seqs]
                        c_max_len = max(len(seq) for _, _, seq in sub)
                        c_padded_toks = [seq + [UTF8.pad_id] * (c_max_len - len(seq)) for _, _, seq in sub]
                        c_padded_segs = [[0] * len(seq) + [-1] * (c_max_len - len(seq)) for _, _, seq in sub]
                        c_toks = torch.tensor(c_padded_toks, device=dev, dtype=torch.int64)
                        c_segs = torch.tensor(c_padded_segs, device=dev, dtype=torch.int64)
                        c_logits = self.backend.logits(c_toks, c_segs)
                        c_last = torch.tensor([len(seq) - 1 for _, _, seq in sub], device=dev, dtype=torch.int64)
                        c_arange = torch.arange(len(sub), device=dev, dtype=torch.int64)
                        c_log_probs = torch.log_softmax(c_logits[c_arange, c_last, :], dim=-1)
                        lp_eos_cpu = c_log_probs[:, UTF8.eos_id].cpu()
                        for k, (row_idx, label_idx, _) in enumerate(sub):
                            lp_eos_19_list[row_idx][label_idx] = float(lp_eos_cpu[k].item())
        finally:
            for module, was_training in modes:
                module.training = was_training

        raw_proposals: list[PolicyProposal] = []
        resolved_dist0: list[torch.Tensor] = []
        for i, obs in enumerate(obs_list):
            d0 = dist0_list[i]
            assert d0 is not None
            resolved_dist0.append(d0)
            rtoks = raw_tokens[i]
            channels = (_CHANNEL,) if obs.query_available else ()
            raw_op, raw_arg = parse_action(rtoks, channels)
            try:
                raw_str = UTF8.decode([t for t in rtoks if t >= 4])
            except ValueError:
                raw_str = repr(rtoks)
            raw_proposals.append(
                PolicyProposal(
                    operation=raw_op,
                    argument=raw_arg,
                    joint_log_probability=raw_logp[i],
                    raw_output=raw_str,
                    model_calls=len(rtoks),
                    evaluated_positions=positions[i],
                    mode="raw",
                    model_identity=self.model_identity,
                )
            )
        return raw_proposals, resolved_dist0, lp_eos_19_list, prompt_lens

    def extract_b0_step_scores_batch(
        self,
        obs_list: Sequence[PublicInput],
        *,
        max_cand_batch_seqs: int = 64,
    ) -> list[tuple[PolicyProposal, torch.Tensor, torch.Tensor]]:
        """Return (raw_proposal, dist0_cpu, lp_eos_19_cpu) per observation for exact multi-mode scoring."""

        if not obs_list:
            return []
        all_19 = list(range(19))
        raw_proposals, dist0_list, lp_eos_19_list, _ = self._run_batch_inference(
            obs_list,
            label_indices_per_row=[all_19 for _ in obs_list],
            max_cand_batch_seqs=max_cand_batch_seqs,
        )
        assert lp_eos_19_list is not None
        return [(raw_proposals[i], dist0_list[i], lp_eos_19_list[i]) for i in range(len(obs_list))]

    def score_b0_all_modes_batch(
        self,
        obs_list: Sequence[PublicInput],
        *,
        modes: Sequence[str] = POLICY_MODES,
        max_cand_batch_seqs: int = 64,
    ) -> dict[str, list[tuple[PolicyProposal, PolicyProposal]]]:
        """Compute exact (executed_proposal, raw_proposal) for all requested modes sharing one GPU pass."""

        for mode in modes:
            if mode not in POLICY_MODES:
                raise ValueError(f"unsupported policy mode: {mode!r}")
        if not obs_list:
            return {mode: [] for mode in modes}
        if not modes:
            return {}

        if all(mode == "raw" for mode in modes):
            raw_proposals, _, _, _ = self._run_batch_inference(
                obs_list,
                label_indices_per_row=None,
                max_cand_batch_seqs=max_cand_batch_seqs,
            )
            return {mode: [(r, r) for r in raw_proposals] for mode in modes}

        # Score only the union of candidate labels required across requested modes per observation.
        non_raw_modes = [m for m in modes if m != "raw"]
        union_indices_per_row: list[list[int]] = []
        for obs in obs_list:
            _ensure_public_input(obs)
            seen: set[int] = set()
            ordered: list[int] = []
            for m in non_raw_modes:
                for idx in _valid_action_19_indices(
                    m,
                    query_available=obs.query_available,
                    memory_available=obs.memory_available,
                ):
                    if idx not in seen:
                        seen.add(idx)
                        ordered.append(idx)
            union_indices_per_row.append(ordered)

        raw_proposals, dist0_list, lp_eos_19_list, prompt_lens = self._run_batch_inference(
            obs_list,
            label_indices_per_row=union_indices_per_row,
            max_cand_batch_seqs=max_cand_batch_seqs,
        )
        assert lp_eos_19_list is not None

        out: dict[str, list[tuple[PolicyProposal, PolicyProposal]]] = {}
        for mode in modes:
            if mode == "raw":
                out[mode] = [(r, r) for r in raw_proposals]
            else:
                out[mode] = [
                    (
                        _build_b0_exec_proposal(
                            raw_proposal=raw_proposals[i],
                            dist0_cpu=dist0_list[i],
                            lp_eos_19_cpu=lp_eos_19_list[i],
                            obs=obs,
                            prompt_len=prompt_lens[i],
                            mode=mode,
                            model_identity=self.model_identity,
                        ),
                        raw_proposals[i],
                    )
                    for i, obs in enumerate(obs_list)
                ]
        return out

    def propose_batch(
        self,
        obs_list: Sequence[PublicInput],
        *,
        max_cand_batch_seqs: int = 64,
    ) -> list[tuple[PolicyProposal, PolicyProposal]]:
        """Batched equivalent of propose(obs) returning (executed_mode_proposal, raw_proposal) per item."""

        if not obs_list:
            return []
        if self.mode == "raw":
            raw_proposals, _, _, _ = self._run_batch_inference(
                obs_list,
                label_indices_per_row=None,
                max_cand_batch_seqs=max_cand_batch_seqs,
            )
            return [(r, r) for r in raw_proposals]

        for obs in obs_list:
            _ensure_public_input(obs)
        label_indices_per_row = [
            _valid_action_19_indices(
                self.mode,
                query_available=obs.query_available,
                memory_available=obs.memory_available,
            )
            for obs in obs_list
        ]
        raw_proposals, dist0_list, lp_eos_19_list, prompt_lens = self._run_batch_inference(
            obs_list,
            label_indices_per_row=label_indices_per_row,
            max_cand_batch_seqs=max_cand_batch_seqs,
        )
        assert lp_eos_19_list is not None
        return [
            (
                _build_b0_exec_proposal(
                    raw_proposal=raw_proposals[i],
                    dist0_cpu=dist0_list[i],
                    lp_eos_19_cpu=lp_eos_19_list[i],
                    obs=obs,
                    prompt_len=prompt_lens[i],
                    mode=self.mode,
                    model_identity=self.model_identity,
                ),
                raw_proposals[i],
            )
            for i, obs in enumerate(obs_list)
        ]

    def propose(self, obs: PublicInput) -> tuple[PolicyProposal, PolicyProposal]:
        """Return (executed_mode_proposal, unconstrained_raw_proposal)."""

        return self.propose_batch([obs])[0]


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
