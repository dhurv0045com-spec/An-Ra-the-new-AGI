"""Planthir ExperimentModule registration and Y1 legacy raw byte adapter for Avendi V8-Y2."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.elendril.codec import digest
from anra_v5.elendril.hitchhiker_diagnostics import RawHitchhikerPolicy
from anra_v5.elendril.representation import UTF8, SpecialToken
from anra_v5.elendril.v6_evaluation import parse_action
from anra_v5.planthir import (
    ExperimentModule,
    GenericDatasetBundle,
    UniversalAction,
    UniversalObservation,
)
from anra_v5.planthir.contracts import SubgoalProbeSpec
from .frozen import generate_raw_bytes_and_first_logits
from .metrics import AP_TOKEN_IDS
from .oracle import parse_tokenized_public_observation, walk_parsed_graph
from .protocol import BACKEND_ID, DESIGN_SHA, SYMBOLS
from .render import RenderedRow, encode_prompt_and_target, render_avendi_prompt
from .transition import (
    AvendiContinuousRecurrentControl,
    AvendiOneStepAttentionControl,
    AvendiSymbolicTransitionModel,
    IDX_TO_SYM,
    SYM_TO_IDX,
)
from .worlds import AvendiCase, build_avendi_suite


def case_to_planthir_world(case: AvendiCase) -> dict[str, Any]:
    """Convert an AvendiCase into a Planthir world dict with truth firewalled under 'hidden'."""
    return {
        "world_id": case.case_id,
        "family_id": case.family_id,
        "split": case.split,
        "panel": case.panel,
        "cell_key": case.cell_key,
        "hops": case.hops,
        "query_subject": case.query_subject,
        "public_prompt": case.public_prompt,
        "prompt_tokens": list(case.prompt_tokens),
        "query_available": case.query_available,
        "memory_available": case.memory_available,
        "canonical_family_key": case.canonical_family_key,
        "hidden": {
            "expected": case.expected_answer,
            "rival": case.rival_answer,
            "expected_operation": case.expected_operation,
            "proof_path": list(case.proof_path),
            "target_tokens": list(case.target_tokens),
        },
    }


class AvendiEnvironmentKernel:
    """Planthir EnvironmentKernel for Avendi V8-Y2 worlds with strict actor truth firewall."""

    kernel_id = "avendi-v8-y2-kernel-v1"

    def structural_signature(self, world: Mapping[str, Any]) -> str:
        return str(world.get("canonical_family_key", f"hops:{world.get('hops', 1)}"))

    def inject_distractor(self, world: Mapping[str, Any]) -> dict[str, Any]:
        w = dict(world)
        p_toks = w.get("prompt_tokens") or [UTF8.bos_id, *UTF8.encode(str(w["public_prompt"])), UTF8.special(SpecialToken.ANSWER)]
        parsed = parse_tokenized_public_observation(p_toks)
        used = {r.subject for r in parsed.rows} | {r.value for r in parsed.rows if r.value is not None}
        unused = [s for s in SYMBOLS if s not in used]
        d_subj = unused[0] if unused else "P"
        rows = [RenderedRow(r.subject, r.value, r.tick, r.source) for r in parsed.rows]
        rows.append(RenderedRow(subject=d_subj, value=None, tick=9876, source="s3"))
        new_prompt = render_avendi_prompt(
            query_subject=parsed.query_subject,
            hops=parsed.hops,
            rows=rows,
            query_available=parsed.query_available,
            memory_available=parsed.memory_available,
            is_literal_copy=parsed.is_literal_copy,
        )
        enc = encode_prompt_and_target(new_prompt, str(w["hidden"]["expected"]))
        w["world_id"] = f"{world['world_id']}::dist"
        w["public_prompt"] = new_prompt
        w["prompt_tokens"] = list(enc["prompt_tokens"])
        return w

    def permute_world(self, world: Mapping[str, Any], shift: int = 1) -> dict[str, Any]:
        w = dict(world)
        p_toks = w.get("prompt_tokens") or [UTF8.bos_id, *UTF8.encode(str(w["public_prompt"])), UTF8.special(SpecialToken.ANSWER)]
        parsed = parse_tokenized_public_observation(p_toks)
        if parsed.is_literal_copy:
            w["world_id"] = f"{world['world_id']}::perm{shift}"
            return w
        n = len(SYMBOLS)
        s_mod = (int(shift) % (n - 1)) or 1
        bij = {SYMBOLS[i]: SYMBOLS[(i + s_mod) % n] for i in range(n)}
        rows = [
            RenderedRow(
                subject=bij[r.subject],
                value=bij[r.value] if r.value is not None else None,
                tick=r.tick,
                source=r.source,
            )
            for r in parsed.rows
        ]
        new_q = bij[parsed.query_subject]
        new_exp = bij.get(str(w["hidden"]["expected"]), str(w["hidden"]["expected"]))
        new_riv = bij.get(str(w["hidden"]["rival"]), str(w["hidden"]["rival"]))
        new_prompt = render_avendi_prompt(
            query_subject=new_q,
            hops=parsed.hops,
            rows=rows,
            query_available=parsed.query_available,
            memory_available=parsed.memory_available,
        )
        enc = encode_prompt_and_target(new_prompt, new_exp)
        w["world_id"] = f"{world['world_id']}::perm{shift}"
        w["query_subject"] = new_q
        w["public_prompt"] = new_prompt
        w["prompt_tokens"] = list(enc["prompt_tokens"])
        w["hidden"] = {
            **dict(w["hidden"]),
            "expected": new_exp,
            "rival": new_riv,
            "proof_path": [bij.get(x, x) for x in w["hidden"].get("proof_path", ())],
            "target_tokens": list(enc["target_tokens"]),
        }
        return w

    def reverse_world(self, world: Mapping[str, Any]) -> dict[str, Any]:
        w = dict(world)
        p_toks = w.get("prompt_tokens") or [UTF8.bos_id, *UTF8.encode(str(w["public_prompt"])), UTF8.special(SpecialToken.ANSWER)]
        parsed = parse_tokenized_public_observation(p_toks)
        rows = [RenderedRow(r.subject, r.value, r.tick, r.source) for r in reversed(parsed.rows)]
        new_prompt = render_avendi_prompt(
            query_subject=parsed.query_subject,
            hops=parsed.hops,
            rows=rows,
            query_available=parsed.query_available,
            memory_available=parsed.memory_available,
            is_literal_copy=parsed.is_literal_copy,
        )
        enc = encode_prompt_and_target(new_prompt, str(w["hidden"]["expected"]))
        w["world_id"] = f"{world['world_id']}::rev"
        w["public_prompt"] = new_prompt
        w["prompt_tokens"] = list(enc["prompt_tokens"])
        return w

    def initial_state(self, world: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "world_id": str(world["world_id"]),
            "public_prompt": str(world["public_prompt"]),
            "prompt_tokens": tuple(int(t) for t in world.get("prompt_tokens", (UTF8.bos_id, *UTF8.encode(str(world["public_prompt"])), UTF8.special(SpecialToken.ANSWER)))),
            "query_available": bool(world.get("query_available", False)),
            "memory_available": bool(world.get("memory_available", False)),
            "consumed_tools": frozenset(),
            "step": 0,
            "terminal": False,
        }

    def observe(self, state: Mapping[str, Any]) -> UniversalObservation:
        return UniversalObservation(
            obs_id=f"{state['world_id']}@s{state['step']}",
            rendered_text=str(state["public_prompt"]),
            available_tools={
                "query": bool(state["query_available"]) and ("query" not in state["consumed_tools"]),
                "retrieve": bool(state["memory_available"]) and ("retrieve" not in state["consumed_tools"]),
            },
            consumed_tools=frozenset(state["consumed_tools"]),
            decision_index=int(state["step"]),
            budget_remaining=max(0, 2 - int(state["step"])),
            metadata={"world_id": str(state["world_id"])},
        )

    def ideal_action(self, state: Mapping[str, Any]) -> tuple[str, str | None]:
        parsed = parse_tokenized_public_observation(state["prompt_tokens"])
        walked = walk_parsed_graph(parsed)
        op = walked["operation"]
        if op == "ANSWER":
            return ("ANSWER", str(walked["argument"]))
        if op == "QUERY":
            return ("QUERY", "request-current-status")
        return ("ABSTAIN", None)

    def step(self, state: Mapping[str, Any], operation: str, argument: str | None) -> dict[str, Any]:
        consumed = set(state["consumed_tools"])
        if operation == "QUERY":
            consumed.add("query")
        elif operation == "RETRIEVE":
            consumed.add("retrieve")
        return {
            **dict(state),
            "consumed_tools": frozenset(consumed),
            "query_available": False if operation == "QUERY" else bool(state["query_available"]),
            "memory_available": False if operation == "RETRIEVE" else bool(state["memory_available"]),
            "step": int(state["step"]) + 1,
            "terminal": True,
        }

    def verify(self, world: Mapping[str, Any], actions: Sequence[tuple[str, str | None]]) -> dict[str, Any]:
        if not actions:
            return {"correct": False, "terminal_status": "empty_actions"}
        op, arg = actions[-1]
        hidden = world.get("hidden", {})
        exp_op = str(hidden.get("expected_operation", "ANSWER"))
        exp_ans = str(hidden.get("expected", ""))
        if exp_op == "ANSWER":
            ok = op == "ANSWER" and arg == exp_ans
        elif exp_op == "QUERY":
            ok = op == "QUERY"
        else:
            ok = op == "ABSTAIN"
        return {
            "correct": bool(ok),
            "terminal_status": "verified",
            "expected_operation": exp_op,
            "expected_argument": exp_ans if exp_op == "ANSWER" else None,
            "observed_operation": op,
            "observed_argument": arg,
        }

    def decompose_subgoals(self, world: Mapping[str, Any]) -> tuple[SubgoalProbeSpec, ...]:
        p_toks = world.get("prompt_tokens") or [UTF8.bos_id, *UTF8.encode(str(w["public_prompt"])), UTF8.special(SpecialToken.ANSWER)]
        parsed = parse_tokenized_public_observation(p_toks)
        if parsed.hops <= 1 or parsed.is_literal_copy:
            return ()
        walked = walk_parsed_graph(parsed)
        proof = walked["proof_path"]
        if len(proof) < 2:
            return ()
        rows = [RenderedRow(r.subject, r.value, r.tick, r.source) for r in parsed.rows]
        sub_prompt = render_avendi_prompt(
            query_subject=parsed.query_subject,
            hops=1,
            rows=rows,
            query_available=False,
            memory_available=False,
        )
        hop1_target = proof[1]
        enc = encode_prompt_and_target(sub_prompt, hop1_target)
        mod_world = {
            **dict(world),
            "world_id": f"{world['world_id']}::subgoal_hop1",
            "hops": 1,
            "public_prompt": sub_prompt,
            "prompt_tokens": list(enc["prompt_tokens"]),
            "hidden": {
                **dict(world.get("hidden", {})),
                "expected": hop1_target,
                "expected_operation": "ANSWER",
                "proof_path": list(proof[:2]),
                "target_tokens": list(enc["target_tokens"]),
            },
        }
        return (
            SubgoalProbeSpec(
                subgoal_id=f"{world['world_id']}::hop1",
                parent_world_id=str(world["world_id"]),
                hop_index=1,
                modified_world=mod_world,
                expected_operation="ANSWER",
                expected_argument=hop1_target,
            ),
        )


class AvendiY1LegacyByteAdapter:
    """Explicit Y1 legacy raw byte adapter preserving native 260-way byte logits without X1 19-action conversion."""

    policy_id = "avendi-y1-legacy-byte-adapter-v1"

    def __init__(self, backend: Any, *, model_identity: str = "y1-legacy-s3", mode: str = "raw") -> None:
        self.backend = backend
        self.model_identity = model_identity
        self.mode = mode
        self._raw_hh = RawHitchhikerPolicy(backend)

    def propose_raw_bytes(self, prompt_tokens: Sequence[int]) -> tuple[list[int], Any]:
        """Generate raw bytes until EOS and return (raw_tokens, first_logits_260)."""
        return generate_raw_bytes_and_first_logits(self.backend, prompt_tokens)

    def verify_token_equivalence_with_y1_case(self, y1_case: Any) -> dict[str, Any]:
        """Verify token-for-token equivalence between this adapter and RawHitchhikerPolicy on a clean Y1 case."""
        from anra_v5.elendril.y1_evaluation import evaluate_y1_cases

        prompt_tokens = (UTF8.bos_id, *UTF8.encode(y1_case.public_prompt), UTF8.special(SpecialToken.ANSWER))
        adapter_tokens, first_logits = self.propose_raw_bytes(prompt_tokens)

        pred_doc = evaluate_y1_cases(
            (y1_case,),
            self._raw_hh,
            job_id="avendi-eq-check",
            checkpoint_sha256="0" * 64,
        )
        hh_tokens = list(pred_doc["predictions"][0]["decisions"][0]["raw_tokens"])

        matched = adapter_tokens == hh_tokens
        return {
            "case_id": y1_case.case_id,
            "matched": matched,
            "adapter_tokens": adapter_tokens,
            "hitchhiker_raw_tokens": hh_tokens,
            "first_logits_shape": list(first_logits.shape),
        }

    def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction:
        return self.propose_batch([obs])[0]

    def propose_batch(self, obs_batch: Sequence[UniversalObservation]) -> list[UniversalAction]:
        out: list[UniversalAction] = []
        for obs in obs_batch:
            p_toks = (UTF8.bos_id, *UTF8.encode(obs.rendered_text), UTF8.special(SpecialToken.ANSWER))
            raw_toks, logits_260 = self.propose_raw_bytes(p_toks)
            probs = F.softmax(torch.tensor(logits_260, dtype=torch.float32), dim=-1)
            channels = ("request-current-status",) if obs.available_tools.get("query") else ()
            op, arg = parse_action(tuple(raw_toks), channels, "utf8-raw")
            if op is None:
                op, arg = ("ABSTAIN", None)
            first_tok = raw_toks[0] if raw_toks else UTF8.eos_id
            lp = float(torch.log(probs[first_tok].clamp_min(1e-12)).item())
            dist = {f"ANSWER:{s}": float(probs[AP_TOKEN_IDS[s]].item()) for s in SYMBOLS}
            out.append(
                UniversalAction(
                    operation=op,
                    argument=arg,
                    log_probability=lp,
                    action_distribution=dist,
                    mode=self.mode,
                    raw_operation=op,
                    raw_argument=arg,
                )
            )
        return out


class _AvendiPlanthirTransitionWrapper(nn.Module):
    """Adapter wrapping P4 transition models for Planthir ExperimentModule Gate 1-5 execution."""

    def __init__(self, arch: str, width: int, seed: int) -> None:
        super().__init__()
        self.arch = arch
        if arch == "symbolic_bottleneck_soft":
            self.inner = AvendiSymbolicTransitionModel(seed=seed)
        elif arch == "continuous_recurrent":
            self.inner = AvendiContinuousRecurrentControl(recurrent_width=width, seed=seed)
        elif arch == "one_step_attention":
            self.inner = AvendiOneStepAttentionControl(seed=seed)
        else:
            raise ValueError(f"unsupported Avendi Planthir architecture: {arch!r}")
        self.trained_compute_steps = 3
        self._ablate = False
        self._depth_override: int | None = None

    @contextmanager
    def ablate_mechanism(self):
        self._ablate = True
        try:
            yield "avendi_transition_bottleneck"
        finally:
            self._ablate = False

    @contextmanager
    def compute_depth_context(self, steps: int):
        old = self._depth_override
        self._depth_override = int(steps)
        try:
            yield
        finally:
            self._depth_override = old

    def forward_prompt(self, prompt: str) -> dict[str, Any]:
        res = self.inner.forward_prompt(prompt)
        if self._ablate:
            logits = torch.zeros_like(res["final_logits"])
            return {**res, "final_logits": logits, "predicted_symbol": "A"}
        return res


class AvendiPlanthirPolicy:
    """ObservablePolicy wrapper for _AvendiPlanthirTransitionWrapper."""

    policy_id = "avendi-planthir-policy-v1"

    def __init__(
        self,
        model: _AvendiPlanthirTransitionWrapper,
        model_spec: Any,
        *,
        model_identity: str,
        mode: str = "raw",
        copy_enabled_override: bool | None = None,
    ) -> None:
        self.model = model
        self.model_spec = model_spec
        self.model_identity = model_identity
        self.mode = mode
        self.batched_calls = 0

    def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction:
        return self.propose_batch([obs])[0]

    def propose_batch(self, obs_batch: Sequence[UniversalObservation]) -> list[UniversalAction]:
        self.batched_calls += 1
        out: list[UniversalAction] = []
        self.model.eval()
        with torch.no_grad():
            for obs in obs_batch:
                res = self.model.forward_prompt(obs.rendered_text)
                logits = res["final_logits"]
                probs = F.softmax(logits, dim=-1)
                pred_idx = int(torch.argmax(probs).item())
                sym = IDX_TO_SYM[pred_idx]
                lp = float(torch.log(probs[pred_idx].clamp_min(1e-12)).item())
                dist = {f"ANSWER:{IDX_TO_SYM[i]}": float(probs[i].item()) for i in range(16)}
                out.append(
                    UniversalAction(
                        operation="ANSWER",
                        argument=sym,
                        log_probability=lp,
                        action_distribution=dist,
                        mode=self.mode,
                        raw_operation="ANSWER",
                        raw_argument=sym,
                    )
                )
        return out

    def extract_representations(self, obs_batch: Sequence[UniversalObservation]) -> dict[str, torch.Tensor]:
        vecs: list[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for obs in obs_batch:
                res = self.model.forward_prompt(obs.rendered_text)
                vecs.append(res["final_logits"].detach().cpu())
        return {"decision_hidden": torch.stack(vecs, dim=0)}


def _avendi_model_factory(arch: str, model_config: Mapping[str, Any], seed: int) -> nn.Module:
    width = int(model_config.get("recurrent_width", 56))
    return _AvendiPlanthirTransitionWrapper(arch=arch, width=width, seed=seed)


def _avendi_dataset_factory(dataset_config: Mapping[str, Any], cpu_canary: bool) -> dict[str, Any]:
    sf = 64 if cpu_canary else int(dataset_config.get("scale_factor", 32))
    suite = build_avendi_suite(scale_factor=sf)
    all_train_fams = suite["splits"]["train_p3"]["one_hop_families"]
    n_tr = 4 if cpu_canary else 16
    train_fams = all_train_fams[:n_tr]
    probe_fams = all_train_fams[n_tr : n_tr + 2]
    dev_fams = suite["splits"]["dev"]["one_hop_families"][: (2 if cpu_canary else 8)]
    mh_fams = suite["splits"]["dev"]["three_hop_families"][: (2 if cpu_canary else 8)]

    train_cases = [c for f in train_fams for c in f.cells]
    probe_cases = [c for f in probe_fams for c in f.cells]
    eval_cases = [c for f in dev_fams for c in f.cells]
    mh_cases = [c for f in mh_fams for c in f.cells]

    items = tuple(
        {
            "case_id": c.case_id,
            "public_prompt": c.public_prompt,
            "target_idx": SYM_TO_IDX[c.expected_answer],
            "rival_idx": SYM_TO_IDX[c.rival_answer],
        }
        for c in train_cases
    )
    train_worlds = tuple(case_to_planthir_world(c) for c in train_cases)
    probe_worlds = tuple(case_to_planthir_world(c) for c in probe_cases)
    eval_worlds = tuple(case_to_planthir_world(c) for c in eval_cases)
    mh_worlds = tuple(case_to_planthir_world(c) for c in mh_cases)
    data_ident = digest([c.prompt_token_sha256 for c in train_cases])

    return {
        "train_data": GenericDatasetBundle(
            dataset_name=str(dataset_config.get("dataset_name", "avendi-crossover")),
            items=items,
            data_identity=data_ident,
            manifest={"schema": "avendi-planthir-dataset-v1", "design_sha256": DESIGN_SHA, "items": len(items)},
        ),
        "train_worlds": train_worlds,
        "eval_worlds": eval_worlds,
        "multihop_worlds": mh_worlds,
        "probe_train_worlds": probe_worlds,
    }


def _avendi_loss_fn(model: _AvendiPlanthirTransitionWrapper, batch: Sequence[Mapping[str, Any]], recipe: Any) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    lam = 0.25 if getattr(recipe, "objective", "ce") == "ce_pair_margin" else 0.0
    for item in batch:
        res = model.forward_prompt(str(item["public_prompt"]))
        logits = res["final_logits"]
        gold_idx = int(item["target_idx"])
        rival_idx = int(item["rival_idx"])
        ce = F.cross_entropy(logits.unsqueeze(0), torch.tensor([gold_idx], dtype=torch.int64, device=logits.device))
        if lam > 0.0:
            s_i = logits[gold_idx] - logits[rival_idx]
            ce = ce + lam * F.relu(1.0 - s_i)
        losses.append(ce)
    return torch.stack(losses).mean()


AVENDI_EXPERIMENT_MODULE = ExperimentModule(
    backend_id=BACKEND_ID,
    backend_version="avendi-v8-y2-1.0",
    program="AVENDI",
    architectures=("symbolic_bottleneck_soft", "continuous_recurrent", "one_step_attention"),
    objectives=("ce", "ce_pair_margin"),
    model_factory=_avendi_model_factory,
    dataset_factory=_avendi_dataset_factory,
    loss_fn=_avendi_loss_fn,
    kernel_factory=AvendiEnvironmentKernel,
    policy_factory=AvendiPlanthirPolicy,
    default_dataset_config={"dataset_name": "avendi-crossover", "scale_factor": 64},
    default_model_config={"recurrent_width": 56},
    treatment_architecture="symbolic_bottleneck_soft",
    treatment_model_config={"recurrent_width": 56},
    extra_model_keys=("recurrent_width",),
    extra_dataset_keys=("scale_factor",),
    supports_grouped_loss=True,
).register()


def register_planthir_backends() -> ExperimentModule:
    """Entry point invoked automatically when PLANTHIR_BACKEND_MODULES includes this module."""
    return AVENDI_EXPERIMENT_MODULE.register()
