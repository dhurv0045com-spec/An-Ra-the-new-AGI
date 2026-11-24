"""Instrument validation, invariant enforcement, and controlled fault injection."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import math
from hashlib import sha256
import random
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import torch

from anra_v5.elendril.representation import SpecialToken, UTF8
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8.contracts import PublicInput, build_v8_model_spec
from anra_v5.elendril.v8.data import build_d1_training_dataset, render_public_v8
from anra_v5.elendril.v8.evaluation import run_episode as v8_run_episode
from anra_v5.elendril.v8.models import build_v8_model, compute_typed_nll
from anra_v5.elendril.v8.teacher import build_teacher_steps_for_world
from anra_v5.elendril.v8.training import V8TypedTrainer, prepare_v8_training_data
from anra_v5.elendril.foundation_training import Recipe

from .adapters.v8_adapter import V8EnvironmentKernel
from .contracts import EnvironmentKernel, ObservablePolicy, UniversalObservation
from .scoring import decode_generated_bytes

_TOOL_OP_TO_KEY = {"QUERY": "query", "RETRIEVE": "retrieve"}


class _OracleV8Actor:
    """Minimal truth-firewalled oracle actor used purely for instrument parity checks."""

    mode = "raw"
    model_identity = "planthir-oracle-probe"

    def __init__(self, kernel: V8EnvironmentKernel) -> None:
        self._kernel = kernel

    def propose(self, obs: PublicInput):
        from anra_v5.elendril.v8.contracts import PolicyProposal
        from anra_v5.elendril.v8.teacher import PublicFact, teacher_decide
        import re

        # Parse only the visible prompt lines cleanly without touching world truth
        lines = obs.text_utf8.splitlines()
        header = lines[1] if len(lines) > 1 else ""
        hops = 1
        m_hops = re.search(r"Follow (\d+) links from ([A-Za-z0-9-]+)\.", header)
        m_curr = re.search(r"Current value of ([A-Za-z0-9-]+)\?", header)
        if m_hops:
            hops = int(m_hops.group(1))
            goal_raw = m_hops.group(2)
        elif m_curr:
            goal_raw = m_curr.group(1)
        else:
            raise ValueError(f"cannot parse public goal from prompt header: {header!r}")

        alias_prefix = ""
        for pref in ("tr-", "dev-"):
            if goal_raw.startswith(pref):
                alias_prefix = pref
                break
        goal = goal_raw[len(alias_prefix) :]

        facts_line = lines[2] if len(lines) > 2 else ""
        parsed_facts: list[PublicFact] = []
        if facts_line and facts_line != "Action:":
            for chunk in facts_line.split("; "):
                m0 = re.match(r"^([^=]+)=([^@]+)@(-?\d+)/(.+)$", chunk.strip())
                if not m0:
                    raise ValueError(f"unsupported template_0 row in oracle probe: {chunk!r}")
                subj_r, val_r, tick_s, src = m0.groups()
                subj = subj_r[len(alias_prefix) :] if alias_prefix and subj_r.startswith(alias_prefix) else subj_r
                val = None if val_r == "null" else (
                    val_r[len(alias_prefix) :] if alias_prefix and val_r.startswith(alias_prefix) else val_r
                )
                parsed_facts.append(PublicFact(subject=subj, source=src, tick=int(tick_s), value=val))

        op, arg = teacher_decide(tuple(parsed_facts), goal, hops, obs.query_available, obs.memory_available)
        prop = PolicyProposal(
            operation=op,
            argument=("request-current-status" if op == "QUERY" and obs.query_available else (arg if op == "ANSWER" else None)),
            joint_log_probability=0.0,
            raw_output=op,
            model_calls=0,
            evaluated_positions=0,
            mode="raw",
            model_identity=self.model_identity,
        )
        return prop, prop


def validate_environment_kernel(
    kernel: EnvironmentKernel,
    worlds: Sequence[Mapping[str, Any]],
    *,
    static_steps_fn: Callable[[Mapping[str, Any]], Sequence[Any]] | None = None,
    check_v8_evaluator_parity: bool = False,
    v8_enforce_tool_consumption: bool = True,
) -> dict[str, Any]:
    """Verify kernel rollout parity, tool consumption monotonicity, truth firewall, and subgoals."""

    if not worlds:
        raise ValueError("validate_environment_kernel requires at least one world")

    defects: list[dict[str, Any]] = []
    tri_matches = 0
    tri_total = 0
    tool_checks = 0
    tool_violations = 0
    firewall_checks = 0
    firewall_violations = 0
    subgoal_checks = 0
    subgoal_violations = 0
    oracle_traces: list[tuple[list[UniversalObservation], list[tuple[str, str | None]]]] = []

    if static_steps_fn is None and isinstance(kernel, V8EnvironmentKernel):
        static_steps_fn = lambda w: build_teacher_steps_for_world(dict(w), render_fn=render_public_v8)

    for world in worlds:
        wid = str(world.get("world_id", "unknown"))
        state = kernel.initial_state(world)
        actions: list[tuple[str, str | None]] = []
        observations: list[UniversalObservation] = []
        consumed_history: set[str] = set()

        for step_idx in range(8):
            obs = kernel.observe(state)
            observations.append(obs)
            inv_errors = obs.validate_invariants()
            for err in inv_errors:
                if "consumed_tool_still_available" in err:
                    tool_violations += 1
                if "truth_leak_marker" in err or "forbidden_metadata_key" in err:
                    firewall_violations += 1
                defects.append({"world_id": wid, "step": step_idx, "check": "observation_invariant", "detail": err})

            for prev_tool in consumed_history:
                tool_checks += 1
                if obs.available_tools.get(prev_tool, False):
                    tool_violations += 1
                    defects.append({
                        "world_id": wid,
                        "step": step_idx,
                        "check": "tool_consumption_monotonicity",
                        "detail": f"tool {prev_tool!r} remained available after consumption",
                    })

            op, arg = kernel.ideal_action(state)
            actions.append((op, arg))
            if op in _TOOL_OP_TO_KEY:
                consumed_history.add(_TOOL_OP_TO_KEY[op])

            next_state = kernel.step(state, op, arg)
            state = next_state
            if getattr(state, "terminal", False) or op not in _TOOL_OP_TO_KEY:
                break

        oracle_traces.append((observations, actions))
        outcome = kernel.verify(world, actions)
        tri_total += 1
        oracle_ok = bool(outcome.get("correct", False))
        if not oracle_ok:
            defects.append({
                "world_id": wid,
                "check": "train_rollout_tri_parity",
                "detail": f"oracle trace failed verifier: {outcome}",
            })

        static_ok = True
        if static_steps_fn is not None:
            static_steps = static_steps_fn(world)
            if len(static_steps) != len(observations):
                static_ok = False
                defects.append({
                    "world_id": wid,
                    "check": "train_rollout_tri_parity",
                    "detail": f"step count mismatch: rollout={len(observations)} vs static={len(static_steps)}",
                })
            else:
                for st_idx, (obs, st) in enumerate(zip(observations, static_steps)):
                    if obs.rendered_text != st.public_input.text_utf8:
                        static_ok = False
                        defects.append({
                            "world_id": wid,
                            "step": st_idx,
                            "check": "train_rollout_tri_parity",
                            "detail": "prompt_drift between training step and rollout observation",
                        })
                    if (
                        bool(obs.available_tools.get("query", False)) != bool(st.public_input.query_available)
                        or bool(obs.available_tools.get("retrieve", False)) != bool(st.public_input.memory_available)
                    ):
                        static_ok = False
                        defects.append({
                            "world_id": wid,
                            "step": st_idx,
                            "check": "train_rollout_tri_parity",
                            "detail": "tool availability mismatch between training step and rollout observation",
                        })

        if oracle_ok and static_ok:
            tri_matches += 1

        # Truth firewall counterfactual check: mutating hidden expected answer must not alter observations
        if "hidden" in world and isinstance(world["hidden"], Mapping) and "expected" in world["hidden"]:
            firewall_checks += 1
            mutated = deepcopy(dict(world))
            orig_exp = mutated["hidden"]["expected"]
            mutated["hidden"]["expected"] = "Z" if orig_exp != "Z" else "Y"
            m_state = kernel.initial_state(mutated)
            for step_idx, orig_obs in enumerate(observations):
                m_obs = kernel.observe(m_state)
                if m_obs != orig_obs:
                    firewall_violations += 1
                    defects.append({
                        "world_id": wid,
                        "step": step_idx,
                        "check": "actor_truth_firewall",
                        "detail": "mutating hidden.expected altered actor-visible observation",
                    })
                    break
                m_op, m_arg = actions[step_idx]
                m_state = kernel.step(m_state, m_op, m_arg)

        # Subgoal decomposition validity check
        subgoals = kernel.decompose_subgoals(world)
        for sg in subgoals:
            subgoal_checks += 1
            sg_state = kernel.initial_state(sg.modified_world)
            sg_op, sg_arg = kernel.ideal_action(sg_state)
            if (sg_op, sg_arg) != (sg.expected_operation, sg.expected_argument):
                subgoal_violations += 1
                defects.append({
                    "world_id": wid,
                    "subgoal_id": sg.subgoal_id,
                    "check": "subgoal_decomposition_validity",
                    "detail": f"subgoal oracle {(sg_op, sg_arg)!r} != expected {(sg.expected_operation, sg.expected_argument)!r}",
                })

    v8_eval_check: dict[str, Any] = {"checked": False}
    if check_v8_evaluator_parity and isinstance(kernel, V8EnvironmentKernel):
        oracle_actor = _OracleV8Actor(kernel)
        v8_eval_ok = 0
        v8_template0_worlds = [w for w in worlds if int(w.get("renderer_template", 0)) == 0]
        if not v8_template0_worlds:
            defects.append({"check": "v8_run_episode_oracle_parity", "detail": "no template-0 worlds available for requested parity check"})
        for w, (observations, actions) in ((w, t) for w, t in zip(worlds, oracle_traces) if int(w.get("renderer_template", 0)) == 0):
            ep = v8_run_episode(
                oracle_actor,
                dict(w),
                render_fn=render_public_v8,
                enforce_tool_consumption=v8_enforce_tool_consumption,
                max_actions=kernel.max_actions,
            )
            expected_hashes = [sha256(o.rendered_text.encode("utf-8")).hexdigest() for o in observations]
            actual_hashes = [step.public_text_sha256 for step in ep.steps]
            def normalized(action):
                op, arg = action
                return (op, arg if op == "ANSWER" else None)
            actual_actions = [(step.executed_proposal.operation, step.executed_proposal.argument) for step in ep.steps]
            trace_matches = expected_hashes == actual_hashes and list(map(normalized, actions)) == list(map(normalized, actual_actions))
            if ep.outcome.correct and trace_matches:
                v8_eval_ok += 1
            else:
                defects.append({
                    "world_id": str(w["world_id"]),
                    "check": "v8_run_episode_oracle_parity",
                    "detail": f"v8 oracle trace or outcome mismatch: status={ep.outcome.terminal_status}, trace_matches={trace_matches}",
                })
        v8_eval_check = {
            "checked": True,
            "worlds": len(v8_template0_worlds),
            "oracle_passed": v8_eval_ok,
            "all_passed": bool(v8_template0_worlds) and v8_eval_ok == len(v8_template0_worlds),
        }

    passed = len(defects) == 0
    return {
        "schema": "anra-planthir-instrument-check-v1",
        "kernel_id": kernel.kernel_id,
        "passed": passed,
        "status": "PASS" if passed else "INSTRUMENT_DEFECT_DETECTED",
        "worlds_checked": len(worlds),
        "checks": {
            "train_rollout_tri_parity": {
                "passed": tri_matches == tri_total and not any(d["check"] == "train_rollout_tri_parity" for d in defects),
                "matched_worlds": tri_matches,
                "total_worlds": tri_total,
            },
            "tool_consumption_monotonicity": {
                "passed": tool_violations == 0,
                "post_consumption_steps_checked": tool_checks,
                "violations": tool_violations,
            },
            "actor_truth_firewall": {
                "passed": firewall_violations == 0,
                "worlds_checked": firewall_checks,
                "violations": firewall_violations,
            },
            "subgoal_decomposition_validity": {
                "passed": subgoal_violations == 0,
                "subgoals_checked": subgoal_checks,
                "violations": subgoal_violations,
            },
            "v8_evaluator_parity": v8_eval_check,
        },
        "defects": defects,
    }


def validate_metric_aggregator(
    aggregator_fn: Callable[[Sequence[tuple[str, dict[str, Any]]]], dict[str, Any]],
) -> dict[str, Any]:
    """Verify multi-epoch aggregation properly tracks the final epoch without shadowing."""

    synthetic_epochs: list[tuple[str, dict[str, Any]]] = [
        ("epoch-1.json", {"epoch": 1, "accuracy": 0.25}),
        ("epoch-2.json", {"epoch": 2, "accuracy": 0.50}),
        ("epoch-4.json", {"epoch": 4, "accuracy": 0.75}),
        ("epoch-10.json", {"epoch": 10, "accuracy": 0.90}),
    ]
    defects: list[str] = []
    result: dict[str, Any] = {}
    try:
        result = aggregator_fn(synthetic_epochs)
    except Exception as exc:
        defects.append(f"aggregator_exception:{type(exc).__name__}:{exc}")
    else:
        final_epoch = result.get("final_epoch")
        final_acc = result.get("final_accuracy")
        if final_epoch != 10:
            defects.append(f"epoch_shadowing:expected_final_epoch_10_got_{final_epoch!r}")
        if not isinstance(final_acc, (int, float)) or abs(float(final_acc) - 0.90) > 1e-9:
            defects.append(f"accuracy_shadowing:expected_0.90_got_{final_acc!r}")

    passed = len(defects) == 0
    return {
        "schema": "anra-planthir-aggregator-check-v1",
        "passed": passed,
        "status": "PASS" if passed else "INSTRUMENT_DEFECT_DETECTED",
        "result": result,
        "defects": defects,
    }


@contextmanager
def _preserve_model_and_rng_state(model: torch.nn.Module):
    """Snapshot and restore model training mode, buffers, grads, and RNG state across non-mutating probes."""

    modes = [(m, m.training) for m in model.modules()]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    gradients = [(p, p.grad, None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    python_rng = random.getstate()
    try:
        import numpy as np
    except ImportError:
        np = None
    numpy_rng = np.random.get_state() if np is not None else None
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)
            for param, original, value in gradients:
                param.grad = original
                if original is not None:
                    original.copy_(value)
        for m, was in modes:
            m.training = was
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
        if np is not None:
            np.random.set_state(numpy_rng)


def validate_fp16_autocast_forward_backward(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.nn.Module], torch.Tensor],
    *,
    active_param_contract: set[str] | frozenset[str] | None = None,
    dtype: torch.dtype = torch.float16,
) -> dict[str, Any]:
    """Perform a non-mutating FP16 autocast forward/backward probe catching Half overflow and dead grads."""

    device_type = "cuda" if any(p.is_cuda for p in model.parameters()) else "cpu"
    defects: list[str] = []
    non_none_params: set[str] = set()
    nonzero_params: set[str] = set()
    loss_value: float | None = None

    with _preserve_model_and_rng_state(model):
        try:
            model.train()
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type, dtype=dtype):
                loss = loss_fn(model)
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                defects.append(f"nonfinite_fp16_loss:{loss_value!r}")
            else:
                loss.backward()
                for name, param in model.named_parameters():
                    if not param.requires_grad or param.grad is None:
                        continue
                    if not torch.isfinite(param.grad).all():
                        defects.append(f"nonfinite_fp16_grad:{name}")
                        continue
                    norm = float(param.grad.detach().float().norm().item())
                    non_none_params.add(name)
                    if norm > 0.0:
                        nonzero_params.add(name)
                if not nonzero_params:
                    defects.append("no_nonzero_fp16_gradients")
                if active_param_contract is not None:
                    for name in sorted(active_param_contract - nonzero_params):
                        defects.append(f"missing_active_fp16_gradient:{name}")
        except (RuntimeError, FloatingPointError, OverflowError) as exc:
            defects.append(f"fp16_autocast_runtime_error:{type(exc).__name__}:{exc}")

    passed = len(defects) == 0
    return {
        "schema": "anra-planthir-fp16-autocast-check-v1",
        "passed": passed,
        "status": "PASS" if passed else "INSTRUMENT_DEFECT_DETECTED",
        "device_type": device_type,
        "dtype": str(dtype),
        "loss": loss_value,
        "non_none_grad_count": len(non_none_params),
        "nonzero_grad_count": len(nonzero_params),
        "defects": defects,
    }


def validate_model_and_gradients(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.nn.Module], torch.Tensor],
    *,
    active_param_contract: set[str] | frozenset[str] | None = None,
    check_fp16_autocast: bool = False,
) -> dict[str, Any]:
    """Perform a non-mutating forward/backward graph integrity and active-parameter check."""

    defects: list[str] = []
    grad_norms: dict[str, float] = {}
    non_none_params: set[str] = set()
    nonzero_params: set[str] = set()
    loss_value: float | None = None

    with _preserve_model_and_rng_state(model):
        model.train()
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model)
        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            defects.append(f"nonfinite_loss:{loss_value!r}")
        loss.backward()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                defects.append(f"nonfinite_grad:{name}")
                continue
            norm = float(param.grad.detach().float().norm().item())
            grad_norms[name] = norm
            non_none_params.add(name)
            if norm > 0.0:
                nonzero_params.add(name)

        if not nonzero_params:
            defects.append("no_nonzero_gradients")
        if active_param_contract is not None:
            missing = sorted(active_param_contract - nonzero_params)
            for name in missing:
                defects.append(f"missing_active_gradient:{name}")

    fp16_report: dict[str, Any] | None = None
    if check_fp16_autocast:
        fp16_report = validate_fp16_autocast_forward_backward(
            model,
            loss_fn,
            active_param_contract=active_param_contract,
        )
        defects.extend(fp16_report["defects"])

    passed = len(defects) == 0
    return {
        "schema": "anra-planthir-grad-check-v1",
        "passed": passed,
        "status": "PASS" if passed else "INSTRUMENT_DEFECT_DETECTED",
        "loss": loss_value,
        "non_none_grad_count": len(non_none_params),
        "nonzero_grad_count": len(nonzero_params),
        "fp16_autocast": fp16_report,
        "defects": defects,
    }


def _tiny_cpu_base_spec():
    return replace(
        MODEL_PROFILES["S20"],
        width=32,
        layers=1,
        query_heads=2,
        kv_heads=1,
        head_dimension=16,
        ffn_width=64,
        context_length=512,
    )


def run_fault_injection_suite() -> dict[str, Any]:
    """Execute 6 controlled fault injections and verify Planthir catches and localizes all 6."""

    d1 = build_d1_training_dataset(quota_per_family=16)
    t0_multi_tool_need = [
        w
        for w in d1.worlds
        if int(w["renderer_template"]) == 0
        and w["family"] in ("missing_query", "conflict_query")
        and w["allowed_public_tools"]["query_available"]
        and w["allowed_public_tools"]["memory_available"]
    ]
    t0_other_tool_need = [
        w
        for w in d1.worlds
        if int(w["renderer_template"]) == 0
        and w["family"] in ("missing_query", "conflict_query", "memory_retrieval")
        and w not in t0_multi_tool_need
    ]
    template0_tool_worlds = (t0_multi_tool_need[:4] + t0_other_tool_need[:4])[:8]

    cases: list[dict[str, Any]] = []

    # 1. V8 consumed-tool availability defect (reproducing the historical V8 evaluator bug)
    healthy_kernel = V8EnvironmentKernel(enforce_tool_consumption=True)
    healthy_report = validate_environment_kernel(
        healthy_kernel,
        template0_tool_worlds,
        check_v8_evaluator_parity=True,
        v8_enforce_tool_consumption=True,
    )
    buggy_kernel = V8EnvironmentKernel(enforce_tool_consumption=False)
    buggy_report = validate_environment_kernel(
        buggy_kernel,
        template0_tool_worlds,
        check_v8_evaluator_parity=True,
        v8_enforce_tool_consumption=False,
    )
    caught_1 = (
        healthy_report["passed"] is True
        and buggy_report["status"] == "INSTRUMENT_DEFECT_DETECTED"
        and not buggy_report["checks"]["tool_consumption_monotonicity"]["passed"]
        and not buggy_report["checks"]["v8_evaluator_parity"]["all_passed"]
    )
    cases.append({
        "fault_id": "v8_consumed_tool_availability_bug",
        "detected": caught_1,
        "status": buggy_report["status"],
        "healthy_v8_eval_passed": healthy_report["checks"]["v8_evaluator_parity"]["oracle_passed"],
        "faulty_v8_eval_passed": buggy_report["checks"]["v8_evaluator_parity"]["oracle_passed"],
        "localized_check": "tool_consumption_monotonicity + v8_evaluator_parity",
        "defect_count": len(buggy_report["defects"]),
    })

    # 2. Truth firewall leak
    class _LeakingKernel(V8EnvironmentKernel):
        def observe(self, state):
            obs = super().observe(state)
            exp = state.world["hidden"]["expected"]
            return replace(obs, rendered_text=obs.rendered_text + f" expected={exp}")

    leak_report = validate_environment_kernel(_LeakingKernel(), template0_tool_worlds[:3])
    caught_2 = (
        leak_report["status"] == "INSTRUMENT_DEFECT_DETECTED"
        and not leak_report["checks"]["actor_truth_firewall"]["passed"]
    )
    cases.append({
        "fault_id": "truth_firewall_leak",
        "detected": caught_2,
        "status": leak_report["status"],
        "localized_check": "actor_truth_firewall",
        "defect_count": len(leak_report["defects"]),
    })

    # 3. Train/eval prompt drift
    def _drifted_render(**kwargs) -> str:
        return render_public_v8(**kwargs).replace("Action:", "AnswerNow:")

    drift_kernel = V8EnvironmentKernel(render_fn=_drifted_render)
    drift_report = validate_environment_kernel(drift_kernel, template0_tool_worlds[:3])
    caught_3 = (
        drift_report["status"] == "INSTRUMENT_DEFECT_DETECTED"
        and not drift_report["checks"]["train_rollout_tri_parity"]["passed"]
    )
    cases.append({
        "fault_id": "train_eval_prompt_drift",
        "detected": caught_3,
        "status": drift_report["status"],
        "localized_check": "train_rollout_tri_parity",
        "defect_count": len(drift_report["defects"]),
    })

    # 4. Disconnected policy head gradient
    spec_f = build_v8_model_spec("F", base_spec=_tiny_cpu_base_spec())
    model_f = build_v8_model(spec_f, 7, torch_module=torch)

    def _detached_val_loss(m):
        toks = torch.tensor([[1, 20, 21, 3]], dtype=torch.int64)
        segs = torch.zeros_like(toks)
        pos = torch.tensor([3], dtype=torch.int64)
        out = m.forward_action(toks, segs, pos)
        # Deliberately detach val_head path while computing op loss
        detached_val = out.log_p_val.detach()
        return -(out.log_p_op[0, 0] + detached_val[0, 0])

    grad_report = validate_model_and_gradients(
        model_f,
        _detached_val_loss,
        active_param_contract={"op_head.weight", "val_head.weight"},
    )
    caught_4 = (
        grad_report["status"] == "INSTRUMENT_DEFECT_DETECTED"
        and "missing_active_gradient:val_head.weight" in grad_report["defects"]
    )
    cases.append({
        "fault_id": "disconnected_policy_head_grad",
        "detected": caught_4,
        "status": grad_report["status"],
        "localized_check": "missing_active_gradient:val_head.weight",
        "defects": grad_report["defects"],
    })

    # 5. V16 epoch aggregation shadowing (int > (int, dict) bug)
    def _v16_buggy_aggregator(items: Sequence[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        final_epoch: tuple[int, dict[str, Any]] | None = None
        for _, data in items:
            ep = int(data["epoch"])
            try:
                # Reproduces tools/elendril_v716_report.py bug comparing int > tuple
                if final_epoch is None or ep > final_epoch:  # type: ignore[operator]
                    final_epoch = (ep, data)
            except TypeError:
                pass
        assert final_epoch is not None
        return {"final_epoch": final_epoch[0], "final_accuracy": final_epoch[1]["accuracy"]}

    agg_report = validate_metric_aggregator(_v16_buggy_aggregator)
    caught_5 = agg_report["status"] == "INSTRUMENT_DEFECT_DETECTED"
    cases.append({
        "fault_id": "v16_epoch_aggregation_shadowing",
        "detected": caught_5,
        "status": agg_report["status"],
        "localized_check": "validate_metric_aggregator",
        "defects": agg_report["defects"],
    })

    # 6. Checkpoint RNG corruption caught on store restore validation
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_data = prepare_v8_training_data("D1", d1_worlds_per_family=16, sample_limit_for_cpu=12)
        recipe = Recipe(seed=7, total_updates=4, batch_size=4, micro_batch_size=2, lr=1e-4, schedule="cosine", precision="fp32")
        src = "a" * 40
        trainer = V8TypedTrainer(
            train_data.typed_samples,
            v8_spec=spec_f,
            recipe=recipe,
            order_seed=7,
            source_commit=src,
            root=tmp_path / "ckpt",
            lineage="fault-rng",
            data_provenance=train_data.manifest,
            device="cpu",
            cpu_canary=True,
        )
        trainer.step()
        ckpt_sha = trainer.save()
        rng_file = trainer.store.objects / ckpt_sha / "rng.bin"
        raw = bytearray(rng_file.read_bytes())
        raw[0] ^= 0xFF
        rng_file.write_bytes(bytes(raw))
        caught_6 = False
        error_msg = ""
        try:
            trainer.store.restore(ckpt_sha)
        except ValueError as exc:
            caught_6 = True
            error_msg = str(exc)
        cases.append({
            "fault_id": "checkpoint_rng_corruption",
            "detected": caught_6,
            "status": "INSTRUMENT_DEFECT_DETECTED" if caught_6 else "MISSED",
            "localized_check": "checkpoint_store_restore",
            "error": error_msg,
        })

    sci_cal = run_scientific_shortcut_calibration()
    shield_report = run_level7_runtime_shield_suite()
    detected_count = sum(1 for c in cases if c["detected"])
    return {
        "schema": "anra-planthir-fault-injection-v1",
        "total_faults": len(cases),
        "detected_faults": detected_count,
        "scientific_shortcuts_rejected": bool(sci_cal["passed"]),
        "level7_shield_passed": bool(shield_report["passed"]),
        "all_detected": detected_count == len(cases) and bool(sci_cal["passed"]) and bool(shield_report["passed"]),
        "healthy_v8_kernel_passed": healthy_report["passed"],
        "cases": cases,
        "scientific_shortcut_calibration": sci_cal,
        "level7_runtime_shield": shield_report,
    }


def _teacher_signature(decision: Any) -> Any:
    if isinstance(decision, Mapping):
        return {str(k): float(v) for k, v in sorted(decision.items())}
    if hasattr(decision, "operation") and hasattr(decision, "argument"):
        return (str(decision.operation), decision.argument)
    if isinstance(decision, tuple) and len(decision) == 2:
        return (str(decision[0]), decision[1])
    return decision


def _signatures_equal(a: Any, b: Any, *, tol: float = 1e-6) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(abs(float(a[k]) - float(b[k])) <= tol for k in a)
    return a == b


def validate_nonanticipating_teacher(
    kernel: EnvironmentKernel,
    world_pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    teacher_fn: Callable[[Any, Mapping[str, Any], UniversalObservation], Any] | None = None,
    action_prefixes: Sequence[Sequence[UniversalAction]] | None = None,
    distribution_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Fail closed when two worlds share identical visible history/availability yet yield different teacher outputs."""

    if not world_pairs:
        return {
            "schema": "anra-planthir-nonanticipating-teacher-v1",
            "passed": False,
            "status": "NO_WORLD_PAIRS",
            "checked_states": 0,
            "comparable_pairs": 0,
            "leaks": [{"code": "NO_WORLD_PAIRS"}],
        }

    leaks: list[dict[str, Any]] = []
    checked_states = 0
    comparable_pairs = 0

    for idx, (world_a, world_b) in enumerate(world_pairs):
        prefixes: Sequence[Sequence[UniversalAction]] = action_prefixes if action_prefixes is not None else ((),)
        pair_had_comparable = False

        for p_idx, prefix in enumerate(prefixes):
            st_a = kernel.initial_state(world_a)
            st_b = kernel.initial_state(world_b)
            valid_prefix = True
            for step_t in range(len(prefix) + 1):
                if getattr(st_a, "terminal", False) or getattr(st_b, "terminal", False):
                    break
                obs_a = kernel.observe(st_a)
                obs_b = kernel.observe(st_b)
                vis_a = (
                    obs_a.rendered_text,
                    tuple(sorted(obs_a.available_tools.items())),
                    tuple(sorted(obs_a.consumed_tools)),
                    int(obs_a.decision_index),
                )
                vis_b = (
                    obs_b.rendered_text,
                    tuple(sorted(obs_b.available_tools.items())),
                    tuple(sorted(obs_b.consumed_tools)),
                    int(obs_b.decision_index),
                )
                if vis_a != vis_b:
                    break
                pair_had_comparable = True
                checked_states += 1

                if teacher_fn is not None:
                    dec_a = _teacher_signature(teacher_fn(st_a, world_a, obs_a))
                    dec_b = _teacher_signature(teacher_fn(st_b, world_b, obs_b))
                else:
                    dec_a = _teacher_signature(kernel.ideal_action(st_a))
                    dec_b = _teacher_signature(kernel.ideal_action(st_b))

                if not _signatures_equal(dec_a, dec_b, tol=distribution_tolerance):
                    leaks.append(
                        {
                            "code": "ANTICIPATING_TEACHER_LEAK",
                            "pair_index": idx,
                            "prefix_index": p_idx,
                            "step_index": step_t,
                            "world_a_id": str(world_a.get("world_id", f"a-{idx}")),
                            "world_b_id": str(world_b.get("world_id", f"b-{idx}")),
                            "decision_a": dec_a,
                            "decision_b": dec_b,
                        }
                    )
                    valid_prefix = False
                    break

                if step_t < len(prefix):
                    act = prefix[step_t]
                    op = str(act.operation if hasattr(act, "operation") else act[0])
                    arg = act.argument if hasattr(act, "operation") else act[1]
                    tool_key = _TOOL_OP_TO_KEY.get(op)
                    if tool_key is not None and not (
                        obs_a.available_tools.get(tool_key, False) and obs_b.available_tools.get(tool_key, False)
                    ):
                        break
                    st_a = kernel.step(st_a, op, arg)
                    st_b = kernel.step(st_b, op, arg)
            if not valid_prefix:
                break
        if pair_had_comparable:
            comparable_pairs += 1

    if comparable_pairs == 0 and not leaks:
        leaks.append({"code": "NO_COMPARABLE_VISIBLE_HISTORIES"})

    passed = len(leaks) == 0 and checked_states > 0
    return {
        "schema": "anra-planthir-nonanticipating-teacher-v1",
        "passed": passed,
        "status": "NONANTICIPATING_OK" if passed else "INSTRUMENT_DEFECT_DETECTED",
        "checked_states": checked_states,
        "comparable_pairs": comparable_pairs,
        "leak_count": len(leaks),
        "leaks": leaks,
    }


_DEFAULT_OP_NAMES = {0: "ANSWER", 1: "QUERY", 2: "RETRIEVE", 3: "ABSTAIN"}


def audit_sampled_strata(
    records_or_worlds: Sequence[Any],
    *,
    kernel: EnvironmentKernel | None = None,
    required_operations: Sequence[str] = (),
) -> dict[str, Any]:
    """Audit operation & family counts on the exact truncated/sampled dataset used for training or eval."""

    op_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for item in records_or_worlds:
        op_name: str | None = None
        fam_name = "unknown"
        if isinstance(item, Mapping):
            fam_name = str(item.get("family", item.get("schema", "unknown")))
            raw_op = item.get("ideal_operation", item.get("target_op", item.get("gold_op")))
            if isinstance(raw_op, int):
                op_name = _DEFAULT_OP_NAMES.get(raw_op, f"OP_{raw_op}")
            elif isinstance(raw_op, str) and raw_op:
                op_name = raw_op
            elif kernel is not None:
                st = kernel.initial_state(item)
                op_name = str(kernel.ideal_action(st)[0])
        else:
            fam_name = str(getattr(item, "family", getattr(item, "task_family", "unknown")))
            raw_op = getattr(item, "gold_op", getattr(item, "target_op", None))
            if isinstance(raw_op, int):
                op_name = _DEFAULT_OP_NAMES.get(raw_op, f"OP_{raw_op}")
            elif isinstance(raw_op, str) and raw_op:
                op_name = raw_op
        key_op = op_name or "UNKNOWN"
        op_counts[key_op] = op_counts.get(key_op, 0) + 1
        family_counts[fam_name] = family_counts.get(fam_name, 0) + 1

    missing_ops = [op for op in required_operations if op_counts.get(str(op), 0) <= 0]
    passed = len(records_or_worlds) > 0 and len(missing_ops) == 0
    interactive_ok = bool(
        op_counts.get("ANSWER", 0) > 0
        and (op_counts.get("QUERY", 0) + op_counts.get("RETRIEVE", 0)) > 0
        and op_counts.get("ABSTAIN", 0) > 0
    )
    return {
        "schema": "anra-planthir-sampled-strata-audit-v1",
        "passed": passed,
        "status": "STRATA_OK" if passed else "STRATA_DEFECT",
        "total_items": len(records_or_worlds),
        "operation_counts": dict(sorted(op_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "interactive_claim_eligible": interactive_ok,
        "required_operations": [str(x) for x in required_operations],
        "missing_required_operations": missing_ops,
    }


def run_scientific_shortcut_calibration(*, seed: int = 1801, families: int = 64) -> dict[str, Any]:
    """Calibrate whether Planthir distinguishes true traversal from timestamp-rank & terminal-copy shortcuts."""

    rng = random.Random(seed)
    alphabet = [chr(ord("A") + i) for i in range(16)]
    crossover_families: list[list[dict[str, Any]]] = []
    interactive_cases: list[dict[str, Any]] = []

    for f_idx in range(families):
        a, b, c, d, e, f, g, h = rng.sample(alphabet, 8)
        order = rng.sample(range(6), 6)
        ticks = rng.sample(range(1000, 9000), 6)
        fam_cases: list[dict[str, Any]] = []
        for swapped in (False, True):
            edges = [(a, c), (b, d), (c, f if swapped else e), (d, e if swapped else f), (e, g), (f, h)]
            rows = [(edges[i][0], edges[i][1], ticks[i]) for i in order]
            for query, answer in ((a, h if swapped else g), (b, g if swapped else h)):
                fam_cases.append({"rows": rows, "query": query, "answer": answer})
        crossover_families.append(fam_cases)

        # Missing internal edge interactive probes (1 solvable via QUERY, 1 unsolvable requiring ABSTAIN)
        hidden_row = (c, e, ticks[2])
        vis_partial = [r for idx_r, r in enumerate(rows) if idx_r != order.index(2)] if 2 in order else list(rows[:-1])
        # Build exact missing-edge case with C->E absent initially
        base_rows = [(a, c, ticks[0]), (c, e, ticks[2]), (e, g, ticks[4])]
        missing_init = [(a, c, ticks[0]), (e, g, ticks[4])]
        interactive_cases.append(
            {
                "kind": "missing_edge_queryable",
                "init_rows": missing_init,
                "query_payload": [(c, e, ticks[2])],
                "query_available": True,
                "query": a,
                "expected": ("QUERY", "ANSWER", g),
                "new_terminal_val": e,  # returned query row ends in E, whereas full 3-hop answer is G!
            }
        )
        interactive_cases.append(
            {
                "kind": "missing_edge_unsolvable",
                "init_rows": missing_init,
                "query_payload": [],
                "query_available": False,
                "query": a,
                "expected": ("ABSTAIN", "ABSTAIN", None),
                "new_terminal_val": g,
            }
        )

    def _eval_policy(policy_kind: str) -> dict[str, Any]:
        fam_ok = 0
        case_ok = 0
        inter_ok = 0
        for fam in crossover_families:
            hits = 0
            for case in fam:
                rows = case["rows"]
                if policy_kind == "timestamp_rank_shortcut":
                    pred = sorted(rows, key=lambda r: r[2])[2][1]
                elif policy_kind == "terminal_value_copy_shortcut":
                    pred = rows[-1][1]
                else:
                    edge_map = {s: v for s, v, _ in rows}
                    cur = case["query"]
                    for _ in range(3):
                        cur = edge_map[cur]
                    pred = cur
                if pred == case["answer"]:
                    case_ok += 1
                    hits += 1
            if hits == 4:
                fam_ok += 1

        for ic in interactive_cases:
            if policy_kind == "traversal_oracle":
                rows_now = list(ic["init_rows"])
                edge_map = {s: v for s, v, _ in rows_now}
                cur: str | None = ic["query"]
                resolved = True
                for _ in range(3):
                    if cur not in edge_map:
                        resolved = False
                        break
                    cur = edge_map[cur]
                if not resolved and ic["query_available"]:
                    rows_now.extend(ic["query_payload"])
                    edge_map = {s: v for s, v, _ in rows_now}
                    cur = ic["query"]
                    resolved = True
                    for _ in range(3):
                        if cur not in edge_map:
                            resolved = False
                            break
                        cur = edge_map[cur]
                    act_seq = ("ANSWER", cur) if resolved else ("ABSTAIN", None)
                    ok = act_seq == (ic["expected"][1], ic["expected"][2]) and ic["expected"][0] == "QUERY"
                elif not resolved:
                    ok = ic["expected"][0] == "ABSTAIN"
                else:
                    ok = False
                if ok:
                    inter_ok += 1
            elif policy_kind == "terminal_value_copy_shortcut":
                if ic["query_available"]:
                    act_seq = ("ANSWER", ic["new_terminal_val"])
                    ok = ic["expected"][0] == "QUERY" and act_seq == (ic["expected"][1], ic["expected"][2])
                else:
                    act_seq = ("ANSWER", ic["init_rows"][-1][1])
                    ok = act_seq[0] == ic["expected"][0]
                if ok:
                    inter_ok += 1
            else:
                act_seq = ("ANSWER", sorted(ic["init_rows"], key=lambda r: r[2])[-1][1])
                ok = act_seq == (ic["expected"][1], ic["expected"][2]) and ic["expected"][0] == "ANSWER"
                if ok:
                    inter_ok += 1

        comp_acc = float(fam_ok / max(1, len(crossover_families)))
        inter_acc = float(inter_ok / max(1, len(interactive_cases)))
        claims: list[str] = []
        if comp_acc == 1.0:
            claims.append("general_traversal")
        if inter_acc == 1.0:
            claims.append("interactive_tool_decision")
        return {
            "policy_id": policy_kind,
            "crossover_case_accuracy": float(case_ok / max(1, 4 * len(crossover_families))),
            "complete_family_accuracy": comp_acc,
            "interactive_tool_accuracy": inter_acc,
            "shortcut_detected": bool(comp_acc < 1.0 or inter_acc < 1.0),
            "qualifies_for_claims": claims,
        }

    policies = {
        name: _eval_policy(name)
        for name in ("timestamp_rank_shortcut", "terminal_value_copy_shortcut", "traversal_oracle")
    }
    passed = (
        policies["timestamp_rank_shortcut"]["shortcut_detected"] is True
        and policies["timestamp_rank_shortcut"]["complete_family_accuracy"] == 0.0
        and policies["timestamp_rank_shortcut"]["qualifies_for_claims"] == []
        and policies["terminal_value_copy_shortcut"]["shortcut_detected"] is True
        and policies["terminal_value_copy_shortcut"]["complete_family_accuracy"] == 0.0
        and policies["terminal_value_copy_shortcut"]["qualifies_for_claims"] == []
        and policies["traversal_oracle"]["shortcut_detected"] is False
        and policies["traversal_oracle"]["complete_family_accuracy"] == 1.0
        and policies["traversal_oracle"]["interactive_tool_accuracy"] == 1.0
        and set(policies["traversal_oracle"]["qualifies_for_claims"]) == {"general_traversal", "interactive_tool_decision"}
    )
    return {
        "schema": "anra-planthir-scientific-shortcut-calibration-v1",
        "passed": passed,
        "status": "SHORTCUTS_REJECTED_AND_ORACLE_QUALIFIED" if passed else "SCIENTIFIC_CALIBRATION_FAILED",
        "seed": seed,
        "families": families,
        "interactive_cases": len(interactive_cases),
        "policies": policies,
    }


def _xf_get(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj and obj[name] is not None:
                return obj[name]
            hidden = obj.get("hidden")
            if isinstance(hidden, Mapping) and name in hidden and hidden[name] is not None:
                return hidden[name]
        elif hasattr(obj, name):
            val = getattr(obj, name)
            if val is not None:
                return val
    return default


def _xf_norm_action(val: Any) -> tuple[str, str | None]:
    if isinstance(val, (tuple, list)) and len(val) >= 2:
        return (str(val[0]), None if val[1] is None else str(val[1]))
    if isinstance(val, Mapping):
        arg = val.get("argument")
        return (str(val.get("operation", "ANSWER")), None if arg is None else str(arg))
    if hasattr(val, "operation") and hasattr(val, "argument"):
        arg = getattr(val, "argument")
        return (str(getattr(val, "operation")), None if arg is None else str(arg))
    if val is None or str(val) in ("!", "ABSTAIN"):
        return ("ABSTAIN", None)
    if str(val) in ("Q", "QUERY"):
        return ("QUERY", None)
    if str(val) in ("R", "RETRIEVE"):
        return ("RETRIEVE", None)
    return ("ANSWER", str(val))


def verify_crossover_fixture(
    families: Sequence[Any],
    *,
    oracle_fn: Callable[[Any], Any] | None = None,
    blind_policies: Mapping[str, Callable[[Any], Any]] | None = None,
    erase_goal_fn: Callable[[Any], Any] | None = None,
    erase_bridge_fn: Callable[[Any], Any] | None = None,
    splits_by_name: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Verify a 4-cell crossover fixture (oracle=1.0, blind policies=0.0, goal/bridge erasures fail, split custody clean)."""
    fams = list(families)
    if not fams:
        raise ValueError("verify_crossover_fixture requires at least one 4-cell family")

    defects: list[dict[str, Any]] = []
    fam_cells: list[tuple[str, list[Any]]] = []
    for idx, fam in enumerate(fams):
        fid = str(_xf_get(fam, "family_id", "world_id", "id", default=f"fam_{idx}"))
        cells = list(_xf_get(fam, "cells", default=()) or ())
        if len(cells) != 4:
            defects.append({"code": "INVALID_FAMILY_CELL_COUNT", "family_id": fid, "count": len(cells)})
        fam_cells.append((fid, cells))

    def _eval_fn(fn: Callable[[Any], Any], transform: Callable[[Any], Any] | None = None) -> dict[str, float]:
        fam_ok = 0
        case_ok = 0
        total_cases = 0
        for _fid, cells in fam_cells:
            all_ok = bool(cells)
            for cell in cells:
                total_cases += 1
                exp_op = _xf_get(cell, "expected_operation", default=None)
                exp_ans = _xf_get(cell, "expected_answer", "expected", "target", default=None)
                expected = _xf_norm_action((exp_op, exp_ans) if exp_op is not None else exp_ans)
                probe_cell = transform(cell) if transform is not None else cell
                try:
                    pred = _xf_norm_action(fn(probe_cell))
                except (ValueError, KeyError, IndexError):
                    pred = ("ABSTAIN", None)
                ok = pred == expected
                if ok:
                    case_ok += 1
                else:
                    all_ok = False
            if all_ok:
                fam_ok += 1
        return {
            "complete_family_accuracy": float(fam_ok / max(1, len(fam_cells))),
            "case_accuracy": float(case_ok / max(1, total_cases)),
        }

    oracle_res: dict[str, Any] | None = None
    if oracle_fn is not None:
        oracle_res = _eval_fn(oracle_fn)
        if oracle_res["complete_family_accuracy"] < 1.0:
            defects.append({"code": "ORACLE_BELOW_UNITY", **oracle_res})

    blind_res: dict[str, dict[str, Any]] = {}
    for pname, pfn in dict(blind_policies or {}).items():
        stats = _eval_fn(pfn)
        rejected = stats["complete_family_accuracy"] == 0.0
        if not rejected:
            defects.append({"code": "BLIND_POLICY_SOLVED_CROSSOVER", "policy_id": pname, **stats})
        blind_res[str(pname)] = {**stats, "shortcut_rejected": rejected}

    erasure_res: dict[str, Any] = {}
    if oracle_fn is not None:
        for ename, efn in (("goal_erased", erase_goal_fn), ("bridge_erased", erase_bridge_fn)):
            if efn is not None:
                stats = _eval_fn(oracle_fn, transform=efn)
                dropped = stats["complete_family_accuracy"] < 1.0
                if not dropped:
                    defects.append({"code": f"ERASURE_NO_EFFECT:{ename}", **stats})
                erasure_res[ename] = {**stats, "accuracy_dropped": dropped}

    overlap_report: dict[str, Any] = {"checked": bool(splits_by_name), "overlaps": []}
    if splits_by_name:
        seen_prompts: dict[str, set[str]] = {}
        seen_ids: dict[str, set[str]] = {}
        for split_name, entries in splits_by_name.items():
            sname = str(split_name)
            for item in entries:
                sub_cells = list(_xf_get(item, "cells", default=()) or ()) or [item]
                for c in sub_cells:
                    prompt = str(_xf_get(c, "public_prompt", "rendered_text", "prompt", default="") or "").strip()
                    if prompt:
                        seen_prompts.setdefault(prompt, set()).add(sname)
                    cid = str(_xf_get(c, "case_id", "world_id", "id", default="") or "").strip()
                    if cid:
                        seen_ids.setdefault(cid, set()).add(sname)
        prompt_overlaps = [{"key": k, "splits": sorted(sp)} for k, sp in seen_prompts.items() if len(sp) > 1]
        id_overlaps = [{"key": k, "splits": sorted(sp)} for k, sp in seen_ids.items() if len(sp) > 1]
        all_overlaps = prompt_overlaps + id_overlaps
        overlap_report["overlaps"] = all_overlaps
        if all_overlaps:
            defects.append({"code": "INSTANTIATED_STATE_OVERLAP", "count": len(all_overlaps), "sample": all_overlaps[:5]})

    passed = len(defects) == 0
    return {
        "schema": "anra-planthir-crossover-fixture-v1",
        "passed": passed,
        "status": "FIXTURE_VERIFIED" if passed else "FIXTURE_DEFECT_DETECTED",
        "family_count": len(fam_cells),
        "oracle": oracle_res,
        "blind_policies": blind_res,
        "erasure_controls": erasure_res,
        "split_custody": overlap_report,
        "defects": defects,
    }


def score_crossover_panel(
    families: Sequence[Any],
    case_predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute 4-cell raw accuracy, latent_flip, min_signed_margin, signed interaction, and decoding gap."""
    fams = list(families)
    if not fams:
        raise ValueError("score_crossover_panel requires at least one family")

    per_family: dict[str, dict[str, Any]] = {}
    total_cases = 0
    case_correct = 0
    false_abst = 0
    malformed = 0

    for idx, fam in enumerate(fams):
        fid = str(_xf_get(fam, "family_id", "world_id", "id", default=f"fam_{idx}"))
        cells = list(_xf_get(fam, "cells", default=()) or ())
        if not cells:
            raise ValueError(f"family {fid!r} has no cells")

        raw_flags: list[bool] = []
        ap_flags: list[bool] = []
        s_vals: list[float] = []
        m_vals: list[float] = []
        m_by_cell: dict[str, float] = {}

        for c_idx, cell in enumerate(cells):
            cid = str(_xf_get(cell, "case_id", "world_id", "id", default=""))
            if cid not in case_predictions:
                raise KeyError(f"missing prediction record for case {cid!r}")
            pred = case_predictions[cid]
            ckey = str(_xf_get(cell, "cell_key", default=f"c{c_idx}"))

            total_cases += 1
            raw_ok = bool(pred.get("raw_exact_correct", pred.get("correct", False)))
            ap_ok = bool(pred.get("ap_constrained_correct", raw_ok))
            if raw_ok:
                case_correct += 1
            if bool(pred.get("false_abstention", False)):
                false_abst += 1
            if bool(pred.get("malformed_action", False)):
                malformed += 1

            s_i = float(pred["signed_margin"]) if pred.get("signed_margin") is not None else (1.0 if raw_ok else -1.0)
            m_gh = float(pred["endpoint_margin"]) if pred.get("endpoint_margin") is not None else s_i

            raw_flags.append(raw_ok)
            ap_flags.append(ap_ok)
            s_vals.append(s_i)
            m_vals.append(m_gh)
            m_by_cell[ckey] = m_gh

        if all(k in m_by_cell for k in ("q0_b0", "q0_b1", "q1_b0", "q1_b1")):
            interaction = (m_by_cell["q0_b0"] - m_by_cell["q0_b1"] - m_by_cell["q1_b0"] + m_by_cell["q1_b1"]) / 4.0
        elif len(m_vals) == 4:
            interaction = (m_vals[0] - m_vals[1] - m_vals[2] + m_vals[3]) / 4.0
        else:
            interaction = 0.0

        per_family[fid] = {
            "all4_raw_correct": all(raw_flags),
            "ap_constrained_all4_correct": all(ap_flags),
            "latent_flip": all(s > 0.0 for s in s_vals),
            "min_signed_margin": float(min(s_vals)),
            "signed_interaction": float(interaction),
        }

    n_fams = len(per_family)
    all4_acc = float(sum(1 for r in per_family.values() if r["all4_raw_correct"]) / n_fams)
    ap_all4_acc = float(sum(1 for r in per_family.values() if r["ap_constrained_all4_correct"]) / n_fams)
    latent_rate = float(sum(1 for r in per_family.values() if r["latent_flip"]) / n_fams)
    return {
        "schema": "anra-planthir-crossover-panel-v1",
        "family_count": n_fams,
        "case_count": total_cases,
        "case_accuracy": float(case_correct / max(1, total_cases)),
        "all4_raw_accuracy": all4_acc,
        "ap_constrained_all4_accuracy": ap_all4_acc,
        "decoding_gap": float(ap_all4_acc - all4_acc),
        "latent_flip_rate": latent_rate,
        "mean_min_signed_margin": float(sum(r["min_signed_margin"] for r in per_family.values()) / n_fams),
        "mean_signed_interaction": float(sum(r["signed_interaction"] for r in per_family.values()) / n_fams),
        "false_abstention_rate": float(false_abst / max(1, total_cases)),
        "malformed_rate": float(malformed / max(1, total_cases)),
        "per_family": per_family,
    }


ADVERSARIAL_BYTE_SEQUENCES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "lone_continuation_0xB5",
        "tokens": (1, 4 + 0xB5, 2),
        "expect_valid_utf8_default": False,
    },
    {
        "case_id": "truncated_2byte_0xC3",
        "tokens": (4 + ord("A"), 4 + 0xC3, 2),
        "expect_valid_utf8_default": False,
    },
    {
        "case_id": "truncated_4byte_0xF0_0x9F",
        "tokens": (4 + 0xF0, 4 + 0x9F, 4 + ord("B"), 2),
        "expect_valid_utf8_default": False,
    },
    {
        "case_id": "invalid_0xFF_0xFE",
        "tokens": (4 + 0xFF, 4 + 0xFE, 2),
        "expect_valid_utf8_default": False,
    },
    {
        "case_id": "embedded_0xB5_with_answer",
        "tokens": tuple(4 + b for b in b"ANSWER:ok") + (4 + 0xB5, 4 + ord("!"), 2),
        "expect_valid_utf8_default": False,
    },
    {
        "case_id": "special_tokens_only",
        "tokens": (0, 1, 3, 2),
        "expect_valid_utf8_default": True,
    },
    {
        "case_id": "stop_newline_before_bad_byte",
        "tokens": (4 + ord("4"), 4 + ord("2"), 4 + ord("\n"), 4 + 0xB5, 2),
        "expect_valid_utf8_default": True,
    },
    {
        "case_id": "empty_sequence",
        "tokens": (),
        "expect_valid_utf8_default": True,
    },
)


def validate_byte_decoder_resilience(
    decode_fn: Callable[[Sequence[int]], Any] | None = None,
) -> dict[str, Any]:
    """Probe a byte decoder against adversarial non-UTF-8 neural token sequences without crashing."""

    decoder = decode_fn if decode_fn is not None else decode_generated_bytes
    using_default = decode_fn is None
    defects: list[str] = []
    case_results: list[dict[str, Any]] = []

    for spec in ADVERSARIAL_BYTE_SEQUENCES:
        cid = str(spec["case_id"])
        toks = list(spec["tokens"])
        exp_valid = bool(spec["expect_valid_utf8_default"])
        try:
            raw_out = decoder(toks)
            if isinstance(raw_out, Mapping):
                text = str(raw_out.get("text", raw_out.get("raw_text", "")))
                valid_flag = raw_out.get("valid_utf8")
            elif isinstance(raw_out, tuple) and len(raw_out) == 2 and isinstance(raw_out[1], bool):
                text = str(raw_out[0])
                valid_flag = bool(raw_out[1])
            elif isinstance(raw_out, str):
                text = raw_out
                valid_flag = None
            else:
                text = str(raw_out)
                valid_flag = None

            if using_default and valid_flag is not exp_valid:
                defects.append(f"unexpected_valid_utf8:{cid}:got={valid_flag!r},expected={exp_valid!r}")
            elif (not exp_valid) and (valid_flag is True):
                defects.append(f"false_valid_utf8_reported:{cid}")
            case_results.append(
                {
                    "case_id": cid,
                    "passed": True,
                    "text": text,
                    "valid_utf8": valid_flag,
                }
            )
        except BaseException as exc:
            err = f"decoder_raised:{cid}:{type(exc).__name__}:{exc}"
            defects.append(err)
            case_results.append(
                {
                    "case_id": cid,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    passed = len(defects) == 0 and len(case_results) == len(ADVERSARIAL_BYTE_SEQUENCES)
    return {
        "schema": "anra-planthir-byte-decoder-shield-v1",
        "passed": passed,
        "status": "PASS" if passed else "INSTRUMENT_DEFECT_DETECTED",
        "using_default_decoder": using_default,
        "checked_cases": len(case_results),
        "cases": case_results,
        "defects": defects,
    }


def validate_end_to_end_lifecycle(
    policy: ObservablePolicy,
    kernel: EnvironmentKernel,
    worlds: Sequence[Mapping[str, Any]],
    *,
    decode_fn: Callable[[Sequence[int]], Any] | None = None,
    check_json_roundtrip: bool = True,
) -> dict[str, Any]:
    """Run integrated pre-flight over byte decoding, kernel contracts, policy rollouts, and JSON serialization."""

    import json
    from .execution import dump_json_safe

    world_list = [dict(w) for w in worlds]
    if not world_list:
        raise ValueError("validate_end_to_end_lifecycle requires at least one world")

    defects: list[str] = []
    byte_shield = validate_byte_decoder_resilience(decode_fn=decode_fn)
    if not byte_shield["passed"]:
        defects.extend(byte_shield["defects"])

    kernel_report = validate_environment_kernel(kernel, world_list)
    if not kernel_report["passed"]:
        defects.extend(kernel_report["defects"])

    rollout_records: list[dict[str, Any]] = []
    for idx, world in enumerate(world_list[:4]):
        try:
            if callable(getattr(policy, "reset_episode", None)):
                policy.reset_episode()  # type: ignore[attr-defined]
            state = kernel.initial_state(world)
            actions_taken: list[tuple[str, Any]] = []
            for _step_i in range(8):
                is_term = (
                    bool(kernel.is_terminal(state))
                    if callable(getattr(kernel, "is_terminal", None))
                    else (
                        bool(state.get("terminal", False))
                        if isinstance(state, Mapping)
                        else bool(getattr(state, "terminal", False))
                    )
                )
                if is_term:
                    break
                obs = kernel.observe(state)
                if callable(getattr(policy, "propose_with_telemetry", None)):
                    proposal = policy.propose_with_telemetry(obs)
                else:
                    proposal = policy.propose(obs)  # type: ignore[attr-defined]
                op_name = str(getattr(proposal, "operation", "ABSTAIN"))
                arg_val = getattr(proposal, "argument", None)
                actions_taken.append((op_name, arg_val))
                state = kernel.step(state, op_name, arg_val)
            ver = kernel.verify(world, actions_taken)
            rollout_records.append(
                {
                    "world_index": idx,
                    "steps": len(actions_taken),
                    "verified": dict(ver) if isinstance(ver, Mapping) else {"result": ver},
                }
            )
        except BaseException as exc:
            defects.append(f"rollout_failed:world_{idx}:{type(exc).__name__}:{exc}")

    json_roundtrip_ok = True
    if check_json_roundtrip:
        try:
            raw_json = dump_json_safe(
                {
                    "byte_shield": byte_shield,
                    "kernel_status": kernel_report["status"],
                    "rollouts": rollout_records,
                }
            )
            parsed = json.loads(raw_json)
            json_roundtrip_ok = isinstance(parsed, Mapping)
        except BaseException as exc:
            json_roundtrip_ok = False
            defects.append(f"json_roundtrip_failed:{type(exc).__name__}:{exc}")

    passed = len(defects) == 0 and bool(rollout_records) and json_roundtrip_ok
    return {
        "schema": "anra-planthir-lifecycle-preflight-v1",
        "passed": passed,
        "status": "PASS" if passed else "INSTRUMENT_DEFECT_DETECTED",
        "byte_decoder_shield": byte_shield,
        "kernel_passed": bool(kernel_report["passed"]),
        "rollouts_checked": len(rollout_records),
        "json_roundtrip_ok": json_roundtrip_ok,
        "defects": defects,
    }


def run_level7_runtime_shield_suite() -> dict[str, Any]:
    """Run adversarial Level-7 regression shields covering 0xB5 decode, FP16 -1e9, JSON NaN/Tensor, and archive caps."""

    import json
    import torch.nn as nn
    import torch.nn.functional as F

    from .archives import export_session_archives, verify_session_archive
    from .execution import dump_json_safe

    cases: list[dict[str, Any]] = []

    # 1. Unconstrained non-UTF-8 byte 0xB5 decoding (strict fails, Level-7 lossy shield survives)
    strict_raised = False
    try:
        UTF8.decode([UTF8.bos_id, 4 + 0xB5, UTF8.eos_id])
    except ValueError:
        strict_raised = True
    strict_caught = validate_byte_decoder_resilience(decode_fn=UTF8.decode)
    lossy_shield = validate_byte_decoder_resilience()
    bad_decoded = decode_generated_bytes([UTF8.bos_id, 4 + 0xB5, UTF8.eos_id])
    case1_passed = (
        strict_raised
        and (not strict_caught["passed"])
        and bool(lossy_shield["passed"])
        and (bad_decoded["valid_utf8"] is False)
    )
    cases.append(
        {
            "shield_id": "adversarial_0xB5_byte_emission",
            "passed": case1_passed,
            "strict_decode_raised": strict_raised,
            "strict_detector_blocked": not strict_caught["passed"],
            "lossy_shield_passed": lossy_shield["passed"],
            "sample_0xB5_valid_utf8": bad_decoded["valid_utf8"],
        }
    )

    # 2. FP16 autocast -1e9 mask overflow caught on uncast half mask and clean on FP32 softmax mask
    class _UnsafeMaskNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            q = self.proj(x)
            scores = torch.matmul(q, q.transpose(-2, -1))
            scores = scores.masked_fill(mask, -1e9)
            probs = F.softmax(scores, dim=-1)
            return (probs @ q).sum()

    class _SafeMaskNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            q = self.proj(x)
            scores = torch.matmul(q.float(), q.float().transpose(-2, -1))
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min / 2.0)
            probs = F.softmax(scores, dim=-1)
            return (probs @ q.float()).sum()

    x_probe = torch.ones((2, 4), dtype=torch.float32)
    mask_probe = torch.tensor([[False, True], [False, False]], dtype=torch.bool)
    unsafe_net = _UnsafeMaskNet()
    safe_net = _SafeMaskNet()
    unsafe_report = validate_fp16_autocast_forward_backward(
        unsafe_net,
        lambda m: m(x_probe, mask_probe),
    )
    safe_report = validate_fp16_autocast_forward_backward(
        safe_net,
        lambda m: m(x_probe, mask_probe),
        active_param_contract={"proj.weight", "proj.bias"},
    )
    case2_passed = (not unsafe_report["passed"]) and bool(safe_report["passed"])
    cases.append(
        {
            "shield_id": "fp16_autocast_minus_1e9_overflow",
            "passed": case2_passed,
            "unsafe_detected": not unsafe_report["passed"],
            "safe_passed": safe_report["passed"],
            "unsafe_defects": unsafe_report["defects"],
        }
    )

    # 3. JSON sanitization against NaN, Inf, Tensors, Paths, sets, and raw bytes
    dirty_payload: dict[str, Any] = {
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        "neg_inf_val": float("-inf"),
        "tensor_scalar": torch.tensor(2.5),
        "tensor_big": torch.zeros(5000),
        "path_obj": Path("ckpt/step_001.bin"),
        "raw_bytes": b"\xb5ok",
        "tag_set": {"alpha", "beta"},
    }
    raw_json_rejected = False
    try:
        json.dumps(dirty_payload, allow_nan=False)
    except (TypeError, ValueError):
        raw_json_rejected = True
    safe_text = dump_json_safe(dirty_payload)
    reparsed = json.loads(safe_text)
    case3_passed = (
        raw_json_rejected
        and reparsed["nan_val"] is None
        and reparsed["inf_val"] is None
        and reparsed["neg_inf_val"] is None
        and abs(float(reparsed["tensor_scalar"]) - 2.5) < 1e-6
        and reparsed["tensor_big"]["__tensor__"] is True
        and isinstance(reparsed["tag_set"], list)
    )
    cases.append(
        {
            "shield_id": "telemetry_json_tensor_nan_inf_sanitizer",
            "passed": case3_passed,
            "raw_json_rejected": raw_json_rejected,
        }
    )

    # 4. Safe archive ordering (ANALYSIS/EVIDENCE/FULL written before capped SUMMARY compaction)
    with tempfile.TemporaryDirectory(prefix="l7sh_", ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        src_dir.mkdir(parents=True, exist_ok=True)
        core_f = src_dir / "CORE_METRICS.json"
        core_f.write_text('{"ok": true}\n', encoding="utf-8")
        evid_f = src_dir / "JOB_RECEIPT.json"
        evid_f.write_text('{"job_id": "j1"}\n', encoding="utf-8")
        # Incompressible high-entropy payload so SUMMARY cap overflows and triggers compaction
        huge_f = src_dir / "HEAVY_TRACE.bin"
        huge_f.write_bytes(random.Random(7).randbytes(64 * 1024))

        receipt = export_session_archives(
            tmp_path,
            out_dir,
            prefix="SHIELD",
            summary_files={"CORE_METRICS.json": core_f, "HEAVY_TRACE.bin": huge_f},
            evidence_files={"CORE_METRICS.json": core_f, "JOB_RECEIPT.json": evid_f},
            analysis_files={"CORE_METRICS.json": core_f},
            full_files={"CORE_METRICS.json": core_f, "JOB_RECEIPT.json": evid_f, "HEAVY_TRACE.bin": huge_f},
            status="COMPLETE",
            source_commit="7" * 40,
            summary_max_bytes=12 * 1024,
            strict_summary_cap=False,
        )
        ver_full = verify_session_archive(receipt["archives"]["full"], expected_source_commit="7" * 40)
        ver_evid = verify_session_archive(receipt["archives"]["evidence"], expected_source_commit="7" * 40)
        ver_sum = verify_session_archive(
            receipt["archives"]["summary"],
            max_bytes=12 * 1024,
            expected_source_commit="7" * 40,
        )
        deliv = json.loads((out_dir / "DELIVERY_RECEIPT.json").read_text(encoding="utf-8"))
        case4_passed = (
            bool(ver_full["verified"])
            and bool(ver_evid["verified"])
            and bool(ver_sum["verified"])
            and bool(receipt.get("summary_compacted"))
            and "HEAVY_TRACE.bin" in receipt.get("summary_omitted_for_cap", [])
            and bool(deliv.get("summary_compacted"))
        )
    cases.append(
        {
            "shield_id": "archive_full_evidence_before_capped_summary",
            "passed": case4_passed,
            "summary_compacted": bool(receipt.get("summary_compacted")),
            "omitted_for_cap": receipt.get("summary_omitted_for_cap", []),
        }
    )

    passed = all(c["passed"] for c in cases)
    return {
        "schema": "anra-planthir-level7-runtime-shield-v1",
        "passed": passed,
        "status": "LEVEL7_SHIELD_PASS" if passed else "LEVEL7_SHIELD_FAILED",
        "total_shields": len(cases),
        "passed_shields": sum(1 for c in cases if c["passed"]),
        "shields": cases,
    }


__all__ = [
    "ADVERSARIAL_BYTE_SEQUENCES",
    "audit_sampled_strata",
    "run_fault_injection_suite",
    "run_level7_runtime_shield_suite",
    "run_scientific_shortcut_calibration",
    "score_crossover_panel",
    "validate_byte_decoder_resilience",
    "validate_end_to_end_lifecycle",
    "validate_environment_kernel",
    "validate_fp16_autocast_forward_backward",
    "validate_metric_aggregator",
    "validate_model_and_gradients",
    "validate_nonanticipating_teacher",
    "verify_crossover_fixture",
]
