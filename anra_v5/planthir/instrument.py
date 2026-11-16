"""Instrument validation, invariant enforcement, and controlled fault injection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from hashlib import sha256
import random
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import torch

from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8.contracts import PublicInput, build_v8_model_spec
from anra_v5.elendril.v8.data import build_d1_training_dataset, render_public_v8
from anra_v5.elendril.v8.evaluation import run_episode as v8_run_episode
from anra_v5.elendril.v8.models import build_v8_model, compute_typed_nll
from anra_v5.elendril.v8.teacher import build_teacher_steps_for_world
from anra_v5.elendril.v8.training import V8TypedTrainer, prepare_v8_training_data
from anra_v5.elendril.foundation_training import Recipe

from .adapters.v8_adapter import V8EnvironmentKernel
from .contracts import EnvironmentKernel, UniversalObservation

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


def validate_model_and_gradients(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.nn.Module], torch.Tensor],
    *,
    active_param_contract: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Perform a non-mutating forward/backward graph integrity and active-parameter check."""

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
    defects: list[str] = []
    grad_norms: dict[str, float] = {}
    non_none_params: set[str] = set()
    nonzero_params: set[str] = set()
    loss_value: float | None = None

    try:
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

    passed = len(defects) == 0
    return {
        "schema": "anra-planthir-grad-check-v1",
        "passed": passed,
        "status": "PASS" if passed else "INSTRUMENT_DEFECT_DETECTED",
        "loss": loss_value,
        "non_none_grad_count": len(non_none_params),
        "nonzero_grad_count": len(nonzero_params),
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

    detected_count = sum(1 for c in cases if c["detected"])
    return {
        "schema": "anra-planthir-fault-injection-v1",
        "total_faults": len(cases),
        "detected_faults": detected_count,
        "all_detected": detected_count == len(cases),
        "healthy_v8_kernel_passed": healthy_report["passed"],
        "cases": cases,
    }


__all__ = [
    "run_fault_injection_suite",
    "validate_environment_kernel",
    "validate_metric_aggregator",
    "validate_model_and_gradients",
]
