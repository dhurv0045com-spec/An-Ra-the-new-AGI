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

    sci_cal = run_scientific_shortcut_calibration()
    detected_count = sum(1 for c in cases if c["detected"])
    return {
        "schema": "anra-planthir-fault-injection-v1",
        "total_faults": len(cases),
        "detected_faults": detected_count,
        "scientific_shortcuts_rejected": bool(sci_cal["passed"]),
        "all_detected": detected_count == len(cases) and bool(sci_cal["passed"]),
        "healthy_v8_kernel_passed": healthy_report["passed"],
        "cases": cases,
        "scientific_shortcut_calibration": sci_cal,
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


__all__ = [
    "audit_sampled_strata",
    "run_fault_injection_suite",
    "run_scientific_shortcut_calibration",
    "validate_environment_kernel",
    "validate_metric_aggregator",
    "validate_model_and_gradients",
    "validate_nonanticipating_teacher",
]
