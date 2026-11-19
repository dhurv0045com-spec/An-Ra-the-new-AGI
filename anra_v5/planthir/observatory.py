"""Scientific observability: internal learning telemetry and 4-stage process funnel."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .contracts import EnvironmentKernel, ObservablePolicy, UniversalAction, UniversalObservation

_TOOL_OP_TO_KEY = {"QUERY": "query", "RETRIEVE": "retrieve"}

DIVERGENCE_TAXONOMY = (
    "NONE",
    "MODE_MASK_OVERRIDE",
    "REPEATED_CONSUMED_TOOL",
    "MISSED_REQUIRED_TOOL",
    "UNNECESSARY_TOOL_CALL",
    "PREMATURE_ABSTAIN",
    "FAILED_ABSTAIN_ON_UNSOLVABLE",
    "HOP1_VALUE_ERROR",
    "MULTI_HOP_CHAIN_ERROR",
)


def compute_effective_rank(hidden: torch.Tensor) -> float:
    """Compute Shannon effective rank exp(H(sigma / sum(sigma))) of centered [N, D] representations."""

    if hidden.ndim != 2:
        raise ValueError(f"expected 2D hidden tensor [N, D], got shape {tuple(hidden.shape)}")
    if hidden.shape[0] < 2:
        return 1.0
    centered = hidden.float() - hidden.float().mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(centered)
    pos = sv[sv > 1e-12]
    if pos.numel() == 0:
        return 1.0
    p = pos / pos.sum()
    entropy = -(p * torch.log(p)).sum().item()
    return float(math.exp(entropy))


def compute_representation_isotropy(hidden: torch.Tensor) -> dict[str, float | None]:
    """Measure directional spread using the normalized second-moment spectrum.

    A score of one requires energy equally spread across all feature dimensions;
    antipodal vectors on one axis therefore do not appear isotropic. This is
    sample-size limited: at most min(N, D) directions can be observed.
    """
    if hidden.ndim != 2:
        raise ValueError(f"expected 2D hidden tensor [N, D], got shape {tuple(hidden.shape)}")
    n, d = hidden.shape
    if n < 2 or d == 0:
        return {"mean_offdiag_cosine": None, "isotropy": None}
    normed = F.normalize(hidden.float(), dim=-1, eps=1e-8)
    if not torch.isfinite(normed).all():
        raise ValueError("hidden representations must be finite")
    cos = normed @ normed.transpose(0, 1)
    mask = ~torch.eye(n, dtype=torch.bool, device=cos.device)
    mean_offdiag = float(cos[mask].mean().item())
    singular = torch.linalg.svdvals(normed)
    eig = singular.square()
    total = eig.sum()
    if total <= 1e-12:
        return {"mean_offdiag_cosine": mean_offdiag, "isotropy": None}
    probabilities = eig / total
    participation = 1.0 / float(probabilities.square().sum().item())
    # Normalize so a one-dimensional collapse maps to zero when d > 1.
    isotropy = (participation - 1.0) / (d - 1) if d > 1 else 1.0
    return {"mean_offdiag_cosine": mean_offdiag, "isotropy": max(0.0, min(1.0, isotropy))}


def classify_parameter_group(param_name: str, custom_map: Mapping[str, str] | None = None) -> str:
    """Map parameter names across architectures into universal functional groups."""

    if custom_map:
        if param_name in custom_map:
            return str(custom_map[param_name])
        low_orig = param_name.lower()
        for pat, grp in custom_map.items():
            if pat.lower() in low_orig or param_name.startswith(pat):
                return str(grp)

    low = param_name.lower()
    if "embed" in low or low.startswith(("token_emb.", "pos_emb.")):
        return "embedding"
    if any(
        k in low
        for k in (
            "q_proj",
            "k_proj",
            "gate",
            "mem",
            "tok_ln",
            "tok_proj",
            "slots_init",
            "slot_",
            "agg_",
            "rel_",
            "edge_",
            "msg_",
            "noedge_",
            "goal_q_proj",
            "read_ln",
            "r_to_ptr",
            "cap_r_",
        )
    ):
        return "pointer_or_memory"
    if any(k in low for k in ("u_q", "u_k", "u_v", "u_out", "u_gru", "recur", "u_blocks")):
        return "recurrent"
    if any(k in low for k in ("head", "cap_in", "cap_out", "joint_head", "op_head", "val_head")):
        return "head"
    return "backbone"


class InternalTelemetryCollector:
    """Collect lightweight weight, gradient, representation, and mechanism telemetry."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def measure_snapshot(
        self,
        model: torch.nn.Module,
        *,
        step: int = 0,
        lr: float = 0.0,
        policy: ObservablePolicy | None = None,
        probe_observations: Sequence[UniversalObservation] = (),
    ) -> dict[str, Any]:
        groups: dict[str, dict[str, float]] = {}
        group_sq_w: dict[str, float] = {}
        group_sq_g: dict[str, float] = {}
        group_count: dict[str, int] = {}
        group_dormant: dict[str, int] = {}

        for name, param in model.named_parameters():
            grp = classify_parameter_group(name)
            w = param.detach().float()
            group_sq_w[grp] = group_sq_w.get(grp, 0.0) + float((w ** 2).sum().item())
            group_count[grp] = group_count.get(grp, 0) + int(w.numel())
            group_dormant[grp] = group_dormant.get(grp, 0) + int((w.abs() < 1e-6).sum().item())
            if param.grad is not None and torch.isfinite(param.grad).all():
                g = param.grad.detach().float()
                group_sq_g[grp] = group_sq_g.get(grp, 0.0) + float((g ** 2).sum().item())
            else:
                group_sq_g.setdefault(grp, 0.0)

        for grp in sorted(group_count):
            w_norm = math.sqrt(group_sq_w[grp])
            g_norm = math.sqrt(group_sq_g[grp])
            n_el = max(1, group_count[grp])
            groups[grp] = {
                "param_count": float(group_count[grp]),
                "weight_l2": w_norm,
                "grad_l2": g_norm,
                "update_to_weight_ratio": (lr * g_norm / w_norm) if w_norm > 0.0 else 0.0,
                "dormant_fraction": group_dormant[grp] / n_el,
            }

        rep_metrics: dict[str, Any] = {}
        mech_telemetry: dict[str, float] = {}
        if policy is not None and probe_observations:
            if callable(getattr(policy, "reset_episode", None)):
                policy.reset_episode()
            reps = policy.extract_representations(probe_observations)
            if "decision_hidden" in reps:
                h = reps["decision_hidden"]
                rep_metrics["effective_rank"] = compute_effective_rank(h)
                rep_metrics["hidden_dim"] = int(h.shape[-1])
                rep_metrics.update(compute_representation_isotropy(h))
            if "pointer_gate" in reps:
                gate_t = reps["pointer_gate"].float()
                mech_telemetry["pointer_gate_mean"] = float(gate_t.mean().item())
                mech_telemetry["pointer_gate_min"] = float(gate_t.min().item())
                mech_telemetry["pointer_gate_max"] = float(gate_t.max().item())
            if "pointer_entropy" in reps:
                mech_telemetry["pointer_attention_entropy_mean"] = float(reps["pointer_entropy"].float().mean().item())

            sample_obs = probe_observations[: min(8, len(probe_observations))]
            rec_means: list[float] = []
            rec_finals: list[float] = []
            for obs in sample_obs:
                if callable(getattr(policy, "reset_episode", None)):
                    policy.reset_episode()
                act = policy.propose_with_telemetry(obs)
                if "recurrent_mean_delta" in act.telemetry:
                    rec_means.append(act.telemetry["recurrent_mean_delta"])
                if "recurrent_final_delta" in act.telemetry:
                    rec_finals.append(act.telemetry["recurrent_final_delta"])
            if rec_means:
                mech_telemetry["recurrent_mean_delta"] = sum(rec_means) / len(rec_means)
                mech_telemetry["recurrent_final_delta"] = sum(rec_finals) / len(rec_finals)

        snapshot = {
            "step": step,
            "lr": lr,
            "parameter_groups": groups,
            "representation": rep_metrics,
            "mechanism_telemetry": mech_telemetry,
        }
        self.history.append(snapshot)
        return snapshot


@dataclass(frozen=True, slots=True)
class FunnelEpisodeTrace:
    """Structured rollout trace produced by ProcessFunnelAnalyzer."""

    world_id: str
    family: str
    hops: int
    expected: str | None
    correct: bool
    terminal_status: str
    first_divergence: str
    stage1_tool_need_match: bool
    initial_needed_tool: bool
    initial_called_tool: bool
    initial_called_valid_tool: bool
    repeated_consumed_tool: bool
    executed_actions: tuple[tuple[str, str | None], ...]
    ideal_first_action: tuple[str, str | None]
    confidence: float
    action_correct: bool
    telemetry_summary: dict[str, float]


def _classify_divergence(
    obs: UniversalObservation,
    action: UniversalAction,
    ideal_op: str,
    ideal_arg: str | None,
    hops: int,
) -> str:
    if (action.operation, action.argument) != (action.raw_operation, action.raw_argument):
        return "MODE_MASK_OVERRIDE"
    op = action.operation
    arg = action.argument
    if op == ideal_op and (op != "ANSWER" or arg == ideal_arg):
        return "NONE"
    if op in _TOOL_OP_TO_KEY:
        tkey = _TOOL_OP_TO_KEY[op]
        if tkey in obs.consumed_tools or not obs.available_tools.get(tkey, False):
            return "REPEATED_CONSUMED_TOOL"
        if ideal_op not in _TOOL_OP_TO_KEY:
            return "UNNECESSARY_TOOL_CALL"
        return "MISSED_REQUIRED_TOOL"  # Called wrong tool (e.g., QUERY when RETRIEVE was ideal)
    if ideal_op in _TOOL_OP_TO_KEY:
        return "MISSED_REQUIRED_TOOL"
    if op == "ABSTAIN" and ideal_op == "ANSWER":
        return "PREMATURE_ABSTAIN"
    if op == "ANSWER" and ideal_op == "ABSTAIN":
        return "FAILED_ABSTAIN_ON_UNSOLVABLE"
    if op == "ANSWER" and ideal_op == "ANSWER" and arg != ideal_arg:
        return "HOP1_VALUE_ERROR" if hops <= 1 else "MULTI_HOP_CHAIN_ERROR"
    return "MULTI_HOP_CHAIN_ERROR"


def _calibration_metrics(confidences: Sequence[float], outcomes: Sequence[bool], *, bins: int = 10) -> dict[str, float | None]:
    n = len(confidences)
    if n != len(outcomes):
        raise ValueError("confidence and outcome lengths must match")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if any(not math.isfinite(float(c)) or not 0 <= float(c) <= 1 for c in confidences):
        raise ValueError("confidences must be finite probabilities")
    if n == 0:
        return {"brier_score": None, "ece": None, "aurc": None}
    clipped = [max(0.0, min(1.0, float(c))) for c in confidences]
    ys = [1.0 if y else 0.0 for y in outcomes]
    brier = sum((p - y) ** 2 for p, y in zip(clipped, ys)) / n

    ece = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        members = [
            (p, y)
            for p, y in zip(clipped, ys)
            if (p >= lo and (p < hi or (b == bins - 1 and p <= hi)))
        ]
        if members:
            avg_p = sum(p for p, _ in members) / len(members)
            avg_y = sum(y for _, y in members) / len(members)
            ece += (len(members) / n) * abs(avg_p - avg_y)

    # Area Under Risk-Coverage Curve (lower risk is better)
    paired = sorted(zip(clipped, ys), key=lambda pair: pair[0], reverse=True)
    cum_errors = 0.0
    risks: list[float] = []
    for idx, (_, y) in enumerate(paired, start=1):
        cum_errors += 1.0 - y
        risks.append(cum_errors / idx)
    aurc = sum(risks) / n
    return {"brier_score": brier, "ece": ece, "aurc": aurc}


class ProcessFunnelAnalyzer:
    """Decompose policy execution across 4 causal process stages and divergence taxonomy."""

    def evaluate_suite(
        self,
        policy: ObservablePolicy,
        kernel: EnvironmentKernel,
        worlds: Sequence[Mapping[str, Any]],
        *,
        include_subgoals: bool = True,
    ) -> dict[str, Any]:
        if not worlds:
            raise ValueError("ProcessFunnelAnalyzer requires at least one world")

        traces: list[FunnelEpisodeTrace] = []
        subgoal_hop1_total = 0
        subgoal_hop1_correct = 0

        for world in worlds:
            if callable(getattr(policy, "reset_episode", None)):
                policy.reset_episode()
            state = kernel.initial_state(world)
            executed: list[tuple[str, str | None]] = []
            first_div = "NONE"
            repeated_consumed = False
            stage1_match = False
            initial_needed_tool = False
            initial_called_tool = False
            initial_called_valid_tool = False
            ideal_first: tuple[str, str | None] = ("ABSTAIN", None)
            confidences: list[float] = []
            action_correctness: list[bool] = []
            telemetry_accum: dict[str, list[float]] = {}

            for step_idx in range(8):
                obs = kernel.observe(state)
                ideal_op, ideal_arg = kernel.ideal_action(state)
                action = policy.propose_with_telemetry(obs)
                conf = math.exp(action.log_probability) if math.isfinite(action.log_probability) else 0.0
                confidences.append(max(0.0, min(1.0, conf)))
                action_correctness.append((action.operation, action.argument) == (ideal_op, ideal_arg))
                for tk, tv in action.telemetry.items():
                    if math.isfinite(tv):
                        telemetry_accum.setdefault(tk, []).append(float(tv))

                div = _classify_divergence(obs, action, ideal_op, ideal_arg, int(world.get("hops", 1)))
                if div == "REPEATED_CONSUMED_TOOL":
                    repeated_consumed = True
                if first_div == "NONE" and div != "NONE":
                    first_div = div

                if step_idx == 0:
                    ideal_first = (ideal_op, ideal_arg)
                    initial_needed_tool = ideal_op in _TOOL_OP_TO_KEY
                    initial_called_tool = action.operation in _TOOL_OP_TO_KEY
                    stage1_match = (action.operation == ideal_op) if initial_needed_tool else not initial_called_tool
                    if initial_called_tool:
                        tkey = _TOOL_OP_TO_KEY[action.operation]
                        initial_called_valid_tool = bool(obs.available_tools.get(tkey, False))

                executed.append((action.operation, action.argument))
                state = kernel.step(state, action.operation, action.argument)
                if bool(state.get("terminal", False)) if isinstance(state, Mapping) else bool(getattr(state, "terminal", False)):
                    break

            outcome = kernel.verify(world, executed)
            correct = bool(outcome.get("correct", False))
            if not correct and first_div == "NONE":
                first_div = "MULTI_HOP_CHAIN_ERROR"

            tel_summary = {k: sum(v) / len(v) for k, v in telemetry_accum.items() if v}
            traces.append(
                FunnelEpisodeTrace(
                    world_id=str(world["world_id"]),
                    family=str(world.get("family", "")),
                    hops=int(world.get("hops", 1)),
                    expected=world.get("hidden", {}).get("expected"),
                    correct=correct,
                    terminal_status=str(outcome.get("terminal_status", "")),
                    first_divergence=first_div,
                    stage1_tool_need_match=stage1_match,
                    initial_needed_tool=initial_needed_tool,
                    initial_called_tool=initial_called_tool,
                    initial_called_valid_tool=initial_called_valid_tool,
                    repeated_consumed_tool=repeated_consumed,
                    executed_actions=tuple(executed),
                    ideal_first_action=ideal_first,
                    confidence=confidences[-1] if confidences else 0.0,
                    action_correct=action_correctness[-1] if action_correctness else False,
                    telemetry_summary=tel_summary,
                )
            )

            if include_subgoals:
                for sg in kernel.decompose_subgoals(world):
                    if sg.stage_name == "hop_1":
                        sg_state = kernel.initial_state(sg.modified_world)
                        sg_obs = kernel.observe(sg_state)
                        sg_act = policy.propose_with_telemetry(sg_obs)
                        subgoal_hop1_total += 1
                        if (sg_act.operation, sg_act.argument) == (sg.expected_operation, sg.expected_argument):
                            subgoal_hop1_correct += 1

        n = len(traces)
        need_tool = [t for t in traces if t.initial_needed_tool]
        no_need_tool = [t for t in traces if not t.initial_needed_tool]
        valid_tool_callers = [t for t in traces if t.initial_called_valid_tool]
        stage2_disciplined = [
            t for t in valid_tool_callers if (not t.repeated_consumed_tool) and len(t.executed_actions) >= 2
        ]
        native_1hop = [t for t in traces if t.hops <= 1 and t.expected is not None]
        multi_hop = [t for t in traces if t.hops >= 2 and t.expected is not None]
        unsolvable = [t for t in traces if t.expected is None]

        def _acc(rows: Sequence[FunnelEpisodeTrace]) -> float | None:
            return (sum(1 for r in rows if r.correct) / len(rows)) if rows else None

        div_counts = Counter(t.first_divergence for t in traces)
        telemetry_means: dict[str, float] = {}
        all_tel_keys = {k for t in traces for k in t.telemetry_summary}
        for key in sorted(all_tel_keys):
            vals = [t.telemetry_summary[key] for t in traces if key in t.telemetry_summary]
            telemetry_means[key] = sum(vals) / len(vals)

        cal = _calibration_metrics([t.confidence for t in traces], [t.action_correct for t in traces])

        native_1hop_acc = _acc(native_1hop)
        subgoal_1hop_acc = (subgoal_hop1_correct / subgoal_hop1_total) if subgoal_hop1_total else None

        return {
            "schema": "anra-planthir-process-funnel-v1",
            "policy_id": str(getattr(policy, "policy_id", getattr(policy, "model_identity", type(policy).__name__))),
            "kernel_id": kernel.kernel_id,
            "episodes": n,
            "overall_accuracy": sum(1 for t in traces if t.correct) / n,
            "funnel": {
                "stage1_info_seeking": sum(1 for t in traces if t.stage1_tool_need_match) / n,
                "stage1_tool_recall_when_needed": (
                    sum(1 for t in need_tool if t.stage1_tool_need_match) / len(need_tool)
                ) if need_tool else None,
                "stage1_unnecessary_tool_rate": (
                    sum(1 for t in no_need_tool if t.initial_called_tool) / len(no_need_tool)
                ) if no_need_tool else None,
                "stage2_tool_discipline": (
                    len(stage2_disciplined) / len(valid_tool_callers)
                ) if valid_tool_callers else None,
                "stage3_single_hop_grounding": native_1hop_acc,
                "stage3_subgoal_hop1_grounding": subgoal_1hop_acc,
                "stage4_multi_hop_composition": _acc(multi_hop),
                "stage4_epistemic_abstention": _acc(unsolvable),
            },
            "counts": {
                "needed_tool": len(need_tool),
                "valid_tool_callers": len(valid_tool_callers),
                "native_1hop_answerable": len(native_1hop),
                "subgoal_hop1_probes": subgoal_hop1_total,
                "multi_hop_answerable": len(multi_hop),
                "unsolvable": len(unsolvable),
            },
            "divergence_taxonomy": {k: div_counts.get(k, 0) for k in DIVERGENCE_TAXONOMY},
            "calibration": {**cal, "target": "last_action_matches_ideal_action"},
            "telemetry_means": telemetry_means,
            "traces": [
                {
                    "world_id": t.world_id,
                    "family": t.family,
                    "hops": t.hops,
                    "correct": t.correct,
                    "terminal_status": t.terminal_status,
                    "first_divergence": t.first_divergence,
                    "executed_actions": [list(a) for a in t.executed_actions],
                    "confidence": t.confidence,
                }
                for t in traces
            ],
        }


def analyze_checkpoint_trajectory(
    checkpoint_records: Sequence[Any],
    *,
    load_fn: Any,
    kernel: EnvironmentKernel,
    eval_worlds: Sequence[Mapping[str, Any]],
    probe_train_worlds: Sequence[Mapping[str, Any]],
    probe_fn: Any,
) -> dict[str, Any]:
    """Track process funnel, representation rank, and linear probe lead across milestone checkpoints."""

    if not checkpoint_records:
        raise ValueError("checkpoint_records cannot be empty")
    if not eval_worlds:
        raise ValueError("eval_worlds cannot be empty")

    ordered = sorted(checkpoint_records, key=lambda r: (int(r.global_update), str(r.checkpoint_sha256)))
    analyzer = ProcessFunnelAnalyzer()
    obs_batch = [kernel.observe(kernel.initial_state(w)) for w in eval_worlds]
    points: list[dict[str, Any]] = []

    for rec in ordered:
        loaded = load_fn(rec, mode="raw")
        policy = loaded.get("policy")
        if policy is None:
            continue
        funnel = analyzer.evaluate_suite(policy, kernel, eval_worlds, include_subgoals=False)
        reps = policy.extract_representations(obs_batch)["decision_hidden"]
        eff_rank = compute_effective_rank(reps)
        iso = compute_representation_isotropy(reps)["isotropy"]
        probe = probe_fn(policy, kernel, probe_train_worlds, eval_worlds)
        fdict = funnel["funnel"]
        points.append({
            "global_update": int(rec.global_update),
            "checkpoint_sha256": str(rec.checkpoint_sha256),
            "overall_accuracy": float(funnel["overall_accuracy"]),
            "stage1_info_seeking": float(fdict["stage1_info_seeking"]),
            "stage3_single_hop_grounding": fdict.get("stage3_single_hop_grounding"),
            "stage4_multi_hop_composition": fdict.get("stage4_multi_hop_composition"),
            "effective_rank": float(eff_rank),
            "isotropy": float(iso) if iso is not None else None,
            "probe_final_symbol_accuracy": probe.get("probe_final_symbol_accuracy"),
            "head_minus_probe_gap": probe.get("head_minus_probe_gap"),
        })

    if not points:
        raise ValueError("no policy-enabled checkpoints found in checkpoint_records")

    lead_vals = [float(p["head_minus_probe_gap"]) for p in points if p.get("head_minus_probe_gap") is not None]
    max_lead = max(lead_vals) if lead_vals else 0.0
    acc_gain = float(points[-1]["overall_accuracy"] - points[0]["overall_accuracy"])
    rank_collapse = bool(
        len(points) >= 2 and points[0]["effective_rank"] > 0.0 and points[-1]["effective_rank"] < 0.70 * points[0]["effective_rank"]
    )
    return {
        "schema": "anra-planthir-checkpoint-trajectory-v1",
        "checkpoints_evaluated": len(points),
        "points": points,
        "accuracy_gain_first_to_last": acc_gain,
        "max_probe_lead_over_head": float(max_lead),
        "representation_leads_readout_detected": bool(max_lead >= 0.10),
        "rank_collapse_detected": rank_collapse,
    }


__all__ = [
    "DIVERGENCE_TAXONOMY",
    "FunnelEpisodeTrace",
    "InternalTelemetryCollector",
    "ProcessFunnelAnalyzer",
    "analyze_checkpoint_trajectory",
    "classify_parameter_group",
    "compute_effective_rank",
    "compute_representation_isotropy",
]
