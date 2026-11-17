"""Causal diagnostics & controlled counterfactual engine on frozen models."""

from __future__ import annotations

from copy import deepcopy
import random
from typing import Any, Callable, Mapping, Sequence

import torch

from anra_v5.elendril.codec import digest
from anra_v5.elendril.v8.contracts import SYMBOLS
from anra_v5.elendril.v8.holdout import canonical_graph_signature

from .contracts import EnvironmentKernel, ObservablePolicy, UniversalObservation


def _run_single_episode(
    policy: ObservablePolicy,
    kernel: EnvironmentKernel,
    world: Mapping[str, Any],
) -> tuple[bool, tuple[str, str | None] | None, list[tuple[str, str | None]]]:
    state = kernel.initial_state(world)
    executed: list[tuple[str, str | None]] = []
    for _ in range(8):
        obs = kernel.observe(state)
        action = policy.propose_with_telemetry(obs)
        executed.append((action.operation, action.argument))
        state = kernel.step(state, action.operation, action.argument)
        if getattr(state, "terminal", False):
            break
    outcome = kernel.verify(world, executed)
    final_act = executed[-1] if executed else None
    return bool(outcome.get("correct", False)), final_act, executed


def _paired_bootstrap_ci(
    diffs: Sequence[float],
    *,
    samples: int = 400,
    seed: int = 202611,
) -> list[float] | None:
    # A singleton or empty sample cannot estimate sampling uncertainty.
    if len(diffs) < 2:
        return None
    if samples < 2:
        raise ValueError("bootstrap requires at least two resamples")
    rng = random.Random(int(digest(["planthir-bootstrap-v1", seed, len(diffs)]), 16))
    n = len(diffs)
    means: list[float] = []
    for _ in range(samples):
        draw = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()
    lo = means[int(0.025 * samples)]
    hi = means[min(int(0.975 * samples), samples - 1)]
    return [float(lo), float(hi)]



def _clustered_bootstrap_ci(
    diffs: Sequence[float], worlds: Sequence[Mapping[str, Any]], *,
    assume_independent_clusters: bool,
) -> tuple[list[float] | None, int | None]:
    """Resample linked parent/pair groups only under an explicit independence assumption."""
    if not assume_independent_clusters or not worlds or len(diffs) != len(worlds):
        return None, None
    if any(not w.get("world_id") for w in worlds):
        return None, None
    parent: dict[str, str] = {}
    def find(k: str) -> str:
        parent.setdefault(k, k)
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)
    for w in worlds:
        wid = "world:" + str(w["world_id"])
        find(wid)
        if w.get("parent_world_id"):
            union(wid, "world:" + str(w["parent_world_id"]))
        if w.get("pair_id"):
            union(wid, "pair:" + str(w["pair_id"]))
        if w.get("base_world_id"):
            union(wid, "world:" + str(w["base_world_id"]))
    groups: dict[str, list[float]] = {}
    for w, diff in zip(worlds, diffs):
        groups.setdefault(find("world:" + str(w["world_id"])), []).append(float(diff))
    clusters = list(groups.values())
    if len(clusters) < 2:
        return None, len(clusters)
    rng = random.Random(int(digest(["planthir-cluster-bootstrap-v1", len(clusters), len(diffs)]), 16))
    means = []
    for _ in range(400):
        draw = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        means.append(sum(sum(group) for group in draw) / sum(len(group) for group in draw))
    means.sort()
    return [float(means[10]), float(means[390])], len(clusters)


def _probe_split_keys(worlds: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    """Collect lineage identifiers and, for V8-shaped worlds, graph signatures."""
    lineage: set[str] = set()
    signatures: set[str] = set()
    for w in worlds:
        for field in ("world_id", "parent_world_id", "base_world_id"):
            if w.get(field):
                lineage.add("world:" + str(w[field]))
        if w.get("pair_id"):
            lineage.add("pair:" + str(w["pair_id"]))
        if w.get("schema") == "elendril-x1-d1h-v1" or "structure_hash" in w:
            from anra_v5.elendril.v8x.data import canonical_x1_graph_signature

            signatures.add(canonical_x1_graph_signature(w))
        elif all(k in w for k in ("goal_subject", "hops", "initial_facts", "allowed_public_tools", "hidden")):
            signature = canonical_graph_signature(dict(w))
            if not signature["exact_canonicalization"]:
                raise ValueError("probe split cannot establish graph disjointness: inexact V8 canonicalization")
            signatures.add(str(signature["canonical_hash"]))
    return lineage, signatures

def run_oracle_information_intervention(
    policy: ObservablePolicy,
    kernel: EnvironmentKernel,
    worlds: Sequence[Mapping[str, Any]],
    *, assume_independent_clusters: bool = False,
) -> dict[str, Any]:
    """Compare interactive rollout vs pre-merged hidden tool payloads (isolating tool-use bottlenecks)."""

    if not worlds:
        raise ValueError("worlds cannot be empty")
    if not hasattr(kernel, "inject_oracle_information"):
        raise TypeError("kernel must provide inject_oracle_information(world)")

    normal_ok: list[int] = []
    oracle_ok: list[int] = []
    tool_diffs: list[float] = []
    tool_world_rows: list[Mapping[str, Any]] = []
    all_diffs: list[float] = []
    tool_normal = 0
    tool_oracle = 0
    tool_n = 0

    for w in worlds:
        c_norm, _, _ = _run_single_episode(policy, kernel, w)
        w_oracle = kernel.inject_oracle_information(w)  # type: ignore[attr-defined]
        c_orac, _, _ = _run_single_episode(policy, kernel, w_oracle)
        n_int = 1 if c_norm else 0
        o_int = 1 if c_orac else 0
        normal_ok.append(n_int)
        oracle_ok.append(o_int)
        all_diffs.append(float(o_int - n_int))

        tools = w.get("allowed_public_tools", {})
        if tools.get("query_available") or tools.get("memory_available"):
            tool_n += 1
            tool_normal += n_int
            tool_oracle += o_int
            tool_diffs.append(float(o_int - n_int))
            tool_world_rows.append(w)

    n = len(worlds)
    norm_acc = sum(normal_ok) / n
    orac_acc = sum(oracle_ok) / n
    all_ci, cluster_count = _clustered_bootstrap_ci(all_diffs, worlds, assume_independent_clusters=assume_independent_clusters)
    tool_ci, tool_cluster_count = _clustered_bootstrap_ci(tool_diffs, tool_world_rows, assume_independent_clusters=assume_independent_clusters)
    return {
        "intervention": "oracle_information_injection",
        "worlds": n,
        "normal_accuracy": norm_acc,
        "oracle_injected_accuracy": orac_acc,
        "delta_oracle": orac_acc - norm_acc,
        "delta_oracle_95ci": all_ci,
        "tool_worlds": tool_n,
        "tool_normal_accuracy": (tool_normal / tool_n) if tool_n else None,
        "tool_oracle_accuracy": (tool_oracle / tool_n) if tool_n else None,
        "tool_delta_oracle": ((tool_oracle - tool_normal) / tool_n) if tool_n else None,
        "tool_delta_oracle_95ci": tool_ci,
        "bootstrap_unit": "linked_base_world_cluster",
        "bootstrap_clusters": cluster_count,
        "tool_bootstrap_clusters": tool_cluster_count,
        "uncertainty_scope": "requires assumed independent linked-world clusters; conditional on supplied worlds and model, not training seeds",
    }


def run_subgoal_hop_decomposition(
    policy: ObservablePolicy,
    kernel: EnvironmentKernel,
    multi_hop_worlds: Sequence[Mapping[str, Any]],
    *, assume_independent_clusters: bool = False,
) -> dict[str, Any]:
    """Decompose multi-hop reasoning on identical visible facts into Hop-1, Hop-2, and Full Chain."""

    evaluated = 0
    hop1_ok = 0
    hop2_ok = 0
    both_ok = 0
    full_ok = 0
    gap_diffs: list[float] = []
    evaluated_worlds: list[Mapping[str, Any]] = []

    for w in multi_hop_worlds:
        subgoals = list(kernel.decompose_subgoals(w))
        by_stage = {sg.stage_name: sg for sg in subgoals}
        if "hop_1" not in by_stage or "hop_2_given_bridge" not in by_stage:
            continue

        sg1 = by_stage["hop_1"]
        sg2 = by_stage["hop_2_given_bridge"]
        c1, _, _ = _run_single_episode(policy, kernel, sg1.modified_world)
        c2, _, _ = _run_single_episode(policy, kernel, sg2.modified_world)
        visible_full = (
            kernel.inject_oracle_information(w)  # type: ignore[attr-defined]
            if hasattr(kernel, "inject_oracle_information")
            else w
        )
        c_full, _, _ = _run_single_episode(policy, kernel, visible_full)

        evaluated += 1
        hop1_ok += 1 if c1 else 0
        hop2_ok += 1 if c2 else 0
        joint = 1 if (c1 and c2) else 0
        both_ok += joint
        full_int = 1 if c_full else 0
        full_ok += full_int
        gap_diffs.append(float(joint - full_int))
        evaluated_worlds.append(w)

    if evaluated == 0:
        return {
            "intervention": "subgoal_hop_decomposition",
            "worlds_decomposed": 0,
            "hop1_accuracy": None,
            "hop2_given_bridge_accuracy": None,
            "joint_individual_hops_solvable": None,
            "full_chain_accuracy": None,
            "composition_gap": None,
            "composition_gap_95ci": None,
        }

    h1 = hop1_ok / evaluated
    h2 = hop2_ok / evaluated
    joint_acc = both_ok / evaluated
    full_acc = full_ok / evaluated
    gap_ci, cluster_count = _clustered_bootstrap_ci(gap_diffs, evaluated_worlds, assume_independent_clusters=assume_independent_clusters)
    return {
        "intervention": "subgoal_hop_decomposition",
        "worlds_decomposed": evaluated,
        "hop1_accuracy": h1,
        "hop2_given_bridge_accuracy": h2,
        "joint_individual_hops_solvable": joint_acc,
        "full_chain_accuracy": full_acc,
        "composition_gap": joint_acc - full_acc,
        "composition_gap_95ci": gap_ci,
        "bootstrap_unit": "linked_base_world_cluster",
        "bootstrap_clusters": cluster_count,
        "uncertainty_scope": "requires assumed independent linked-world clusters; conditional on supplied worlds and model, not training seeds",
    }


def _cyclic_shift_symbol(sym: str | None, shift: int = 1) -> str | None:
    if sym is None or sym not in SYMBOLS:
        return sym
    return SYMBOLS[(SYMBOLS.index(sym) + shift) % len(SYMBOLS)]


def _permute_world(world: Mapping[str, Any], shift: int = 1) -> dict[str, Any]:
    w = deepcopy(dict(world))
    w["world_id"] = f"{w['world_id']}::sym_perm_{shift}"
    w["goal_subject"] = _cyclic_shift_symbol(str(w["goal_subject"]), shift)
    for row in w["initial_facts"]:
        row["subject"] = _cyclic_shift_symbol(str(row["subject"]), shift)
        row["value"] = _cyclic_shift_symbol(row["value"], shift)
    for key in ("query_payload", "memory_payload"):
        for row in w["hidden"][key]:
            row["subject"] = _cyclic_shift_symbol(str(row["subject"]), shift)
            row["value"] = _cyclic_shift_symbol(row["value"], shift)
    w["hidden"]["expected"] = _cyclic_shift_symbol(w["hidden"]["expected"], shift)
    return w


def _reverse_world_rows(world: Mapping[str, Any]) -> dict[str, Any]:
    w = deepcopy(dict(world))
    w["world_id"] = f"{w['world_id']}::row_reverse"
    w["row_order_seed"] = 0
    w["initial_facts"] = list(reversed(w["initial_facts"]))
    w["hidden"]["query_payload"] = list(reversed(w["hidden"]["query_payload"]))
    w["hidden"]["memory_payload"] = list(reversed(w["hidden"]["memory_payload"]))
    return w


def _inject_distractor_chain(world: Mapping[str, Any]) -> dict[str, Any]:
    w = deepcopy(dict(world))
    w["world_id"] = f"{w['world_id']}::distractor"
    used: set[str] = {str(w["goal_subject"])}
    for row in list(w["initial_facts"]) + list(w["hidden"]["query_payload"]) + list(w["hidden"]["memory_payload"]):
        used.add(str(row["subject"]))
        if row["value"] is not None:
            used.add(str(row["value"]))
    unused = [s for s in SYMBOLS if s not in used]
    if len(unused) >= 3:
        d1, d2, d3 = unused[0], unused[1], unused[2]
        is_x1 = world.get("schema") == "elendril-x1-d1h-v1" or any(
            isinstance(r, Mapping) and "epoch" in r for r in w.get("initial_facts", ())
        )
        if is_x1:
            w["initial_facts"] = list(w["initial_facts"]) + [
                {"subject": d1, "source": "S0", "epoch": 1, "value": d2},
                {"subject": d2, "source": "S0", "epoch": 1, "value": d3},
            ]
        else:
            w["initial_facts"] = list(w["initial_facts"]) + [
                {"subject": d1, "source": "s0", "tick": 901, "value": d2},
                {"subject": d2, "source": "s0", "tick": 902, "value": d3},
            ]
    return w


def run_invariance_counterfactuals(
    policy: ObservablePolicy,
    kernel: EnvironmentKernel,
    worlds: Sequence[Mapping[str, Any]],
    *,
    mode_policies: Mapping[str, ObservablePolicy] | None = None,
) -> dict[str, Any]:
    """Test symbol permutation equivariance, row reversal, distractors, and decoder modes."""

    if not worlds:
        raise ValueError("worlds cannot be empty")

    n = len(worlds)
    base_ok = 0
    perm_ok = 0
    perm_equivariant = 0
    rev_ok = 0
    rev_invariant = 0
    dist_ok = 0
    dist_invariant = 0

    for w in worlds:
        c_base, act_base, _ = _run_single_episode(policy, kernel, w)
        c_perm, act_perm, _ = _run_single_episode(policy, kernel, _permute_world(w, shift=1))
        c_rev, act_rev, _ = _run_single_episode(policy, kernel, _reverse_world_rows(w))
        c_dist, act_dist, _ = _run_single_episode(policy, kernel, _inject_distractor_chain(w))

        base_ok += 1 if c_base else 0
        perm_ok += 1 if c_perm else 0
        rev_ok += 1 if c_rev else 0
        dist_ok += 1 if c_dist else 0

        if act_base is not None and act_perm is not None:
            expected_perm_arg = _cyclic_shift_symbol(act_base[1], 1) if act_base[0] == "ANSWER" else act_base[1]
            if (act_perm[0], act_perm[1]) == (act_base[0], expected_perm_arg):
                perm_equivariant += 1
        if act_base == act_rev:
            rev_invariant += 1
        if act_base == act_dist:
            dist_invariant += 1

    mode_results: dict[str, float] = {}
    if mode_policies:
        for mode_name, m_pol in sorted(mode_policies.items()):
            ok = sum(1 for w in worlds if _run_single_episode(m_pol, kernel, w)[0])
            mode_results[mode_name] = ok / n

    return {
        "intervention": "invariance_counterfactuals",
        "worlds": n,
        "baseline_accuracy": base_ok / n,
        "symbol_permutation": {
            "accuracy": perm_ok / n,
            "accuracy_delta": (perm_ok - base_ok) / n,
            "equivariance_rate": perm_equivariant / n,
        },
        "row_order_reversal": {
            "accuracy": rev_ok / n,
            "accuracy_delta": (rev_ok - base_ok) / n,
            "prediction_invariance_rate": rev_invariant / n,
        },
        "distractor_injection": {
            "accuracy": dist_ok / n,
            "accuracy_delta": (dist_ok - base_ok) / n,
            "prediction_invariance_rate": dist_invariant / n,
        },
        "decoder_modes": mode_results,
    }


def _fit_ridge_classifier(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int,
    *,
    l2_reg: float = 1.0,
) -> torch.Tensor:
    """Closed-form multi-class ridge probe with unpenalized bias column."""

    n, d = x_train.shape
    ones = torch.ones((n, 1), dtype=torch.float32)
    xa = torch.cat([x_train.float(), ones], dim=1)
    y_onehot = torch.zeros((n, num_classes), dtype=torch.float32)
    y_onehot.scatter_(1, y_train.long().unsqueeze(1), 1.0)
    reg_diag = torch.cat([torch.full((d,), float(l2_reg)), torch.tensor([1e-6])])
    gram = xa.T @ xa + torch.diag(reg_diag)
    rhs = xa.T @ y_onehot
    return torch.linalg.solve(gram, rhs)  # [D+1, C]


def _eval_ridge_classifier(weights: torch.Tensor, x_eval: torch.Tensor, y_eval: torch.Tensor) -> float:
    ones = torch.ones((x_eval.shape[0], 1), dtype=torch.float32)
    xa = torch.cat([x_eval.float(), ones], dim=1)
    logits = xa @ weights
    preds = logits.argmax(dim=-1)
    return float((preds == y_eval.long()).float().mean().item())


def run_representation_linear_probe(
    policy: ObservablePolicy,
    kernel: EnvironmentKernel,
    train_worlds: Sequence[Mapping[str, Any]],
    eval_worlds: Sequence[Mapping[str, Any]],
    *,
    l2_reg: float = 1.0,
) -> dict[str, Any]:
    """Fit frozen-backbone linear probes for ideal operation, Hop-1 symbol, and final symbol."""

    if len(train_worlds) < 4 or len(eval_worlds) < 2:
        raise ValueError("linear probe requires at least 4 train worlds and 2 eval worlds")

    train_lineage, train_graphs = _probe_split_keys(train_worlds)
    eval_lineage, eval_graphs = _probe_split_keys(eval_worlds)
    if (train_lineage & eval_lineage or train_graphs & eval_graphs
            or any(a == b for a in train_worlds for b in eval_worlds)):
        raise ValueError("probe training and evaluation worlds must be disjoint by lineage and V8 graph signature")

    op_vocab = ("ANSWER", "QUERY", "RETRIEVE", "ABSTAIN")

    def _collect_op_xy(worlds: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        obs_list: list[UniversalObservation] = []
        y_list: list[int] = []
        for w in worlds:
            st = kernel.initial_state(w)
            obs_list.append(kernel.observe(st))
            op, _ = kernel.ideal_action(st)
            y_list.append(op_vocab.index(op))
        reps = policy.extract_representations(obs_list)["decision_hidden"]
        return reps, torch.tensor(y_list, dtype=torch.int64)

    x_op_tr, y_op_tr = _collect_op_xy(train_worlds)
    x_op_ev, y_op_ev = _collect_op_xy(eval_worlds)
    w_op = _fit_ridge_classifier(x_op_tr, y_op_tr, len(op_vocab), l2_reg=l2_reg)
    probe_op_acc = _eval_ridge_classifier(w_op, x_op_ev, y_op_ev)

    def _collect_symbol_xy(
        worlds: Sequence[Mapping[str, Any]],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, list[bool]]:
        obs_list: list[UniversalObservation] = []
        y_hop1: list[int] = []
        y_final: list[int] = []
        head_ok: list[bool] = []
        for w in worlds:
            exp = w.get("hidden", {}).get("expected")
            if exp is None or exp not in SYMBOLS:
                continue
            vis_w = (
                kernel.inject_oracle_information(w)  # type: ignore[attr-defined]
                if hasattr(kernel, "inject_oracle_information")
                else w
            )
            st = kernel.initial_state(vis_w)
            obs = kernel.observe(st)
            obs_list.append(obs)
            y_final.append(SYMBOLS.index(str(exp)))

            subgoals = list(kernel.decompose_subgoals(vis_w))
            by_stage = {sg.stage_name: sg for sg in subgoals}
            if "hop_1" in by_stage and by_stage["hop_1"].expected_argument in SYMBOLS:
                y_hop1.append(SYMBOLS.index(str(by_stage["hop_1"].expected_argument)))
            else:
                y_hop1.append(SYMBOLS.index(str(exp)))

            act = policy.propose_with_telemetry(obs)
            head_ok.append(act.operation == "ANSWER" and act.argument == str(exp))

        if not obs_list:
            return None, None, None, []
        reps = policy.extract_representations(obs_list)["decision_hidden"]
        return (
            reps,
            torch.tensor(y_hop1, dtype=torch.int64),
            torch.tensor(y_final, dtype=torch.int64),
            head_ok,
        )

    x_tr, yh1_tr, yf_tr, _ = _collect_symbol_xy(train_worlds)
    x_ev, yh1_ev, yf_ev, head_ev_ok = _collect_symbol_xy(eval_worlds)

    probe_hop1_acc: float | None = None
    probe_final_acc: float | None = None
    head_final_acc: float | None = None
    if x_tr is not None and x_ev is not None and yh1_tr is not None and yh1_ev is not None and yf_tr is not None and yf_ev is not None:
        w_h1 = _fit_ridge_classifier(x_tr, yh1_tr, len(SYMBOLS), l2_reg=l2_reg)
        w_fin = _fit_ridge_classifier(x_tr, yf_tr, len(SYMBOLS), l2_reg=l2_reg)
        probe_hop1_acc = _eval_ridge_classifier(w_h1, x_ev, yh1_ev)
        probe_final_acc = _eval_ridge_classifier(w_fin, x_ev, yf_ev)
        head_final_acc = sum(1 for ok in head_ev_ok if ok) / len(head_ev_ok)

    # Multi-hop only linear probe accuracy when multi-hop eval worlds are present
    mh_eval = [w for w in eval_worlds if int(w.get("hops", 1)) >= 2 and w.get("hidden", {}).get("expected") in SYMBOLS]
    probe_mh_hop1: float | None = None
    probe_mh_final: float | None = None
    head_mh_final: float | None = None
    if len(mh_eval) >= 2 and x_tr is not None and yh1_tr is not None and yf_tr is not None:
        xm, ym_h1, ym_fin, hm_ok = _collect_symbol_xy(mh_eval)
        if xm is not None and ym_h1 is not None and ym_fin is not None:
            w_h1 = _fit_ridge_classifier(x_tr, yh1_tr, len(SYMBOLS), l2_reg=l2_reg)
            w_fin = _fit_ridge_classifier(x_tr, yf_tr, len(SYMBOLS), l2_reg=l2_reg)
            probe_mh_hop1 = _eval_ridge_classifier(w_h1, xm, ym_h1)
            probe_mh_final = _eval_ridge_classifier(w_fin, xm, ym_fin)
            head_mh_final = sum(1 for ok in hm_ok if ok) / len(hm_ok)

    return {
        "intervention": "representation_linear_probe",
        "train_worlds": len(train_worlds),
        "eval_worlds": len(eval_worlds),
        "probe_operation_accuracy": probe_op_acc,
        "probe_hop1_symbol_accuracy": probe_hop1_acc,
        "probe_final_symbol_accuracy": probe_final_acc,
        "model_head_final_accuracy_on_visible": head_final_acc,
        "head_minus_probe_gap": (
            (probe_final_acc - head_final_acc)
            if (probe_final_acc is not None and head_final_acc is not None)
            else None
        ),
        "multihop_probe_hop1_accuracy": probe_mh_hop1,
        "multihop_probe_final_accuracy": probe_mh_final,
        "multihop_head_final_accuracy": head_mh_final,
    }


class _SpecStepProxy:
    """Lightweight read-only proxy overriding recurrent_steps without mutating frozen specs."""

    def __init__(self, base: Any, recurrent_steps: int) -> None:
        self._base = base
        self.recurrent_steps = int(recurrent_steps)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def run_surgical_mechanism_ablation(
    policy: Any,
    kernel: EnvironmentKernel,
    worlds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Zero-retraining surgical ablation testing whether specialized modules causally contribute or are bypassed."""

    if not worlds:
        raise ValueError("worlds cannot be empty")

    baseline_outcomes: list[bool] = []
    baseline_first_actions: list[tuple[str, str | None] | None] = []
    for w in worlds:
        ok, _, executed = _run_single_episode(policy, kernel, w)
        baseline_outcomes.append(ok)
        baseline_first_actions.append(executed[0] if executed else None)
    baseline_acc = sum(1 for ok in baseline_outcomes if ok) / len(baseline_outcomes)

    v8_spec = getattr(policy, "v8_spec", None)
    model = getattr(policy, "model", None)
    arch = getattr(v8_spec, "architecture", None)

    mechanism_type = "none_baseline_control"
    ablation_supported = False
    ablated_outcomes: list[bool] = list(baseline_outcomes)
    ablated_first_actions: list[tuple[str, str | None] | None] = list(baseline_first_actions)

    if v8_spec is not None and arch == "P":
        mechanism_type = "pointer_copy_gate"
        ablation_supported = True
        ablated_policy = policy.__class__(
            model,
            v8_spec,
            model_identity=f"{getattr(policy, 'model_identity', 'policy')}:no-copy",
            mode=getattr(policy, "mode", "raw"),
            copy_enabled_override=False,
        )
        ablated_outcomes = []
        ablated_first_actions = []
        for w in worlds:
            ok, _, executed = _run_single_episode(ablated_policy, kernel, w)
            ablated_outcomes.append(ok)
            ablated_first_actions.append(executed[0] if executed else None)
    elif v8_spec is not None and arch == "U" and model is not None and hasattr(model, "v8_spec"):
        mechanism_type = "recurrent_refinement"
        ablation_supported = True
        orig_model_spec = model.v8_spec
        orig_policy_spec = policy.v8_spec
        zero_spec = _SpecStepProxy(orig_model_spec, recurrent_steps=0)
        try:
            model.v8_spec = zero_spec
            policy.v8_spec = _SpecStepProxy(orig_policy_spec, recurrent_steps=0)
            ablated_outcomes = []
            ablated_first_actions = []
            for w in worlds:
                ok, _, executed = _run_single_episode(policy, kernel, w)
                ablated_outcomes.append(ok)
                ablated_first_actions.append(executed[0] if executed else None)
        finally:
            model.v8_spec = orig_model_spec
            policy.v8_spec = orig_policy_spec
    elif v8_spec is not None and arch == "F-cap" and model is not None and hasattr(model, "cap_out"):
        mechanism_type = "value_residual_bottleneck"
        ablation_supported = True
        saved_w = model.cap_out.weight.detach().clone()
        saved_b = model.cap_out.bias.detach().clone() if getattr(model.cap_out, "bias", None) is not None else None
        try:
            with torch.no_grad():
                model.cap_out.weight.zero_()
                if saved_b is not None:
                    model.cap_out.bias.zero_()
            ablated_outcomes = []
            ablated_first_actions = []
            for w in worlds:
                ok, _, executed = _run_single_episode(policy, kernel, w)
                ablated_outcomes.append(ok)
                ablated_first_actions.append(executed[0] if executed else None)
        finally:
            with torch.no_grad():
                model.cap_out.weight.copy_(saved_w)
                if saved_b is not None:
                    model.cap_out.bias.copy_(saved_b)
    elif getattr(policy, "x1_spec", None) is not None and model is not None:
        x1_spec = policy.x1_spec
        x1_arch = getattr(x1_spec, "architecture", None)
        if x1_arch in ("X1-R", "X1-R-noedge", "R", "R-noedge") and hasattr(model, "r_to_ptr"):
            mechanism_type = "x1_relational_slot_readout"
            ablation_supported = True
            saved_w = model.r_to_ptr.weight.detach().clone()
            try:
                with torch.no_grad():
                    model.r_to_ptr.weight.zero_()
                ablated_outcomes = []
                ablated_first_actions = []
                for w in worlds:
                    ok, _, executed = _run_single_episode(policy, kernel, w)
                    ablated_outcomes.append(ok)
                    ablated_first_actions.append(executed[0] if executed else None)
            finally:
                with torch.no_grad():
                    model.r_to_ptr.weight.copy_(saved_w)
        elif x1_arch in ("X1-P-capR", "P-capR") and hasattr(model, "cap_r_out"):
            mechanism_type = "x1_query_residual_control"
            ablation_supported = True
            saved_w = model.cap_r_out.weight.detach().clone()
            saved_b = model.cap_r_out.bias.detach().clone() if getattr(model.cap_r_out, "bias", None) is not None else None
            try:
                with torch.no_grad():
                    model.cap_r_out.weight.zero_()
                    if saved_b is not None:
                        model.cap_r_out.bias.zero_()
                ablated_outcomes = []
                ablated_first_actions = []
                for w in worlds:
                    ok, _, executed = _run_single_episode(policy, kernel, w)
                    ablated_outcomes.append(ok)
                    ablated_first_actions.append(executed[0] if executed else None)
            finally:
                with torch.no_grad():
                    model.cap_r_out.weight.copy_(saved_w)
                    if saved_b is not None:
                        model.cap_r_out.bias.copy_(saved_b)
        elif x1_arch == "P":
            mechanism_type = "pointer_copy_gate"
            ablation_supported = True
            ablated_policy = policy.__class__(
                model,
                x1_spec,
                model_identity=f"{getattr(policy, 'model_identity', 'policy')}:no-copy",
                mode=getattr(policy, "mode", "raw"),
                copy_enabled_override=False,
            )
            ablated_outcomes = []
            ablated_first_actions = []
            for w in worlds:
                ok, _, executed = _run_single_episode(ablated_policy, kernel, w)
                ablated_outcomes.append(ok)
                ablated_first_actions.append(executed[0] if executed else None)
        elif x1_arch == "F-cap" and hasattr(model, "cap_out"):
            mechanism_type = "value_residual_bottleneck"
            ablation_supported = True
            saved_w = model.cap_out.weight.detach().clone()
            saved_b = model.cap_out.bias.detach().clone() if getattr(model.cap_out, "bias", None) is not None else None
            try:
                with torch.no_grad():
                    model.cap_out.weight.zero_()
                    if saved_b is not None:
                        model.cap_out.bias.zero_()
                ablated_outcomes = []
                ablated_first_actions = []
                for w in worlds:
                    ok, _, executed = _run_single_episode(policy, kernel, w)
                    ablated_outcomes.append(ok)
                    ablated_first_actions.append(executed[0] if executed else None)
            finally:
                with torch.no_grad():
                    model.cap_out.weight.copy_(saved_w)
                    if saved_b is not None:
                        model.cap_out.bias.copy_(saved_b)
        elif str(x1_arch).startswith("U-") or x1_arch == "U2-untied":
            mechanism_type = "recurrent_refinement"
            ablation_supported = True
            if hasattr(model, "x1_spec"):
                orig_model_spec = model.x1_spec
                orig_policy_spec = policy.x1_spec
                try:
                    model.x1_spec = _SpecStepProxy(orig_model_spec, recurrent_steps=0)
                    policy.x1_spec = _SpecStepProxy(orig_policy_spec, recurrent_steps=0)
                    ablated_outcomes = []
                    ablated_first_actions = []
                    for w in worlds:
                        ok, _, executed = _run_single_episode(policy, kernel, w)
                        ablated_outcomes.append(ok)
                        ablated_first_actions.append(executed[0] if executed else None)
                finally:
                    model.x1_spec = orig_model_spec
                    policy.x1_spec = orig_policy_spec

    ablated_acc = sum(1 for ok in ablated_outcomes if ok) / len(ablated_outcomes)
    agreements = sum(
        1 for a, b in zip(baseline_first_actions, ablated_first_actions) if a == b
    )
    agreement_rate = agreements / len(worlds)
    acc_delta = float(baseline_acc - ablated_acc)
    ornamental_bypass = bool(
        ablation_supported and agreement_rate >= 0.98 and abs(acc_delta) <= 1e-6
    )

    return {
        "intervention": "surgical_mechanism_ablation",
        "mechanism_type": mechanism_type,
        "ablation_supported": ablation_supported,
        "worlds": len(worlds),
        "baseline_accuracy": float(baseline_acc),
        "ablated_accuracy": float(ablated_acc),
        "accuracy_delta": acc_delta,
        "action_agreement_rate": float(agreement_rate),
        "ornamental_mechanism_bypass": ornamental_bypass,
    }


def run_compute_depth_scaling_intervention(
    policy: Any,
    kernel: EnvironmentKernel,
    worlds: Sequence[Mapping[str, Any]],
    *,
    step_grid: Sequence[int] = (0, 1, 2, 4),
) -> dict[str, Any]:
    """Sweep test-time recurrent depth without retraining to measure compute scaling & extrapolation."""

    if not worlds:
        raise ValueError("worlds cannot be empty")
    v8_spec = getattr(policy, "v8_spec", None)
    x1_spec = getattr(policy, "x1_spec", None)
    model = getattr(policy, "model", None)
    if (
        v8_spec is not None
        and getattr(v8_spec, "architecture", None) == "U"
        and model is not None
        and hasattr(model, "v8_spec")
    ):
        spec_attr = "v8_spec"
        active_spec = v8_spec
    elif (
        x1_spec is not None
        and (str(getattr(x1_spec, "architecture", "")).startswith("U-") or getattr(x1_spec, "architecture", "") == "U2-untied")
        and model is not None
        and hasattr(model, "x1_spec")
    ):
        spec_attr = "x1_spec"
        active_spec = x1_spec
    else:
        return {
            "intervention": "compute_depth_scaling",
            "supports_variable_depth": False,
            "trained_steps": 0,
            "accuracy_by_steps": {},
            "recurrence_gain_over_zero": 0.0,
            "extrapolation_gain_above_trained": 0.0,
        }

    trained_steps = int(active_spec.recurrent_steps)
    steps_to_eval = sorted({int(k) for k in step_grid if int(k) >= 0} | {0, trained_steps})
    orig_model_spec = getattr(model, spec_attr)
    orig_policy_spec = getattr(policy, spec_attr)
    acc_by_steps: dict[str, float] = {}

    try:
        for k in steps_to_eval:
            setattr(model, spec_attr, _SpecStepProxy(orig_model_spec, recurrent_steps=k))
            setattr(policy, spec_attr, _SpecStepProxy(orig_policy_spec, recurrent_steps=k))
            correct = sum(1 for w in worlds if _run_single_episode(policy, kernel, w)[0])
            acc_by_steps[str(k)] = float(correct / len(worlds))
    finally:
        setattr(model, spec_attr, orig_model_spec)
        setattr(policy, spec_attr, orig_policy_spec)

    trained_acc = acc_by_steps[str(trained_steps)]
    zero_acc = acc_by_steps["0"]
    higher = [acc_by_steps[str(k)] for k in steps_to_eval if k > trained_steps]
    extrap_gain = float(max(higher) - trained_acc) if higher else 0.0

    return {
        "intervention": "compute_depth_scaling",
        "supports_variable_depth": True,
        "trained_steps": trained_steps,
        "accuracy_by_steps": acc_by_steps,
        "recurrence_gain_over_zero": float(trained_acc - zero_acc),
        "extrapolation_gain_above_trained": extrap_gain,
    }


__all__ = [
    "run_compute_depth_scaling_intervention",
    "run_invariance_counterfactuals",
    "run_oracle_information_intervention",
    "run_representation_linear_probe",
    "run_subgoal_hop_decomposition",
    "run_surgical_mechanism_ablation",
]
