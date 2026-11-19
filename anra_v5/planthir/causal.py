"""Causal diagnostics & controlled counterfactual engine on frozen models."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import random
from typing import Any, Callable, Mapping, Sequence

import torch

from anra_v5.elendril.codec import digest
from anra_v5.elendril.v8.contracts import SYMBOLS
from anra_v5.elendril.v8.holdout import canonical_graph_signature

from .contracts import EnvironmentKernel, ObservablePolicy, UniversalObservation


def summarize_diagnostic_coverage(
    worlds: Sequence[Mapping[str, Any]],
    *,
    target_families: Sequence[str] = (),
    seed_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Summarize sampled worlds, base clusters, families, views, seeds, and target-family coverage."""

    base_worlds: set[str] = set()
    families: set[str] = set()
    views: set[str] = set()
    seeds: set[int] = {int(s) for s in seed_ids}

    for idx, w in enumerate(worlds):
        base_id = w.get("base_world_id") or w.get("family_id") or w.get("world_id") or f"w_{idx}"
        fam_id = w.get("family") or w.get("family_id") or w.get("category") or "default"
        view_id = w.get("form") or w.get("view") or w.get("cell_key") or "canonical"
        base_worlds.add(str(base_id))
        families.add(str(fam_id))
        views.add(str(view_id))
        for sk in ("seed", "init_seed"):
            if w.get(sk) is not None:
                seeds.add(int(w[sk]))

    targets = sorted({str(f) for f in target_families if str(f)})
    covered = [f for f in targets if f in families]
    missing = [f for f in targets if f not in families]
    if targets:
        ratio = f"{len(covered)}/{len(targets)}"
        coverage_complete = len(missing) == 0
    else:
        ratio = "N/A"
        coverage_complete = len(worlds) > 0
    eligible = bool(coverage_complete and len(worlds) > 0)

    sorted_fams = sorted(families)
    if targets and not coverage_complete:
        fam_label = ",".join(sorted_fams) if len(sorted_fams) <= 4 else f"{len(sorted_fams)} families"
        summary = (
            f"{len(base_worlds)} base worlds ({len(worlds)} views) across {fam_label}; "
            f"target-family coverage {ratio} (missing {missing}). Mechanism unresolved."
        )
    else:
        summary = (
            f"{len(base_worlds)} base worlds ({len(worlds)} views) across {len(sorted_fams)} "
            f"family/families ({len(views)} view(s)), target-family coverage {ratio}."
        )

    return {
        "schema": "anra-planthir-diagnostic-coverage-v1",
        "sample_episodes": len(worlds),
        "distinct_base_world_count": len(base_worlds),
        "distinct_base_worlds": sorted(base_worlds),
        "distinct_family_count": len(sorted_fams),
        "distinct_families": sorted_fams,
        "distinct_view_count": len(views),
        "distinct_views": sorted(views),
        "distinct_seed_count": len(seeds),
        "distinct_seeds": sorted(seeds),
        "target_families": targets,
        "covered_target_families": covered,
        "missing_target_families": missing,
        "target_family_coverage_ratio": ratio,
        "coverage_complete": coverage_complete,
        "mechanism_resolution_eligible": eligible,
        "coverage_summary": summary,
    }


def stratify_intervention_worlds(
    worlds: Sequence[Mapping[str, Any]],
    *,
    max_worlds: int = 48,
    family_key: str = "family",
    base_key: str = "base_world_id",
    target_families: Sequence[str] = (),
) -> list[Mapping[str, Any]]:
    """Select base-world view groups round-robin across families so diagnostic samples cover all mechanisms."""

    limit = max(1, int(max_worlds))
    if len(worlds) <= limit:
        return list(worlds)

    by_fam_base: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    fam_order: list[str] = []
    seen_fams: set[str] = set()
    for idx, w in enumerate(worlds):
        fam = str(w.get(family_key) or w.get("family_id") or w.get("category") or "default")
        base = str(w.get(base_key) or w.get("family_id") or w.get("world_id") or f"w_{idx}")
        if fam not in seen_fams:
            seen_fams.add(fam)
            fam_order.append(fam)
        by_fam_base[fam][base].append(w)

    fam_queues: dict[str, list[list[Mapping[str, Any]]]] = {
        fam: list(by_fam_base[fam].values()) for fam in fam_order
    }
    target_set = {str(f) for f in target_families if str(f)}
    ordered_fams = [f for f in target_families if f in fam_queues]
    ordered_fams = list(dict.fromkeys(ordered_fams)) + [f for f in sorted(fam_queues) if f not in set(ordered_fams)]

    selected: list[Mapping[str, Any]] = []
    selected_ids: set[int] = set()

    # If full first groups exceed `limit`, ensure every family gets represented on pass 1 unless target_families fit whole.
    first_group_sizes = [len(fam_queues[f][0]) for f in ordered_fams if fam_queues[f]]
    full_groups_needed = sum(first_group_sizes)
    target_group_sum = sum(len(fam_queues[f][0]) for f in ordered_fams if f in target_set and fam_queues[f])
    use_target_whole = bool(target_set) and target_group_sum <= limit

    if not use_target_whole and len(ordered_fams) <= limit and full_groups_needed > limit:
        per_fam_pass1 = max(1, limit // len(ordered_fams))
        for fam in ordered_fams:
            if len(selected) >= limit:
                break
            q = fam_queues[fam]
            if not q:
                continue
            group = q[0]
            take_n = min(per_fam_pass1, limit - len(selected))
            for w in group[:take_n]:
                if id(w) not in selected_ids:
                    selected_ids.add(id(w))
                    selected.append(w)
            rem = group[take_n:]
            if rem:
                q[0] = rem
            else:
                q.pop(0)

    while len(selected) < limit:
        progressed = False
        for fam in ordered_fams:
            if len(selected) >= limit:
                break
            q = fam_queues[fam]
            if not q:
                continue
            group = q.pop(0)
            take = group[: limit - len(selected)]
            for w in take:
                if id(w) not in selected_ids:
                    selected_ids.add(id(w))
                    selected.append(w)
                    progressed = True
        if not progressed:
            break
    return selected


def _summarize_paired_episode_changes(
    worlds: Sequence[Mapping[str, Any]],
    baseline_outcomes: Sequence[bool],
    baseline_first_actions: Sequence[tuple[str, str | None] | None],
    intervention_outcomes: Sequence[bool],
    intervention_first_actions: Sequence[tuple[str, str | None] | None],
    *,
    baseline_final_actions: Sequence[tuple[str, str | None] | None] | None = None,
    intervention_final_actions: Sequence[tuple[str, str | None] | None] | None = None,
) -> dict[str, Any]:
    gained = 0
    lost = 0
    unchanged_ok = 0
    unchanged_bad = 0
    first_op_changed = 0
    first_act_changed = 0
    final_act_changed = 0
    changed_ids: list[str] = []

    fin_base = baseline_final_actions or [None] * len(worlds)
    fin_int = intervention_final_actions or [None] * len(worlds)
    for idx, (w, ok0, ok1, fa0, fa1, la0, la1) in enumerate(
        zip(worlds, baseline_outcomes, intervention_outcomes, baseline_first_actions, intervention_first_actions, fin_base, fin_int)
    ):
        if ok1 and not ok0:
            gained += 1
        elif ok0 and not ok1:
            lost += 1
        elif ok0 and ok1:
            unchanged_ok += 1
        else:
            unchanged_bad += 1

        op0 = fa0[0] if fa0 is not None else None
        op1 = fa1[0] if fa1 is not None else None
        if op0 != op1:
            first_op_changed += 1
        if fa0 != fa1:
            first_act_changed += 1
        if la0 != la1:
            final_act_changed += 1
        if (ok0 != ok1 or fa0 != fa1 or la0 != la1) and len(changed_ids) < 32:
            changed_ids.append(str(w.get("world_id", f"w_{idx}")))

    return {
        "total_episodes": len(worlds),
        "gained_correct_count": gained,
        "lost_correct_count": lost,
        "unchanged_correct_count": unchanged_ok,
        "unchanged_wrong_count": unchanged_bad,
        "net_correct_delta": gained - lost,
        "first_operation_changed_count": first_op_changed,
        "first_action_changed_count": first_act_changed,
        "final_action_changed_count": final_act_changed,
        "changed_world_ids": changed_ids,
    }


def _run_single_episode(
    policy: ObservablePolicy,
    kernel: EnvironmentKernel,
    world: Mapping[str, Any],
) -> tuple[bool, tuple[str, str | None] | None, list[tuple[str, str | None]]]:
    if callable(getattr(policy, "reset_episode", None)):
        policy.reset_episode()  # type: ignore[attr-defined]
    state = kernel.initial_state(world)
    executed: list[tuple[str, str | None]] = []
    max_steps = int(getattr(kernel, "max_actions", 8) or 8)
    for _ in range(max_steps):
        obs = kernel.observe(state)
        action = policy.propose_with_telemetry(obs)
        executed.append((action.operation, action.argument))
        state = kernel.step(state, action.operation, action.argument)
        if bool(state.get("terminal", False)) if isinstance(state, Mapping) else bool(getattr(state, "terminal", False)):
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


def _probe_split_keys(
    worlds: Sequence[Mapping[str, Any]],
    *,
    kernel: Any = None,
) -> tuple[set[str], set[str]]:
    """Collect lineage identifiers and structural signatures via kernel hook or V8/X1 fallback."""
    lineage: set[str] = set()
    signatures: set[str] = set()
    sig_fn = getattr(kernel, "structural_signature", None) if kernel is not None else None
    for w in worlds:
        for field in ("world_id", "parent_world_id", "base_world_id"):
            if w.get(field):
                lineage.add("world:" + str(w[field]))
        if w.get("pair_id"):
            lineage.add("pair:" + str(w["pair_id"]))
        if callable(sig_fn):
            signatures.add(str(sig_fn(w)))
        elif w.get("structural_signature"):
            signatures.add(str(w["structural_signature"]))
        elif w.get("schema") == "elendril-x1-d1h-v1" or "structure_hash" in w:
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
    *,
    assume_independent_clusters: bool = False,
    target_families: Sequence[str] = (),
    seed_ids: Sequence[int] = (),
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
        "coverage": summarize_diagnostic_coverage(worlds, target_families=target_families, seed_ids=seed_ids),
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


def _permute_world(world: Mapping[str, Any], shift: int = 1, *, kernel: Any = None) -> dict[str, Any]:
    if kernel is not None and callable(getattr(kernel, "permute_world", None)):
        return deepcopy(dict(kernel.permute_world(world, shift=shift)))
    w = deepcopy(dict(world))
    if "world_id" in w:
        w["world_id"] = f"{w['world_id']}::sym_perm_{shift}"
    if "goal_subject" in w:
        w["goal_subject"] = _cyclic_shift_symbol(str(w["goal_subject"]), shift)
    for row in w.get("initial_facts", ()):
        if isinstance(row, dict) and "subject" in row and "value" in row:
            row["subject"] = _cyclic_shift_symbol(str(row["subject"]), shift)
            row["value"] = _cyclic_shift_symbol(row["value"], shift)
    hidden = w.get("hidden")
    if isinstance(hidden, dict):
        for key in ("query_payload", "memory_payload"):
            for row in hidden.get(key, ()):
                if isinstance(row, dict) and "subject" in row and "value" in row:
                    row["subject"] = _cyclic_shift_symbol(str(row["subject"]), shift)
                    row["value"] = _cyclic_shift_symbol(row["value"], shift)
        if "expected" in hidden:
            hidden["expected"] = _cyclic_shift_symbol(hidden["expected"], shift)
    return w


def _reverse_world_rows(world: Mapping[str, Any], *, kernel: Any = None) -> dict[str, Any]:
    if kernel is not None and callable(getattr(kernel, "reverse_world", None)):
        return deepcopy(dict(kernel.reverse_world(world)))
    w = deepcopy(dict(world))
    if "world_id" in w:
        w["world_id"] = f"{w['world_id']}::row_reverse"
    w["row_order_seed"] = 0
    if isinstance(w.get("initial_facts"), list):
        w["initial_facts"] = list(reversed(w["initial_facts"]))
    hidden = w.get("hidden")
    if isinstance(hidden, dict):
        if isinstance(hidden.get("query_payload"), list):
            hidden["query_payload"] = list(reversed(hidden["query_payload"]))
        if isinstance(hidden.get("memory_payload"), list):
            hidden["memory_payload"] = list(reversed(hidden["memory_payload"]))
    return w


def _inject_distractor_chain(world: Mapping[str, Any], *, kernel: Any = None) -> dict[str, Any]:
    if kernel is not None and callable(getattr(kernel, "inject_distractor", None)):
        return deepcopy(dict(kernel.inject_distractor(world)))
    w = deepcopy(dict(world))
    if "world_id" in w:
        w["world_id"] = f"{w['world_id']}::distractor"
    if not isinstance(w.get("initial_facts"), list):
        return w
    hidden = w.get("hidden") if isinstance(w.get("hidden"), dict) else {}
    used: set[str] = {str(w.get("goal_subject", ""))}
    for row in list(w.get("initial_facts", ())) + list(hidden.get("query_payload", ())) + list(hidden.get("memory_payload", ())):
        if isinstance(row, Mapping):
            if "subject" in row:
                used.add(str(row["subject"]))
            if row.get("value") is not None:
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
    target_families: Sequence[str] = (),
    seed_ids: Sequence[int] = (),
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
        c_perm, act_perm, _ = _run_single_episode(policy, kernel, _permute_world(w, shift=1, kernel=kernel))
        c_rev, act_rev, _ = _run_single_episode(policy, kernel, _reverse_world_rows(w, kernel=kernel))
        c_dist, act_dist, _ = _run_single_episode(policy, kernel, _inject_distractor_chain(w, kernel=kernel))

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
        "coverage": summarize_diagnostic_coverage(worlds, target_families=target_families, seed_ids=seed_ids),
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

    train_lineage, train_graphs = _probe_split_keys(train_worlds, kernel=kernel)
    eval_lineage, eval_graphs = _probe_split_keys(eval_worlds, kernel=kernel)
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

    def _extract_target_symbol(world: Mapping[str, Any]) -> str | None:
        if callable(getattr(kernel, "target_symbol", None)):
            exp_val = kernel.target_symbol(world)  # type: ignore[attr-defined]
        else:
            hidden = world.get("hidden")
            exp_val = None
            if isinstance(hidden, Mapping):
                for key in ("expected", "expected_answer", "true_answer"):
                    if hidden.get(key) is not None:
                        exp_val = hidden.get(key)
                        break
            if exp_val is None:
                exp_val = world.get("target")
        if isinstance(exp_val, str) and len(exp_val) == 1 and exp_val in SYMBOLS:
            return exp_val
        return None

    def _collect_symbol_xy(
        worlds: Sequence[Mapping[str, Any]],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, list[bool]]:
        obs_list: list[UniversalObservation] = []
        y_hop1: list[int] = []
        y_final: list[int] = []
        head_ok: list[bool] = []
        for w in worlds:
            exp = _extract_target_symbol(w)
            if exp is None:
                continue
            vis_w = (
                kernel.inject_oracle_information(w)  # type: ignore[attr-defined]
                if hasattr(kernel, "inject_oracle_information")
                else w
            )
            st = kernel.initial_state(vis_w)
            obs = kernel.observe(st)
            obs_list.append(obs)
            y_final.append(SYMBOLS.index(exp))

            subgoals = list(kernel.decompose_subgoals(vis_w))
            by_stage = {sg.stage_name: sg for sg in subgoals}
            sg_arg = by_stage["hop_1"].expected_argument if "hop_1" in by_stage else None
            if isinstance(sg_arg, str) and len(sg_arg) == 1 and sg_arg in SYMBOLS:
                y_hop1.append(SYMBOLS.index(sg_arg))
            else:
                y_hop1.append(SYMBOLS.index(exp))

            if callable(getattr(policy, "reset_episode", None)):
                policy.reset_episode()
            act = policy.propose_with_telemetry(obs)
            head_ok.append(act.operation == "ANSWER" and act.argument == exp)

        if not obs_list:
            return None, None, None, []
        if callable(getattr(policy, "reset_episode", None)):
            policy.reset_episode()
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
    mh_eval = [w for w in eval_worlds if int(w.get("hops", 1)) >= 2 and _extract_target_symbol(w) is not None]
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
    *,
    target_families: Sequence[str] = (),
    seed_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Zero-retraining surgical ablation testing whether specialized modules causally contribute or are bypassed."""

    if not worlds:
        raise ValueError("worlds cannot be empty")

    def _run_pass(pol: Any) -> tuple[list[bool], list[tuple[str, str | None] | None], list[tuple[str, str | None] | None]]:
        outcomes: list[bool] = []
        firsts: list[tuple[str, str | None] | None] = []
        finals: list[tuple[str, str | None] | None] = []
        for w in worlds:
            ok, fin, executed = _run_single_episode(pol, kernel, w)
            outcomes.append(ok)
            firsts.append(executed[0] if executed else None)
            finals.append(fin)
        return outcomes, firsts, finals

    baseline_outcomes, baseline_first_actions, baseline_final_actions = _run_pass(policy)
    baseline_acc = sum(1 for ok in baseline_outcomes if ok) / len(baseline_outcomes)

    v8_spec = getattr(policy, "v8_spec", None)
    model = getattr(policy, "model", None)
    arch = getattr(v8_spec, "architecture", None)

    mechanism_type = "none_baseline_control"
    ablation_supported = False
    ablated_outcomes: list[bool] = list(baseline_outcomes)
    ablated_first_actions: list[tuple[str, str | None] | None] = list(baseline_first_actions)
    ablated_final_actions: list[tuple[str, str | None] | None] = list(baseline_final_actions)

    custom_ablate = getattr(policy, "ablate_mechanism", None) or getattr(model, "ablate_mechanism", None)
    if callable(custom_ablate):
        with custom_ablate() as mtype:
            if mtype:
                mechanism_type = str(mtype)
                ablation_supported = True
                ablated_outcomes, ablated_first_actions, ablated_final_actions = _run_pass(policy)
    elif v8_spec is not None and arch == "P":
        mechanism_type = "pointer_copy_gate"
        ablation_supported = True
        ablated_policy = policy.__class__(
            model,
            v8_spec,
            model_identity=f"{getattr(policy, 'model_identity', 'policy')}:no-copy",
            mode=getattr(policy, "mode", "raw"),
            copy_enabled_override=False,
        )
        ablated_outcomes, ablated_first_actions, ablated_final_actions = _run_pass(ablated_policy)
    elif v8_spec is not None and arch == "U" and model is not None and hasattr(model, "v8_spec"):
        mechanism_type = "recurrent_refinement"
        ablation_supported = True
        orig_model_spec = model.v8_spec
        orig_policy_spec = policy.v8_spec
        zero_spec = _SpecStepProxy(orig_model_spec, recurrent_steps=0)
        try:
            model.v8_spec = zero_spec
            policy.v8_spec = _SpecStepProxy(orig_policy_spec, recurrent_steps=0)
            ablated_outcomes, ablated_first_actions, ablated_final_actions = _run_pass(policy)
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
            ablated_outcomes, ablated_first_actions, ablated_final_actions = _run_pass(policy)
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
                ablated_outcomes, ablated_first_actions, ablated_final_actions = _run_pass(policy)
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
                ablated_outcomes, ablated_first_actions, ablated_final_actions = _run_pass(policy)
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
            ablated_outcomes, ablated_first_actions, ablated_final_actions = _run_pass(ablated_policy)
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
                ablated_outcomes, ablated_first_actions, ablated_final_actions = _run_pass(policy)
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
                    ablated_outcomes, ablated_first_actions, ablated_final_actions = _run_pass(policy)
                finally:
                    model.x1_spec = orig_model_spec
                    policy.x1_spec = orig_policy_spec

    ablated_acc = sum(1 for ok in ablated_outcomes if ok) / len(ablated_outcomes)
    agreements = sum(
        1 for a, b in zip(baseline_first_actions, ablated_first_actions) if a == b
    )
    agreement_rate = agreements / len(worlds)
    acc_delta = float(baseline_acc - ablated_acc)
    cov = summarize_diagnostic_coverage(worlds, target_families=target_families, seed_ids=seed_ids)
    paired_changes = _summarize_paired_episode_changes(
        worlds,
        baseline_outcomes,
        baseline_first_actions,
        ablated_outcomes,
        ablated_first_actions,
        baseline_final_actions=baseline_final_actions,
        intervention_final_actions=ablated_final_actions,
    )

    fam_label = ",".join(cov["distinct_families"]) if len(cov["distinct_families"]) <= 3 else f"{cov['distinct_family_count']} families"
    if cov["target_families"] and not cov["coverage_complete"]:
        ornamental_bypass = False
        mechanism_status = "MECHANISM_UNRESOLVED_MISSING_TARGET_FAMILIES"
        prefix = "Zero net change" if abs(acc_delta) <= 1e-6 else f"Net change {acc_delta:+.3f}"
        headline = (
            f"{prefix} on {cov['distinct_base_world_count']} {fam_label} worlds. "
            f"Target-family coverage {cov['target_family_coverage_ratio']}. Mechanism unresolved."
        )
    else:
        ornamental_bypass = bool(
            ablation_supported and agreement_rate >= 0.98 and abs(acc_delta) <= 1e-6
        )
        mechanism_status = (
            "ORNAMENTAL_MECHANISM_BYPASS"
            if ornamental_bypass
            else ("ACTIVE_MECHANISM" if ablation_supported and (abs(acc_delta) > 1e-6 or paired_changes["first_action_changed_count"] > 0) else "NO_ABLATION_SUPPORTED")
        )
        headline = (
            f"Delta {acc_delta:+.3f} on {cov['distinct_base_world_count']} {fam_label} worlds "
            f"(+{paired_changes['gained_correct_count']} / -{paired_changes['lost_correct_count']}, "
            f"first-op changed {paired_changes['first_operation_changed_count']})."
        )

    return {
        "intervention": "surgical_mechanism_ablation",
        "mechanism_type": mechanism_type,
        "ablation_supported": ablation_supported,
        "worlds": len(worlds),
        "coverage": cov,
        "paired_changes": paired_changes,
        "baseline_accuracy": float(baseline_acc),
        "ablated_accuracy": float(ablated_acc),
        "accuracy_delta": acc_delta,
        "action_agreement_rate": float(agreement_rate),
        "ornamental_mechanism_bypass": ornamental_bypass,
        "mechanism_status": mechanism_status,
        "headline": headline,
    }


def run_compute_depth_scaling_intervention(
    policy: Any,
    kernel: EnvironmentKernel,
    worlds: Sequence[Mapping[str, Any]],
    *,
    step_grid: Sequence[int] = (0, 1, 2, 4),
    target_families: Sequence[str] = (),
    seed_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Sweep test-time recurrent depth without retraining to measure compute scaling & extrapolation."""

    if not worlds:
        raise ValueError("worlds cannot be empty")
    v8_spec = getattr(policy, "v8_spec", None)
    x1_spec = getattr(policy, "x1_spec", None)
    model = getattr(policy, "model", None)
    depth_ctx = getattr(policy, "compute_depth_context", None) or getattr(model, "compute_depth_context", None)
    if callable(depth_ctx):
        trained_steps = int(getattr(policy, "trained_compute_steps", getattr(model, "trained_compute_steps", 0)))
        steps_to_eval = sorted({int(k) for k in step_grid if int(k) >= 0} | {0, trained_steps})
        acc_by_steps: dict[str, float] = {}
        for k in steps_to_eval:
            with depth_ctx(k):
                correct = sum(1 for w in worlds if _run_single_episode(policy, kernel, w)[0])
                acc_by_steps[str(k)] = float(correct / len(worlds))
        trained_acc = acc_by_steps[str(trained_steps)]
        zero_acc = acc_by_steps["0"]
        higher = [acc_by_steps[str(k)] for k in steps_to_eval if k > trained_steps]
        return {
            "intervention": "compute_depth_scaling",
            "supports_variable_depth": True,
            "coverage": summarize_diagnostic_coverage(worlds, target_families=target_families, seed_ids=seed_ids),
            "trained_steps": trained_steps,
            "accuracy_by_steps": acc_by_steps,
            "recurrence_gain_over_zero": float(trained_acc - zero_acc),
            "extrapolation_gain_above_trained": float(max(higher) - trained_acc) if higher else 0.0,
        }
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
            "coverage": summarize_diagnostic_coverage(worlds, target_families=target_families, seed_ids=seed_ids),
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
        "coverage": summarize_diagnostic_coverage(worlds, target_families=target_families, seed_ids=seed_ids),
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
    "stratify_intervention_worlds",
    "summarize_diagnostic_coverage",
]
