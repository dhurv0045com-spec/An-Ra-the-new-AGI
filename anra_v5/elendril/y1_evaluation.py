"""AN-RA V8-Y1 evaluation, heuristic controls, cluster bootstrap, and fail-closed gates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import math
import random
from typing import Any

from .codec import canonical, digest
from .cognition import (
    CognitiveState,
    OperationCandidate,
    belief,
    run_episode,
    verify,
)
from .hitchhiker_data import public_case_for, render_hitchhiker_public
from .representation import UTF8
from .v6_data import SYMBOLS, public_action
from .v6_evaluation import DEFAULT_BUDGET, parse_action
from .y1_data import Y1Case
from .y1_protocol import (
    BOTTLENECK_CRITERIA,
    CONFIRMATION_GATE,
    PROTOCOL_SHA,
    SCREENING_GATE,
    SEEDS,
    SHORTCUT_HEURISTICS,
    T0_SPEC,
    T3_GROUP_CE_SPEC,
    TIE_BREAK_ORDER,
    derive_seed_int,
)


ORIGINAL_PRESERVATION_KINDS = ("conflict", "known", "memory", "missing", "null", "route", "unavailable")


def _first_divergence(
    case: Y1Case,
    trace: Any,
    outcome: Any,
) -> tuple[str, int | None, bool]:
    """Classify first wrong decision and whether queried/retrieved evidence reached context."""
    evidence_reached = any(d.operation in ("QUERY", "RETRIEVE") and d.argument is not None for d in trace.decisions)
    if outcome.correct:
        return "OK", None, evidence_reached
    if trace.status == "invalid_action":
        return "INVALID_FORMAT", max(0, len(trace.decisions) - 1), evidence_reached
    if trace.status == "budget_exhausted":
        return "BUDGET_EXHAUSTED", max(0, len(trace.decisions) - 1), evidence_reached
    if trace.decisions:
        d0 = trace.decisions[0]
        if d0.operation == "QUERY" and not case.environment.channels:
            return "ILLEGAL_ACTION", 0, evidence_reached
        if d0.operation == "RETRIEVE" and case.memory is None:
            return "ILLEGAL_ACTION", 0, evidence_reached
    if case.solvability == "query_needed" and (not trace.decisions or trace.decisions[0].operation != "QUERY"):
        return "MISSED_QUERY", 0, evidence_reached
    if case.solvability in ("unavailable", "null"):
        return "UNSAFE_ANSWER", max(0, len(trace.decisions) - 1), evidence_reached
    if trace.status == "abstained":
        return "FALSE_ABSTAIN", max(0, len(trace.decisions) - 1), evidence_reached
    if evidence_reached and len(trace.decisions) > 1:
        return "WRONG_ANSWER_POST_QUERY", len(trace.decisions) - 1, True
    return "WRONG_ANSWER_DIRECT", max(0, len(trace.decisions) - 1), evidence_reached


class Y1HeuristicPolicy:
    """Explicit preregistered heuristic or oracle baseline on exact Y1 public prompts."""

    model_identity = None

    def __init__(
        self,
        mode: str,
        *,
        surface: Any = None,
        memory_available: bool = False,
        truth: Any = None,
        majority_symbol: str = "A",
        seed: int = 7,
    ) -> None:
        if mode not in (
            "last-row-object",
            "nearest-to-query-object",
            "highest-timestamp-object",
            "global-most-frequent-answer",
            "always-abstain",
            "random-legal",
            "oracle-traversal",
        ):
            raise ValueError(f"unknown Y1 heuristic mode: {mode!r}")
        self.mode = mode
        self.surface = surface
        self.memory_available = memory_available
        self.truth = truth
        self.majority_symbol = majority_symbol
        self.seed = seed
        self.origin = "privileged_oracle" if mode == "oracle-traversal" else "fixed_baseline"
        self.decoder_schema = f"heuristic:{mode}"

    def for_case(self, case: Any, *, memory_available: bool | None = None) -> "Y1HeuristicPolicy":
        return Y1HeuristicPolicy(
            self.mode,
            surface=case.surface,
            memory_available=(case.memory is not None if memory_available is None else memory_available),
            truth=case.truth if self.mode == "oracle-traversal" else None,
            majority_symbol=self.majority_symbol,
            seed=self.seed,
        )

    def propose(self, state: CognitiveState, legal: frozenset[str], channels: tuple[str, ...], positions: int) -> OperationCandidate:
        if self.mode == "oracle-traversal":
            if self.truth is None or state.goal.episode != self.truth.episode:
                raise ValueError("oracle episode mismatch")
            return OperationCandidate("ABSTAIN") if self.truth.expected is None else OperationCandidate("ANSWER", self.truth.expected)
        if self.mode == "always-abstain":
            return OperationCandidate("ABSTAIN")
        if self.mode == "global-most-frequent-answer":
            return OperationCandidate("ANSWER", self.majority_symbol)
        if self.mode == "last-row-object":
            # Copies object of the last serialized fact row (positional shortcut).
            val = state.events[-1].value if state.events else None
            return OperationCandidate("ABSTAIN") if val is None else OperationCandidate("ANSWER", val)
        if self.mode == "nearest-to-query-object":
            # Copies object of the first serialized fact row immediately below query header (row 0).
            val = state.events[0].value if state.events else None
            return OperationCandidate("ABSTAIN") if val is None else OperationCandidate("ANSWER", val)
        if self.mode == "highest-timestamp-object":
            # Copies object of the highest-timestamp fact (tie-breaking on last serialized order).
            if not state.events:
                return OperationCandidate("ABSTAIN")
            best = max(enumerate(state.events), key=lambda pair: (pair[1].tick, pair[0]))[1]
            return OperationCandidate("ABSTAIN") if best.value is None else OperationCandidate("ANSWER", best.value)
        # random-legal
        rng = random.Random(derive_seed_int("y1-random-legal", self.seed, state.render(), channels, self.memory_available))
        options: list[tuple[str, str | None]] = [("ANSWER", s) for s in SYMBOLS] + [("ABSTAIN", None)]
        if "QUERY" in legal and channels:
            options.append(("QUERY", channels[0]))
        if "RETRIEVE" in legal and self.memory_available:
            options.append(("RETRIEVE", None))
        return OperationCandidate(*rng.choice(options))


def evaluate_y1_cases(
    cases: tuple[Y1Case, ...],
    policy_factory: Any,
    *,
    job_id: str,
    decoder_condition: str = "raw",
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Run public-only interactive episode rollout across cases and emit replayable prediction rows."""
    if not cases:
        raise ValueError("evaluate_y1_cases requires non-empty cases")
    predictions: list[dict[str, Any]] = []
    budget = DEFAULT_BUDGET

    for case in cases:
        is_oracle = getattr(policy_factory, "mode", None) == "oracle-traversal"
        actor_case = case.v6_case if is_oracle else public_case_for(case.v6_case)
        actor = policy_factory.for_case(actor_case, memory_available=(case.memory is not None))
        if actor_case is case.v6_case and getattr(actor, "origin", None) != "privileged_oracle":
            raise ValueError("oracle case leaked to non-oracle policy")
        trace = run_episode(case.goal, case.environment, actor, budget=budget, memory=case.memory)
        outcome = verify(trace, case.truth)
        div_cat, div_idx, ev_reached = _first_divergence(case, trace, outcome)

        first_raw = tuple(trace.decisions[0].raw_tokens) if trace.decisions else ()
        ended_eos = bool(first_raw) and first_raw[-1] == UTF8.eos_id
        predictions.append(
            {
                "job_id": job_id,
                "checkpoint_sha256": checkpoint_sha256,
                "case_id": case.case_id,
                "world_id": case.world_id,
                "family_id": case.family_id,
                "pair_id": case.pair_id,
                "panel": case.panel,
                "factor": case.factor,
                "variant_index": case.variant_index,
                "hops": case.hops,
                "solvability": case.solvability,
                "applicable": case.applicable,
                "decoder_condition": decoder_condition,
                "decisions": [
                    {
                        "operation": d.operation,
                        "argument": d.argument,
                        "raw_tokens": list(d.raw_tokens),
                        "ended_eos": bool(d.raw_tokens) and d.raw_tokens[-1] == UTF8.eos_id,
                        "model_calls": d.model_calls,
                        "model_positions": d.model_positions,
                    }
                    for d in trace.decisions
                ],
                "first_action": trace.decisions[0].operation if trace.decisions else "NONE",
                "first_argument": trace.decisions[0].argument if trace.decisions else None,
                "ended_eos": ended_eos,
                "trace_status": trace.status,
                "final_answer": trace.answer,
                "valid": trace.status not in ("invalid_action", "budget_exhausted"),
                "correct": bool(outcome.correct),
                "first_divergence": div_cat,
                "first_divergence_decision_index": div_idx,
                "evidence_reached_context": ev_reached,
                "requests": trace.requests,
                "model_calls": sum(d.model_calls for d in trace.decisions),
                "model_positions": sum(d.model_positions for d in trace.decisions),
                "origin": getattr(actor, "origin", "model"),
            }
        )
    return {
        "schema": "anra-v8-y1-predictions-v1",
        "job_id": job_id,
        "decoder_condition": decoder_condition,
        "checkpoint_sha256": checkpoint_sha256,
        "predictions": predictions,
    }


def summarize_y1_predictions(
    cases: tuple[Y1Case, ...],
    pred_doc: dict[str, Any],
    *,
    historical_continuity_cases: tuple[Y1Case, ...] = (),
    historical_pred_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute exact primary endpoint, applicable-only effects, panels, preservation, and failure audit."""
    by_cid = {c.case_id: c for c in cases}
    rows = pred_doc["predictions"]
    if len(rows) != len(cases) or {r["case_id"] for r in rows} != set(by_cid):
        raise ValueError("prediction rows do not match case inventory 1-to-1")

    # 1. Primary endpoint: mean across hops 1..4 of known-evidence directly answerable, macro-averaged over Case 0 & Case 1.
    hop_cells: dict[int, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    family_primary_outcomes: dict[str, dict[str, Any]] = {}
    for r in rows:
        c = by_cid[r["case_id"]]
        if c.panel in ("main-base", "main-row-perm") and c.solvability == "answerable":
            hop_cells[c.hops][c.variant_index].append(bool(r["correct"]))
            entry = family_primary_outcomes.setdefault(
                c.family_id,
                {"family_id": c.family_id, "hops": c.hops, "solvability": c.solvability, "scores": {}},
            )
            entry["scores"][c.variant_index] = int(r["correct"])

    per_hop_primary: dict[str, Any] = {}
    hop_means: list[float] = []
    for h in (1, 2, 3, 4):
        c0 = hop_cells[h].get(0, [])
        c1 = hop_cells[h].get(1, [])
        a0 = sum(c0) / len(c0) if c0 else None
        a1 = sum(c1) / len(c1) if c1 else None
        hm = 0.5 * (a0 + a1) if (a0 is not None and a1 is not None) else None
        if hm is not None:
            hop_means.append(hm)
        per_hop_primary[f"hop-{h}"] = {
            "base_correct": sum(c0),
            "base_total": len(c0),
            "base_acc": a0,
            "row_perm_correct": sum(c1),
            "row_perm_total": len(c1),
            "row_perm_acc": a1,
            "macro_acc": hm,
        }
    primary_accuracy = math.fsum(hop_means) / len(hop_means) if len(hop_means) == 4 else None

    # 2. Global action safety & solvability rates across main suite (applicable cases)
    main_rows = [r for r in rows if r["panel"].startswith("main-")]
    valid_rate = sum(1 for r in rows if r["valid"]) / len(rows)
    unavail_rows = [r for r in rows if r["solvability"] == "unavailable"]
    unavail_abstain = (
        sum(1 for r in unavail_rows if r["correct"]) / len(unavail_rows)
        if unavail_rows
        else None
    )
    qry_rows = [r for r in rows if r["solvability"] == "query_needed"]
    qry_terminal = sum(1 for r in qry_rows if r["correct"]) / len(qry_rows) if qry_rows else None
    qry_call_rate = (
        sum(1 for r in qry_rows if r["first_action"] == "QUERY") / len(qry_rows) if qry_rows else None
    )

    # 3. Variant-specific breakdown & excluded applicable=false counts
    by_factor: dict[str, dict[str, Any]] = {}
    grouped_panel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped_panel[r["panel"]].append(r)
    for pname, prows in sorted(grouped_panel.items()):
        app = [r for r in prows if r["applicable"]]
        excl = len(prows) - len(app)
        by_factor[pname] = {
            "total": len(prows),
            "applicable_count": len(app),
            "excluded_inapplicable_count": excl,
            "correct": sum(1 for r in app if r["correct"]),
            "accuracy": sum(1 for r in app if r["correct"]) / len(app) if app else None,
            "invalid": sum(1 for r in prows if not r["valid"]),
        }

    # 4. Edge-change panel (256 families: orig_ref + fixed_pos + moved_pos)
    ep_groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in grouped_panel.get("edge-change-panel", []):
        ep_groups[r["world_id"]][r["factor"]] = r
    ep_orig_ok = ep_fix_ok = ep_mov_ok = ep_both_fix = ep_both_mov = ep_all3 = ep_follow = 0
    for fam_rows in ep_groups.values():
        o = fam_rows["orig_ref"]
        fx = fam_rows["fixed_pos"]
        mv = fam_rows["moved_pos"]
        ep_orig_ok += int(o["correct"])
        ep_fix_ok += int(fx["correct"])
        ep_mov_ok += int(mv["correct"])
        ep_both_fix += int(o["correct"] and fx["correct"])
        ep_both_mov += int(o["correct"] and mv["correct"])
        ep_all3 += int(o["correct"] and fx["correct"] and mv["correct"])
        ep_follow += int(o["correct"] and fx["final_answer"] != o["final_answer"] and fx["correct"])
    n_ep = len(ep_groups)
    edge_panel = {
        "families": n_ep,
        "orig_acc": ep_orig_ok / n_ep if n_ep else None,
        "fixed_pos_acc": ep_fix_ok / n_ep if n_ep else None,
        "moved_pos_acc": ep_mov_ok / n_ep if n_ep else None,
        "both_correct_orig_fixed": ep_both_fix / n_ep if n_ep else None,
        "both_correct_orig_moved": ep_both_mov / n_ep if n_ep else None,
        "both_correct_all_three": ep_all3 / n_ep if n_ep else None,
        "change_following_rate": ep_follow / ep_orig_ok if ep_orig_ok else 0.0,
    }

    # 5. Missing-bridge panel
    bp_groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in grouped_panel.get("missing-bridge-panel", []):
        bp_groups[r["world_id"]][r["factor"]] = r
    bp_av_ok = bp_un_ok = bp_both = 0
    for fam_rows in bp_groups.values():
        av = fam_rows["query_available"]
        un = fam_rows["query_unavailable"]
        bp_av_ok += int(av["correct"])
        bp_un_ok += int(un["correct"])
        bp_both += int(av["correct"] and un["correct"])
    n_bp = len(bp_groups)
    bridge_panel = {
        "families": n_bp,
        "query_available_terminal_acc": bp_av_ok / n_bp if n_bp else None,
        "query_unavailable_abstain_acc": bp_un_ok / n_bp if n_bp else None,
        "both_correct_pair_rate": bp_both / n_bp if n_bp else None,
    }

    # 6. Dual-chain fixed-depth query panel
    dq_groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in grouped_panel.get("dual-chain-query-panel", []):
        dq_groups[r["world_id"]][r["factor"]] = r
    dq_a = dq_b = dq_both = 0
    for fam_rows in dq_groups.values():
        ra = fam_rows["query_chain_a"]
        rb = fam_rows["query_chain_b"]
        dq_a += int(ra["correct"])
        dq_b += int(rb["correct"])
        dq_both += int(ra["correct"] and rb["correct"])
    n_dq = len(dq_groups)
    dual_query_panel = {
        "families": n_dq,
        "chain_a_acc": dq_a / n_dq if n_dq else None,
        "chain_b_acc": dq_b / n_dq if n_dq else None,
        "both_correct_rate": dq_both / n_dq if n_dq else None,
    }

    # 7. Serialization holdout
    sh_rows = grouped_panel.get("serialization-holdout", [])
    sh_acc = sum(1 for r in sh_rows if r["correct"]) / len(sh_rows) if sh_rows else None

    # 8. Preservation suite (7 original V6 kinds)
    pres_by_kind: dict[str, dict[str, Any]] = {}
    for kind in ORIGINAL_PRESERVATION_KINDS:
        krows = [r for r in grouped_panel.get("preservation-suite", []) if r["solvability"] == kind]
        kcor = sum(1 for r in krows if r["correct"])
        pres_by_kind[kind] = {
            "episodes": len(krows),
            "correct": kcor,
            "accuracy": kcor / len(krows) if krows else None,
        }

    # 9. Failure counts + representative failed case_ids + divergence breakdown
    div_counts: dict[str, int] = defaultdict(int)
    failed_ids: list[str] = []
    for r in sorted(rows, key=lambda item: item["case_id"]):
        div_counts[r["first_divergence"]] += 1
        if not r["correct"]:
            failed_ids.append(r["case_id"])

    # 10. Separate historical continuity
    hist_summary = None
    if historical_continuity_cases and historical_pred_doc is not None:
        hrows = historical_pred_doc["predictions"]
        hist_summary = {
            "episodes": len(hrows),
            "correct": sum(1 for r in hrows if r["correct"]),
            "accuracy": sum(1 for r in hrows if r["correct"]) / len(hrows) if hrows else None,
        }

    # Family cluster scores for primary endpoint bootstrap (each answerable family macro score (s0+s1)/2)
    family_cluster_scores = {
        fid: 0.5 * (info["scores"][0] + info["scores"][1])
        for fid, info in family_primary_outcomes.items()
        if 0 in info["scores"] and 1 in info["scores"]
    }
    family_strata = {
        fid: (info["hops"], info["solvability"])
        for fid, info in family_primary_outcomes.items()
        if fid in family_cluster_scores
    }

    return {
        "schema": "anra-v8-y1-metrics-v1",
        "job_id": pred_doc["job_id"],
        "decoder_condition": pred_doc["decoder_condition"],
        "checkpoint_sha256": pred_doc["checkpoint_sha256"],
        "episodes_total": len(rows),
        "primary_endpoint": primary_accuracy,
        "per_hop_primary": per_hop_primary,
        "guardrails": {
            "valid_action_rate": valid_rate,
            "unavailable_abstain_rate": unavail_abstain,
            "query_needed_raw_terminal_rate": qry_terminal,
            "query_needed_call_rate": qry_call_rate,
            "edge_both_correct_all_three": edge_panel["both_correct_all_three"],
            "edge_both_correct_orig_fixed": edge_panel["both_correct_orig_fixed"],
            "edge_both_correct_orig_moved": edge_panel["both_correct_orig_moved"],
            "dual_query_both_correct": dual_query_panel["both_correct_rate"],
        },
        "by_panel": by_factor,
        "edge_change_panel": edge_panel,
        "missing_bridge_panel": bridge_panel,
        "dual_chain_query_panel": dual_query_panel,
        "serialization_holdout": {"episodes": len(sh_rows), "accuracy": sh_acc},
        "preservation_suite": pres_by_kind,
        "first_divergence_counts": dict(sorted(div_counts.items())),
        "failed_count": len(failed_ids),
        "representative_failed_case_ids": failed_ids[:32],
        "historical_continuity": hist_summary,
        "family_cluster_scores": family_cluster_scores,
        "family_strata": {k: list(v) for k, v in family_strata.items()},
    }


def evaluate_all_heuristics(
    cases: tuple[Y1Case, ...],
    *,
    historical_cases: tuple[Y1Case, ...] = (),
    seed: int = 7,
) -> dict[str, Any]:
    """Evaluate all 7 preregistered heuristic/oracle controls on the exact identical prompts."""
    out: dict[str, Any] = {}
    for mode in (
        "last-row-object",
        "nearest-to-query-object",
        "highest-timestamp-object",
        "global-most-frequent-answer",
        "always-abstain",
        "random-legal",
        "oracle-traversal",
    ):
        pol = Y1HeuristicPolicy(mode, majority_symbol="A", seed=seed)
        pred = evaluate_y1_cases(cases, pol, job_id=f"Y1-heuristic-{mode}", decoder_condition=f"heuristic:{mode}")
        hist_pred = (
            evaluate_y1_cases(historical_cases, pol, job_id=f"Y1-heuristic-{mode}-hist", decoder_condition=f"heuristic:{mode}")
            if historical_cases
            else None
        )
        summ = summarize_y1_predictions(
            cases,
            pred,
            historical_continuity_cases=historical_cases,
            historical_pred_doc=hist_pred,
        )
        # Keep heuristic receipts compact while preserving exact primary/panel numbers.
        summ.pop("family_cluster_scores", None)
        summ.pop("family_strata", None)
        out[mode] = {
            "rule": {
                "last-row-object": "object of last serialized initial row or ! if null",
                "nearest-to-query-object": "object of row 0 adjacent to query header or ! if null",
                "highest-timestamp-object": "object of highest-tick initial row (tie last row) or ! if null",
                "global-most-frequent-answer": "constant majority answer symbol A",
                "always-abstain": "always ! (ABSTAIN)",
                "random-legal": "uniform SHA256-seeded choice over legal actions",
                "oracle-traversal": "privileged evaluator oracle using Truth (not a shortcut)",
            }[mode],
            "primary_endpoint": summ["primary_endpoint"],
            "per_hop_primary": summ["per_hop_primary"],
            "guardrails": summ["guardrails"],
            "edge_change_panel": summ["edge_change_panel"],
            "missing_bridge_panel": summ["missing_bridge_panel"],
            "dual_chain_query_panel": summ["dual_chain_query_panel"],
            "preservation_suite": summ["preservation_suite"],
            "historical_continuity": summ["historical_continuity"],
        }
    return out


def bootstrap_paired_family_ci(
    cand_by_seed: dict[int, dict[str, Any]],
    base_by_seed: dict[int, dict[str, Any]],
    *,
    n_resamples: int = 2000,
    seed: int = 20261116,
) -> dict[str, Any]:
    """Stratified (hop, solvability) 2,000-resample family-cluster percentile 95% CI per seed and mean."""
    seeds = sorted(cand_by_seed)
    first_s = seeds[0]
    fids_by_stratum: dict[tuple[int, str], list[str]] = defaultdict(list)
    strata_map = cand_by_seed[first_s]["family_strata"]
    for fid in sorted(cand_by_seed[first_s]["family_cluster_scores"]):
        h, solv = strata_map[fid]
        fids_by_stratum[(int(h), str(solv))].append(fid)

    rng = random.Random(derive_seed_int("y1-bootstrap", seed, n_resamples, sorted(seeds)))
    per_seed_diffs: dict[int, list[float]] = {s: [] for s in seeds}
    mean_diffs: list[float] = []

    strata_keys = sorted(fids_by_stratum)
    total_families = sum(len(fids_by_stratum[k]) for k in strata_keys)

    for _ in range(n_resamples):
        sampled_fids: list[str] = []
        for sk in strata_keys:
            pool = fids_by_stratum[sk]
            sampled_fids.extend(rng.choice(pool) for _ in range(len(pool)))
        seed_d: list[float] = []
        for s in seeds:
            cs = cand_by_seed[s]["family_cluster_scores"]
            bs = base_by_seed[s]["family_cluster_scores"]
            diff = sum(cs[f] - bs[f] for f in sampled_fids) / total_families
            per_seed_diffs[s].append(diff)
            seed_d.append(diff)
        mean_diffs.append(sum(seed_d) / len(seed_d))

    def ci_bounds(vals: list[float]) -> tuple[float, float]:
        sv = sorted(vals)
        lo = sv[max(0, min(len(sv) - 1, int(0.025 * len(sv))))]
        hi = sv[max(0, min(len(sv) - 1, int(0.975 * len(sv)) - 1))]
        return lo, hi

    mean_lo, mean_hi = ci_bounds(mean_diffs)
    return {
        "resamples": n_resamples,
        "sampling_unit": "stratified_family_cluster(hop, solvability)",
        "mean_contrast": sum(
            cand_by_seed[s]["primary_endpoint"] - base_by_seed[s]["primary_endpoint"] for s in seeds
        ) / len(seeds),
        "mean_ci95_low": mean_lo,
        "mean_ci95_high": mean_hi,
        "per_seed": {
            str(s): {
                "observed_diff": cand_by_seed[s]["primary_endpoint"] - base_by_seed[s]["primary_endpoint"],
                "ci95_low": ci_bounds(per_seed_diffs[s])[0],
                "ci95_high": ci_bounds(per_seed_diffs[s])[1],
            }
            for s in seeds
        },
    }


def evaluate_t1_or_t2_gate(
    *,
    stage: str,
    arms_metrics: dict[str, dict[int, dict[str, Any]]],
    heuristics: dict[str, Any],
    require_ci_lower_gt_zero: bool = False,
) -> dict[str, Any]:
    """Fail-closed evaluation of T1 factorial screening or T2 locked confirmation."""
    gate_cfg = CONFIRMATION_GATE if require_ci_lower_gt_zero else SCREENING_GATE
    if "F00" not in arms_metrics or set(arms_metrics["F00"]) != set(SEEDS):
        return {"schema": "anra-v8-y1-gate-v1", "stage": stage, "status": "INCOMPLETE", "reason": "missing F00 seeds"}

    best_shortcut_name = max(SHORTCUT_HEURISTICS, key=lambda h: heuristics[h]["primary_endpoint"] or 0.0)
    best_shortcut_primary = heuristics[best_shortcut_name]["primary_endpoint"] or 0.0
    f00_by_seed = arms_metrics["F00"]

    candidates = [a for a in sorted(arms_metrics) if a != "F00"]
    arm_reports: dict[str, Any] = {}
    passing_candidates: list[tuple[float, str]] = []

    for arm in candidates:
        cand_by_seed = arms_metrics[arm]
        if set(cand_by_seed) != set(SEEDS):
            arm_reports[arm] = {"status": "INCOMPLETE", "missing_seeds": sorted(set(SEEDS) - set(cand_by_seed))}
            continue

        ci = bootstrap_paired_family_ci(
            cand_by_seed,
            f00_by_seed,
            n_resamples=int(gate_cfg["bootstrap_resamples"]),
        )
        mean_primary = sum(cand_by_seed[s]["primary_endpoint"] for s in SEEDS) / len(SEEDS)
        f00_mean_primary = sum(f00_by_seed[s]["primary_endpoint"] for s in SEEDS) / len(SEEDS)
        mean_gain = mean_primary - f00_mean_primary
        adv_vs_shortcut = mean_primary - best_shortcut_primary

        seed_checks: dict[str, Any] = {}
        reasons: list[str] = []
        if mean_gain < gate_cfg["min_mean_primary_gain"]:
            reasons.append(f"mean_gain {mean_gain:.4f} < {gate_cfg['min_mean_primary_gain']}")
        if adv_vs_shortcut < gate_cfg["min_advantage_over_best_shortcut"]:
            reasons.append(f"advantage_vs_{best_shortcut_name} {adv_vs_shortcut:.4f} < {gate_cfg['min_advantage_over_best_shortcut']}")
        if require_ci_lower_gt_zero and ci["mean_ci95_low"] <= 0.0:
            reasons.append(f"mean_ci95_low {ci['mean_ci95_low']:.4f} <= 0")

        for s in SEEDS:
            cm = cand_by_seed[s]
            bm = f00_by_seed[s]
            diff = cm["primary_endpoint"] - bm["primary_endpoint"]
            guards = cm["guardrails"]
            kind_regs = {
                k: (bm["preservation_suite"][k]["accuracy"] - cm["preservation_suite"][k]["accuracy"])
                for k in ORIGINAL_PRESERVATION_KINDS
            }
            worst_kind = max(kind_regs, key=lambda k: kind_regs[k])
            worst_reg = kind_regs[worst_kind]
            edge_both = guards["edge_both_correct_all_three"] or 0.0
            dq_both = guards["dual_query_both_correct"] or 0.0
            qry_term = guards["query_needed_raw_terminal_rate"] or 0.0
            val_rate = guards["valid_action_rate"] or 0.0
            un_rate = guards["unavailable_abstain_rate"] or 0.0

            ok_s = (
                diff > 0.0
                and edge_both >= gate_cfg["min_edge_panel_both_correct_per_seed"]
                and dq_both >= gate_cfg["min_dual_query_both_correct_per_seed"]
                and qry_term >= gate_cfg["min_query_needed_raw_terminal_per_seed"]
                and val_rate >= gate_cfg["min_valid_action_rate_per_seed"]
                and un_rate >= gate_cfg["min_unavailable_abstain_rate_per_seed"]
                and worst_reg <= gate_cfg["max_original_kind_regression_vs_f00_per_seed"]
            )
            seed_checks[str(s)] = {
                "primary": cm["primary_endpoint"],
                "f00_primary": bm["primary_endpoint"],
                "gain_vs_f00": diff,
                "edge_both_correct": edge_both,
                "dual_query_both_correct": dq_both,
                "query_needed_raw_terminal": qry_term,
                "valid_action_rate": val_rate,
                "unavailable_abstain_rate": un_rate,
                "worst_original_kind": worst_kind,
                "worst_original_kind_regression": worst_reg,
                "passed_all_seed_guards": ok_s,
            }
            if diff <= 0.0:
                reasons.append(f"seed {s} gain {diff:.4f} <= 0")
            if edge_both < gate_cfg["min_edge_panel_both_correct_per_seed"]:
                reasons.append(f"seed {s} edge_both_correct {edge_both:.4f} < 0.50")
            if dq_both < gate_cfg["min_dual_query_both_correct_per_seed"]:
                reasons.append(f"seed {s} dual_query_both_correct {dq_both:.4f} < 0.50")
            if qry_term < gate_cfg["min_query_needed_raw_terminal_per_seed"]:
                reasons.append(f"seed {s} query_needed_raw_terminal {qry_term:.4f} < 0.50")
            if val_rate < gate_cfg["min_valid_action_rate_per_seed"]:
                reasons.append(f"seed {s} valid_action_rate {val_rate:.4f} < 0.95")
            if un_rate < gate_cfg["min_unavailable_abstain_rate_per_seed"]:
                reasons.append(f"seed {s} unavailable_abstain_rate {un_rate:.4f} < 0.90")
            if worst_reg > gate_cfg["max_original_kind_regression_vs_f00_per_seed"]:
                reasons.append(f"seed {s} preservation regression {worst_kind}={worst_reg:.4f} > 0.05")

        passed = not reasons
        if passed:
            passing_candidates.append((mean_primary, arm))
        arm_reports[arm] = {
            "status": "PASS" if passed else "FAIL",
            "reasons": reasons,
            "mean_primary": mean_primary,
            "f00_mean_primary": f00_mean_primary,
            "mean_gain_vs_f00": mean_gain,
            "best_shortcut_heuristic": best_shortcut_name,
            "best_shortcut_primary": best_shortcut_primary,
            "advantage_vs_best_shortcut": adv_vs_shortcut,
            "bootstrap_ci": ci,
            "seeds": seed_checks,
        }

    # Resolve winner with exact tie-break priority F01, F10, F11 within 1e-12
    locked_winner = None
    if passing_candidates:
        max_score = max(p[0] for p in passing_candidates)
        near_ties = {arm for score, arm in passing_candidates if abs(max_score - score) <= 1e-12}
        for preferred in TIE_BREAK_ORDER:
            if preferred in near_ties:
                locked_winner = preferred
                break
        if locked_winner is None:
            locked_winner = sorted(near_ties)[0]

    # Exploratory 2x2 factorial contrasts when all 4 arms present
    factorial_contrasts = None
    if all(a in arms_metrics and set(arms_metrics[a]) == set(SEEDS) for a in ("F00", "F10", "F01", "F11")):
        means = {
            a: sum(arms_metrics[a][s]["primary_endpoint"] for s in SEEDS) / len(SEEDS)
            for a in ("F00", "F10", "F01", "F11")
        }
        factorial_contrasts = {
            "label": "EXPLORATORY_THREE_SEED_DESCRIPTIVE_CONTRASTS",
            "F10_minus_F00_fresh": means["F10"] - means["F00"],
            "F01_minus_F00_coverage": means["F01"] - means["F00"],
            "F11_minus_F10_minus_F01_plus_F00_interaction": means["F11"] - means["F10"] - means["F01"] + means["F00"],
        }

    any_incomplete = any(r.get("status") == "INCOMPLETE" for r in arm_reports.values())
    status = "INCOMPLETE" if any_incomplete else ("PASS" if locked_winner is not None else "FAIL")
    return {
        "schema": "anra-v8-y1-gate-v1",
        "protocol_sha256": PROTOCOL_SHA,
        "stage": stage,
        "status": status,
        "locked_winner": locked_winner,
        "best_shortcut_heuristic": best_shortcut_name,
        "best_shortcut_primary": best_shortcut_primary,
        "arms": arm_reports,
        "exploratory_factorial_contrasts": factorial_contrasts,
    }


def evaluate_f11_bottleneck_admission(
    f11_d1_by_seed: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Check whether completed T1 F11 models satisfy all output-bottleneck criteria for Group-CE T3."""
    if set(f11_d1_by_seed) != set(SEEDS):
        return {"admitted": False, "reason": "incomplete F11 D1 evaluations", "seeds": {}}
    seed_details: dict[str, Any] = {}
    all_pass = True
    for s in SEEDS:
        d = f11_d1_by_seed[s]
        rank1 = float(d.get("known_base_rank1_within_ap_rate") or 0.0)
        raw_known = float(d.get("known_base_raw_exact_rate") or 0.0)
        gap = rank1 - raw_known
        ce_edge = float(d.get("constrained_edge_both_correct") or 0.0)
        ce_dq = float(d.get("constrained_dual_query_both_correct") or 0.0)
        ok = (
            rank1 >= BOTTLENECK_CRITERIA["min_answer_only_rank1_base_known_per_seed"]
            and gap >= BOTTLENECK_CRITERIA["min_rank1_minus_raw_gap_per_seed"]
            and ce_edge >= BOTTLENECK_CRITERIA["min_constrained_edge_both_correct_per_seed"]
            and ce_dq >= BOTTLENECK_CRITERIA["min_constrained_dual_query_both_correct_per_seed"]
        )
        all_pass = all_pass and ok
        seed_details[str(s)] = {
            "rank1_within_ap": rank1,
            "raw_exact_known_base": raw_known,
            "rank1_minus_raw_gap": gap,
            "constrained_edge_both_correct": ce_edge,
            "constrained_dual_query_both_correct": ce_dq,
            "passed": ok,
        }
    return {
        "schema": "anra-v8-y1-bottleneck-admission-v1",
        "admitted": all_pass,
        "recommended_branch": "targeted_group_ce" if all_pass else "capacity_s20",
        "seeds": seed_details,
    }


def evaluate_t0_learnability(pred_doc: dict[str, Any], cases: tuple[Y1Case, ...], final_loss: float) -> dict[str, Any]:
    """Check exact raw interactive sequence+value+EOS accuracy >= 95% on T0's 128 balanced families."""
    by_cid = {c.case_id: c for c in cases}
    rows = pred_doc["predictions"]
    if len(rows) != len(cases) or {r["case_id"] for r in rows} != set(by_cid):
        raise ValueError("T0 prediction mismatch against 128 families")
    correct = sum(1 for r in rows if r["correct"])
    acc = correct / len(rows)
    passed = math.isfinite(final_loss) and acc >= T0_SPEC["min_raw_accuracy"]
    return {
        "schema": "anra-v8-y1-t0-gate-v1",
        "families": len(rows),
        "correct": correct,
        "raw_exact_accuracy": acc,
        "final_loss": final_loss,
        "threshold": T0_SPEC["min_raw_accuracy"],
        "status": "PASS" if passed else "FAIL",
    }
