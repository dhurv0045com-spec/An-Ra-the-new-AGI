"""ELENDRIL V8-X1 primary endpoint aggregation, 10,000-replicate cluster bootstrap, and reports."""

from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import Any, Mapping, Sequence

from anra_v5.planthir.design import analyze_paired_seed_runs

from .contracts import X1_FAMILIES
from .protocol import X1_ADVANCEMENT_THRESHOLDS

_R_MECHANISTIC_FAMILIES = frozenset({"X02", "X03", "X05", "X06", "X07", "X08", "X09"})


def summarize_x1_episodes(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute primary X1 macro-12-family endpoints, view conjunctions, CF-both-correct, and divergence counts."""

    if not records:
        raise ValueError("cannot summarize empty X1 episode records")

    by_family: dict[str, list[bool]] = defaultdict(list)
    by_form: dict[str, list[bool]] = defaultdict(list)
    by_hop: dict[int, list[bool]] = defaultdict(list)
    by_base: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    div_counts: Counter[str] = Counter()
    lucky_count = 0

    for rec in records:
        ok = bool(rec["correct"])
        by_family[str(rec["family"])].append(ok)
        by_form[str(rec["form"])].append(ok)
        by_hop[int(rec["hops"])].append(ok)
        by_base[str(rec["base_world_id"])][str(rec["form"])] = rec
        div_counts[str((rec.get("first_divergence") or {}).get("category", "none"))] += 1
        if rec.get("lucky_unsupported_guess"):
            lucky_count += 1

    fam_rates = {fam: (sum(vals) / len(vals)) for fam, vals in sorted(by_family.items()) if vals}
    present_fams = [fam for fam in X1_FAMILIES if fam in fam_rates]
    macro_12 = (sum(fam_rates[f] for f in present_fams) / len(present_fams)) if present_fams else (
        sum(1 for r in records if r["correct"]) / len(records)
    )

    # Linked counterfactuals & view conjunctions over shared base_world_id
    rel_cf_n = 0
    rel_cf_both = 0
    r_mech_n = 0
    r_mech_both = 0
    irrel_n = 0
    irrel_pres = 0
    perm_n = 0
    perm_pres = 0
    conj_n = 0
    conj_ok = 0

    for _base_id, forms in by_base.items():
        c = forms.get("canonical")
        if c is None:
            continue
        p = forms.get("entity_code_perm")
        ir = forms.get("irrel_facts")
        cf = forms.get("relevant_cf")
        if p is not None and ir is not None:
            conj_n += 1
            if bool(c["correct"]) and bool(p["correct"]) and bool(ir["correct"]):
                conj_ok += 1
        if p is not None and bool(c["correct"]):
            perm_n += 1
            if bool(p["correct"]):
                perm_pres += 1
        if ir is not None and bool(c["correct"]):
            irrel_n += 1
            if bool(ir["correct"]):
                irrel_pres += 1
        if cf is not None and cf.get("cf_transition_class") == "determinate_answer_change":
            rel_cf_n += 1
            both = bool(c["correct"]) and bool(cf["correct"])
            if both:
                rel_cf_both += 1
            if str(c["family"]) in _R_MECHANISTIC_FAMILIES:
                r_mech_n += 1
                if both:
                    r_mech_both += 1

    return {
        "schema": "elendril-x1-summary-v1",
        "episodes": len(records),
        "base_worlds": len(by_base),
        "overall_accuracy": float(sum(1 for r in records if r["correct"]) / len(records)),
        "macro_12_family_accuracy": float(macro_12),
        "by_family": fam_rates,
        "by_form": {k: (sum(v) / len(v)) for k, v in sorted(by_form.items()) if v},
        "by_hop": {str(h): (sum(v) / len(v)) for h, v in sorted(by_hop.items()) if v},
        "relevant_cf_both_correct": (rel_cf_both / rel_cf_n) if rel_cf_n else None,
        "relevant_cf_counts": {"both_correct": rel_cf_both, "eligible": rel_cf_n},
        "r_mechanistic_cf_both_correct": (r_mech_both / r_mech_n) if r_mech_n else None,
        "r_mechanistic_cf_counts": {"both_correct": r_mech_both, "eligible": r_mech_n},
        "d2_view_conjunction_accuracy": (conj_ok / conj_n) if conj_n else None,
        "d2_view_conjunction_counts": {"all_correct": conj_ok, "eligible": conj_n},
        "perm_preservation_among_correct": (perm_pres / perm_n) if perm_n else None,
        "irrel_preservation_among_correct": (irrel_pres / irrel_n) if irrel_n else None,
        "lucky_unsupported_guesses": int(lucky_count),
        "first_divergence_histogram": dict(sorted(div_counts.items())),
    }


def cluster_bootstrap_paired_world_diff(
    ctrl_records: Sequence[Mapping[str, Any]],
    trt_records: Sequence[Mapping[str, Any]],
    *,
    replicates: int = int(X1_ADVANCEMENT_THRESHOLDS["bootstrap_replicates"]),
    seed: int = 17,
) -> dict[str, Any]:
    """Paired cluster bootstrap keeping all linked views of each base_world_id together (Section 9.3)."""

    c_by_base: dict[str, list[bool]] = defaultdict(list)
    t_by_base: dict[str, list[bool]] = defaultdict(list)
    for r in ctrl_records:
        c_by_base[str(r["base_world_id"])].append(bool(r["correct"]))
    for r in trt_records:
        t_by_base[str(r["base_world_id"])].append(bool(r["correct"]))

    shared_bases = sorted(set(c_by_base) & set(t_by_base))
    if not shared_bases:
        raise ValueError("paired cluster bootstrap requires overlapping base_world_ids")

    per_base_deltas = [
        (sum(t_by_base[b]) / len(t_by_base[b])) - (sum(c_by_base[b]) / len(c_by_base[b]))
        for b in shared_bases
    ]
    n = len(per_base_deltas)
    obs_mean = sum(per_base_deltas) / n
    rng = random.Random(seed)
    boot_means = sorted(
        sum(per_base_deltas[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(max(1, int(replicates)))
    )
    lo = boot_means[int(0.025 * (len(boot_means) - 1))]
    hi = boot_means[int(0.975 * (len(boot_means) - 1))]
    return {
        "base_world_clusters": n,
        "mean_paired_delta": float(obs_mean),
        "replicates": int(replicates),
        "ci95": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def evaluate_x1_development_selection(
    seed_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate Section 9.3 development candidate selection rules without fabricating promotion."""

    if not seed_pairs:
        return {"qualified": False, "reason": "NO_SEED_PAIRS", "deltas": []}
    deltas = [float(p["trt_macro_12"] - p["ctrl_macro_12"]) for p in seed_pairs]
    mean_gain = sum(deltas) / len(deltas)
    pos_count = sum(1 for d in deltas if d > 0.0)
    min_delta = min(deltas)

    # Also invoke Planthir's exact paired-seed inferential statistical summary
    obs = [
        {
            "pair_id": f"{p['init_seed']}:{p['order_seed']}",
            "init_seed": p["init_seed"],
            "order_seed": p["order_seed"],
            "control_value": p["ctrl_macro_12"],
            "treatment_value": p["trt_macro_12"],
        }
        for p in seed_pairs
    ]
    paired_stats = analyze_paired_seed_runs(obs) if len(seed_pairs) >= 2 else None

    mean_ok = mean_gain >= float(X1_ADVANCEMENT_THRESHOLDS["development_mean_gain_min"])
    pos_ok = pos_count >= int(X1_ADVANCEMENT_THRESHOLDS["development_positive_seeds_min"])
    drop_ok = min_delta >= float(X1_ADVANCEMENT_THRESHOLDS["development_max_single_seed_drop"])
    return {
        "qualified": bool(mean_ok and pos_ok and drop_ok and paired_stats and paired_stats["passed"]),
        "mean_gain": float(mean_gain),
        "positive_seeds": int(pos_count),
        "total_seeds": len(deltas),
        "min_seed_delta": float(min_delta),
        "seed_deltas": deltas,
        "paired_seed_stats": paired_stats,
    }


def build_x1_report_markdown(
    *,
    session: str,
    source_commit: str,
    execution_status: str,
    delivery_status: str,
    scientific_status: str,
    cause_status: str,
    job_summaries: Sequence[Mapping[str, Any]],
    notes: Sequence[str] = (),
) -> str:
    """Render concise V8-X1 session report separating execution/delivery vs scientific/cause status."""

    lines = [
        f"# ELENDRIL V8-X1 Session Report — {session}",
        "",
        f"- **Source Commit:** `{source_commit}`",
        f"- **Execution Status:** `{execution_status}`",
        f"- **Delivery Status:** `{delivery_status}`",
        f"- **Scientific Status:** `{scientific_status}`",
        f"- **Cause Status:** `{cause_status}`",
        "",
        "## Job Summary Table",
        "",
        "| Job ID | Arch | Objective | Seed | Updates | Overall Acc | Macro-12 Acc | R-CF Both | D2 Conj |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for j in job_summaries:
        s = j.get("summary", {})
        rcf = s.get("relevant_cf_both_correct")
        d2c = s.get("d2_view_conjunction_accuracy")
        ov_str = f"{s['overall_accuracy']:.4f}" if s.get("overall_accuracy") is not None else "N/A"
        m12_str = f"{s['macro_12_family_accuracy']:.4f}" if s.get("macro_12_family_accuracy") is not None else "N/A"
        rcf_str = f"{rcf:.4f}" if rcf is not None else "N/A"
        d2c_str = f"{d2c:.4f}" if d2c is not None else "N/A"
        lines.append(
            f"| `{j['job_id']}` | `{j['architecture']}` | `{j['objective']}` | `{j['init_seed']}` | "
            f"{j.get('completed_updates', 0)} | {ov_str} | {m12_str} | {rcf_str} | {d2c_str} |"
        )
    if notes:
        lines.extend(["", "## Notes", ""])
        for n in notes:
            lines.append(f"- {n}")
    return "\n".join(lines) + "\n"


__all__ = [
    "build_x1_report_markdown",
    "cluster_bootstrap_paired_world_diff",
    "evaluate_x1_development_selection",
    "summarize_x1_episodes",
]
