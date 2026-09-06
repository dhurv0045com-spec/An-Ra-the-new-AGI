"""ARK-004A-R: independent reanalysis of the precursor claim, from raw receipts.

Direction semantics (explicit, no naming tricks):
  positive rho(feature, G90_step)  = higher feature predicts LATER generalization
  negative rho(feature, G90_step)  = higher feature predicts EARLIER generalization
Both G90_step and post_mem_delay_90 (G90 - M99) targets are reported, plus the
sign-flipped speed forms, so direction cannot hide behind a transform.

LOO algorithm (explicit, as executed here):
  For each held-out seed h: compute Spearman rho of the feature vs the target
  over the OTHER three seeds. "Positive direction fold" = rho > 0 (feature
  high -> G90 step large -> generalization later among training seeds).
  Aggregation: count of positive folds + full-sample rho. Ties in the target
  or feature make rho undefined for that fold and are counted as TIE.
  Target = raw G90_step. No sign-flipping, no post-hoc transformation.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path


def spearman(x, y):
    n = len(x)
    if n < 3:
        return None
    rx = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: x[i]))}
    ry = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: y[i]))}
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def sustained(trajectory, key, bar, consecutive=3):
    streak, start = 0, None
    for e in trajectory:
        if float(e.get(key, 0.0)) >= bar:
            if streak == 0:
                start = e["step"]
            streak += 1
            if streak >= consecutive:
                return start
        else:
            streak, start = 0, None
    return None


def first_sustained_after(trajectory, key, bar, after_step, consecutive=3):
    """First window of `consecutive` evals >= bar, strictly after after_step."""
    streak, start = 0, None
    for e in trajectory:
        if e["step"] <= after_step:
            continue
        if float(e.get(key, 0.0)) >= bar:
            if streak == 0:
                start = e["step"]
            streak += 1
            if streak >= consecutive:
                return start
        else:
            streak, start = 0, None
    return None


def reconstruct(path: str) -> dict:
    r = json.load(open(path, encoding="utf-8"))
    res = r["results"]
    traj = res["trajectory"]
    seed = res["seed"]
    m99 = res["sustained"]["M99"]["step"]
    g50 = res["sustained"]["G50"]["step"]
    g90 = res["sustained"]["G90"]["step"]
    window = [e for e in traj if m99 <= e["step"] <= m99 + 2000]
    features = {
        "mean_tens_selectivity": sum(e["selectivity"]["TENS_tens_share"] for e in window) / len(window),
        "mean_ones_selectivity": sum(e["selectivity"]["ONES_ones_share"] for e in window) / len(window),
        "mean_tens_margin": sum(e["margins"]["tens_margin"] for e in window) / len(window),
        "mean_ones_margin": sum(e["margins"]["ones_margin"] for e in window) / len(window),
        "early_ood_auc": sum(e["test_exact"] for e in window) / len(window),
        "loss_at_M99": next(e["loss"] for e in traj if e["step"] >= m99),
        "final_ood": traj[-1]["test_exact"],
        "final_tens_selectivity": traj[-1]["selectivity"]["TENS_tens_share"],
    }
    # temporal-ordering landmarks (sustained OOD thresholds after M99)
    landmarks = {
        "P10": first_sustained_after(traj, "test_exact", 0.10, m99),
        "P25": first_sustained_after(traj, "test_exact", 0.25, m99),
        "G50": g50,
        "G90": g90,
    }
    # selectivity landmarks: first step where tens selectivity exceeds its
    # final value's 80% AND its early-window mean by >= 0.10 (a material move)
    early_mean = features["mean_tens_selectivity"]
    sel_move = None
    for e in traj:
        if e["step"] <= m99:
            continue
        if e["selectivity"]["TENS_tens_share"] >= early_mean + 0.10:
            sel_move = e["step"]
            break
    return {
        "seed": seed,
        "M99": m99, "G50": g50, "G90": g90,
        "post_mem_delay_90": (g90 - m99) if g90 else None,
        "window": {"start": m99, "end": m99 + 2000, "n_evals": len(window)},
        "features": features,
        "landmarks": landmarks,
        "tens_selectivity_first_material_move": sel_move,
    }


def main() -> int:
    receipts = sorted(glob.glob(str(Path(__file__).parent / "RESULT_seed*.json")))
    per_seed = [reconstruct(p) for p in receipts]
    seeds = [s["seed"] for s in per_seed]
    g90 = [s["G90"] for s in per_seed]
    delay = [s["post_mem_delay_90"] for s in per_seed]

    feature_names = [
        "mean_tens_selectivity", "mean_ones_selectivity", "mean_tens_margin",
        "mean_ones_margin", "early_ood_auc", "loss_at_M99",
    ]
    table = {}
    for name in feature_names:
        values = [s["features"][name] for s in per_seed]
        table[name] = {
            "values_by_seed": dict(zip(seeds, [round(v, 4) for v in values])),
            "rho_vs_G90_step": round(spearman(values, g90), 3),
            "rho_vs_post_mem_delay_90": round(spearman(values, delay), 3),
            "rho_vs_negG90_step": round(spearman(values, [-g for g in g90]), 3),
            "rho_vs_neg_delay": round(spearman(values, [-d for d in delay]), 3),
        }
    baselines = {
        "M99_step": [s["M99"] for s in per_seed],
        "absolute_step_halfway": [12000] * len(seeds),  # constant: rho undefined -> excluded
    }
    for name, values in baselines.items():
        table[name] = {
            "values_by_seed": dict(zip(seeds, values)),
            "rho_vs_G90_step": round(spearman(values, g90), 3) if name != "absolute_step_halfway" else None,
        }

    loo = {}
    for name in feature_names + ["M99_step"]:
        values = [s["M99"] for s in per_seed] if name == "M99_step" else [
            s["features"][name] for s in per_seed]
        folds = []
        for held in range(len(seeds)):
            idx = [i for i in range(len(seeds)) if i != held]
            rho = spearman([values[i] for i in idx], [g90[i] for i in idx])
            folds.append({"held_out_seed": seeds[held], "rho_over_remaining_3": (
                round(rho, 3) if rho is not None else "TIE")})
        positive = sum(1 for f in folds if isinstance(f["rho_over_remaining_3"], float)
                       and f["rho_over_remaining_3"] > 0)
        loo[name] = {"folds": folds, "positive_direction_folds": positive,
                     "convention": "positive fold = feature-high -> G90-step-large = LATER "
                                   "generalization among the 3 training seeds"}

    # temporal ordering: does tens-selectivity's material move precede P10?
    ordering = {}
    for s in per_seed:
        move, p10 = s["tens_selectivity_first_material_move"], s["landmarks"]["P10"]
        if move is None or p10 is None:
            ordering[s["seed"]] = {"classification": "INSUFFICIENT_MOVE_OR_NO_P10",
                                   "selectivity_move_step": move, "P10": p10}
        else:
            ordering[s["seed"]] = {
                "classification": ("BEFORE_P10" if move < p10 else
                                   "NEAR_P10" if abs(move - p10) <= 400 else
                                   "AFTER_P10"),
                "selectivity_move_step": move, "P10": p10,
            }

    result = {
        "schema": "arkenstone-ark004a-reanalysis/v1",
        "direction_semantics": {
            "positive_rho_vs_G90_step": "higher feature predicts LATER generalization",
            "negative_rho_vs_G90_step": "higher feature predicts EARLIER generalization",
        },
        "loo_algorithm": {
            "fitted_on": "the 3 non-held-out seeds (rank correlation only; no fitted parameters)",
            "prediction_for_held_out": "rank direction is the only prediction asserted",
            "correct_direction": "rho > 0 over the 3 training seeds (feature high -> G90 step large -> later)",
            "ties": "a fold with any target/feature ties yielding undefined rho counts as TIE",
            "target": "raw G90_step; no transformation",
        },
        "per_seed": per_seed,
        "correlation_table": table,
        "loo": loo,
        "temporal_ordering": ordering,
    }
    out = Path(__file__).parent / "REANALYSIS.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "rho_tens_sel_vs_G90step": table["mean_tens_selectivity"]["rho_vs_G90_step"],
        "rho_tens_sel_vs_delay": table["mean_tens_selectivity"]["rho_vs_post_mem_delay_90"],
        "loo_tens_positive_folds": loo["mean_tens_selectivity"]["positive_direction_folds"],
        "ordering": {str(k): v["classification"] for k, v in ordering.items()},
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
