"""Frontier-stability checks (Mission 7).

A ladder need not be perfectly monotonic, but wild non-monotonicity from a
tiny noisy pilot must yield CALIBRATION_UNSTABLE, not an invented "pocket".
Distinguishes REAL_NONMONOTONICITY (CIs exclude noise) from
CALIBRATION_NOISE (CIs overlap wildly / N tiny).
"""

from __future__ import annotations

from .status import wilson


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for pos, i in enumerate(order):
        r[i] = pos + 1.0
    return r


def spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx <= 0 or vy <= 0:
        return 0.0
    return cov / (vx * vy) ** 0.5


def check_frontier(rung_stats: list[dict]) -> dict:
    """rung_stats: ordered [{rung, raw_k, n}] easiest-first.

    Returns stability verdict + per-adjacent inversion flags. Conservative:
    small N or overlapping CIs -> CALIBRATION_NOISE, never a strong claim.
    """
    accs = [s["raw_k"] / s["n"] if s["n"] else 0.0 for s in rung_stats]
    ranks = list(range(len(accs)))
    rho = spearman(ranks, accs)
    inversions = []
    for a, b in zip(rung_stats, rung_stats[1:]):
        pa = a["raw_k"] / a["n"] if a["n"] else 0.0
        pb = b["raw_k"] / b["n"] if b["n"] else 0.0
        if pb > pa:  # harder rung scores higher: inversion
            loa, hia = wilson(a["raw_k"], a["n"])
            lob, hib = wilson(b["raw_k"], b["n"])
            separated = lob > hia and min(a["n"], b["n"]) >= 20
            inversions.append({"pair": [a["rung"], b["rung"]],
                               "rates": [round(pa, 4), round(pb, 4)],
                               "kind": "REAL_NONMONOTONICITY" if separated else "CALIBRATION_NOISE"})
    total_n = sum(s["n"] for s in rung_stats)
    if total_n < 40 or rho > -0.3:
        verdict = "CALIBRATION_UNSTABLE" if (inversions or rho > -0.3) else "STABLE"
    else:
        verdict = "REAL_NONMONOTONICITY" if any(i["kind"] == "REAL_NONMONOTONICITY"
                                               for i in inversions) else "STABLE"
    return {"spearman_difficulty_accuracy": round(rho, 4), "inversions": inversions,
            "verdict": verdict}
