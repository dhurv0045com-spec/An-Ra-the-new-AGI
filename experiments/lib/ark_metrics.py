"""Arkenstone research metrics: sustained thresholds and receipt binding.

Sustained-threshold semantics (preregistered before ARK-002B): a threshold
counts only as the FIRST of >=3 consecutive evaluations at or above the bar.
A run that ends before three confirming observations marks the threshold
NOT_DEMONSTRATED. Max-accuracy claims are forbidden.
"""

from __future__ import annotations

import hashlib
import json


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sustained_threshold(
    trajectory: list[dict],
    key: str,
    threshold: float,
    *,
    consecutive: int = 3,
) -> dict:
    """First window of `consecutive` consecutive evals with metric >= threshold.

    Returns {"step": int|None, "tokens": int|None, "exposures": float|None,
    "status": "DEMONSTRATED"|"NOT_DEMONSTRATED"}.
    """

    streak = 0
    window_start: dict | None = None
    for entry in trajectory:
        if float(entry.get(key, 0.0)) >= threshold:
            if streak == 0:
                window_start = entry
            streak += 1
            if streak >= consecutive:
                return {
                    "step": window_start["step"],
                    "tokens": window_start.get("tokens"),
                    "exposures": window_start.get("exposures"),
                    "status": "DEMONSTRATED",
                }
        else:
            streak = 0
            window_start = None
    return {"step": None, "tokens": None, "exposures": None, "status": "NOT_DEMONSTRATED"}


def sustained_summary(
    trajectory: list[dict],
    *,
    memory_key: str = "train_exact",
    ood_key: str = "test_exact",
) -> dict:
    """M99 / G50 / G90 / G95 + delay, exposure ratio, and OOD-AUC after M99."""

    m99 = sustained_threshold(trajectory, memory_key, 0.99)
    summary: dict[str, object] = {
        "M99": m99,
        "G50": sustained_threshold(trajectory, ood_key, 0.50),
        "G90": sustained_threshold(trajectory, ood_key, 0.90),
        "G95": sustained_threshold(trajectory, ood_key, 0.95),
    }
    if m99["step"] is not None and summary["G90"]["step"] is not None:
        summary["post_mem_delay_90_steps"] = summary["G90"]["step"] - m99["step"]
        if summary["G90"]["exposures"] is not None and m99["exposures"]:
            summary["exposure_ratio_90"] = (
                float(summary["G90"]["exposures"]) / float(m99["exposures"])
            )
    else:
        summary["post_mem_delay_90_steps"] = None
        summary["exposure_ratio_90"] = None
    after_mem = [e for e in trajectory if m99["step"] is not None and e["step"] >= m99["step"]]
    if len(after_mem) >= 2:
        steps = [e["step"] for e in after_mem]
        width = max(steps) - min(steps)
        summary["ood_auc_after_M99"] = (
            sum(float(e.get(ood_key, 0.0)) for e in after_mem) / len(after_mem)
        )
        summary["ood_auc_window_steps"] = width
    else:
        summary["ood_auc_after_M99"] = None
    summary["max_ood_exact_forbidden_as_claim"] = True
    return summary


def bind_receipt(
    *,
    experiment_id: str,
    plan_commit_sha: str,
    code_paths: dict[str, str],
    config: dict,
    results: dict,
) -> dict:
    """Receipt bound to plan commit, code hashes, and config identity."""

    for name, path in code_paths.items():
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        code_paths[name] = digest
    receipt = {
        "experiment_id": experiment_id,
        "plan_commit_sha256": plan_commit_sha,
        "code_sha256": code_paths,
        "config": config,
        "results": results,
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    return receipt


def verify_receipt(receipt: dict, code_paths: dict[str, str]) -> bool:
    """Fail-closed check: receipt hash and bound code hashes must match."""

    stored = receipt["receipt_sha256"]
    candidate = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if hashlib.sha256(_canonical(candidate)).hexdigest() != stored:
        return False
    for name, path in code_paths.items():
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if receipt["code_sha256"].get(name) != digest:
            return False
    return True
