"""Conditional V8 retention engine (N, N-extra, R, S, G) gated by qualified Task A/B artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any

from anra_v5.elendril.codec import canonical, digest

from .protocol import ADVANCEMENT_RULES


@dataclass(frozen=True, slots=True)
class ReplayQualification:
    task_name: str
    raw_accuracy: float
    relevant_cf_both_correct: float
    source_commit: str
    checkpoint_sha256: str


def verify_retention_prerequisites(
    task_a: ReplayQualification | None,
    task_b: ReplayQualification | None,
) -> dict[str, object]:
    """Require verified >=80% raw & >=70% relevant CF on both Task A and Task B before replay runs."""

    rules = ADVANCEMENT_RULES["retention"]
    if task_a is None or task_b is None:
        return {
            "qualified": False,
            "status": "PREREQUISITE_NOT_QUALIFIED",
            "reasons": ["missing verified Task A or Task B qualification artifact"],
        }
    reasons: list[str] = []
    for q in (task_a, task_b):
        if q.raw_accuracy < float(rules["a_b_qualification_raw_min"]):
            reasons.append(f"{q.task_name} raw_accuracy {q.raw_accuracy:.3f} < {rules['a_b_qualification_raw_min']}")
        if q.relevant_cf_both_correct < float(rules["a_b_qualification_cf_both_correct_min"]):
            reasons.append(
                f"{q.task_name} cf_both_correct {q.relevant_cf_both_correct:.3f} < {rules['a_b_qualification_cf_both_correct_min']}"
            )
    return {
        "qualified": len(reasons) == 0,
        "status": "QUALIFIED" if not reasons else "PREREQUISITE_NOT_QUALIFIED",
        "reasons": reasons,
    }


def build_stratified_support_index(samples: tuple[Any, ...], *, num_strata: int = 8, target_size: int = 128) -> dict[int, list[int]]:
    """Partition Task A support samples into 8 deterministic strata (total 128 support slots when available)."""

    buckets: dict[int, list[int]] = {s: [] for s in range(num_strata)}
    for idx, sample in enumerate(samples):
        cat = getattr(sample, "category", "ordinary")
        st_idx = (sum(ord(ch) for ch in str(cat)) + idx) % num_strata
        buckets[st_idx].append(idx)
    per_stratum = max(1, target_size // num_strata)
    return {s: idxs[:per_stratum] for s, idxs in buckets.items()}


class GuardianMonitorState:
    """Checkpointed G monitor: 64-update interval, 64 monitor worlds, EMA=0.5, weight=error+0.05."""

    def __init__(
        self,
        *,
        num_strata: int = 8,
        ema_alpha: float = 0.5,
        floor: float = 0.05,
        monitor_interval: int = 64,
        monitor_worlds: int = 64,
    ) -> None:
        self.num_strata = num_strata
        self.ema_alpha = ema_alpha
        self.floor = floor
        self.monitor_interval = monitor_interval
        self.monitor_worlds = monitor_worlds
        self.ema_errors: list[float] = [0.5] * num_strata
        self.monitor_calls: int = 0
        self.monitor_history: list[dict[str, object]] = []

    def observe_stratum_errors(self, update: int, stratum_errors: dict[int, float], *, work_cost: float = 64.0) -> None:
        for s in range(self.num_strata):
            err = float(stratum_errors.get(s, self.ema_errors[s]))
            self.ema_errors[s] = self.ema_alpha * err + (1.0 - self.ema_alpha) * self.ema_errors[s]
        self.monitor_calls += 1
        self.monitor_history.append(
            {
                "update": update,
                "ema_errors": list(self.ema_errors),
                "charged_monitor_worlds": self.monitor_worlds,
                "work_cost": work_cost,
            }
        )

    def sampling_weights(self) -> list[float]:
        raw = [max(0.0, e) + self.floor for e in self.ema_errors]
        total = sum(raw)
        return [w / total for w in raw]

    def sample_replay_indices(
        self,
        support: dict[int, list[int]],
        *,
        quota: int = 4,
        step_seed: int,
    ) -> list[int]:
        weights = self.sampling_weights()
        rng = random.Random(int(digest(["v8-guardian-replay", step_seed]), 16))
        available_strata = [s for s in range(self.num_strata) if support.get(s)]
        if not available_strata:
            return []
        chosen: list[int] = []
        avail_weights = [weights[s] for s in available_strata]
        for _ in range(quota):
            stratum = rng.choices(available_strata, weights=avail_weights, k=1)[0]
            pool = support[stratum]
            chosen.append(pool[rng.randrange(len(pool))])
        return chosen

    def to_dict(self) -> dict[str, object]:
        return {
            "num_strata": self.num_strata,
            "ema_alpha": self.ema_alpha,
            "floor": self.floor,
            "monitor_interval": self.monitor_interval,
            "monitor_worlds": self.monitor_worlds,
            "ema_errors": list(self.ema_errors),
            "monitor_calls": self.monitor_calls,
            "monitor_history": list(self.monitor_history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "GuardianMonitorState":
        obj = cls(
            num_strata=int(data["num_strata"]),
            ema_alpha=float(data["ema_alpha"]),
            floor=float(data["floor"]),
            monitor_interval=int(data["monitor_interval"]),
            monitor_worlds=int(data["monitor_worlds"]),
        )
        obj.ema_errors = [float(x) for x in data["ema_errors"]]
        obj.monitor_calls = int(data["monitor_calls"])
        obj.monitor_history = list(data["monitor_history"])
        return obj


def sample_fixed_replay_batch(
    task_b_samples: tuple[Any, ...],
    support: dict[int, list[int]],
    task_a_samples: tuple[Any, ...],
    *,
    regime: str,
    b_indices: tuple[int, ...],
    step_seed: int,
    guardian: GuardianMonitorState | None = None,
) -> tuple[Any, ...]:
    """Assemble matched-quota replay + new-B microbatch for N, N-extra, R, S, or G."""

    if regime not in ("N", "N-extra", "R", "S", "G"):
        raise ValueError(f"unsupported retention regime: {regime!r}")
    b_batch = [task_b_samples[i] for i in b_indices]
    if regime in ("N", "N-extra"):
        return tuple(b_batch)

    quota = int(ADVANCEMENT_RULES["retention"]["replay_quota_per_16_b"])
    rng = random.Random(int(digest(["v8-replay-select", regime, step_seed]), 16))
    if regime == "R":
        pool = [idx for idxs in support.values() for idx in idxs]
        rep_idxs = [pool[rng.randrange(len(pool))] for _ in range(min(quota, len(pool)))] if pool else []
    elif regime == "S":
        strata = [s for s, idxs in sorted(support.items()) if idxs]
        rep_idxs = []
        for k in range(quota):
            if strata:
                st = strata[k % len(strata)]
                rep_idxs.append(support[st][rng.randrange(len(support[st]))])
    else:
        if guardian is None:
            raise ValueError("regime G requires GuardianMonitorState")
        rep_idxs = guardian.sample_replay_indices(support, quota=quota, step_seed=step_seed)
    return tuple(b_batch + [task_a_samples[i] for i in rep_idxs])


def evaluate_retention_outcome(
    *,
    regime_scores: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Check whether replay (R/S/G) preserves old-A >=+0.10 vs N without hurting new-B >0.05."""

    if "N" not in regime_scores:
        return {"qualified": False, "reasons": ["missing no-replay comparator N"]}
    rules = ADVANCEMENT_RULES["retention"]
    n_a = float(regime_scores["N"]["old_a_accuracy"])
    n_b = float(regime_scores["N"]["new_b_accuracy"])
    results: dict[str, dict[str, object]] = {}
    for name, sc in sorted(regime_scores.items()):
        if name in ("N", "N-extra"):
            continue
        da = float(sc["old_a_accuracy"]) - n_a
        db = float(sc["new_b_accuracy"]) - n_b
        ok = (da >= float(rules["old_a_improvement_vs_no_replay_min"])) and (
            db >= -float(rules["new_b_within_no_replay_max_drop"])
        )
        results[name] = {"old_a_gain": da, "new_b_diff": db, "meets_retention_rule": ok}
    return {"baseline_N": regime_scores["N"], "comparisons": results}


__all__ = [
    "GuardianMonitorState",
    "ReplayQualification",
    "build_stratified_support_index",
    "evaluate_retention_outcome",
    "sample_fixed_replay_batch",
    "verify_retention_prerequisites",
]
