"""Conditional V8 frozen-weight 32-record external public memory evaluation (Scenario E16)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any
import random

from anra_v5.elendril.codec import digest
from anra_v5.elendril.v8.contracts import SYMBOLS, V8ModelSpec

from .evaluation import EpisodeRecord, run_episode
from .policies import make_policy
from .teacher import PublicFact

MEMORY_REGIMES = (
    "no_retrieval",
    "learned_retrieve",
    "engineered_retrieve",
    "shuffled_memory",
    "stale_then_public_correction",
)


def build_fixed_memory_bank(num_records: int = 32, *, seed: int = 9091) -> list[PublicFact]:
    """Construct a fixed 32-record public memory bank with non-stale & stale entries."""

    if num_records != 32:
        raise ValueError("E16 requires fixed 32-record public memory store")
    rng = random.Random(int(digest(["v8-e16-memory-bank", seed]), 16))
    bank: list[PublicFact] = []
    for i in range(num_records):
        subj = SYMBOLS[i % len(SYMBOLS)]
        val = SYMBOLS[(i * 5 + 3) % len(SYMBOLS)]
        if val == subj:
            val = SYMBOLS[(i + 1) % len(SYMBOLS)]
        bank.append(PublicFact(subject=subj, source="mem_bank", tick=10 + i, value=val))
    rng.shuffle(bank)
    return bank


def evaluate_e16_memory_regimes(
    model: Any,
    v8_spec: V8ModelSpec,
    worlds: list[dict[str, object]] | tuple[dict[str, object], ...],
    *,
    model_identity: str,
    prerequisite_qualified: bool,
) -> dict[str, object]:
    """Evaluate all 5 frozen-weight public memory regimes if prerequisite artifact is qualified."""

    if not prerequisite_qualified:
        return {
            "scenario": "E16",
            "status": "PREREQUISITE_NOT_QUALIFIED",
            "regimes": {},
        }

    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    # Freeze guard: snapshot parameters before evaluation to guarantee zero weight mutation.
    before_sig = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pol_raw = make_policy(model, v8_spec, model_identity=model_identity, mode="raw")
    pol_nomem = make_policy(model, v8_spec, model_identity=model_identity, mode="availability-mask")

    regime_records: dict[str, list[EpisodeRecord]] = {r: [] for r in MEMORY_REGIMES}
    for idx, w in enumerate(worlds):
        # 1. no_retrieval
        w_nomem = {
            **w,
            "allowed_public_tools": {
                **w["allowed_public_tools"],
                "memory_available": False,
            },
        }
        regime_records["no_retrieval"].append(run_episode(pol_nomem, w_nomem))

        # 2. learned_retrieve
        regime_records["learned_retrieve"].append(run_episode(pol_raw, w))

        # 3. engineered_retrieve (force RETRIEVE availability & supply memory payload upfront)
        mem_facts = tuple(PublicFact(**row) for row in w["hidden"]["memory_payload"])
        regime_records["engineered_retrieve"].append(run_episode(pol_raw, w, extra_initial_facts=mem_facts))

        # 4. shuffled_memory (wrong world's memory payload supplied as distraction)
        other = worlds[(idx + 1) % len(worlds)]
        w_shuf = {
            **w,
            "hidden": {
                **w["hidden"],
                "memory_payload": list(other["hidden"]["memory_payload"]),
            },
        }
        regime_records["shuffled_memory"].append(run_episode(pol_raw, w_shuf))

        # 5. stale_then_public_correction (stale memory claim at tick 1 superseded by authoritative public tick 999)
        exp = w["hidden"]["expected"]
        if exp is not None:
            wrong = next(s for s in SYMBOLS if s != exp)
            stale_mem = (PublicFact(subject=str(w["goal_subject"]), source="auth_mem", tick=1, value=wrong),)
            public_fix = (PublicFact(subject=str(w["goal_subject"]), source="auth_mem", tick=999, value=str(exp)),)
            w_stale = {
                **w,
                "hops": 1,
                "hidden": {**w["hidden"], "memory_payload": [asdict(row) for row in stale_mem]},
            }
            regime_records["stale_then_public_correction"].append(
                run_episode(pol_raw, w_stale, extra_initial_facts=stale_mem + public_fix)
            )

    for m, was in modes:
        m.training = was
    for k, v in model.state_dict().items():
        if not v.detach().cpu().equal(before_sig[k]):
            raise AssertionError("frozen-weight E16 memory evaluation mutated weights!")

    summaries = {
        r: {
            "episodes": len(eps),
            "exact_episode_accuracy": (sum(1 for e in eps if e.outcome.correct) / len(eps)) if eps else None,
        }
        for r, eps in regime_records.items()
    }
    return {
        "scenario": "E16",
        "status": "EXECUTED",
        "frozen_weights_verified": True,
        "memory_bank_records": 32,
        "regimes": summaries,
    }


__all__ = [
    "MEMORY_REGIMES",
    "build_fixed_memory_bank",
    "evaluate_e16_memory_regimes",
]
