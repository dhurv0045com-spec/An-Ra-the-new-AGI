from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class BenchmarkResult:
    val_perplexity: float
    rlvr_pass_at_1: float
    civ_score: float
    coherence: float


def run_benchmark(*, val_loss: float, rlvr_rewards: list[float], civ_score: float, coherence: float) -> BenchmarkResult:
    ppl = float(math.exp(min(20.0, max(-20.0, float(val_loss)))))
    pass1 = float(sum(1 for r in rlvr_rewards if r >= 0.999) / max(1, len(rlvr_rewards)))
    return BenchmarkResult(val_perplexity=ppl, rlvr_pass_at_1=pass1, civ_score=float(civ_score), coherence=float(coherence))
