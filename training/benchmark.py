from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
import subprocess
import time
from pathlib import Path


@dataclass
class BenchmarkResult:
    val_perplexity: float
    rlvr_pass_at_1: float
    civ_score: float
    coherence: float
    success: bool = True
    command_return_code: int | None = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_benchmark(
    *,
    val_loss: float | None = None,
    rlvr_rewards: list[float] | None = None,
    civ_score: float = 1.0,
    coherence: float = 1.0,
    command: list[str] | None = None,
    cwd: str | Path = ".",
    output_path: str | Path | None = None,
    timeout: int = 300,
) -> BenchmarkResult:
    start = time.time()
    return_code = None
    success = True
    if command:
        proc = subprocess.run(command, capture_output=True, text=True, cwd=str(cwd), timeout=timeout)
        return_code = int(proc.returncode)
        success = proc.returncode == 0
    loss = 0.0 if val_loss is None else float(val_loss)
    ppl = float(math.exp(min(20.0, max(-20.0, loss))))
    rewards = rlvr_rewards or []
    pass1 = float(sum(1 for r in rewards if r >= 0.999) / max(1, len(rewards)))
    result = BenchmarkResult(
        val_perplexity=ppl,
        rlvr_pass_at_1=pass1,
        civ_score=float(civ_score),
        coherence=float(coherence),
        success=success,
        command_return_code=return_code,
        elapsed_seconds=round(time.time() - start, 4),
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result
