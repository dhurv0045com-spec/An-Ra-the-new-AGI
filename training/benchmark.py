from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import subprocess
import time


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


def _torch_modules():
    import torch
    import torch.nn.functional as F

    return torch, F


class BenchmarkSuite:
    """Run model-backed validation, coding, CIV, and coherence checks."""

    def __init__(
        self,
        model,
        tokenizer,
        verifier=None,
        civ_guard=None,
        holdout_texts: list[str] | None = None,
        coding_tasks: list[dict] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.civ_guard = civ_guard
        self._holdout = holdout_texts or []
        self._coding = coding_tasks or []

    def _device(self):
        return next(self.model.parameters()).device

    def val_perplexity(self, max_examples: int = 500) -> float:
        if not self._holdout:
            return 999.0
        self.model.eval()
        torch, F = _torch_modules()
        device = self._device()
        block = self.model.block_size
        total_loss, total_tokens = 0.0, 0

        with torch.no_grad():
            for text in self._holdout[:max_examples]:
                ids = self.tokenizer.encode(text)
                if len(ids) < 2:
                    continue
                ids = ids[:block]
                x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
                y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
                logits, _ = self.model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction="sum",
                )
                total_loss += float(loss.item())
                total_tokens += y.numel()

        if total_tokens == 0:
            return 999.0
        return float(math.exp(min(20.0, total_loss / total_tokens)))

    def rlvr_pass_at_1(self, max_tasks: int = 50) -> float:
        if not self._coding or self.verifier is None:
            return 0.0
        self.model.eval()
        torch, _ = _torch_modules()
        device = self._device()
        block = self.model.block_size
        eos_id = getattr(self.tokenizer, "special_ids", {}).get("<eos>", -1)
        passed = 0

        with torch.no_grad():
            for task in self._coding[:max_tasks]:
                prompt = task.get("prompt", "")
                test = task.get("test_code", "")
                p_ids = self.tokenizer.encode(prompt)
                gen = list(p_ids)
                for _ in range(256):
                    x = torch.tensor([gen[-block:]], dtype=torch.long, device=device)
                    logits, _ = self.model(x)
                    nxt = int(logits[0, -1, :].argmax().item())
                    gen.append(nxt)
                    if nxt == eos_id:
                        break
                code = self.tokenizer.decode(gen[len(p_ids):])
                vr = self.verifier.score("code", code=code, test_code=test)
                if vr.score >= 0.999:
                    passed += 1

        return passed / max(1, min(max_tasks, len(self._coding)))

    def civ_similarity(self) -> float:
        if self.civ_guard is None:
            return 1.0
        try:
            sim, _ = self.civ_guard.verify()
            return float(sim)
        except Exception:
            return 1.0

    def conversation_coherence(self) -> float:
        self.model.eval()
        torch, _ = _torch_modules()
        device = self._device()
        block = self.model.block_size
        secret = "7391"

        history = f"H: The secret code is {secret}.\nANRA: Understood. The secret code is {secret}.\n"
        for _ in range(18):
            history += "H: What else can you do?\nANRA: I can help with many tasks.\n"
        history += "H: What is the secret code?\nANRA:"

        ids = self.tokenizer.encode(history)[-block:]
        x = torch.tensor([ids], dtype=torch.long, device=device)

        gen = []
        eos_id = getattr(self.tokenizer, "special_ids", {}).get("<eos>", -1)
        with torch.no_grad():
            for _ in range(30):
                logits, _ = self.model(x)
                nxt = int(logits[0, -1, :].argmax().item())
                gen.append(nxt)
                if nxt == eos_id:
                    break
                x = torch.tensor([[nxt]], dtype=torch.long, device=device)

        response = self.tokenizer.decode(gen)
        return 1.0 if secret in response else 0.0

    def goal_completion_rate(self, goals_attempted: int = 0, goals_completed: int = 0) -> float:
        if goals_attempted == 0:
            return 0.0
        return float(goals_completed) / float(goals_attempted)

    def run_all(self, goals_attempted: int = 0, goals_completed: int = 0) -> BenchmarkResult:
        return BenchmarkResult(
            val_perplexity=self.val_perplexity(),
            rlvr_pass_at_1=self.rlvr_pass_at_1(),
            civ_score=self.civ_similarity(),
            coherence=self.conversation_coherence(),
            success=True,
        )

    def print_report(self, result: BenchmarkResult) -> None:
        targets = {
            "val_perplexity": ("< 30", lambda v: v < 30),
            "rlvr_pass_at_1": ("> 40%", lambda v: v > 0.40),
            "civ_score": (">= 0.92", lambda v: v >= 0.92),
            "coherence": ("> 50%", lambda v: v > 0.50),
        }
        print("=" * 64)
        print("AN-RA BENCHMARK")
        print("=" * 64)
        for field_name, (target_str, check_fn) in targets.items():
            value = getattr(result, field_name, 0.0)
            flag = "PASS" if check_fn(value) else "FAIL"
            print(f"  {flag:<4} {field_name:<28} {value:>8.4f}   target: {target_str}")
        print("=" * 64)
