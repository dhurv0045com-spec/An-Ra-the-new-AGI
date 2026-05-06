"""STaR - Self-Taught Reasoning.

The model generates its own reasoning chains. Only chains that produce verified
correct answers become training data. If all direct attempts fail but the answer
is known, STaR adds a lower-weight rationalization example.

Chain format:
  <think>
  [step-by-step reasoning]
  </think>
  [final answer]
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - structural tests can inspect without torch.
    torch = None
    F = None


def _no_grad():
    if torch is not None:
        return torch.no_grad()

    def decorator(fn):
        return fn

    return decorator


@dataclass
class STaRExample:
    prompt: str
    chain: str
    answer: str
    score: float
    source: str
    weight: float = 1.0

    @property
    def rationale(self) -> str:
        return self.chain


@dataclass
class STaRLoop:
    model: object
    tokenizer: object
    verifier: object
    threshold: float = 0.9
    max_tokens: int = 512
    accepted: list[STaRExample] = field(default_factory=list)

    def _device(self) -> torch.device:
        if self.model is not None and hasattr(self.model, "parameters"):
            return next(self.model.parameters()).device
        if torch is None:
            raise ImportError("STaRLoop generation and fine-tuning require torch.")
        return torch.device("cpu")

    @_no_grad()
    def _generate_tokens(self, prompt_ids: list[int], temperature: float = 0.7) -> list[int]:
        if torch is None or F is None:
            raise ImportError("STaRLoop generation requires torch.")
        eos_id = getattr(self.tokenizer, "special_ids", {}).get("<eos>", -1)
        block = getattr(self.model, "block_size", 512)
        device = self._device()
        gen = list(prompt_ids)

        self.model.eval()
        for _ in range(self.max_tokens):
            x = torch.tensor([gen[-block:]], dtype=torch.long, device=device)
            logits, _ = self.model(x)
            logits = logits[0, -1, :] / max(temperature, 1e-6)
            nxt = int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())
            gen.append(nxt)
            if nxt == eos_id:
                break

        return gen[len(prompt_ids) :]

    def _generate_chain(self, prompt: str) -> str:
        """Generate one chain with <think> scaffolding."""
        think_prompt = prompt + "\n<think>\n"
        ids = self.tokenizer.encode(think_prompt)
        comp = self._generate_tokens(ids)
        return think_prompt + self.tokenizer.decode(comp)

    def _extract_answer(self, chain: str) -> str:
        """Extract the final answer after </think>."""
        if "</think>" in chain:
            answer = chain.split("</think>", 1)[-1].strip()
        else:
            lines = [line.strip() for line in chain.splitlines() if line.strip()]
            answer = lines[-1] if lines else chain.strip()

        match = re.match(r"^(?:final\s+answer|answer)\s*:\s*(.+)$", answer, flags=re.I | re.S)
        return match.group(1).strip() if match else answer

    def step(
        self,
        prompt: str,
        task_type: str = "open",
        correct_answer: str = "",
        n_attempts: int = 4,
    ) -> list[STaRExample]:
        """Generate chains, verify answers, and keep accepted examples."""
        results: list[STaRExample] = []
        found = False

        for _ in range(n_attempts):
            chain = self._generate_chain(prompt)
            answer = self._extract_answer(chain)
            vr = self.verifier.score(
                task_type,
                code=answer,
                expression=answer,
                expected=correct_answer,
                response=answer,
                task=prompt,
            )
            ex = STaRExample(
                prompt=prompt,
                chain=chain,
                answer=answer,
                score=float(vr.score),
                source="direct",
                weight=1.0,
            )
            results.append(ex)
            if vr.score >= self.threshold:
                self.accepted.append(ex)
                found = True

        if not found and correct_answer:
            rat_prompt = (
                f"{prompt}\n"
                f"The correct answer is: {correct_answer}\n"
                f"Explain step-by-step:\n<think>\n"
            )
            ids = self.tokenizer.encode(rat_prompt)
            comp = self._generate_tokens(ids)
            chain = rat_prompt + self.tokenizer.decode(comp)
            ex = STaRExample(
                prompt=prompt,
                chain=chain,
                answer=correct_answer,
                score=0.5,
                source="rationalization",
                weight=0.5,
            )
            results.append(ex)
            self.accepted.append(ex)

        return results

    def finetune_on_chains(
        self,
        optimizer: torch.optim.Optimizer,
        n_steps: int = 50,
        chain_token_weight: float = 1.25,
    ) -> list[float]:
        """Fine-tune the model on accepted reasoning chains."""
        if not self.accepted:
            return []

        self.model.train()
        device = self._device()
        block = getattr(self.model, "block_size", 512)
        losses: list[float] = []

        for i in range(n_steps):
            ex = self.accepted[i % len(self.accepted)]
            p_ids = self.tokenizer.encode(ex.prompt)
            c_ids = self.tokenizer.encode(ex.chain)
            all_ids = (p_ids + c_ids)[-block:]

            if len(all_ids) < 2:
                continue

            x = torch.tensor([all_ids[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([all_ids[1:]], dtype=torch.long, device=device)
            logits, _ = self.model(x)

            time_steps = logits.shape[1]
            weights = torch.ones(time_steps, device=device)
            prompt_end = min(len(p_ids), time_steps)
            weights[prompt_end:] = chain_token_weight

            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="none",
            ).view(1, time_steps)
            loss = (ce * weights.unsqueeze(0)).mean() * ex.weight

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        return losses

    def replay_buffer(self) -> list[dict]:
        return [
            {
                "prompt": x.prompt,
                "chain": x.chain,
                "rationale": x.rationale,
                "answer": x.answer,
                "score": x.score,
                "source": x.source,
                "weight": x.weight,
            }
            for x in self.accepted
        ]

    def generate(self, prompt: str, n: int = 4) -> list[str]:
        """Backward-compatible generation API."""
        return [self._generate_chain(prompt) for _ in range(n)]
