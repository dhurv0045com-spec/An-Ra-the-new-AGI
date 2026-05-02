from __future__ import annotations
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class STaRExample:
    prompt: str
    chain: str
    answer: str
    score: float
    source: str


@dataclass
class STaRLoop:
    model: object
    tokenizer: object
    verifier: object
    threshold: float = 0.9
    max_tokens: int = 512
    accepted: list[STaRExample] = field(default_factory=list)

    def _device(self):
        return next(self.model.parameters()).device if hasattr(self.model, 'parameters') else torch.device('cpu')

    def _extract_answer(self, chain: str) -> str:
        tail = chain.split('</think>', 1)[-1] if '</think>' in chain else chain
        for marker in ('Answer:', 'answer:', 'Final:', 'final:'):
            if marker in tail:
                return tail.split(marker, 1)[-1].strip()
        lines = [line.strip() for line in tail.splitlines() if line.strip()]
        return lines[-1] if lines else ''

    @torch.no_grad()
    def _generate_chain(self, prompt: str) -> str:
        scaffold = f"{prompt}\n<think>\n"
        if hasattr(self.model, 'generate'):
            ids = torch.tensor([self.tokenizer.encode(scaffold)], dtype=torch.long, device=self._device())
            out = self.model.generate(ids, max_new_tokens=self.max_tokens)
            return self.tokenizer.decode(out[0].detach().cpu().tolist())
        return f"{scaffold}</think>\nAnswer:"

    def step(self, prompt: str, task_type: str = 'open', correct_answer: str = '', n_attempts: int = 4) -> list[STaRExample]:
        results = []
        found = False
        for _ in range(n_attempts):
            chain = self._generate_chain(prompt)
            if '<think>' not in chain:
                chain = f"{prompt}\n<think>\n{chain}\n</think>\n{self._extract_answer(chain)}"
            answer = self._extract_answer(chain)
            vr = self.verifier.score(task_type, code=answer, expression=answer, expected=correct_answer, response=answer, task=prompt)
            ex = STaRExample(prompt, chain, answer, float(vr.score), 'direct')
            results.append(ex)
            if ex.score >= self.threshold:
                self.accepted.append(ex)
                found = True
        if not found and correct_answer:
            chain = f"{prompt}\n<think>\nUse the verified answer and derive a concise rationale.\n</think>\nAnswer: {correct_answer}"
            ex = STaRExample(prompt, chain, correct_answer, 1.0, 'rationalization')
            self.accepted.append(ex)
            results.append(ex)
        return results

    def _loss_for_text(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text)[-self.model.block_size:]
        if len(ids) < 2:
            return torch.tensor(0.0, device=self._device(), requires_grad=True)
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=self._device())
        y = torch.tensor([ids[1:]], dtype=torch.long, device=self._device())
        logits, _ = self.model(x)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=1)

    def finetune_on_chains(self, optimizer: torch.optim.Optimizer, n_steps: int = 50, chain_weight: float = 1.25) -> list[float]:
        if not self.accepted:
            return []
        losses = []
        self.model.train()
        for step in range(int(n_steps)):
            ex = self.accepted[step % len(self.accepted)]
            optimizer.zero_grad(set_to_none=True)
            loss = self._loss_for_text(ex.chain) * float(chain_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        return losses

    def replay_buffer(self) -> list[dict]:
        return [x.__dict__ for x in self.accepted]
