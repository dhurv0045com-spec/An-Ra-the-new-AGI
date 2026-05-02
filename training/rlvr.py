from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy

import torch
import torch.nn.functional as F


@dataclass
class RLVRTask:
    prompt: str
    task_type: str
    test_code: str = ''
    expected: str = ''
    task_id: str = ''


@dataclass
class RLVRStep:
    task: RLVRTask
    completions: list[str]
    rewards: list[float]
    advantages: list[float]
    loss: float
    mean_reward: float


class RLVRTrainer:
    def __init__(self, model, tokenizer, optimizer: torch.optim.Optimizer, verifier, G:int=4, kl_coeff:float=0.04, max_new_tokens:int=256, grad_clip:float=1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.verifier = verifier
        self.G = int(G)
        self.kl_coeff = float(kl_coeff)
        self.max_new_tokens = int(max_new_tokens)
        self.grad_clip = float(grad_clip)
        self.reference_model = deepcopy(model).eval()
        for param in self.reference_model.parameters():
            param.requires_grad_(False)

    def _device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def _generate_completions(self, prompt: str, n: int) -> list[str]:
        if hasattr(self.model, "generate"):
            prompt_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self._device())
            completions = []
            for _ in range(n):
                out = self.model.generate(prompt_ids, max_new_tokens=self.max_new_tokens)
                ids = out[0].detach().cpu().tolist()[len(prompt_ids[0]):]
                completions.append(self.tokenizer.decode(ids))
            return completions
        seed = self.tokenizer.decode(self.tokenizer.encode(prompt)[-16:])
        return [seed for _ in range(n)]

    def _sequence_terms(self, prompt: str, completion: str) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_ids = self.tokenizer.encode(prompt)
        ids = (prompt_ids + self.tokenizer.encode(completion))[-self.model.block_size:]
        if len(ids) < 2:
            zero = torch.tensor(0.0, device=self._device(), requires_grad=True)
            return zero, zero
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=self._device())
        y = torch.tensor([ids[1:]], dtype=torch.long, device=self._device())
        logits, _ = self.model(x)
        with torch.no_grad():
            ref_logits, _ = self.reference_model.to(self._device())(x)
        logp = F.log_softmax(logits, dim=-1)
        ref_logp = F.log_softmax(ref_logits, dim=-1)
        start = min(len(prompt_ids), len(ids) - 1)
        if start >= y.shape[1]:
            zero = torch.tensor(0.0, device=self._device(), requires_grad=True)
            return zero, zero
        selected_y = y[:, start:]
        token_logp = logp[:, start:, :].gather(2, selected_y.unsqueeze(-1)).squeeze(-1)
        policy_logp = token_logp.sum()
        probs = logp[:, start:, :].exp()
        kl = (probs * (logp[:, start:, :] - ref_logp[:, start:, :])).sum(dim=-1).mean()
        return policy_logp, kl

    def train_step(self, task: RLVRTask) -> RLVRStep:
        completions = self._generate_completions(task.prompt, self.G)
        rewards = []
        for c in completions:
            vr = self.verifier.score(task.task_type, code=c, test_code=task.test_code, expression=c, expected=task.expected, response=c, task=task.prompt)
            rewards.append(float(vr.score))
        r = torch.tensor(rewards, dtype=torch.float32)
        mean_r = r.mean()
        std_r = r.std(unbiased=False)
        adv = ((r - mean_r) / (std_r + 1e-8)).tolist()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        policy_loss = torch.tensor(0.0, device=self._device())
        kl_loss = torch.tensor(0.0, device=self._device())
        for completion, advantage in zip(completions, adv):
            logp, kl = self._sequence_terms(task.prompt, completion)
            policy_loss = policy_loss + (-float(advantage) * logp)
            kl_loss = kl_loss + kl
        loss = (policy_loss / self.G) + self.kl_coeff * (kl_loss / self.G)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return RLVRStep(task, completions, rewards, adv, float(loss.item()), float(mean_r.item()))
