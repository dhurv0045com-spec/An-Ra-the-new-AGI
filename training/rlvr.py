from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class RLVRTask:
    prompt: str
    task_type: str
    test_code: str = ""
    expected: str = ""
    task_id: str = ""


@dataclass
class RLVRStep:
    task: RLVRTask
    completions: list[str]
    rewards: list[float]
    advantages: list[float]
    loss: float
    mean_reward: float


class RLVRTrainer:
    def __init__(self, model, tokenizer, verifier, block_size: int = 2048, beta: float = 0.05):
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.block_size = int(block_size)
        self.beta = float(beta)

    def _log_prob(self, prompt: str, completion: str) -> torch.Tensor:
        p_ids = self.tokenizer.encode(prompt)
        c_ids = self.tokenizer.encode(completion)
        ids = (p_ids + c_ids)[-self.block_size :]
        x = torch.tensor(ids, dtype=torch.long, device=next(self.model.parameters()).device).unsqueeze(0)
        logits, _ = self.model(x)
        logp = F.log_softmax(logits[:, :-1, :], dim=-1)

        prompt_len = min(len(p_ids), len(ids))
        completion_start = max(prompt_len - 1, 0)
        target = x[:, 1:]
        comp_targets = target[:, completion_start:]
        comp_logp = logp[:, completion_start:, :].gather(-1, comp_targets.unsqueeze(-1)).squeeze(-1)
        return comp_logp.sum()

    def train_step(self, task: RLVRTask, completions: list[str]) -> RLVRStep:
        rewards = []
        logps = []
        for c in completions:
            v = self.verifier.score(task.task_type, code=c, test_code=task.test_code, expected=task.expected, response=c, task=task.prompt)
            rewards.append(float(v.score))
            logps.append(self._log_prob(task.prompt, c))

        r = torch.tensor(rewards, device=logps[0].device)
        adv = r - r.mean()
        policy_loss = torch.stack([-(a * lp) for a, lp in zip(adv, logps)]).mean()
        kl_term = torch.stack([lp.pow(2) for lp in logps]).mean() * self.beta
        loss = policy_loss + kl_term

        return RLVRStep(task, completions, rewards, adv.detach().cpu().tolist(), float(loss.item()), float(r.mean().item()))
